import numpy as np
from utils.config import SplitterConfig


class BboxAnomalySplitter:
    """
    Split tracklets based on bbox velocity anomalies (spatial jumps).
    
    When an ID switch occurs, the bounding box "teleports" from one player to another,
    causing an abnormally high frame-to-frame movement for just a few frames.
    
    This is often the MOST RELIABLE signal for ID switches because:
    - It's independent of jersey numbers or team colors
    - It works even when players move at similar speeds
    - The spike is very distinctive and short-lived
    """
    
    def __init__(self, config=None):
        self.config = config if config else SplitterConfig()
        
        # Velocity spike detection parameters
        self.lookback_window = 20        # frames to analyze for baseline velocity
        self.lookahead_window = 10       # frames to check after spike
        self.std_threshold = 4         # sigma threshold for anomaly detection
        self.min_spike_velocity = 35.0   # minimum pixels/frame to consider
        self.min_fragment_length = 10    # minimum frames in a fragment
        
        # Spike must be SHORT (1-3 frames) - longer movements are likely real player motion
        self.max_spike_duration = 3      # frames
        
    
    def split_tracklet(self, tracklet, next_available_id):
        """
        Detect velocity anomalies and split tracklet at these points
        """
        if len(tracklet.bboxes) < self.lookback_window + self.lookahead_window:
            return None
        
        # Calculate frame-to-frame velocities
        velocities = self.calculate_velocities(tracklet.bboxes)
        
        # Detect velocity spikes
        spike_indices = self.detect_velocity_spikes(velocities)
        
        if not spike_indices:
            return None
        
        # Create fragments
        fragments = self.create_fragments(tracklet, spike_indices, next_available_id)
        
        return fragments
    
    
    def calculate_velocities(self, bboxes):
        """
        Calculate frame-to-frame velocities of bbox centers
        Returns array of velocity magnitudes (pixels/frame)
        """
        centers = []
        for bbox in bboxes:
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            centers.append([center_x, center_y])
        
        centers = np.array(centers)
        
        # Calculate displacements between consecutive frames
        displacements = np.diff(centers, axis=0)
        
        # Calculate velocity magnitudes
        velocities = np.linalg.norm(displacements, axis=1)
        
        return velocities
    
    
    def detect_velocity_spikes(self, velocities):
        """
        Detect anomalous velocity spikes that indicate ID switches
        
        Returns list of frame indices where spikes occur
        """
        spike_indices = []
        n = len(velocities)
        
        for i in range(self.lookback_window, n - self.lookahead_window):
            # Get baseline velocity statistics (before this frame)
            lookback_start = max(0, i - self.lookback_window)
            baseline_velocities = velocities[lookback_start:i]
            
            if len(baseline_velocities) == 0:
                continue
            
            baseline_mean = np.mean(baseline_velocities)
            baseline_std = np.std(baseline_velocities)
            
            # Current velocity
            current_velocity = velocities[i]
            
            # Check if current velocity is anomalously high
            if baseline_std > 0:
                z_score = (current_velocity - baseline_mean) / baseline_std
            else:
                z_score = 0
            
            # Spike detection criteria
            is_spike = (
                z_score > self.std_threshold and 
                current_velocity > self.min_spike_velocity
            )
            
            if not is_spike:
                continue
            
            # Check that spike is SHORT (1-3 frames)
            spike_duration = self.measure_spike_duration(velocities, i, baseline_mean, baseline_std)
            
            if spike_duration > self.max_spike_duration:
                # This is sustained fast movement, not an ID switch
                continue
            
            # Check that velocity returns to normal afterwards
            lookahead_start = i + 1
            lookahead_end = min(n, i + 1 + self.lookahead_window)
            lookahead_velocities = velocities[lookahead_start:lookahead_end]
            
            if len(lookahead_velocities) > 0:
                lookahead_mean = np.mean(lookahead_velocities)
                
                # After spike, velocity should return close to baseline
                # (allows some change, but not sustained high velocity)
                returns_to_normal = lookahead_mean < baseline_mean * 2.0
                
                if not returns_to_normal:
                    continue
            
            spike_indices.append(i + 1)  # Split AFTER the spike frame
            
            print(f"    Velocity spike detected at frame_idx {i}:")
            print(f"      Baseline: {baseline_mean:.1f} ± {baseline_std:.1f} px/f")
            print(f"      Spike: {current_velocity:.1f} px/f (z={z_score:.2f})")
            print(f"      Duration: {spike_duration} frames")
            print(f"      After spike: {lookahead_mean:.1f} px/f")
        
        return spike_indices
    
    
    def measure_spike_duration(self, velocities, spike_idx, baseline_mean, baseline_std):
        """
        Measure how many consecutive frames have elevated velocity
        """
        threshold = baseline_mean + self.std_threshold * baseline_std
        duration = 1
        
        # Check forward
        i = spike_idx + 1
        while i < len(velocities) and velocities[i] > threshold:
            duration += 1
            i += 1
            if duration > self.max_spike_duration:
                break
        
        return duration
    
    
    def create_fragments(self, tracklet, spike_indices, next_available_id):
        """
        Create fragment tracklets from split points
        CRITICAL FIX: Merge small fragments to prevent losing detections
        """
        # Remove any split indices that are out of bounds
        valid_splits = [idx for idx in spike_indices if 0 < idx < len(tracklet.frames)]
        
        if not valid_splits:
            return None
        
        # Create initial boundaries
        boundaries = [0] + sorted(valid_splits) + [len(tracklet.frames)]
        
        # FIXED: Remove boundaries that would create too-small fragments
        # This merges small fragments with adjacent ones
        merged_boundaries = [boundaries[0]]  # Always keep start
        
        for i in range(1, len(boundaries) - 1):  # Check middle boundaries only
            # Check if this boundary creates a valid fragment from the last kept boundary
            fragment_length = boundaries[i] - merged_boundaries[-1]
            
            if fragment_length >= self.min_fragment_length:
                # Check if remaining segment is also large enough
                remaining_length = boundaries[-1] - boundaries[i]
                
                if remaining_length >= self.min_fragment_length:
                    # Both fragments are valid, keep this boundary
                    merged_boundaries.append(boundaries[i])
                # else: don't add boundary, merge with next fragment
            # else: don't add boundary, merge with previous fragment
        
        merged_boundaries.append(boundaries[-1])  # Always keep end
        
        # Create fragments from merged boundaries
        fragments = []
        current_id = next_available_id
        
        for i in range(len(merged_boundaries) - 1):
            start = merged_boundaries[i]
            end = merged_boundaries[i + 1]
            
            # Extract sub-tracklet (end-1 because extract uses inclusive end)
            fragment = tracklet.extract(start, end - 1)
            fragment.track_id = current_id
            fragment.parent_id = tracklet.parent_id
            
            fragments.append(fragment)
            current_id += 1
        
        # Return None if only 1 fragment left (no actual split occurred)
        if len(fragments) <= 1:
            return None
        
        return fragments