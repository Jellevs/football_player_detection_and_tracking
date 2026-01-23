import numpy as np
from collections import defaultdict

from utils.config import SplitterConfig


class TrajectorySplitter:
    def __init__(self, config=None):
        self.config = config if config else SplitterConfig()
        
        # Proximity detection
        self.proximity_distance = 80  # pixels - bboxes closer than this
        self.min_overlap_frames = 3   # minimum frames of proximity to consider
        
        # Velocity analysis
        self.velocity_window = 5       # frames before/after to calculate velocity
        self.min_velocity_change = 8.0 # pixels/frame - minimum speed change
        self.direction_change_threshold = 60  # degrees - minimum direction change
        
        # Trajectory swap detection
        self.swap_similarity_threshold = 0.6  # cosine similarity for velocity swap
        
        # Fragment filtering
        self.min_fragment_length = 15  # frames
        
    
    def split_all_tracklets(self, tracklets_dict):
        """
        Analyze all tracklets together and split based on proximity + trajectory
        Returns new tracklets dict with splits applied
        """
        print(f"\n=== Trajectory-Proximity Splitting ===")
        
        # Build frame-to-tracklets index for fast proximity lookup
        frame_index = self.build_frame_index(tracklets_dict)
        
        # Detect proximity events and potential splits
        split_decisions = self.detect_all_splits(tracklets_dict, frame_index)
        
        # Apply splits
        new_tracklets = self.apply_splits(tracklets_dict, split_decisions)
        
        return new_tracklets
    
    
    def build_frame_index(self, tracklets_dict):
        """
        Build index: frame_idx -> list of (track_id, local_idx, bbox, center)
        """
        frame_index = defaultdict(list)
        
        for track_id, tracklet in tracklets_dict.items():
            for local_idx, frame_idx in enumerate(tracklet.frames):
                bbox = tracklet.bboxes[local_idx]
                center = self.get_bbox_center(bbox)
                
                frame_index[frame_idx].append({
                    'track_id': track_id,
                    'local_idx': local_idx,
                    'bbox': bbox,
                    'center': center
                })
        
        return frame_index
    
    
    def detect_all_splits(self, tracklets_dict, frame_index):
        """
        Detect all split points across all tracklets
        Returns: dict of {track_id: [split_indices]}
        """
        split_decisions = defaultdict(list)
        processed_pairs = set()
        
        # Analyze each frame for proximity events
        for frame_idx, detections in frame_index.items():
            # Check all pairs of tracklets in this frame
            for i in range(len(detections)):
                for j in range(i + 1, len(detections)):
                    det_A = detections[i]
                    det_B = detections[j]
                    
                    track_A = det_A['track_id']
                    track_B = det_B['track_id']
                    
                    # Skip if already processed this pair at this frame
                    pair_key = (min(track_A, track_B), max(track_A, track_B), frame_idx)
                    if pair_key in processed_pairs:
                        continue
                    processed_pairs.add(pair_key)
                    
                    # Check if bboxes are close enough
                    if self.are_bboxes_close(det_A['bbox'], det_B['bbox'], det_A['center'], det_B['center']):
                        # Analyze this proximity event
                        split_info = self.analyze_proximity_event(
                            tracklets_dict[track_A],
                            tracklets_dict[track_B],
                            det_A['local_idx'],
                            det_B['local_idx'],
                            frame_idx
                        )
                        
                        if split_info:
                            split_decisions[track_A].append(split_info['split_idx_A'])
                            split_decisions[track_B].append(split_info['split_idx_B'])
                            
                            print(f"  Split detected: Track {track_A} & {track_B} at frame {frame_idx}")
                            print(f"    Reason: {split_info['reason']}")
        
        # Deduplicate and sort split points for each tracklet
        for track_id in split_decisions:
            splits = sorted(set(split_decisions[track_id]))
            split_decisions[track_id] = splits
        
        return split_decisions
    
    
    def are_bboxes_close(self, bbox_A, bbox_B, center_A, center_B):
        """
        Check if two bboxes are close enough to potentially cause identity switch
        Uses both IoU and center distance
        """
        # Calculate IoU
        iou = self.calculate_iou(bbox_A, bbox_B)
        if iou > 0.05:  # Any overlap
            return True
        
        # Calculate center distance
        distance = np.linalg.norm(center_A - center_B)
        if distance < self.proximity_distance:
            return True
        
        return False
    
    
    def analyze_proximity_event(self, tracklet_A, tracklet_B, local_idx_A, local_idx_B, frame_idx):
        """
        Analyze a proximity event to determine if identity switch occurred
        Returns split info dict or None
        """
        # Get velocities before and after this frame
        vel_A_before = self.get_velocity(tracklet_A, local_idx_A, before=True)
        vel_A_after = self.get_velocity(tracklet_A, local_idx_A, before=False)
        
        vel_B_before = self.get_velocity(tracklet_B, local_idx_B, before=True)
        vel_B_after = self.get_velocity(tracklet_B, local_idx_B, before=False)
        
        # Skip if we don't have enough data
        if vel_A_before is None or vel_A_after is None or vel_B_before is None or vel_B_after is None:
            return None
        
        # Signal 1: Velocity magnitude change
        speed_change_A = abs(np.linalg.norm(vel_A_after) - np.linalg.norm(vel_A_before))
        speed_change_B = abs(np.linalg.norm(vel_B_after) - np.linalg.norm(vel_B_before))
        
        # Signal 2: Direction change
        dir_change_A = self.calculate_direction_change(vel_A_before, vel_A_after)
        dir_change_B = self.calculate_direction_change(vel_B_before, vel_B_after)
        
        # Signal 3: Velocity swap detection
        swap_score = self.detect_velocity_swap(vel_A_before, vel_A_after, vel_B_before, vel_B_after)
        
        # Decision logic: Multiple signals increase confidence
        score = 0
        reasons = []
        
        # High velocity swap score is strong indicator
        if swap_score > self.swap_similarity_threshold:
            score += 3
            reasons.append(f"velocity swap (score={swap_score:.2f})")
        
        # Significant speed change in either tracklet
        if speed_change_A > self.min_velocity_change:
            score += 1
            reasons.append(f"speed change A ({speed_change_A:.1f}px/f)")
        if speed_change_B > self.min_velocity_change:
            score += 1
            reasons.append(f"speed change B ({speed_change_B:.1f}px/f)")
        
        # Significant direction change
        if dir_change_A > self.direction_change_threshold:
            score += 1
            reasons.append(f"direction change A ({dir_change_A:.0f}°)")
        if dir_change_B > self.direction_change_threshold:
            score += 1
            reasons.append(f"direction change B ({dir_change_B:.0f}°)")
        
        # Require minimum score to trigger split
        if score >= 2:  # At least 2 signals
            return {
                'split_idx_A': local_idx_A,
                'split_idx_B': local_idx_B,
                'frame_idx': frame_idx,
                'score': score,
                'reason': ', '.join(reasons)
            }
        
        return None
    
    
    def get_velocity(self, tracklet, local_idx, before=True):
        """
        Calculate average velocity before or after a given index
        Returns velocity vector (dx, dy) or None if insufficient data
        """
        window = self.velocity_window
        
        if before:
            start = max(0, local_idx - window)
            end = local_idx
        else:
            start = local_idx + 1
            end = min(len(tracklet.frames), local_idx + 1 + window)
        
        if end - start < 2:
            return None
        
        # Get bbox centers in this window
        centers = []
        for i in range(start, end):
            bbox = tracklet.bboxes[i]
            center = self.get_bbox_center(bbox)
            centers.append(center)
        
        centers = np.array(centers)
        
        # Calculate velocities (frame-to-frame)
        velocities = np.diff(centers, axis=0)
        
        # Return average velocity
        avg_velocity = np.mean(velocities, axis=0)
        return avg_velocity
    
    
    def calculate_direction_change(self, vel_before, vel_after):
        """
        Calculate angle change between two velocity vectors in degrees
        """
        # Normalize vectors
        vel_before_norm = vel_before / (np.linalg.norm(vel_before) + 1e-6)
        vel_after_norm = vel_after / (np.linalg.norm(vel_after) + 1e-6)
        
        # Calculate angle using dot product
        cos_angle = np.clip(np.dot(vel_before_norm, vel_after_norm), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    
    def detect_velocity_swap(self, vel_A_before, vel_A_after, vel_B_before, vel_B_after):
        """
        Detect if two tracklets swapped velocities
        Returns swap score (0-1), higher = more likely swap occurred
        """
        # Check if A's new velocity matches B's old velocity
        sim_A_to_B = self.cosine_similarity(vel_A_after, vel_B_before)
        
        # Check if B's new velocity matches A's old velocity
        sim_B_to_A = self.cosine_similarity(vel_B_after, vel_A_before)
        
        # Average similarity (both should be high for clean swap)
        swap_score = (sim_A_to_B + sim_B_to_A) / 2.0
        
        return swap_score
    
    
    def cosine_similarity(self, vec_A, vec_B):
        """
        Calculate cosine similarity between two vectors
        """
        norm_A = np.linalg.norm(vec_A)
        norm_B = np.linalg.norm(vec_B)
        
        if norm_A < 1e-6 or norm_B < 1e-6:
            return 0.0
        
        cos_sim = np.dot(vec_A, vec_B) / (norm_A * norm_B)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        
        # Convert to 0-1 range (0 = opposite, 1 = same)
        return (cos_sim + 1.0) / 2.0
    
    
    def get_bbox_center(self, bbox):
        """
        Get center point of bbox [x1, y1, x2, y2]
        """
        return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    
    
    def calculate_iou(self, bbox_A, bbox_B):
        """
        Calculate IoU between two bboxes
        """
        x1 = max(bbox_A[0], bbox_B[0])
        y1 = max(bbox_A[1], bbox_B[1])
        x2 = min(bbox_A[2], bbox_B[2])
        y2 = min(bbox_A[3], bbox_B[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area_A = (bbox_A[2] - bbox_A[0]) * (bbox_A[3] - bbox_A[1])
        area_B = (bbox_B[2] - bbox_B[0]) * (bbox_B[3] - bbox_B[1])
        
        union = area_A + area_B - intersection
        
        return intersection / (union + 1e-6)
    
    
    def apply_splits(self, tracklets_dict, split_decisions):
        """
        Apply split decisions to create new tracklets
        """
        if not split_decisions:
            print("No trajectory-proximity splits detected")
            return tracklets_dict
        
        max_existing_id = max(tracklets_dict.keys()) if tracklets_dict else 0
        next_available_id = max_existing_id + 1
        
        new_tracklets = {}
        split_count = 0
        
        for track_id, tracklet in tracklets_dict.items():
            if track_id in split_decisions:
                split_indices = split_decisions[track_id]
                
                # Create fragments
                fragments = self.create_fragments(tracklet, split_indices, next_available_id)
                
                if fragments and len(fragments) > 1:
                    split_count += 1
                    print(f"  Tracklet {track_id} split into {len(fragments)} fragments")
                    
                    for fragment in fragments:
                        new_tracklets[fragment.track_id] = fragment
                        next_available_id = max(next_available_id, fragment.track_id + 1)
                else:
                    # Splits didn't result in valid fragments
                    new_tracklets[tracklet.track_id] = tracklet
            else:
                # No splits for this tracklet
                new_tracklets[tracklet.track_id] = tracklet
        
        print(f"Trajectory-proximity splitting: {split_count}/{len(tracklets_dict)} tracklets split")
        return new_tracklets
    
    
    def create_fragments(self, tracklet, split_indices, next_available_id):
        """
        Create fragment tracklets from split points
        """
        # Create boundaries
        boundaries = [0] + sorted(split_indices) + [len(tracklet.frames)]
        
        fragments = []
        current_id = next_available_id
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1] - 1
            
            # Check minimum fragment size
            fragment_length = end - start + 1
            if fragment_length < self.min_fragment_length:
                continue
            
            # Extract sub-tracklet
            fragment = tracklet.extract(start, end)
            fragment.track_id = current_id
            fragment.parent_id = tracklet.parent_id
            
            fragments.append(fragment)
            current_id += 1
        
        # Return None if filtering removed everything or only 1 fragment left
        if len(fragments) <= 1:
            return None
        
        return fragments














