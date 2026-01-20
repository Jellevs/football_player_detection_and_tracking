import numpy as np
from collections import Counter
from utils.config import SplitterConfig


class JerseySplitter():
    """ 
    Splitting tracklet into multiple subtracklets when a jersey number switch is detected:
    - Uses entropy threshold to only use jerseys with a entropy below 0.2 
    - Checks for earlier split point by checking for abnormal fast moving bounding boxes 
    """
    def __init__(self, config=None):
        """ Initialize JerseySplitter with configuration """
        self.config = config if config else SplitterConfig()


    def split_tracklet(self, tracklet, next_available_id):
        """ Split tracklet at persistent jersey number switches """

        jerseys = tracklet.pred_attributes.get('jerseys', [])
        entropies = tracklet.pred_attributes.get('jersey_entropies', [])

        if not jerseys or len(jerseys) == 0:
            return None

        # Detect switch points
        switch_indices = self.detect_id_switch(jerseys, entropies)

        if not switch_indices:
            return None
        
        refined_switch_indices = []
        for switch_idx in switch_indices:
            refined_switch_idx = self.refine_split_point(tracklet, switch_idx)
            refined_switch_indices.append(refined_switch_idx)


        # Create fragment boundaries
        boundaries = [0] + refined_switch_indices + [len(jerseys)]

        fragments = []
        current_id = next_available_id

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1] - 1

            # Check minimum fragment size
            fragment_length = end - start + 1
            if fragment_length < self.config.jersey_min_fragment:
                # Skip fragments that are too short
                continue

            # Extract sub-tracklet
            fragment = tracklet.extract(start, end)
            fragment.track_id = current_id
            fragment.parent_id = tracklet.parent_id

            fragments.append(fragment)
            current_id += 1

        # If filtering removed all fragments, return None
        if len(fragments) == 0:
            return None

        # If only one fragment remains, no split occurred
        if len(fragments) == 1:
            return None

        return fragments

    
    def detect_id_switch(self, jerseys, entropies):
        """ Detect switch points where jersey identity changes persistently """
        switch_points = []
        n = len(jerseys)

        current_jersey = None

        i = 0
        while i < n:
            if self.is_valid_number(jerseys[i], entropies[i]):
                candidate = jerseys[i]

                # Check if first value is actually persistent and not noise
                if self.is_persistent_switch(jerseys, entropies, i, candidate):
                    current_jersey = candidate
                    break

            i += 1

        if current_jersey is None:
            return []

        # Scan through predictions
        while i < n:
            if not self.is_valid_number(jerseys[i], entropies[i]):
                i += 1
                continue

            # Check if this is a different number
            if jerseys[i] != current_jersey:
                candidate = jerseys[i]

                # Verify persistence using a window
                if self.is_persistent_switch(jerseys, entropies, i, candidate):
                    switch_points.append(i)
                    current_jersey = candidate

            i += 1

        return switch_points


    def is_valid_number(self, value, entropy):
        """ Check if jersey value is valid and confident """
        if value is None:
            return False
        if isinstance(value, float) and np.isnan(value):
            return False
        if entropy > self.config.jersey_entropy_threshold:
            return False
        return True


    def is_persistent_switch(self, jerseys, entropies, start_idx, new_jersey):
        """ Check if new jersey persists in the lookahead window """
        lookahead = self.config.jersey_lookahead
        min_persistence = self.config.jersey_min_persistence

        # Extract window
        window_end = min(start_idx + lookahead, len(jerseys))
        window = [
            j for i, j in enumerate(jerseys[start_idx:window_end])
            if self.is_valid_number(j, entropies[start_idx + 1])
        ]

        # Count occurrences of new jersey
        new_count = sum(1 for j in window if j == new_jersey)

        # Must appear at least min_persistence times
        if new_count < min_persistence:
            return False

        # Should be the dominant value in the window (filters out noise)
        if len(window) > 0:
            counts = Counter(window)
            most_common_jersey = counts.most_common(1)[0][0]
            return most_common_jersey == new_jersey

        return new_count >= min_persistence


    def refine_split_point(self, tracklet, detected_idx):
        """ Look for earlier split point by bbox anomalies """
        bboxes = tracklet.bboxes

        if bboxes is None or len(bboxes) == 0:
            return detected_idx

        # Set search window
        start_search = max(0, detected_idx - self.config.jersey_lookback)
        search_slice = np.array(bboxes[start_search : detected_idx + 1])

        # Not enough data
        if len(search_slice) < 4:
            return detected_idx

        centers = search_slice[:, :2] + search_slice[:, 2:] / 2

        # Calculate velocity
        velocities = np.linalg.norm(np.diff(centers, axis=0), axis=1)

        mean_v = np.mean(velocities)
        std_v = np.std(velocities)
        
        # Find biggest peak
        max_v_idx = np.argmax(velocities)
        max_v = velocities[max_v_idx]

        # Define threshold
        threshold = mean_v + (2.5 * std_v)

        if max_v > threshold and max_v > self.config.jersey_min_pixel_jump:
            refined_split_idx = start_search + max_v_idx + 1
            
            # If earlier split point is found, return it
            if refined_split_idx < detected_idx:
                return refined_split_idx

        return detected_idx