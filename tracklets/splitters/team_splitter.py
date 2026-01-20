import numpy as np
from collections import Counter

from utils.config import SplitterConfig


class TeamSplitter():
    """ 
    Splitting tracklet into multiple subtracklets when a team ID switch is detected:
    - Ignores very short team ID switches occuring by occlusion
    """
    def __init__(self, config=None):
        """ Initialize TeamSplitter with configuration """
        self.config = config if config else SplitterConfig()


    def split_tracklet(self, tracklet, next_available_id):
        """ Split tracklet at persistent team id switches """

        teams = tracklet.pred_attributes.get('teams', [])

        if not teams or len(teams) == 0:
            return None

        # Detect switch points
        switch_indices = self.detect_id_switch(teams)

        if not switch_indices:
            return None
        
        # Create fragment boundaries
        boundaries = [0] + switch_indices + [len(teams)]

        fragments = []
        current_id = next_available_id

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1] - 1

            # Check minimum fragment size
            fragment_length = end - start + 1
            if fragment_length < self.config.team_min_fragment:
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
    


    def detect_id_switch(self, teams):
        """ Detect switch points where team id changes persistently """
        switch_points = []
        n = len(teams)

        current_team = None
        i = 0
        while i < n:
            if self.is_valid_number(teams[i]):
                candidate = teams[i]

                # Check if first value is actually persistent and not noise
                if self.is_persistent_switch(teams, i, candidate):
                    current_team = candidate
                    break

            i += 1

        if current_team is None:
            return []

        # Scan through predictions
        while i < n:
            if not self.is_valid_number(teams[i]):
                i += 1
                continue

            # Check if this is a different number
            if teams[i] != current_team:
                candidate = teams[i]

                # Verify persistence using a window
                if self.is_persistent_switch(teams, i, candidate):
                    switch_points.append(i)
                    current_team = candidate

            i += 1

        return switch_points
    

    def is_persistent_switch(self, teams, start_idx, new_team):
        """ Check if new team persists in the lookahead window """
        lookahead = self.config.team_lookahead
        min_persistence = self.config.team_min_persistence

        # Extract window
        window_end = min(start_idx + lookahead, len(teams))
        window = [j for j in teams[start_idx:window_end] if self.is_valid_number(j)]

        # Count occurrences of new teams
        new_count = sum(1 for j in window if j == new_team)

        # Must appear at least min_persistence times
        if new_count < min_persistence:
            return False

        # Should be the dominant value in the window (filters out noise)
        if len(window) > 0:
            counts = Counter(window)
            most_common_team = counts.most_common(1)[0][0]
            return most_common_team == new_team

        return new_count >= min_persistence
    

    def is_valid_number(self, value):
            """ Check if team value is valid """
            if value is None:
                return False
            if isinstance(value, int) and np.isnan(value):
                return False
            return True