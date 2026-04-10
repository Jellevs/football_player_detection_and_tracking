import numpy as np
from collections import Counter
from utils.config import SplitterConfig


class UnifiedSplitter:
    """
    Simultaneous jersey+team splitter.

    Runs a single scan over all frames and evaluates both signals at every
    frame. A split is triggered when either signal independently detects a
    persistent switch — using the exact same persistence logic as JerseySplitter
    and TeamSplitter respectively.

    After each split, both signal references reset to the new identity.

    This is structurally equivalent to the sequential design (same split
    decisions, same thresholds) but avoids the implicit priority hierarchy:
    in the sequential design, jersey runs first and the team splitter never
    sees across jersey-split boundaries. Here both signals are always active.
    """

    def __init__(self, config: SplitterConfig = None):
        self.config = config if config else SplitterConfig()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def split_tracklet(self, tracklet, next_available_id):
        jerseys    = tracklet.pred_attributes.get("jerseys", [])
        entropies  = tracklet.pred_attributes.get("jersey_entropies", [])
        teams      = tracklet.pred_attributes.get("teams", [])
        team_confs = tracklet.pred_attributes.get("team_confs", [])

        n = len(tracklet.frames)
        if n == 0:
            return None

        # Pad to length n so indexing is always safe
        jerseys    = list(jerseys)    + [None] * max(0, n - len(jerseys))
        entropies  = list(entropies)  + [1.0]  * max(0, n - len(entropies))
        teams      = list(teams)      + [None] * max(0, n - len(teams))
        team_confs = list(team_confs) + [0.0]  * max(0, n - len(team_confs))

        switch_indices = self._detect_switches(jerseys, entropies, teams, team_confs)

        if not switch_indices:
            return None

        return self._create_fragments(tracklet, switch_indices, n, next_available_id)

    def _create_fragments(self, tracklet, switch_indices, n, next_available_id):
        boundaries = [0] + switch_indices + [n]
        fragments  = []
        current_id = next_available_id

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end   = boundaries[i + 1] - 1
            if end - start + 1 < self.config.min_fragment_length:
                continue
            fragment = tracklet.extract(start, end)
            fragment.track_id  = current_id
            fragment.parent_id = tracklet.parent_id
            fragments.append(fragment)
            current_id += 1

        if len(fragments) < 2:
            return None
        return fragments

    # ------------------------------------------------------------------
    # Single-pass scan
    # ------------------------------------------------------------------

    def _detect_switches(self, jerseys, entropies, teams, team_confs):
        n = len(jerseys)
        switch_points = []

        # Establish initial references using the same logic as each splitter
        ref_jersey = self._establish_jersey_ref(jerseys, entropies)
        ref_team   = self._establish_team_ref(teams, team_confs)

        i = 0
        while i < n:
            # --- Jersey signal (same logic as JerseySplitter) ---
            jersey_fired = False
            if ref_jersey is not None and self._jersey_valid(jerseys[i], entropies[i]):
                candidate = jerseys[i]
                if (candidate != ref_jersey
                        and not self._digit_compatible(candidate, ref_jersey)
                        and self._jersey_persistent(jerseys, entropies, i, candidate)):
                    jersey_fired = True
                    new_jersey   = candidate

            # --- Team signal (same logic as TeamSplitter) ---
            team_fired = False
            if ref_team is not None and self._team_valid(teams[i], team_confs[i]):
                candidate = teams[i]
                if (candidate != ref_team
                        and self._team_persistent(teams, team_confs, i, candidate)):
                    team_fired = True
                    new_team   = candidate

            # --- Split if either fires ---
            if jersey_fired or team_fired:
                switch_points.append(i)
                # Update both references; keep old value if signal didn't fire
                ref_jersey = new_jersey if jersey_fired else ref_jersey
                ref_team   = new_team   if team_fired   else ref_team

            i += 1

        return switch_points

    # ------------------------------------------------------------------
    # Reference establishment (mirrors each splitter's init scan)
    # ------------------------------------------------------------------

    def _establish_jersey_ref(self, jerseys, entropies):
        n = len(jerseys)
        for i in range(n):
            if self._jersey_valid(jerseys[i], entropies[i]):
                if self._jersey_persistent(jerseys, entropies, i, jerseys[i]):
                    return jerseys[i]
        return None

    def _establish_team_ref(self, teams, team_confs):
        n = len(teams)
        for i in range(n):
            if self._team_valid(teams[i], team_confs[i]):
                if self._team_persistent(teams, team_confs, i, teams[i]):
                    return teams[i]
        return None

    # ------------------------------------------------------------------
    # Persistence checks — identical to JerseySplitter / TeamSplitter
    # ------------------------------------------------------------------

    # def _jersey_persistent(self, jerseys, entropies, start, candidate):
    #     lookahead  = self.config.jersey_lookahead
    #     window_end = min(start + lookahead, len(jerseys))
    #     window = [jerseys[k] for k in range(start, window_end)
    #               if self._jersey_valid(jerseys[k], entropies[k])]
    #     if not window:
    #         return False
    #     count = sum(1 for j in window if j == candidate)
    #     if count < self.config.jersey_min_persistence:
    #         return False
    #     return Counter(window).most_common(1)[0][0] == candidate

    # TODO: test this persistent version without most common value, imagine if 
    # we have window=100, with reference number = 9, then 45 times number = 7, then 55 times number = 9. 
    # Most common would not split, but if the 45 times 7 appears close to each other it could very much be a short id switch
    # But i think we should only split when then we actually have a very high ratio, because if window = 100 and 20 frames is new number
    # then change is high that its just occlusion, so maybe its actually ok like this, TEST! ALSO FOR TEAMS

    def _jersey_persistent(self, jerseys, entropies, start, candidate):
        lookahead  = self.config.jersey_lookahead
        window_end = min(start + lookahead, len(jerseys))
        
        # Collect first jersey_min_persistence reliable predictions after start
        reliable = []
        for k in range(start, window_end):
            if self._jersey_valid(jerseys[k], entropies[k]):
                reliable.append(jerseys[k])
            if len(reliable) >= self.config.jersey_min_persistence:
                break
        
        if len(reliable) < self.config.jersey_min_persistence:
            return False
        
        # Check ratio of candidate in those first P_j reliable predictions
        ratio = sum(1 for j in reliable if j == candidate) / len(reliable)
        return ratio >= self.config.jersey_min_persistence_ratio


    def _team_persistent(self, teams, team_confs, start, candidate):
        lookahead  = self.config.team_lookahead
        window_end = min(start + lookahead, len(teams))
        window = [teams[k] for k in range(start, window_end)
                  if self._team_valid(teams[k], team_confs[k])]
        if not window:
            return False
        count = sum(1 for t in window if t == candidate)
        # if count < self.config.team_min_persistence:
        #     return False
        if count / len(window) < self.config.team_min_persistence_ratio:
            return False
        # return Counter(window).most_common(1)[0][0] == candidate
        return True

    # ------------------------------------------------------------------
    # Validity helpers — identical to each splitter
    # ------------------------------------------------------------------

    def _jersey_valid(self, value, entropy):
        if value is None:
            return False
        if isinstance(value, float) and np.isnan(value):
            return False
        return entropy <= self.config.jersey_entropy_threshold

    def _team_valid(self, value, confidence=1.0):
        if value is None:
            return False
        if isinstance(value, float) and np.isnan(value):
            return False
        if isinstance(confidence, float) and np.isnan(confidence):
            return False
        return confidence >= self.config.team_confidence_threshold

    @staticmethod
    def _digit_compatible(a, b):
        s_a, s_b = str(int(a)), str(int(b))
        if len(s_a) == 1 and len(s_b) == 2:
            return s_a in s_b
        if len(s_a) == 2 and len(s_b) == 1:
            return s_b in s_a
        return False
