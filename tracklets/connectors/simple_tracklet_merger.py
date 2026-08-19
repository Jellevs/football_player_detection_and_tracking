import numpy as np
from typing import Dict, Tuple, Optional
from collections import defaultdict


class SimpleTrackletMerger:
    """
    Two-pass tracklet merger:
      Pass 1: Merge tracklets with exact (jersey, team) identity match.
      Pass 2: Merge remaining tracklets using ReID similarity, but hard-block
              when confident jersey numbers or consistent team IDs conflict.
    """
    # 87.806
    def __init__(self, reid_threshold=0.4, jersey_entropy_threshold=0.2,
                 team_consistency_threshold=0.9, team_confidence_threshold=0.6,
                 min_jersey_predictions=5):
        self.reid_threshold = reid_threshold
        self.jersey_entropy_threshold = jersey_entropy_threshold
        self.team_consistency_threshold = team_consistency_threshold
        self.team_confidence_threshold = team_confidence_threshold
        self.min_jersey_predictions = min_jersey_predictions

    def merge(self, tracklets: Dict) -> Dict:
        if len(tracklets) < 2:
            return tracklets

        # Pass 1: Exact identity merge (jersey + team both known & match)
        identity_groups = defaultdict(list)
        remaining = []

        for tid, tracklet in tracklets.items():
            jersey, entropy = self._get_jersey_stats(tracklet)
            team, consistency = self._get_team_stats(tracklet)

            has_confident_jersey = jersey is not None
            has_team = team is not None

            if has_confident_jersey and has_team:
                identity_groups[(jersey, team)].append(tid)
            else:
                remaining.append(tid)

        # Merge exact-match groups
        merged_tracklets = {}
        next_id = 1

        for (jersey, team), member_ids in identity_groups.items():
            base, overflow = self._merge_group(tracklets, member_ids)
            merged_tracklets[next_id] = base
            next_id += 1
            for t in overflow:
                remaining.append(None)  # placeholder
                merged_tracklets[next_id] = t
                next_id += 1

        # Add remaining as-is for now
        for tid in remaining:
            if tid is not None:
                merged_tracklets[next_id] = tracklets[tid]
                next_id += 1

        # Pass 2: ReID-based merge with hard constraints
        merged_tracklets = self._reid_merge_pass(merged_tracklets)

        return merged_tracklets

    def _reid_merge_pass(self, tracklets: Dict) -> Dict:
        """Greedily merge tracklet pairs using ReID, respecting hard constraints."""
        # Build list of tracklet ids we can work with
        tid_list = sorted(tracklets.keys())
        merged_into = {}  # tid -> tid it was merged into

        # Greedy: find best ReID pair, merge, repeat
        changed = True
        while changed:
            changed = False
            best_dist = float('inf')
            best_pair = None

            active_ids = [t for t in tid_list if t not in merged_into]

            for i_idx in range(len(active_ids)):
                for j_idx in range(i_idx + 1, len(active_ids)):
                    ti = tracklets[active_ids[i_idx]]
                    tj = tracklets[active_ids[j_idx]]

                    # Hard block: temporal overlap
                    if set(ti.frames) & set(tj.frames):
                        continue

                    # Hard block: confident jersey mismatch
                    if self._jersey_conflict(ti, tj):
                        continue

                    # Hard block: consistent team mismatch
                    if self._team_conflict(ti, tj):
                        continue

                    # Compute ReID distance
                    if not ti.embeddings or not tj.embeddings:
                        continue

                    dist = self._reid_distance(ti, tj)

                    # Bonus: if both have matching jersey+team, reduce distance
                    j1, _ = self._get_jersey_stats(ti)
                    j2, _ = self._get_jersey_stats(tj)
                    t1, _ = self._get_team_stats(ti)
                    t2, _ = self._get_team_stats(tj)
                    if j1 is not None and j2 is not None and j1 == j2 and t1 == t2:
                        dist = 0.05  # guaranteed merge for exact identity

                    if dist < best_dist:
                        best_dist = dist
                        best_pair = (active_ids[i_idx], active_ids[j_idx])

            if best_pair and best_dist < self.reid_threshold:
                id_a, id_b = best_pair
                self._merge_tracklet_into(tracklets[id_a], tracklets[id_b])
                merged_into[id_b] = id_a
                changed = True

        # Build final output
        result = {}
        next_id = 1
        for tid in tid_list:
            if tid not in merged_into:
                result[next_id] = tracklets[tid]
                next_id += 1

        return result

    def _jersey_conflict(self, t1, t2) -> bool:
        """True if both have confident jerseys that differ."""
        j1, _ = self._get_jersey_stats(t1)
        j2, _ = self._get_jersey_stats(t2)
        return j1 is not None and j2 is not None and j1 != j2

    def _team_conflict(self, t1, t2) -> bool:
        """True if both have consistent teams that differ."""
        team1, cons1 = self._get_team_stats(t1)
        team2, cons2 = self._get_team_stats(t2)

        both_consistent = (team1 is not None and team2 is not None
                          and cons1 >= self.team_consistency_threshold
                          and cons2 >= self.team_consistency_threshold)

        return both_consistent and team1 != team2

    @staticmethod
    def _reid_distance(t1, t2) -> float:
        """Average pairwise cosine distance between two tracklets' embeddings."""
        feat1 = np.stack(t1.embeddings).astype(np.float32)
        feat2 = np.stack(t2.embeddings).astype(np.float32)
        norms1 = np.linalg.norm(feat1, axis=1, keepdims=True) + 1e-6
        norms2 = np.linalg.norm(feat2, axis=1, keepdims=True) + 1e-6
        cos_sim = (feat1 @ feat2.T) / (norms1 @ norms2.T)
        return float(1.0 - cos_sim.mean())

    def _get_jersey_stats(self, tracklet) -> Tuple[Optional[float], float]:
        """Get mode jersey number and its consistency, using only low-entropy predictions."""
        jerseys   = tracklet.pred_attributes.get('jerseys', [])
        entropies = tracklet.pred_attributes.get('jersey_entropies', [])

        valid = [
            jerseys[i] for i in range(len(jerseys))
            if jerseys[i] is not None
            and not (isinstance(jerseys[i], float) and np.isnan(jerseys[i]))
            and i < len(entropies)
            and entropies[i] <= self.jersey_entropy_threshold
        ]

        if len(valid) < self.min_jersey_predictions:
            return None, 0.0
        mode = max(set(valid), key=valid.count)
        consistency = valid.count(mode) / len(valid)
        return mode, consistency

    def _get_team_stats(self, tracklet) -> Tuple[Optional[float], float]:
        """Get mode team and its consistency, using only confident predictions."""
        teams      = tracklet.pred_attributes.get('teams', [])
        team_confs = tracklet.pred_attributes.get('team_confs', [])

        valid = [
            teams[i] for i in range(len(teams))
            if teams[i] is not None
            and not (isinstance(teams[i], float) and np.isnan(teams[i]))
            and i < len(team_confs)
            and team_confs[i] is not None
            and not (isinstance(team_confs[i], float) and np.isnan(team_confs[i]))
            and team_confs[i] >= self.team_confidence_threshold
        ]

        if not valid:
            return None, 0.0
        mode = max(set(valid), key=valid.count)
        consistency = valid.count(mode) / len(valid)
        return mode, consistency

    def _merge_group(self, tracklets, member_ids):
        """Merge a list of tracklets chronologically. Returns (base, overflow_list)."""
        member_ids.sort(key=lambda x: tracklets[x].frames[0])

        base = tracklets[member_ids[0]]
        current_frames = set(base.frames)
        overflow = []

        for next_tid in member_ids[1:]:
            other = tracklets[next_tid]

            if current_frames & set(other.frames):
                overflow.append(other)
                continue

            self._merge_tracklet_into(base, other)
            current_frames |= set(other.frames)

        return base, overflow

    @staticmethod
    def _merge_tracklet_into(base, other):
        """Append all data from `other` into `base`."""
        base.frames.extend(other.frames)
        base.bboxes.extend(other.bboxes)
        base.scores.extend(other.scores)
        base.embeddings.extend(other.embeddings)

        for key in base.pred_attributes:
            base.pred_attributes[key].extend(other.pred_attributes.get(key, []))

        for key in base.gt_attributes:
            base.gt_attributes[key].extend(other.gt_attributes.get(key, []))
