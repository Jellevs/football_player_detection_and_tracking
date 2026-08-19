"""
Rule-based tracklet merger with ReID fallback.

Decision for each pair (in order):

  1. Temporal overlap              → block
  2. Jersey conflict               → block  (both confident, numbers differ)
  3. Jersey match + team match     → merge  (both confident, both agree)
  4. Jersey match + team conflict  → block  (jersey agrees, teams definitely differ)
  5. Jersey match + team unclear   → reid   (jersey agrees, team not confident for both)
  6. Jersey unclear                → reid   (no confident jersey for at least one tracklet)
"""

import csv
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union


class DecisionMerger:
    def __init__(
        self,
        reid_threshold: float = 0.4,
        jersey_entropy_threshold: float = 0.1,
        min_jersey_predictions: int = 5,
        team_confidence_threshold: float = 0.5,
        team_consistency_threshold: float = 0.9,
        log_path: Optional[Union[Path, str]] = None,
    ):
        self.reid_threshold = reid_threshold
        self.jersey_entropy_threshold = jersey_entropy_threshold
        self.min_jersey_predictions = min_jersey_predictions
        self.team_confidence_threshold = team_confidence_threshold
        self.team_consistency_threshold = team_consistency_threshold
        self.log_path = Path(log_path) if log_path is not None else None

    def merge(self, tracklets: Dict, sequence_name: str = "") -> Dict:
        if len(tracklets) < 2:
            return tracklets

        tid_list = sorted(tracklets.keys())
        merged_into = {}

        changed = True
        while changed:
            changed = False
            best_dist = float('inf')
            best_pair = None

            active = [t for t in tid_list if t not in merged_into]

            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    ti = tracklets[active[i]]
                    tj = tracklets[active[j]]

                    decision = self._pair_decision(ti, tj)

                    if decision == 'block':
                        continue
                    elif decision == 'merge':
                        dist = 0.0  # guaranteed merge, always selected first
                    else:  # 'reid'
                        if not ti.embeddings or not tj.embeddings:
                            continue
                        dist = self._reid_distance(ti, tj)
                        if dist >= self.reid_threshold:
                            continue

                    if dist < best_dist:
                        best_dist = dist
                        best_pair = (active[i], active[j])

            if best_pair is not None:
                id_a, id_b = best_pair
                self._log_merge(sequence_name, id_a, id_b, tracklets[id_a], tracklets[id_b])
                self._merge_into(tracklets[id_a], tracklets[id_b])
                merged_into[id_b] = id_a
                changed = True

        result = {}
        next_id = 1
        for tid in tid_list:
            if tid not in merged_into:
                result[next_id] = tracklets[tid]
                next_id += 1
        return result

    def _log_merge(self, sequence_name, base_id, absorbed_id, base, absorbed):
        if self.log_path is None:
            return
        write_header = not self.log_path.exists()
        with self.log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "merger_type", "sequence",
                    "base_id", "absorbed_id",
                    "base_start_frame", "base_end_frame",
                    "absorbed_start_frame", "absorbed_end_frame",
                ])
            writer.writerow([
                "DecisionMerger", sequence_name,
                base_id, absorbed_id,
                base.frames[0], base.frames[-1],
                absorbed.frames[0], absorbed.frames[-1],
            ])

    # Decision tree

    def _pair_decision(self, t1, t2) -> str:
        """Returns 'block', 'merge', or 'reid'."""
        if set(t1.frames) & set(t2.frames):
            return 'block'

        j1 = self._get_jersey(t1)
        j2 = self._get_jersey(t2)

        if j1 is not None and j2 is not None:
            if j1 != j2:
                return 'block'                  # jersey conflict

            # Same jersey, check team
            team1, cons1 = self._get_team(t1)
            team2, cons2 = self._get_team(t2)
            t1_conf = team1 is not None and cons1 >= self.team_consistency_threshold
            t2_conf = team2 is not None and cons2 >= self.team_consistency_threshold

            if t1_conf and t2_conf:
                return 'merge' if team1 == team2 else 'block'

            # Jersey matches but team is not confident for both → fall back to ReID
            return 'reid'

        # At least one jersey is not confident → fall back to ReID 
        return 'reid'

    # Identity extraction

    def _get_jersey(self, tracklet) -> Optional[int]:
        """Mode jersey number if enough low-entropy predictions exist, else None."""
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
            return None
        return max(set(valid), key=valid.count)

    def _get_team(self, tracklet) -> Tuple[Optional[int], float]:
        """Mode team and its consistency, using only high-confidence predictions."""
        teams      = tracklet.pred_attributes.get('teams', [])
        team_confs = tracklet.pred_attributes.get('team_confs', [])
        valid = [
            teams[i] for i in range(len(teams))
            if teams[i] is not None
            and not (isinstance(teams[i], float) and np.isnan(teams[i]))
            and i < len(team_confs)
            and team_confs[i] is not None
            and not (isinstance(team_confs[i], float) and np.isnan(team_confs[i]))
            and float(team_confs[i]) >= self.team_confidence_threshold
        ]
        if not valid:
            return None, 0.0
        mode = max(set(valid), key=valid.count)
        return mode, valid.count(mode) / len(valid)

    # ReID distance

    @staticmethod
    def _reid_distance(t1, t2) -> float:
        """Cosine distance between mean normalized tracklet embeddings."""
        feat1 = np.stack(t1.embeddings).astype(np.float32)
        feat2 = np.stack(t2.embeddings).astype(np.float32)
        norms1 = np.linalg.norm(feat1, axis=1, keepdims=True) + 1e-6
        norms2 = np.linalg.norm(feat2, axis=1, keepdims=True) + 1e-6
        mean1 = (feat1 / norms1).mean(axis=0)
        mean2 = (feat2 / norms2).mean(axis=0)
        mean1 /= np.linalg.norm(mean1) + 1e-6
        mean2 /= np.linalg.norm(mean2) + 1e-6
        return float(1.0 - np.dot(mean1, mean2))

    # Merge utility

    @staticmethod
    def _merge_into(base, other):
        """Append all data from `other` into `base`."""
        base.frames.extend(other.frames)
        base.bboxes.extend(other.bboxes)
        base.scores.extend(other.scores)
        base.embeddings.extend(other.embeddings)
        for key in base.pred_attributes:
            base.pred_attributes[key].extend(other.pred_attributes.get(key, []))
        for key in base.gt_attributes:
            base.gt_attributes[key].extend(other.gt_attributes.get(key, []))
