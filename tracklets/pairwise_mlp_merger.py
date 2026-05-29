"""
pairwise_mlp_merger.py

Drop-in replacement merger using the Pairwise MLP model.

Since the MLP only uses 12 pairwise features (no per-frame data), this is
very fast — no sequence extraction or padding needed.

Usage in main.py:
    from tracklets.pairwise_mlp_merger import PairwiseMLPMerger

    merger = PairwiseMLPMerger(
        model_path  = "path/to/best_model.pt",
        merge_threshold = 0.5,
    )
    merged_tracklets = merger.merge(splitted_tracklets)
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from experiments.tracklet_merger.transformers.pairwise_mlp.model import PairwiseMLP
from experiments.tracklet_merger.transformers.utils import compute_pairwise_features


class PairwiseMLPMerger:
    def __init__(
        self,
        model_path: Path,
        merge_threshold: float = 0.5,
        linkage_method: str = "average",
        jersey_entropy_threshold: float = 0.15,
        team_consistency_threshold: float = 0.9,
        team_confidence_threshold: float = 0.6,
        device: str = None,
    ):
        self.merge_threshold = merge_threshold
        self.linkage_method = linkage_method
        self.jersey_entropy_threshold = jersey_entropy_threshold
        self.team_consistency_threshold = team_consistency_threshold
        self.team_confidence_threshold = team_confidence_threshold

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model
        checkpoint = torch.load(str(model_path), map_location=self.device, weights_only=False)
        self.model = PairwiseMLP(pairwise_dim=12, dropout=0.0).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def merge(self, tracklets: Dict) -> Dict:
        tracklet_ids = sorted(tracklets.keys())
        n = len(tracklet_ids)
        if n < 2:
            return tracklets

        dist_matrix = self._build_distance_matrix(tracklets, tracklet_ids)

        condensed = squareform(dist_matrix)
        Z = linkage(condensed, method=self.linkage_method)
        labels = fcluster(Z, t=self.merge_threshold, criterion="distance")

        return self._apply_merges(tracklets, tracklet_ids, labels)

    # ------------------------------------------------------------------
    # Distance matrix
    # ------------------------------------------------------------------

    def _build_distance_matrix(self, tracklets, tracklet_ids):
        n = len(tracklet_ids)
        dist_matrix = np.ones((n, n))
        np.fill_diagonal(dist_matrix, 0.0)

        pair_indices = []
        pair_pw = []

        for i in range(n):
            for j in range(i + 1, n):
                t_a = tracklets[tracklet_ids[i]]
                t_b = tracklets[tracklet_ids[j]]

                # Hard constraint: temporal overlap
                if set(t_a.frames) & set(t_b.frames):
                    dist_matrix[i, j] = dist_matrix[j, i] = 2.0
                    continue

                # Hard constraint: confident jersey mismatch
                if self._jersey_conflict(t_a, t_b):
                    dist_matrix[i, j] = dist_matrix[j, i] = 2.0
                    continue

                # Hard constraint: consistent team mismatch
                if self._team_conflict(t_a, t_b):
                    dist_matrix[i, j] = dist_matrix[j, i] = 2.0
                    continue

                pw = compute_pairwise_features(t_a, t_b)
                pair_indices.append((i, j))
                pair_pw.append(pw)

        # Batch score all valid pairs
        if pair_indices:
            probs = self._batch_score(pair_pw)
            for (i, j), p in zip(pair_indices, probs):
                d = float(1.0 - p)
                dist_matrix[i, j] = dist_matrix[j, i] = d

        return dist_matrix

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _batch_score(self, pairwise_features):
        """Score all pairs in one batch (tiny model, no need to chunk)."""
        pw = torch.tensor(
            np.stack(pairwise_features), dtype=torch.float32
        ).to(self.device)

        logits = self.model(pw)
        probs = torch.sigmoid(logits).cpu().numpy()
        return probs.tolist()

    # ------------------------------------------------------------------
    # Hard-constraint helpers
    # ------------------------------------------------------------------

    def _jersey_conflict(self, t1, t2) -> bool:
        j1, e1 = self._jersey_stats(t1)
        j2, e2 = self._jersey_stats(t2)
        both_conf = (j1 is not None and j2 is not None
                     and e1 < self.jersey_entropy_threshold
                     and e2 < self.jersey_entropy_threshold)
        return bool(both_conf and j1 != j2)

    def _team_conflict(self, t1, t2) -> bool:
        tm1, c1 = self._team_stats(t1)
        tm2, c2 = self._team_stats(t2)
        both_cons = (tm1 is not None and tm2 is not None
                     and c1 >= self.team_consistency_threshold
                     and c2 >= self.team_consistency_threshold)
        return bool(both_cons and tm1 != tm2)

    @staticmethod
    def _jersey_stats(tracklet):
        jerseys = tracklet.pred_attributes.get("jerseys", [])
        entropies = tracklet.pred_attributes.get("jersey_entropies", [])
        valid = [(j, entropies[i] if i < len(entropies) else 1.0)
                 for i, j in enumerate(jerseys)
                 if not (isinstance(j, float) and np.isnan(j))]
        if not valid:
            return None, 1.0
        js, es = zip(*valid)
        mode = max(set(js), key=js.count)
        return mode, float(np.mean([e for j, e in zip(js, es) if j == mode]))

    def _team_stats(self, tracklet):
        teams = tracklet.pred_attributes.get("teams", [])
        team_confs = tracklet.pred_attributes.get("team_confs", [])
        valid = [
            teams[i] for i in range(len(teams))
            if not (isinstance(teams[i], float) and np.isnan(teams[i]))
            and i < len(team_confs)
            and team_confs[i] is not None
            and not (isinstance(team_confs[i], float) and np.isnan(team_confs[i]))
            and float(team_confs[i]) >= self.team_confidence_threshold
        ]
        if not valid:
            return None, 0.0
        mode = max(set(valid), key=valid.count)
        return mode, valid.count(mode) / len(valid)

    # ------------------------------------------------------------------
    # Merge application
    # ------------------------------------------------------------------

    def _apply_merges(self, tracklets, tracklet_ids, cluster_labels):
        clusters = {}
        for idx, cid in enumerate(cluster_labels):
            clusters.setdefault(cid, []).append(tracklet_ids[idx])

        result = {}
        overflow_id = max(clusters.keys()) + 1

        for cid, members in clusters.items():
            members.sort(key=lambda x: tracklets[x].frames[0])
            base = tracklets[members[0]]
            current_frames = set(base.frames)

            for next_id in members[1:]:
                other = tracklets[next_id]
                if current_frames & set(other.frames):
                    result[overflow_id] = other
                    overflow_id += 1
                    continue
                base.frames.extend(other.frames)
                base.bboxes.extend(other.bboxes)
                base.scores.extend(other.scores)
                base.embeddings.extend(other.embeddings)
                for k in base.pred_attributes:
                    base.pred_attributes[k].extend(other.pred_attributes.get(k, []))
                current_frames |= set(other.frames)

            result[cid] = base

        return result
