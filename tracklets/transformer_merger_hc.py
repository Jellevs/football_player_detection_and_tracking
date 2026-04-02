"""
transformer_merger_hc.py

Hierarchical-clustering version of the Transformer merger.

Uses the Siamese CLS Transformer purely as an *encoder*: each tracklet is
encoded once into a CLS embedding, and pairwise cosine similarity between
those embeddings is used as the similarity score fed into average-linkage
hierarchical clustering — identical pipeline to XGBoostMerger, just a
different similarity-score source.

Usage in main.py:
    from tracklets.transformer_merger_hc import TransformerMergerHC

    merger = TransformerMergerHC(
        model_path = "weights/transformer/best_model.pt",
        meta_path  = "weights/transformer/transformer_merger_meta.json",
        merge_threshold = 0.5,   # distance threshold (= 1 - cosine_sim)
    )
    merged_tracklets = merger.merge(splitted_tracklets)
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from transformer.model import SiameseCLSTransformer
from transformer.dataset import extract_frame_features, RAW_DIM


class TransformerMergerHC:
    def __init__(
        self,
        model_path: Path,
        meta_path: Path,
        merge_threshold: float = 0.5,
        linkage_method: str = "average",
        jersey_entropy_threshold: float = 0.15,
        team_consistency_threshold: float = 0.9,
        device: str = None,
    ):
        self.merge_threshold = merge_threshold
        self.linkage_method = linkage_method
        self.jersey_entropy_threshold = jersey_entropy_threshold
        self.team_consistency_threshold = team_consistency_threshold

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        checkpoint = torch.load(str(model_path), map_location=self.device, weights_only=False)
        self.config = checkpoint["config"]
        self.model = SiameseCLSTransformer(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        with open(meta_path) as f:
            meta = json.load(f)
        self.t_max = meta.get("t_max", self.config.t_max)
        self.max_frame_value = meta.get("max_frame_value", self.config.max_frame_value)

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
    # Distance matrix: encode all tracklets once, then cosine similarity
    # ------------------------------------------------------------------

    def _build_distance_matrix(self, tracklets, tracklet_ids):
        n = len(tracklet_ids)
        dist_matrix = np.ones((n, n))
        np.fill_diagonal(dist_matrix, 0.0)

        # Encode all tracklets into L2-normalised CLS embeddings
        cls_embeddings = {}
        for idx, tid in enumerate(tracklet_ids):
            cls_embeddings[idx] = self._encode_tracklet(tracklets[tid])  # (d_model,) normalised

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

                # Cosine similarity → distance
                sim = float(torch.dot(cls_embeddings[i], cls_embeddings[j]).item())
                sim = max(-1.0, min(1.0, sim))   # numerical safety
                d = 1.0 - sim                     # maps [-1,1] → [0,2]; same scale as hard constraints
                dist_matrix[i, j] = dist_matrix[j, i] = d

        return dist_matrix

    # ------------------------------------------------------------------
    # Transformer encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_tracklet(self, tracklet) -> torch.Tensor:
        """Encode a single tracklet → L2-normalised CLS embedding (d_model,)."""
        features, frames = extract_frame_features(tracklet)
        tokens, mask, positions, frame_nums = self._prepare_sequence(features, frames)

        tokens = tokens.unsqueeze(0).to(self.device)
        mask = mask.unsqueeze(0).to(self.device)
        positions = positions.unsqueeze(0).to(self.device)
        frame_nums = frame_nums.unsqueeze(0).to(self.device)

        cls = self.model.encode_tracklet(tokens, mask, positions, frame_nums)
        cls = cls.squeeze(0)                          # (d_model,)
        return F.normalize(cls, dim=0)                # L2-normalise for cosine sim

    def _prepare_sequence(self, features, frames):
        n = len(frames)

        if n > self.t_max:
            indices = np.linspace(0, n - 1, self.t_max, dtype=int)
            features = features[indices]
            frames = frames[indices]
            n = self.t_max

        padded = np.zeros((self.t_max, RAW_DIM), dtype=np.float32)
        padded[:n] = features

        mask = np.ones(self.t_max, dtype=bool)
        mask[:n] = False

        positions = np.zeros(self.t_max, dtype=np.float32)
        if n > 1:
            positions[:n] = np.arange(n, dtype=np.float32) / (n - 1)

        frame_nums = np.zeros(self.t_max, dtype=np.float32)
        frame_nums[:n] = frames / self.max_frame_value

        return (
            torch.from_numpy(padded),
            torch.from_numpy(mask),
            torch.from_numpy(positions),
            torch.from_numpy(frame_nums),
        )

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

    @staticmethod
    def _team_stats(tracklet):
        teams = [t for t in tracklet.pred_attributes.get("teams", [])
                 if not (isinstance(t, float) and np.isnan(t))]
        if not teams:
            return None, 0.0
        mode = max(set(teams), key=teams.count)
        return mode, teams.count(mode) / len(teams)

    # ------------------------------------------------------------------
    # Merge application (identical to XGBoostMerger)
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
