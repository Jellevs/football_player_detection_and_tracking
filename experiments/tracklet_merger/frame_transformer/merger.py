"""
Frame Transformer merger — drop-in replacement for TemporalBinMerger / XGBoostMerger.

Usage in main.py:
    from experiments.tracklet_merger.frame_transformer.merger import FrameTransformerMerger

    merger = FrameTransformerMerger(
        model_path  = "weights/frame_transformer/best_model.pt",
        meta_path   = "weights/frame_transformer/meta.json",
        merge_threshold = 0.50,
    )
    merged_tracklets = merger.merge(splitted_tracklets)

Unlike the temporal bin transformer which encodes each tracklet independently
and then classifies cached CLS pairs, this model requires a full forward pass
per pair because cross-attention between frames IS the key mechanism.
For N tracklets, this is N*(N-1)/2 forward passes. With batch processing and
small model size (d_model=128, 4 layers), this is tractable for typical
sequence sizes (~50 tracklets -> ~1225 pairs).
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

from experiments.tracklet_merger.frame_transformer.model import FramePairTransformer
from transformer.dataset import (
    extract_frame_features, compute_pairwise_features,
    _tracklet_stats, RAW_DIM, TRACKLET_STATS_DIM,
)


class FrameTransformerMerger:
    def __init__(
        self,
        model_path: Path,
        meta_path: Path,
        merge_threshold: float = 0.5,
        linkage_method: str = "average",
        jersey_entropy_threshold: float = 0.15,
        team_consistency_threshold: float = 0.9,
        team_confidence_threshold: float = 0.6,
        use_jersey_constraint: bool = True,
        use_team_constraint: bool = True,
        device: str = None,
    ):
        self.merge_threshold = merge_threshold
        self.linkage_method = linkage_method
        self.jersey_entropy_threshold = jersey_entropy_threshold
        self.team_consistency_threshold = team_consistency_threshold
        self.team_confidence_threshold = team_confidence_threshold
        self.use_jersey_constraint = use_jersey_constraint
        self.use_team_constraint = use_team_constraint

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model
        checkpoint = torch.load(str(model_path), map_location=self.device, weights_only=False)
        self.config = checkpoint["config"]
        self.model = FramePairTransformer(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Load metadata
        with open(meta_path) as f:
            meta = json.load(f)
        self.max_frames = meta.get("max_frames", self.config.max_frames_per_tracklet)
        self.min_frames = meta.get("min_frames", self.config.min_frames)

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

        # Pre-extract features for all tracklets
        tracklet_feats = {}
        for idx, tid in enumerate(tracklet_ids):
            feats, frames = extract_frame_features(tracklets[tid])
            tracklet_feats[idx] = (feats, frames)

        # Collect valid pairs and their data
        pair_indices = []
        pair_data = []

        for i in range(n):
            for j in range(i + 1, n):
                t_a = tracklets[tracklet_ids[i]]
                t_b = tracklets[tracklet_ids[j]]

                # Hard constraint: temporal overlap
                if set(t_a.frames) & set(t_b.frames):
                    dist_matrix[i, j] = dist_matrix[j, i] = 2.0
                    continue

                # Hard constraint: confident jersey mismatch
                if self.use_jersey_constraint and self._jersey_conflict(t_a, t_b):
                    dist_matrix[i, j] = dist_matrix[j, i] = 2.0
                    continue

                # Hard constraint: consistent team mismatch
                if self.use_team_constraint and self._team_conflict(t_a, t_b):
                    dist_matrix[i, j] = dist_matrix[j, i] = 2.0
                    continue

                # Extended pairwise features
                pw_base = compute_pairwise_features(t_a, t_b)
                feats_a, frames_a = tracklet_feats[i]
                feats_b, frames_b = tracklet_feats[j]
                stats_a = _tracklet_stats(feats_a, frames_a)
                stats_b = _tracklet_stats(feats_b, frames_b)
                extended_pw = np.concatenate([pw_base, stats_a, stats_b])

                pair_indices.append((i, j))
                pair_data.append({
                    "feats_a": feats_a,
                    "frames_a": frames_a,
                    "feats_b": feats_b,
                    "frames_b": frames_b,
                    "pairwise": extended_pw,
                })

        # Batch scoring
        if pair_data:
            probs = self._batch_score(pair_data)
            for (i, j), p in zip(pair_indices, probs):
                d = float(1.0 - p)
                dist_matrix[i, j] = dist_matrix[j, i] = d

        return dist_matrix

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _batch_score(self, pair_data, batch_size=64):
        """Score all pairs through the full model."""
        all_probs = []

        for start in range(0, len(pair_data), batch_size):
            end = min(start + batch_size, len(pair_data))
            batch = pair_data[start:end]

            # Adaptive frame sampling (deterministic for inference)
            sampled = []
            for pd in batch:
                fa, fra, fb, frb, pw = (
                    pd["feats_a"], pd["frames_a"],
                    pd["feats_b"], pd["frames_b"],
                    pd["pairwise"],
                )
                n_a, n_b = len(fa), len(fb)
                K = min(n_a, n_b, self.max_frames)
                K = max(K, self.min_frames)

                # Uniform deterministic sampling
                idx_a = np.round(np.linspace(0, n_a - 1, min(K, n_a))).astype(int)
                idx_b = np.round(np.linspace(0, n_b - 1, min(K, n_b))).astype(int)

                sampled.append({
                    "feats_a": fa[idx_a],
                    "frames_a": fra[idx_a],
                    "feats_b": fb[idx_b],
                    "frames_b": frb[idx_b],
                    "pairwise": pw,
                })

            # Pad and batch
            B = len(sampled)
            max_a = max(len(s["feats_a"]) for s in sampled)
            max_b = max(len(s["feats_b"]) for s in sampled)

            t_feats_a = torch.zeros(B, max_a, RAW_DIM, device=self.device)
            t_frames_a = torch.zeros(B, max_a, device=self.device)
            t_mask_a = torch.ones(B, max_a, dtype=torch.bool, device=self.device)

            t_feats_b = torch.zeros(B, max_b, RAW_DIM, device=self.device)
            t_frames_b = torch.zeros(B, max_b, device=self.device)
            t_mask_b = torch.ones(B, max_b, dtype=torch.bool, device=self.device)

            t_pw = torch.zeros(B, 34, device=self.device)

            for i, s in enumerate(sampled):
                na = len(s["feats_a"])
                nb = len(s["feats_b"])
                t_feats_a[i, :na] = torch.tensor(s["feats_a"], dtype=torch.float32)
                t_frames_a[i, :na] = torch.tensor(s["frames_a"], dtype=torch.float32)
                t_mask_a[i, :na] = False
                t_feats_b[i, :nb] = torch.tensor(s["feats_b"], dtype=torch.float32)
                t_frames_b[i, :nb] = torch.tensor(s["frames_b"], dtype=torch.float32)
                t_mask_b[i, :nb] = False
                t_pw[i] = torch.tensor(s["pairwise"], dtype=torch.float32)

            logits = self.model(
                t_feats_a, t_frames_a, t_mask_a,
                t_feats_b, t_frames_b, t_mask_b,
                t_pw,
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())

        return all_probs

    # ------------------------------------------------------------------
    # Hard-constraint helpers (identical to other mergers)
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
    # Merge application (identical to other mergers)
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
