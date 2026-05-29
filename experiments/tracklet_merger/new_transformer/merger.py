"""
Late Cross-Attention Transformer merger -- drop-in replacement for XGBoostMerger.

Key improvements over the temporal bin transformer merger (Plan Sections 4-5):

B3. Two-stage inference with cross-attention:
    1. Encode each tracklet ONCE -> cache full token set (not just CLS)
    2. For each pair: run cross-attention over cached token sets + classifier

C3. Per-sequence affinity normalization:
    - Rank-normalize or z-score the distance matrix within each sequence
      before fcluster. This targets the exact "absolute precision at the
      boundary" problem and is robust to scene-to-scene scale shifts.
    - Supports average, complete, and ward linkage methods.

Usage:
    from experiments.tracklet_merger.new_transformer.merger import NewTransformerMerger

    merger = NewTransformerMerger(
        model_path  = "weights/new_transformer/best_model.pt",
        meta_path   = "weights/new_transformer/meta.json",
        merge_threshold = 0.5,
    )
    merged_tracklets = merger.merge(splitted_tracklets)
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

from experiments.tracklet_merger.new_transformer.model import LateCrossAttentionTransformer
from experiments.tracklet_merger.new_transformer.dataset import (
    NewTransformerDataset, _PW_SCALE, EXTENDED_PAIRWISE_DIM,
)
from transformer.dataset import (
    extract_frame_features, compute_pairwise_features,
    _tracklet_stats, RAW_DIM, TRACKLET_STATS_DIM,
)


class NewTransformerMerger:
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
        # C3: Per-sequence affinity normalization
        normalization_method: str = "none",  # "rank", "zscore", or "none"
        # Calibration
        calibration_temperature: float = None,
        calibration_platt: tuple = None,
        device: str = None,
    ):
        self.merge_threshold = merge_threshold
        self.linkage_method = linkage_method
        self.jersey_entropy_threshold = jersey_entropy_threshold
        self.team_consistency_threshold = team_consistency_threshold
        self.team_confidence_threshold = team_confidence_threshold
        self.use_jersey_constraint = use_jersey_constraint
        self.use_team_constraint = use_team_constraint
        self.normalization_method = normalization_method
        self.calibration_temperature = calibration_temperature
        self.calibration_platt = calibration_platt

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model
        checkpoint = torch.load(str(model_path), map_location=self.device, weights_only=False)
        self.config = checkpoint["config"]
        self.model = LateCrossAttentionTransformer(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Load metadata
        with open(meta_path) as f:
            meta = json.load(f)
        self.n_bins = meta.get("n_temporal_bins", self.config.n_temporal_bins)
        self.boundary_k = meta.get("boundary_k", self.config.boundary_k)
        self.max_frame_value = meta.get("max_frame_value", self.config.max_frame_value)

        # Dummy dataset for bin representation
        self._dummy_dataset = NewTransformerDataset.__new__(NewTransformerDataset)
        self._dummy_dataset.n_bins = self.n_bins
        self._dummy_dataset.boundary_k = self.boundary_k
        self._dummy_dataset.max_frame_value = self.max_frame_value

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def merge(self, tracklets: Dict) -> Dict:
        tracklet_ids = sorted(tracklets.keys())
        n = len(tracklet_ids)
        if n < 2:
            return tracklets

        dist_matrix = self._build_distance_matrix(tracklets, tracklet_ids)

        # C3: Per-sequence affinity normalization
        dist_matrix = self._normalize_distances(dist_matrix)

        condensed = squareform(dist_matrix)
        Z = linkage(condensed, method=self.linkage_method)
        labels = fcluster(Z, t=self.merge_threshold, criterion="distance")

        return self._apply_merges(tracklets, tracklet_ids, labels)

    # ------------------------------------------------------------------
    # Distance matrix with two-stage inference
    # ------------------------------------------------------------------

    def _build_distance_matrix(self, tracklets, tracklet_ids):
        n = len(tracklet_ids)
        dist_matrix = np.ones((n, n))
        np.fill_diagonal(dist_matrix, 0.0)

        # Stage 1: Encode all tracklets once -> full token sets
        token_cache = {}
        for idx, tid in enumerate(tracklet_ids):
            token_cache[idx] = self._encode_tracklet_tokens(tracklets[tid])

        # Stage 2: Score all valid pairs using cross-attention
        pair_indices = []
        pair_pairwise = []

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

                # Extended pairwise features (with same normalization as training)
                pw_base = compute_pairwise_features(t_a, t_b)
                feats_a, frames_a = extract_frame_features(t_a)
                feats_b, frames_b = extract_frame_features(t_b)
                stats_a = _tracklet_stats(feats_a, frames_a)
                stats_b = _tracklet_stats(feats_b, frames_b)
                extended_pw = np.concatenate([pw_base, stats_a, stats_b])
                extended_pw = np.where(np.isfinite(extended_pw), extended_pw, 0.0).astype(np.float32)
                extended_pw = extended_pw / _PW_SCALE
                extended_pw = np.clip(extended_pw, -50.0, 50.0)

                pair_indices.append((i, j))
                pair_pairwise.append(extended_pw)

        # Batch classify using cross-attention
        if pair_indices:
            probs = self._batch_classify_cross(token_cache, pair_indices, pair_pairwise)
            for (i, j), p in zip(pair_indices, probs):
                d = float(1.0 - p)
                dist_matrix[i, j] = dist_matrix[j, i] = d

        return dist_matrix

    # ------------------------------------------------------------------
    # C3: Per-sequence affinity normalization
    # ------------------------------------------------------------------

    def _normalize_distances(self, dist_matrix: np.ndarray) -> np.ndarray:
        """
        Normalize the distance matrix within the sequence.

        Methods:
        - "rank": Rank-normalize distances to [0, 1] -- most robust
        - "zscore": Z-score normalize non-constrained distances
        - "none": No normalization
        """
        if self.normalization_method == "none":
            return dist_matrix

        n = dist_matrix.shape[0]
        if n < 3:
            return dist_matrix

        # Extract upper triangle (non-diagonal, non-hard-constrained)
        mask = np.zeros_like(dist_matrix, dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                if dist_matrix[i, j] < 1.5:  # not hard-constrained (2.0)
                    mask[i, j] = True

        valid_dists = dist_matrix[mask]
        if len(valid_dists) < 2:
            return dist_matrix

        result = dist_matrix.copy()

        if self.normalization_method == "rank":
            # Rank-normalize valid distances to [0, 1]
            ranks = rankdata(valid_dists) / len(valid_dists)
            result[mask] = ranks
            # Mirror
            for i in range(n):
                for j in range(i + 1, n):
                    result[j, i] = result[i, j]

        elif self.normalization_method == "zscore":
            # Z-score normalize, then rescale to [0, 1] via sigmoid
            mean = valid_dists.mean()
            std = valid_dists.std() + 1e-8
            z_scores = (valid_dists - mean) / std
            # Map to [0, 1] using sigmoid
            normalized = 1.0 / (1.0 + np.exp(-z_scores))
            result[mask] = normalized
            for i in range(n):
                for j in range(i + 1, n):
                    result[j, i] = result[i, j]

        return result

    # ------------------------------------------------------------------
    # Encoding & classification
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_tracklet_tokens(self, tracklet) -> torch.Tensor:
        """Encode a single tracklet -> full token set (n_tokens, d_model)."""
        feats, frames = extract_frame_features(tracklet)
        bins = self._dummy_dataset._create_bin_representation(feats, frames)
        bins = bins.unsqueeze(0).to(self.device)
        tokens = self.model.encode_tracklet(bins)  # (1, n_tokens, d_model)
        return tokens.squeeze(0)  # (n_tokens, d_model)

    @torch.no_grad()
    def _batch_classify_cross(self, token_cache, pair_indices, pair_pairwise):
        """Run cross-attention + classifier on batches of pre-encoded token sets."""
        batch_size = 128
        all_probs = []

        for start in range(0, len(pair_indices), batch_size):
            end = min(start + batch_size, len(pair_indices))
            batch_idx = pair_indices[start:end]
            batch_pw = pair_pairwise[start:end]

            tokens_a = torch.stack([token_cache[i] for i, j in batch_idx]).to(self.device)
            tokens_b = torch.stack([token_cache[j] for i, j in batch_idx]).to(self.device)
            pw = torch.tensor(np.stack(batch_pw), dtype=torch.float32).to(self.device)

            logits = self.model.forward_from_cached(tokens_a, tokens_b, pw)

            # Apply calibration if configured
            if self.calibration_platt is not None:
                a, b = self.calibration_platt
                probs = torch.sigmoid(a * logits + b).cpu().numpy()
            elif self.calibration_temperature is not None:
                probs = torch.sigmoid(logits / self.calibration_temperature).cpu().numpy()
            else:
                probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.extend(probs.tolist())

        return all_probs

    # ------------------------------------------------------------------
    # Hard-constraint helpers (identical to XGBoostMerger)
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
