"""
Ensemble merger: blend Transformer + XGBoost probabilities in logit space.

Plan Section 4, B2 — "near-guaranteed win, do this first":
    Blend transformer and XGBoost probabilities in logit space, then sweep
    the blend weight and merge threshold against HOTA.  Because XGBoost is
    already strong and the transformer makes *different* errors, the ensemble
    almost certainly beats either alone.

Usage:
    from experiments.tracklet_merger.new_transformer.ensemble import EnsembleMerger

    merger = EnsembleMerger(
        transformer_model_path = "weights/new_transformer/best_model.pt",
        transformer_meta_path  = "weights/new_transformer/meta.json",
        xgboost_model_path     = "weights/xgboost/xgboost_merger.json",
        xgboost_meta_path      = "weights/xgboost/xgboost_merger_meta.json",
        blend_weight           = 0.5,   # 0 = pure XGBoost, 1 = pure Transformer
        merge_threshold        = 0.5,
    )
    merged_tracklets = merger.merge(splitted_tracklets)

Logit-space blending:
    logit_blend = (1 - w) * logit_xgb + w * logit_transformer
    prob_blend  = sigmoid(logit_blend)
    distance    = 1 - prob_blend
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Union
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from scipy.special import logit as sp_logit, expit as sp_expit

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

from experiments.tracklet_merger.new_transformer.model import LateCrossAttentionTransformer
from experiments.tracklet_merger.new_transformer.dataset import (
    NewTransformerDataset, _PW_SCALE, EXTENDED_PAIRWISE_DIM,
)
from tracklets.xgboost_merger import XGBoostMerger
from transformer.dataset import (
    extract_frame_features, compute_pairwise_features,
    _tracklet_stats, RAW_DIM, TRACKLET_STATS_DIM,
)


def _safe_logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Numerically safe logit transform."""
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


class EnsembleMerger:
    """
    Ensemble of Late Cross-Attention Transformer + XGBoost.

    Blends in logit space for well-calibrated combination:
        logit_blend = (1 - blend_weight) * logit_xgb + blend_weight * logit_transformer
    """

    def __init__(
        self,
        # Transformer paths
        transformer_model_path: Path,
        transformer_meta_path: Path,
        # XGBoost paths
        xgboost_model_path: Union[Path, str, List[Union[Path, str]]],
        xgboost_meta_path: Path,
        # Ensemble config
        blend_weight: float = 0.5,
        merge_threshold: float = 0.5,
        linkage_method: str = "average",
        # Hard constraints
        jersey_entropy_threshold: float = 0.15,
        team_consistency_threshold: float = 0.9,
        team_confidence_threshold: float = 0.6,
        use_jersey_constraint: bool = True,
        use_team_constraint: bool = True,
        # C3: Normalization
        normalization_method: str = "none",
        device: str = None,
    ):
        self.blend_weight = blend_weight
        self.merge_threshold = merge_threshold
        self.linkage_method = linkage_method
        self.jersey_entropy_threshold = jersey_entropy_threshold
        self.team_consistency_threshold = team_consistency_threshold
        self.team_confidence_threshold = team_confidence_threshold
        self.use_jersey_constraint = use_jersey_constraint
        self.use_team_constraint = use_team_constraint
        self.normalization_method = normalization_method

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ---- Load Transformer ----
        checkpoint = torch.load(
            str(transformer_model_path), map_location=self.device, weights_only=False
        )
        self.transformer_config = checkpoint["config"]
        self.transformer = LateCrossAttentionTransformer(self.transformer_config).to(self.device)
        self.transformer.load_state_dict(checkpoint["model_state_dict"])
        self.transformer.eval()

        with open(transformer_meta_path) as f:
            t_meta = json.load(f)
        self.n_bins = t_meta.get("n_temporal_bins", self.transformer_config.n_temporal_bins)
        self.boundary_k = t_meta.get("boundary_k", self.transformer_config.boundary_k)
        self.max_frame_value = t_meta.get("max_frame_value", self.transformer_config.max_frame_value)

        # Dummy dataset for bin creation
        self._dummy_dataset = NewTransformerDataset.__new__(NewTransformerDataset)
        self._dummy_dataset.n_bins = self.n_bins
        self._dummy_dataset.boundary_k = self.boundary_k
        self._dummy_dataset.max_frame_value = self.max_frame_value

        # ---- Load XGBoost ----
        self.xgboost_merger = XGBoostMerger(
            model_path=xgboost_model_path,
            meta_path=xgboost_meta_path,
            merge_threshold=merge_threshold,
            linkage_method=linkage_method,
            jersey_entropy_threshold=jersey_entropy_threshold,
            team_consistency_threshold=team_consistency_threshold,
            team_confidence_threshold=team_confidence_threshold,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def merge(self, tracklets: Dict) -> Dict:
        tracklet_ids = sorted(tracklets.keys())
        n = len(tracklet_ids)
        if n < 2:
            return tracklets

        dist_matrix = self._build_distance_matrix(tracklets, tracklet_ids)

        # C3: Normalization
        dist_matrix = self._normalize_distances(dist_matrix)

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

        # Stage 1: Encode all tracklets for transformer
        token_cache = {}
        for idx, tid in enumerate(tracklet_ids):
            token_cache[idx] = self._encode_tracklet_tokens(tracklets[tid])

        # Stage 2: Score all pairs with both models
        pair_indices = []
        pair_pw_transformer = []  # extended pairwise for transformer

        for i in range(n):
            for j in range(i + 1, n):
                t_a = tracklets[tracklet_ids[i]]
                t_b = tracklets[tracklet_ids[j]]

                # Hard constraints
                if set(t_a.frames) & set(t_b.frames):
                    dist_matrix[i, j] = dist_matrix[j, i] = 2.0
                    continue
                if self.use_jersey_constraint and self._jersey_conflict(t_a, t_b):
                    dist_matrix[i, j] = dist_matrix[j, i] = 2.0
                    continue
                if self.use_team_constraint and self._team_conflict(t_a, t_b):
                    dist_matrix[i, j] = dist_matrix[j, i] = 2.0
                    continue

                # Extended pairwise for transformer (with same normalization as training)
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
                pair_pw_transformer.append(extended_pw)

        if pair_indices:
            # Get XGBoost probabilities
            xgb_probs = self._get_xgboost_probs(tracklets, tracklet_ids, pair_indices)

            # Get Transformer probabilities
            tf_probs = self._get_transformer_probs(token_cache, pair_indices, pair_pw_transformer)

            # Blend in logit space
            w = self.blend_weight
            xgb_logits = _safe_logit(np.array(xgb_probs))
            tf_logits = _safe_logit(np.array(tf_probs))
            blended_logits = (1 - w) * xgb_logits + w * tf_logits
            blended_probs = sp_expit(blended_logits)

            for (i, j), p in zip(pair_indices, blended_probs):
                d = float(1.0 - p)
                dist_matrix[i, j] = dist_matrix[j, i] = d

        return dist_matrix

    # ------------------------------------------------------------------
    # XGBoost scoring
    # ------------------------------------------------------------------

    def _get_xgboost_probs(self, tracklets, tracklet_ids, pair_indices):
        """Get XGBoost merge probabilities for the given pairs."""
        import pandas as pd
        pair_rows = []
        for i, j in pair_indices:
            t_a = tracklets[tracklet_ids[i]]
            t_b = tracklets[tracklet_ids[j]]
            agg_a = self.xgboost_merger._aggregate(t_a)
            agg_b = self.xgboost_merger._aggregate(t_b)
            row = self.xgboost_merger._build_feature_row(agg_a, agg_b)
            pair_rows.append(row)

        X = pd.DataFrame(pair_rows, columns=self.xgboost_merger.feature_cols).fillna(0).values
        probs_per_model = np.stack(
            [m.predict_proba(X)[:, 1] for m in self.xgboost_merger.models], axis=0
        )
        probs = probs_per_model.mean(axis=0)
        return probs.tolist()

    # ------------------------------------------------------------------
    # Transformer scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_tracklet_tokens(self, tracklet) -> torch.Tensor:
        feats, frames = extract_frame_features(tracklet)
        bins = self._dummy_dataset._create_bin_representation(feats, frames)
        bins = bins.unsqueeze(0).to(self.device)
        tokens = self.transformer.encode_tracklet(bins)
        return tokens.squeeze(0)

    @torch.no_grad()
    def _get_transformer_probs(self, token_cache, pair_indices, pair_pairwise):
        batch_size = 128
        all_probs = []

        for start in range(0, len(pair_indices), batch_size):
            end = min(start + batch_size, len(pair_indices))
            batch_idx = pair_indices[start:end]
            batch_pw = pair_pairwise[start:end]

            tokens_a = torch.stack([token_cache[i] for i, j in batch_idx]).to(self.device)
            tokens_b = torch.stack([token_cache[j] for i, j in batch_idx]).to(self.device)
            pw = torch.tensor(np.stack(batch_pw), dtype=torch.float32).to(self.device)

            logits = self.transformer.forward_from_cached(tokens_a, tokens_b, pw)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())

        return all_probs

    # ------------------------------------------------------------------
    # C3: Normalization (shared with NewTransformerMerger)
    # ------------------------------------------------------------------

    def _normalize_distances(self, dist_matrix: np.ndarray) -> np.ndarray:
        if self.normalization_method == "none":
            return dist_matrix

        n = dist_matrix.shape[0]
        if n < 3:
            return dist_matrix

        mask = np.zeros_like(dist_matrix, dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                if dist_matrix[i, j] < 1.5:
                    mask[i, j] = True

        valid_dists = dist_matrix[mask]
        if len(valid_dists) < 2:
            return dist_matrix

        result = dist_matrix.copy()

        if self.normalization_method == "rank":
            ranks = rankdata(valid_dists) / len(valid_dists)
            result[mask] = ranks
            for i in range(n):
                for j in range(i + 1, n):
                    result[j, i] = result[i, j]

        elif self.normalization_method == "zscore":
            mean = valid_dists.mean()
            std = valid_dists.std() + 1e-8
            z_scores = (valid_dists - mean) / std
            normalized = 1.0 / (1.0 + np.exp(-z_scores))
            result[mask] = normalized
            for i in range(n):
                for j in range(i + 1, n):
                    result[j, i] = result[i, j]

        return result

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
