"""
Temporal Bin Transformer merger — drop-in replacement for XGBoostMerger.

Usage in main.py:
    from experiments.tracklet_merger.transformer.merger import TemporalBinMerger

    merger = TemporalBinMerger(
        model_path  = "weights/temporal_bin_transformer/best_model.pt",
        meta_path   = "weights/temporal_bin_transformer/meta.json",
        merge_threshold = 0.5,
    )
    merged_tracklets = merger.merge(splitted_tracklets)

Two-stage inference for efficiency:
  1. Encode each tracklet ONCE → cache CLS embeddings (12 tokens → 1 CLS)
  2. For each pair: run only the lightweight classifier head on cached embeddings

This is efficient because the temporal binning + encoding is O(T) per tracklet,
and the classifier head is a small MLP operating on fixed-size inputs.
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

from experiments.tracklet_merger.transformer.model import TemporalBinTransformer
from experiments.tracklet_merger.transformer.dataset import TemporalBinDataset
from transformer.dataset import (
    extract_frame_features, compute_pairwise_features,
    _tracklet_stats, RAW_DIM, TRACKLET_STATS_DIM,
)


class TemporalBinMerger:
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
        self.calibration_temperature = calibration_temperature
        self.calibration_platt = calibration_platt  # (a, b) tuple

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model
        checkpoint = torch.load(str(model_path), map_location=self.device, weights_only=False)
        self.config = checkpoint["config"]
        self.model = TemporalBinTransformer(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Load metadata
        with open(meta_path) as f:
            meta = json.load(f)
        self.n_bins = meta.get("n_temporal_bins", self.config.n_temporal_bins)
        self.boundary_k = meta.get("boundary_k", self.config.boundary_k)
        self.max_frame_value = meta.get("max_frame_value", self.config.max_frame_value)

        # A dummy dataset instance for the bin representation method
        # (we only use _create_bin_representation, not __getitem__)
        self._dummy_dataset = TemporalBinDataset.__new__(TemporalBinDataset)
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

        # Stage 1: encode all tracklets once
        cls_embeddings = {}
        for idx, tid in enumerate(tracklet_ids):
            cls_embeddings[idx] = self._encode_tracklet(tracklets[tid])

        # Stage 2: score all valid pairs
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

                # # Hard constraint: consistent team mismatch
                if self.use_team_constraint and self._team_conflict(t_a, t_b):
                    dist_matrix[i, j] = dist_matrix[j, i] = 2.0
                    continue

                # Extended pairwise features
                pw_base = compute_pairwise_features(t_a, t_b)
                feats_a, frames_a = extract_frame_features(t_a)
                feats_b, frames_b = extract_frame_features(t_b)
                stats_a = _tracklet_stats(feats_a, frames_a)
                stats_b = _tracklet_stats(feats_b, frames_b)
                extended_pw = np.concatenate([pw_base, stats_a, stats_b])

                pair_indices.append((i, j))
                pair_pairwise.append(extended_pw)

        # Batch classify
        if pair_indices:
            probs = self._batch_classify(cls_embeddings, pair_indices, pair_pairwise)
            for (i, j), p in zip(pair_indices, probs):
                d = float(1.0 - p)
                dist_matrix[i, j] = dist_matrix[j, i] = d

        return dist_matrix

    # ------------------------------------------------------------------
    # Encoding & classification
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_tracklet(self, tracklet) -> torch.Tensor:
        """Encode a single tracklet -> CLS embedding (d_model,)."""
        feats, frames = extract_frame_features(tracklet)
        bins = self._dummy_dataset._create_bin_representation(feats, frames)
        bins = bins.unsqueeze(0).to(self.device)  # (1, n_tokens, dim)
        cls = self.model.encode_tracklet(bins)
        return cls.squeeze(0)  # (d_model,)

    @torch.no_grad()
    def _batch_classify(self, cls_embeddings, pair_indices, pair_pairwise):
        """Run classifier head on batches of pre-encoded CLS pairs."""
        batch_size = 256
        all_probs = []

        for start in range(0, len(pair_indices), batch_size):
            end = min(start + batch_size, len(pair_indices))
            batch_idx = pair_indices[start:end]
            batch_pw = pair_pairwise[start:end]

            cls_a = torch.stack([cls_embeddings[i] for i, j in batch_idx])
            cls_b = torch.stack([cls_embeddings[j] for i, j in batch_idx])
            pw = torch.tensor(np.stack(batch_pw), dtype=torch.float32).to(self.device)

            logits = self.model.classifier(cls_a, cls_b, pw)

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
    # Hard-constraint helpers (identical to XGBoostMerger / TransformerMerger)
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
