"""
Dataset for the Late Cross-Attention Transformer.

Reuses the temporal bin representation from the existing transformer dataset
but adds support for:
  - Hard-negative weighting (pairs carry a `weight` field)
  - Metadata for per-sequence grouping (needed for contrastive loss and
    clustering proxy evaluation)

The dataset produces the same bin representation as TemporalBinDataset:
  Tokens 0..n_bins-1:  temporal bin averages
  Token n_bins:        start boundary
  Token n_bins+1:      end boundary
  Token n_bins+2:      aggregate statistics

Online augmentations are identical to the temporal bin transformer.
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Reuse feature extraction from the original transformer code
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

from transformer.dataset import (
    extract_frame_features, _tracklet_stats,
    RAW_DIM, REID_START, REID_END, SIGLIP_START, SIGLIP_END,
    BBOX_START, BBOX_END, SCALAR_START, SCALAR_END, SCORE_START, SCORE_END,
    TRACKLET_STATS_DIM,
    PW_ENDPOINT_DX_IDX, PW_ENDPOINT_DY_IDX, PW_BBOX_HEIGHT_RATIO_IDX,
)

EXTENDED_PAIRWISE_DIM = 34

# ---------------------------------------------------------------------------
# Pairwise feature normalization constants.
#
# The 34-dim extended pairwise vector has wildly different scales:
#   idx 0   temporal_gap        (0 - 10 000+ frames)
#   idx 1   spatial_distance    (0 - 2 000 pixels)
#   idx 2-3 endpoint_dx/dy      (-2 000 - 2 000)
#   idx 4   bbox_height_ratio   (0.1 - 10+)
#   idx 5-6 cosine similarities (already -1 to 1, no scaling needed)
#   idx 7-11 binary/flag features (0 or 1, no scaling needed)
#   idx 12+ tracklet stats (duration 0-10k, n_frames 0-1k, coords 0-2k,
#                            bbox_height 0-600, coverages 0-1)
#
# Without normalization, the baseline linear head and the correction MLP
# are dominated by the large-magnitude features (temporal_gap, area, ...).
# Under AMP float16 (max ~65 504) intermediate products can overflow -> inf
# -> NaN in attention -> inf loss.
#
# We apply per-feature scaling so every dimension is roughly O(1).
# ---------------------------------------------------------------------------
_PW_SCALE = np.ones(EXTENDED_PAIRWISE_DIM, dtype=np.float32)
# Base 12: [temp_gap, spat_dist, dx, dy, h_ratio, reid_cos, sig_cos,
#           j_match, j_conflict, j_conf, tm_match, tm_consist]
_PW_SCALE[0]  = 500.0    # temporal_gap
_PW_SCALE[1]  = 500.0    # spatial_distance
_PW_SCALE[2]  = 500.0    # endpoint_dx
_PW_SCALE[3]  = 500.0    # endpoint_dy
_PW_SCALE[4]  = 2.0      # bbox_height_ratio
# indices 5-11: cosines and binary flags -- already O(1)
# Stats block A (indices 12-22) and Stats block B (indices 23-33):
for offset in [12, 23]:
    _PW_SCALE[offset + 0] = 500.0   # duration
    _PW_SCALE[offset + 1] = 100.0   # n_frames
    _PW_SCALE[offset + 2] = 1000.0  # start_cx  (pixel coord)
    _PW_SCALE[offset + 3] = 1000.0  # start_cy
    _PW_SCALE[offset + 4] = 1000.0  # end_cx
    _PW_SCALE[offset + 5] = 1000.0  # end_cy
    _PW_SCALE[offset + 6] = 200.0   # mean_bbox_height
    # indices 7-10: coverages/consistencies -- already 0-1


class NewTransformerDataset(Dataset):
    """
    Dataset for the Late Cross-Attention Transformer.

    Each sample returns:
        bins_a:    (n_bins + 3, raw_dim + 2) -- temporal bins + boundary + stats
        bins_b:    (n_bins + 3, raw_dim + 2) -- same for tracklet B
        pairwise:  (34,) extended pairwise features
        label:     scalar (0 or 1)
        weight:    scalar sample weight (higher for hard negatives)
        seq_id:    integer sequence identifier (for per-sequence grouping)
    """

    def __init__(
        self,
        pair_metadata: List[dict],
        cache_root: Path,
        n_bins: int = 8,
        boundary_k: int = 5,
        max_frame_value: float = 750.0,
        training: bool = True,
        frame_dropout: float = 0.15,
        augment_swap: bool = True,
        noise_std_reid: float = 0.01,
        noise_std_siglip: float = 0.02,
        modality_mask_prob: float = 0.10,
        temporal_crop_prob: float = 0.15,
    ):
        self.n_bins = n_bins
        self.boundary_k = boundary_k
        self.max_frame_value = max_frame_value
        self.training = training
        self.frame_dropout = frame_dropout
        self.augment_swap = augment_swap
        self.noise_std_reid = noise_std_reid
        self.noise_std_siglip = noise_std_siglip
        self.modality_mask_prob = modality_mask_prob
        self.temporal_crop_prob = temporal_crop_prob

        # Build sequence ID mapping
        self._seq_to_id: Dict[str, int] = {}

        # Pre-extract features for all referenced tracklets
        self._feature_cache: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {}
        self._load_features(pair_metadata, cache_root)

        # Store pair metadata
        self.pairs = []
        for p in pair_metadata:
            seq = p.get("sequence", "unknown")
            if seq not in self._seq_to_id:
                self._seq_to_id[seq] = len(self._seq_to_id)

            if p.get("is_synthetic", False):
                self.pairs.append({
                    "feats_a": p["feats_a"],
                    "frames_a": p["frames_a"],
                    "feats_b": p["feats_b"],
                    "frames_b": p["frames_b"],
                    "pairwise": p["pairwise"],
                    "label": p["label"],
                    "weight": p.get("weight", 1.0),
                    "seq_id": self._seq_to_id[seq],
                    "is_synthetic": True,
                    "gt_id_a": p.get("gt_id_a", -1),
                    "gt_id_b": p.get("gt_id_b", -1),
                })
            else:
                key_a = (p["sequence"], p["tid_a"])
                key_b = (p["sequence"], p["tid_b"])
                if key_a in self._feature_cache and key_b in self._feature_cache:
                    self.pairs.append({
                        "key_a": key_a,
                        "key_b": key_b,
                        "pairwise": p["pairwise"],
                        "label": p["label"],
                        "weight": p.get("weight", 1.0),
                        "seq_id": self._seq_to_id[seq],
                        "is_synthetic": False,
                        "gt_id_a": p.get("gt_id_a", -1),
                        "gt_id_b": p.get("gt_id_b", -1),
                    })

    def _load_features(self, pair_metadata, cache_root):
        """Load pickle caches and extract features for all real tracklets."""
        needed = set()
        for p in pair_metadata:
            if not p.get("is_synthetic", False):
                needed.add((p["sequence"], p["tid_a"]))
                needed.add((p["sequence"], p["tid_b"]))

        by_seq: Dict[str, set] = {}
        for seq, tid in needed:
            by_seq.setdefault(seq, set()).add(tid)

        for seq, tids in by_seq.items():
            cache_path = Path(cache_root) / f"cache_split_{seq}.pkl"
            if not cache_path.exists():
                cache_path = Path(cache_root) / f"cache_attributes_{seq}.pkl"
            if not cache_path.exists():
                print(f"  [WARN] Cache not found for {seq}")
                continue
            with open(cache_path, "rb") as f:
                tracklets = pickle.load(f)
            for tid in tids:
                if tid in tracklets:
                    feats, frames = extract_frame_features(tracklets[tid])
                    self._feature_cache[(seq, tid)] = (feats, frames)

        print(f"  Loaded features for {len(self._feature_cache)} tracklets")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]

        if p["is_synthetic"]:
            feats_a, frames_a = p["feats_a"].copy(), p["frames_a"].copy()
            feats_b, frames_b = p["feats_b"].copy(), p["frames_b"].copy()
        else:
            feats_a, frames_a = [x.copy() for x in self._feature_cache[p["key_a"]]]
            feats_b, frames_b = [x.copy() for x in self._feature_cache[p["key_b"]]]

        # Apply augmentations before binning
        if self.training:
            feats_a, frames_a = self._augment_sequence(feats_a, frames_a)
            feats_b, frames_b = self._augment_sequence(feats_b, frames_b)

        # Create temporal bin representation
        bins_a = self._create_bin_representation(feats_a, frames_a)
        bins_b = self._create_bin_representation(feats_b, frames_b)

        pw_raw = p["pairwise"].copy()
        # Normalize pairwise features to O(1) scale and clamp NaN/inf
        pw_raw = np.where(np.isfinite(pw_raw), pw_raw, 0.0).astype(np.float32)
        pw_raw = pw_raw / _PW_SCALE
        pw_raw = np.clip(pw_raw, -50.0, 50.0)
        pairwise = torch.from_numpy(pw_raw)

        label = torch.tensor(p["label"], dtype=torch.float32)
        weight = torch.tensor(p["weight"], dtype=torch.float32)
        seq_id = torch.tensor(p["seq_id"], dtype=torch.long)

        # A/B swap augmentation
        if self.training and self.augment_swap and torch.rand(1).item() > 0.5:
            bins_a, bins_b = bins_b, bins_a
            pairwise[PW_ENDPOINT_DX_IDX] = -pairwise[PW_ENDPOINT_DX_IDX]
            pairwise[PW_ENDPOINT_DY_IDX] = -pairwise[PW_ENDPOINT_DY_IDX]
            # Invert height ratio correctly: normalized value is raw/scale.
            # Swapped raw = 1/raw, so swapped normalized = (1/raw)/scale.
            ratio_norm = pairwise[PW_BBOX_HEIGHT_RATIO_IDX].item()
            scale = _PW_SCALE[PW_BBOX_HEIGHT_RATIO_IDX]
            raw_ratio = ratio_norm * scale
            if raw_ratio != 0:
                pairwise[PW_BBOX_HEIGHT_RATIO_IDX] = (1.0 / raw_ratio) / scale
            # Swap A/B stats blocks
            pairwise[12:23], pairwise[23:34] = pairwise[23:34].clone(), pairwise[12:23].clone()

        return bins_a, bins_b, pairwise, label, weight, seq_id

    def _augment_sequence(
        self, feats: np.ndarray, frames: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply online augmentations to a raw frame sequence before binning."""
        n = len(frames)

        # 1. Temporal cropping
        if self.temporal_crop_prob > 0 and np.random.rand() < self.temporal_crop_prob and n > 10:
            crop_frac = np.random.uniform(0.10, 0.30)
            crop_n = max(1, int(n * crop_frac))
            if np.random.rand() > 0.5:
                feats = feats[crop_n:]
                frames = frames[crop_n:]
            else:
                feats = feats[:-crop_n]
                frames = frames[:-crop_n]
            n = len(frames)

        # 2. Frame dropout (preserve first and last)
        if self.frame_dropout > 0 and n > 4:
            keep = np.random.rand(n) > self.frame_dropout
            keep[0] = True
            keep[-1] = True
            if keep.sum() >= 3:
                feats = feats[keep]
                frames = frames[keep]
                n = int(keep.sum())

        # 3. Gaussian noise on embeddings
        if self.noise_std_reid > 0:
            feats[:, REID_START:REID_END] += np.random.randn(
                n, REID_END - REID_START
            ).astype(np.float32) * self.noise_std_reid

        if self.noise_std_siglip > 0:
            feats[:, SIGLIP_START:SIGLIP_END] += np.random.randn(
                n, SIGLIP_END - SIGLIP_START
            ).astype(np.float32) * self.noise_std_siglip

        # 4. Modality masking
        if self.modality_mask_prob > 0 and np.random.rand() < self.modality_mask_prob:
            group = np.random.choice(5)
            slices = [
                (REID_START, REID_END),
                (SIGLIP_START, SIGLIP_END),
                (BBOX_START, BBOX_END),
                (SCALAR_START, SCALAR_END),
                (SCORE_START, SCORE_END),
            ]
            s, e = slices[group]
            feats[:, s:e] = 0.0

        return feats, frames

    def _create_bin_representation(
        self, feats: np.ndarray, frames: np.ndarray
    ) -> torch.Tensor:
        """
        Create the compact temporal bin representation for one tracklet.

        Returns: (n_bins + 3, raw_dim + 2) tensor
        """
        # L2-normalize ReID and SigLIP slices per frame before binning.
        # Raw embeddings can reach magnitudes of 10k-22k, which overflow
        # float16 under AMP. The projectors learn scale anyway, so unit
        # vectors are strictly equivalent in expressiveness.
        feats = feats.copy()
        # Replace any NaN/inf in raw features before processing
        feats = np.where(np.isfinite(feats), feats, 0.0).astype(np.float32)
        for start, end in [(REID_START, REID_END), (SIGLIP_START, SIGLIP_END)]:
            norms = np.linalg.norm(feats[:, start:end], axis=1, keepdims=True)
            feats[:, start:end] /= np.maximum(norms, 1e-8)

        n = len(frames)
        token_dim = RAW_DIM + 2
        n_tokens = self.n_bins + 3

        result = np.zeros((n_tokens, token_dim), dtype=np.float32)

        # --- Temporal bins ---
        bin_edges = np.linspace(0, n, self.n_bins + 1, dtype=int)
        for b in range(self.n_bins):
            start_idx = bin_edges[b]
            end_idx = max(bin_edges[b + 1], start_idx + 1)
            end_idx = min(end_idx, n)

            if start_idx < n:
                bin_feats = feats[start_idx:end_idx]
                bin_frames = frames[start_idx:end_idx]

                result[b, :RAW_DIM] = bin_feats.mean(axis=0)

                center_pos = (start_idx + end_idx - 1) / 2.0 / max(n - 1, 1)
                result[b, RAW_DIM] = center_pos

                center_frame = bin_frames.mean() / self.max_frame_value
                result[b, RAW_DIM + 1] = center_frame

        # --- Boundary tokens ---
        k = min(self.boundary_k, n)

        start_feats = feats[:k].mean(axis=0)
        result[self.n_bins, :RAW_DIM] = start_feats
        result[self.n_bins, RAW_DIM] = 0.0
        result[self.n_bins, RAW_DIM + 1] = frames[:k].mean() / self.max_frame_value

        end_feats = feats[-k:].mean(axis=0)
        result[self.n_bins + 1, :RAW_DIM] = end_feats
        result[self.n_bins + 1, RAW_DIM] = 1.0
        result[self.n_bins + 1, RAW_DIM + 1] = frames[-k:].mean() / self.max_frame_value

        # --- Aggregate statistics token ---
        stats = _tracklet_stats(feats, frames)
        result[self.n_bins + 2, :TRACKLET_STATS_DIM] = stats
        result[self.n_bins + 2, RAW_DIM] = 0.5
        result[self.n_bins + 2, RAW_DIM + 1] = frames.mean() / self.max_frame_value

        return torch.from_numpy(result)
