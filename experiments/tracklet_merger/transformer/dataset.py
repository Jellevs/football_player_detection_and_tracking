"""
Dataset for the Temporal Bin Transformer.

Instead of feeding raw frame sequences (up to 200 frames x 1293 dims) through
a transformer, we create a compact fixed-size representation per tracklet:

  Token 0:      [CLS] (learnable, added by the model)
  Tokens 1-8:   Temporal bins — divide the tracklet into N equal time slices
                 and average the per-frame features within each bin.  This
                 captures *how* features evolve over time (e.g., ReID embedding
                 drift, jersey visibility changes) without the memory cost of
                 processing every frame.
  Token 9:      Start boundary — average of first K frames.  The merge decision
                 depends heavily on where tracklets start/end.
  Token 10:     End boundary — average of last K frames.
  Token 11:     Aggregate statistics — a vector of 11 scalars (duration,
                 jersey coverage, team consistency, etc.) that mirror the
                 XGBoost per-tracklet features.

Total: 11 tokens returned by the dataset (CLS is prepended by the model).
Each token is raw_dim=1293 and will be projected to d_model by the model.

Online augmentations:
  - Frame dropout before binning (15% of frames, preserving first/last)
  - Gaussian noise on ReID/SigLIP embeddings
  - Modality masking (zero out one feature group 10% of the time)
  - Temporal cropping (remove 10-30% from a random end 15% of the time)
  - A/B pair swap (50% of the time)
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

# Pairwise dimension: 12 base + 11 stats_A + 11 stats_B = 34
EXTENDED_PAIRWISE_DIM = 34


class TemporalBinDataset(Dataset):
    """
    Dataset that produces compact temporal bin representations of tracklet pairs.

    Each sample returns:
        bins_a:    (n_bins + 3, raw_dim) — temporal bins + boundary + stats for tracklet A
        bins_b:    (n_bins + 3, raw_dim) — same for tracklet B
        pairwise:  (34,) extended pairwise features
        label:     scalar (0 or 1)

    The model will prepend CLS tokens and project to d_model.
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

        # Pre-extract features for all referenced tracklets
        self._feature_cache: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {}
        self._load_features(pair_metadata, cache_root)

        # Store pair metadata
        self.pairs = []
        for p in pair_metadata:
            if p.get("is_synthetic", False):
                # Synthetic pairs carry their own features
                self.pairs.append({
                    "feats_a": p["feats_a"],
                    "frames_a": p["frames_a"],
                    "feats_b": p["feats_b"],
                    "frames_b": p["frames_b"],
                    "pairwise": p["pairwise"],
                    "label": p["label"],
                    "is_synthetic": True,
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
                        "is_synthetic": False,
                    })

    def _load_features(self, pair_metadata, cache_root):
        """Load pickle caches and extract features for all real (non-synthetic) tracklets."""
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
                # Fall back to pre-split cache
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

        pairwise = torch.tensor(p["pairwise"].copy(), dtype=torch.float32)
        label = torch.tensor(p["label"], dtype=torch.float32)

        # A/B swap augmentation
        if self.training and self.augment_swap and torch.rand(1).item() > 0.5:
            bins_a, bins_b = bins_b, bins_a
            # Adjust pairwise features
            pairwise[PW_ENDPOINT_DX_IDX] = -pairwise[PW_ENDPOINT_DX_IDX]
            pairwise[PW_ENDPOINT_DY_IDX] = -pairwise[PW_ENDPOINT_DY_IDX]
            ratio = pairwise[PW_BBOX_HEIGHT_RATIO_IDX]
            if ratio != 0:
                pairwise[PW_BBOX_HEIGHT_RATIO_IDX] = 1.0 / ratio
            # Swap A/B stats blocks (indices 12:23 and 23:34)
            pairwise[12:23], pairwise[23:34] = pairwise[23:34].clone(), pairwise[12:23].clone()

        return bins_a, bins_b, pairwise, label

    def _augment_sequence(
        self, feats: np.ndarray, frames: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply online augmentations to a raw frame sequence before binning."""
        n = len(frames)

        # 1. Temporal cropping: remove 10-30% from a random end
        if self.temporal_crop_prob > 0 and np.random.rand() < self.temporal_crop_prob and n > 10:
            crop_frac = np.random.uniform(0.10, 0.30)
            crop_n = max(1, int(n * crop_frac))
            if np.random.rand() > 0.5:
                # Crop from start
                feats = feats[crop_n:]
                frames = frames[crop_n:]
            else:
                # Crop from end
                feats = feats[:-crop_n]
                frames = frames[:-crop_n]
            n = len(frames)

        # 2. Frame dropout (preserve first and last)
        if self.frame_dropout > 0 and n > 4:
            keep = np.random.rand(n) > self.frame_dropout
            keep[0] = True
            keep[-1] = True
            if keep.sum() >= 3:  # need at least 3 frames for meaningful bins
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

        # 4. Modality masking: zero out one feature group
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
            - Tokens 0..n_bins-1:  temporal bin averages
            - Token n_bins:        start boundary (avg of first K frames)
            - Token n_bins+1:      end boundary (avg of last K frames)
            - Token n_bins+2:      aggregate statistics (zero-padded to raw_dim+2)

        Each token has raw_dim features + 2 positional features:
            [0:raw_dim]     average raw features within the bin
            [raw_dim]       normalized intra-tracklet position (center of bin, 0..1)
            [raw_dim+1]     normalized absolute frame number (center of bin)
        """
        n = len(frames)
        token_dim = RAW_DIM + 2  # features + 2 positional scalars
        n_tokens = self.n_bins + 3  # bins + start_boundary + end_boundary + stats

        result = np.zeros((n_tokens, token_dim), dtype=np.float32)

        # --- Temporal bins ---
        bin_edges = np.linspace(0, n, self.n_bins + 1, dtype=int)
        for b in range(self.n_bins):
            start_idx = bin_edges[b]
            end_idx = max(bin_edges[b + 1], start_idx + 1)  # at least 1 frame
            end_idx = min(end_idx, n)

            if start_idx < n:
                bin_feats = feats[start_idx:end_idx]
                bin_frames = frames[start_idx:end_idx]

                result[b, :RAW_DIM] = bin_feats.mean(axis=0)

                # Positional: center of bin in normalized tracklet position
                center_pos = (start_idx + end_idx - 1) / 2.0 / max(n - 1, 1)
                result[b, RAW_DIM] = center_pos

                # Positional: center of bin in absolute frame number
                center_frame = bin_frames.mean() / self.max_frame_value
                result[b, RAW_DIM + 1] = center_frame

        # --- Boundary tokens ---
        k = min(self.boundary_k, n)

        # Start boundary
        start_feats = feats[:k].mean(axis=0)
        result[self.n_bins, :RAW_DIM] = start_feats
        result[self.n_bins, RAW_DIM] = 0.0  # position = start
        result[self.n_bins, RAW_DIM + 1] = frames[:k].mean() / self.max_frame_value

        # End boundary
        end_feats = feats[-k:].mean(axis=0)
        result[self.n_bins + 1, :RAW_DIM] = end_feats
        result[self.n_bins + 1, RAW_DIM] = 1.0  # position = end
        result[self.n_bins + 1, RAW_DIM + 1] = frames[-k:].mean() / self.max_frame_value

        # --- Aggregate statistics token ---
        # Pack 11 stats into the first 11 positions, zero-pad the rest
        stats = _tracklet_stats(feats, frames)
        result[self.n_bins + 2, :TRACKLET_STATS_DIM] = stats
        # Positional: midpoint
        result[self.n_bins + 2, RAW_DIM] = 0.5
        result[self.n_bins + 2, RAW_DIM + 1] = frames.mean() / self.max_frame_value

        return torch.from_numpy(result)
