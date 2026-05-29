"""
Dataset for the Frame-level Cross-Attention Pair Transformer.

Key difference from TemporalBinDataset: instead of compressing tracklets
into temporal bins, this dataset returns per-frame features directly.

Frame matching strategy (user-specified):
  For each pair (A, B), K = min(len(A), len(B), max_frames).
  During training: randomly sample K frames from each tracklet.
  During inference: uniformly sample K frames (deterministic).

This ensures balanced information from both tracklets. The random sampling
during training acts as strong augmentation and naturally handles occluded
or noisy frames.

A custom collate_fn pads variable-length sequences within each batch and
produces attention masks for the transformer.
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

from transformer.dataset import (
    extract_frame_features,
    RAW_DIM, REID_START, REID_END, SIGLIP_START, SIGLIP_END,
    PW_ENDPOINT_DX_IDX, PW_ENDPOINT_DY_IDX, PW_BBOX_HEIGHT_RATIO_IDX,
)


class FramePairDataset(Dataset):
    """
    Dataset that returns per-frame features for tracklet pairs.

    Each sample returns a dict with:
        feats_a:   (K_a, 1293)  per-frame features for tracklet A
        frames_a:  (K_a,)       frame numbers for tracklet A
        feats_b:   (K_b, 1293)  per-frame features for tracklet B
        frames_b:  (K_b,)       frame numbers for tracklet B
        pairwise:  (34,)        extended pairwise features
        label:     scalar       0 or 1
    """

    def __init__(
        self,
        pair_metadata: List[dict],
        cache_root: Path,
        max_frames: int = 200,
        min_frames: int = 2,
        training: bool = True,
        frame_dropout: float = 0.2,
        augment_swap: bool = True,
        noise_std_reid: float = 0.01,
        noise_std_siglip: float = 0.02,
    ):
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.training = training
        self.frame_dropout = frame_dropout
        self.augment_swap = augment_swap
        self.noise_std_reid = noise_std_reid
        self.noise_std_siglip = noise_std_siglip

        # Pre-extract features for all referenced tracklets
        self._feature_cache: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {}
        self._load_features(pair_metadata, cache_root)

        # Store pair metadata (only pairs where both tracklets were found)
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

        print(f"  FramePairDataset: {len(self.pairs)} pairs "
              f"({'training' if training else 'validation'})")

    def _load_features(self, pair_metadata, cache_root):
        """Load pickle caches and extract features for all referenced tracklets."""
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
            feats_a = p["feats_a"].copy()
            frames_a = p["frames_a"].copy()
            feats_b = p["feats_b"].copy()
            frames_b = p["frames_b"].copy()
        else:
            feats_a, frames_a = [x.copy() for x in self._feature_cache[p["key_a"]]]
            feats_b, frames_b = [x.copy() for x in self._feature_cache[p["key_b"]]]

        pairwise = p["pairwise"].copy().astype(np.float32)
        label = float(p["label"])

        # ── Adaptive frame matching ──
        # K = min(len_a, len_b, max_frames): match to shorter tracklet
        n_a, n_b = len(feats_a), len(feats_b)
        K = min(n_a, n_b, self.max_frames)
        K = max(K, self.min_frames)  # at least min_frames

        if self.training:
            # Random sampling from each tracklet
            idx_a = np.sort(np.random.choice(n_a, min(K, n_a), replace=K > n_a))
            idx_b = np.sort(np.random.choice(n_b, min(K, n_b), replace=K > n_b))
        else:
            # Deterministic uniform sampling
            idx_a = np.round(np.linspace(0, n_a - 1, min(K, n_a))).astype(int)
            idx_b = np.round(np.linspace(0, n_b - 1, min(K, n_b))).astype(int)

        feats_a, frames_a = feats_a[idx_a], frames_a[idx_a]
        feats_b, frames_b = feats_b[idx_b], frames_b[idx_b]

        # ── Frame dropout (training only) ──
        if self.training and self.frame_dropout > 0:
            feats_a, frames_a = self._apply_frame_dropout(feats_a, frames_a)
            feats_b, frames_b = self._apply_frame_dropout(feats_b, frames_b)

        # ── Noise augmentation (training only) ──
        if self.training:
            feats_a = self._add_noise(feats_a)
            feats_b = self._add_noise(feats_b)

        # ── Swap augmentation ──
        if self.training and self.augment_swap and np.random.rand() > 0.5:
            feats_a, feats_b = feats_b, feats_a
            frames_a, frames_b = frames_b, frames_a
            # Flip sign of directional pairwise features
            pairwise[PW_ENDPOINT_DX_IDX] *= -1
            pairwise[PW_ENDPOINT_DY_IDX] *= -1
            if pairwise[PW_BBOX_HEIGHT_RATIO_IDX] > 0:
                pairwise[PW_BBOX_HEIGHT_RATIO_IDX] = 1.0 / (
                    pairwise[PW_BBOX_HEIGHT_RATIO_IDX] + 1e-6
                )

        return {
            "feats_a": torch.tensor(feats_a, dtype=torch.float32),
            "frames_a": torch.tensor(frames_a, dtype=torch.float32),
            "feats_b": torch.tensor(feats_b, dtype=torch.float32),
            "frames_b": torch.tensor(frames_b, dtype=torch.float32),
            "pairwise": torch.tensor(pairwise, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
        }

    def _apply_frame_dropout(self, feats, frames):
        """Randomly drop frames, keeping at least min_frames."""
        n = len(feats)
        if n <= self.min_frames:
            return feats, frames
        keep = np.random.rand(n) > self.frame_dropout
        # Always keep first and last frame
        keep[0] = True
        keep[-1] = True
        if keep.sum() < self.min_frames:
            # Force keep enough random frames
            false_idx = np.where(~keep)[0]
            need = self.min_frames - int(keep.sum())
            force_keep = np.random.choice(false_idx, need, replace=False)
            keep[force_keep] = True
        return feats[keep], frames[keep]

    def _add_noise(self, feats):
        """Add Gaussian noise to ReID and SigLIP embeddings."""
        if self.noise_std_reid > 0:
            feats[:, REID_START:REID_END] += np.random.randn(
                len(feats), REID_END - REID_START
            ).astype(np.float32) * self.noise_std_reid
        if self.noise_std_siglip > 0:
            feats[:, SIGLIP_START:SIGLIP_END] += np.random.randn(
                len(feats), SIGLIP_END - SIGLIP_START
            ).astype(np.float32) * self.noise_std_siglip
        return feats


# ---------------------------------------------------------------------------
# Custom collate for variable-length frame sequences
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """
    Pad variable-length frame sequences to the max length in the batch.

    Returns a tuple of tensors ready for the model:
        feats_a, frames_a, mask_a, feats_b, frames_b, mask_b, pairwise, labels
    """
    B = len(batch)
    raw_dim = batch[0]["feats_a"].size(1)

    # Find max lengths in this batch
    max_a = max(b["feats_a"].size(0) for b in batch)
    max_b = max(b["feats_b"].size(0) for b in batch)

    # Allocate padded tensors
    feats_a = torch.zeros(B, max_a, raw_dim)
    frames_a = torch.zeros(B, max_a)
    mask_a = torch.ones(B, max_a, dtype=torch.bool)    # True = padding (ignore)

    feats_b = torch.zeros(B, max_b, raw_dim)
    frames_b = torch.zeros(B, max_b)
    mask_b = torch.ones(B, max_b, dtype=torch.bool)

    pairwise = torch.stack([b["pairwise"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])

    for i, b in enumerate(batch):
        na = b["feats_a"].size(0)
        nb = b["feats_b"].size(0)

        feats_a[i, :na] = b["feats_a"]
        frames_a[i, :na] = b["frames_a"]
        mask_a[i, :na] = False

        feats_b[i, :nb] = b["feats_b"]
        frames_b[i, :nb] = b["frames_b"]
        mask_b[i, :nb] = False

    return feats_a, frames_a, mask_a, feats_b, frames_b, mask_b, pairwise, labels
