"""
Dataset for the Siamese CLS Transformer.

The dataset loads pre-generated pair metadata and extracts per frame
features from the original pickle caches on initialization.
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..utils import (
    extract_frame_features,
    RAW_DIM,
    PW_ENDPOINT_DX_IDX,
    PW_ENDPOINT_DY_IDX,
    PW_BBOX_HEIGHT_RATIO_IDX,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SequencePairDataset(Dataset):
    """
    PyTorch Dataset for tracklet pair classification.

    Loads pair metadata and pre extracts per frame features from pickle
    caches into memory at initialization. __getitem__ performs temporal
    subsampling, frame dropout, padding, and optional A/B swap augmentation.
    """

    def __init__(
        self,
        pair_metadata: List[dict],
        cache_root: Path,
        t_max: int = 200,
        max_frame_value: float = 750.0,
        training: bool = True,
        frame_dropout: float = 0.15,
        augment_swap: bool = True,
    ):
        """
        Args:
            pair_metadata: list of dicts with keys:
                sequence, tid_a, tid_b, pairwise (ndarray), label (int)
            cache_root: path to directory containing cache_split_*.pkl
            t_max: max sequence length (subsample longer tracklets)
            max_frame_value: normalization constant for absolute frame numbers
            training: enable augmentation
            frame_dropout: fraction of frames to drop during training
            augment_swap: enable A/B swap augmentation
        """
        self.t_max = t_max
        self.max_frame_value = max_frame_value
        self.training = training
        self.frame_dropout = frame_dropout
        self.augment_swap = augment_swap

        # Pre extract features for all referenced tracklets
        self._feature_cache: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._load_features(pair_metadata, cache_root)

        # Store pair metadata (only keep what we need)
        self.pairs = []
        for p in pair_metadata:
            # Synthetic pairs carry their own pre extracted features
            if p.get("is_synthetic", False) and "feats_a" in p:
                key_a = (p["sequence"], p["tid_a"])
                key_b = (p["sequence"], p["tid_b"])
                self._feature_cache[key_a] = (p["feats_a"], p["frames_a"])
                self._feature_cache[key_b] = (p["feats_b"], p["frames_b"])

            key_a = (p["sequence"], p["tid_a"])
            key_b = (p["sequence"], p["tid_b"])
            if key_a in self._feature_cache and key_b in self._feature_cache:
                self.pairs.append({
                    "key_a": key_a,
                    "key_b": key_b,
                    "pairwise": p["pairwise"],
                    "label": p["label"],
                    "weight": p.get("weight", 1.0),
                })

    def _load_features(self, pair_metadata, cache_root):
        """Load pickle caches and extract features for all referenced tracklets.

        Tries both post split cache (cache_split_{seq}.pkl) and original cache
        (cache_attributes_{seq}.pkl) filenames for compatibility.
        """
        # Collect needed (sequence, tid) keys (skip synthetic pairs)
        needed = set()
        for p in pair_metadata:
            if p.get("is_synthetic", False) and "feats_a" in p:
                continue  # synthetic pairs carry their own features
            needed.add((p["sequence"], p["tid_a"]))
            needed.add((p["sequence"], p["tid_b"]))

        # Group by sequence to load each cache once
        by_seq: Dict[str, set] = {}
        for seq, tid in needed:
            by_seq.setdefault(seq, set()).add(tid)

        for seq, tids in by_seq.items():
            # Try split cache first, then original cache
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
        feats_a, frames_a = self._feature_cache[p["key_a"]]
        feats_b, frames_b = self._feature_cache[p["key_b"]]

        tok_a, mask_a, pos_a, fn_a = self._prepare_sequence(feats_a, frames_a)
        tok_b, mask_b, pos_b, fn_b = self._prepare_sequence(feats_b, frames_b)

        pairwise = torch.tensor(p["pairwise"].copy(), dtype=torch.float32)
        label = torch.tensor(p["label"], dtype=torch.float32)
        weight = torch.tensor(p["weight"], dtype=torch.float32)

        # A/B swap augmentation
        if self.training and self.augment_swap and torch.rand(1).item() > 0.5:
            tok_a, tok_b = tok_b, tok_a
            mask_a, mask_b = mask_b, mask_a
            pos_a, pos_b = pos_b, pos_a
            fn_a, fn_b = fn_b, fn_a
            pairwise[PW_ENDPOINT_DX_IDX] = -pairwise[PW_ENDPOINT_DX_IDX]
            pairwise[PW_ENDPOINT_DY_IDX] = -pairwise[PW_ENDPOINT_DY_IDX]
            ratio = pairwise[PW_BBOX_HEIGHT_RATIO_IDX]
            if ratio != 0:
                pairwise[PW_BBOX_HEIGHT_RATIO_IDX] = 1.0 / ratio

        return tok_a, mask_a, pos_a, fn_a, tok_b, mask_b, pos_b, fn_b, pairwise, label, weight

    def _prepare_sequence(
        self, features: np.ndarray, frames: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample, apply frame dropout, pad, and create position tensors."""
        n = len(frames)

        # Temporal subsampling if too long
        if n > self.t_max:
            indices = np.linspace(0, n - 1, self.t_max, dtype=int)
            features = features[indices]
            frames = frames[indices]
            n = self.t_max

        # Frame dropout during training (always keep first and last)
        if self.training and self.frame_dropout > 0 and n > 2:
            keep = np.random.rand(n) > self.frame_dropout
            keep[0] = True
            keep[-1] = True
            if keep.sum() >= 2:
                features = features[keep]
                frames = frames[keep]
                n = int(keep.sum())

        # Pad to t_max
        padded = np.zeros((self.t_max, RAW_DIM), dtype=np.float32)
        padded[:n] = features

        # Mask: True = padding (ignored by attention)
        mask = np.ones(self.t_max, dtype=bool)
        mask[:n] = False

        # Intra tracklet positions (normalized 0..1)
        positions = np.zeros(self.t_max, dtype=np.float32)
        if n > 1:
            positions[:n] = np.arange(n, dtype=np.float32) / (n - 1)

        # Absolute frame numbers (normalized)
        frame_nums = np.zeros(self.t_max, dtype=np.float32)
        frame_nums[:n] = frames / self.max_frame_value

        return (
            torch.from_numpy(padded),
            torch.from_numpy(mask),
            torch.from_numpy(positions),
            torch.from_numpy(frame_nums),
        )
