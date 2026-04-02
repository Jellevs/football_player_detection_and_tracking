"""
Dataset for the per-frame split-point transformer.

Each sample is a single tracklet (real or synthetically concatenated) with a
per-frame binary label vector indicating split points.

Features are extracted on-the-fly from the same pickle caches used by the
merger pipeline via transformer.dataset.extract_frame_features().
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple

from transformer.dataset import extract_frame_features, RAW_DIM


class SplitDataset(Dataset):
    """
    PyTorch Dataset for per-frame split-point detection.

    Sample metadata format (from generate_split_data.py):
        {
            "source_type": "synthetic" | "natural" | "negative",
            "sequence":    str,
            "tid_a":       int,
            "tid_b":       int | None,     # None for natural / negative
            "crop_a":      (start, end),   # frame-index crop for tracklet A
            "crop_b":      (start, end) | None,
            "gap_frames":  int,            # temporal gap inserted at junction
            "split_indices": [int, ...],   # frame indices within the sample
        }
    """

    def __init__(
        self,
        sample_metadata: List[dict],
        cache_root: Path,
        t_max: int = 400,
        max_frame_value: float = 750.0,
        split_label_radius: int = 3,
        training: bool = True,
        frame_dropout: float = 0.10,
    ):
        self.t_max = t_max
        self.max_frame_value = max_frame_value
        self.split_label_radius = split_label_radius
        self.training = training
        self.frame_dropout = frame_dropout

        # Pre-extract features for all referenced tracklets
        self._feature_cache: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._load_features(sample_metadata, cache_root)

        # Keep only samples whose tracklets we successfully loaded
        self.samples = []
        for s in sample_metadata:
            key_a = (s["sequence"], s["tid_a"])
            if key_a not in self._feature_cache:
                continue
            if s["tid_b"] is not None:
                key_b = (s["sequence"], s["tid_b"])
                if key_b not in self._feature_cache:
                    continue
            self.samples.append(s)

    def _load_features(self, sample_metadata, cache_root):
        """Load pickle caches and extract features for all referenced tracklets."""
        needed = set()
        for s in sample_metadata:
            needed.add((s["sequence"], s["tid_a"]))
            if s["tid_b"] is not None:
                needed.add((s["sequence"], s["tid_b"]))

        by_seq: Dict[str, set] = {}
        for seq, tid in needed:
            by_seq.setdefault(seq, set()).add(tid)

        for seq, tids in by_seq.items():
            cache_path = Path(cache_root) / f"cache_attributes_{seq}.pkl"
            if not cache_path.exists():
                print(f"  [WARN] Cache not found: {cache_path}")
                continue
            with open(cache_path, "rb") as f:
                tracklets = pickle.load(f)
            for tid in tids:
                if tid in tracklets:
                    feats, frames = extract_frame_features(tracklets[tid])
                    self._feature_cache[(seq, tid)] = (feats, frames)

        print(f"  Loaded features for {len(self._feature_cache)} tracklets")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        feats_a, frames_a = self._feature_cache[(s["sequence"], s["tid_a"])]

        # Apply crops
        ca = s["crop_a"]
        feats_a = feats_a[ca[0]:ca[1]]
        frames_a = frames_a[ca[0]:ca[1]]

        if s["tid_b"] is not None:
            # Synthetic concatenation
            feats_b, frames_b = self._feature_cache[(s["sequence"], s["tid_b"])]
            cb = s["crop_b"]
            feats_b = feats_b[cb[0]:cb[1]]
            frames_b = frames_b[cb[0]:cb[1]]

            # Insert temporal gap: shift B's frame numbers
            gap = s.get("gap_frames", 0)
            frames_b = frames_b.copy()
            frames_b += (frames_a[-1] - frames_b[0] + 1 + gap)

            # Concatenate
            features = np.concatenate([feats_a, feats_b], axis=0)
            frames = np.concatenate([frames_a, frames_b], axis=0)
        else:
            # Natural or negative — single tracklet
            features = feats_a
            frames = frames_a

        n = len(frames)

        # Build label vector
        labels = np.zeros(n, dtype=np.float32)
        for si in s["split_indices"]:
            lo = max(0, si - self.split_label_radius)
            hi = min(n, si + self.split_label_radius + 1)
            labels[lo:hi] = 1.0

        # Prepare sequence tensors
        tokens, mask, positions, frame_nums, labels = self._prepare_sequence(
            features, frames, labels
        )

        return tokens, mask, positions, frame_nums, labels

    def _prepare_sequence(
        self,
        features: np.ndarray,
        frames: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample, apply frame dropout, pad, create position tensors."""
        n = len(frames)

        # Temporal subsampling if too long — preserve split-point frames
        if n > self.t_max:
            # Always keep split-point frames
            split_mask = labels > 0
            split_idxs = set(np.where(split_mask)[0])

            indices = set(np.linspace(0, n - 1, self.t_max, dtype=int).tolist())
            indices |= split_idxs
            indices = sorted(indices)[:self.t_max]

            features = features[indices]
            frames = frames[indices]
            labels = labels[indices]
            n = len(indices)

        # Frame dropout during training — never drop split-point frames
        if self.training and self.frame_dropout > 0 and n > 2:
            keep = np.random.rand(n) > self.frame_dropout
            keep[0] = True
            keep[-1] = True
            keep[labels > 0] = True  # protect split labels
            if keep.sum() >= 2:
                features = features[keep]
                frames = frames[keep]
                labels = labels[keep]
                n = int(keep.sum())

        # Pad to t_max
        padded = np.zeros((self.t_max, RAW_DIM), dtype=np.float32)
        padded[:n] = features

        mask = np.ones(self.t_max, dtype=bool)
        mask[:n] = False

        positions = np.zeros(self.t_max, dtype=np.float32)
        if n > 1:
            positions[:n] = np.arange(n, dtype=np.float32) / (n - 1)

        frame_nums = np.zeros(self.t_max, dtype=np.float32)
        frame_nums[:n] = frames / self.max_frame_value

        padded_labels = np.zeros(self.t_max, dtype=np.float32)
        padded_labels[:n] = labels

        return (
            torch.from_numpy(padded),
            torch.from_numpy(mask),
            torch.from_numpy(positions),
            torch.from_numpy(frame_nums),
            torch.from_numpy(padded_labels),
        )
