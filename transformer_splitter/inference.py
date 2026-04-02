"""
Inference wrapper for the per-frame split-point transformer.

Drop-in replacement for UnifiedSplitter — same interface:
    splitter.split_tracklet(tracklet, next_available_id) → list[Tracklet] | None
"""

import json
import numpy as np
import torch
from pathlib import Path
from scipy.signal import find_peaks

from transformer.dataset import extract_frame_features, RAW_DIM
from transformer_splitter.model import SplitPointTransformer


class TransformerSplitter:
    """
    Detects identity switches within a single tracklet using a transformer
    that outputs per-frame split probabilities.
    """

    def __init__(
        self,
        model_path: Path,
        meta_path: Path,
        split_threshold: float = None,
        min_peak_distance: int = 15,
        min_fragment_length: int = 20,
        device: str = None,
    ):
        self.min_peak_distance = min_peak_distance
        self.min_fragment_length = min_fragment_length

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model
        checkpoint = torch.load(str(model_path), map_location=self.device, weights_only=False)
        self.config = checkpoint["config"]
        self.model = SplitPointTransformer(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Load metadata
        with open(meta_path) as f:
            meta = json.load(f)
        self.t_max = meta.get("t_max", self.config.t_max)
        self.max_frame_value = meta.get("max_frame_value", self.config.max_frame_value)

        # Use recall-optimized threshold from training, or override
        if split_threshold is not None:
            self.split_threshold = split_threshold
        else:
            self.split_threshold = meta.get("threshold", 0.3)

    # ------------------------------------------------------------------
    # Public interface (matches UnifiedSplitter)
    # ------------------------------------------------------------------

    def split_tracklet(self, tracklet, next_available_id):
        """
        Detect split points in a single tracklet.

        Returns:
            List of fragment Tracklets, or None if no split needed.
        """
        n = len(tracklet.frames)
        if n < 2 * self.min_fragment_length:
            return None

        # Get per-frame split probabilities
        probs = self._predict(tracklet)  # (n,)

        # Peak detection
        peaks, properties = find_peaks(
            probs,
            height=self.split_threshold,
            distance=self.min_peak_distance,
        )

        if len(peaks) == 0:
            return None

        # Convert peak indices to split boundaries
        split_indices = sorted(peaks.tolist())

        # Build fragments
        boundaries = [0] + split_indices + [n]
        fragments = []
        current_id = next_available_id

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1] - 1  # inclusive for tracklet.extract
            if end - start + 1 < self.min_fragment_length:
                continue
            fragment = tracklet.extract(start, end)
            fragment.track_id = current_id
            fragment.parent_id = tracklet.parent_id
            fragments.append(fragment)
            current_id += 1

        if len(fragments) < 2:
            return None
        return fragments

    # ------------------------------------------------------------------
    # Model inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict(self, tracklet) -> np.ndarray:
        """Run the model on a single tracklet, return per-frame probabilities."""
        features, frames = extract_frame_features(tracklet)
        n = len(frames)

        # Truncate to expected dim if needed (backward compat)
        expected_dim = self.config.raw_token_dim
        if features.shape[1] > expected_dim:
            features = features[:, :expected_dim]

        # Subsample if too long (preserve order)
        if n > self.t_max:
            indices = np.linspace(0, n - 1, self.t_max, dtype=int)
        else:
            indices = np.arange(n)

        sub_features = features[indices]
        sub_frames = frames[indices]
        n_sub = len(indices)

        # Pad to t_max
        feat_dim = sub_features.shape[1]
        padded = np.zeros((self.t_max, feat_dim), dtype=np.float32)
        padded[:n_sub] = sub_features

        mask = np.ones(self.t_max, dtype=bool)
        mask[:n_sub] = False

        positions = np.zeros(self.t_max, dtype=np.float32)
        if n_sub > 1:
            positions[:n_sub] = np.arange(n_sub, dtype=np.float32) / (n_sub - 1)

        frame_nums = np.zeros(self.t_max, dtype=np.float32)
        frame_nums[:n_sub] = sub_frames / self.max_frame_value

        # Forward pass
        tokens = torch.from_numpy(padded).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        pos_t = torch.from_numpy(positions).unsqueeze(0).to(self.device)
        fn_t = torch.from_numpy(frame_nums).unsqueeze(0).to(self.device)

        logits = self.model(tokens, mask_t, pos_t, fn_t)  # (1, t_max)
        probs_sub = torch.sigmoid(logits[0, :n_sub]).cpu().numpy()

        # Map back to original frame indices if we subsampled
        if n > self.t_max:
            probs_full = np.zeros(n, dtype=np.float32)
            # Interpolate probabilities back to all original frames
            probs_full[indices] = probs_sub
            # Fill gaps via linear interpolation
            for k in range(len(indices) - 1):
                i_start = indices[k]
                i_end = indices[k + 1]
                if i_end - i_start > 1:
                    frac = np.linspace(0, 1, i_end - i_start + 1)
                    probs_full[i_start:i_end + 1] = (
                        probs_sub[k] * (1 - frac) + probs_sub[k + 1] * frac
                    )
            return probs_full
        else:
            return probs_sub
