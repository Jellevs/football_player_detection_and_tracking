"""Shared building blocks for the Siamese CLS Transformer."""

import math
import copy
import torch
import torch.nn as nn


class FrameProjector(nn.Module):
    """
    Projects raw per-frame features (1292 dim) into d_model tokens.

    Single linear projection with layer norm - simple and effective.
    Same design as used in the Cross-Attention and Hybrid architectures.

    Input layout (per frame, 1292 dim total):
        [0:512)      ReID embedding (L2 normalized)
        [512:1280)   SigLIP embedding (L2 normalized, 768 dim)
        [1280:1287)  BBox derived: cx, cy, w, h, aspect, area, cy_norm
        [1287:1292)  Scalars: jersey, entropy, jersey_conf, team, team_conf
    """

    def __init__(self, config):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(config.raw_token_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.projection_dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, T, 1292)
        Returns: (B, T, d_model)
        """
        return self.proj(tokens)


class DualPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding using two continuous signals:
      - Intra tracklet position (normalized 0 to 1, scaled to 0 to 100)
      - Absolute frame number (normalized by max_frame, scaled to 0 to 100)

    Each signal produces d_model/2 dims; concatenated to d_model.
    Supports arbitrary float positions (not just integers).
    """

    def __init__(self, d_model: int):
        super().__init__()
        half_dim = d_model // 2
        # Frequency bands: log spaced from 1 to 10000
        freqs = torch.exp(
            torch.arange(0, half_dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / half_dim)
        )
        self.register_buffer("freqs", freqs)  # (half_dim / 2,)

    def _encode(self, positions: torch.Tensor) -> torch.Tensor:
        """
        positions: (B, T) continuous float values
        Returns: (B, T, half_dim)
        """
        args = positions.unsqueeze(-1) * self.freqs  # (B, T, half_dim/2)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, T, half_dim)

    def forward(self, intra_positions: torch.Tensor,
                frame_numbers: torch.Tensor) -> torch.Tensor:
        """
        intra_positions: (B, T) normalized 0..1 within tracklet
        frame_numbers:   (B, T) normalized absolute frame numbers
        Returns: (B, T, d_model)
        """
        pe_intra    = self._encode(intra_positions * 100.0)
        pe_absolute = self._encode(frame_numbers * 100.0)
        return torch.cat([pe_intra, pe_absolute], dim=-1)


class ClassificationHead(nn.Module):
    """
    Combines two CLS embeddings with pairwise features and classifies.

    Input: [cls_a, cls_b, |cls_a  cls_b|, cls_a * cls_b, pairwise]
    Output: scalar logit
    """

    def __init__(self, config):
        super().__init__()
        self.use_pairwise = config.use_pairwise_features
        pw_dim = config.pairwise_dim if self.use_pairwise else 0
        self.pairwise_dim = pw_dim
        input_dim = config.d_model * 4 + pw_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(config.transformer_dropout),
            nn.Linear(64, 1),
        )

    def forward(self, cls_a: torch.Tensor, cls_b: torch.Tensor,
                pairwise: torch.Tensor) -> torch.Tensor:
        """
        cls_a, cls_b: (B, d_model)
        pairwise: (B, pairwise_dim)
        Returns: (B,) logits
        """
        parts = [
            cls_a,
            cls_b,
            torch.abs(cls_a - cls_b),
            cls_a * cls_b,
        ]
        if self.use_pairwise:
            if pairwise.size(-1) != self.pairwise_dim:
                raise ValueError(
                    f"Expected {self.pairwise_dim} pairwise features, "
                    f"got {pairwise.size(-1)}"
                )
            parts.append(pairwise)
        combined = torch.cat(parts, dim=-1)
        return self.net(combined).squeeze(-1)
