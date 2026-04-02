"""Shared building blocks for the Siamese CLS Transformer."""

import math
import copy
import torch
import torch.nn as nn


class TokenProjector(nn.Module):
    """
    Projects raw per-frame features (1292-dim) into d_model tokens.

    Separate projection heads per modality allow different treatment of
    embeddings (high-dim, pre-trained) vs scalars (low-dim, heterogeneous).

    Input layout (per frame):
        [0:512)      ReID embedding (L2-normalized)
        [512:1280)   SigLIP embedding
        [1280:1287)  BBox derived: cx, cy, w, h, aspect, area, cy_norm
        [1287:1292)  Scalars: jersey, entropy, jersey_conf, team, team_conf
        [1292:1293)  Detection score
    """

    def __init__(self, config):
        super().__init__()
        c = config

        self.reid_proj = nn.Sequential(
            nn.Linear(c.reid_dim, c.reid_proj_dim),
            nn.LayerNorm(c.reid_proj_dim),
            nn.GELU(),
        )
        self.siglip_proj = nn.Sequential(
            nn.Linear(c.siglip_dim, c.siglip_proj_dim),
            nn.LayerNorm(c.siglip_proj_dim),
            nn.GELU(),
        )
        self.bbox_proj = nn.Sequential(
            nn.Linear(c.bbox_feat_dim, c.bbox_proj_dim),
            nn.LayerNorm(c.bbox_proj_dim),
            nn.GELU(),
        )
        self.scalar_proj = nn.Sequential(
            nn.Linear(c.scalar_dim, c.scalar_proj_dim),
            nn.LayerNorm(c.scalar_proj_dim),
            nn.GELU(),
        )
        self.score_proj = nn.Sequential(
            nn.Linear(c.score_dim, c.score_proj_dim),
            nn.LayerNorm(c.score_proj_dim),
            nn.GELU(),
        )

        self.final_proj = nn.Linear(c.projected_dim, c.d_model)

        # Store slice boundaries
        self._reid_end = c.reid_dim
        self._siglip_end = c.reid_dim + c.siglip_dim
        self._bbox_end = self._siglip_end + c.bbox_feat_dim
        self._scalar_end = self._bbox_end + c.scalar_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, T, 1292)
        Returns: (B, T, d_model)
        """
        reid   = tokens[..., :self._reid_end]
        siglip = tokens[..., self._reid_end:self._siglip_end]
        bbox   = tokens[..., self._siglip_end:self._bbox_end]
        scalar = tokens[..., self._bbox_end:self._scalar_end]
        score  = tokens[..., self._scalar_end:]

        projected = torch.cat([
            self.reid_proj(reid),
            self.siglip_proj(siglip),
            self.bbox_proj(bbox),
            self.scalar_proj(scalar),
            self.score_proj(score),
        ], dim=-1)

        return self.final_proj(projected)


class DualPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding using two continuous signals:
      - Intra-tracklet position (normalized 0→1, scaled to 0→100)
      - Absolute frame number (normalized by max_frame, scaled to 0→100)

    Each signal produces d_model/2 dims; concatenated to d_model.
    Supports arbitrary float positions (not just integers).
    """

    def __init__(self, d_model: int):
        super().__init__()
        half_dim = d_model // 2
        # Frequency bands: log-spaced from 1 to 10000
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

    Input: [cls_a, cls_b, |cls_a - cls_b|, cls_a * cls_b, pairwise]
    Output: scalar logit
    """

    def __init__(self, config):
        super().__init__()
        input_dim = config.d_model * 4 + config.pairwise_dim

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
        combined = torch.cat([
            cls_a,
            cls_b,
            torch.abs(cls_a - cls_b),
            cls_a * cls_b,
            pairwise,
        ], dim=-1)
        return self.net(combined).squeeze(-1)
