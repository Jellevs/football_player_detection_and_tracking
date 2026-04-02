"""
Per-frame split-point transformer.

Processes a single tracklet and outputs a split probability for every frame.
Reuses TokenProjector and DualPositionalEncoding from the merger codebase.
"""

import copy
import torch
import torch.nn as nn

from transformer.modules import TokenProjector, DualPositionalEncoding
from .config import SplitTransformerConfig


class SplitPointHead(nn.Module):
    """Per-frame MLP that maps d_model → 1 logit per frame."""

    def __init__(self, config: SplitTransformerConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.head_hidden),
            nn.GELU(),
            nn.LayerNorm(config.head_hidden),
            nn.Dropout(config.head_dropout),
            nn.Linear(config.head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        Returns: (B, T) logits
        """
        return self.net(x).squeeze(-1)


class SplitPointTransformer(nn.Module):
    """
    Transformer encoder with per-frame binary classification head.

    Architecture:
        Per-frame tokens → TokenProjector → + DualPositionalEncoding
        → TransformerEncoder → SplitPointHead → per-frame logits
    """

    def __init__(self, config: SplitTransformerConfig = None):
        super().__init__()
        if config is None:
            config = SplitTransformerConfig()
        self.config = config

        # Reused modules (same architecture as merger)
        self.token_projector = TokenProjector(config)
        self.pos_encoding = DualPositionalEncoding(config.d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(config.num_layers)]
        )
        self.encoder_norm = nn.LayerNorm(config.d_model)

        # Per-frame classification head
        self.head = SplitPointHead(config)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        positions: torch.Tensor,
        frame_numbers: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tokens:        (B, T, raw_token_dim) per-frame features
            mask:          (B, T) True where padded
            positions:     (B, T) normalized intra-tracklet positions 0..1
            frame_numbers: (B, T) normalized absolute frame numbers

        Returns:
            (B, T) per-frame logits (split probability before sigmoid)
        """
        x = self.token_projector(tokens)
        x = x + self.pos_encoding(positions, frame_numbers)

        # Transformer with stochastic depth
        sd = self.config.stochastic_depth
        for i, layer in enumerate(self.encoder_layers):
            if self.training and sd > 0:
                drop_prob = sd * (i + 1) / len(self.encoder_layers)
                if torch.rand(1).item() < drop_prob:
                    continue
            x = layer(x, src_key_padding_mask=mask)

        x = self.encoder_norm(x)
        return self.head(x)
