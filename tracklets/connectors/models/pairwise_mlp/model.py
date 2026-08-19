"""
Pairwise MLP baseline for tracklet merge classification.

Uses ONLY the 13-dim pairwise features (no per-frame sequence data).
This serves as a baseline to measure whether the transformer architectures
extract useful information from frame-level features beyond what the
handcrafted pairwise features already capture.

Architecture:
    pairwise (13) -> MLP -> sigmoid
"""

import torch
import torch.nn as nn


class PairwiseMLP(nn.Module):
    """
    Simple MLP classifier operating only on pairwise features.

    Architecture:
        13 -> 64 -> 32 -> 1
    """

    def __init__(self, pairwise_dim: int = 13, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(pairwise_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, pairwise: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pairwise: (B, 13) pairwise features
        Returns:
            (B,) logits
        """
        return self.classifier(pairwise).squeeze(-1)
