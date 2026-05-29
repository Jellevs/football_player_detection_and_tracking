"""
Hybrid Transformer for tracklet merge classification.

Two-stage architecture combining self-attention and cross-attention:
  Stage 1 (Self-attention encoder): each tracklet is encoded independently
           with a shared Transformer encoder. Unlike the Siamese CLS model,
           we retain the full output sequence (not just a CLS summary).
  Stage 2 (Cross-attention): each tracklet's contextualized frames attend to
           the other tracklet's frames, enabling direct frame-to-frame
           comparison on top of already-enriched representations.

This is the natural progression from the two simpler architectures:
  - Siamese CLS uses self-attention only (compare compressed summaries)
  - Cross-attention uses cross-attention only (compare raw projected frames)
  - Hybrid uses both (build context first, then compare)

The self-attention stage lets each frame "see" its own tracklet's temporal
context (e.g., frame 50 knows about ReID drift from frame 1 to frame 100),
so cross-attention can make more informed comparisons.
"""

import copy
import math
import torch
import torch.nn as nn

from .config import HybridConfig


# ---------------------------------------------------------------------------
# Shared components (same as cross-attention model)
# ---------------------------------------------------------------------------

class FrameProjector(nn.Module):
    """Projects raw per-frame features (525 dim) into d_model tokens."""

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(config.raw_token_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.projection_dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(tokens)


class DualPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding using two continuous signals:
    intra-tracklet position and absolute frame number.
    """

    def __init__(self, d_model: int):
        super().__init__()
        half_dim = d_model // 2
        freqs = torch.exp(
            torch.arange(0, half_dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / half_dim)
        )
        self.register_buffer("freqs", freqs)

    def _encode(self, positions: torch.Tensor) -> torch.Tensor:
        args = positions.unsqueeze(-1) * self.freqs
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def forward(self, intra_positions, frame_numbers):
        pe_intra = self._encode(intra_positions * 100.0)
        pe_absolute = self._encode(frame_numbers * 100.0)
        return torch.cat([pe_intra, pe_absolute], dim=-1)


# ---------------------------------------------------------------------------
# Stage 1: Self-attention encoder
# ---------------------------------------------------------------------------

class SelfAttentionEncoder(nn.Module):
    """
    Shared self-attention encoder applied independently to each tracklet.
    Retains the full output sequence for downstream cross-attention.
    Uses a learned CLS token at position 0 to aggregate the sequence.
    """

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, config.d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.self_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(config.num_self_layers)]
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            x:    (B, T, d_model) projected frame tokens
            mask: (B, T) True where padded
        Returns:
            seq:      (B, T+1, d_model) full sequence with CLS at position 0
            pad_mask: (B, T+1) extended padding mask
        """
        B = x.size(0)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Extend mask (CLS is never masked)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        pad_mask = torch.cat([cls_mask, mask], dim=1)

        # Self-attention with stochastic depth
        sd = self.config.stochastic_depth
        for i, layer in enumerate(self.layers):
            if self.training and sd > 0:
                drop_prob = sd * (i + 1) / len(self.layers)
                if torch.rand(1).item() < drop_prob:
                    continue
            x = layer(x, src_key_padding_mask=pad_mask)

        x = self.norm(x)
        return x, pad_mask


# ---------------------------------------------------------------------------
# Stage 2: Cross-attention (separate weights, sequential update)
# ---------------------------------------------------------------------------

class CrossAttentionLayer(nn.Module):
    """
    Bidirectional cross-attention with separate weights for A->B and B->A.
    Uses pre-norm and sequential update (matching the best-performing
    cross-attention model configuration).
    """

    def __init__(self, config: HybridConfig):
        super().__init__()
        d = config.d_model

        # A -> B cross-attention
        self.norm_a = nn.LayerNorm(d)
        self.norm_b_for_a = nn.LayerNorm(d)
        self.cross_attn_a = nn.MultiheadAttention(
            embed_dim=d, num_heads=config.nhead,
            dropout=config.cross_dropout, batch_first=True,
        )

        # B -> A cross-attention
        self.norm_b = nn.LayerNorm(d)
        self.norm_a_for_b = nn.LayerNorm(d)
        self.cross_attn_b = nn.MultiheadAttention(
            embed_dim=d, num_heads=config.nhead,
            dropout=config.cross_dropout, batch_first=True,
        )

        # Feed-forward (separate weights per side)
        self.ff_a = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Dropout(config.cross_dropout),
            nn.Linear(d * 2, d),
            nn.Dropout(config.cross_dropout),
        )
        self.ff_b = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Dropout(config.cross_dropout),
            nn.Linear(d * 2, d),
            nn.Dropout(config.cross_dropout),
        )

    def forward(self, x_a, x_b, mask_a, mask_b):
        # A attends to B
        q_a = self.norm_a(x_a)
        kv_b = self.norm_b_for_a(x_b)
        attn_a, _ = self.cross_attn_a(q_a, kv_b, kv_b, key_padding_mask=mask_b)
        x_a = x_a + attn_a
        x_a = x_a + self.ff_a(x_a)

        # B attends to A (sequential: sees A's updated state)
        q_b = self.norm_b(x_b)
        kv_a = self.norm_a_for_b(x_a)
        attn_b, _ = self.cross_attn_b(q_b, kv_a, kv_a, key_padding_mask=mask_a)
        x_b = x_b + attn_b
        x_b = x_b + self.ff_b(x_b)

        return x_a, x_b


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class HybridTransformer(nn.Module):
    """
    Hybrid (self-attention -> cross-attention) Transformer.

    Architecture:
        1. Project per-frame features to d_model
        2. Add positional encoding
        3. Self-attention encoder (shared, applied to each tracklet independently)
        4. Cross-attention (A's frames attend to B's and vice versa)
        5. Extract CLS token from each side
        6. Classify: [cls_a, cls_b, |a-b|, a*b, pairwise] -> logit
    """

    def __init__(self, config: HybridConfig = None):
        super().__init__()
        if config is None:
            config = HybridConfig()
        self.config = config
        d = config.d_model

        # Frame projection
        self.projector = FrameProjector(config)

        # Positional encoding
        self.pos_encoding = DualPositionalEncoding(d)

        # Stage 1: self-attention encoder (shared for both tracklets)
        self.self_encoder = SelfAttentionEncoder(config)

        # Stage 2: cross-attention layers
        self.cross_layers = nn.ModuleList([
            CrossAttentionLayer(config)
            for _ in range(config.num_cross_layers)
        ])

        # Classification head
        clf_input = d * 4 + config.pairwise_dim
        self.classifier = nn.Sequential(
            nn.Linear(clf_input, config.classifier_hidden),
            nn.LayerNorm(config.classifier_hidden),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.classifier_hidden, 1),
        )

    def forward(
        self,
        tokens_a: torch.Tensor, mask_a: torch.Tensor,
        pos_a: torch.Tensor, frames_a: torch.Tensor,
        tokens_b: torch.Tensor, mask_b: torch.Tensor,
        pos_b: torch.Tensor, frames_b: torch.Tensor,
        pairwise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for a batch of tracklet pairs.
        Returns: (B,) logits for merge probability
        """
        # Project frames
        x_a = self.projector(tokens_a)
        x_b = self.projector(tokens_b)

        # Add positional encoding
        x_a = x_a + self.pos_encoding(pos_a, frames_a)
        x_b = x_b + self.pos_encoding(pos_b, frames_b)

        # Stage 1: self-attention (shared encoder, applied independently)
        seq_a, pad_a = self.self_encoder(x_a, mask_a)
        seq_b, pad_b = self.self_encoder(x_b, mask_b)

        # Stage 2: cross-attention
        for layer in self.cross_layers:
            seq_a, seq_b = layer(seq_a, seq_b, pad_a, pad_b)

        # Extract CLS tokens (position 0)
        cls_a = seq_a[:, 0]
        cls_b = seq_b[:, 0]

        # Classify
        combined = torch.cat([
            cls_a,
            cls_b,
            torch.abs(cls_a - cls_b),
            cls_a * cls_b,
            pairwise,
        ], dim=-1)

        return self.classifier(combined).squeeze(-1)
