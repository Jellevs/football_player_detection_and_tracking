"""
Cross Attention Transformer for tracklet merge classification.

Instead of encoding each tracklet independently, frames from tracklet A
directly attend to frames from tracklet B (and vice versa). This lets the
model discover frame level correspondences and mismatches that aggregate
features like mean cosine similarity would miss.

Architecture:
    Per frame tokens -> Linear projection -> + DualPositionalEncoding
    -> Cross attention (A attends to B, B attends to A)
    -> Attention pooling (learned query summarizes each cross attended sequence)
    -> ClassificationHead([pool_a, pool_b, |a-b|, a*b, pairwise])
"""

import math
import torch
import torch.nn as nn

from .config import CrossAttentionConfig


class FrameProjector(nn.Module):
    """
    Projects raw per frame features (1292 dim) into d_model tokens.
    Single linear projection with layer norm, much lighter than the
    multi head projector in siamese_cls.
    """

    def __init__(self, config: CrossAttentionConfig):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(config.raw_token_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.projection_dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T, 1292) -> (B, T, d_model)"""
        return self.proj(tokens)


class DualPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding using two continuous signals:
    intra tracklet position and absolute frame number.
    Each produces d_model/2 dims, concatenated to d_model.
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


class CrossAttentionLayer(nn.Module):
    """
    Bidirectional cross attention: A attends to B, B attends to A.
    Uses pre norm for stable training.
    """

    def __init__(self, config: CrossAttentionConfig):
        super().__init__()
        d = config.d_model

        # A -> B cross attention
        self.norm_a = nn.LayerNorm(d)
        self.norm_b_for_a = nn.LayerNorm(d)
        self.cross_attn_a = nn.MultiheadAttention(
            embed_dim=d, num_heads=config.nhead,
            dropout=config.cross_dropout, batch_first=True,
        )

        # B -> A cross attention
        self.norm_b = nn.LayerNorm(d)
        self.norm_a_for_b = nn.LayerNorm(d)
        self.cross_attn_b = nn.MultiheadAttention(
            embed_dim=d, num_heads=config.nhead,
            dropout=config.cross_dropout, batch_first=True,
        )

        # Feed forward (shared structure, separate weights)
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
        """
        x_a: (B, T, d_model), x_b: (B, T, d_model)
        mask_a, mask_b: (B, T) True where padded
        Returns updated x_a, x_b
        """
        # A attends to B (query=A, key/value=B)
        q_a = self.norm_a(x_a)
        kv_b = self.norm_b_for_a(x_b)
        attn_a, _ = self.cross_attn_a(
            q_a, kv_b, kv_b, key_padding_mask=mask_b,
        )
        x_a = x_a + attn_a
        x_a = x_a + self.ff_a(x_a)

        # B attends to A (query=B, key/value=A)
        q_b = self.norm_b(x_b)
        kv_a = self.norm_a_for_b(x_a)
        attn_b, _ = self.cross_attn_b(
            q_b, kv_a, kv_a, key_padding_mask=mask_a,
        )
        x_b = x_b + attn_b
        x_b = x_b + self.ff_b(x_b)

        return x_a, x_b


class AttentionPooling(nn.Module):
    """
    Pools a variable length sequence into a fixed vector using a learned
    query that attends over all frames. Much cheaper than a full CLS token
    through multiple self attention layers.
    """

    def __init__(self, d_model: int, num_heads: int = 2):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        mask: (B, T) True where padded
        Returns: (B, d_model)
        """
        B = x.size(0)
        q = self.query.expand(B, -1, -1)  # (B, 1, d_model)
        pooled, _ = self.attn(q, x, x, key_padding_mask=mask)
        return self.norm(pooled.squeeze(1))  # (B, d_model)


class CrossAttentionTransformer(nn.Module):
    """
    Cross Attention Transformer for tracklet pair classification.

    Architecture:
        1. Project per frame features to d_model
        2. Add positional encoding
        3. Cross attention: A's frames attend to B's frames and vice versa
        4. Attention pool each sequence to a fixed vector
        5. Classify: [pool_a, pool_b, |a-b|, a*b, pairwise] -> logit
    """

    def __init__(self, config: CrossAttentionConfig = None):
        super().__init__()
        if config is None:
            config = CrossAttentionConfig()
        self.config = config
        d = config.d_model

        # Frame projection
        self.projector = FrameProjector(config)

        # Positional encoding
        self.pos_encoding = DualPositionalEncoding(d)

        # Cross attention layers
        self.cross_layers = nn.ModuleList([
            CrossAttentionLayer(config)
            for _ in range(config.num_cross_layers)
        ])

        # Attention pooling (one shared pooler for both tracklets)
        self.pooler = AttentionPooling(d, num_heads=config.pool_heads)

        # Classification head
        # Input: [pool_a, pool_b, |pool_a - pool_b|, pool_a * pool_b, pairwise]
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
        x_a = self.projector(tokens_a)  # (B, T, d)
        x_b = self.projector(tokens_b)

        # Add positional encoding
        x_a = x_a + self.pos_encoding(pos_a, frames_a)
        x_b = x_b + self.pos_encoding(pos_b, frames_b)

        # Cross attention
        for layer in self.cross_layers:
            x_a, x_b = layer(x_a, x_b, mask_a, mask_b)

        # Pool each sequence
        pool_a = self.pooler(x_a, mask_a)  # (B, d)
        pool_b = self.pooler(x_b, mask_b)  # (B, d)

        # Classify
        combined = torch.cat([
            pool_a,
            pool_b,
            torch.abs(pool_a - pool_b),
            pool_a * pool_b,
            pairwise,
        ], dim=-1)

        return self.classifier(combined).squeeze(-1)  # (B,)
