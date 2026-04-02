"""
Cross-attention Siamese Transformer for tracklet merge classification.

Two-stage architecture:
  Stage 1 (Siamese encoder): each tracklet is encoded independently with a
           shared Transformer encoder — identical to SiameseCLSTransformer.
           The full output sequence is retained (not just the CLS token).
  Stage 2 (Cross-attention): each tracklet's sequence attends to the other,
           allowing direct frame-to-frame interaction before classification.

The key advantage over the vanilla Siamese CLS model:
  The model can ask "does the exit of A look like the entry of B?" at the
  frame level, rather than comparing two compressed 128-dim vectors.

Inference efficiency:
  Stage 1 (expensive) is computed once per tracklet and cached.
  Stage 2 (cross-attention) runs per pair — more expensive than the MLP head
  in the vanilla model but far more expressive.
"""

import copy
from dataclasses import dataclass
import torch
import torch.nn as nn

from .config import TransformerMergerConfig
from .modules import TokenProjector, DualPositionalEncoding, ClassificationHead


# ---------------------------------------------------------------------------
# Config extension
# ---------------------------------------------------------------------------

@dataclass
class CrossAttnConfig(TransformerMergerConfig):
    # ---- Encoder (same width as vanilla model) ----
    d_model:          int   = 128
    nhead:            int   = 4
    num_layers:       int   = 4
    dim_feedforward:  int   = 256
    transformer_dropout: float = 0.2
    stochastic_depth: float = 0.1

    # ---- Per-modality projections (same as vanilla) ----
    reid_proj_dim:    int   = 64
    siglip_proj_dim:  int   = 64
    bbox_proj_dim:    int   = 16
    scalar_proj_dim:  int   = 16
    score_proj_dim:   int   = 8

    # ---- Cross-attention ----
    num_cross_layers: int   = 2
    cross_dropout:    float = 0.1

    # ---- Classifier head ----
    classifier_dropout: float = 0.3

    # ---- Sequence length ----
    # Reduced from 200: cross-attention is O(T^2), so T=200 is ~10x more
    # expensive than T=64. Boundary frames carry the merge signal anyway.
    t_max: int = 64

    # ---- Training ----
    lr:           float = 2e-4
    batch_size:   int   = 64
    max_epochs:   int   = 120
    patience:     int   = 20
    warmup_epochs: int  = 5        # linear LR warmup before cosine decay


# ---------------------------------------------------------------------------
# Cross-attention block (pre-norm, residual, FFN)
# ---------------------------------------------------------------------------

class CrossAttnBlock(nn.Module):
    """
    One layer of cross-attention: query attends to key_value.

    Pre-norm design (same as the self-attention encoder) with residual
    connection and a position-wise FFN.

    Shared weights are used for A→B and B→A within the same layer,
    which enforces symmetry and halves cross-attention parameters.
    """

    def __init__(self, d_model: int, nhead: int, dim_ffn: int, dropout: float):
        super().__init__()
        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn    = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ffn, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        kv_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        query:           (B, T_q, d_model)
        key_value:       (B, T_kv, d_model)
        kv_padding_mask: (B, T_kv) True = padding position (ignored by attention)
        Returns:         (B, T_q, d_model)
        """
        # Cross-attention with pre-norm
        attn_out, _ = self.attn(
            self.norm_q(query),
            self.norm_kv(key_value),
            self.norm_kv(key_value),
            key_padding_mask=kv_padding_mask,
        )
        query = query + self.drop(attn_out)

        # FFN with pre-norm
        query = query + self.drop(self.ffn(self.norm_ffn(query)))
        return query


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class CrossAttentionTransformer(nn.Module):
    """
    Cross-attention Siamese Transformer.

    forward() signature matches SiameseCLSTransformer so the same training
    loop works without modification.

    For efficient inference, encode_tracklet() returns the full self-encoded
    sequence (not just CLS), and cross_and_classify() does the rest per pair.
    """

    def __init__(self, config: CrossAttnConfig = None):
        super().__init__()
        if config is None:
            config = CrossAttnConfig()
        self.config = config

        # ---- Stage 1: shared siamese encoder (same as SiameseCLSTransformer) ----
        self.token_projector = TokenProjector(config)
        self.pos_encoding    = DualPositionalEncoding(config.d_model)
        self.cls_token       = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = config.d_model,
            nhead          = config.nhead,
            dim_feedforward= config.dim_feedforward,
            dropout        = config.transformer_dropout,
            activation     = "gelu",
            batch_first    = True,
            norm_first     = True,
        )
        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(config.num_layers)]
        )
        self.encoder_norm = nn.LayerNorm(config.d_model)

        # ---- Stage 2: bidirectional cross-attention (shared weights A↔B) ----
        self.cross_layers = nn.ModuleList([
            CrossAttnBlock(
                d_model = config.d_model,
                nhead   = config.nhead,
                dim_ffn = config.dim_feedforward,
                dropout = config.cross_dropout,
            )
            for _ in range(config.num_cross_layers)
        ])

        # ---- Stage 3: classification head (same interface as before) ----
        self.classifier = ClassificationHead(config)

    # ------------------------------------------------------------------
    # Encoding (stage 1)
    # ------------------------------------------------------------------

    def encode_tracklet(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        positions: torch.Tensor,
        frame_numbers: torch.Tensor,
    ):
        """
        Encode a tracklet through the siamese self-attention encoder.

        Returns the full output sequence (B, T+1, d_model) and the
        corresponding padding mask (B, T+1) — NOT just the CLS token.
        The CLS token sits at position 0.
        """
        B = tokens.size(0)

        x = self.token_projector(tokens)                            # (B, T, d_model)
        x = x + self.pos_encoding(positions, frame_numbers)        # (B, T, d_model)

        cls      = self.cls_token.expand(B, -1, -1)                # (B, 1, d_model)
        x        = torch.cat([cls, x], dim=1)                      # (B, T+1, d_model)

        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        pad_mask = torch.cat([cls_mask, mask], dim=1)              # (B, T+1)

        sd = self.config.stochastic_depth
        for i, layer in enumerate(self.encoder_layers):
            if self.training and sd > 0:
                drop_prob = sd * (i + 1) / len(self.encoder_layers)
                if torch.rand(1).item() < drop_prob:
                    continue
            x = layer(x, src_key_padding_mask=pad_mask)

        x = self.encoder_norm(x)
        return x, pad_mask   # (B, T+1, d_model), (B, T+1)

    # ------------------------------------------------------------------
    # Cross-attention + classify (stages 2–3)
    # ------------------------------------------------------------------

    def cross_and_classify(
        self,
        seq_a: torch.Tensor, pad_a: torch.Tensor,
        seq_b: torch.Tensor, pad_b: torch.Tensor,
        pairwise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run cross-attention on a batch of pre-encoded sequence pairs and
        return classification logits.

        seq_a/b:    (B, T+1, d_model)  — full self-encoded sequences
        pad_a/b:    (B, T+1)           — True = padding
        pairwise:   (B, pairwise_dim)
        Returns:    (B,) logits
        """
        # Both cross-attentions are computed from the *previous* layer's
        # representations (not each other's updated version), so update
        # is parallel rather than sequential — avoids asymmetric leakage.
        for cross_layer in self.cross_layers:
            new_a = cross_layer(seq_a, seq_b, kv_padding_mask=pad_b)
            new_b = cross_layer(seq_b, seq_a, kv_padding_mask=pad_a)
            seq_a, seq_b = new_a, new_b

        # CLS token (position 0) aggregates the cross-attended context
        pool_a = seq_a[:, 0]   # (B, d_model)
        pool_b = seq_b[:, 0]   # (B, d_model)

        return self.classifier(pool_a, pool_b, pairwise)

    # ------------------------------------------------------------------
    # Full forward (used during training)
    # ------------------------------------------------------------------

    def forward(
        self,
        tokens_a: torch.Tensor, mask_a: torch.Tensor,
        pos_a: torch.Tensor,    frames_a: torch.Tensor,
        tokens_b: torch.Tensor, mask_b: torch.Tensor,
        pos_b: torch.Tensor,    frames_b: torch.Tensor,
        pairwise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full forward pass for a batch of pairs (used during training).
        Signature is identical to SiameseCLSTransformer.forward().

        Returns: (B,) logits
        """
        seq_a, pad_a = self.encode_tracklet(tokens_a, mask_a, pos_a, frames_a)
        seq_b, pad_b = self.encode_tracklet(tokens_b, mask_b, pos_b, frames_b)
        return self.cross_and_classify(seq_a, pad_a, seq_b, pad_b, pairwise)
