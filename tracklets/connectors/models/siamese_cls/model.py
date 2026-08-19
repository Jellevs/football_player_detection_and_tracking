"""
Siamese CLS Transformer for tracklet merge classification.

Each tracklet is independently encoded by a shared Transformer encoder.
A learned [CLS] token aggregates the sequence representation.
The two CLS embeddings are compared via a classification head.
"""

import copy
import torch
import torch.nn as nn

from .config import TransformerMergerConfig
from .modules import FrameProjector, DualPositionalEncoding, ClassificationHead


class SiameseCLSTransformer(nn.Module):
    """
    Siamese Transformer that encodes two variable-length tracklet
    frame sequences and predicts whether they should be merged.

    Architecture:
        Per-frame tokens → FrameProjector → + DualPositionalEncoding
        → Prepend [CLS] → TransformerEncoder (shared weights)
        → Extract CLS → ClassificationHead([cls_a, cls_b, |a-b|, a*b, pw])
    """

    def __init__(self, config: TransformerMergerConfig = None):
        super().__init__()
        if config is None:
            config = TransformerMergerConfig()
        self.config = config

        # Token projection: raw per-frame features → d_model
        self.token_projector = FrameProjector(config)

        # Positional encoding
        self.pos_encoding = DualPositionalEncoding(config.d_model)

        # Learned CLS token
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, config.d_model) * 0.02
        )

        # Transformer encoder layers (manual list for stochastic depth)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-norm for more stable training
        )
        encoder_layer.self_attn.dropout = config.attention_dropout
        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(config.num_layers)]
        )
        self.encoder_norm = nn.LayerNorm(config.d_model)

        # Classification head
        self.classifier = ClassificationHead(config)

    def encode_tracklet(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        positions: torch.Tensor,
        frame_numbers: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a single tracklet into a CLS embedding.

        Args:
            tokens:        (B, T, 1292) raw per-frame features
            mask:          (B, T) True where padded (ignored by attention)
            positions:     (B, T) normalized intra-tracklet positions (0..1)
            frame_numbers: (B, T) normalized absolute frame numbers

        Returns:
            (B, d_model) CLS embedding
        """
        B = tokens.size(0)

        # Project tokens to d_model
        x = self.token_projector(tokens)                          # (B, T, d_model)

        # Add positional encoding
        x = x + self.pos_encoding(positions, frame_numbers)       # (B, T, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)                    # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                            # (B, T+1, d_model)

        # Extend mask: CLS is never masked
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        pad_mask = torch.cat([cls_mask, mask], dim=1)             # (B, T+1)

        # Transformer with stochastic depth
        sd = self.config.stochastic_depth
        for i, layer in enumerate(self.encoder_layers):
            if self.training and sd > 0:
                # Linearly increasing drop probability per layer
                drop_prob = sd * (i + 1) / len(self.encoder_layers)
                if torch.rand(1).item() < drop_prob:
                    continue
            x = layer(x, src_key_padding_mask=pad_mask)

        x = self.encoder_norm(x)

        # Return CLS token output (position 0)
        return x[:, 0]

    def forward(
        self,
        tokens_a: torch.Tensor,  mask_a: torch.Tensor,
        pos_a: torch.Tensor,     frames_a: torch.Tensor,
        tokens_b: torch.Tensor,  mask_b: torch.Tensor,
        pos_b: torch.Tensor,     frames_b: torch.Tensor,
        pairwise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for a batch of tracklet pairs.

        Returns: (B,) logits for merge probability
        """
        cls_a = self.encode_tracklet(tokens_a, mask_a, pos_a, frames_a)
        cls_b = self.encode_tracklet(tokens_b, mask_b, pos_b, frames_b)
        return self.classifier(cls_a, cls_b, pairwise)
