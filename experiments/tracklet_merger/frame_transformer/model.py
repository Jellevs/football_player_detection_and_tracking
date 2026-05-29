"""
Frame-level Cross-Attention Pair Transformer.

Instead of compressing tracklets into temporal bins, this model takes
per-frame features from both tracklets and lets them cross-attend directly.
The transformer sees [CLS] + frames_A + frames_B and learns which specific
frames across the two tracklets are most informative for the merge decision.

Architecture:
    1. Linear projection: raw_dim (1293) -> d_model (128)
    2. Add tracklet identity embedding (A=0, B=1)
    3. Add sinusoidal temporal encoding from actual frame numbers
    4. Prepend [CLS] token
    5. Transformer encoder with full self-attention across all frames
    6. CLS output + pairwise features (34d) -> MLP -> logit

The cross-attention between frames of different tracklets is the key
advantage over XGBoost: the model can compare specific frames (e.g.,
"frame 45 of tracklet A shows the same jersey as frame 12 of B").
"""

import math
import torch
import torch.nn as nn

from experiments.tracklet_merger.frame_transformer.config import FrameTransformerConfig


class FramePairTransformer(nn.Module):
    def __init__(self, config: FrameTransformerConfig):
        super().__init__()
        self.config = config
        d = config.d_model

        # ── Feature projection ──
        self.feature_proj = nn.Sequential(
            nn.Linear(config.raw_dim, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # ── Tracklet identity embedding (A=0, B=1) ──
        self.tracklet_emb = nn.Embedding(2, d)

        # ── CLS token ──
        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        # ── Transformer encoder ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,       # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_layers
        )
        self.post_norm = nn.LayerNorm(d)

        # ── Classifier head: CLS embedding + pairwise features -> logit ──
        self.classifier = nn.Sequential(
            nn.Linear(d + config.pairwise_dim, config.classifier_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden, config.classifier_hidden // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier/Kaiming init for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _sinusoidal_encoding(self, frame_numbers: torch.Tensor, d_model: int) -> torch.Tensor:
        """
        Sinusoidal positional encoding based on actual frame numbers.

        Args:
            frame_numbers: (B, N) integer/float frame numbers
            d_model: embedding dimension

        Returns:
            (B, N, d_model) positional encoding
        """
        pe = torch.zeros(*frame_numbers.shape, d_model, device=frame_numbers.device)
        position = frame_numbers.unsqueeze(-1).float()  # (B, N, 1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=frame_numbers.device).float()
            * -(math.log(10000.0) / d_model)
        )
        pe[..., 0::2] = torch.sin(position * div_term)
        pe[..., 1::2] = torch.cos(position * div_term)
        return pe

    def forward(
        self,
        feats_a: torch.Tensor,     # (B, N_a, raw_dim)
        frames_a: torch.Tensor,    # (B, N_a)
        mask_a: torch.Tensor,      # (B, N_a) True = padding
        feats_b: torch.Tensor,     # (B, N_b, raw_dim)
        frames_b: torch.Tensor,    # (B, N_b)
        mask_b: torch.Tensor,      # (B, N_b) True = padding
        pairwise: torch.Tensor,    # (B, pairwise_dim)
    ) -> torch.Tensor:
        """
        Forward pass. Returns raw logits (B,).

        The full sequence is [CLS] + A_frames + B_frames.
        Self-attention runs over the entire sequence, so A frames
        attend to B frames and vice versa (cross-attention emerges
        naturally from the shared transformer encoder).
        """
        B = feats_a.size(0)
        d = self.config.d_model

        # Project raw features to d_model
        tokens_a = self.feature_proj(feats_a)  # (B, N_a, d)
        tokens_b = self.feature_proj(feats_b)  # (B, N_b, d)

        # Add tracklet identity embeddings
        id_a = torch.zeros(B, tokens_a.size(1), dtype=torch.long, device=feats_a.device)
        id_b = torch.ones(B, tokens_b.size(1), dtype=torch.long, device=feats_b.device)
        tokens_a = tokens_a + self.tracklet_emb(id_a)
        tokens_b = tokens_b + self.tracklet_emb(id_b)

        # Add sinusoidal temporal encoding from actual frame numbers
        tokens_a = tokens_a + self._sinusoidal_encoding(frames_a, d)
        tokens_b = tokens_b + self._sinusoidal_encoding(frames_b, d)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d)

        # Concatenate: [CLS] + A_tokens + B_tokens
        tokens = torch.cat([cls, tokens_a, tokens_b], dim=1)  # (B, 1+N_a+N_b, d)

        # Build attention mask: True = ignore position
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=feats_a.device)
        attn_mask = torch.cat([cls_mask, mask_a, mask_b], dim=1)  # (B, 1+N_a+N_b)

        # Transformer encoder
        out = self.transformer(tokens, src_key_padding_mask=attn_mask)
        cls_out = self.post_norm(out[:, 0])  # (B, d)

        # Classify: CLS embedding + precomputed pairwise features
        combined = torch.cat([cls_out, pairwise], dim=1)  # (B, d + pw_dim)
        logit = self.classifier(combined).squeeze(-1)      # (B,)
        return logit
