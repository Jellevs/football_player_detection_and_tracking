"""
Late Cross-Attention Transformer for tracklet merge classification.

Architecture (Plan Section 4, B3)
=================================
This is the recommended architecture from the plan — a late cross-attention
bi-encoder that sits between the pure bi-encoder and the full frame cross-encoder:

  Per tracklet (encode once, cache):
      11 tokens (bins + boundaries + stats)
      -> ModalityProjector / StatsProjector -> d_model
      -> + TokenTypeEmbedding
      -> Prepend [CLS]
      -> 3-layer self-attention TransformerEncoder
      -> Output: 12 token embeddings (including CLS)

  Per pair (lightweight, runs on cached tokens):
      Concatenate token sets: [pair_CLS, A_tokens, B_tokens]
      -> 2-layer cross-attention TransformerEncoder
         (A tokens attend to B tokens and vice versa)
      -> Extract pair_CLS embedding
      -> Residual classifier head: [pair_CLS, |A-B|, A*B, pairwise_34]

The cross-attention layers give the model a genuine capability that XGBoost
cannot have: directly comparing specific tokens across the two tracklets
(e.g., matching the one clear jersey frame in A to the one clear frame in B).

Residual head (B1): The classifier is structured so that with the learned
(CLS-derived) terms zeroed, it reproduces a logistic regression on the
pairwise features. The model thus *starts* at the linear-baseline operating
point and the transformer learns only the residual correction.

Total parameters: ~280K (vs ~200K for the temporal bin transformer,
still far smaller than the frame cross-encoder).
"""

import copy
import math
import torch
import torch.nn as nn

from .config import NewTransformerConfig


# ---------------------------------------------------------------------------
# Reuse projectors and token type embedding from the temporal bin transformer
# ---------------------------------------------------------------------------

class ModalityProjector(nn.Module):
    """
    Projects raw per-token features (1293-dim + 2 positional) to d_model.
    Separate projection heads per modality.
    """

    def __init__(self, config: NewTransformerConfig):
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
        self.pos_proj = nn.Sequential(
            nn.Linear(2, 8),
            nn.GELU(),
        )

        input_dim = c.projected_dim + 8
        self.final_proj = nn.Linear(input_dim, c.d_model)

        self._reid_end = c.reid_dim
        self._siglip_end = c.reid_dim + c.siglip_dim
        self._bbox_end = self._siglip_end + c.bbox_feat_dim
        self._scalar_end = self._bbox_end + c.scalar_dim
        self._score_end = self._scalar_end + c.score_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        reid = tokens[..., :self._reid_end]
        siglip = tokens[..., self._reid_end:self._siglip_end]
        bbox = tokens[..., self._siglip_end:self._bbox_end]
        scalar = tokens[..., self._bbox_end:self._scalar_end]
        score = tokens[..., self._scalar_end:self._score_end]
        pos = tokens[..., self._score_end:]

        projected = torch.cat([
            self.reid_proj(reid),
            self.siglip_proj(siglip),
            self.bbox_proj(bbox),
            self.scalar_proj(scalar),
            self.score_proj(score),
            self.pos_proj(pos),
        ], dim=-1)

        return self.final_proj(projected)


class StatsProjector(nn.Module):
    """Separate projector for the aggregate statistics token."""

    def __init__(self, config: NewTransformerConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.stats_dim + 2, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
        )
        self.stats_dim = config.stats_dim

    def forward(self, stats_token: torch.Tensor) -> torch.Tensor:
        meaningful = torch.cat([
            stats_token[..., :self.stats_dim],
            stats_token[..., -2:],
        ], dim=-1)
        return self.net(meaningful)


class TokenTypeEmbedding(nn.Module):
    """
    Learned embedding to distinguish token types.
    Types: 0=CLS, 1=temporal_bin, 2=start_boundary, 3=end_boundary, 4=stats
    """

    def __init__(self, config: NewTransformerConfig):
        super().__init__()
        self.embedding = nn.Embedding(5, config.d_model)

    def forward(self, n_bins: int, device: torch.device) -> torch.Tensor:
        types = torch.zeros(1 + n_bins + 3, dtype=torch.long, device=device)
        types[0] = 0                         # CLS
        types[1:1 + n_bins] = 1              # temporal bins
        types[1 + n_bins] = 2                # start boundary
        types[2 + n_bins] = 3                # end boundary
        types[3 + n_bins] = 4                # stats
        return self.embedding(types).unsqueeze(0)


# ---------------------------------------------------------------------------
# Cross-attention layer
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """
    A single cross-attention block for the pair interaction stage.

    This is a standard transformer encoder layer applied to the concatenation
    of [pair_CLS, A_tokens, B_tokens].  Since all tokens attend to all tokens
    via self-attention, A tokens naturally attend to B tokens and vice versa
    (cross-attention emerges from the shared self-attention).

    We add a tracklet identity embedding so the model can distinguish which
    tokens belong to which tracklet.
    """

    def __init__(self, config: NewTransformerConfig):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.cross_attn_nhead,
            dim_feedforward=config.cross_attn_dim_feedforward,
            dropout=config.cross_attn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


# ---------------------------------------------------------------------------
# Residual classifier head (B1)
# ---------------------------------------------------------------------------

class ResidualPairClassifier(nn.Module):
    """
    Classification head with residual structure (Plan B1).

    The head is structured so that with the learned (CLS-derived) terms zeroed,
    it reduces to a logistic regression on the pairwise features:

        logit = linear_baseline(pairwise) + neural_correction(pair_cls, pairwise)

    This means the model *starts* at the linear-baseline operating point and
    the cross-attention + transformer can only add value — it can never throw
    away the cosine-similarity signal.
    """

    def __init__(self, config: NewTransformerConfig):
        super().__init__()
        d = config.d_model
        pw_dim = config.pairwise_dim

        # Baseline branch: logistic regression on pairwise features
        self.baseline = nn.Linear(pw_dim, 1)

        # Neural correction branch: pair_CLS + pairwise -> residual logit
        # Input: [pair_cls, |cls_a - cls_b|, cls_a * cls_b, pairwise]
        correction_input_dim = d + d * 2 + pw_dim  # pair_cls + diff + prod + pw
        self.correction = nn.Sequential(
            nn.Linear(correction_input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(config.transformer_dropout),
            nn.Linear(64, 1),
        )

        # Learned gate: controls how much the correction contributes
        # Initialized near zero so the model starts at the baseline
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        pair_cls: torch.Tensor,      # (B, d_model) — from cross-attention
        cls_a: torch.Tensor,          # (B, d_model) — per-tracklet CLS
        cls_b: torch.Tensor,          # (B, d_model) — per-tracklet CLS
        pairwise: torch.Tensor,       # (B, pairwise_dim)
    ) -> torch.Tensor:
        # Baseline: logistic regression on pairwise features
        baseline_logit = self.baseline(pairwise).squeeze(-1)  # (B,)

        # Neural correction
        correction_input = torch.cat([
            pair_cls,
            torch.abs(cls_a - cls_b),
            cls_a * cls_b,
            pairwise,
        ], dim=-1)
        correction_logit = self.correction(correction_input).squeeze(-1)  # (B,)

        # Gated residual
        return baseline_logit + self.gate * correction_logit


class StandardPairClassifier(nn.Module):
    """
    Standard (non-residual) classifier head — fallback if residual head is disabled.
    Same structure as the temporal bin transformer's PairClassifier but takes
    pair_cls from cross-attention instead of concatenated CLS embeddings.
    """

    def __init__(self, config: NewTransformerConfig):
        super().__init__()
        d = config.d_model
        # [pair_cls, cls_a, cls_b, |a-b|, a*b, pairwise]
        input_dim = d * 5 + config.pairwise_dim

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

    def forward(
        self,
        pair_cls: torch.Tensor,
        cls_a: torch.Tensor,
        cls_b: torch.Tensor,
        pairwise: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([
            pair_cls,
            cls_a,
            cls_b,
            torch.abs(cls_a - cls_b),
            cls_a * cls_b,
            pairwise,
        ], dim=-1)
        return self.net(combined).squeeze(-1)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class LateCrossAttentionTransformer(nn.Module):
    """
    Late Cross-Attention Transformer for tracklet merge classification.

    Two-stage architecture:
        Stage 1 (encode once, cache per tracklet):
            bin_tokens -> project -> self-attention -> token set + CLS

        Stage 2 (run per pair):
            [pair_CLS] + A_tokens + B_tokens + tracklet_id_embedding
            -> cross-attention layers
            -> pair_CLS output
            -> residual classifier head
    """

    def __init__(self, config: NewTransformerConfig = None):
        super().__init__()
        if config is None:
            config = NewTransformerConfig()
        self.config = config

        # ---- Stage 1: Per-tracklet encoding (identical to temporal bin) ----
        self.modality_proj = ModalityProjector(config)
        self.stats_proj = StatsProjector(config)
        self.type_embedding = TokenTypeEmbedding(config)

        # Learnable CLS token for per-tracklet encoding
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, config.d_model) * 0.02
        )

        # Self-attention encoder (per-tracklet)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.self_attn_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(config.num_self_attn_layers)]
        )
        self.self_attn_norm = nn.LayerNorm(config.d_model)

        # ---- Stage 2: Cross-attention (per-pair) ----
        # Learnable pair CLS token
        self.pair_cls_token = nn.Parameter(
            torch.randn(1, 1, config.d_model) * 0.02
        )

        # Tracklet identity embedding (A=0, B=1)
        self.tracklet_id_embedding = nn.Embedding(2, config.d_model)

        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList(
            [CrossAttentionBlock(config) for _ in range(config.num_cross_attn_layers)]
        )
        self.cross_attn_norm = nn.LayerNorm(config.d_model)

        # ---- Classification head ----
        if config.use_residual_head:
            self.classifier = ResidualPairClassifier(config)
        else:
            self.classifier = StandardPairClassifier(config)

        self._init_weights()

    def _init_weights(self):
        """Xavier init for linear layers, small normal for embeddings."""
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

    # ------------------------------------------------------------------
    # Stage 1: Per-tracklet encoding
    # ------------------------------------------------------------------

    def encode_tracklet(self, bin_tokens: torch.Tensor) -> torch.Tensor:
        """
        Encode a single tracklet into its full token set.

        Args:
            bin_tokens: (B, n_bins+3, raw_dim+2)

        Returns:
            token_set: (B, n_tokens, d_model) — full token set including CLS
                       token_set[:, 0] is the CLS embedding
        """
        B = bin_tokens.size(0)
        n_bins = self.config.n_temporal_bins

        # Split into regular tokens and stats token
        regular_tokens = bin_tokens[:, :n_bins + 2]
        stats_token = bin_tokens[:, n_bins + 2]

        # Project
        regular_proj = self.modality_proj(regular_tokens)
        stats_proj = self.stats_proj(stats_token).unsqueeze(1)

        # Concatenate: [bins, boundaries, stats]
        x = torch.cat([regular_proj, stats_proj], dim=1)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Add token type embedding
        type_emb = self.type_embedding(n_bins, x.device)
        x = x + type_emb

        # Self-attention encoder with stochastic depth
        sd = self.config.stochastic_depth
        for i, layer in enumerate(self.self_attn_layers):
            if self.training and sd > 0:
                drop_prob = sd * (i + 1) / len(self.self_attn_layers)
                if torch.rand(1).item() < drop_prob:
                    continue
            x = layer(x)

        x = self.self_attn_norm(x)
        return x  # (B, n_tokens, d_model) — CLS at position 0

    def encode_tracklet_cls(self, bin_tokens: torch.Tensor) -> torch.Tensor:
        """Convenience: encode and return only the CLS embedding."""
        return self.encode_tracklet(bin_tokens)[:, 0]

    # ------------------------------------------------------------------
    # Stage 2: Cross-attention pair scoring
    # ------------------------------------------------------------------

    def cross_attend(
        self,
        tokens_a: torch.Tensor,  # (B, n_tokens, d_model)
        tokens_b: torch.Tensor,  # (B, n_tokens, d_model)
    ) -> torch.Tensor:
        """
        Run cross-attention over two tracklet token sets.

        Args:
            tokens_a: (B, n_tokens, d_model) — encoded token set for tracklet A
            tokens_b: (B, n_tokens, d_model) — encoded token set for tracklet B

        Returns:
            pair_cls: (B, d_model) — the pair CLS output for classification
        """
        B = tokens_a.size(0)

        # Add tracklet identity embeddings
        id_a = torch.zeros(B, tokens_a.size(1), dtype=torch.long, device=tokens_a.device)
        id_b = torch.ones(B, tokens_b.size(1), dtype=torch.long, device=tokens_b.device)
        tokens_a = tokens_a + self.tracklet_id_embedding(id_a)
        tokens_b = tokens_b + self.tracklet_id_embedding(id_b)

        # Prepend pair CLS token
        pair_cls = self.pair_cls_token.expand(B, -1, -1)

        # Concatenate: [pair_CLS, A_tokens, B_tokens]
        x = torch.cat([pair_cls, tokens_a, tokens_b], dim=1)

        # Cross-attention layers
        for layer in self.cross_attn_layers:
            x = layer(x)

        x = self.cross_attn_norm(x)

        # Return pair CLS (position 0)
        return x[:, 0]

    # ------------------------------------------------------------------
    # Full forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        bins_a: torch.Tensor,      # (B, n_bins+3, raw_dim+2)
        bins_b: torch.Tensor,      # (B, n_bins+3, raw_dim+2)
        pairwise: torch.Tensor,    # (B, 34)
        return_cls: bool = False,
    ):
        """
        Full forward pass for training.

        Args:
            return_cls: If True, also return per-tracklet CLS embeddings
                        (needed for contrastive loss without a second forward).

        Returns:
            logits: (B,)
            If return_cls=True: (logits, cls_a, cls_b)
        """
        # Stage 1: encode each tracklet
        tokens_a = self.encode_tracklet(bins_a)   # (B, n_tokens, d_model)
        tokens_b = self.encode_tracklet(bins_b)   # (B, n_tokens, d_model)

        cls_a = tokens_a[:, 0]  # (B, d_model)
        cls_b = tokens_b[:, 0]  # (B, d_model)

        # Stage 2: cross-attention
        pair_cls = self.cross_attend(tokens_a, tokens_b)

        # Classification
        logits = self.classifier(pair_cls, cls_a, cls_b, pairwise)

        if return_cls:
            return logits, cls_a, cls_b
        return logits

    def forward_from_cached(
        self,
        tokens_a: torch.Tensor,    # (B, n_tokens, d_model) — pre-encoded
        tokens_b: torch.Tensor,    # (B, n_tokens, d_model) — pre-encoded
        pairwise: torch.Tensor,    # (B, 34)
    ) -> torch.Tensor:
        """
        Forward pass using pre-encoded token sets (for efficient inference).
        Stage 1 is skipped; only cross-attention + classifier run.
        """
        cls_a = tokens_a[:, 0]
        cls_b = tokens_b[:, 0]

        pair_cls = self.cross_attend(tokens_a, tokens_b)
        return self.classifier(pair_cls, cls_a, cls_b, pairwise)


# ---------------------------------------------------------------------------
# EMA wrapper (C4)
# ---------------------------------------------------------------------------

class EMAModel:
    """
    Exponential Moving Average of model weights for smoother inference.

    Maintains a shadow copy of the model parameters updated as:
        shadow = decay * shadow + (1 - decay) * current

    At inference, swap in the shadow weights for better-calibrated predictions.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: nn.Module):
        """Swap in EMA weights for inference."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original weights after inference."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
