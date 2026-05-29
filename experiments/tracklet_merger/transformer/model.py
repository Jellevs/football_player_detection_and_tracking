"""
Temporal Bin Transformer for tracklet merge classification.

Architecture overview
=====================
Each tracklet is represented as a compact sequence of 12 tokens:
  - 1 learnable [CLS] token
  - 8 temporal bin tokens (averaged features within time slices)
  - 2 boundary tokens (first/last K frames)
  - 1 aggregate statistics token

These are processed by a shared 3-layer Transformer encoder with d_model=96.
The CLS output is used to represent the entire tracklet.

Two tracklet CLS embeddings are then compared via a classification head that
also receives 34-dim extended pairwise features (temporal gap, spatial distance,
embedding similarities, jersey/team agreement, per-tracklet statistics).

Design rationale:
  - 12 tokens vs 200: dramatically more data-efficient, attention has O(T^2) cost
  - Temporal bins preserve dynamics (how features change over time)
  - Boundary tokens emphasize merge-relevant information (tracklet endpoints)
  - Statistics token provides invariant summary (like XGBoost features)
  - Modality-specific projection (from existing codebase, proven to work)
  - Small model (~200K params) with strong regularisation to prevent overfitting

Total parameters: ~200K (vs ~450K for the original SiameseCLS with t_max=200)
"""

import copy
import math
import torch
import torch.nn as nn

from .config import TemporalBinConfig


class ModalityProjector(nn.Module):
    """
    Projects raw per-token features (1293-dim + 2 positional) to d_model.

    Separate projection heads per modality allow different learned representations
    for high-dimensional pre-trained embeddings vs low-dimensional scalars.

    Input layout per token:
        [0:512)       ReID embedding (L2-normalized)
        [512:1280)    SigLIP embedding
        [1280:1287)   BBox derived: cx, cy, w, h, aspect, area, cy_norm
        [1287:1292)   Scalars: jersey, entropy, jersey_conf, team, team_conf
        [1292:1293)   Detection score
        [1293:1294)   Intra-tracklet position (0..1)
        [1294:1295)   Normalized absolute frame number
    """

    def __init__(self, config: TemporalBinConfig):
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

        # Positional features (2-dim) get a small projection
        self.pos_proj = nn.Sequential(
            nn.Linear(2, 8),
            nn.GELU(),
        )

        # Final projection to d_model
        input_dim = c.projected_dim + 8  # modality projections + positional
        self.final_proj = nn.Linear(input_dim, c.d_model)

        # Boundaries for slicing
        self._reid_end = c.reid_dim
        self._siglip_end = c.reid_dim + c.siglip_dim
        self._bbox_end = self._siglip_end + c.bbox_feat_dim
        self._scalar_end = self._bbox_end + c.scalar_dim
        self._score_end = self._scalar_end + c.score_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, T, raw_dim+2) where raw_dim=1293, +2 for positional
        Returns: (B, T, d_model)
        """
        reid = tokens[..., :self._reid_end]
        siglip = tokens[..., self._reid_end:self._siglip_end]
        bbox = tokens[..., self._siglip_end:self._bbox_end]
        scalar = tokens[..., self._bbox_end:self._scalar_end]
        score = tokens[..., self._scalar_end:self._score_end]
        pos = tokens[..., self._score_end:]  # last 2 dims

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
    """
    Separate projector for the aggregate statistics token.

    The stats token has a different structure from the bin/boundary tokens:
    only 11 meaningful values in positions [0:11], rest is zeros.
    We project just those 11 stats to d_model directly.
    """

    def __init__(self, config: TemporalBinConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.stats_dim + 2, config.d_model),  # +2 for positional
            nn.LayerNorm(config.d_model),
            nn.GELU(),
        )
        self.stats_dim = config.stats_dim

    def forward(self, stats_token: torch.Tensor) -> torch.Tensor:
        """
        stats_token: (B, raw_dim+2) — only first stats_dim values meaningful
        Returns: (B, d_model)
        """
        # Extract the meaningful part: 11 stats + 2 positional
        meaningful = torch.cat([
            stats_token[..., :self.stats_dim],
            stats_token[..., -2:],  # positional features at the end
        ], dim=-1)
        return self.net(meaningful)


class TokenTypeEmbedding(nn.Module):
    """
    Learned embedding to distinguish token types within the sequence.

    Types: 0=CLS, 1=temporal_bin, 2=start_boundary, 3=end_boundary, 4=stats
    """

    def __init__(self, config: TemporalBinConfig):
        super().__init__()
        self.embedding = nn.Embedding(5, config.d_model)

    def forward(self, n_bins: int, device: torch.device) -> torch.Tensor:
        """
        Returns: (1, n_tokens, d_model) type embeddings for the full sequence.
        Order: [CLS, bin_0, ..., bin_7, start_boundary, end_boundary, stats]
        """
        types = torch.zeros(1 + n_bins + 3, dtype=torch.long, device=device)
        types[0] = 0                         # CLS
        types[1:1 + n_bins] = 1              # temporal bins
        types[1 + n_bins] = 2                # start boundary
        types[2 + n_bins] = 3                # end boundary
        types[3 + n_bins] = 4                # stats
        return self.embedding(types).unsqueeze(0)  # (1, n_tokens, d_model)


class PairClassifier(nn.Module):
    """
    Combines two CLS embeddings with extended pairwise features.

    Input: [cls_a, cls_b, |cls_a - cls_b|, cls_a * cls_b, pairwise_34dim]
    Output: scalar logit

    The pairwise features (34-dim) carry strong signal — XGBoost feature
    ablation showed 13 pairwise features alone get HOTA 90.54.  Including
    them here lets the transformer focus on learning what the aggregate
    statistics miss, while still having direct access to the proven signals.
    """

    def __init__(self, config: TemporalBinConfig):
        super().__init__()
        input_dim = config.d_model * 4 + config.pairwise_dim  # 96*4 + 34 = 418

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
        cls_a: torch.Tensor,
        cls_b: torch.Tensor,
        pairwise: torch.Tensor,
    ) -> torch.Tensor:
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


class TemporalBinTransformer(nn.Module):
    """
    Temporal Bin Transformer for tracklet merge classification.

    Architecture:
        Per tracklet:
            11 tokens (bins + boundaries + stats)
            → ModalityProjector / StatsProjector → d_model
            → + TokenTypeEmbedding
            → Prepend [CLS]
            → 3-layer TransformerEncoder (shared weights, pre-norm)
            → Extract [CLS] embedding

        Pair classification:
            [CLS_A, CLS_B, |A-B|, A*B, pairwise_34] → MLP → logit
    """

    def __init__(self, config: TemporalBinConfig = None):
        super().__init__()
        if config is None:
            config = TemporalBinConfig()
        self.config = config

        # Projectors
        self.modality_proj = ModalityProjector(config)
        self.stats_proj = StatsProjector(config)

        # Token type embedding
        self.type_embedding = TokenTypeEmbedding(config)

        # Learnable CLS token
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, config.d_model) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm for stable training
        )
        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(config.num_layers)]
        )
        self.encoder_norm = nn.LayerNorm(config.d_model)

        # Classification head
        self.classifier = PairClassifier(config)

    def encode_tracklet(self, bin_tokens: torch.Tensor) -> torch.Tensor:
        """
        Encode a single tracklet from its bin representation into a CLS embedding.

        Args:
            bin_tokens: (B, n_bins+3, raw_dim+2) — output of dataset

        Returns:
            (B, d_model) CLS embedding
        """
        B = bin_tokens.size(0)
        n_bins = self.config.n_temporal_bins

        # Split into regular tokens (bins + boundaries) and stats token
        regular_tokens = bin_tokens[:, :n_bins + 2]      # (B, n_bins+2, raw_dim+2)
        stats_token = bin_tokens[:, n_bins + 2]           # (B, raw_dim+2)

        # Project regular tokens through modality projector
        regular_proj = self.modality_proj(regular_tokens)  # (B, n_bins+2, d_model)

        # Project stats token through dedicated projector
        stats_proj = self.stats_proj(stats_token)           # (B, d_model)
        stats_proj = stats_proj.unsqueeze(1)                # (B, 1, d_model)

        # Concatenate: [bins, boundaries, stats]
        x = torch.cat([regular_proj, stats_proj], dim=1)    # (B, n_bins+3, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)              # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                      # (B, n_bins+4, d_model)

        # Add token type embedding
        type_emb = self.type_embedding(n_bins, x.device)    # (1, n_bins+4, d_model)
        x = x + type_emb

        # Transformer encoder with stochastic depth
        sd = self.config.stochastic_depth
        for i, layer in enumerate(self.encoder_layers):
            if self.training and sd > 0:
                drop_prob = sd * (i + 1) / len(self.encoder_layers)
                if torch.rand(1).item() < drop_prob:
                    continue
            x = layer(x)  # No padding mask needed — all tokens are valid

        x = self.encoder_norm(x)

        # Return CLS token (position 0)
        return x[:, 0]

    def forward(
        self,
        bins_a: torch.Tensor,
        bins_b: torch.Tensor,
        pairwise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for a batch of tracklet pairs.

        Args:
            bins_a:   (B, n_bins+3, raw_dim+2) temporal bin representation of tracklet A
            bins_b:   (B, n_bins+3, raw_dim+2) temporal bin representation of tracklet B
            pairwise: (B, 34) extended pairwise features

        Returns:
            (B,) logits for merge probability
        """
        cls_a = self.encode_tracklet(bins_a)
        cls_b = self.encode_tracklet(bins_b)
        return self.classifier(cls_a, cls_b, pairwise)
