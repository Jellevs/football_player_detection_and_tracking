from dataclasses import dataclass


@dataclass
class HybridConfig:
    """Configuration for the Hybrid (self-attn -> cross-attn) Transformer."""

    # Input dimensions
    reid_dim: int = 512
    siglip_dim: int = 768
    bbox_feat_dim: int = 7       # cx, cy, w, h, aspect, area, cy_norm
    scalar_dim: int = 5          # jersey, entropy, jersey_conf, team, team_conf

    # Projection
    d_model: int = 64
    projection_dropout: float = 0.1

    # Self-attention encoder
    nhead: int = 4
    num_self_layers: int = 2
    dim_feedforward: int = 128   # 2x d_model
    self_dropout: float = 0.1
    stochastic_depth: float = 0.1

    # Cross-attention
    num_cross_layers: int = 2
    cross_dropout: float = 0.1

    # Sequence handling
    t_max: int = 200

    # Classification head
    pairwise_dim: int = 13       # 13 dim pairwise features (with SigLIP and team conflict)
    classifier_hidden: int = 64
    classifier_dropout: float = 0.3

    # Training
    lr: float = 5e-4
    weight_decay: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 15
    label_smoothing: float = 0.05
    frame_dropout: float = 0.15
    augment_swap: bool = True

    # Data
    max_frame_value: float = 750.0

    @property
    def raw_token_dim(self) -> int:
        return self.reid_dim + self.siglip_dim + self.bbox_feat_dim + self.scalar_dim
