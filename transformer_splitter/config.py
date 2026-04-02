from dataclasses import dataclass


@dataclass
class SplitTransformerConfig:
    """Configuration for the per-frame split-point transformer."""

    # ---- Input dimensions (must match transformer/modules.py) ----
    reid_dim: int = 512
    siglip_dim: int = 768
    bbox_feat_dim: int = 7
    scalar_dim: int = 5          # jersey, entropy, jersey_conf, team, team_conf
    score_dim: int = 1

    # ---- Projection dimensions (same as merger → reuse TokenProjector) ----
    reid_proj_dim: int = 64
    siglip_proj_dim: int = 64
    bbox_proj_dim: int = 16
    scalar_proj_dim: int = 16
    score_proj_dim: int = 8

    # ---- Transformer encoder ----
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 256
    transformer_dropout: float = 0.2
    stochastic_depth: float = 0.1

    # ---- Sequence handling ----
    t_max: int = 400             # concatenated tracklets can be long

    # ---- Per-frame classification head ----
    head_hidden: int = 64
    head_dropout: float = 0.3

    # ---- Training ----
    lr: float = 3e-4
    weight_decay: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 15
    label_smoothing: float = 0.05
    frame_dropout: float = 0.10  # lower than merger — context near splits matters
    pos_weight: float = 40.0     # heavy recall bias: oversplit > undersplit

    # ---- Data generation ----
    max_frame_value: float = 750.0
    min_crop_len: int = 20       # minimum frames per half when generating synthetic samples
    split_label_radius: int = 3  # +/- frames around true split labeled as positive
    max_temporal_gap: int = 30   # max random gap inserted at synthetic junction
    positive_negative_ratio: float = 3.0  # target pos:neg ratio

    @property
    def projected_dim(self) -> int:
        return (self.reid_proj_dim + self.siglip_proj_dim +
                self.bbox_proj_dim + self.scalar_proj_dim + self.score_proj_dim)

    @property
    def raw_token_dim(self) -> int:
        return (self.reid_dim + self.siglip_dim +
                self.bbox_feat_dim + self.scalar_dim + self.score_dim)
