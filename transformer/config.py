from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class TransformerMergerConfig:
    """Configuration for the Siamese CLS Transformer tracklet merger."""

    # ---- Input dimensions ----
    reid_dim: int = 512
    siglip_dim: int = 768
    bbox_feat_dim: int = 7       # cx, cy, w, h, aspect, area, cy_norm
    scalar_dim: int = 5          # jersey, entropy, jersey_conf, team, team_conf
    score_dim: int = 1

    # ---- Projection dimensions ----
    reid_proj_dim: int = 64
    siglip_proj_dim: int = 64
    bbox_proj_dim: int = 16
    scalar_proj_dim: int = 16
    score_proj_dim: int = 8

    # ---- Transformer ----
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 256
    transformer_dropout: float = 0.2
    stochastic_depth: float = 0.1

    # ---- Sequence handling ----
    t_max: int = 200             # max frames per tracklet (subsample if longer)

    # ---- Classification head ----
    pairwise_dim: int = 12
    classifier_dropout: float = 0.3

    # ---- Training ----
    lr: float = 5e-4
    weight_decay: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 15
    label_smoothing: float = 0.05
    frame_dropout: float = 0.15  # fraction of frames randomly dropped during training
    augment_swap: bool = True    # A/B swap augmentation

    # ---- Data generation ----
    max_frame_value: float = 750.0    # sequences are exactly 750 frames long
    negative_ratio: float = 2.0
    min_tracklet_len: int = 5

    @property
    def projected_dim(self) -> int:
        return (self.reid_proj_dim + self.siglip_proj_dim +
                self.bbox_proj_dim + self.scalar_proj_dim + self.score_proj_dim)

    @property
    def raw_token_dim(self) -> int:
        return (self.reid_dim + self.siglip_dim +
                self.bbox_feat_dim + self.scalar_dim + self.score_dim)
