from dataclasses import dataclass

from ..utils import PAIRWISE_DIM


@dataclass
class TransformerMergerConfig:
    """Configuration for the Siamese CLS Transformer tracklet merger."""

    # Input dimensions
    reid_dim: int = 512
    siglip_dim: int = 768
    bbox_feat_dim: int = 7       # cx, cy, w, h, aspect, area, cy_norm
    scalar_dim: int = 5          # jersey, entropy, jersey_conf, team, team_conf

    # Projection
    projection_dropout: float = 0.1

    # Transformer
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 128
    attention_dropout: float = 0.1
    transformer_dropout: float = 0.2
    stochastic_depth: float = 0.1

    # Sequence handling
    t_max: int = 200             # max frames per tracklet (subsample if longer)

    # Classification head
    pairwise_dim: int = PAIRWISE_DIM
    use_pairwise_features: bool = True  # if False, classification head ignores pairwise features
    classifier_dropout: float = 0.3

    # Training
    lr: float = 5e-4
    weight_decay: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 15
    label_smoothing: float = 0.05
    frame_dropout: float = 0.15  # fraction of frames randomly dropped during training
    augment_swap: bool = True    # A/B swap augmentation

    # Data generation
    max_frame_value: float = 750.0    # sequences are exactly 750 frames long
    negative_ratio: float = None       # no downsampling for eval
    min_tracklet_len: int = 0           # canonical: no length filtering

    @property
    def raw_token_dim(self) -> int:
        return (self.reid_dim + self.siglip_dim +
                self.bbox_feat_dim + self.scalar_dim)
