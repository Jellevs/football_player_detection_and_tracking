"""
Configuration for the Temporal Bin Transformer tracklet merger.

Design rationale
================
The original SiameseCLS transformer processes up to 200 raw frames (1293-dim)
per tracklet through a full transformer encoder.  Feature ablation on XGBoost
showed that 13 pairwise features alone nearly match the full 47-feature model
(HOTA 90.54 vs 90.71), suggesting that per-frame detail adds marginal signal
when naively aggregated via CLS token.

This model instead creates a *compact* fixed-size representation per tracklet:
  - 8 temporal bins  (average features within each time slice)
  - 2 boundary tokens (first/last K frames, where the merge signal is strongest)
  - 1 aggregate statistics token (duration, coverage, consistency, etc.)
  - 1 learnable CLS token
Total: 12 tokens per tracklet, each projected to d_model=96.

This is ~17x shorter than t_max=200, making the self-attention layers far
cheaper and the model far more data-efficient (~200K parameters vs ~450K).

Key differences from the original transformer:
  1. Temporal binning instead of raw frame sequences
  2. Post-split training data (matching inference distribution, like XGBoost)
  3. Synthetic positive pair generation (splitting long tracklets)
  4. Extended pairwise features (34-dim = 12 base + 2*11 per-tracklet stats)
  5. Asymmetric focal loss instead of BCE
  6. Cosine annealing with linear warmup instead of ReduceLROnPlateau
  7. Modality masking augmentation (zero out one feature group during training)
"""

from dataclasses import dataclass


@dataclass
class TemporalBinConfig:
    """Configuration for the Temporal Bin Transformer tracklet merger."""

    # ---- Input dimensions (per-frame raw features) ----
    reid_dim: int = 512
    siglip_dim: int = 768
    bbox_feat_dim: int = 7       # cx, cy, w, h, aspect, area, cy_norm
    scalar_dim: int = 5          # jersey, entropy, jersey_conf, team, team_conf
    score_dim: int = 1

    # ---- Projection dimensions (modality-specific) ----
    reid_proj_dim: int = 48      # reduced from 64 — less overfitting risk
    siglip_proj_dim: int = 48    # reduced from 64
    bbox_proj_dim: int = 16
    scalar_proj_dim: int = 16
    score_proj_dim: int = 8

    # ---- Temporal binning ----
    n_temporal_bins: int = 8     # divide tracklet into this many equal time slices
    boundary_k: int = 5          # average first/last K frames for boundary tokens

    # ---- Transformer ----
    d_model: int = 96            # reduced from 128 — more data-efficient
    nhead: int = 4               # 96/4 = 24-dim per head — sufficient
    num_layers: int = 3          # reduced from 4 — fewer parameters
    dim_feedforward: int = 192   # 2x d_model (reduced from 256)
    transformer_dropout: float = 0.15
    stochastic_depth: float = 0.1

    # ---- Classification head ----
    # Extended pairwise: 12 base + 11 stats_A + 11 stats_B = 34
    pairwise_dim: int = 34
    classifier_dropout: float = 0.25   # slightly less aggressive than 0.3

    # ---- Training ----
    lr: float = 3e-4
    min_lr: float = 1e-6              # floor for cosine annealing
    weight_decay: float = 5e-3        # stronger than 1e-3 — small model benefits
    batch_size: int = 64              # larger batch — compact tokens fit easily
    max_epochs: int = 120
    patience: int = 20                # more patience with cosine schedule
    warmup_epochs: int = 5            # linear warmup before cosine decay
    label_smoothing: float = 0.05

    # ---- Focal loss ----
    focal_gamma_pos: float = 1.0      # less aggressive down-weighting of easy positives
    focal_gamma_neg: float = 2.0      # strongly down-weight easy negatives

    # ---- Augmentation ----
    frame_dropout: float = 0.15       # fraction of frames dropped before binning
    augment_swap: bool = True         # A/B swap augmentation
    noise_std_reid: float = 0.01      # Gaussian noise on ReID embeddings
    noise_std_siglip: float = 0.02    # Gaussian noise on SigLIP embeddings
    modality_mask_prob: float = 0.10  # probability of zeroing one feature group
    temporal_crop_prob: float = 0.15  # probability of removing 10-30% from one end

    # ---- Data generation ----
    max_frame_value: float = 750.0    # SoccerNet sequences are 750 frames
    negative_ratio: float = 4.0       # neg/pos ratio (with synthetic pos, need more neg)
    min_tracklet_len: int = 5
    purity_threshold: float = 0.80

    # Synthetic positive generation
    min_split_len: int = 20           # minimum tracklet length to generate a synthetic split
    n_synthetic_splits: int = 3       # number of random split points per eligible tracklet
    synthetic_min_fragment: int = 8   # each fragment must have at least this many frames

    # ---- Aggregate stats dimension (per-tracklet) ----
    stats_dim: int = 11               # matches _tracklet_stats output

    @property
    def projected_dim(self) -> int:
        return (self.reid_proj_dim + self.siglip_proj_dim +
                self.bbox_proj_dim + self.scalar_proj_dim + self.score_proj_dim)

    @property
    def raw_token_dim(self) -> int:
        return (self.reid_dim + self.siglip_dim +
                self.bbox_feat_dim + self.scalar_dim + self.score_dim)

    @property
    def n_tokens_per_tracklet(self) -> int:
        """CLS + temporal bins + 2 boundary + 1 stats."""
        return 1 + self.n_temporal_bins + 2 + 1
