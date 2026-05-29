"""
Configuration for the Late Cross-Attention Transformer tracklet merger.

Architecture (Plan Section 4, B3)
=================================
This model is a *late cross-attention bi-encoder*: the sweet spot between the
pure bi-encoder (temporal bin transformer) and the expensive full frame-level
cross-encoder (frame_transformer).

  1. Encode each tracklet **once** into its compact token set (reuse the 11
     temporal-bin/boundary/stats tokens from the temporal bin transformer).
  2. For each candidate pair, run **2 cross-attention layers** over the
     concatenated ~22 tokens + a pair-CLS, so A's tokens attend to B's tokens
     and vice versa.

This adds genuine cross-tracklet interaction — the model can match *the one
clear jersey frame in A to the one clear frame in B* — which XGBoost
fundamentally cannot do.  Because the per-pair sequence is ~23 tokens, the
O(N^2) pair cost is trivial.

Key improvements over the temporal bin transformer:
  - Late cross-attention between tracklet token sets (B3)
  - Residual classifier head: with CLS terms zeroed, reproduces a
    logistic-regression-on-pairwise baseline (B1)
  - Privileged cosine similarity signal as direct input (B4)
  - HOTA-based model selection instead of AUC (C1)
  - Hard-negative mining in data generation (A2)
  - Realistic synthetic positives with ReID-drift cuts (A1)
  - Supervised contrastive auxiliary loss on CLS embeddings (C2)
  - Per-sequence affinity normalization at inference (C3)
  - Weight EMA for smoother, better-calibrated inference (C4)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NewTransformerConfig:
    """Configuration for the Late Cross-Attention Transformer merger."""

    # ---- Input dimensions (per-frame raw features) ----
    reid_dim: int = 512
    siglip_dim: int = 768
    bbox_feat_dim: int = 7       # cx, cy, w, h, aspect, area, cy_norm
    scalar_dim: int = 5          # jersey, entropy, jersey_conf, team, team_conf
    score_dim: int = 1

    # ---- Projection dimensions (modality-specific) ----
    reid_proj_dim: int = 48
    siglip_proj_dim: int = 48
    bbox_proj_dim: int = 16
    scalar_proj_dim: int = 16
    score_proj_dim: int = 8

    # ---- Temporal binning (shared with temporal bin transformer) ----
    n_temporal_bins: int = 8
    boundary_k: int = 5

    # ---- Self-attention encoder (per-tracklet) ----
    d_model: int = 96
    nhead: int = 4
    num_self_attn_layers: int = 3
    dim_feedforward: int = 192
    transformer_dropout: float = 0.15
    stochastic_depth: float = 0.1

    # ---- Cross-attention layers (B3 — the key architectural addition) ----
    num_cross_attn_layers: int = 2
    cross_attn_nhead: int = 4
    cross_attn_dim_feedforward: int = 192
    cross_attn_dropout: float = 0.15

    # ---- Classification head (B1 — residual structure) ----
    # Extended pairwise: 12 base + 11 stats_A + 11 stats_B = 34
    pairwise_dim: int = 34
    classifier_dropout: float = 0.25

    # The classifier is structured so that with CLS-derived terms zeroed,
    # it reproduces a logistic regression on the pairwise features alone.
    # This means the model *starts* at the linear-baseline operating point
    # and the transformer can only add value.
    use_residual_head: bool = True

    # ---- Training ----
    lr: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 5e-3
    batch_size: int = 64
    max_epochs: int = 150
    patience: int = 25
    warmup_epochs: int = 5
    label_smoothing: float = 0.05

    # ---- Focal loss ----
    focal_gamma_pos: float = 1.0
    focal_gamma_neg: float = 2.0

    # ---- Auxiliary losses (C2) ----
    # Supervised contrastive loss on CLS embeddings within a sequence
    contrastive_weight: float = 0.1
    contrastive_temperature: float = 0.07

    # ---- HOTA-based selection (C1) ----
    # Evaluate clustering proxy every N epochs (set to 0 to disable)
    hota_eval_interval: int = 5
    # Use V-measure as a fast clustering proxy instead of full HOTA
    use_vmeasure_proxy: bool = True

    # ---- Weight EMA (C4) ----
    use_ema: bool = True
    ema_decay: float = 0.999

    # ---- Augmentation ----
    frame_dropout: float = 0.15
    augment_swap: bool = True
    noise_std_reid: float = 0.01
    noise_std_siglip: float = 0.02
    modality_mask_prob: float = 0.10
    temporal_crop_prob: float = 0.15

    # ---- Data generation ----
    max_frame_value: float = 750.0
    negative_ratio: float = 4.0
    min_tracklet_len: int = 5

    # Purity thresholds (A5 — tighter for positives)
    purity_threshold_positive: float = 0.90   # clean merges only
    purity_threshold_negative: float = 0.80   # route impure to hard-neg pool

    # Synthetic positive generation (A1 — realistic)
    min_split_len: int = 20
    n_synthetic_splits: int = 3
    synthetic_min_fragment: int = 8
    # Cut at maximum ReID drift within tracklet
    use_reid_drift_cuts: bool = True
    # Inject temporal gap at cut point
    inject_temporal_gap: bool = True
    temporal_gap_range: tuple = (5, 60)  # min/max frames to delete

    # Hard-negative mining (A2)
    hard_negative_ratio: float = 0.5   # fraction of negatives that should be hard
    hard_neg_min_cosine: float = 0.3   # minimum ReID cosine sim to qualify as hard
    hard_neg_max_temporal_gap: int = 200  # max temporal gap for hard negatives

    # ---- Per-sequence normalization (C3) ----
    use_affinity_normalization: bool = True
    normalization_method: str = "rank"  # "rank", "zscore", or "none"

    # ---- Aggregate stats dimension (per-tracklet) ----
    stats_dim: int = 11

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
