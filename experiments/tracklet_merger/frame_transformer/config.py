"""
Configuration for the Frame-level Cross-Attention Pair Transformer.
"""

from dataclasses import dataclass


@dataclass
class FrameTransformerConfig:
    # ── Model architecture ──
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1

    # ── Input dimensions ──
    raw_dim: int = 1293          # 512 ReID + 768 SigLIP + 7 bbox + 5 scalar + 1 score
    pairwise_dim: int = 34       # 12 base + 2*11 tracklet stats

    # ── Frame handling ──
    max_frames_per_tracklet: int = 200   # hard cap per tracklet
    min_frames: int = 2                  # minimum frames after dropout

    # ── Augmentation (training only) ──
    frame_dropout: float = 0.2           # fraction of frames to drop
    augment_swap: bool = True            # randomly swap A/B
    noise_std_reid: float = 0.01         # Gaussian noise on ReID embeddings
    noise_std_siglip: float = 0.02       # Gaussian noise on SigLIP embeddings

    # ── Classifier head ──
    classifier_hidden: int = 128
