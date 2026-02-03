from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class MergeClassifierConfig:
    # Architecture
    fragment_input_dim: int = 41
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [64, 64, 32])
    classifier_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    pairwise_input_dim: int = 12
    encoder_dropout: float = 0.3
    classifier_dropout: float = 0.4

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 200
    early_stopping_patience: int = 20
    lr_patience: int = 10
    lr_factor: float = 0.5

    # Data
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    augment_swap: bool = True

    # Paths
    model_save_dir: Path = field(default_factory=lambda: Path("weights/merge_classifier"))
    training_data_dir: Path = field(default_factory=lambda: Path("output/training_data"))
