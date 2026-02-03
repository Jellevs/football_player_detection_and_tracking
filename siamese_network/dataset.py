import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import GroupShuffleSplit


class FeatureNormalizer:
    """Per-column standardization (mean=0, std=1) fit on training data only."""

    def __init__(self):
        self.mean = None
        self.std = None
        self.is_fit = False

    def fit(self, df: pd.DataFrame):
        self.mean = df.mean()
        self.std = df.std().replace(0, 1)  # avoid division by zero
        self.is_fit = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.is_fit, "Call fit() first"
        return (df - self.mean) / self.std

    def state_dict(self) -> Dict:
        return {"mean": self.mean.to_dict(), "std": self.std.to_dict()}

    def load_state_dict(self, state: Dict):
        self.mean = pd.Series(state["mean"])
        self.std = pd.Series(state["std"])
        self.is_fit = True


class TrackletPairDataset(Dataset):
    """
    PyTorch Dataset for tracklet pair CSVs.

    Loads per-sequence CSVs from a directory or a single combined CSV.
    Supports A<->B swap augmentation for symmetry.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        a_columns: List[str],
        b_columns: List[str],
        pairwise_columns: List[str],
        normalizer: Optional[FeatureNormalizer] = None,
        augment_swap: bool = True,
    ):
        self.augment_swap = augment_swap
        self.labels = torch.tensor(dataframe["label"].values, dtype=torch.float32)

        # Normalize feature columns
        feature_df = dataframe[a_columns + b_columns + pairwise_columns]
        if normalizer is not None and normalizer.is_fit:
            feature_df = normalizer.transform(feature_df)

        self.a_features = torch.tensor(
            feature_df[a_columns].values, dtype=torch.float32
        )
        self.b_features = torch.tensor(
            feature_df[b_columns].values, dtype=torch.float32
        )
        self.pairwise_features = torch.tensor(
            feature_df[pairwise_columns].values, dtype=torch.float32
        )

        # Store raw spatial columns for recomputing endpoint_distance on swap
        self.a_start_x = dataframe["A_start_x"].values
        self.a_start_y = dataframe["A_start_y"].values
        self.b_end_x = dataframe["B_end_x"].values
        self.b_end_y = dataframe["B_end_y"].values

        # Find pairwise column indices for swap adjustments
        self.pw_endpoint_idx = pairwise_columns.index("pairwise_endpoint_distance")
        self.pw_bbox_ratio_idx = pairwise_columns.index("pairwise_bbox_height_ratio")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        a = self.a_features[idx]
        b = self.b_features[idx]
        pw = self.pairwise_features[idx].clone()
        label = self.labels[idx]

        if self.augment_swap and self.training_mode and torch.rand(1).item() > 0.5:
            # Swap A and B
            a, b = b, a
            # Invert bbox_height_ratio (A/B -> B/A)
            if pw[self.pw_bbox_ratio_idx] != 0:
                pw[self.pw_bbox_ratio_idx] = 1.0 / pw[self.pw_bbox_ratio_idx]
            # endpoint_distance: use spatial_distance as proxy (symmetric)
            # since exact recomputation requires unnormalized values
            # spatial_distance is already in pairwise and is symmetric

        return a, b, pw, label

    @property
    def training_mode(self):
        """Only augment during training, not validation/test."""
        return self._training_mode

    @training_mode.setter
    def training_mode(self, value: bool):
        self._training_mode = value

    _training_mode: bool = True


def load_data(
    data_dir: Path,
) -> Tuple[pd.DataFrame, List[str], List[str], List[str], np.ndarray]:
    """
    Load training data from per-sequence CSVs or combined CSV.

    Returns:
        (dataframe, a_columns, b_columns, pairwise_columns, sequence_labels)
    """
    data_dir = Path(data_dir)

    # Try per-sequence CSVs first
    csv_files = sorted(data_dir.glob("*_pairs.csv"))
    if csv_files:
        dfs = []
        sequences = []
        for csv_path in csv_files:
            seq_name = csv_path.stem.replace("_pairs", "")
            df = pd.read_csv(csv_path)
            df["sequence"] = seq_name
            dfs.append(df)
            sequences.append(seq_name)
        combined = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(csv_files)} per-sequence CSVs: {len(combined)} total pairs")
    else:
        # Fall back to combined CSV
        combined_path = data_dir / "combined_training_data.csv"
        assert combined_path.exists(), f"No CSVs found in {data_dir}"
        combined = pd.read_csv(combined_path)
        if "sequence" not in combined.columns:
            combined["sequence"] = "unknown"
        print(f"Loaded combined CSV: {len(combined)} total pairs")

    # Detect column groups
    a_columns = sorted([c for c in combined.columns if c.startswith("A_")])
    b_columns = sorted([c for c in combined.columns if c.startswith("B_")])
    pairwise_columns = sorted([c for c in combined.columns if c.startswith("pairwise_")])

    # Handle NaN values
    feature_cols = a_columns + b_columns + pairwise_columns
    nan_count = combined[feature_cols].isna().sum().sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values found, filling with 0")
        combined[feature_cols] = combined[feature_cols].fillna(0)

    sequence_labels = combined["sequence"].values

    print(f"Features: {len(a_columns)} A + {len(b_columns)} B + {len(pairwise_columns)} pairwise")
    print(f"Labels: {int(combined['label'].sum())} positive, {int((combined['label'] == 0).sum())} negative")
    print(f"Sequences: {combined['sequence'].nunique()}")

    return combined, a_columns, b_columns, pairwise_columns, sequence_labels


def split_by_sequence(
    df: pd.DataFrame,
    sequence_labels: np.ndarray,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by sequence to avoid leakage."""

    unique_sequences = np.unique(sequence_labels)
    n_seq = len(unique_sequences)

    if n_seq < 3:
        print(f"Warning: only {n_seq} sequences, falling back to random split")
        n = len(df)
        indices = np.random.RandomState(random_state).permutation(n)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        test_idx = indices[:n_test]
        val_idx = indices[n_test : n_test + n_val]
        train_idx = indices[n_test + n_val :]
        return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]

    # First split: separate test set
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
    trainval_idx, test_idx = next(gss_test.split(df, groups=sequence_labels))

    # Second split: separate val from trainval
    trainval_sequences = sequence_labels[trainval_idx]
    val_ratio_adjusted = val_ratio / (1 - test_ratio)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio_adjusted, random_state=random_state)
    train_idx_inner, val_idx_inner = next(
        gss_val.split(df.iloc[trainval_idx], groups=trainval_sequences)
    )
    train_idx = trainval_idx[train_idx_inner]
    val_idx = trainval_idx[val_idx_inner]

    train_seqs = set(sequence_labels[train_idx])
    val_seqs = set(sequence_labels[val_idx])
    test_seqs = set(sequence_labels[test_idx])

    print(f"Split: train={len(train_idx)} ({len(train_seqs)} seq), "
          f"val={len(val_idx)} ({len(val_seqs)} seq), "
          f"test={len(test_idx)} ({len(test_seqs)} seq)")

    # Verify no sequence overlap
    assert not (train_seqs & val_seqs), "Train/val sequence overlap!"
    assert not (train_seqs & test_seqs), "Train/test sequence overlap!"
    assert not (val_seqs & test_seqs), "Val/test sequence overlap!"

    return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]
