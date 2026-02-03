# import pandas as pd
# import argparse
# import json
# import numpy as np
# from pathlib import Path
# from xgboost import XGBClassifier
# from sklearn.metrics import roc_auc_score

# from config import MergeClassifierConfig
# from dataset import load_data, split_by_sequence
# from evaluate import evaluate_model, print_evaluation_report, evaluate_per_sequence


# def train_xgboost(config: MergeClassifierConfig):
#     # Load data
#     train_df, val_df, test_df, a_cols, b_cols, pw_cols = load_presplit_data(
#         config.training_data_dir
#     )

#     feature_cols = a_cols + b_cols + pw_cols

#     X_train = train_df[feature_cols].values
#     y_train = train_df["label"].values
#     X_val = val_df[feature_cols].values
#     y_val = val_df["label"].values
#     X_test = test_df[feature_cols].values
#     y_test = test_df["label"].values

#     # Class weight
#     n_pos = y_train.sum()
#     n_neg = len(y_train) - n_pos
#     scale_pos_weight = n_neg / max(n_pos, 1)

#     print(f"\nTraining XGBoost baseline...")
#     print(f"  Features: {len(feature_cols)}")
#     print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
#     print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    
#     model = XGBClassifier(
#         n_estimators=1000,
#         max_depth=6,
#         learning_rate=0.05,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         scale_pos_weight=scale_pos_weight,
#         eval_metric="auc",
#         early_stopping_rounds=30,
#         random_state=42,
#         verbosity=1,
#     )

#     model.fit(
#         X_train, y_train,
#         eval_set=[(X_val, y_val)],
#         verbose=20,
#     )

#     # Evaluate on test set
#     test_scores = model.predict_proba(X_test)[:, 1]
#     test_metrics = evaluate_model(y_test, test_scores)
#     print_evaluation_report(test_metrics, title="XGBoost Test Results")
    
#     # Feature importance
#     importances = model.feature_importances_
#     sorted_idx = np.argsort(importances)[::-1]
#     print("\nTop 20 Feature Importances:")
#     for i, idx in enumerate(sorted_idx[:20]):
#         print(f"  {i+1:2d}. {feature_cols[idx]:40s} {importances[idx]:.4f}")
    
#     # Save
#     save_dir = Path(config.model_save_dir)
#     save_dir.mkdir(parents=True, exist_ok=True)
#     model.save_model(str(save_dir / "xgboost_presplit.json"))
    
#     with open(save_dir / "xgboost_metrics.json", "w") as f:
#         json.dump(
#             {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float, np.floating))},
#             f, indent=2,
#         )
    
#     print(f"\nModel saved to {save_dir / 'xgboost_presplit.json'}")
#     return model, test_metrics


# def load_presplit_data(data_dir: Path):
#     """Load from pre-split train/valid/test directories."""
    
#     data_dir = Path(data_dir)
    
#     # Load each split
#     train_csvs = sorted((data_dir / "train").glob("*_pairs.csv"))
#     val_csvs = sorted((data_dir / "valid").glob("*_pairs.csv"))
#     test_csvs = sorted((data_dir / "test").glob("*_pairs.csv"))
    
#     print(f"Found {len(train_csvs)} train CSVs")
#     print(f"Found {len(val_csvs)} val CSVs")
#     print(f"Found {len(test_csvs)} test CSVs")
    
#     # Load and combine per split
#     train_dfs = [pd.read_csv(f) for f in train_csvs]
#     val_dfs = [pd.read_csv(f) for f in val_csvs]
#     test_dfs = [pd.read_csv(f) for f in test_csvs]
    
#     train_df = pd.concat(train_dfs, ignore_index=True)
#     val_df = pd.concat(val_dfs, ignore_index=True)
#     test_df = pd.concat(test_dfs, ignore_index=True)
    
#     print(f"\nTrain: {len(train_df)} pairs")
#     print(f"Val:   {len(val_df)} pairs")
#     print(f"Test:  {len(test_df)} pairs")
#     print(f"Total: {len(train_df) + len(val_df) + len(test_df)} pairs")
    
#     # Detect columns
#     a_columns = sorted([c for c in train_df.columns if c.startswith("A_")])
#     b_columns = sorted([c for c in train_df.columns if c.startswith("B_")])
#     pairwise_columns = sorted([c for c in train_df.columns if c.startswith("pairwise_")])
    
#     # Handle NaN
#     feature_cols = a_columns + b_columns + pairwise_columns
#     for df in [train_df, val_df, test_df]:
#         df[feature_cols] = df[feature_cols].fillna(0)
    
#     return train_df, val_df, test_df, a_columns, b_columns, pairwise_columns


# if __name__ == "__main__":
#     data_dir = r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\training_data"

#     save_dir = r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\weights\xgboost"
#     config = MergeClassifierConfig(
#         training_data_dir=Path(data_dir),
#         model_save_dir=Path(save_dir),
#     )

#     train_xgboost(config)


import pandas as pd
import json
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier

from config import MergeClassifierConfig
from evaluate import evaluate_model, print_evaluation_report


def train_xgboost(config: MergeClassifierConfig):
    # Load data
    train_df, val_df, test_df, a_cols, b_cols, pw_cols = load_presplit_data(
        config.training_data_dir
    )

    feature_cols = a_cols + b_cols + pw_cols

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    # Class weight
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    print(f"\nTraining XGBoost baseline...")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    
    model = XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        early_stopping_rounds=30,
        random_state=42,
        verbosity=1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=20,
    )

    # Evaluate on test set
    test_scores = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_model(y_test, test_scores)
    print_evaluation_report(test_metrics, title="XGBoost Test Results")
    
    # Feature importance
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nTop 20 Feature Importances:")
    for i, idx in enumerate(sorted_idx[:20]):
        print(f"  {i+1:2d}. {feature_cols[idx]:40s} {importances[idx]:.4f}")
    
    # Save
    save_dir = Path(config.model_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(save_dir / "xgboost_presplit.json"))
    
    with open(save_dir / "xgboost_metrics.json", "w") as f:
        json.dump(
            {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float, np.floating))},
            f, indent=2,
        )
    
    print(f"\nModel saved to {save_dir / 'xgboost_presplit.json'}")
    return model, test_metrics


def load_presplit_data(data_dir: Path):
    """Load pre-combined train/valid/test CSVs from generate_train_data.py."""
    
    data_dir = Path(data_dir)
    
    train_path = data_dir / "combined_train_data.csv"
    val_path   = data_dir / "combined_valid_data.csv"
    test_path  = data_dir / "combined_test_data.csv"
    
    # Verify all three exist before loading
    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")
    
    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df  = pd.read_csv(test_path)
    
    print(f"Train: {len(train_df)} pairs")
    print(f"Val:   {len(val_df)} pairs")
    print(f"Test:  {len(test_df)} pairs")
    print(f"Total: {len(train_df) + len(val_df) + len(test_df)} pairs")
    
    # Detect columns
    a_columns = sorted([c for c in train_df.columns if c.startswith("A_")])
    b_columns = sorted([c for c in train_df.columns if c.startswith("B_")])
    pairwise_columns = sorted([c for c in train_df.columns if c.startswith("pairwise_")])
    
    # Handle NaN
    feature_cols = a_columns + b_columns + pairwise_columns
    for df in [train_df, val_df, test_df]:
        df[feature_cols] = df[feature_cols].fillna(0)
    
    return train_df, val_df, test_df, a_columns, b_columns, pairwise_columns


if __name__ == "__main__":
    data_dir = r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\training_data"

    save_dir = r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\weights\xgboost"
    config = MergeClassifierConfig(
        training_data_dir=Path(data_dir),
        model_save_dir=Path(save_dir),
    )

    train_xgboost(config)