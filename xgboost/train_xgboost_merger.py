"""
train_xgboost_merger.py

Trains an XGBoost binary classifier on the pairwise tracklet data produced by
generate_merger_training_data.py.

The model predicts the probability that two tracklets should be merged.
This probability is converted to a distance (1 - p) and fed into the same
hierarchical clustering loop used by SoccerAwareMerger.

Usage:
    python train_xgboost_merger.py

Outputs (written to SAVE_DIR):
    xgboost_merger.json       — trained XGBoost model
    xgboost_merger_meta.json  — threshold + column order needed at inference
    xgboost_merger_metrics.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Paths — adjust to your environment
# ---------------------------------------------------------------------------
OUTPUT_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\weights\xgboost_5_neg_ratio")
SAVE_DIR    = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\weights\xgboost_5_neg_ratio")
DATA_DIR    = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\train_data\xgboost_5_neg_ratio")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_splits(data_dir: Path):
    """
    Load the three combined CSVs produced by generate_merger_training_data.py.
    Returns (train_df, val_df, test_df, feature_cols).

    Feature columns are all A_*, B_*, pairwise_* columns.
    The reid / siglip embedding columns (reid_mean_*, reid_std_*, etc.) are
    included directly — no PCA.
    """
    train_path = data_dir / "combined_train_data.csv"
    val_path   = data_dir / "combined_valid_data.csv"
    test_path  = data_dir / "combined_test_data.csv"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing: {p}\n"
                f"Run generate_merger_training_data.py for each split first."
            )

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df  = pd.read_csv(test_path)

    print(f"Loaded  train : {len(train_df):>7,} rows")
    print(f"        valid : {len(val_df):>7,} rows")
    print(f"        test  : {len(test_df):>7,} rows")

    # Identify feature columns — everything except label, sequence, track_id
    non_feature = {"label", "sequence", "A_track_id", "B_track_id"}
    all_cols     = train_df.columns.tolist()
    feature_cols = [c for c in all_cols
                    if c not in non_feature
                    and (c.startswith("A_") or c.startswith("B_") or c.startswith("pairwise_"))]

    print(f"        feature cols : {len(feature_cols)}")
    print(f"          A_*        : {sum(1 for c in feature_cols if c.startswith('A_'))}")
    print(f"          B_*        : {sum(1 for c in feature_cols if c.startswith('B_'))}")
    print(f"          pairwise_* : {sum(1 for c in feature_cols if c.startswith('pairwise_'))}")

    # Fill NaN (e.g. jersey_mode for tracklets with no legible jersey)
    for df in [train_df, val_df, test_df]:
        df[feature_cols] = df[feature_cols].fillna(0)

    return train_df, val_df, test_df, feature_cols


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def find_best_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Find the probability threshold that maximises F1 on labels/scores."""
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(labels: np.ndarray, scores: np.ndarray, threshold: float, title: str = ""):
    preds = (scores >= threshold).astype(int)
    cm    = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "threshold"      : threshold,
        "auc_roc"        : roc_auc_score(labels, scores),
        "avg_precision"  : average_precision_score(labels, scores),
        "f1"             : f1_score(labels, preds, zero_division=0),
        "precision"      : precision_score(labels, preds, zero_division=0),
        "recall"         : recall_score(labels, preds, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

    if title:
        bar = "=" * 52
        print(f"\n{bar}")
        print(f"  {title}")
        print(bar)
        print(f"  AUC-ROC        : {metrics['auc_roc']:.4f}")
        print(f"  Avg Precision  : {metrics['avg_precision']:.4f}")
        print(f"  F1             : {metrics['f1']:.4f}")
        print(f"  Precision      : {metrics['precision']:.4f}")
        print(f"  Recall         : {metrics['recall']:.4f}")
        print(f"  Threshold      : {metrics['threshold']:.3f}")
        print(f"  TN / FP        : {tn:5d} / {fp:5d}")
        print(f"  FN / TP        : {fn:5d} / {tp:5d}")
        print(bar)

    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    train_df, val_df, test_df, feature_cols = load_splits(DATA_DIR)

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_val   = val_df[feature_cols].values
    y_val   = val_df["label"].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df["label"].values

    # Class weight
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"\nClass balance: {n_pos} pos / {n_neg} neg  (scale_pos_weight={scale_pos_weight:.2f})")

    # Model
    print("\nTraining XGBoost...")
    model = XGBClassifier(
        n_estimators         = 1000,
        max_depth            = 6,
        learning_rate        = 0.05,
        subsample            = 0.8,
        colsample_bytree     = 0.8,
        min_child_weight     = 5,
        gamma                = 1,
        scale_pos_weight     = scale_pos_weight,
        eval_metric          = "auc",
        early_stopping_rounds= 30,
        random_state         = 42,
        n_jobs               = -1,
        verbosity            = 1,
        device               = "cpu",         # change to "cuda" if GPU available
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=20,
    )

    # Save per-round training log
    results = model.evals_result()
    train_auc = results["validation_0"]["auc"]
    val_auc   = results["validation_1"]["auc"]
    log_path  = SAVE_DIR / "training_log.csv"
    with open(log_path, "w") as f:
        f.write("round,train_auc,val_auc\n")
        for r, (ta, va) in enumerate(zip(train_auc, val_auc)):
            f.write(f"{r},{ta},{va}\n")
    print(f"Training log saved → {log_path}  ({len(train_auc)} rounds)")

    # Find optimal threshold on validation set
    val_scores = model.predict_proba(X_val)[:, 1]
    threshold  = find_best_threshold(y_val, val_scores)
    print(f"\nOptimal threshold (from val set): {threshold:.3f}")
    evaluate(y_val, val_scores, threshold, title="Validation Results")

    # Evaluate on held-out test set
    test_scores  = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_scores, threshold, title="Test Results")

    # Feature importance — top 30
    importances = model.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:30]
    print("\nTop 30 Feature Importances:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {feature_cols[idx]:55s} {importances[idx]:.4f}")

    # Save model
    model_path = SAVE_DIR / "xgboost_merger.json"
    model.save_model(str(model_path))
    print(f"\nModel saved  → {model_path}")

    # Save metadata needed at inference time
    meta = {
        "feature_cols"   : feature_cols,
        "threshold"      : threshold,
        "reid_dim"       : 512,
        "siglip_dim"     : 768,
    }
    meta_path = SAVE_DIR / "xgboost_merger_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Meta saved   → {meta_path}")

    # Save metrics
    metrics_path = SAVE_DIR / "xgboost_merger_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({k: v for k, v in test_metrics.items() if isinstance(v, (int, float))}, f, indent=2)
    print(f"Metrics saved → {metrics_path}")

    return model, feature_cols, threshold


if __name__ == "__main__":
    train()