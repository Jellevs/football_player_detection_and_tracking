"""
train_xgboost_merger_tuned.py

Optimized training pipeline for the XGBoost tracklet merger.

Improvements over train_xgboost_merger.py:
  1. Optuna Bayesian hyperparameter search with MedianPruner.
  2. GroupKFold(n_splits=5) on the `sequence` column inside the objective,
     so hyperparameter selection does not leak across sequences and does
     not see the held-out validation set.
  3. Multi-seed ensemble of the best configuration — final predictions are
     averaged over N models trained with different random seeds.
  4. Optional monotonic constraints on jersey_match / team_match features.
  5. Rich threshold analysis: F1, MCC, PR-AUC and precision@recall=0.95
     are all reported; F1 is used for the operating point (same as baseline).
  6. Full trial history is saved for later inspection.

Requires:
    pip install optuna

Usage:
    python train_xgboost_merger_tuned.py
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    confusion_matrix,
    precision_recall_curve,
)

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


# ---------------------------------------------------------------------------
# Paths — adjust to your environment
# ---------------------------------------------------------------------------
DATA_DIR = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\train_data\xgboost_5_neg_ratio")
SAVE_DIR = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\weights\xgboost_tuned")

# ---------------------------------------------------------------------------
# Search / training settings
# ---------------------------------------------------------------------------
N_TRIALS         = 200          # Optuna trials
CV_FOLDS         = 5            # GroupKFold folds inside each trial
ENSEMBLE_SEEDS   = [0, 1, 2, 3, 4]
INNER_EARLY_STOP = 50           # early stopping inside CV folds
FINAL_EARLY_STOP = 75           # early stopping when fitting the final models
MAX_ESTIMATORS   = 3000         # upper bound; early stopping decides actual
DEVICE           = "cpu"        # set to "cuda" if GPU available

# Monotonic constraints — set to True to apply them. These encode domain
# knowledge: if these pairwise features go up, merge probability should go up.
USE_MONOTONE_CONSTRAINTS = True
MONOTONE_INCREASING = {
    "pairwise_jersey_match",
    "pairwise_team_match",
}


# ---------------------------------------------------------------------------
# Data loading (same column conventions as baseline script)
# ---------------------------------------------------------------------------

def load_splits(data_dir: Path):
    train_path = data_dir / "combined_train_data.csv"
    val_path   = data_dir / "combined_valid_data.csv"
    test_path  = data_dir / "combined_test_data.csv"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df  = pd.read_csv(test_path)

    print(f"Loaded  train : {len(train_df):>7,} rows  "
          f"({train_df['sequence'].nunique()} sequences)")
    print(f"        valid : {len(val_df):>7,} rows  "
          f"({val_df['sequence'].nunique()} sequences)")
    print(f"        test  : {len(test_df):>7,} rows  "
          f"({test_df['sequence'].nunique()} sequences)")

    non_feature = {"label", "sequence", "A_track_id", "B_track_id"}
    all_cols = train_df.columns.tolist()
    feature_cols = [
        c for c in all_cols
        if c not in non_feature
        and (c.startswith("A_") or c.startswith("B_") or c.startswith("pairwise_"))
    ]
    print(f"        feature cols: {len(feature_cols)}")

    for df in [train_df, val_df, test_df]:
        df[feature_cols] = df[feature_cols].fillna(0)

    return train_df, val_df, test_df, feature_cols


def build_monotone_vector(feature_cols):
    """Build XGBoost's monotone_constraints vector: +1, -1, or 0 per feature."""
    vec = []
    for c in feature_cols:
        if c in MONOTONE_INCREASING:
            vec.append(1)
        else:
            vec.append(0)
    # XGBoost expects a tuple-like string e.g. "(0,1,0,-1,...)"
    return "(" + ",".join(str(v) for v in vec) + ")"


# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------

def find_best_threshold(labels, scores, metric="f1"):
    best_score, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (scores >= t).astype(int)
        if metric == "f1":
            s = f1_score(labels, preds, zero_division=0)
        elif metric == "mcc":
            s = matthews_corrcoef(labels, preds) if preds.sum() > 0 else 0.0
        else:
            raise ValueError(f"Unknown metric: {metric}")
        if s > best_score:
            best_score, best_t = s, t
    return float(best_t), float(best_score)


def precision_at_recall(labels, scores, target_recall=0.95):
    p, r, thr = precision_recall_curve(labels, scores)
    valid = r[:-1] >= target_recall
    if not valid.any():
        return 0.0, 1.0
    idx = np.where(valid)[0][-1]
    return float(p[idx]), float(thr[idx])


def evaluate(labels, scores, threshold, title=""):
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    metrics = {
        "threshold":      float(threshold),
        "auc_roc":        float(roc_auc_score(labels, scores)),
        "avg_precision":  float(average_precision_score(labels, scores)),
        "f1":             float(f1_score(labels, preds, zero_division=0)),
        "precision":      float(precision_score(labels, preds, zero_division=0)),
        "recall":         float(recall_score(labels, preds, zero_division=0)),
        "mcc":            float(matthews_corrcoef(labels, preds)) if preds.sum() > 0 else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

    if title:
        bar = "=" * 52
        print(f"\n{bar}\n  {title}\n{bar}")
        print(f"  AUC-ROC        : {metrics['auc_roc']:.4f}")
        print(f"  Avg Precision  : {metrics['avg_precision']:.4f}")
        print(f"  F1             : {metrics['f1']:.4f}")
        print(f"  Precision      : {metrics['precision']:.4f}")
        print(f"  Recall         : {metrics['recall']:.4f}")
        print(f"  MCC            : {metrics['mcc']:.4f}")
        print(f"  Threshold      : {metrics['threshold']:.3f}")
        print(f"  TN / FP        : {tn:5d} / {fp:5d}")
        print(f"  FN / TP        : {fn:5d} / {tp:5d}")
        print(bar)
    return metrics


# ---------------------------------------------------------------------------
# Optuna objective: GroupKFold CV on sequence
# ---------------------------------------------------------------------------

def make_objective(X_train, y_train, groups_train, feature_cols, monotone_str):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel":trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 5.0),
        }

        cv = GroupKFold(n_splits=CV_FOLDS)
        fold_aucs = []

        for fold_idx, (tr_idx, vl_idx) in enumerate(
            cv.split(X_train, y_train, groups_train)
        ):
            model = XGBClassifier(
                n_estimators         = MAX_ESTIMATORS,
                eval_metric          = "auc",
                early_stopping_rounds= INNER_EARLY_STOP,
                random_state         = 42,
                n_jobs               = -1,
                verbosity            = 0,
                device               = DEVICE,
                monotone_constraints = monotone_str if USE_MONOTONE_CONSTRAINTS else None,
                **params,
            )
            model.fit(
                X_train[tr_idx], y_train[tr_idx],
                eval_set=[(X_train[vl_idx], y_train[vl_idx])],
                verbose=False,
            )
            scores = model.predict_proba(X_train[vl_idx])[:, 1]
            auc    = roc_auc_score(y_train[vl_idx], scores)
            fold_aucs.append(auc)

            # Pruning: report running mean after each fold, prune if hopeless
            trial.report(float(np.mean(fold_aucs)), fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(fold_aucs))
    return objective


# ---------------------------------------------------------------------------
# Ensemble training with best params
# ---------------------------------------------------------------------------

def train_ensemble(X_train, y_train, X_val, y_val, best_params, monotone_str):
    """Train one model per seed with early stopping on the validation set."""
    models = []
    for seed in ENSEMBLE_SEEDS:
        model = XGBClassifier(
            n_estimators         = MAX_ESTIMATORS,
            eval_metric          = "auc",
            early_stopping_rounds= FINAL_EARLY_STOP,
            random_state         = seed,
            n_jobs               = -1,
            verbosity            = 0,
            device               = DEVICE,
            monotone_constraints = monotone_str if USE_MONOTONE_CONSTRAINTS else None,
            **best_params,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        best_iter = getattr(model, "best_iteration", None)
        print(f"  seed={seed}: best_iter={best_iter}  "
              f"best_score={getattr(model, 'best_score', float('nan')):.4f}")
        models.append(model)
    return models


def ensemble_predict(models, X):
    scores = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    print("Loading data...")
    train_df, val_df, test_df, feature_cols = load_splits(DATA_DIR)

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    groups_train = train_df["sequence"].values
    X_val   = val_df[feature_cols].values
    y_val   = val_df["label"].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df["label"].values

    n_pos, n_neg = int(y_train.sum()), int(len(y_train) - y_train.sum())
    print(f"\nClass balance (train): {n_pos} pos / {n_neg} neg  "
          f"(baseline scale_pos_weight = {n_neg/max(n_pos,1):.2f})")

    monotone_str = build_monotone_vector(feature_cols)
    if USE_MONOTONE_CONSTRAINTS:
        n_mono = sum(1 for c in feature_cols if c in MONOTONE_INCREASING)
        print(f"Monotonic constraints: +1 on {n_mono} features")

    # ---- Optuna search ----
    print(f"\n{'='*60}")
    print(f"  Optuna search: {N_TRIALS} trials, {CV_FOLDS}-fold GroupKFold CV")
    print(f"{'='*60}")

    sampler = TPESampler(seed=42, multivariate=True)
    pruner  = MedianPruner(n_startup_trials=10, n_warmup_steps=2)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="xgboost_merger_tuning",
    )

    t0 = time.time()
    study.optimize(
        make_objective(X_train, y_train, groups_train, feature_cols, monotone_str),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )
    elapsed = time.time() - t0
    print(f"\nOptuna done in {elapsed/60:.1f} min  "
          f"(completed={len(study.trials)}, best CV-AUC={study.best_value:.4f})")

    best_params = dict(study.best_params)
    print("\nBest params:")
    for k, v in best_params.items():
        print(f"  {k:22s} = {v}")

    # Save trial history (regardless of later steps succeeding)
    trials_df = study.trials_dataframe()
    trials_df.to_csv(SAVE_DIR / "optuna_trials.csv", index=False)

    # ---- Final ensemble training ----
    print(f"\n{'='*60}")
    print(f"  Training ensemble: {len(ENSEMBLE_SEEDS)} seeds")
    print(f"{'='*60}")
    models = train_ensemble(X_train, y_train, X_val, y_val, best_params, monotone_str)

    # ---- Threshold selection on validation ----
    val_scores = ensemble_predict(models, X_val)
    t_f1, f1_val = find_best_threshold(y_val, val_scores, metric="f1")
    t_mcc, mcc_val = find_best_threshold(y_val, val_scores, metric="mcc")
    p_at_r95, thr_at_r95 = precision_at_recall(y_val, val_scores, target_recall=0.95)

    print(f"\nThreshold analysis on validation set (ensemble):")
    print(f"  F1-optimal threshold      : {t_f1:.3f}  (F1={f1_val:.4f})")
    print(f"  MCC-optimal threshold     : {t_mcc:.3f}  (MCC={mcc_val:.4f})")
    print(f"  Precision @ Recall=0.95   : {p_at_r95:.4f}  (threshold={thr_at_r95:.3f})")

    threshold = t_f1  # match baseline convention
    evaluate(y_val, val_scores, threshold, title="Validation Results (ensemble)")

    # ---- Test evaluation ----
    test_scores  = ensemble_predict(models, X_test)
    test_metrics = evaluate(y_test, test_scores, threshold, title="Test Results (ensemble)")

    # Also evaluate single best model (first seed) for comparison
    single_test = models[0].predict_proba(X_test)[:, 1]
    evaluate(y_test, single_test, threshold, title="Test Results (single seed=0)")

    # ---- Feature importance (averaged across ensemble) ----
    importances = np.mean([m.feature_importances_ for m in models], axis=0)
    top_idx = np.argsort(importances)[::-1][:30]
    print("\nTop 30 Feature Importances (ensemble-avg):")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {feature_cols[idx]:55s} {importances[idx]:.4f}")

    # ---- Save artifacts ----
    for i, m in enumerate(models):
        m.save_model(str(SAVE_DIR / f"xgboost_merger_seed{ENSEMBLE_SEEDS[i]}.json"))

    meta = {
        "feature_cols":     feature_cols,
        "threshold":        threshold,
        "ensemble_seeds":   ENSEMBLE_SEEDS,
        "best_params":      best_params,
        "cv_best_auc":      float(study.best_value),
        "monotone_used":    USE_MONOTONE_CONSTRAINTS,
        "monotone_string":  monotone_str if USE_MONOTONE_CONSTRAINTS else None,
        "alt_thresholds": {
            "f1":        {"threshold": t_f1,  "score": f1_val},
            "mcc":       {"threshold": t_mcc, "score": mcc_val},
            "prec@r095": {"threshold": thr_at_r95, "precision": p_at_r95},
        },
        "reid_dim":   512,
        "siglip_dim": 768,
    }
    with open(SAVE_DIR / "xgboost_merger_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(SAVE_DIR / "xgboost_merger_metrics.json", "w") as f:
        json.dump({k: v for k, v in test_metrics.items() if isinstance(v, (int, float))},
                  f, indent=2)

    print(f"\nArtifacts saved to: {SAVE_DIR}")
    print(f"  ensemble models : xgboost_merger_seed*.json")
    print(f"  meta            : xgboost_merger_meta.json")
    print(f"  metrics         : xgboost_merger_metrics.json")
    print(f"  optuna trials   : optuna_trials.csv")


if __name__ == "__main__":
    train()
