"""
train_unfiltered_team.py

Trains an XGBoost merger model using UNFILTERED team predictions
(team_confidence_threshold = 0.0) so that all team predictions, regardless
of KMeans confidence, are included in the team_mode / team_consistency /
team_coverage aggregation.

Uses the optimal training data parameters found in the sweep:
    negative_ratio = 3
    min_tracklet_len = 10
    tracklet_purity = 1.0
    splitted = False

The resulting model can be compared against the filtered baseline
(team_confidence_threshold = 0.6) to determine whether the confidence
filter helps or whether XGBoost can learn to handle noisy team predictions
on its own.

Usage:
    python experiments/tracklet_merger/xgboost/train_unfiltered_team.py

Output:
    experiments/tracklet_merger/xgboost/unfiltered_team_results/
        combined_train_data.csv
        combined_valid_data.csv
        combined_test_data.csv
        xgboost_merger.json
        xgboost_merger_meta.json
        metrics.json
        feature_importance.csv
        training_log.csv
"""

import sys
sys.path.insert(0, r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch")

import json
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional

from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from utils.config import SplitterConfig


# ============================================================================
# Configuration
# ============================================================================
PROJECT_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch")
DATA_ROOT    = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\soccernet-player-tracking")
CACHE_ROOT   = PROJECT_ROOT / "output" / "cache"
OUTPUT_DIR   = PROJECT_ROOT / "experiments" / "tracklet_merger" / "xgboost" / "unfiltered_team_results"

# Optimal training data parameters (from sweep)
NEGATIVE_RATIO      = 3
MIN_TRACKLET_LEN    = 10
TRACKLET_PURITY     = 1.0
USE_SPLITTED        = False

# KEY CHANGE: no confidence filtering on team predictions
TEAM_CONFIDENCE_THRESHOLD = 0.6

# Fixed
REID_DIM   = 512
SIGLIP_DIM = 768

DATA_SPLITS = {
    "train": ["train"],
    "valid": ["valid"],
    "test":  ["test"],
}


# ============================================================================
# Cache loading (pre-split tracklets only, since splitted=False)
# ============================================================================
def load_tracklets(sequence: str) -> Optional[dict]:
    path = CACHE_ROOT / f"cache_attributes_{sequence}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# ============================================================================
# Tracklet aggregation (with unfiltered team)
# ============================================================================
def aggregate_tracklet(tracklet, track_id: int) -> Optional[dict]:
    frames     = tracklet.frames
    bboxes     = tracklet.bboxes
    embeddings = tracklet.embeddings

    if len(frames) < MIN_TRACKLET_LEN:
        return None
    if not embeddings:
        return None

    feats = {}

    # ---- ReID ----
    emb_arr  = np.stack(embeddings).astype(np.float32)
    norms    = np.linalg.norm(emb_arr, axis=1, keepdims=True) + 1e-6
    emb_norm = emb_arr / norms
    reid_mean_vec = emb_norm.mean(axis=0)

    # ---- SigLIP ----
    siglip_all   = tracklet.pred_attributes.get("siglip_embeddings", [])
    valid_siglip = [s for s in siglip_all if np.any(np.array(s) != 0)]
    if valid_siglip:
        siglip_mean_vec = np.stack(valid_siglip).astype(np.float32).mean(axis=0)
    else:
        siglip_mean_vec = np.zeros(SIGLIP_DIM, dtype=np.float32)

    # ---- Jersey ----
    jerseys   = tracklet.pred_attributes.get("jerseys", [])
    entropies = tracklet.pred_attributes.get("jersey_entropies", [1.0] * len(frames))
    confs     = tracklet.pred_attributes.get("jersey_confs_mean", [0.0] * len(frames))

    valid_j = [(j, entropies[i] if i < len(entropies) else 1.0,
                   confs[i]     if i < len(confs)     else 0.0)
               for i, j in enumerate(jerseys) if not np.isnan(j)]

    if valid_j:
        js, es, cs  = zip(*valid_j)
        mode_jersey = max(set(js), key=js.count)
        mode_es     = [e for j, e in zip(js, es) if j == mode_jersey]
        feats["jersey_mode"]         = float(mode_jersey)
        feats["jersey_entropy_mean"] = float(np.mean(mode_es))
        feats["jersey_conf_mean"]    = float(np.mean(cs))
        feats["jersey_coverage"]     = len(valid_j) / max(len(frames), 1)
    else:
        feats["jersey_mode"]         = np.nan
        feats["jersey_entropy_mean"] = 1.0
        feats["jersey_conf_mean"]    = 0.0
        feats["jersey_coverage"]     = 0.0

    # ---- Team (UNFILTERED: threshold = 0.0) ----
    teams_raw  = tracklet.pred_attributes.get("teams", [])
    team_confs = tracklet.pred_attributes.get("team_confs", [])

    # All non-NaN confidences (for mean)
    valid_team_confs = [
        team_confs[i] for i in range(len(teams_raw))
        if teams_raw[i] is not None
        and not (isinstance(teams_raw[i], float) and np.isnan(teams_raw[i]))
        and i < len(team_confs)
        and team_confs[i] is not None
        and not (isinstance(team_confs[i], float) and np.isnan(team_confs[i]))
    ]

    # All non-NaN team predictions (no confidence filter!)
    teams_valid = [
        teams_raw[i] for i in range(len(teams_raw))
        if teams_raw[i] is not None
        and not (isinstance(teams_raw[i], float) and np.isnan(teams_raw[i]))
        and i < len(team_confs)
        and team_confs[i] is not None
        and not (isinstance(team_confs[i], float) and np.isnan(team_confs[i]))
        and team_confs[i] >= TEAM_CONFIDENCE_THRESHOLD  # 0.0, so everything passes
    ]

    if teams_valid:
        mode_team   = max(set(teams_valid), key=teams_valid.count)
        consistency = teams_valid.count(mode_team) / len(teams_valid)
        feats["team_mode"]        = float(mode_team)
        feats["team_consistency"] = float(consistency)
        feats["team_coverage"]    = len(teams_valid) / max(len(frames), 1)
    else:
        feats["team_mode"]        = np.nan
        feats["team_consistency"] = 0.0
        feats["team_coverage"]    = 0.0

    feats["team_conf_mean"] = float(np.mean(valid_team_confs)) if valid_team_confs else 0.0

    # ---- Temporal / spatial ----
    feats["start_frame"]      = float(frames[0])
    feats["end_frame"]        = float(frames[-1])
    feats["duration"]         = float(frames[-1] - frames[0] + 1)
    feats["n_frames"]         = float(len(frames))

    def bbox_centre(bbox):
        return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

    sx, sy = bbox_centre(bboxes[0])
    ex, ey = bbox_centre(bboxes[-1])
    feats["start_x"]          = float(sx)
    feats["start_y"]          = float(sy)
    feats["end_x"]            = float(ex)
    feats["end_y"]            = float(ey)
    feats["mean_bbox_height"] = float(np.mean([b[3] - b[1] for b in bboxes]))

    # ---- GT label ----
    gt_ids   = tracklet.gt_attributes.get("track_ids", [])
    valid_gt = [g for g in gt_ids if g is not None and not (isinstance(g, float) and np.isnan(g))]
    if valid_gt:
        majority_gt = max(set(valid_gt), key=valid_gt.count)
        purity      = valid_gt.count(majority_gt) / len(valid_gt)
        gt_mode     = majority_gt if purity >= TRACKLET_PURITY else None
    else:
        gt_mode = None

    return {
        "track_id":        track_id,
        "gt_id":           gt_mode,
        "features":        feats,
        "reid_mean_vec":   reid_mean_vec,
        "siglip_mean_vec": siglip_mean_vec,
    }


# ============================================================================
# Pairwise features (identical to sweep script)
# ============================================================================
def compute_pairwise_features(agg_a: dict, agg_b: dict) -> dict:
    fa, fb = agg_a["features"], agg_b["features"]
    pw = {}

    if fa["end_frame"] <= fb["start_frame"]:
        exit_feat, entry_feat = fa, fb
    elif fb["end_frame"] <= fa["start_frame"]:
        exit_feat, entry_feat = fb, fa
    else:
        exit_feat, entry_feat = None, None

    pw["temporal_gap"] = (
        float(entry_feat["start_frame"] - exit_feat["end_frame"])
        if exit_feat is not None else 0.0
    )

    if exit_feat is not None:
        dx = entry_feat["start_x"] - exit_feat["end_x"]
        dy = entry_feat["start_y"] - exit_feat["end_y"]
        pw["spatial_distance"] = float(np.sqrt(dx**2 + dy**2))
        pw["endpoint_dx"]      = float(dx)
        pw["endpoint_dy"]      = float(dy)
    else:
        pw["spatial_distance"] = 0.0
        pw["endpoint_dx"]      = 0.0
        pw["endpoint_dy"]      = 0.0

    h_a, h_b = fa["mean_bbox_height"], fb["mean_bbox_height"]
    pw["bbox_height_ratio"] = float(h_a / h_b) if h_b > 0 else 1.0

    reid_a, reid_b = agg_a["reid_mean_vec"], agg_b["reid_mean_vec"]
    na, nb = np.linalg.norm(reid_a) + 1e-6, np.linalg.norm(reid_b) + 1e-6
    pw["reid_cosine_sim"] = float(np.dot(reid_a / na, reid_b / nb))

    sig_a, sig_b = agg_a["siglip_mean_vec"], agg_b["siglip_mean_vec"]
    na_s, nb_s = np.linalg.norm(sig_a) + 1e-6, np.linalg.norm(sig_b) + 1e-6
    pw["siglip_cosine_sim"] = float(np.dot(sig_a / na_s, sig_b / nb_s))

    j_a, j_b = fa["jersey_mode"], fb["jersey_mode"]
    both_j = not (np.isnan(j_a) if isinstance(j_a, float) else False) \
         and not (np.isnan(j_b) if isinstance(j_b, float) else False)
    pw["jersey_match"]          = float(both_j and j_a == j_b)
    pw["jersey_conflict"]       = float(both_j and j_a != j_b)
    pw["jersey_both_confident"] = float(
        both_j and fa["jersey_entropy_mean"] < 0.15 and fb["jersey_entropy_mean"] < 0.15
    )

    t_a, t_b = fa["team_mode"], fb["team_mode"]
    both_t = not (np.isnan(t_a) if isinstance(t_a, float) else False) \
         and not (np.isnan(t_b) if isinstance(t_b, float) else False)
    pw["team_match"]           = float(both_t and t_a == t_b)
    pw["team_conflict"]        = float(both_t and t_a != t_b)
    pw["team_both_consistent"] = float(
        both_t and fa["team_consistency"] > 0.9 and fb["team_consistency"] > 0.9
    )

    return pw


# ============================================================================
# Pair generation
# ============================================================================
def generate_pairs_for_sequence(aggregated: List[dict], sequence: str) -> pd.DataFrame:
    rows = []
    n = len(aggregated)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = aggregated[i], aggregated[j]
            fa, fb = a["features"], b["features"]

            # Skip overlapping tracklets
            if fa["end_frame"] <= fb["start_frame"]:
                gap = fb["start_frame"] - fa["end_frame"]
            elif fb["end_frame"] <= fa["start_frame"]:
                gap = fa["start_frame"] - fb["end_frame"]
            else:
                continue

            gt_a, gt_b = a["gt_id"], b["gt_id"]
            if gt_a is None or gt_b is None:
                label = 0
            else:
                label = int(gt_a == gt_b)

            pw  = compute_pairwise_features(a, b)
            row = {"sequence": sequence, "label": label}
            for k, v in fa.items():
                row[f"A_{k}"] = v
            for k, v in fb.items():
                row[f"B_{k}"] = v
            for k, v in pw.items():
                row[f"pairwise_{k}"] = v
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Apply negative ratio
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    n_keep = min(len(neg), int(len(pos) * NEGATIVE_RATIO))
    if n_keep < len(neg):
        neg = neg.sample(n=n_keep, random_state=42)
    df = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=42)

    return df


# ============================================================================
# Data generation
# ============================================================================
def generate_data(split_name: str, split_folders: List[str]) -> Optional[pd.DataFrame]:
    all_dfs = []

    for folder in split_folders:
        split_root = DATA_ROOT / folder
        if not split_root.exists():
            print(f"  [SKIP] {split_root} not found")
            continue

        sequences = sorted([d.name for d in split_root.iterdir() if d.is_dir()])

        for seq in tqdm(sequences, desc=f"  {split_name}"):
            tracklets = load_tracklets(seq)
            if tracklets is None:
                continue

            aggregated = []
            for tid, t in tracklets.items():
                agg = aggregate_tracklet(t, tid)
                if agg is not None:
                    aggregated.append(agg)

            if len(aggregated) < 2:
                continue

            df_seq = generate_pairs_for_sequence(aggregated, sequence=seq)
            if not df_seq.empty:
                all_dfs.append(df_seq)

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    gt_cols  = [c for c in combined.columns if c.endswith("_gt_id") or c == "gt_id"]
    combined.drop(columns=gt_cols, inplace=True, errors="ignore")
    return combined


# ============================================================================
# Training
# ============================================================================
def find_best_threshold(labels, scores):
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def train_and_evaluate():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Generate data ---
    print("=" * 60)
    print("Generating training data (unfiltered team, threshold=0.0)")
    print(f"  negative_ratio={NEGATIVE_RATIO}, min_tracklet_len={MIN_TRACKLET_LEN}")
    print(f"  tracklet_purity={TRACKLET_PURITY}, splitted={USE_SPLITTED}")
    print(f"  team_confidence_threshold={TEAM_CONFIDENCE_THRESHOLD}")
    print("=" * 60)

    for split_name, split_folders in DATA_SPLITS.items():
        csv_path = OUTPUT_DIR / f"combined_{split_name}_data.csv"
        if csv_path.exists():
            print(f"  {split_name}: already exists, skipping generation")
            continue

        print(f"\n  Generating {split_name} data...")
        df = generate_data(split_name, split_folders)
        if df is not None:
            df.to_csv(csv_path, index=False)
            n_pos = int(df["label"].sum())
            print(f"  {split_name}: {len(df):,} pairs (pos={n_pos}, neg={len(df) - n_pos})")
        else:
            print(f"  {split_name}: NO DATA generated")
            return

    # --- Step 2: Train ---
    print("\n" + "=" * 60)
    print("Training XGBoost...")
    print("=" * 60)

    train_df = pd.read_csv(OUTPUT_DIR / "combined_train_data.csv")
    val_df   = pd.read_csv(OUTPUT_DIR / "combined_valid_data.csv")
    test_df  = pd.read_csv(OUTPUT_DIR / "combined_test_data.csv")

    non_feature  = {"label", "sequence", "A_track_id", "B_track_id"}
    feature_cols = [c for c in train_df.columns
                    if c not in non_feature
                    and (c.startswith("A_") or c.startswith("B_") or c.startswith("pairwise_"))]

    for df in [train_df, val_df, test_df]:
        df[feature_cols] = df[feature_cols].fillna(0)

    X_train, y_train = train_df[feature_cols].values, train_df["label"].values
    X_val,   y_val   = val_df[feature_cols].values,   val_df["label"].values
    X_test,  y_test  = test_df[feature_cols].values,  test_df["label"].values

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    print(f"  Train: {len(train_df):,} pairs (pos={n_pos}, neg={n_neg}, spw={scale_pos_weight:.2f})")
    print(f"  Val:   {len(val_df):,} pairs")
    print(f"  Test:  {len(test_df):,} pairs")

    model = XGBClassifier(
        n_estimators          = 1000,
        max_depth             = 6,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        min_child_weight      = 5,
        gamma                 = 1,
        scale_pos_weight      = scale_pos_weight,
        eval_metric           = ["auc", "logloss", "aucpr", "error"],
        early_stopping_rounds = 30,
        random_state          = 42,
        n_jobs                = -1,
        verbosity             = 0,
        device                = "cpu",
    )

    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    train_time = time.time() - t0

    # Save training log
    results      = model.evals_result()
    metric_names = list(results["validation_0"].keys())
    n_rounds     = len(results["validation_0"][metric_names[0]])

    log_rows = []
    for r in range(n_rounds):
        row = {"round": r}
        for m in metric_names:
            row[f"train_{m}"] = results["validation_0"][m][r]
            row[f"val_{m}"]   = results["validation_1"][m][r]
        log_rows.append(row)
    pd.DataFrame(log_rows).to_csv(OUTPUT_DIR / "training_log.csv", index=False)

    # --- Step 3: Evaluate ---
    val_scores  = model.predict_proba(X_val)[:, 1]
    threshold   = find_best_threshold(y_val, val_scores)
    val_preds   = (val_scores >= threshold).astype(int)
    val_tn, val_fp, val_fn, val_tp = confusion_matrix(y_val, val_preds).ravel()

    test_scores = model.predict_proba(X_test)[:, 1]
    test_preds  = (test_scores >= threshold).astype(int)
    test_tn, test_fp, test_fn, test_tp = confusion_matrix(y_test, test_preds).ravel()

    metrics = {
        "team_confidence_threshold": TEAM_CONFIDENCE_THRESHOLD,
        "negative_ratio":     NEGATIVE_RATIO,
        "min_tracklet_len":   MIN_TRACKLET_LEN,
        "tracklet_purity":    TRACKLET_PURITY,
        "splitted":           USE_SPLITTED,
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "train_pos_ratio":    float(n_pos / max(len(train_df), 1)),
        "scale_pos_weight":   scale_pos_weight,
        "n_rounds":           n_rounds,
        "best_iteration":     getattr(model, "best_iteration", n_rounds),
        "threshold":          threshold,
        "train_time_seconds": train_time,
        # Validation
        "val_auc_roc":       roc_auc_score(y_val, val_scores),
        "val_avg_precision": average_precision_score(y_val, val_scores),
        "val_f1":            f1_score(y_val, val_preds, zero_division=0),
        "val_precision":     precision_score(y_val, val_preds, zero_division=0),
        "val_recall":        recall_score(y_val, val_preds, zero_division=0),
        "val_tp": int(val_tp), "val_fp": int(val_fp),
        "val_fn": int(val_fn), "val_tn": int(val_tn),
        # Test
        "test_auc_roc":       roc_auc_score(y_test, test_scores),
        "test_avg_precision": average_precision_score(y_test, test_scores),
        "test_f1":            f1_score(y_test, test_preds, zero_division=0),
        "test_precision":     precision_score(y_test, test_preds, zero_division=0),
        "test_recall":        recall_score(y_test, test_preds, zero_division=0),
        "test_tp": int(test_tp), "test_fp": int(test_fp),
        "test_fn": int(test_fn), "test_tn": int(test_tn),
    }

    # Save model + meta
    model.save_model(str(OUTPUT_DIR / "xgboost_merger.json"))

    meta = {
        "feature_cols":              feature_cols,
        "threshold":                 threshold,
        "reid_dim":                  REID_DIM,
        "siglip_dim":               SIGLIP_DIM,
        "team_confidence_threshold": TEAM_CONFIDENCE_THRESHOLD,
    }
    with open(OUTPUT_DIR / "xgboost_merger_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Feature importance
    fi_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    # --- Print results ---
    print(f"\n{'=' * 60}")
    print("RESULTS (unfiltered team, threshold=0.0)")
    print(f"{'=' * 60}")
    print(f"  Rounds: {n_rounds}  Best iter: {metrics['best_iteration']}  Threshold: {threshold:.3f}")
    print(f"  Train time: {train_time:.1f}s")
    print(f"\n  Validation:")
    print(f"    AUC-ROC:  {metrics['val_auc_roc']:.4f}")
    print(f"    F1:       {metrics['val_f1']:.4f}")
    print(f"    Prec:     {metrics['val_precision']:.4f}  Recall: {metrics['val_recall']:.4f}")
    print(f"\n  Test:")
    print(f"    AUC-ROC:  {metrics['test_auc_roc']:.4f}")
    print(f"    F1:       {metrics['test_f1']:.4f}")
    print(f"    Prec:     {metrics['test_precision']:.4f}  Recall: {metrics['test_recall']:.4f}")
    print(f"\n  Output: {OUTPUT_DIR}")

    # Compare hint
    print(f"\n{'=' * 60}")
    print("To compare with the filtered baseline, check the sweep results:")
    print(f"  Filtered (0.6): experiments/tracklet_merger/xgboost/sweep_results/neg3_minlen10_purity1.0_splitFalse/")
    print(f"  Unfiltered (0.0): {OUTPUT_DIR}")
    print(f"\nTo use this model in the full pipeline, set in main.py:")
    print(f'  model_path = "{OUTPUT_DIR / "xgboost_merger.json"}"')
    print(f'  meta_path  = "{OUTPUT_DIR / "xgboost_merger_meta.json"}"')
    print(f"  team_confidence_threshold = 0.0")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    train_and_evaluate()
