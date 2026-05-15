"""
sweep_xgboost_training.py

Grid search over training data parameters for the XGBoost tracklet merger.
For each combination, generates training data (if not already cached), trains
a model with fixed hyperparameters, and logs all results into a single
comparison CSV.

Swept parameters:
    - NEGATIVE_RATIO:          controls class balance in training data
    - MIN_TRACKLET_LEN:        minimum tracklet length to include
    - TRACKLET_PURITY_THRESHOLD: label noise control
    - SPLITTED:                whether to use post-split tracklets

Fixed (not swept):
    - XGBoost hyperparameters: kept constant so differences are due to data only
    - TEAM_CONFIDENCE_THRESHOLD: 0.6
    - MAX_TEMPORAL_GAP: None

Usage:
    python sweep_xgboost_training.py

Outputs:
    sweep_results/
        sweep_summary.csv          — one row per config with all metrics
        <config_name>/
            combined_train_data.csv
            combined_valid_data.csv
            combined_test_data.csv
            xgboost_merger.json
            xgboost_merger_meta.json
            xgboost_merger_metrics.json
            training_log.csv
            feature_importance.csv
"""

import sys
sys.path.insert(0, r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch")

import json
import time
import pickle
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
from dataclasses import dataclass

from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from tracklets.split_tracklets import split_tracklets
from utils.config import SplitterConfig


# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch")
DATA_ROOT    = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\soccernet-player-tracking")
CACHE_ROOT   = PROJECT_ROOT / "output" / "cache"
CACHE_SPLIT_ROOT = PROJECT_ROOT / "output" / "cache_split"
SWEEP_ROOT   = PROJECT_ROOT / "experiments" / "tracklet_merger" / "xgboost" / "sweep_results"

SPLITTER_CFG = SplitterConfig()

# Fixed constants
REID_DIM   = 512
SIGLIP_DIM = 768
TEAM_CONFIDENCE_THRESHOLD = 0.6

# ============================================================================
# Sweep grid
# ============================================================================
SWEEP_GRID = {
    "negative_ratio":          [None, 3, 5, 10],
    "min_tracklet_len":        [0, 5, 10],
    "tracklet_purity":         [0.6, 0.8, 1.0],
    "splitted":                [True, False],
}

# Data splits to process for each CSV
DATA_SPLITS = {
    "train": ["train"],
    "valid": ["valid"],
    "test":  ["test"],
}


# ============================================================================
# Config dataclass
# ============================================================================
@dataclass
class SweepConfig:
    negative_ratio: float
    min_tracklet_len: int
    tracklet_purity: float
    splitted: bool

    @property
    def name(self) -> str:
        return (
            f"neg{self.negative_ratio}"
            f"_minlen{self.min_tracklet_len}"
            f"_purity{self.tracklet_purity}"
            f"_split{self.splitted}"
        )


# ============================================================================
# Cache loading
# ============================================================================
def load_presplit_tracklets(sequence: str) -> Optional[dict]:
    path = CACHE_ROOT / f"cache_attributes_{sequence}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def load_split_tracklets(sequence: str) -> Optional[dict]:
    path = CACHE_SPLIT_ROOT / f"cache_split_{sequence}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)

    # Cache miss — compute from presplit
    tracklets = load_presplit_tracklets(sequence)
    if tracklets is None:
        return None

    tracklets = split_tracklets(tracklets, SPLITTER_CFG)

    CACHE_SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(tracklets, f)
    return tracklets


# ============================================================================
# Tracklet aggregation (parameterized)
# ============================================================================
def aggregate_tracklet(tracklet, track_id: int, cfg: SweepConfig) -> Optional[dict]:
    frames     = tracklet.frames
    bboxes     = tracklet.bboxes
    embeddings = tracklet.embeddings

    if len(frames) < cfg.min_tracklet_len:
        return None
    if not embeddings:
        return None

    feats = {}

    # ReID mean vector (for pairwise cosine sim, not written to CSV)
    emb_arr  = np.stack(embeddings).astype(np.float32)
    norms    = np.linalg.norm(emb_arr, axis=1, keepdims=True) + 1e-6
    emb_norm = emb_arr / norms
    reid_mean_vec = emb_norm.mean(axis=0)

    # SigLIP mean vector
    siglip_all   = tracklet.pred_attributes.get("siglip_embeddings", [])
    valid_siglip = [s for s in siglip_all if np.any(np.array(s) != 0)]
    if valid_siglip:
        siglip_mean_vec = np.stack(valid_siglip).astype(np.float32).mean(axis=0)
    else:
        siglip_mean_vec = np.zeros(SIGLIP_DIM, dtype=np.float32)

    # Jersey
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

    # Team
    teams_raw  = tracklet.pred_attributes.get("teams", [])
    team_confs = tracklet.pred_attributes.get("team_confs", [])

    valid_team_confs = [
        team_confs[i] for i in range(len(teams_raw))
        if teams_raw[i] is not None
        and not (isinstance(teams_raw[i], float) and np.isnan(teams_raw[i]))
        and i < len(team_confs)
        and team_confs[i] is not None
        and not (isinstance(team_confs[i], float) and np.isnan(team_confs[i]))
    ]
    teams_confident = [
        teams_raw[i] for i in range(len(teams_raw))
        if teams_raw[i] is not None
        and not (isinstance(teams_raw[i], float) and np.isnan(teams_raw[i]))
        and i < len(team_confs)
        and team_confs[i] is not None
        and not (isinstance(team_confs[i], float) and np.isnan(team_confs[i]))
        and team_confs[i] >= TEAM_CONFIDENCE_THRESHOLD
    ]

    if teams_confident:
        mode_team   = max(set(teams_confident), key=teams_confident.count)
        consistency = teams_confident.count(mode_team) / len(teams_confident)
        feats["team_mode"]        = float(mode_team)
        feats["team_consistency"] = float(consistency)
        feats["team_coverage"]    = len(teams_confident) / max(len(frames), 1)
    else:
        feats["team_mode"]        = np.nan
        feats["team_consistency"] = 0.0
        feats["team_coverage"]    = 0.0

    feats["team_conf_mean"] = float(np.mean(valid_team_confs)) if valid_team_confs else 0.0

    # Temporal / spatial
    feats["start_frame"]     = float(frames[0])
    feats["end_frame"]       = float(frames[-1])
    feats["duration"]        = float(frames[-1] - frames[0] + 1)
    feats["n_frames"]        = float(len(frames))

    def bbox_centre(bbox):
        return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

    sx, sy = bbox_centre(bboxes[0])
    ex, ey = bbox_centre(bboxes[-1])
    feats["start_x"]         = float(sx)
    feats["start_y"]         = float(sy)
    feats["end_x"]           = float(ex)
    feats["end_y"]           = float(ey)
    feats["mean_bbox_height"] = float(np.mean([b[3] - b[1] for b in bboxes]))

    # GT label
    gt_ids   = tracklet.gt_attributes.get("track_ids", [])
    valid_gt = [g for g in gt_ids if g is not None and not (isinstance(g, float) and np.isnan(g))]
    if valid_gt:
        majority_gt = max(set(valid_gt), key=valid_gt.count)
        purity      = valid_gt.count(majority_gt) / len(valid_gt)
        gt_mode     = majority_gt if purity >= cfg.tracklet_purity else None
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
# Pairwise features
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
# Pair generation for one sequence
# ============================================================================
def generate_pairs_for_sequence(
    aggregated: List[dict],
    sequence: str,
    negative_ratio: Optional[float] = None,
) -> pd.DataFrame:
    rows = []
    n = len(aggregated)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = aggregated[i], aggregated[j]
            fa, fb = a["features"], b["features"]

            if fa["end_frame"] <= fb["start_frame"]:
                gap = fb["start_frame"] - fa["end_frame"]
            elif fb["end_frame"] <= fa["start_frame"]:
                gap = fa["start_frame"] - fb["end_frame"]
            else:
                continue  # overlapping

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

    if negative_ratio is not None:
        pos = df[df["label"] == 1]
        neg = df[df["label"] == 0]
        n_keep = min(len(neg), int(len(pos) * negative_ratio))
        if n_keep < len(neg):
            neg = neg.sample(n=n_keep, random_state=42)
        df = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=42)

    return df


# ============================================================================
# Data generation for one config + one data split
# ============================================================================
def generate_data_for_split(cfg: SweepConfig, split_name: str, split_folders: List[str]) -> Optional[pd.DataFrame]:
    all_dfs = []

    for folder in split_folders:
        split_root = DATA_ROOT / folder
        if not split_root.exists():
            print(f"  [SKIP] {split_root} not found")
            continue

        sequences = sorted([d.name for d in split_root.iterdir() if d.is_dir()])

        for seq in sequences:
            if cfg.splitted:
                tracklets = load_split_tracklets(seq)
            else:
                tracklets = load_presplit_tracklets(seq)

            if tracklets is None:
                continue

            aggregated = []
            for tid, t in tracklets.items():
                agg = aggregate_tracklet(t, tid, cfg)
                if agg is not None:
                    aggregated.append(agg)

            if len(aggregated) < 2:
                continue

            df_seq = generate_pairs_for_sequence(
                aggregated, sequence=seq, negative_ratio=cfg.negative_ratio
            )
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


def train_and_evaluate(cfg: SweepConfig, config_dir: Path) -> Optional[dict]:
    """Train XGBoost on pre-generated CSVs in config_dir. Returns metrics dict."""
    train_path = config_dir / "combined_train_data.csv"
    val_path   = config_dir / "combined_valid_data.csv"
    test_path  = config_dir / "combined_test_data.csv"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            print(f"  [SKIP] Missing: {p}")
            return None

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df  = pd.read_csv(test_path)

    # Feature columns
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

    # Train with fixed hyperparameters
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

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    # Save training log
    results      = model.evals_result()
    metric_names = list(results["validation_0"].keys())
    n_rounds     = len(results["validation_0"][metric_names[0]])

    log_path = config_dir / "training_log.csv"
    header   = ["round"]
    for m in metric_names:
        header.extend([f"train_{m}", f"val_{m}"])
    with open(log_path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in range(n_rounds):
            row = [str(r)]
            for m in metric_names:
                row.append(str(results["validation_0"][m][r]))
                row.append(str(results["validation_1"][m][r]))
            f.write(",".join(row) + "\n")

    # Threshold selection on validation
    val_scores = model.predict_proba(X_val)[:, 1]
    threshold  = find_best_threshold(y_val, val_scores)

    # Validation metrics
    val_preds = (val_scores >= threshold).astype(int)
    val_tn, val_fp, val_fn, val_tp = confusion_matrix(y_val, val_preds).ravel()

    # Test metrics
    test_scores = model.predict_proba(X_test)[:, 1]
    test_preds  = (test_scores >= threshold).astype(int)
    test_tn, test_fp, test_fn, test_tp = confusion_matrix(y_test, test_preds).ravel()

    metrics = {
        # Config
        "config_name":        cfg.name,
        "negative_ratio":     cfg.negative_ratio,
        "min_tracklet_len":   cfg.min_tracklet_len,
        "tracklet_purity":    cfg.tracklet_purity,
        "splitted":           cfg.splitted,
        # Data stats
        "n_train":            len(train_df),
        "n_val":              len(val_df),
        "n_test":             len(test_df),
        "train_pos_ratio":    float(n_pos / max(len(train_df), 1)),
        "scale_pos_weight":   scale_pos_weight,
        "n_rounds":           n_rounds,
        "best_iteration":     getattr(model, "best_iteration", n_rounds),
        # Threshold
        "threshold":          threshold,
        # Validation metrics
        "val_auc_roc":        roc_auc_score(y_val, val_scores),
        "val_avg_precision":  average_precision_score(y_val, val_scores),
        "val_f1":             f1_score(y_val, val_preds, zero_division=0),
        "val_precision":      precision_score(y_val, val_preds, zero_division=0),
        "val_recall":         recall_score(y_val, val_preds, zero_division=0),
        "val_tp": int(val_tp), "val_fp": int(val_fp),
        "val_fn": int(val_fn), "val_tn": int(val_tn),
        # Test metrics
        "test_auc_roc":       roc_auc_score(y_test, test_scores),
        "test_avg_precision": average_precision_score(y_test, test_scores),
        "test_f1":            f1_score(y_test, test_preds, zero_division=0),
        "test_precision":     precision_score(y_test, test_preds, zero_division=0),
        "test_recall":        recall_score(y_test, test_preds, zero_division=0),
        "test_tp": int(test_tp), "test_fp": int(test_fp),
        "test_fn": int(test_fn), "test_tn": int(test_tn),
    }

    # Save model
    model.save_model(str(config_dir / "xgboost_merger.json"))

    # Save meta
    meta = {
        "feature_cols": feature_cols,
        "threshold":    threshold,
        "reid_dim":     REID_DIM,
        "siglip_dim":   SIGLIP_DIM,
        "config":       {
            "negative_ratio":     cfg.negative_ratio,
            "min_tracklet_len":   cfg.min_tracklet_len,
            "tracklet_purity":    cfg.tracklet_purity,
            "splitted":           cfg.splitted,
        },
    }
    with open(config_dir / "xgboost_merger_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save test metrics
    with open(config_dir / "xgboost_merger_metrics.json", "w") as f:
        json.dump({k: v for k, v in metrics.items() if isinstance(v, (int, float, bool))}, f, indent=2)

    # Save feature importance
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    fi_df.to_csv(config_dir / "feature_importance.csv", index=False)

    return metrics


# ============================================================================
# Main sweep
# ============================================================================
def main():
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)

    # Build all configs
    keys   = list(SWEEP_GRID.keys())
    values = list(SWEEP_GRID.values())
    configs = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        configs.append(SweepConfig(**params))

    print(f"Total configurations: {len(configs)}")
    print(f"Sweep output: {SWEEP_ROOT}\n")

    all_results = []

    for idx, cfg in enumerate(configs):
        config_dir = SWEEP_ROOT / cfg.name
        config_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"  [{idx+1}/{len(configs)}] {cfg.name}")
        print(f"    negative_ratio={cfg.negative_ratio}  min_tracklet_len={cfg.min_tracklet_len}  "
              f"tracklet_purity={cfg.tracklet_purity}  splitted={cfg.splitted}")
        print(f"{'='*70}")

        t0 = time.time()

        # --- Step 1: Generate data (skip if CSVs already exist) ---
        all_csvs_exist = all(
            (config_dir / f"combined_{split}_data.csv").exists()
            for split in ["train", "valid", "test"]
        )

        if all_csvs_exist:
            print("  Data CSVs already exist, skipping generation.")
        else:
            print("  Generating training data...")
            for split_name, split_folders in DATA_SPLITS.items():
                csv_path = config_dir / f"combined_{split_name}_data.csv"
                if csv_path.exists():
                    print(f"    {split_name}: already exists, skipping")
                    continue

                print(f"    {split_name}: generating...")
                df = generate_data_for_split(cfg, split_name, split_folders)
                if df is not None:
                    df.to_csv(csv_path, index=False)
                    n_pos = int(df["label"].sum())
                    print(f"    {split_name}: {len(df):,} pairs (pos={n_pos}, neg={len(df)-n_pos})")
                else:
                    print(f"    {split_name}: NO DATA generated (missing caches?)")

        # --- Step 2: Train and evaluate ---
        print("  Training XGBoost...")
        metrics = train_and_evaluate(cfg, config_dir)

        elapsed = time.time() - t0

        if metrics is not None:
            metrics["elapsed_seconds"] = elapsed
            all_results.append(metrics)
            print(f"  Val  AUC={metrics['val_auc_roc']:.4f}  F1={metrics['val_f1']:.4f}")
            print(f"  Test AUC={metrics['test_auc_roc']:.4f}  F1={metrics['test_f1']:.4f}")
            print(f"  Rounds={metrics['n_rounds']}  Threshold={metrics['threshold']:.3f}  ({elapsed:.1f}s)")
        else:
            print(f"  FAILED — no metrics produced ({elapsed:.1f}s)")

        # Save intermediate summary after each config (in case of crash)
        if all_results:
            summary_df = pd.DataFrame(all_results)
            summary_df.to_csv(SWEEP_ROOT / "sweep_summary.csv", index=False)

    # Final summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv(SWEEP_ROOT / "sweep_summary.csv", index=False)

        print(f"\n{'='*70}")
        print(f"  SWEEP COMPLETE — {len(all_results)} / {len(configs)} configs succeeded")
        print(f"{'='*70}")

        # Print top 10 by test AUC
        top = summary_df.nlargest(10, "test_auc_roc")
        print("\nTop 10 configs by test AUC-ROC:")
        print(f"  {'Config':<50s} {'Test AUC':>10s} {'Test F1':>10s} {'Val AUC':>10s} {'Val F1':>10s}")
        print(f"  {'-'*50} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for _, row in top.iterrows():
            print(f"  {row['config_name']:<50s} {row['test_auc_roc']:>10.4f} {row['test_f1']:>10.4f} "
                  f"{row['val_auc_roc']:>10.4f} {row['val_f1']:>10.4f}")

        print(f"\nFull results: {SWEEP_ROOT / 'sweep_summary.csv'}")
    else:
        print("\n[ERROR] No configs completed successfully.")


if __name__ == "__main__":
    main()
