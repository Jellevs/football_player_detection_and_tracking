"""
ablation_features.py

Feature ablation study for the XGBoost merger.

Two types of ablation:
  1. REMOVE one feature group at a time  → shows each group's contribution
  2. KEEP only one feature group at a time → shows each group's standalone value

Feature groups:
  - jersey:      A/B jersey_mode, jersey_entropy_mean, jersey_conf_mean, jersey_coverage
                 + pairwise jersey_match, jersey_conflict, jersey_both_confident
  - team:        A/B team_mode, team_consistency, team_coverage, team_conf_mean
                 + pairwise team_match, team_conflict, team_both_consistent
  - temporal:    A/B start_frame, end_frame, duration, n_frames
                 + pairwise temporal_gap
  - spatial:     A/B start_x, start_y, end_x, end_y, mean_bbox_height
                 + pairwise spatial_distance, endpoint_dx, endpoint_dy, bbox_height_ratio
  - reid:        pairwise reid_cosine_sim
  - siglip:      pairwise siglip_cosine_sim

For each ablation variant, the model is retrained on the same training data
(from the best sweep config) with only the selected features, then evaluated
through the full pipeline.

Usage:
    python -m experiments.tracklet_merger.xgboost.ablation_features \
        --config neg3_minlen10_purity0.8_splitTrue \
        --split valid

    # Only run "remove" ablations:
    python -m experiments.tracklet_merger.xgboost.ablation_features \
        --config neg3_minlen10_purity0.8_splitTrue \
        --split valid --mode remove

Outputs:
    experiments/tracklet_merger/xgboost/feature_ablation_results.csv
    experiments/tracklet_merger/xgboost/feature_ablation_models/<variant>/
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import settings
from experiments.idk.runner import (
    ExperimentRunner,
    default_xgboost_factory,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SWEEP_ROOT = Path(__file__).resolve().parent / "sweep_results"
OUTPUT_CSV = Path(__file__).resolve().parent / "feature_ablation_results.csv"
MODELS_DIR = Path(__file__).resolve().parent / "feature_ablation_models"


# ---------------------------------------------------------------------------
# Feature group definitions
# ---------------------------------------------------------------------------

# Per-tracklet feature stems (will be prefixed with A_ and B_)
JERSEY_STEMS   = ["jersey_mode", "jersey_entropy_mean", "jersey_conf_mean", "jersey_coverage"]
TEAM_STEMS     = ["team_mode", "team_consistency", "team_coverage", "team_conf_mean"]
TEMPORAL_STEMS = ["start_frame", "end_frame", "duration", "n_frames"]
SPATIAL_STEMS  = ["start_x", "start_y", "end_x", "end_y", "mean_bbox_height"]

# Pairwise features per group
PAIRWISE_JERSEY   = ["pairwise_jersey_match", "pairwise_jersey_conflict", "pairwise_jersey_both_confident"]
PAIRWISE_TEAM     = ["pairwise_team_match", "pairwise_team_conflict", "pairwise_team_both_consistent"]
PAIRWISE_TEMPORAL = ["pairwise_temporal_gap"]
PAIRWISE_SPATIAL  = ["pairwise_spatial_distance", "pairwise_endpoint_dx", "pairwise_endpoint_dy", "pairwise_bbox_height_ratio"]
PAIRWISE_REID     = ["pairwise_reid_cosine_sim"]
PAIRWISE_SIGLIP   = ["pairwise_siglip_cosine_sim"]


def get_group_cols(group: str) -> Set[str]:
    """Return the set of column names belonging to a feature group."""
    cols = set()
    if group == "jersey":
        for s in JERSEY_STEMS:
            cols.add(f"A_{s}")
            cols.add(f"B_{s}")
        cols.update(PAIRWISE_JERSEY)
    elif group == "team":
        for s in TEAM_STEMS:
            cols.add(f"A_{s}")
            cols.add(f"B_{s}")
        cols.update(PAIRWISE_TEAM)
    elif group == "temporal":
        for s in TEMPORAL_STEMS:
            cols.add(f"A_{s}")
            cols.add(f"B_{s}")
        cols.update(PAIRWISE_TEMPORAL)
    elif group == "spatial":
        for s in SPATIAL_STEMS:
            cols.add(f"A_{s}")
            cols.add(f"B_{s}")
        cols.update(PAIRWISE_SPATIAL)
    elif group == "reid":
        cols.update(PAIRWISE_REID)
    elif group == "siglip":
        cols.update(PAIRWISE_SIGLIP)
    else:
        raise ValueError(f"Unknown feature group: {group}")
    return cols


ALL_GROUPS = ["jersey", "team", "temporal", "spatial", "reid", "siglip"]


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "variant", "mode", "groups_description", "n_features",
    "config_name", "merge_threshold",
    "HOTA", "DetA", "AssA", "IDF1", "MOTA",
    "val_f1", "val_auc",
    "split", "timestamp",
]


def append_result(row: dict) -> None:
    write_header = not OUTPUT_CSV.exists()
    with open(OUTPUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def already_evaluated(eval_split: str) -> set:
    if not OUTPUT_CSV.exists():
        return set()
    df = pd.read_csv(OUTPUT_CSV)
    return set(df[df["split"] == eval_split]["variant"])


# ---------------------------------------------------------------------------
# Resolve model directory
# ---------------------------------------------------------------------------

def resolve_model_dir(config_name: str) -> Path:
    d = SWEEP_ROOT / config_name
    if d.exists() and (d / "xgboost_merger.json").exists():
        return d
    d_best = SWEEP_ROOT / f"{config_name}_BEST"
    if d_best.exists() and (d_best / "xgboost_merger.json").exists():
        return d_best
    raise FileNotFoundError(f"No model found for '{config_name}' in {SWEEP_ROOT}")


# ---------------------------------------------------------------------------
# Retrain with feature subset
# ---------------------------------------------------------------------------

def load_data(data_dir: Path, feature_cols: List[str]):
    """Load train/val CSVs with only the specified feature columns."""
    train_df = pd.read_csv(data_dir / "combined_train_data.csv")
    val_df   = pd.read_csv(data_dir / "combined_valid_data.csv")

    # Only keep features that exist in the data
    available = [c for c in feature_cols if c in train_df.columns]
    missing   = [c for c in feature_cols if c not in train_df.columns]
    if missing:
        print(f"    Warning: {len(missing)} features not in data: {missing[:5]}...")

    X_train = train_df[available].fillna(0).values
    y_train = train_df["label"].values
    X_val   = val_df[available].fillna(0).values
    y_val   = val_df["label"].values

    return X_train, y_train, X_val, y_val, available


def train_ablation_model(
    data_dir: Path,
    feature_cols: List[str],
    save_dir: Path,
) -> dict:
    """
    Train an XGBoost model on the given feature subset.
    Returns dict with val_f1, val_auc, threshold, feature_cols.
    """
    from sklearn.metrics import f1_score, roc_auc_score

    X_train, y_train, X_val, y_val, used_cols = load_data(data_dir, feature_cols)

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=1,
        scale_pos_weight=scale_pos_weight,
        eval_metric=["auc", "logloss"],
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        device="cpu",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_scores = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_scores)

    # Find best F1 threshold
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (val_scores >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    # Save
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(save_dir / "xgboost_merger.json"))
    meta = {"feature_cols": used_cols, "threshold": float(best_t)}
    with open(save_dir / "xgboost_merger_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {"val_f1": best_f1, "val_auc": val_auc, "threshold": best_t, "n_features": len(used_cols)}


# ---------------------------------------------------------------------------
# Build ablation variants
# ---------------------------------------------------------------------------

def build_variants(all_feature_cols: List[str], mode: str) -> List[dict]:
    """
    Build list of ablation variants.

    Each variant is a dict with:
      - name: human-readable variant name
      - mode: "remove" or "keep"
      - feature_cols: the feature columns to use
      - description: what was removed/kept
    """
    all_cols_set = set(all_feature_cols)
    variants = []

    if mode in ("remove", "both"):
        # Baseline: all features
        variants.append({
            "name": "baseline_all",
            "mode": "baseline",
            "feature_cols": all_feature_cols,
            "description": "all features",
        })

        # Remove one group at a time
        for group in ALL_GROUPS:
            group_cols = get_group_cols(group)
            remaining  = [c for c in all_feature_cols if c not in group_cols]
            variants.append({
                "name": f"remove_{group}",
                "mode": "remove",
                "feature_cols": remaining,
                "description": f"all minus {group} ({len(group_cols)} cols removed)",
            })

        # Remove jersey + team together (all attribute info)
        attr_cols = get_group_cols("jersey") | get_group_cols("team")
        remaining = [c for c in all_feature_cols if c not in attr_cols]
        variants.append({
            "name": "remove_jersey_team",
            "mode": "remove",
            "feature_cols": remaining,
            "description": f"all minus jersey+team ({len(attr_cols)} cols removed)",
        })

        # Remove reid + siglip together (all appearance embeddings)
        emb_cols = get_group_cols("reid") | get_group_cols("siglip")
        remaining = [c for c in all_feature_cols if c not in emb_cols]
        variants.append({
            "name": "remove_reid_siglip",
            "mode": "remove",
            "feature_cols": remaining,
            "description": f"all minus reid+siglip ({len(emb_cols)} cols removed)",
        })

        # Remove temporal + spatial together (all positional info)
        pos_cols = get_group_cols("temporal") | get_group_cols("spatial")
        remaining = [c for c in all_feature_cols if c not in pos_cols]
        variants.append({
            "name": "remove_temporal_spatial",
            "mode": "remove",
            "feature_cols": remaining,
            "description": f"all minus temporal+spatial ({len(pos_cols)} cols removed)",
        })

    if mode in ("keep", "both"):
        # Keep only one group at a time
        for group in ALL_GROUPS:
            group_cols = get_group_cols(group)
            kept = [c for c in all_feature_cols if c in group_cols]
            if kept:
                variants.append({
                    "name": f"only_{group}",
                    "mode": "keep",
                    "feature_cols": kept,
                    "description": f"only {group} ({len(kept)} cols)",
                })

        # Keep only pairwise features (no per-tracklet A_/B_ features)
        pw_cols = [c for c in all_feature_cols if c.startswith("pairwise_")]
        variants.append({
            "name": "only_pairwise",
            "mode": "keep",
            "feature_cols": pw_cols,
            "description": f"only pairwise features ({len(pw_cols)} cols)",
        })

        # Keep only per-tracklet features (no pairwise)
        pt_cols = [c for c in all_feature_cols if c.startswith("A_") or c.startswith("B_")]
        variants.append({
            "name": "only_per_tracklet",
            "mode": "keep",
            "feature_cols": pt_cols,
            "description": f"only per-tracklet A_/B_ features ({len(pt_cols)} cols)",
        })

        # --- Per-tracklet only removal (keep pairwise, drop per-tracklet of one group) ---
        for group, stems in [("jersey", JERSEY_STEMS), ("team", TEAM_STEMS),
                             ("temporal", TEMPORAL_STEMS), ("spatial", SPATIAL_STEMS)]:
            pt_group_cols = {f"A_{s}" for s in stems} | {f"B_{s}" for s in stems}
            remaining = [c for c in all_feature_cols if c not in pt_group_cols]
            variants.append({
                "name": f"remove_pt_{group}",
                "mode": "remove",
                "feature_cols": remaining,
                "description": f"remove per-tracklet {group} only, keep pairwise {group} ({len(pt_group_cols)} cols removed)",
            })

    return variants


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_ablation(
    config_name: str,
    eval_split: str = "valid",
    merge_threshold: float = 0.5,
    mode: str = "both",
    skip_existing: bool = True,
):
    model_dir = resolve_model_dir(config_name)

    # Load the full feature column list from the best model's meta
    with open(model_dir / "xgboost_merger_meta.json") as f:
        meta = json.load(f)
    all_feature_cols = meta["feature_cols"]

    variants = build_variants(all_feature_cols, mode)

    print(f"\n{'='*60}")
    print(f"  Feature Ablation Study")
    print(f"  Model config: {model_dir.name}")
    print(f"  Split: {eval_split}")
    print(f"  Merge threshold: {merge_threshold}")
    print(f"  Total features: {len(all_feature_cols)}")
    print(f"  Variants to test: {len(variants)}")
    print(f"{'='*60}\n")

    done = already_evaluated(eval_split) if skip_existing else set()
    remaining = [v for v in variants if v["name"] not in done]
    if done:
        print(f"  Skipping {len(variants) - len(remaining)} already-evaluated variants")
    if not remaining:
        print("  All variants already evaluated. Use --force to re-run.")
        _print_summary(eval_split)
        return

    # Prepare the runner once
    runner = ExperimentRunner(
        exp_id="feat_ablation",
        split=eval_split,
    )
    runner.prepare()

    for i, variant in enumerate(remaining, 1):
        name = variant["name"]
        print(f"\n--- [{i}/{len(remaining)}] {name}: {variant['description']} ---")

        # Train (or reuse) the ablation model
        ablation_model_dir = MODELS_DIR / name
        if (ablation_model_dir / "xgboost_merger.json").exists() and skip_existing:
            print(f"  Reusing existing model at {ablation_model_dir}")
            train_info = {"val_f1": "", "val_auc": "", "n_features": len(variant["feature_cols"])}
        else:
            print(f"  Training with {len(variant['feature_cols'])} features...")
            t0 = time.time()
            train_info = train_ablation_model(
                data_dir=model_dir,
                feature_cols=variant["feature_cols"],
                save_dir=ablation_model_dir,
            )
            print(f"  Trained in {time.time() - t0:.1f}s  "
                  f"val_f1={train_info['val_f1']:.4f}  val_auc={train_info['val_auc']:.4f}")

        # Evaluate through full pipeline
        factory = default_xgboost_factory(
            model_dir=ablation_model_dir,
            merge_threshold=merge_threshold,
        )

        t0 = time.time()
        result = runner.run(
            run_name=f"feat_ablation_{name}",
            merger_factory=factory,
            config={"variant": name},
            reuse_if_exists=skip_existing,
        )
        dt = time.time() - t0

        append_result({
            "variant":          name,
            "mode":             variant["mode"],
            "groups_description": variant["description"],
            "n_features":       train_info.get("n_features", len(variant["feature_cols"])),
            "config_name":      model_dir.name,
            "merge_threshold":  merge_threshold,
            "HOTA":             result.metrics.get("HOTA", ""),
            "DetA":             result.metrics.get("DetA", ""),
            "AssA":             result.metrics.get("AssA", ""),
            "IDF1":             result.metrics.get("IDF1", ""),
            "MOTA":             result.metrics.get("MOTA", ""),
            "val_f1":           train_info.get("val_f1", ""),
            "val_auc":          train_info.get("val_auc", ""),
            "split":            eval_split,
            "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        print(f"  HOTA={result.metrics.get('HOTA', '?'):.3f}  "
              f"AssA={result.metrics.get('AssA', '?'):.3f}  "
              f"({dt:.1f}s)")

    _print_summary(eval_split)


def _print_summary(eval_split: str):
    if not OUTPUT_CSV.exists():
        return
    df = pd.read_csv(OUTPUT_CSV)
    df = df[df["split"] == eval_split].sort_values("HOTA", ascending=False)
    if df.empty:
        return

    # Find baseline for delta computation
    baseline = df[df["variant"] == "baseline_all"]
    baseline_hota = baseline.iloc[0]["HOTA"] if len(baseline) else None

    print(f"\n{'='*70}")
    print(f"  Feature Ablation Results ({eval_split} split)")
    print(f"{'='*70}")
    print(f"{'Variant':<30s} {'Mode':<8s} {'#Feat':>5s} {'HOTA':>7s} {'AssA':>7s} {'Delta':>7s}")
    print("-" * 70)
    for _, r in df.iterrows():
        delta = ""
        if baseline_hota is not None and r["variant"] != "baseline_all":
            d = r["HOTA"] - baseline_hota
            delta = f"{d:+.3f}"
        elif r["variant"] == "baseline_all":
            delta = "  base"
        print(f"{r['variant']:<30s} {r['mode']:<8s} {int(r['n_features']):>5d} "
              f"{r['HOTA']:>7.3f} {r['AssA']:>7.3f} {delta:>7s}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature ablation study for XGBoost merger"
    )
    parser.add_argument(
        "--config", type=str, required=False, default="neg3_minlen0_purity0.8_splitTrue",
        help="Config name of the model (e.g., neg3_minlen10_purity0.8_splitTrue)"
    )
    parser.add_argument(
        "--split", type=str, default="valid",
        choices=["train", "valid", "test"],
    )
    parser.add_argument(
        "--merge-threshold", type=float, default=0.7,
    )
    parser.add_argument(
        "--mode", type=str, default="both",
        choices=["remove", "keep", "both"],
        help="Ablation mode: 'remove' one group, 'keep' only one group, or 'both'"
    )
    parser.add_argument(
        "--force", action="store_true",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_ablation(
        config_name=args.config,
        eval_split=args.split,
        merge_threshold=args.merge_threshold,
        mode=args.mode,
        skip_existing=not args.force,
    )


if __name__ == "__main__":
    main()
