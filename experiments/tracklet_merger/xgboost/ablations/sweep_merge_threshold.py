"""
sweep_merge_threshold.py

Sweep merge distance threshold and linkage method for hierarchical clustering
on a single XGBoost model (the best from the training-data sweep).

The XGBoost merger converts predicted merge probability to distance (1 - p),
then feeds this into hierarchical agglomerative clustering. This experiment
keeps the model fixed and only varies the clustering parameters (threshold
and linkage method), so no retraining is needed — just re-merge and
re-evaluate.

Usage:
    python -m experiments.tracklet_merger.xgboost.sweep_merge_threshold
        --config neg3_minlen0_purity0.6_splitTrue --split valid

    # Custom threshold range and linkage methods:
    python -m experiments.tracklet_merger.xgboost.sweep_merge_threshold
        --config neg3_minlen0_purity0.6_splitTrue
        --thresholds 0.3 0.4 0.5 0.6 0.7
        --linkage-methods average single complete

Outputs:
    experiments/tracklet_merger/xgboost/merge_threshold_results.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import List

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import settings
from experiments.idk.runner import (
    ExperimentRunner,
    default_xgboost_factory,
    summarize_results,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SWEEP_ROOT = Path(__file__).resolve().parent / "sweep_results"
OUTPUT_CSV = Path(__file__).resolve().parent / "merge_threshold_results.csv"


# ---------------------------------------------------------------------------
# CSV output (incremental, crash-safe)
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "config_name", "merge_threshold", "linkage_method",
    "HOTA", "DetA", "AssA", "IDF1", "MOTA",
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
    """Return (config_name, threshold, linkage) tuples already evaluated."""
    if not OUTPUT_CSV.exists():
        return set()
    df = pd.read_csv(OUTPUT_CSV)
    df_split = df[df["split"] == eval_split]
    if "linkage_method" not in df_split.columns:
        # Old CSV without linkage column — treat all as "average"
        return set(
            (cn, t, "average")
            for cn, t in zip(df_split["config_name"], df_split["merge_threshold"])
        )
    return set(zip(
        df_split["config_name"],
        df_split["merge_threshold"],
        df_split["linkage_method"],
    ))


# ---------------------------------------------------------------------------
# Resolve model directory
# ---------------------------------------------------------------------------

def resolve_model_dir(config_name: str) -> Path:
    """Find the sweep results directory for a config name."""
    d = SWEEP_ROOT / config_name
    if d.exists() and (d / "xgboost_merger.json").exists():
        return d
    # Try with _BEST suffix
    d_best = SWEEP_ROOT / f"{config_name}_BEST"
    if d_best.exists() and (d_best / "xgboost_merger.json").exists():
        return d_best
    raise FileNotFoundError(
        f"No model found for config '{config_name}' in {SWEEP_ROOT}"
    )


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_threshold_sweep(
    config_name: str,
    thresholds: List[float],
    linkage_methods: List[str],
    eval_split: str = "valid",
    skip_existing: bool = True,
):
    model_dir = resolve_model_dir(config_name)

    # Build all (threshold, linkage) combos
    combos = [(t, lm) for lm in linkage_methods for t in thresholds]

    print(f"\n{'='*60}")
    print(f"  Merge Threshold & Linkage Sweep")
    print(f"  Model: {model_dir.name}")
    print(f"  Split: {eval_split}")
    print(f"  Thresholds: {thresholds}")
    print(f"  Linkage methods: {linkage_methods}")
    print(f"  Total combinations: {len(combos)}")
    print(f"  Output: {OUTPUT_CSV}")
    print(f"{'='*60}\n")

    # Check which combos are already done
    done = already_evaluated(eval_split) if skip_existing else set()
    remaining = [(t, lm) for t, lm in combos
                 if (model_dir.name, t, lm) not in done]
    if done:
        skipped = len(combos) - len(remaining)
        if skipped:
            print(f"  Skipping {skipped} already-evaluated combinations")
    if not remaining:
        print("  All combinations already evaluated. Use --force to re-run.")
        _print_summary(config_name, eval_split)
        return

    # Prepare the runner once (loads attribute caches + runs splitter)
    runner = ExperimentRunner(
        exp_id="thresh_sweep",
        split=eval_split,
    )
    runner.prepare()

    results = []
    for i, (threshold, lm) in enumerate(remaining, 1):
        print(f"\n--- [{i}/{len(remaining)}] threshold={threshold:.3f}  linkage={lm} ---")

        factory = default_xgboost_factory(
            model_dir=model_dir,
            merge_threshold=threshold,
            linkage_method=lm,
        )

        t0 = time.time()
        result = runner.run(
            run_name=f"{model_dir.name}_t{threshold:.3f}_{lm}",
            merger_factory=factory,
            config={
                "config_name": model_dir.name,
                "merge_threshold": threshold,
                "linkage_method": lm,
            },
            reuse_if_exists=skip_existing,
        )
        dt = time.time() - t0

        append_result({
            "config_name":     model_dir.name,
            "merge_threshold": threshold,
            "linkage_method":  lm,
            "HOTA":            result.metrics.get("HOTA", ""),
            "DetA":            result.metrics.get("DetA", ""),
            "AssA":            result.metrics.get("AssA", ""),
            "IDF1":            result.metrics.get("IDF1", ""),
            "MOTA":            result.metrics.get("MOTA", ""),
            "split":           eval_split,
            "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        results.append(result)
        print(f"  HOTA={result.metrics.get('HOTA', '?'):.3f}  "
              f"AssA={result.metrics.get('AssA', '?'):.3f}  "
              f"({dt:.1f}s)")

    # Summary
    if results:
        summarize_results(results, sort_by="HOTA")

    _print_summary(config_name, eval_split)


def _print_summary(config_name: str, eval_split: str):
    """Print all results for this config and split, grouped by linkage method."""
    if not OUTPUT_CSV.exists():
        return
    df = pd.read_csv(OUTPUT_CSV)
    mask = (df["split"] == eval_split) & (
        df["config_name"].str.replace("_BEST", "") == config_name.replace("_BEST", "")
    )
    df_filtered = df[mask].copy()
    if df_filtered.empty:
        return

    # Add linkage column for old CSVs
    if "linkage_method" not in df_filtered.columns:
        df_filtered["linkage_method"] = "average"

    df_filtered = df_filtered.sort_values(["linkage_method", "merge_threshold"])

    print(f"\n{'='*70}")
    print(f"  Clustering parameter sweep for {config_name} ({eval_split} split)")
    print(f"{'='*70}")

    for lm in df_filtered["linkage_method"].unique():
        df_lm = df_filtered[df_filtered["linkage_method"] == lm]
        print(f"\n  --- {lm} linkage ---")
        print(df_lm[["merge_threshold", "HOTA", "DetA", "AssA", "IDF1", "MOTA"]].to_string(index=False))
        best = df_lm.loc[df_lm["HOTA"].idxmax()]
        print(f"  Best: threshold={best['merge_threshold']:.3f}  HOTA={best['HOTA']:.3f}")

    overall_best = df_filtered.loc[df_filtered["HOTA"].idxmax()]
    print(f"\n  Overall best: linkage={overall_best['linkage_method']}  "
          f"threshold={overall_best['merge_threshold']:.3f}  "
          f"HOTA={overall_best['HOTA']:.3f}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep merge threshold for a single XGBoost merger model"
    )
    parser.add_argument(
        "--config", type=str, required=False, default="neg3_minlen0_purity0.8_splitTrue",
        help="Config name of the model to evaluate (e.g., neg3_minlen10_purity0.8_splitTrue)"
    )
    parser.add_argument(
        "--split", type=str, default="valid",
        choices=["train", "valid", "test"],
        help="Dataset split to evaluate on (default: valid)"
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=None,
        help="Custom threshold values (default: 0.30 to 0.80 in steps of 0.05)"
    )
    parser.add_argument(
        "--linkage-methods", type=str, nargs="+", default=None,
        help="Linkage methods to test (default: average single complete)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-evaluate combinations that already have results"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.thresholds is None:
        thresholds = [round(t, 3) for t in
                      [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]]
    else:
        thresholds = [round(t, 3) for t in args.thresholds]

    linkage_methods = args.linkage_methods or ["average", "single", "complete"]

    run_threshold_sweep(
        config_name=args.config,
        thresholds=thresholds,
        linkage_methods=linkage_methods,
        eval_split=args.split,
        skip_existing=not args.force,
    )


if __name__ == "__main__":
    main()
