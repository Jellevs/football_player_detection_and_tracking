"""
sweep_hota_eval.py

Evaluate XGBoost sweep models using the full pipeline (split → merge → HOTA).

The sweep_xgboost_training.py script found optimal training-data configs by
XGBoost classifier metrics (AUC, F1).  However, those metrics don't correlate
well with end-to-end tracking performance (HOTA).  This script bridges the gap
by running each sweep model through the full pipeline on the **validation** set
and recording the resulting HOTA (and AssA, DetA, IDF1, MOTA).

It reuses the ExperimentRunner infrastructure from experiments/idk/runner.py,
which:
  1. loads cached attribute tracklets (output/cache/cache_attributes_<seq>.pkl)
  2. runs the splitter once (cached in memory)
  3. deep-copies split output per model, runs the merger, writes MOT files
  4. calls sn-trackeval for HOTA evaluation

Usage:
    python -m experiments.tracklet_merger.xgboost.sweep_hota_eval          # all 72
    python -m experiments.tracklet_merger.xgboost.sweep_hota_eval --top 15 # top 15 by test F1
    python -m experiments.tracklet_merger.xgboost.sweep_hota_eval --configs neg3_minlen10_purity1.0_splitFalse neg5_minlen10_purity1.0_splitFalse

Outputs:
    experiments/tracklet_merger/xgboost/sweep_hota_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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
SWEEP_ROOT   = Path(__file__).resolve().parent / "sweep_results"
SUMMARY_CSV  = SWEEP_ROOT / "sweep_summary.csv"
OUTPUT_CSV   = Path(__file__).resolve().parent / "sweep_hota_results.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_sweep_dirs() -> List[Path]:
    """Return all sweep result directories that contain a trained model."""
    dirs = []
    for d in sorted(SWEEP_ROOT.iterdir()):
        if d.is_dir() and (d / "xgboost_merger.json").exists():
            dirs.append(d)
    return dirs


def load_sweep_summary() -> Optional[pd.DataFrame]:
    """Load sweep_summary.csv if it exists (for ranking/filtering)."""
    if SUMMARY_CSV.exists():
        return pd.read_csv(SUMMARY_CSV)
    return None


def top_n_by_metric(n: int, metric: str = "test_f1") -> List[str]:
    """Return the top-N config names ranked by a sweep metric."""
    df = load_sweep_summary()
    if df is None:
        print(f"Warning: {SUMMARY_CSV} not found, evaluating all configs")
        return [d.name for d in discover_sweep_dirs()]
    df_sorted = df.sort_values(metric, ascending=False).head(n)
    return df_sorted["config_name"].tolist()


def filter_configs(configs: List[str]) -> List[Path]:
    """Convert config names to validated directory paths."""
    dirs = []
    for name in configs:
        # Strip _BEST suffix if present for matching
        d = SWEEP_ROOT / name
        if not d.exists():
            # Try with _BEST suffix
            d_best = SWEEP_ROOT / f"{name}_BEST"
            if d_best.exists():
                d = d_best
            else:
                print(f"  Warning: config directory not found: {name}")
                continue
        if not (d / "xgboost_merger.json").exists():
            print(f"  Warning: no model in {d.name}, skipping")
            continue
        dirs.append(d)
    return dirs


# ---------------------------------------------------------------------------
# CSV output (incremental, crash-safe)
# ---------------------------------------------------------------------------

HOTA_CSV_COLUMNS = [
    "config_name", "HOTA", "DetA", "AssA", "IDF1", "MOTA",
    "merge_threshold", "split", "timestamp",
]


def append_hota_result(row: dict) -> None:
    write_header = not OUTPUT_CSV.exists()
    with open(OUTPUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HOTA_CSV_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in HOTA_CSV_COLUMNS})


def already_evaluated(eval_split: str) -> set:
    """Return config names that already have HOTA results for this split."""
    if not OUTPUT_CSV.exists():
        return set()
    df = pd.read_csv(OUTPUT_CSV)
    df_split = df[df["split"] == eval_split]
    return set(df_split["config_name"].tolist())


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_sweep_hota_eval(
    config_dirs: List[Path],
    eval_split: str = "test",
    merge_threshold: float = 0.5,
    skip_existing: bool = True,
):
    """
    Evaluate each sweep model on the given split using the full pipeline.

    The splitter is run once and cached; only the merger changes per config.
    """
    print(f"\n{'='*60}")
    print(f"  XGBoost Sweep → HOTA Evaluation")
    print(f"  Split: {eval_split}")
    print(f"  Models to evaluate: {len(config_dirs)}")
    print(f"  Merge threshold: {merge_threshold}")
    print(f"  Output: {OUTPUT_CSV}")
    print(f"{'='*60}\n")

    # Check which configs already have results for THIS split
    done = already_evaluated(eval_split) if skip_existing else set()
    remaining = [d for d in config_dirs if d.name not in done]
    if done:
        print(f"  Skipping {len(config_dirs) - len(remaining)} already-evaluated configs")
    if not remaining:
        print("  All configs already evaluated. Use --force to re-run.")
        return

    # Prepare the runner (loads caches + runs splitter once)
    runner = ExperimentRunner(
        exp_id="sweep_hota",
        split=eval_split,
    )
    runner.prepare()

    results = []
    for i, model_dir in enumerate(remaining, 1):
        config_name = model_dir.name
        print(f"\n--- [{i}/{len(remaining)}] {config_name} ---")

        # Build merger factory for this sweep model
        factory = default_xgboost_factory(
            model_dir=model_dir,
            merge_threshold=merge_threshold,
        )

        # Run through full pipeline
        t0 = time.time()
        result = runner.run(
            run_name=config_name,
            merger_factory=factory,
            config={"config_name": config_name, "merge_threshold": merge_threshold},
            reuse_if_exists=skip_existing,
        )
        dt = time.time() - t0

        # Save incrementally
        append_hota_result({
            "config_name":     config_name,
            "HOTA":            result.metrics.get("HOTA", ""),
            "DetA":            result.metrics.get("DetA", ""),
            "AssA":            result.metrics.get("AssA", ""),
            "IDF1":            result.metrics.get("IDF1", ""),
            "MOTA":            result.metrics.get("MOTA", ""),
            "merge_threshold": merge_threshold,
            "split":           eval_split,
            "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        results.append(result)
        print(f"  HOTA={result.metrics.get('HOTA', '?'):.3f}  "
              f"AssA={result.metrics.get('AssA', '?'):.3f}  "
              f"({dt:.1f}s)")

    # Final summary
    if results:
        summarize_results(results, sort_by="HOTA")

    # Print overall best (including previously evaluated) for THIS split only
    if OUTPUT_CSV.exists():
        df = pd.read_csv(OUTPUT_CSV)
        df_split = df[df["split"] == eval_split]
        if not df_split.empty:
            df_sorted = df_split.sort_values("HOTA", ascending=False)
            print(f"\n{'='*60}")
            print(f"  Top 10 configs by HOTA ({eval_split} split)")
            print(f"{'='*60}")
            print(df_sorted.head(10).to_string(index=False))
            print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate XGBoost sweep models with full-pipeline HOTA"
    )
    parser.add_argument(
        "--top", type=int, default=None,
        help="Only evaluate the top N configs by test F1 (default: all)"
    )
    parser.add_argument(
        "--metric", type=str, default="test_f1",
        help="Metric to rank by when using --top (default: test_f1)"
    )
    parser.add_argument(
        "--configs", nargs="+", default=None,
        help="Specific config names to evaluate (e.g., neg3_minlen10_purity1.0_splitFalse)"
    )
    parser.add_argument(
        "--split", type=str, default="valid",
        choices=["train", "valid", "test"],
        help="Dataset split to evaluate on (default: test)"
    )
    parser.add_argument(
        "--merge-threshold", type=float, default=0.7,
        help="Merge distance threshold for hierarchical clustering (default: 0.5)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-evaluate configs that already have results"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine which configs to evaluate
    if args.configs:
        config_dirs = filter_configs(args.configs)
        print(f"Evaluating {len(config_dirs)} specified configs")
    elif args.top:
        names = top_n_by_metric(args.top, args.metric)
        config_dirs = filter_configs(names)
        print(f"Evaluating top {args.top} configs by {args.metric}")
    else:
        config_dirs = discover_sweep_dirs()
        print(f"Evaluating all {len(config_dirs)} sweep configs")

    if not config_dirs:
        print("No configs to evaluate. Check sweep_results/ directory.")
        return

    run_sweep_hota_eval(
        config_dirs=config_dirs,
        eval_split=args.split,
        merge_threshold=args.merge_threshold,
        skip_existing=not args.force,
    )


if __name__ == "__main__":
    main()
