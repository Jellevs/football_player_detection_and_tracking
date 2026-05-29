"""
ablation_hard_constraints.py

Ablation study on the three hard constraints in the XGBoost merger:
  1. Temporal overlap  → distance = 2.0 (never merge co-occurring tracklets)
  2. Jersey conflict   → distance = 2.0 (different confident jersey numbers)
  3. Team conflict     → distance = 2.0 (different consistent team labels)

Tests all 8 combinations (each on/off) to show individual and combined impact.
No retraining is needed — the same model is used, only the constraint flags change.

Usage:
    python -m experiments.tracklet_merger.xgboost.ablation_hard_constraints \
        --config neg3_minlen10_purity0.8_splitTrue \
        --split valid

    # With custom merge threshold (e.g., from threshold sweep):
    python -m experiments.tracklet_merger.xgboost.ablation_hard_constraints \
        --config neg3_minlen10_purity0.8_splitTrue \
        --split valid --merge-threshold 0.45

Outputs:
    experiments/tracklet_merger/xgboost/hard_constraint_ablation_results.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from itertools import product
from pathlib import Path

import pandas as pd

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
OUTPUT_CSV = Path(__file__).resolve().parent / "hard_constraint_ablation_results.csv"


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "config_name", "merge_threshold",
    "temporal_enabled", "jersey_enabled", "team_enabled",
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


def already_evaluated(eval_split: str, config_name: str) -> set:
    """Return (temporal, jersey, team) tuples already evaluated."""
    if not OUTPUT_CSV.exists():
        return set()
    df = pd.read_csv(OUTPUT_CSV)
    df = df[(df["split"] == eval_split) & (df["config_name"] == config_name)]
    return set(zip(df["temporal_enabled"], df["jersey_enabled"], df["team_enabled"]))


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
# Main
# ---------------------------------------------------------------------------

def run_ablation(
    config_name: str,
    eval_split: str = "valid",
    merge_threshold: float = 0.5,
    skip_existing: bool = True,
):
    model_dir = resolve_model_dir(config_name)

    # All 8 combinations: (temporal_enabled, jersey_enabled, team_enabled)
    combos = list(product([True, False], repeat=3))
    # Sort so baseline (all True) is first
    combos.sort(key=lambda x: (-sum(x), x), reverse=False)

    print(f"\n{'='*60}")
    print(f"  Hard Constraint Ablation")
    print(f"  Model: {model_dir.name}")
    print(f"  Split: {eval_split}")
    print(f"  Merge threshold: {merge_threshold}")
    print(f"  Combinations: {len(combos)}")
    print(f"{'='*60}\n")

    done = already_evaluated(eval_split, model_dir.name) if skip_existing else set()
    remaining = [c for c in combos if c not in done]
    if done:
        print(f"  Skipping {len(combos) - len(remaining)} already-evaluated combos")
    if not remaining:
        print("  All combinations already evaluated. Use --force to re-run.")
        _print_summary(model_dir.name, eval_split)
        return

    # Prepare runner once
    runner = ExperimentRunner(
        exp_id="hc_ablation",
        split=eval_split,
    )
    runner.prepare()

    for i, (temp_en, jer_en, team_en) in enumerate(remaining, 1):
        label = _combo_label(temp_en, jer_en, team_en)
        print(f"\n--- [{i}/{len(remaining)}] {label} ---")

        factory = default_xgboost_factory(
            model_dir=model_dir,
            merge_threshold=merge_threshold,
            disable_temporal_constraint=not temp_en,
            disable_jersey_constraint=not jer_en,
            disable_team_constraint=not team_en,
        )

        run_name = f"{model_dir.name}_{label}"
        t0 = time.time()
        result = runner.run(
            run_name=run_name,
            merger_factory=factory,
            config={
                "config_name": model_dir.name,
                "temporal": temp_en,
                "jersey": jer_en,
                "team": team_en,
            },
            reuse_if_exists=skip_existing,
        )
        dt = time.time() - t0

        append_result({
            "config_name":      model_dir.name,
            "merge_threshold":  merge_threshold,
            "temporal_enabled": temp_en,
            "jersey_enabled":   jer_en,
            "team_enabled":     team_en,
            "HOTA":             result.metrics.get("HOTA", ""),
            "DetA":             result.metrics.get("DetA", ""),
            "AssA":             result.metrics.get("AssA", ""),
            "IDF1":             result.metrics.get("IDF1", ""),
            "MOTA":             result.metrics.get("MOTA", ""),
            "split":            eval_split,
            "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        print(f"  HOTA={result.metrics.get('HOTA', '?'):.3f}  "
              f"AssA={result.metrics.get('AssA', '?'):.3f}  "
              f"({dt:.1f}s)")

    _print_summary(model_dir.name, eval_split)


def _combo_label(temp: bool, jer: bool, team: bool) -> str:
    parts = []
    parts.append("T" if temp else "t")
    parts.append("J" if jer else "j")
    parts.append("M" if team else "m")   # M for teaM (T is taken)
    return "".join(parts)


def _print_summary(config_name: str, eval_split: str):
    if not OUTPUT_CSV.exists():
        return
    df = pd.read_csv(OUTPUT_CSV)
    config_clean = config_name.replace("_BEST", "")
    mask = (df["split"] == eval_split) & (
        df["config_name"].str.replace("_BEST", "") == config_clean
    )
    df_f = df[mask].sort_values("HOTA", ascending=False)
    if df_f.empty:
        return

    print(f"\n{'='*70}")
    print(f"  Hard Constraint Ablation Results ({eval_split} split)")
    print(f"  T=temporal  J=jersey  M=team  (uppercase=enabled)")
    print(f"{'='*70}")

    for _, r in df_f.iterrows():
        label = _combo_label(r["temporal_enabled"], r["jersey_enabled"], r["team_enabled"])
        delta = ""
        baseline = df_f[
            (df_f["temporal_enabled"] == True) &
            (df_f["jersey_enabled"] == True) &
            (df_f["team_enabled"] == True)
        ]
        if len(baseline):
            d = r["HOTA"] - baseline.iloc[0]["HOTA"]
            delta = f"  ({d:+.3f})" if abs(d) > 0.001 else "  (baseline)"
        print(f"  {label}  HOTA={r['HOTA']:.3f}  AssA={r['AssA']:.3f}  "
              f"DetA={r['DetA']:.3f}  IDF1={r['IDF1']:.3f}{delta}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation study on hard constraints in XGBoost merger"
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
        "--force", action="store_true",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_ablation(
        config_name=args.config,
        eval_split=args.split,
        merge_threshold=args.merge_threshold,
        skip_existing=not args.force,
    )


if __name__ == "__main__":
    main()
