"""
step1_per_sequence_decomposition.py — Per-sequence HOTA / DetA / AssA breakdown.

Loads pedestrian_detailed.csv for each configured method, merges on sequence,
and produces:
  1. A wide table with HOTA / DetA / AssA per method per sequence
  2. Delta columns for each method pair
  3. Sequences ranked by the primary gap (e.g. ALERT vs baseline AssA)
  4. A saved CSV and a printed summary

Configuration
-------------
Edit METHODS below:
    key   = display label used in output tables and column names
    value = folder path relative to evaluation/SNPT/
            (can be nested, e.g. "mergers/xgboost/test/merger_jersey_team_bbox")

Set PRIMARY_PAIR to the (label_a, label_b) tuple whose delta drives the ranking.

Run from the repo root:
    python experiments/analysis/step1_per_sequence_decomposition.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))

# ── configuration ─────────────────────────────────────────────────────────────

# key   = display label
# value = folder path relative to evaluation/SNPT/
METHODS: dict[str, str] = {
    "Baseline":  "baseline",
    "XGBoost":   "mergers/xgboost/test/merger_jersey_team_bbox",
    "GTA":       "mergers/gta_split+merge",
}

# The pair whose delta drives the ranking (use display labels from METHODS keys).
# Positive delta = label_b beats label_a.
PRIMARY_PAIR: tuple[str, str] = ("Baseline", "XGBoost")

EVAL_ROOT = REPO / "evaluation" / "SNPT"
OUT_DIR   = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────

# Columns in pedestrian_detailed.csv that represent the AUC aggregate
_COL_MAP = {
    "HOTA": "HOTA___AUC",
    "DetA": "DetA___AUC",
    "AssA": "AssA___AUC",
}


def load_detailed(folder_path: str, label: str) -> pd.DataFrame | None:
    """Load pedestrian_detailed.csv at EVAL_ROOT/folder_path, return slim df or None."""
    path = EVAL_ROOT / folder_path / "pedestrian_detailed.csv"
    if not path.exists():
        print(f"  [WARN] {path} not found — skipping '{label}'")
        return None
    df = pd.read_csv(path, usecols=["seq"] + list(_COL_MAP.values()))
    df = df.rename(columns={v: k for k, v in _COL_MAP.items()})
    # Convert from [0,1] fractions to percentage points (multiply × 100)
    for metric in _COL_MAP:
        df[metric] = df[metric] * 100
    df = df.set_index("seq")
    return df


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Step 1 — Per-sequence HOTA / DetA / AssA decomposition ===\n")

    # Load all methods  (label → DataFrame)
    frames: dict[str, pd.DataFrame] = {}
    for label, folder_path in METHODS.items():
        print(f"Loading '{label}' from {folder_path} …")
        df = load_detailed(folder_path, label)
        if df is not None:
            frames[label] = df
    print()

    if len(frames) < 2:
        print("Need at least 2 methods to compare. Exiting.")
        sys.exit(1)

    # Find common sequences
    all_seqs = set.intersection(*[set(df.index) for df in frames.values()])
    all_seqs = sorted(all_seqs)
    print(f"Common sequences: {len(all_seqs)}\n")

    # ── build wide table ─────────────────────────────────────────────────────
    rows = []
    for seq in all_seqs:
        row: dict = {"seq": seq}
        for label in frames:
            for metric in _COL_MAP:
                row[f"{label}_{metric}"] = frames[label].loc[seq, metric]
        rows.append(row)

    wide = pd.DataFrame(rows).set_index("seq")

    # ── compute deltas between every pair ────────────────────────────────────
    labels = list(frames.keys())
    pairs = [(a, b) for i, a in enumerate(labels) for b in labels[i + 1:]]

    for la, lb in pairs:
        pair_label = f"Δ({lb} − {la})"
        for metric in _COL_MAP:
            wide[f"{pair_label}_{metric}"] = (
                wide[f"{lb}_{metric}"] - wide[f"{la}_{metric}"]
            )

    # ── primary delta for ranking ────────────────────────────────────────────
    la_key, lb_key = PRIMARY_PAIR
    if la_key in frames and lb_key in frames:
        primary_col = f"Δ({lb_key} − {la_key})_AssA"
        wide_sorted = wide.sort_values(primary_col, ascending=False)
    else:
        print(f"[WARN] Primary pair {PRIMARY_PAIR} not both loaded; sorting by seq.")
        wide_sorted = wide.sort_index()
        primary_col = None

    # ── save full table ───────────────────────────────────────────────────────
    out_csv = OUT_DIR / "step1_per_sequence.csv"
    wide_sorted.to_csv(out_csv)
    print(f"Full table saved → {out_csv}\n")

    # ── print summary table ───────────────────────────────────────────────────
    print(f"{'Sequence':<12}", end="")
    for label in frames:
        for metric in ("HOTA", "AssA", "DetA"):
            print(f"  {label[:12]}_{metric}", end="")
    if primary_col:
        print(f"  Δ_AssA({lb_key[:8]}−{la_key[:8]})", end="")
    print()

    print("-" * (12 + len(frames) * 3 * 18 + 30))
    for seq, row in wide_sorted.iterrows():
        print(f"{seq:<12}", end="")
        for label in frames:
            for metric in ("HOTA", "AssA", "DetA"):
                print(f"  {row[f'{label}_{metric}']:>16.3f}", end="")
        if primary_col:
            print(f"  {row[primary_col]:>+22.3f}", end="")
        print()

    # ── print aggregates ──────────────────────────────────────────────────────
    print("\n--- Mean across all sequences ---")
    print(f"{'Method':<30}  {'HOTA':>7}  {'AssA':>7}  {'DetA':>7}")
    for label in frames:
        print(
            f"{label:<30}  "
            f"{wide_sorted[f'{label}_HOTA'].mean():>7.3f}  "
            f"{wide_sorted[f'{label}_AssA'].mean():>7.3f}  "
            f"{wide_sorted[f'{label}_DetA'].mean():>7.3f}"
        )

    # ── print top wins and losses ─────────────────────────────────────────────
    if primary_col:
        n_show = 10
        print(f"\n--- Top {n_show} sequences where {lb_key} GAINS most over {la_key} (AssA) ---")
        for seq, row in wide_sorted.head(n_show).iterrows():
            print(
                f"  {seq}  "
                f"{la_key}: AssA={row[f'{la_key}_AssA']:.3f}  "
                f"{lb_key}: AssA={row[f'{lb_key}_AssA']:.3f}  "
                f"Δ={row[primary_col]:+.3f}"
            )

        print(f"\n--- Top {n_show} sequences where {lb_key} LOSES most vs {la_key} (AssA) ---")
        for seq, row in wide_sorted.tail(n_show).iloc[::-1].iterrows():
            print(
                f"  {seq}  "
                f"{la_key}: AssA={row[f'{la_key}_AssA']:.3f}  "
                f"{lb_key}: AssA={row[f'{lb_key}_AssA']:.3f}  "
                f"Δ={row[primary_col]:+.3f}"
            )

    # ── correlation analysis ──────────────────────────────────────────────────
    if primary_col and la_key in frames:
        baseline_assa = wide_sorted[f"{la_key}_AssA"]
        delta_assa    = wide_sorted[primary_col]
        corr = np.corrcoef(baseline_assa.values, delta_assa.values)[0, 1]
        print(f"\nCorrelation (baseline AssA vs Δ AssA): r = {corr:.3f}")
        if corr < -0.3:
            print("  → Methods tend to help more on sequences where baseline is weaker.")
        elif corr > 0.3:
            print("  → Methods tend to help more on sequences where baseline is already strong.")
        else:
            print("  → No strong correlation with baseline performance level.")

    print(f"\nDone. Full table at: {out_csv}")


if __name__ == "__main__":
    main()
