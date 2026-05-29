#!/usr/bin/env python
"""
gridsearch_reliability.py — Small grid search over the rule-based merger's
attribute reliability thresholds.

The merger's `jersey_entropy_threshold` and `min_jersey_predictions`
control how often a tracklet has a "confident" jersey value, which in
turn controls how often the jersey-team auto-merge fires. Aligning these
with the splitter's stricter values (entropy 0.01, persistence 20) is
methodologically consistent but costs ~0.5 HOTA relative to more
permissive choices, because the auto-merge fires less often.

This script performs a small grid search on the validation set to find
the merger-optimal reliability values, so the rule-based merger is fairly
tuned before the end-to-end connector comparison with XGBoost and the
transformer.

Configuration:
    Gate config:        recommended (no jersey/team conflict blocks).
    Team thresholds:    fixed at confidence 0.6, consistency 0.9.
    Jersey grid:        jersey_entropy_threshold x min_jersey_predictions
                        x reid_threshold.

Prerequisites:
    Attribute caches must exist (output/cache/cache_attributes_*.pkl).

Usage:
    python gridsearch_reliability.py                  # validation only
    python gridsearch_reliability.py --split test     # test split
    python gridsearch_reliability.py --force          # ignore cached runs

Outputs:
    output/decision_merger_reliability_grid.csv
"""

import argparse
import copy
import csv
import itertools
import pickle
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # for ablate_gates import

import settings
from utils.build import build_paths
from utils.config import SplitterConfig
from utils.evaluation_utils import save_mot_file_for_sn_trackeval
from utils.run_evaluation import run_evaluation
from tracklets.split_tracklets import split_tracklets
from ablate_gates import AblatableDecisionMerger


# ---------------------------------------------------------------------------
# Grid definition — edit these lists to widen / narrow the search.
# ---------------------------------------------------------------------------

JERSEY_ENTROPY_GRID    = [0.01, 0.10, 0.20]
MIN_JERSEY_PREDS_GRID  = [5, 10, 20]
REID_THRESHOLD_GRID    = [0.4, 0.5, 0.6]

# Fixed values for everything else (team params are stable across all
# previous experiments and the team conflict block is inert).
FIXED_PARAMS = dict(
    team_confidence_threshold   = 0.6,
    team_consistency_threshold  = 0.9,
)

# Recommended gate config: no attribute conflict blocks.
GATE_OVERRIDES = dict(
    jersey_conflict_block = False,
    team_conflict_block   = False,
)

PRIMARY_METRIC = "HOTA"
EXP_ID = "dm_reliability_grid"

CACHE_DIR  = settings.OUTPUT_ROOT / "cache"
EVAL_DIR   = settings.PROJECT_ROOT / "evaluation"
OUTPUT_CSV = settings.OUTPUT_ROOT / "decision_merger_reliability_grid.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_eval_split(split: str) -> None:
    assert split in ("train", "valid", "test"), f"Unknown split: {split}"
    settings.EVAL_SPLIT = split
    settings.DATA_ROOT = Path(
        rf"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\soccernet-player-tracking\{split}"
    )


def load_attribute_cache(sequence: str) -> dict:
    cache_path = CACHE_DIR / f"cache_attributes_{sequence}.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Attribute cache missing: {cache_path}\n"
            "Run main.py at least once to generate it."
        )
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def make_splitter_config() -> SplitterConfig:
    known_fields = set(SplitterConfig.__dataclass_fields__.keys())
    known = {k: v for k, v in settings.SPLITTER.items() if k in known_fields}
    extra = {k: v for k, v in settings.SPLITTER.items() if k not in known_fields}
    cfg = SplitterConfig(**known)
    for k, v in extra.items():
        setattr(cfg, k, v)
    return cfg


def split_all_sequences(sequences: list) -> dict:
    splitter_cfg = make_splitter_config()
    split_cache = {}
    print("=== Splitting all sequences (one-time) ===")
    for seq in sequences:
        tracklets = load_attribute_cache(seq)
        split_cache[seq] = split_tracklets(tracklets, splitter_cfg)
    print(f"=== Splitting done — {len(split_cache)} sequences cached ===\n")
    return split_cache


def parse_summary(method_name: str) -> dict:
    summary_path = EVAL_DIR / "SNPT" / method_name / "pedestrian_summary.txt"
    if not summary_path.exists():
        return {}
    lines = summary_path.read_text().strip().splitlines()
    if len(lines) < 2:
        return {}
    headers = lines[0].split()
    values  = lines[1].split()
    return {h: float(v) for h, v in zip(headers, values)}


def run_combination(
    sequences: list,
    split_cache: dict,
    jersey_entropy: float,
    min_jersey_preds: int,
    reid_threshold: float,
    split: str,
    force: bool = False,
) -> dict:
    """Merge + evaluate for one (entropy, min_preds, reid_threshold) combo."""
    method_name = (
        f"{EXP_ID}__ent_{jersey_entropy:.3f}__minp_{min_jersey_preds}"
        f"__reid_{reid_threshold:.2f}__{split}"
    )

    # Check cache.
    summary_path = EVAL_DIR / "SNPT" / method_name / "pedestrian_summary.txt"
    if not force and summary_path.exists():
        print(f"  [CACHED] {method_name}")
        return parse_summary(method_name)

    merger = AblatableDecisionMerger(
        jersey_entropy_threshold = jersey_entropy,
        min_jersey_predictions   = min_jersey_preds,
        reid_threshold           = reid_threshold,
        **FIXED_PARAMS,
        **GATE_OVERRIDES,
    )

    t0 = time.time()
    for sequence in sequences:
        paths = build_paths(sequence)
        tracklets = copy.deepcopy(split_cache[sequence])
        merged = merger.merge(tracklets)
        save_mot_file_for_sn_trackeval(
            tracklets_dict=merged,
            output_path=paths.evaluation_path,
            sequence_name=sequence,
            method_name=method_name,
        )

    run_evaluation(method_name, split)
    elapsed = time.time() - t0

    metrics = parse_summary(method_name)
    print(
        f"  ent={jersey_entropy:.3f}  min_preds={min_jersey_preds:>2d}  "
        f"reid={reid_threshold:.2f}  "
        f"HOTA={metrics.get('HOTA', '?'):.3f}  "
        f"AssA={metrics.get('AssA', '?'):.3f}  "
        f"IDF1={metrics.get('IDF1', '?'):.3f}  "
        f"({elapsed:.1f}s)"
    )
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Grid search over rule-based merger reliability thresholds."
    )
    parser.add_argument("--split", default="valid",
                        choices=["train", "valid", "test"],
                        help="Data split to evaluate on (default: valid).")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if cached results exist.")
    args = parser.parse_args()
    args.force = True

    set_eval_split(args.split)

    sequences = sorted([d.name for d in settings.DATA_ROOT.iterdir() if d.is_dir()])
    grid = list(itertools.product(JERSEY_ENTROPY_GRID, MIN_JERSEY_PREDS_GRID, REID_THRESHOLD_GRID))

    print(f"Split:                 {args.split}")
    print(f"Sequences:             {len(sequences)}")
    print(f"Jersey entropy grid:   {JERSEY_ENTROPY_GRID}")
    print(f"Min predictions grid:  {MIN_JERSEY_PREDS_GRID}")
    print(f"ReID threshold grid:   {REID_THRESHOLD_GRID}")
    print(f"Fixed params:          {FIXED_PARAMS}")
    print(f"Gate overrides:        {GATE_OVERRIDES}")
    print(f"Total combinations:    {len(grid)}\n")

    # Split once, reuse across the grid.
    split_cache = split_all_sequences(sequences)

    results = []
    for i, (entropy, min_preds, reid_thr) in enumerate(grid, 1):
        print(f"\n[{i}/{len(grid)}]")
        try:
            metrics = run_combination(
                sequences, split_cache, entropy, min_preds,
                reid_thr, args.split, force=args.force,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            metrics = {}

        results.append({
            "jersey_entropy_threshold": entropy,
            "min_jersey_predictions":   min_preds,
            "reid_threshold":           reid_thr,
            "split":                    args.split,
            **{k: v for k, v in FIXED_PARAMS.items()},
            **metrics,
        })

    # --- Save CSV --------------------------------------------------------
    if results:
        # Union of keys so missing metrics don't break the writer.
        fieldnames = []
        seen = set()
        for r in results:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {OUTPUT_CSV}")

    # --- Ranked summary --------------------------------------------------
    ranked = sorted(results, key=lambda r: r.get(PRIMARY_METRIC, -1), reverse=True)

    print(f"\n{'='*88}")
    print(f"Reliability grid — ranked by {PRIMARY_METRIC} ({args.split} split)")
    print(f"{'='*88}")
    print(f"  {'entropy':>8}  {'min_preds':>10}  {'reid':>6}  {'HOTA':>8}  "
          f"{'AssA':>8}  {'IDF1':>8}  {'DetA':>8}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in ranked:
        print(
            f"  {r['jersey_entropy_threshold']:>8.3f}  "
            f"{r['min_jersey_predictions']:>10d}  "
            f"{r['reid_threshold']:>6.2f}  "
            f"{r.get('HOTA', 0):>8.3f}  "
            f"{r.get('AssA', 0):>8.3f}  "
            f"{r.get('IDF1', 0):>8.3f}  "
            f"{r.get('DetA', 0):>8.3f}"
        )

    if ranked:
        best = ranked[0]
        print(
            f"\nBest: jersey_entropy_threshold={best['jersey_entropy_threshold']:.3f}, "
            f"min_jersey_predictions={best['min_jersey_predictions']}, "
            f"reid_threshold={best['reid_threshold']:.2f}  "
            f"-> HOTA={best.get('HOTA', 'N/A')}"
        )


if __name__ == "__main__":
    main()
