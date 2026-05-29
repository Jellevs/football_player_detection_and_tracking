#!/usr/bin/env python
"""
sweep_reid_threshold.py — Sweep the ReID cosine-distance threshold for
the rule-based DecisionMerger.

The ReID threshold is the single most influential continuous parameter:
it controls the fallback decision when jersey/team attributes are not
confident enough to decide on their own.

The sweep is run on the *recommended* merger configuration justified by
the gate ablation (see ablate_gates.py): temporal-overlap gate,
jersey-team auto-merge, and ReID fallback are all enabled, while both
attribute conflict blocks (jersey_conflict_block, team_conflict_block)
are disabled. This reflects the configuration that is carried forward
to the end-to-end connector comparison.

Prerequisites:
    Attribute caches must exist (output/cache/cache_attributes_*.pkl).

Usage:
    python sweep_reid_threshold.py                     # validation only
    python sweep_reid_threshold.py --split test        # test set
    python sweep_reid_threshold.py --force             # ignore cached results

Outputs:
    output/decision_merger_reid_sweep_results.csv
"""

import argparse
import copy
import csv
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
# Configuration
# ---------------------------------------------------------------------------

# ReID threshold values to sweep (cosine distance, lower = stricter)
REID_THRESHOLDS = [
    0.2, 0.3, 0.4,
    0.5,  0.6, 0.7, 0.8
]

# Fixed DecisionMerger parameters (not swept)
FIXED_PARAMS = dict(
    jersey_entropy_threshold=0.01,
    min_jersey_predictions=20,
    team_confidence_threshold=0.6,
    team_consistency_threshold=0.9,
)

# Recommended-config gate overrides (justified by gate ablation).
# Both attribute conflict blocks are disabled; auto-merge, ReID fallback,
# and temporal-overlap gate are retained (the latter is always on).
GATE_OVERRIDES = dict(
    jersey_conflict_block=False,
    team_conflict_block=False,
)

# Experiment ID for cache folder naming
EXP_ID = "dm_reid_sweep"

# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

CACHE_DIR = settings.OUTPUT_ROOT / "cache"
EVAL_DIR  = settings.PROJECT_ROOT / "evaluation"
OUTPUT_CSV = settings.OUTPUT_ROOT / "decision_merger_reid_sweep_results.csv"


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


def run_threshold(
    sequences: list,
    split_cache: dict,
    reid_threshold: float,
    split: str,
    force: bool = False,
) -> dict:
    """Merge + evaluate for one ReID threshold value."""
    method_name = f"{EXP_ID}__reid_{reid_threshold:.2f}__{split}"

    # Check for cached result
    summary_path = EVAL_DIR / "SNPT" / method_name / "pedestrian_summary.txt"
    if not force and summary_path.exists():
        print(f"  [CACHED] {method_name}")
        return parse_summary(method_name)

    merger = AblatableDecisionMerger(
        reid_threshold=reid_threshold,
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
    print(f"  reid_threshold={reid_threshold:.2f}  "
          f"HOTA={metrics.get('HOTA', '?'):.3f}  "
          f"AssA={metrics.get('AssA', '?'):.3f}  "
          f"IDF1={metrics.get('IDF1', '?'):.3f}  "
          f"({elapsed:.1f}s)")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Sweep ReID threshold for DecisionMerger")
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"],
                        help="Data split to evaluate on (default: valid)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if cached results exist")
    args = parser.parse_args()

    set_eval_split(args.split)

    sequences = sorted([d.name for d in settings.DATA_ROOT.iterdir() if d.is_dir()])
    print(f"Split: {args.split}")
    print(f"Sequences ({len(sequences)}): {sequences[0]} ... {sequences[-1]}")
    print(f"ReID thresholds: {REID_THRESHOLDS}")
    print(f"Fixed params:    {FIXED_PARAMS}")
    print(f"Gate overrides:  {GATE_OVERRIDES}\n")

    # Split once, reuse across all threshold values
    split_cache = split_all_sequences(sequences)

    results = []
    for reid_threshold in REID_THRESHOLDS:
        metrics = run_threshold(
            sequences, split_cache, reid_threshold, args.split, force=args.force,
        )
        results.append({
            "reid_threshold": reid_threshold,
            "split": args.split,
            **FIXED_PARAMS,
            **metrics,
        })

    # Save CSV
    if results:
        fieldnames = list(results[0].keys())
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {OUTPUT_CSV}")

    # Print ranked summary
    ranked = sorted(results, key=lambda r: r.get("HOTA", -1), reverse=True)

    print(f"\n{'='*80}")
    print(f"ReID Threshold Sweep — ranked by HOTA ({args.split} split)")
    print(f"{'='*80}")
    print(f"  {'Threshold':>10}  {'HOTA':>8}  {'AssA':>8}  {'IDF1':>8}  {'DetA':>8}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in ranked:
        print(f"  {r['reid_threshold']:>10.2f}  "
              f"{r.get('HOTA', 0):>8.3f}  "
              f"{r.get('AssA', 0):>8.3f}  "
              f"{r.get('IDF1', 0):>8.3f}  "
              f"{r.get('DetA', 0):>8.3f}")

    best = ranked[0]
    print(f"\nBest: reid_threshold={best['reid_threshold']:.2f}  "
          f"HOTA={best.get('HOTA', 'N/A')}")


if __name__ == "__main__":
    main()
