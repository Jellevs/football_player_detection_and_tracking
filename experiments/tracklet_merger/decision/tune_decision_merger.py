#!/usr/bin/env python
"""
tune_decision_merger.py — Grid search over DecisionMerger hyperparameters.

Prerequisites:
    Run main.py at least once to populate the attribute caches
    (output/cache/cache_attributes_*.pkl). This script skips tracking and
    attribute prediction entirely — it only tunes the merge stage.

    The splitter is run once with the settings from settings.py and cached
    in memory, so only the merger parameters are varied across combinations.

Usage:
    python tune_decision_merger.py

Results are ranked by PRIMARY_METRIC and saved to:
    output/decision_merger_grid_search_results.csv
"""

import copy
import csv
import itertools
import pickle
from pathlib import Path



import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

import settings
from utils.build import build_paths
from utils.config import SplitterConfig
from utils.evaluation_utils import save_mot_file_for_sn_trackeval
from utils.run_evaluation import run_evaluation
from tracklets.split_tracklets import split_tracklets
from tracklets.decision_merger import DecisionMerger

# ---------------------------------------------------------------------------
# Hyperparameter grid — edit these lists to control the search space.
# Each entry is a list of values to try.  To fix a parameter, use a
# 1-element list.
# ---------------------------------------------------------------------------

MERGER_GRID = dict(
    # ReID cosine distance threshold (fallback when attributes are unclear)
    reid_threshold              = [0.3, 0.4, 0.5, 0.6, 0.7],

    # Jersey: max Shannon entropy to count a per-frame prediction as reliable
    jersey_entropy_threshold    = [0.01, 0.05],

    # Jersey: minimum reliable observations needed to consider jersey confident
    min_jersey_predictions      = [10, 20],

    # Team: minimum per-frame classifier confidence to use a prediction
    team_confidence_threshold   = [0.6],

    # Team: minimum fraction of confident predictions that must agree
    team_consistency_threshold  = [0.9],
)

# Metric from pedestrian_summary.txt to rank by (higher is better).
PRIMARY_METRIC = "HOTA"

# Which split to evaluate on.  Use "test" for final results, or temporarily
# switch to "train" for faster iteration during development.
EVAL_SPLIT = settings.EVAL_SPLIT

# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

CACHE_DIR = settings.OUTPUT_ROOT / "cache"
EVAL_DIR  = settings.PROJECT_ROOT / "evaluation"


def load_attribute_cache(sequence: str) -> dict:
    """Load cached attribute tracklets for one sequence."""
    cache_path = CACHE_DIR / f"cache_attributes_{sequence}.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Attribute cache missing: {cache_path}\n"
            "Run main.py at least once to generate it."
        )
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def make_splitter_config() -> SplitterConfig:
    """Build a SplitterConfig from settings.py (used once for splitting)."""
    known_fields = set(SplitterConfig.__dataclass_fields__.keys())
    known = {k: v for k, v in settings.SPLITTER.items() if k in known_fields}
    extra = {k: v for k, v in settings.SPLITTER.items() if k not in known_fields}

    cfg = SplitterConfig(**known)
    for k, v in extra.items():
        setattr(cfg, k, v)
    return cfg


def split_all_sequences(sequences: list) -> dict:
    """
    Run the splitter once on every sequence and return a dict of
    {sequence: split_tracklets}.  This is cached in memory so the
    (expensive) splitting step is not repeated for every merger combo.
    """
    splitter_cfg = make_splitter_config()
    split_cache = {}

    print("=== Splitting all sequences (one-time) ===")
    for seq in sequences:
        tracklets = load_attribute_cache(seq)
        split_cache[seq] = split_tracklets(tracklets, splitter_cfg)
    print(f"=== Splitting done — {len(split_cache)} sequences cached ===\n")
    return split_cache


def parse_summary(method_name: str) -> dict:
    """Read pedestrian_summary.txt and return a dict of metric -> float."""
    summary_path = EVAL_DIR / "SNPT" / method_name / "pedestrian_summary.txt"
    if not summary_path.exists():
        return {}
    lines = summary_path.read_text().strip().splitlines()
    if len(lines) < 2:
        return {}
    headers = lines[0].split()
    values  = lines[1].split()
    return {h: float(v) for h, v in zip(headers, values)}


def run_combination(sequences, split_cache, merger_params, combo_id):
    """Merge + save MOT + evaluate for one parameter combination."""
    method_name = f"dm_grid_{combo_id:04d}"

    merger = DecisionMerger(
        reid_threshold=merger_params["reid_threshold"],
        jersey_entropy_threshold=merger_params["jersey_entropy_threshold"],
        min_jersey_predictions=merger_params["min_jersey_predictions"],
        team_confidence_threshold=merger_params["team_confidence_threshold"],
        team_consistency_threshold=merger_params["team_consistency_threshold"],
    )

    for sequence in sequences:
        paths = build_paths(sequence)

        # Deep copy so we don't mutate the cached split tracklets.
        tracklets = copy.deepcopy(split_cache[sequence])
        merged = merger.merge(tracklets)

        save_mot_file_for_sn_trackeval(
            tracklets_dict=merged,
            output_path=paths.evaluation_path,
            sequence_name=sequence,
            method_name=method_name,
        )

    run_evaluation(method_name, EVAL_SPLIT)
    return method_name, parse_summary(method_name)


def build_combinations():
    keys   = list(MERGER_GRID.keys())
    values = list(MERGER_GRID.values())

    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def main():
    sequences = sorted([d.name for d in settings.DATA_ROOT.iterdir() if d.is_dir()])
    print(f"Sequences ({len(sequences)}): {sequences[0]} ... {sequences[-1]}")

    # Split once, reuse across all merger combos.
    split_cache = split_all_sequences(sequences)

    combos  = build_combinations()
    n_total = len(combos)

    sizes = " x ".join(f"{len(v)}" for v in MERGER_GRID.values())
    print(f"Grid: {sizes} = {n_total} combinations\n")

    results = []

    for combo_id, merger_params in enumerate(combos):
        print(f"\n[{combo_id + 1}/{n_total}]")
        print(f"  params: {merger_params}")

        try:
            method_name, metrics = run_combination(
                sequences, split_cache, merger_params, combo_id
            )
            score = metrics.get(PRIMARY_METRIC, float("nan"))
            print(f"  -> {PRIMARY_METRIC} = {score:.4f}")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            metrics = {}
            method_name = f"dm_grid_{combo_id:04d}"

        results.append({
            "combo_id":    combo_id,
            "method_name": method_name,
            **merger_params,
            **metrics,
        })

    # Sort best -> worst by primary metric.
    results.sort(key=lambda r: r.get(PRIMARY_METRIC, -1.0), reverse=True)

    # Save CSV.
    out_path = settings.OUTPUT_ROOT / "decision_merger_grid_search_results.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {out_path}")

    # Print top 10.
    param_keys = list(MERGER_GRID.keys())

    print(f"\n{'=' * 90}")
    print(f"Top 10 by {PRIMARY_METRIC}:")
    print(f"{'=' * 90}")

    for rank, r in enumerate(results[:10], 1):
        param_str = ", ".join(f"{k}={r.get(k)}" for k in param_keys)
        hota = r.get("HOTA",  "?")
        assa = r.get("AssA",  "?")
        deta = r.get("DetA",  "?")
        idf1 = r.get("IDF1",  "?")
        idsw = r.get("IDSW",  "?")
        print(
            f"  #{rank:2d}  HOTA={hota:.4f}  AssA={assa:.4f}  DetA={deta:.4f}"
            f"  IDF1={idf1:.4f}  IDSW={idsw}"
            f"\n        {param_str}"
        )

    if results:
        best = results[0]
        print(f"\nBest parameter set (combo {best['combo_id']:04d}):")
        for k in param_keys:
            print(f"  {k} = {best.get(k)}")
        print(f"  -> {PRIMARY_METRIC} = {best.get(PRIMARY_METRIC, 'N/A')}")


if __name__ == "__main__":
    main()
