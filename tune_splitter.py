#!/usr/bin/env python
"""
tune_splitter.py — Grid search over UnifiedSplitter + SimpleTrackletMerger hyperparameters.

Prerequisites:
    Run main.py at least once to populate the attribute caches
    (output/cache/cache_attributes_*.pkl). This script skips tracking and
    attribute prediction entirely — it only tunes the split/merge stage.

Usage:
    python tune_splitter.py

Results are ranked by PRIMARY_METRIC and saved to:
    output/grid_search_results.csv
"""

import copy
import csv
import itertools
import pickle
import sys
from pathlib import Path

import settings
from utils.build import build_paths
from utils.config import SplitterConfig
from utils.evaluation_utils import save_mot_file_for_sn_trackeval
from utils.run_evaluation import run_evaluation
from tracklets.split_tracklets import split_tracklets
from tracklets.simple_tracklet_merger import SimpleTrackletMerger

# ---------------------------------------------------------------------------
# Hyperparameter grid — edit these lists to control the search space.
# Each entry is a list of values to try. To fix a parameter, use a 1-element
# list, e.g. jersey_lookahead = [100].
# ---------------------------------------------------------------------------

SPLITTER_GRID = dict(
    # --- Jersey signal (searched) ---
    jersey_entropy_threshold     = [0.01, 0.05, 0.1],  # max entropy to count a prediction as reliable
    jersey_min_persistence       = [5, 10, 20],         # min reliable obs collected before split fires

    # --- Team signal (searched) ---
    team_confidence_threshold    = [0.5, 0.6, 0.7],    # min team-classifier confidence to use a prediction

    # --- Fixed at report / well-motivated values ---
    jersey_lookahead             = [100],               # W_j: max frames searched for reliable jersey obs
    jersey_min_persistence_ratio = [0.8],               # ρ_j: fraction of P_j window that must match
    team_lookahead               = [100],               # W_t: frames in team persistence window
    team_min_persistence_ratio   = [0.8],               # ρ_t: fraction of window that must match
    min_fragment_length          = [20],                # L: discard fragments shorter than this (frames)
)

MERGER_GRID = dict(
    reid_threshold               = [0.3, 0.4, 0.5],    # max cosine distance for ReID-based merge

    # Fixed — mirrors the splitter's jersey_entropy_threshold (set in run_combination)
    team_consistency_threshold   = [0.9],               # min team-mode fraction to count as "consistent"
)

# Metric from pedestrian_summary.txt to rank by (higher is better).
PRIMARY_METRIC = "HOTA"

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


def make_splitter_config(params: dict) -> SplitterConfig:
    """
    Build a SplitterConfig, forwarding only known dataclass fields.
    Any extra params (e.g. min_fragment_length) are monkey-patched onto the
    resulting object so UnifiedSplitter can still access them via self.config.
    """
    known_fields = set(SplitterConfig.__dataclass_fields__.keys())
    known  = {k: v for k, v in params.items() if k in known_fields}
    extra  = {k: v for k, v in params.items() if k not in known_fields}

    cfg = SplitterConfig(**known)
    for k, v in extra.items():
        setattr(cfg, k, v)
    return cfg


def parse_summary(method_name: str) -> dict:
    """Read pedestrian_summary.txt and return a dict of metric → float."""
    summary_path = EVAL_DIR / "SNPT" / method_name / "pedestrian_summary.txt"
    if not summary_path.exists():
        return {}
    lines = summary_path.read_text().strip().splitlines()
    if len(lines) < 2:
        return {}
    headers = lines[0].split()
    values  = lines[1].split()
    return {h: float(v) for h, v in zip(headers, values)}


def run_combination(sequences, splitter_params, merger_params, combo_id):
    """Run split → merge → save MOT → evaluate for one parameter combination."""
    method_name  = f"grid_{combo_id:04d}"
    splitter_cfg = make_splitter_config(splitter_params)
    merger = SimpleTrackletMerger(
        reid_threshold=merger_params["reid_threshold"],
        jersey_entropy_threshold=splitter_params["jersey_entropy_threshold"],
        team_consistency_threshold=merger_params["team_consistency_threshold"],
    )

    for sequence in sequences:
        paths = build_paths(sequence)

        # Deep copy so we don't mutate the cached objects across iterations.
        tracklets = copy.deepcopy(load_attribute_cache(sequence))

        split  = split_tracklets(tracklets, splitter_cfg)
        merged = merger.merge(split)

        save_mot_file_for_sn_trackeval(
            tracklets_dict=merged,
            output_path=paths.evaluation_path,
            sequence_name=sequence,
            method_name=method_name,
        )

    run_evaluation(method_name, settings.EVAL_SPLIT)
    return method_name, parse_summary(method_name)


def build_combinations():
    s_keys   = list(SPLITTER_GRID.keys())
    s_values = list(SPLITTER_GRID.values())
    m_keys   = list(MERGER_GRID.keys())
    m_values = list(MERGER_GRID.values())

    combos = []
    for s_combo, m_combo in itertools.product(
        itertools.product(*s_values),
        itertools.product(*m_values),
    ):
        combos.append(
            (dict(zip(s_keys, s_combo)), dict(zip(m_keys, m_combo)))
        )
    return combos


def main():
    sequences = sorted([d.name for d in settings.DATA_ROOT.iterdir() if d.is_dir()])
    print(f"Sequences ({len(sequences)}): {sequences[0]} … {sequences[-1]}")

    combos  = build_combinations()
    n_total = len(combos)

    n_splitter = 1
    for v in SPLITTER_GRID.values():
        n_splitter *= len(v)
    n_merger = 1
    for v in MERGER_GRID.values():
        n_merger *= len(v)

    print(f"Grid: {n_splitter} splitter × {n_merger} merger = {n_total} combinations")

    results = []

    for combo_id, (splitter_params, merger_params) in enumerate(combos):
        print(f"\n[{combo_id + 1}/{n_total}]")
        print(f"  splitter: {splitter_params}")
        print(f"  merger:   {merger_params}")

        try:
            method_name, metrics = run_combination(
                sequences, splitter_params, merger_params, combo_id
            )
            score = metrics.get(PRIMARY_METRIC, float("nan"))
            print(f"  → {PRIMARY_METRIC} = {score:.4f}")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            metrics = {}
            method_name = f"grid_{combo_id:04d}"

        results.append({
            "combo_id":   combo_id,
            "method_name": method_name,
            **splitter_params,
            **merger_params,
            **metrics,
        })

    # Sort best → worst by primary metric
    results.sort(key=lambda r: r.get(PRIMARY_METRIC, -1.0), reverse=True)

    # Save CSV
    out_path = settings.OUTPUT_ROOT / "grid_search_results.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {out_path}")

    # Print top-10
    s_keys = list(SPLITTER_GRID.keys())
    m_keys = list(MERGER_GRID.keys())

    print(f"\n{'=' * 80}")
    print(f"Top-10 by {PRIMARY_METRIC}:")
    print(f"{'=' * 80}")

    for rank, r in enumerate(results[:10], 1):
        splitter_str = ", ".join(f"{k}={r.get(k)}" for k in s_keys)
        merger_str   = ", ".join(f"{k}={r.get(k)}" for k in m_keys)
        hota  = r.get("HOTA",  "?")
        assa  = r.get("AssA",  "?")
        deta  = r.get("DetA",  "?")
        idf1  = r.get("IDF1",  "?")
        print(
            f"  #{rank:2d}  HOTA={hota:.4f}  AssA={assa:.4f}  DetA={deta:.4f}  IDF1={idf1:.4f}"
            f"\n        {splitter_str}"
            f"\n        {merger_str}"
        )

    if results:
        best = results[0]
        print(f"\nBest parameter set (combo {best['combo_id']:04d}):")
        for k in s_keys + m_keys:
            print(f"  {k} = {best.get(k)}")
        print(f"  → {PRIMARY_METRIC} = {best.get(PRIMARY_METRIC, 'N/A')}")


if __name__ == "__main__":
    main()
