"""
bbox_gridsearch_combined.py — Grid search for BboxAnomalySplitter parameters
in the COMBINED splitter pipeline (STR + Jersey + Team + Bbox).

Loads cached attribute tracklets once, then for each bbox parameter combination:
  1. Runs the full splitting pipeline (all splitters active)
  2. Merges with XGBoost
  3. Evaluates HOTA via TrackEval
  4. Collects results

After all combos, prints top results, saves CSV, and optionally runs
sensitivity analysis around the best found configuration.

Usage:
    python experiments/attributes/bbox_gridsearch_combined.py
    python experiments/attributes/bbox_gridsearch_combined.py --mode sensitivity
    python experiments/attributes/bbox_gridsearch_combined.py --mode both
"""
from __future__ import annotations

import argparse
import csv
import itertools
import pickle
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Repo setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import settings
from utils.config import SplitterConfig
from utils.evaluation_utils import save_mot_file_for_sn_trackeval
from tracklets.split_tracklets import split_tracklets
from tracklets.xgboost_merger import XGBoostMerger

# ---------------------------------------------------------------------------
# Fixed splitter params (non-bbox) — current best from settings.py
# ---------------------------------------------------------------------------
FIXED_SPLITTER_PARAMS = dict(
    # Jersey
    jersey_min_persistence=20,
    jersey_lookahead=150,
    jersey_min_persistence_ratio=0.9,
    jersey_entropy_threshold=0.02,
    # Team
    team_min_persistence=10,
    team_min_persistence_ratio=0.9,
    team_lookahead=100,
    team_confidence_threshold=0.7,
    # STR
    temporal_reid_min_gap_frames=5,
    temporal_reid_threshold=0.15,
    temporal_reid_min_segment_frames=5,
    temporal_reid_n_samples=20,
    # Trajectory (not used in default SIGNALS, but keep for completeness)
    proximity_distance=50.0,
    min_overlap_frames=3,
    velocity_window=5,
    min_velocity_change=30.0,
    direction_change_threshold=120.0,
    swap_similarity_threshold=0.95,
)

# ---------------------------------------------------------------------------
# Bbox grid — focused on the parameters that showed any signal
# ---------------------------------------------------------------------------
BBOX_GRID = {
    "std_threshold":       [3.0, 4.0, 5.0, 6.0],
    "min_spike_velocity":  [20.0, 30.0, 35.0, 50.0],
    "max_spike_duration":  [2, 3, 4],
    "lookback_window":     [10, 15, 20, 30],
    "lookahead_window":    [5, 10, 15],
    "min_fragment_length": [10, 20],
}

# Current defaults (for sensitivity analysis baseline)
BBOX_DEFAULTS = {
    "std_threshold":       3.0,
    "min_spike_velocity":  35.0,
    "max_spike_duration":  3,
    "lookback_window":     20,
    "lookahead_window":    10,
    "min_fragment_length": 10,
}

# Which splitters to activate (must match split_tracklets.py SIGNALS format)
# We temporarily override the module-level SIGNALS for the grid search.
ACTIVE_SIGNALS = ["temporal_reid", "jersey", "team", "bbox"]

# ---------------------------------------------------------------------------
# Merger config
# ---------------------------------------------------------------------------
MERGER_MODEL_PATH = REPO_ROOT / "weights" / "xgboost_5_neg_ratio_no_main_subj_filt" / "xgboost_merger.json"
MERGER_META_PATH  = REPO_ROOT / "weights" / "xgboost_5_neg_ratio_no_main_subj_filt" / "xgboost_merger_meta.json"
MERGER_THRESHOLD  = 0.5

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EVAL_SPLIT  = settings.EVAL_SPLIT
CACHE_DIR   = settings.OUTPUT_ROOT / "cache"
METHOD_NAME = "_bbox_grid_combined"
OUTPUT_DIR  = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_cached_tracklets(cache_dir: Path, sequences: List[str]) -> Dict[str, Dict]:
    """Load all cached attribute tracklet dictionaries."""
    all_caches = {}
    for seq in sequences:
        cache_path = cache_dir / f"cache_attributes_{seq}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                all_caches[seq] = pickle.load(f)
        else:
            print(f"  [WARN] No cache for {seq}")
    return all_caches


def get_sequences() -> List[str]:
    """Read sequence list from seqmap file."""
    seqmap = REPO_ROOT / "evaluation" / "seqmaps" / f"SNPT-{EVAL_SPLIT}.txt"
    with seqmap.open("r") as f:
        return [line.strip() for line in f
                if line.strip() and not line.strip().lower().startswith("name")]


def build_splitter_config(bbox_params: dict) -> SplitterConfig:
    """Build a SplitterConfig combining fixed params with bbox grid params."""
    all_params = {**FIXED_SPLITTER_PARAMS, **bbox_params}
    return SplitterConfig(**all_params)


def create_merger() -> XGBoostMerger:
    return XGBoostMerger(
        model_path=str(MERGER_MODEL_PATH),
        meta_path=str(MERGER_META_PATH),
        merge_threshold=MERGER_THRESHOLD,
    )


def run_trackeval(method_name: str) -> Optional[Dict[str, float]]:
    """Run TrackEval and parse summary results."""
    eval_root = settings.PROJECT_ROOT / "evaluation"
    cmd = [
        sys.executable,
        str(settings.PROJECT_ROOT / "sn-trackeval" / "scripts" / "run_mot_challenge.py"),
        "--BENCHMARK", "SNMOT",
        "--SPLIT_TO_EVAL", f"SNMOT-{EVAL_SPLIT}",
        "--GT_FOLDER", str(settings.DATA_ROOT),
        "--TRACKERS_FOLDER", str(eval_root / "SNPT"),
        "--TRACKERS_TO_EVAL", method_name,
        "--SEQMAP_FILE", str(eval_root / "seqmaps" / f"SNPT-{EVAL_SPLIT}.txt"),
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--DO_PREPROC", "False",
        "--USE_PARALLEL", "False",
        "--TRACKER_SUB_FOLDER", "data",
        "--SKIP_SPLIT_FOL", "True",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] TrackEval failed: {result.stderr[-500:]}")
        return None

    summary_path = eval_root / "SNPT" / method_name / "pedestrian_summary.txt"
    if not summary_path.exists():
        return None

    with open(summary_path, "r") as f:
        lines = f.readlines()

    if len(lines) < 2:
        return None

    headers = lines[0].strip().split()
    values = lines[1].strip().split()
    metrics = {}
    for h, v in zip(headers, values):
        try:
            metrics[h] = float(v)
        except ValueError:
            pass
    return metrics


def run_pipeline_for_params(bbox_params: dict, all_caches: Dict[str, Dict],
                            sequences: List[str], merger: XGBoostMerger) -> int:
    """
    Run the full split + merge pipeline for one parameter configuration.
    Returns total number of splits (fragments created beyond original tracklets).
    """
    import tracklets.split_tracklets as st_module

    # Override the SIGNALS in the split_tracklets module
    original_signals = st_module.SIGNALS
    st_module.SIGNALS = ACTIVE_SIGNALS

    splitter_cfg = build_splitter_config(bbox_params)
    eval_root = settings.PROJECT_ROOT / "evaluation"
    total_splits = 0

    try:
        for seq in sequences:
            if seq not in all_caches:
                continue

            # Deep copy tracklets to avoid mutation across runs
            import copy
            tracklets = copy.deepcopy(all_caches[seq])
            n_before = len(tracklets)

            # Run full splitting pipeline (STR + jersey + team + bbox)
            splitted = split_tracklets(tracklets, splitter_cfg)
            total_splits += len(splitted) - n_before

            # Merge with XGBoost
            merged = merger.merge(splitted)

            # Save MOT file
            save_mot_file_for_sn_trackeval(
                tracklets_dict=merged,
                output_path=eval_root,
                sequence_name=seq,
                method_name=METHOD_NAME,
            )
    finally:
        # Restore original SIGNALS
        st_module.SIGNALS = original_signals

    return total_splits


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GridResult:
    params: dict
    n_splits: int
    hota: float
    assa: float
    deta: float
    idf1: float
    mota: float
    idsw: int


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def run_grid_search():
    """Full grid search over bbox parameters in the combined pipeline."""
    sequences = get_sequences()
    print(f"Sequences: {len(sequences)}")
    print("Loading cached tracklets...")

    all_caches = load_cached_tracklets(CACHE_DIR, sequences)
    print(f"  Loaded {len(all_caches)} sequence caches")

    # Generate all combinations
    keys = list(BBOX_GRID.keys())
    values = [BBOX_GRID[k] for k in keys]
    all_combos = list(itertools.product(*values))
    print(f"\nTotal parameter combinations: {len(all_combos)}")
    print(f"Active signals: {ACTIVE_SIGNALS}")

    results: List[GridResult] = []
    merger = create_merger()

    for combo_idx, combo in enumerate(tqdm(all_combos, desc="Bbox grid search")):
        params = dict(zip(keys, combo))

        total_splits = run_pipeline_for_params(params, all_caches, sequences, merger)

        metrics = run_trackeval(METHOD_NAME)
        if metrics is None:
            continue

        r = GridResult(
            params=params,
            n_splits=total_splits,
            hota=metrics.get("HOTA", 0.0),
            assa=metrics.get("AssA", 0.0),
            deta=metrics.get("DetA", 0.0),
            idf1=metrics.get("IDF1", 0.0),
            mota=metrics.get("MOTA", 0.0),
            idsw=int(metrics.get("IDSW", 0)),
        )
        results.append(r)

        is_default = all(params[k] == BBOX_DEFAULTS[k] for k in keys)
        marker = " *** DEFAULT ***" if is_default else ""

        tqdm.write(
            f"  [{combo_idx+1}/{len(all_combos)}] "
            f"std={params['std_threshold']:.1f} vel={params['min_spike_velocity']:.0f} "
            f"dur={params['max_spike_duration']} lb={params['lookback_window']} "
            f"la={params['lookahead_window']} frag={params['min_fragment_length']} | "
            f"HOTA={r.hota:.3f} AssA={r.assa:.3f} IDF1={r.idf1:.3f} "
            f"IDSW={r.idsw} splits={r.n_splits}{marker}"
        )

    return results


# ---------------------------------------------------------------------------
# Sensitivity analysis (around best config)
# ---------------------------------------------------------------------------

def run_sensitivity(best_params: Optional[dict] = None):
    """
    One-at-a-time sensitivity around the best (or default) configuration.
    Run this AFTER grid search to verify robustness of the chosen config.
    """
    base = best_params or dict(BBOX_DEFAULTS)
    print(f"\nSensitivity baseline: {base}")

    sequences = get_sequences()
    all_caches = load_cached_tracklets(CACHE_DIR, sequences)
    merger = create_merger()

    sensitivity_results = {}

    for param_name, param_values in BBOX_GRID.items():
        print(f"\n--- Sensitivity: {param_name} ---")
        param_results = []

        for val in param_values:
            params = dict(base)
            params[param_name] = val

            total_splits = run_pipeline_for_params(params, all_caches, sequences, merger)

            metrics = run_trackeval(METHOD_NAME)
            if metrics:
                hota = metrics.get("HOTA", 0.0)
                param_results.append((val, hota, total_splits))
                marker = " *" if val == base[param_name] else ""
                print(f"  {param_name}={val} -> HOTA={hota:.3f} splits={total_splits}{marker}")

        sensitivity_results[param_name] = param_results

    return sensitivity_results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_grid_csv(results: List[GridResult], path: Path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "std_threshold", "min_spike_velocity", "max_spike_duration",
            "lookback_window", "lookahead_window", "min_fragment_length",
            "n_splits", "HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW",
        ])
        for r in sorted(results, key=lambda x: -x.hota):
            w.writerow([
                r.params["std_threshold"], r.params["min_spike_velocity"],
                r.params["max_spike_duration"], r.params["lookback_window"],
                r.params["lookahead_window"], r.params["min_fragment_length"],
                r.n_splits, f"{r.hota:.4f}", f"{r.deta:.4f}", f"{r.assa:.4f}",
                f"{r.idf1:.4f}", f"{r.mota:.4f}", r.idsw,
            ])
    print(f"\nGrid results saved to {path}")


def save_sensitivity_csv(sensitivity_results: dict, path: Path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value", "HOTA", "n_splits"])
        for param_name, values in sensitivity_results.items():
            for val, hota, splits in values:
                w.writerow([param_name, val, f"{hota:.4f}", splits])
    print(f"Sensitivity results saved to {path}")


def print_top_results(results: List[GridResult], top_n: int = 20):
    print(f"\n{'='*100}")
    print(f"TOP {top_n} BBOX CONFIGURATIONS — COMBINED PIPELINE (by HOTA)")
    print(f"{'='*100}")

    sorted_results = sorted(results, key=lambda x: -x.hota)[:top_n]

    header = (f"{'#':>3} {'std':>5} {'vel':>5} {'dur':>4} {'lb':>4} {'la':>4} {'frag':>5} "
              f"{'splits':>7} {'HOTA':>8} {'AssA':>8} {'IDF1':>8} {'IDSW':>5}")
    print(header)
    print("-" * len(header))

    for i, r in enumerate(sorted_results, 1):
        print(f"{i:>3} {r.params['std_threshold']:>5.1f} "
              f"{r.params['min_spike_velocity']:>5.0f} "
              f"{r.params['max_spike_duration']:>4d} "
              f"{r.params['lookback_window']:>4d} "
              f"{r.params['lookahead_window']:>4d} "
              f"{r.params['min_fragment_length']:>5d} "
              f"{r.n_splits:>7d} "
              f"{r.hota:>8.3f} {r.assa:>8.3f} {r.idf1:>8.3f} {r.idsw:>5d}")


def print_sensitivity(sensitivity_results: dict, base_params: dict):
    print(f"\n{'='*70}")
    print("SENSITIVITY ANALYSIS (around best config, one parameter at a time)")
    print(f"{'='*70}")

    for param_name, values in sensitivity_results.items():
        if not values:
            continue
        hotas = [h for _, h, _ in values]
        hota_range = max(hotas) - min(hotas)
        sensitivity = "HIGH" if hota_range > 1.0 else "LOW" if hota_range < 0.1 else "MODERATE"

        print(f"\n  {param_name}:")
        print(f"    {'value':>10} {'HOTA':>8} {'splits':>7}")
        for val, hota, splits in values:
            marker = " <-- best" if val == base_params[param_name] else ""
            print(f"    {str(val):>10} {hota:>8.3f} {splits:>7d}{marker}")
        print(f"    Range: {hota_range:.3f} (sensitivity: {sensitivity})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mode = "grid" #sensitivity

    print("=" * 70)
    print("BBOX PARAMETER GRID SEARCH — COMBINED PIPELINE")
    print(f"Active signals: {ACTIVE_SIGNALS}")
    print(f"Eval split: {EVAL_SPLIT}")
    print("=" * 70)

    best_params = dict(BBOX_DEFAULTS)

    if mode in ("grid", "both"):
        print("\n[GRID SEARCH] Running...")
        results = run_grid_search()
        save_grid_csv(results, OUTPUT_DIR / "bbox_gridsearch_combined.csv")
        print_top_results(results)

        # Update best_params from grid search results
        if results:
            best = max(results, key=lambda r: r.hota)
            best_params = dict(best.params)
            print(f"\nBest config: {best_params} -> HOTA={best.hota:.3f}")

    if mode in ("sensitivity", "both"):
        print(f"\n[SENSITIVITY] Running around: {best_params}")
        sens_results = run_sensitivity(best_params)
        save_sensitivity_csv(sens_results, OUTPUT_DIR / "bbox_sensitivity_combined.csv")
        print_sensitivity(sens_results, best_params)


if __name__ == "__main__":
    main()
