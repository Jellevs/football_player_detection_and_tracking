"""
bbox_parameter_sweep.py — End-to-end parameter tuning for BboxAnomalySplitter.

Loads cached tracklets once, then for each parameter combination:
  1. Applies the bbox splitter with the given parameters
  2. Saves MOT output
  3. Runs TrackEval (HOTA, CLEAR, Identity)
  4. Collects results

After all combos are evaluated, prints a comparison table and saves CSV.

Additionally computes a sensitivity analysis: for each parameter, measures
how much HOTA changes as that parameter varies while others stay at their
default values (one-at-a-time sensitivity).

Usage:
    python experiments/attributes/bbox_parameter_sweep.py
"""
from __future__ import annotations

import csv
import itertools
import pickle
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Repo setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import settings
from utils.build import build_paths
from utils.evaluation_utils import save_mot_file_for_sn_trackeval
from tracklets.tracklet import Tracklet
from tracklets.xgboost_merger import XGBoostMerger

# ---------------------------------------------------------------------------
# Grid — edit here
# ---------------------------------------------------------------------------


GRID = {
    "std_threshold":      [3.0, 4.0, 5.0],
    "min_spike_velocity": [30.0, 35.0, 40.0],
    "max_spike_duration": [3],
    "lookback_window":    [20],
    "lookahead_window":   [10],
    "min_fragment_length": [20],
}

# Default values (current best config from settings.py)
DEFAULTS = {
    "std_threshold":      3.0,
    "min_spike_velocity": 35.0,
    "max_spike_duration": 3,
    "lookback_window":    20,
    "lookahead_window":   10,
    "min_fragment_length": 20,
}

EVAL_SPLIT   = settings.EVAL_SPLIT
CACHE_DIR    = settings.OUTPUT_ROOT / "cache"
METHOD_NAME  = "_bbox_sweep_tmp"
OUTPUT_DIR   = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Merger config — update these paths after tuning the merger
# ---------------------------------------------------------------------------
MERGER_MODEL_PATH = REPO_ROOT / "weights" / "xgboost_5_neg_ratio_no_main_subj_filt" / "xgboost_merger.json"
MERGER_META_PATH  = REPO_ROOT / "weights" / "xgboost_5_neg_ratio_no_main_subj_filt" / "xgboost_merger_meta.json"
MERGER_THRESHOLD  = 0.5


# ---------------------------------------------------------------------------
# BboxAnomalySplitter (inline copy with overridable params)
# ---------------------------------------------------------------------------

class BboxAnomalySplitterTunable:
    """
    Exact copy of BboxAnomalySplitter with all params passed through __init__
    so the grid search can vary them freely.
    """

    def __init__(self, std_threshold, min_spike_velocity, max_spike_duration,
                 lookback_window, lookahead_window, min_fragment_length):
        self.lookback_window = lookback_window
        self.lookahead_window = lookahead_window
        self.std_threshold = std_threshold
        self.min_spike_velocity = min_spike_velocity
        self.max_spike_duration = max_spike_duration
        self.min_fragment_length = min_fragment_length

    def split_tracklet(self, tracklet, next_available_id):
        if len(tracklet.bboxes) < self.lookback_window + self.lookahead_window:
            return None

        velocities = self.calculate_velocities(tracklet.bboxes)
        spike_indices = self.detect_velocity_spikes(velocities)

        if not spike_indices:
            return None

        return self.create_fragments(tracklet, spike_indices, next_available_id)

    def calculate_velocities(self, bboxes):
        centers = []
        for bbox in bboxes:
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            centers.append([center_x, center_y])
        centers = np.array(centers)
        displacements = np.diff(centers, axis=0)
        return np.linalg.norm(displacements, axis=1)

    def detect_velocity_spikes(self, velocities):
        spike_indices = []
        n = len(velocities)

        for i in range(self.lookback_window, n - self.lookahead_window):
            lookback_start = max(0, i - self.lookback_window)
            baseline_velocities = velocities[lookback_start:i]

            if len(baseline_velocities) == 0:
                continue

            baseline_mean = np.mean(baseline_velocities)
            baseline_std = np.std(baseline_velocities)
            current_velocity = velocities[i]

            if baseline_std > 0:
                z_score = (current_velocity - baseline_mean) / baseline_std
            else:
                z_score = 0

            is_spike = (
                z_score > self.std_threshold and
                current_velocity > self.min_spike_velocity
            )

            if not is_spike:
                continue

            # Check spike duration
            spike_duration = self.measure_spike_duration(
                velocities, i, baseline_mean, baseline_std
            )
            if spike_duration > self.max_spike_duration:
                continue

            # Check velocity returns to normal
            lookahead_start = i + 1
            lookahead_end = min(n, i + 1 + self.lookahead_window)
            lookahead_velocities = velocities[lookahead_start:lookahead_end]

            if len(lookahead_velocities) > 0:
                lookahead_mean = np.mean(lookahead_velocities)
                if lookahead_mean >= baseline_mean * 2.0:
                    continue

            spike_indices.append(i + 1)

        return spike_indices

    def measure_spike_duration(self, velocities, spike_idx, baseline_mean, baseline_std):
        threshold = baseline_mean + self.std_threshold * baseline_std
        duration = 1
        i = spike_idx + 1
        while i < len(velocities) and velocities[i] > threshold:
            duration += 1
            i += 1
            if duration > self.max_spike_duration:
                break
        return duration

    def create_fragments(self, tracklet, spike_indices, next_available_id):
        valid_splits = [idx for idx in spike_indices if 0 < idx < len(tracklet.frames)]
        if not valid_splits:
            return None

        boundaries = [0] + sorted(valid_splits) + [len(tracklet.frames)]

        # Merge small fragments
        merged_boundaries = [boundaries[0]]
        for i in range(1, len(boundaries) - 1):
            fragment_length = boundaries[i] - merged_boundaries[-1]
            if fragment_length >= self.min_fragment_length:
                remaining_length = boundaries[-1] - boundaries[i]
                if remaining_length >= self.min_fragment_length:
                    merged_boundaries.append(boundaries[i])
        merged_boundaries.append(boundaries[-1])

        fragments = []
        current_id = next_available_id
        for i in range(len(merged_boundaries) - 1):
            start = merged_boundaries[i]
            end = merged_boundaries[i + 1]
            fragment = tracklet.extract(start, end)
            fragment.track_id = current_id
            fragment.parent_id = tracklet.parent_id
            fragments.append(fragment)
            current_id += 1

        if len(fragments) <= 1:
            return None
        return fragments


# ---------------------------------------------------------------------------
# Splitting + Evaluation
# ---------------------------------------------------------------------------

def apply_bbox_splitter(tracklets: Dict[int, Any], params: dict) -> Dict[int, Any]:
    """Apply bbox splitter with given params to all tracklets."""
    splitter = BboxAnomalySplitterTunable(**params)

    max_existing_id = max(tracklets.keys()) if tracklets else 0
    next_available_id = max_existing_id + 1

    new_tracklets = {}
    for track_id, tracklet in tracklets.items():
        fragments = splitter.split_tracklet(tracklet, next_available_id)
        if fragments:
            for frag in fragments:
                new_tracklets[frag.track_id] = frag
                next_available_id = max(next_available_id, frag.track_id + 1)
        else:
            new_tracklets[tracklet.track_id] = tracklet

    return new_tracklets


def create_merger() -> XGBoostMerger:
    """Create the XGBoost merger with current best weights."""
    return XGBoostMerger(
        model_path=str(MERGER_MODEL_PATH),
        meta_path=str(MERGER_META_PATH),
        merge_threshold=MERGER_THRESHOLD,
    )


def run_trackeval(method_name: str) -> Optional[Dict[str, float]]:
    """Run TrackEval and parse the summary results."""
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

    # Parse pedestrian_summary.txt
    summary_path = eval_root / "SNPT" / method_name / "pedestrian_summary.txt"
    if not summary_path.exists():
        print(f"  [ERROR] Summary not found: {summary_path}")
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


def load_cached_tracklets(cache_dir: Path, sequences: List[str]) -> Dict[str, Dict]:
    """Load all cached tracklet dictionaries."""
    all_caches = {}
    for seq in sequences:
        cache_path = cache_dir / f"cache_attributes_{seq}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                all_caches[seq] = pickle.load(f)
        else:
            print(f"  [WARN] No cache for {seq}")
    return all_caches


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    params: dict
    n_splits: int
    hota: float
    deta: float
    assa: float
    idf1: float
    mota: float
    idsw: int


def run_full_sweep():
    """Run full grid search over all parameter combinations."""
    # Load sequences
    seqmap = REPO_ROOT / "evaluation" / "seqmaps" / f"SNPT-{EVAL_SPLIT}.txt"
    with seqmap.open("r") as f:
        sequences = [line.strip() for line in f
                     if line.strip() and not line.strip().lower().startswith("name")]

    print(f"Sequences: {len(sequences)}")
    print(f"Loading cached tracklets...")

    all_caches = load_cached_tracklets(CACHE_DIR, sequences)
    print(f"  Loaded {len(all_caches)} sequence caches")

    # Generate all parameter combinations
    keys = list(GRID.keys())
    values = [GRID[k] for k in keys]
    all_combos = list(itertools.product(*values))
    print(f"\nTotal parameter combinations: {len(all_combos)}")

    results: List[SweepResult] = []
    eval_root = settings.PROJECT_ROOT / "evaluation"
    merger = create_merger()

    for combo_idx, combo in enumerate(tqdm(all_combos, desc="Bbox sweep")):
        params = dict(zip(keys, combo))

        # Apply bbox splitter + merger to all sequences and save MOT files
        total_splits = 0
        for seq in sequences:
            if seq not in all_caches:
                continue

            tracklets = all_caches[seq]
            split_tracklets = apply_bbox_splitter(tracklets, params)
            total_splits += len(split_tracklets) - len(tracklets)

            # Merge split tracklets
            merged_tracklets = merger.merge(split_tracklets)

            # Save MOT file
            mot_dir = eval_root / "SNPT" / METHOD_NAME / "data"
            mot_dir.mkdir(parents=True, exist_ok=True)
            save_mot_file_for_sn_trackeval(
                tracklets_dict=merged_tracklets,
                output_path=eval_root,
                sequence_name=seq,
                method_name=METHOD_NAME,
            )

        # Run evaluation
        metrics = run_trackeval(METHOD_NAME)
        if metrics is None:
            continue

        r = SweepResult(
            params=params,
            n_splits=total_splits,
            hota=metrics.get("HOTA", 0.0),
            deta=metrics.get("DetA", 0.0),
            assa=metrics.get("AssA", 0.0),
            idf1=metrics.get("IDF1", 0.0),
            mota=metrics.get("MOTA", 0.0),
            idsw=int(metrics.get("IDSW", 0)),
        )
        results.append(r)

        print(f"  [{combo_idx+1}/{len(all_combos)}] "
              f"std={params['std_threshold']:.1f} vel={params['min_spike_velocity']:.0f} "
              f"dur={params['max_spike_duration']} lb={params['lookback_window']} "
              f"la={params['lookahead_window']} frag={params['min_fragment_length']} | "
              f"HOTA={r.hota:.3f} AssA={r.assa:.3f} IDF1={r.idf1:.3f} "
              f"IDSW={r.idsw} splits={r.n_splits}")

    return results


def run_sensitivity_analysis():
    """
    One-at-a-time sensitivity: vary each parameter while keeping others at default.
    This shows how sensitive HOTA is to each individual parameter.
    """
    seqmap = REPO_ROOT / "evaluation" / "seqmaps" / f"SNPT-{EVAL_SPLIT}.txt"
    with seqmap.open("r") as f:
        sequences = [line.strip() for line in f
                     if line.strip() and not line.strip().lower().startswith("name")]

    all_caches = load_cached_tracklets(CACHE_DIR, sequences)

    eval_root = settings.PROJECT_ROOT / "evaluation"
    merger = create_merger()
    sensitivity_results = {}

    for param_name, param_values in GRID.items():
        print(f"\n--- Sensitivity: {param_name} ---")
        param_results = []

        for val in param_values:
            params = dict(DEFAULTS)
            params[param_name] = val

            # Apply splitter + merger and evaluate
            total_splits = 0
            for seq in sequences:
                if seq not in all_caches:
                    continue
                tracklets = all_caches[seq]
                split_tracklets = apply_bbox_splitter(tracklets, params)
                total_splits += len(split_tracklets) - len(tracklets)

                # Merge split tracklets
                merged_tracklets = merger.merge(split_tracklets)

                mot_dir = eval_root / "SNPT" / METHOD_NAME / "data"
                mot_dir.mkdir(parents=True, exist_ok=True)
                save_mot_file_for_sn_trackeval(
                    tracklets_dict=merged_tracklets,
                    output_path=eval_root,
                    sequence_name=seq,
                    method_name=METHOD_NAME,
                )

            metrics = run_trackeval(METHOD_NAME)
            if metrics:
                hota = metrics.get("HOTA", 0.0)
                param_results.append((val, hota, total_splits))
                print(f"  {param_name}={val} → HOTA={hota:.3f} splits={total_splits}")

        sensitivity_results[param_name] = param_results

    return sensitivity_results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_sweep_csv(results: List[SweepResult], path: Path):
    """Save full grid search results to CSV."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "std_threshold", "min_spike_velocity", "max_spike_duration",
            "lookback_window", "lookahead_window", "min_fragment_length",
            "n_splits", "HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW",
        ])
        for r in sorted(results, key=lambda x: -x.hota):
            w.writerow([
                r.params["std_threshold"],
                r.params["min_spike_velocity"],
                r.params["max_spike_duration"],
                r.params["lookback_window"],
                r.params["lookahead_window"],
                r.params["min_fragment_length"],
                r.n_splits, r.hota, r.deta, r.assa, r.idf1, r.mota, r.idsw,
            ])
    print(f"\nSweep results saved to {path}")


def save_sensitivity_csv(sensitivity_results: dict, path: Path):
    """Save sensitivity analysis results to CSV."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value", "HOTA", "n_splits"])
        for param_name, values in sensitivity_results.items():
            for val, hota, splits in values:
                w.writerow([param_name, val, f"{hota:.4f}", splits])
    print(f"Sensitivity results saved to {path}")


def print_best_results(results: List[SweepResult], top_n: int = 10):
    """Print top N results by HOTA."""
    print(f"\n{'='*90}")
    print(f"TOP {top_n} BBOX PARAMETER CONFIGURATIONS (by HOTA)")
    print(f"{'='*90}")

    sorted_results = sorted(results, key=lambda x: -x.hota)[:top_n]

    header = (f"{'std':>5} {'vel':>5} {'dur':>4} {'lb':>4} {'la':>4} {'frag':>5} "
              f"{'splits':>7} {'HOTA':>7} {'AssA':>7} {'IDF1':>7} {'IDSW':>5}")
    print(header)
    print("-" * len(header))

    for r in sorted_results:
        print(f"{r.params['std_threshold']:>5.1f} "
              f"{r.params['min_spike_velocity']:>5.0f} "
              f"{r.params['max_spike_duration']:>4d} "
              f"{r.params['lookback_window']:>4d} "
              f"{r.params['lookahead_window']:>4d} "
              f"{r.params['min_fragment_length']:>5d} "
              f"{r.n_splits:>7d} "
              f"{r.hota:>7.3f} {r.assa:>7.3f} {r.idf1:>7.3f} {r.idsw:>5d}")


def print_sensitivity(sensitivity_results: dict):
    """Print sensitivity analysis summary."""
    print(f"\n{'='*70}")
    print("SENSITIVITY ANALYSIS (one parameter at a time, others at default)")
    print(f"{'='*70}")

    for param_name, values in sensitivity_results.items():
        if not values:
            continue
        hotas = [h for _, h, _ in values]
        hota_range = max(hotas) - min(hotas)
        print(f"\n  {param_name}:")
        print(f"    {'value':>10} {'HOTA':>8} {'splits':>7}")
        for val, hota, splits in values:
            marker = " *" if val == DEFAULTS[param_name] else ""
            print(f"    {str(val):>10} {hota:>8.3f} {splits:>7d}{marker}")
        print(f"    Range: {hota_range:.3f} (sensitivity: {'HIGH' if hota_range > 0.5 else 'LOW' if hota_range < 0.1 else 'MODERATE'})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bbox parameter sweep")
    parser.add_argument("--mode", choices=["grid", "sensitivity", "both"],
                        default="grid", help="grid = full grid search, sensitivity = one-at-a-time around DEFAULTS, both = grid then sensitivity around best")
    args = parser.parse_args()

    print("=" * 70)
    print("BBOX ANOMALY SPLITTER — PARAMETER TUNING")
    print("=" * 70)
    print(f"Default params: {DEFAULTS}")
    print(f"Grid sizes: {' x '.join(str(len(v)) for v in GRID.values())} = {len(list(itertools.product(*GRID.values())))} combos")
    print()

    best_params = dict(DEFAULTS)

    if args.mode in ("grid", "both"):
        print("\n[GRID SEARCH] Running full grid search...")
        results = run_full_sweep()
        save_sweep_csv(results, OUTPUT_DIR / "bbox_sweep_results.csv")
        print_best_results(results)

        if results:
            best = max(results, key=lambda x: x.hota)
            best_params = dict(best.params)
            print(f"\nBest config from grid: {best_params} -> HOTA={best.hota:.3f}")

    if args.mode in ("sensitivity", "both"):
        if args.mode == "both":
            # Update DEFAULTS to best found params for sensitivity
            for k in DEFAULTS:
                DEFAULTS[k] = best_params[k]
            print(f"\n[SENSITIVITY] Running around best config: {best_params}")
        else:
            print(f"\n[SENSITIVITY] Running around defaults: {DEFAULTS}")

        sensitivity_results = run_sensitivity_analysis()
        save_sensitivity_csv(sensitivity_results, OUTPUT_DIR / "bbox_sensitivity.csv")
        print_sensitivity(sensitivity_results)


if __name__ == "__main__":
    main()
