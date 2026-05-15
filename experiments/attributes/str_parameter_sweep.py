"""
str_parameter_sweep.py — End-to-end parameter tuning for TemporalReIDSplitter (STR).

Loads cached tracklets once, then for each parameter combination:
  1. Applies the STR splitter with the given parameters
  2. Saves MOT output
  3. Runs TrackEval (HOTA, CLEAR, Identity)
  4. Collects results

Additionally computes a one-at-a-time sensitivity analysis.

Usage:
    python experiments/attributes/str_parameter_sweep.py
"""
from __future__ import annotations

import csv
import itertools
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Repo setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import settings
from utils.evaluation_utils import save_mot_file_for_sn_trackeval
from tracklets.tracklet import Tracklet
from tracklets.xgboost_merger import XGBoostMerger

# ---------------------------------------------------------------------------
# Grid — edit here
# ---------------------------------------------------------------------------
GRID = {
    "reid_threshold":      [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
    "min_gap_frames":      [2, 3, 5, 8, 10, 15],
    "min_segment_frames":  [3, 5, 8, 10],
    "n_samples":           [10, 15, 20, 30],
}

# Default values (current config from settings.py)
DEFAULTS = {
    "reid_threshold":     0.15,
    "min_gap_frames":     5,
    "min_segment_frames": 5,
    "n_samples":          20,
}

EVAL_SPLIT   = settings.EVAL_SPLIT
CACHE_DIR    = settings.OUTPUT_ROOT / "cache"
METHOD_NAME  = "_str_sweep_tmp"
OUTPUT_DIR   = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Merger config — update these paths after tuning the merger
# ---------------------------------------------------------------------------
MERGER_MODEL_PATH = REPO_ROOT / "weights" / "xgboost_5_neg_ratio_no_main_subj_filt" / "xgboost_merger.json"
MERGER_META_PATH  = REPO_ROOT / "weights" / "xgboost_5_neg_ratio_no_main_subj_filt" / "xgboost_merger_meta.json"
MERGER_THRESHOLD  = 0.5


def create_merger() -> XGBoostMerger:
    """Create the XGBoost merger with current best weights."""
    return XGBoostMerger(
        model_path=str(MERGER_MODEL_PATH),
        meta_path=str(MERGER_META_PATH),
        merge_threshold=MERGER_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# STR Splitter (inline with overridable params, no debug prints)
# ---------------------------------------------------------------------------

class TemporalReIDSplitterTunable:
    """TemporalReIDSplitter with all params passed through __init__."""

    def __init__(self, reid_threshold, min_gap_frames, min_segment_frames, n_samples):
        self.reid_threshold = reid_threshold
        self.min_gap_frames = min_gap_frames
        self.min_segment_frames = min_segment_frames
        self.n_samples = n_samples

    def split_tracklet(self, tracklet, next_available_id):
        frames = np.array(tracklet.frames)

        if len(frames) < 2 * self.min_segment_frames:
            return None

        if not tracklet.embeddings or len(tracklet.embeddings) != len(frames):
            return None

        embs = np.stack(tracklet.embeddings)

        # Find temporal gaps
        diffs = np.diff(frames)
        gap_indices = np.where(diffs >= self.min_gap_frames)[0]

        if len(gap_indices) == 0:
            return None

        # Check each gap
        split_points = []
        for gap_idx in gap_indices:
            before_start = max(0, gap_idx + 1 - self.n_samples)
            before_embs = embs[before_start:gap_idx + 1]

            after_end = min(len(embs), gap_idx + 1 + self.n_samples)
            after_embs = embs[gap_idx + 1:after_end]

            if len(before_embs) == 0 or len(after_embs) == 0:
                continue

            mean_before = before_embs.mean(axis=0, keepdims=True)
            mean_after = after_embs.mean(axis=0, keepdims=True)
            cos_dist = cdist(mean_before, mean_after, metric='cosine')[0, 0]

            if cos_dist > self.reid_threshold:
                split_points.append(gap_idx + 1)

        if not split_points:
            return None

        # Create fragments
        boundaries = [0] + split_points + [len(frames)]
        fragments = []
        current_id = next_available_id

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1] - 1

            if (end - start + 1) < self.min_segment_frames:
                continue

            fragment = tracklet.extract(start, end)
            fragment.track_id = current_id
            fragment.parent_id = tracklet.parent_id
            fragments.append(fragment)
            current_id += 1

        if len(fragments) <= 1:
            return None

        return fragments

    def split_all(self, tracklets):
        max_existing_id = max(tracklets.keys()) if tracklets else 0
        next_available_id = max_existing_id + 1

        new_tracklets = {}
        for track_id, tracklet in tracklets.items():
            fragments = self.split_tracklet(tracklet, next_available_id)
            if fragments:
                for fragment in fragments:
                    new_tracklets[fragment.track_id] = fragment
                    next_available_id = max(next_available_id, fragment.track_id + 1)
            else:
                new_tracklets[tracklet.track_id] = tracklet

        return new_tracklets


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

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
# Sensitivity analysis
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


def run_sensitivity_analysis():
    """
    One-at-a-time sensitivity: vary each parameter while keeping others at default.
    """
    seqmap = REPO_ROOT / "evaluation" / "seqmaps" / f"SNPT-{EVAL_SPLIT}.txt"
    with seqmap.open("r") as f:
        sequences = [line.strip() for line in f
                     if line.strip() and not line.strip().lower().startswith("name")]

    print(f"Sequences: {len(sequences)}")
    print(f"Loading cached tracklets...")
    all_caches = load_cached_tracklets(CACHE_DIR, sequences)
    print(f"  Loaded {len(all_caches)} sequence caches")

    eval_root = settings.PROJECT_ROOT / "evaluation"
    merger = create_merger()
    sensitivity_results = {}

    for param_name, param_values in GRID.items():
        print(f"\n--- Sensitivity: {param_name} ---")
        param_results = []

        for val in param_values:
            params = dict(DEFAULTS)
            params[param_name] = val

            # Apply STR splitter + merger to all sequences
            splitter = TemporalReIDSplitterTunable(**params)
            total_splits = 0

            for seq in sequences:
                if seq not in all_caches:
                    continue
                tracklets = all_caches[seq]
                split_tracklets = splitter.split_all(tracklets)
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
            if metrics:
                hota = metrics.get("HOTA", 0.0)
                param_results.append((val, hota, total_splits))
                print(f"  {param_name}={val} → HOTA={hota:.3f} splits={total_splits}")

        sensitivity_results[param_name] = param_results

    return sensitivity_results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_sensitivity_csv(sensitivity_results: dict, path: Path):
    """Save sensitivity analysis results to CSV."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value", "HOTA", "n_splits"])
        for param_name, values in sensitivity_results.items():
            for val, hota, splits in values:
                w.writerow([param_name, val, f"{hota:.4f}", splits])
    print(f"Sensitivity results saved to {path}")


def print_sensitivity(sensitivity_results: dict):
    """Print sensitivity analysis summary."""
    print(f"\n{'='*70}")
    print("STR SPLITTER — SENSITIVITY ANALYSIS")
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
    print("=" * 70)
    print("STR (TEMPORAL REID) SPLITTER — PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"Default params: {DEFAULTS}")
    print()

    sensitivity_results = run_sensitivity_analysis()
    save_sensitivity_csv(sensitivity_results, OUTPUT_DIR / "str_sensitivity.csv")
    print_sensitivity(sensitivity_results)


if __name__ == "__main__":
    main()
