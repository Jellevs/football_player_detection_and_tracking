"""
trajectory_parameter_sweep.py — End-to-end parameter tuning for TrajectorySplitter.

Loads cached tracklets once, then for each parameter combination:
  1. Applies the trajectory splitter with the given parameters
  2. Saves MOT output
  3. Runs TrackEval (HOTA, CLEAR, Identity)
  4. Collects results

Additionally computes a one-at-a-time sensitivity analysis.

Note: The trajectory splitter is cross-tracklet (requires all tracklets
at once to detect proximity events), so each evaluation requires running
the full splitter on all tracklets together.

Usage:
    python experiments/attributes/trajectory_parameter_sweep.py
"""
from __future__ import annotations

import csv
import itertools
import pickle
import subprocess
import sys
from collections import defaultdict
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
from utils.evaluation_utils import save_mot_file_for_sn_trackeval
from tracklets.tracklet import Tracklet
from tracklets.xgboost_merger import XGBoostMerger

# ---------------------------------------------------------------------------
# Grid — edit here
# ---------------------------------------------------------------------------
GRID = {
    "proximity_distance":         [30, 40, 50, 60, 80],
    "min_velocity_change":        [15.0, 20.0, 25.0, 30.0, 40.0],
    "direction_change_threshold": [90, 100, 110, 120, 140],
    "min_overlap_frames":         [2, 3, 5],
    "swap_similarity_threshold":  [0.85, 0.90, 0.95],
    "min_fragment_length":        [5, 10, 15],
}

# Default values (current config)
DEFAULTS = {
    "proximity_distance":         50,
    "min_velocity_change":        30.0,
    "direction_change_threshold": 120,
    "min_overlap_frames":         3,
    "swap_similarity_threshold":  0.95,
    "min_fragment_length":        10,
}

EVAL_SPLIT   = settings.EVAL_SPLIT
CACHE_DIR    = settings.OUTPUT_ROOT / "cache"
METHOD_NAME  = "_traj_sweep_tmp"
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
# TrajectorySplitter (inline with overridable params, no debug prints)
# ---------------------------------------------------------------------------

class TrajectorySplitterTunable:
    """TrajectorySplitter with all params passed through __init__."""

    def __init__(self, proximity_distance, min_velocity_change,
                 direction_change_threshold, min_overlap_frames,
                 swap_similarity_threshold, min_fragment_length):
        self.proximity_distance = proximity_distance
        self.min_overlap_frames = min_overlap_frames
        self.velocity_window = 5
        self.min_velocity_change = min_velocity_change
        self.direction_change_threshold = direction_change_threshold
        self.swap_similarity_threshold = swap_similarity_threshold
        self.min_fragment_length = min_fragment_length

    def split_all_tracklets(self, tracklets_dict):
        """Analyze all tracklets together and split based on proximity + trajectory."""
        frame_index = self._build_frame_index(tracklets_dict)
        split_decisions = self._detect_all_splits(tracklets_dict, frame_index)
        return self._apply_splits(tracklets_dict, split_decisions)

    def _build_frame_index(self, tracklets_dict):
        frame_index = defaultdict(list)
        for track_id, tracklet in tracklets_dict.items():
            for local_idx, frame_idx in enumerate(tracklet.frames):
                bbox = tracklet.bboxes[local_idx]
                center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                frame_index[frame_idx].append({
                    'track_id': track_id,
                    'local_idx': local_idx,
                    'bbox': bbox,
                    'center': center,
                })
        return frame_index

    def _detect_all_splits(self, tracklets_dict, frame_index):
        split_decisions = defaultdict(list)
        processed_pairs = set()

        for frame_idx, detections in frame_index.items():
            for i in range(len(detections)):
                for j in range(i + 1, len(detections)):
                    det_A = detections[i]
                    det_B = detections[j]
                    track_A = det_A['track_id']
                    track_B = det_B['track_id']

                    pair_key = (min(track_A, track_B), max(track_A, track_B), frame_idx)
                    if pair_key in processed_pairs:
                        continue
                    processed_pairs.add(pair_key)

                    if self._are_bboxes_close(det_A, det_B):
                        split_info = self._analyze_proximity_event(
                            tracklets_dict[track_A], tracklets_dict[track_B],
                            det_A['local_idx'], det_B['local_idx'], frame_idx
                        )
                        if split_info:
                            split_decisions[track_A].append(split_info['split_idx_A'] + 1)
                            split_decisions[track_B].append(split_info['split_idx_B'] + 1)

        for track_id in split_decisions:
            split_decisions[track_id] = sorted(set(split_decisions[track_id]))

        return split_decisions

    def _are_bboxes_close(self, det_A, det_B):
        bbox_A, bbox_B = det_A['bbox'], det_B['bbox']
        # IoU check
        x1 = max(bbox_A[0], bbox_B[0])
        y1 = max(bbox_A[1], bbox_B[1])
        x2 = min(bbox_A[2], bbox_B[2])
        y2 = min(bbox_A[3], bbox_B[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_A = (bbox_A[2] - bbox_A[0]) * (bbox_A[3] - bbox_A[1])
        area_B = (bbox_B[2] - bbox_B[0]) * (bbox_B[3] - bbox_B[1])
        union = area_A + area_B - intersection
        iou = intersection / (union + 1e-6)
        if iou > 0.1:
            return True
        # Center distance check
        distance = np.linalg.norm(det_A['center'] - det_B['center'])
        return distance < self.proximity_distance

    def _analyze_proximity_event(self, tracklet_A, tracklet_B,
                                  local_idx_A, local_idx_B, frame_idx):
        vel_A_before = self._get_velocity(tracklet_A, local_idx_A, before=True)
        vel_A_after = self._get_velocity(tracklet_A, local_idx_A, before=False)
        vel_B_before = self._get_velocity(tracklet_B, local_idx_B, before=True)
        vel_B_after = self._get_velocity(tracklet_B, local_idx_B, before=False)

        if any(v is None for v in [vel_A_before, vel_A_after, vel_B_before, vel_B_after]):
            return None

        speed_change_A = abs(np.linalg.norm(vel_A_after) - np.linalg.norm(vel_A_before))
        speed_change_B = abs(np.linalg.norm(vel_B_after) - np.linalg.norm(vel_B_before))

        dir_change_A = self._direction_change(vel_A_before, vel_A_after)
        dir_change_B = self._direction_change(vel_B_before, vel_B_after)

        swap_score = self._detect_velocity_swap(
            vel_A_before, vel_A_after, vel_B_before, vel_B_after
        )

        # Scoring
        score = 0
        if swap_score > self.swap_similarity_threshold:
            score += 2
        if speed_change_A > self.min_velocity_change and speed_change_B > self.min_velocity_change:
            score += 1
        if dir_change_A > self.direction_change_threshold and dir_change_B > self.direction_change_threshold:
            score += 4

        if score >= 5:
            return {
                'split_idx_A': local_idx_A,
                'split_idx_B': local_idx_B,
                'frame_idx': frame_idx,
                'score': score,
            }
        return None

    def _get_velocity(self, tracklet, local_idx, before=True):
        window = self.velocity_window
        if before:
            start = max(0, local_idx - window)
            end = local_idx
        else:
            start = local_idx + 1
            end = min(len(tracklet.frames), local_idx + 1 + window)

        if end - start < 2:
            return None

        centers = []
        for i in range(start, end):
            bbox = tracklet.bboxes[i]
            centers.append([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        centers = np.array(centers)
        velocities = np.diff(centers, axis=0)
        return np.mean(velocities, axis=0)

    def _direction_change(self, vel_before, vel_after):
        v1 = vel_before / (np.linalg.norm(vel_before) + 1e-6)
        v2 = vel_after / (np.linalg.norm(vel_after) + 1e-6)
        cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def _detect_velocity_swap(self, vel_A_before, vel_A_after, vel_B_before, vel_B_after):
        def cosine_sim(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-6 or nb < 1e-6:
                return 0.0
            return (np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0) + 1.0) / 2.0

        sim_A_to_B = cosine_sim(vel_A_after, vel_B_before)
        sim_B_to_A = cosine_sim(vel_B_after, vel_A_before)
        return (sim_A_to_B + sim_B_to_A) / 2.0

    def _apply_splits(self, tracklets_dict, split_decisions):
        if not split_decisions:
            return tracklets_dict

        max_existing_id = max(tracklets_dict.keys()) if tracklets_dict else 0
        next_available_id = max_existing_id + 1

        new_tracklets = {}
        for track_id, tracklet in tracklets_dict.items():
            if track_id in split_decisions:
                fragments = self._create_fragments(
                    tracklet, split_decisions[track_id], next_available_id
                )
                if fragments and len(fragments) > 1:
                    for fragment in fragments:
                        new_tracklets[fragment.track_id] = fragment
                        next_available_id = max(next_available_id, fragment.track_id + 1)
                else:
                    new_tracklets[tracklet.track_id] = tracklet
            else:
                new_tracklets[tracklet.track_id] = tracklet

        return new_tracklets

    def _create_fragments(self, tracklet, split_indices, next_available_id):
        valid_splits = [idx for idx in split_indices if 0 < idx < len(tracklet.frames)]
        if not valid_splits:
            return None

        boundaries = [0] + sorted(valid_splits) + [len(tracklet.frames)]

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

            # Apply trajectory splitter + merger to all sequences
            splitter = TrajectorySplitterTunable(**params)
            total_splits = 0

            for seq in sequences:
                if seq not in all_caches:
                    continue
                tracklets = all_caches[seq]
                split_tracklets = splitter.split_all_tracklets(tracklets)
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
    print("TRAJECTORY SPLITTER — SENSITIVITY ANALYSIS")
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
    print("TRAJECTORY SPLITTER — PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"Default params: {DEFAULTS}")
    print()

    sensitivity_results = run_sensitivity_analysis()
    save_sensitivity_csv(sensitivity_results, OUTPUT_DIR / "trajectory_sensitivity.csv")
    print_sensitivity(sensitivity_results)


if __name__ == "__main__":
    main()
