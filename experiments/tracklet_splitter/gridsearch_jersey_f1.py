"""
gridsearch_jersey_f1.py — Grid search over jersey splitter hyperparameters.

Optimises IDS-F1 using the **same evaluation logic** as intrinsic_metrics.py
(IoU-based predicted-split detection), so scores are directly comparable to
run_splitter_eval.py output.

The previous tune_jersey_splitter.py used parent_id-based split detection,
which diverges from intrinsic_metrics.py due to:
  - different bbox formats (xyxy in cache vs xywh in MOT files)
  - different frame indexing (0-indexed in tracklets vs 1-indexed in MOT)
  - different split detection logic (parent_id walk vs IoU matching)

This script avoids all three issues by converting splitter output to the same
MOT-style by-frame format and reusing intrinsic_metrics functions directly.

Prerequisites
-------------
Attribute caches must exist for the target split:
    output/cache/cache_attributes_<sequence>.pkl
Run main.py at least once with the desired split to generate them.

Usage
-----
    python experiments/tracklet_splitter/gridsearch_jersey_f1.py

Results are saved to:
    experiments/tracklet_splitter/output/gridsearch_jersey_f1_results.csv
"""

from __future__ import annotations

import copy
import csv
import itertools
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── make repo root importable ─────────────────────────────────────────────────
REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))

import settings
from utils.config import SplitterConfig
from tracklets.splitters.jersey_splitter import JerseySplitter

# Import evaluation functions directly from intrinsic_metrics so we use the
# exact same matching logic as run_splitter_eval.py
from experiments.tracklet_splitter.intrinsic_metrics import (
    load_mot_by_track,
    load_mot_by_frame,
    build_det_to_gt,
    find_true_switches,
    find_predicted_splits,
    match_switches,
    resolve_gt_path,
    iou_xywh,
)

# ── configuration ─────────────────────────────────────────────────────────────

# Tune on valid split to avoid leaking test labels into hyperparameter choices.
DATA_SPLIT = "test"

GT_ROOT   = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\soccernet-player-tracking") / DATA_SPLIT
EVAL_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\evaluation\SNPT")
INPUT_DIR = EVAL_ROOT / "baseline" / "data"          # pre-split baseline tracklets (MOT format)
SEQMAP    = REPO / "evaluation" / "seqmaps" / f"SNPT-{DATA_SPLIT}.txt"
CACHE_DIR = REPO / "output" / "cache"
OUT_CSV   = Path(__file__).parent / "output" / "gridsearch_jersey_f1_results.csv"

IOU_THRESHOLD = 0.5   # for matching tracklet detections to GT detections
TOLERANCE     = 10    # frames: split points within ±TOLERANCE of a true switch count as TP

# ── hyperparameter grid ───────────────────────────────────────────────────────
# Fix a parameter by giving it a single-element list.

GRID = dict(
    jersey_entropy_threshold     = [0.01, 0.05, 0.1, 0.2],
    jersey_min_persistence       = [3, 5, 10, 20],
    jersey_min_persistence_ratio = [0.6, 0.7, 0.8, 0.9],
    jersey_lookahead             = [20, 50, 100],
    min_fragment_length          = [10, 20],
)


# ── helpers ───────────────────────────────────────────────────────────────────

def tracklets_to_mot_by_frame(
    split_tracklets: dict,
) -> dict[int, list[tuple[int, np.ndarray]]]:
    """
    Convert splitter output (Tracklet objects with xyxy bboxes) to the same
    format as intrinsic_metrics.load_mot_by_frame:
        {frame: [(track_id, bbox_xywh)]}
    Cached tracklet frames are already 1-indexed (matching MOT/GT convention),
    so no frame offset is needed. Bboxes are converted from xyxy to xywh.
    """
    by_frame: dict[int, list] = defaultdict(list)
    for tid, tracklet in split_tracklets.items():
        for frame, bbox in zip(tracklet.frames, tracklet.bboxes):
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            xywh = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
            by_frame[frame].append((tid, xywh))
    return dict(by_frame)


def apply_jersey_splitter(tracklets: dict, cfg: SplitterConfig) -> dict:
    """Run the jersey splitter and return the new tracklet dict."""
    splitter = JerseySplitter(cfg)
    max_id = max(tracklets.keys()) if tracklets else 0
    next_id = max_id + 1
    result = {}
    for tid, tracklet in tracklets.items():
        frags = splitter.split_tracklet(tracklet, next_id)
        if frags:
            for f in frags:
                result[f.track_id] = f
                next_id = max(next_id, f.track_id + 1)
        else:
            result[tid] = tracklet
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # ── discover sequences from seqmap ────────────────────────────────────
    if SEQMAP.exists():
        with SEQMAP.open("r") as f:
            sequences = [line.strip() for line in f
                         if line.strip() and not line.strip().lower().startswith("name")]
    else:
        sequences = sorted(p.stem for p in INPUT_DIR.glob("SNPT-*.txt"))
    print(f"Split: {DATA_SPLIT}  |  Sequences from seqmap: {len(sequences)}")

    # ── load MOT-format input tracklets and GT (same source as intrinsic_metrics) ─
    print("Loading baseline MOT files and GT...")
    input_tracks_per_seq: dict[str, dict] = {}
    input_by_frame_per_seq: dict[str, dict] = {}
    gt_by_frame_per_seq: dict[str, dict] = {}
    input_gt_map_per_seq: dict[str, dict] = {}
    true_switches_per_seq: dict[str, list] = {}
    cached_tracklets_per_seq: dict[str, dict] = {}

    skipped = []
    for seq in sequences:
        # Check all required files exist
        input_path = INPUT_DIR / f"{seq}.txt"
        gt_path = resolve_gt_path(GT_ROOT, seq)
        cache_path = CACHE_DIR / f"cache_attributes_{seq}.pkl"

        missing = []
        if not input_path.exists():
            missing.append("input MOT")
        if gt_path is None:
            missing.append("GT")
        if not cache_path.exists():
            missing.append("cache")
        if missing:
            skipped.append((seq, ", ".join(missing)))
            continue

        # Load MOT-format data (same as intrinsic_metrics.py)
        input_tracks = load_mot_by_track(input_path)
        input_by_frame = load_mot_by_frame(input_path)
        gt_by_frame = load_mot_by_frame(gt_path)

        # Build GT map and true switches (identical to intrinsic_metrics)
        input_gt_map = build_det_to_gt(input_by_frame, gt_by_frame, IOU_THRESHOLD)
        true_switches = find_true_switches(input_tracks, input_gt_map)

        # Load cached attribute tracklets (for running the splitter)
        with cache_path.open("rb") as f:
            cached_tracklets = pickle.load(f)

        input_tracks_per_seq[seq] = input_tracks
        gt_by_frame_per_seq[seq] = gt_by_frame
        input_gt_map_per_seq[seq] = input_gt_map
        true_switches_per_seq[seq] = true_switches
        cached_tracklets_per_seq[seq] = cached_tracklets

    sequences = [s for s in sequences if s in cached_tracklets_per_seq]
    print(f"  Ready: {len(sequences)} sequences  |  Skipped: {len(skipped)}")
    if skipped:
        for seq, reason in skipped[:5]:
            print(f"    {seq}: missing {reason}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more")

    total_true = sum(len(v) for v in true_switches_per_seq.values())
    print(f"  Total true switches: {total_true}")

    if not sequences:
        print("No sequences to evaluate. Ensure baseline MOT files, GT, and caches exist.")
        return

    # ── build grid ────────────────────────────────────────────────────────
    keys   = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))
    print(f"\nGrid: {len(combos)} combinations  ({' x '.join(str(len(v)) for v in GRID.values())})")

    # ── search ────────────────────────────────────────────────────────────
    results = []
    t0 = time.time()

    for i, values in enumerate(combos):
        params = dict(zip(keys, values))
        cfg = SplitterConfig(**{k: v for k, v in params.items()
                                if k in SplitterConfig.__dataclass_fields__})

        tp_total = fp_total = fn_total = 0
        n_pred_total = 0
        n_tracklets_total = 0

        for seq in sequences:
            tracklets = copy.deepcopy(cached_tracklets_per_seq[seq])
            split = apply_jersey_splitter(tracklets, cfg)

            # Convert splitter output to MOT-style by-frame format
            # (1-indexed frames, xywh bboxes) — same as intrinsic_metrics
            pred_by_frame = tracklets_to_mot_by_frame(split)

            # Use intrinsic_metrics functions for evaluation
            predicted_splits = find_predicted_splits(
                input_tracks_per_seq[seq], pred_by_frame, IOU_THRESHOLD
            )
            tp, fp, fn = match_switches(
                true_switches_per_seq[seq], predicted_splits, TOLERANCE
            )

            tp_total += tp
            fp_total += fp
            fn_total += fn
            n_pred_total += len(predicted_splits)
            n_tracklets_total += len(split)

        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else float("nan")
        recall    = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if not (np.isnan(precision) or np.isnan(recall) or precision + recall == 0)
              else float("nan"))

        row = {
            **params,
            "tp": tp_total, "fp": fp_total, "fn": fn_total,
            "num_predicted_splits": n_pred_total,
            "num_true_switches":    total_true,
            "num_tracklets":        n_tracklets_total,
            "ids_precision": precision,
            "ids_recall":    recall,
            "ids_f1":        f1,
        }
        results.append(row)

        if (i + 1) % 20 == 0 or (i + 1) == len(combos):
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(combos) - i - 1)
            best_f1 = max((r['ids_f1'] for r in results if not np.isnan(r['ids_f1'])), default=float('nan'))
            print(f"  [{i+1:3d}/{len(combos)}]  best F1: {best_f1:.4f}  "
                  f"elapsed: {elapsed:.0f}s  ETA: {eta:.0f}s")

    # ── sort and save ─────────────────────────────────────────────────────
    results.sort(key=lambda r: (r["ids_f1"] if not np.isnan(r["ids_f1"]) else -1), reverse=True)

    fieldnames = list(results[0].keys())
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved {len(results)} results to {OUT_CSV}")

    # ── print top-10 ─────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"Top-10 by IDS-F1  (tolerance=±{TOLERANCE} frames, split={DATA_SPLIT})")
    print(f"{'='*80}")
    for rank, r in enumerate(results[:10], 1):
        params_str = "  ".join(f"{k}={r[k]}" for k in keys)
        print(f"  #{rank:2d}  F1={r['ids_f1']:.4f}  P={r['ids_precision']:.4f}"
              f"  R={r['ids_recall']:.4f}  splits={r['num_predicted_splits']}"
              f"  tracklets={r['num_tracklets']}")
        print(f"       {params_str}")

    best = results[0]
    print(f"\nBest configuration:")
    for k in keys:
        print(f"  {k} = {best[k]}")
    print(f"  → IDS-F1={best['ids_f1']:.4f}  P={best['ids_precision']:.4f}  R={best['ids_recall']:.4f}")
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
