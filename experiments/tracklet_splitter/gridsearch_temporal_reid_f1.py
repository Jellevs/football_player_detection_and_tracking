"""
gridsearch_temporal_reid_f1.py — Grid search over TemporalReIDSplitter hyperparameters.

Evaluates the temporal ReID splitter **in isolation** by comparing the original
tracker output (baseline MOT files) against the temporal-split output.

The baseline is the raw tracker output (pre-any-splitting).  The splitter is
run on tracklets reconstructed from the tracked_detections cache (which
contains ReID embeddings needed by the temporal splitter).

True identity switches are computed within the baseline tracklets, so TP/FP/FN
reflect only how well the temporal-ReID signal detects switches.

Prerequisites
-------------
Tracked-detection caches must exist:
    output/cache/cache_tracked_detections_<sequence>.pkl
Run main.py at least once with the desired split to generate them.

Usage
-----
    python experiments/tracklet_splitter/gridsearch_temporal_reid_f1.py

Results are saved to:
    experiments/tracklet_splitter/output/gridsearch_temporal_reid_f1_results.csv
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
from utils.data_utils import organize_detections_by_track
from tracklets.splitters.temporal_reid_splitter import TemporalReIDSplitter

from experiments.tracklet_splitter.intrinsic_metrics import (
    load_mot_by_track,
    load_mot_by_frame,
    build_det_to_gt,
    find_true_switches,
    find_predicted_splits,
    match_switches,
    resolve_gt_path,
)

# ── configuration ─────────────────────────────────────────────────────────────

DATA_SPLIT = "test"

GT_ROOT   = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\soccernet-player-tracking") / DATA_SPLIT
EVAL_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\evaluation\SNPT")
INPUT_DIR = EVAL_ROOT / "baseline" / "data"          # pre-split baseline tracklets (MOT format)
SEQMAP    = REPO / "evaluation" / "seqmaps" / f"SNPT-{DATA_SPLIT}.txt"
CACHE_DIR = REPO / "output" / "cache"
OUT_CSV   = Path(__file__).parent / "output" / "gridsearch_temporal_reid_f1_results.csv"

IOU_THRESHOLD = 0.5   # for matching tracklet detections to GT detections
TOLERANCE     = 10    # frames: split points within ±TOLERANCE of a true switch count as TP

# ── hyperparameter grid ───────────────────────────────────────────────────────

GRID = dict(
    min_gap_frames      = [3, 5, 10, 15, 20, 30],
    reid_threshold      = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
    min_segment_frames  = [3, 5, 10],
    n_samples           = [10, 20, 30],
)


# ── helpers ───────────────────────────────────────────────────────────────────

def tracklets_to_mot_by_frame(
    tracklets: dict,
) -> dict[int, list[tuple[int, np.ndarray]]]:
    """
    Convert Tracklet objects (xyxy bboxes) to the same format as
    intrinsic_metrics.load_mot_by_frame:
        {frame: [(track_id, bbox_xywh), ...]}
    Bboxes are converted from xyxy to xywh.  Frame indices from the
    tracked_detections cache are 0-indexed, so we add +1 for MOT convention.
    """
    by_frame: dict[int, list] = defaultdict(list)
    for tid, tracklet in tracklets.items():
        for frame, bbox in zip(tracklet.frames, tracklet.bboxes):
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            xywh = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
            # tracked_detections use 0-indexed frames; MOT/GT use 1-indexed
            by_frame[frame + 1].append((tid, xywh))
    return dict(by_frame)


def apply_temporal_splitter(tracklets: dict, params: dict) -> dict:
    """Run the temporal ReID splitter and return the new tracklet dict."""
    splitter = TemporalReIDSplitter(**params)
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
        sequences = sorted(p.stem.replace("cache_tracked_detections_", "")
                           for p in CACHE_DIR.glob("cache_tracked_detections_SNPT-*.pkl"))
    print(f"Split: {DATA_SPLIT}  |  Sequences from seqmap: {len(sequences)}")

    # ── load baseline MOT files, GT, and tracked-detection caches ─────────
    # Baseline = original tracker output (pre-any-splitting) from MOT files.
    # Tracked-detection caches provide the raw tracklets with ReID embeddings.
    print("Loading baseline MOT files, GT, and tracked-detection caches...")
    input_tracks_per_seq: dict[str, dict] = {}
    true_switches_per_seq: dict[str, list] = {}
    raw_tracklets_per_seq: dict[str, dict] = {}

    skipped = []
    for seq in sequences:
        input_path = INPUT_DIR / f"{seq}.txt"
        gt_path = resolve_gt_path(GT_ROOT, seq)
        det_cache_path = CACHE_DIR / f"cache_tracked_detections_{seq}.pkl"

        missing = []
        if not input_path.exists():
            missing.append("input MOT")
        if gt_path is None:
            missing.append("GT")
        if not det_cache_path.exists():
            missing.append("tracked_detections cache")
        if missing:
            skipped.append((seq, ", ".join(missing)))
            continue

        # Load baseline MOT files for evaluation reference
        input_tracks = load_mot_by_track(input_path)
        input_by_frame = load_mot_by_frame(input_path)
        gt_by_frame = load_mot_by_frame(gt_path)

        # Build GT map and true switches within baseline tracklets
        input_gt_map = build_det_to_gt(input_by_frame, gt_by_frame, IOU_THRESHOLD)
        true_switches = find_true_switches(input_tracks, input_gt_map)

        # Load tracked detections and reconstruct tracklets with embeddings
        with det_cache_path.open("rb") as f:
            tracked_detections = pickle.load(f)
        raw_tracklets = organize_detections_by_track(tracked_detections)

        input_tracks_per_seq[seq] = input_tracks
        true_switches_per_seq[seq] = true_switches
        raw_tracklets_per_seq[seq] = raw_tracklets

    sequences = [s for s in sequences if s in raw_tracklets_per_seq]
    print(f"  Ready: {len(sequences)} sequences  |  Skipped: {len(skipped)}")
    if skipped:
        for seq, reason in skipped[:5]:
            print(f"    {seq}: missing {reason}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more")

    total_true = sum(len(v) for v in true_switches_per_seq.values())
    print(f"  Total true switches (within baseline tracklets): {total_true}")

    if not sequences:
        print("No sequences to evaluate. Ensure baseline MOT files, GT, and "
              "tracked_detections caches exist.")
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

        tp_total = fp_total = fn_total = 0
        n_pred_total = 0
        n_splits_total = 0

        for seq in sequences:
            tracklets = copy.deepcopy(raw_tracklets_per_seq[seq])
            split = apply_temporal_splitter(tracklets, params)

            # Convert temporal-split output to MOT-style by-frame format
            pred_by_frame = tracklets_to_mot_by_frame(split)

            # find_predicted_splits compares baseline (original tracker) vs
            # temporal-split output → only temporal-caused ID changes
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
            n_splits_total += len(split) - len(tracklets)

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
            "num_new_tracklets":    n_splits_total,
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
    print(f"  Evaluated in ISOLATION — only temporal-ReID-caused splits counted")
    print(f"{'='*80}")
    for rank, r in enumerate(results[:10], 1):
        params_str = "  ".join(f"{k}={r[k]}" for k in keys)
        print(f"  #{rank:2d}  F1={r['ids_f1']:.4f}  P={r['ids_precision']:.4f}"
              f"  R={r['ids_recall']:.4f}  new_tracklets={r['num_new_tracklets']}"
              f"  pred_splits={r['num_predicted_splits']}")
        print(f"       {params_str}")

    best = results[0]
    print(f"\nBest configuration:")
    for k in keys:
        print(f"  {k} = {best[k]}")
    print(f"  → IDS-F1={best['ids_f1']:.4f}  P={best['ids_precision']:.4f}  R={best['ids_recall']:.4f}")
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
