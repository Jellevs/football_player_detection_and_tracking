#!/usr/bin/env python3
"""
Intrinsic tracklet splitter metrics.

Computes four metrics for a tracklet splitter, all grounded in the
ground-truth identity labels rather than in HOTA:

  - Tracklet purity: for each output tracklet, the fraction of frames that
    carry the dominant ground-truth identity. Reported as macro-mean over
    output tracklets, and as a length-weighted mean.
  - Identity-switch recall (IDS-R): fraction of the ground-truth identity
    switches inside the input tracklets that the splitter splits within
    +/- tolerance frames.
  - Identity-switch precision (IDS-P): fraction of predicted splits that
    occur within +/- tolerance frames of a ground-truth identity switch.
  - Identity-switch F1: harmonic mean of IDS-R and IDS-P.

Ground-truth switch points are defined as positions inside an INPUT (pre-split)
tracklet where the GT identity of consecutive detections changes. A predicted
split is defined as the first frame of every output tracklet after the first
within an input tracklet. Matching is per input-tracklet and uses greedy
nearest-neighbour within +/- tolerance frames.

Inputs are MOT-format files (one per sequence):

    frame, id, x, y, w, h, conf, -1, -1, -1

Expected layout:

  <pred_dir>/SNPT-XXX.txt            predicted (post-split) tracklets
  <input_dir>/SNPT-XXX.txt           input tracklets BEFORE the splitter
                                     (e.g. the raw Deep-EIoU baseline output)
  <gt_dir>/SNPT-XXX/gt/gt.txt        ground-truth tracklets
    or <gt_dir>/SNPT-XXX.txt

Example:

  python intrinsic_metrics.py \\
      --pred evaluation/SNPT/alert_full/data \\
      --input evaluation/SNPT/baseline/data \\
      --gt path/to/test/gt_folder \\
      --tolerance 10 \\
      --iou 0.5 \\
      --out results_alert_full.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# =====================================================================
# MOT file IO
# =====================================================================

def load_mot_by_track(path: Path) -> Dict[int, List[Tuple[int, np.ndarray]]]:
    """track_id -> list of (frame, bbox_xywh) sorted by frame."""
    tracks: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            track_id = int(float(parts[1]))
            box = np.array([float(parts[2]), float(parts[3]),
                            float(parts[4]), float(parts[5])], dtype=np.float32)
            tracks[track_id].append((frame, box))
    for tid in tracks:
        tracks[tid].sort(key=lambda fb: fb[0])
    return tracks


def load_mot_by_frame(path: Path) -> Dict[int, List[Tuple[int, np.ndarray]]]:
    """frame -> list of (track_id, bbox_xywh)."""
    by_frame: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            track_id = int(float(parts[1]))
            box = np.array([float(parts[2]), float(parts[3]),
                            float(parts[4]), float(parts[5])], dtype=np.float32)
            by_frame[frame].append((track_id, box))
    return by_frame


# =====================================================================
# Geometry
# =====================================================================

def iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def best_match(box: np.ndarray, candidates: List[Tuple[int, np.ndarray]],
               iou_threshold: float) -> int:
    """Return the ID of the candidate with highest IoU above threshold, else -1."""
    best_iou = 0.0
    best_id = -1
    for cid, cbox in candidates:
        i = iou_xywh(box, cbox)
        if i > best_iou:
            best_iou = i
            best_id = cid
    return best_id if best_iou >= iou_threshold else -1


# =====================================================================
# Per-detection GT assignment
# =====================================================================

def build_det_to_gt(
    det_by_frame: Dict[int, List[Tuple[int, np.ndarray]]],
    gt_by_frame: Dict[int, List[Tuple[int, np.ndarray]]],
    iou_threshold: float,
) -> Dict[Tuple[int, int, float, float], int]:
    """
    Map each detection, keyed by (frame, track_id, x, y), to its GT track_id.
    Using coords in the key disambiguates duplicated track_ids in the input.
    Returns -1 for unmatched detections.
    """
    out: Dict[Tuple[int, int, float, float], int] = {}
    for frame, dets in det_by_frame.items():
        gts = gt_by_frame.get(frame, [])
        for tid, box in dets:
            key = (frame, tid, float(box[0]), float(box[1]))
            out[key] = best_match(box, gts, iou_threshold)
    return out


# =====================================================================
# Metrics
# =====================================================================

@dataclass
class SeqMetrics:
    sequence: str
    num_tracklets: int = 0
    num_frames_scored: int = 0
    mean_purity: float = float("nan")
    weighted_purity: float = float("nan")
    num_true_switches: int = 0
    num_predicted_splits: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else float("nan")

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else float("nan")

    @property
    def f1(self) -> float:
        r, p = self.recall, self.precision
        if np.isnan(r) or np.isnan(p) or (r + p) == 0:
            return float("nan")
        return 2 * r * p / (r + p)


def compute_purity(
    pred_tracks: Dict[int, List[Tuple[int, np.ndarray]]],
    pred_gt_map: Dict[Tuple[int, int, float, float], int],
) -> Tuple[float, float, int, int]:
    """Return (macro_mean_purity, length_weighted_purity, n_tracklets, n_frames_scored)."""
    purities: List[float] = []
    weights: List[int] = []
    total_scored = 0
    for tid, frames in pred_tracks.items():
        gts = [pred_gt_map.get((f, tid, float(b[0]), float(b[1])), -1)
               for f, b in frames]
        gts = [g for g in gts if g != -1]
        if not gts:
            continue
        counts: Dict[int, int] = defaultdict(int)
        for g in gts:
            counts[g] += 1
        dom = max(counts.values())
        total = len(gts)
        purities.append(dom / total)
        weights.append(total)
        total_scored += total
    if not purities:
        return float("nan"), float("nan"), 0, 0
    parr = np.array(purities)
    warr = np.array(weights, dtype=np.float64)
    macro = float(parr.mean())
    weighted = float((parr * warr).sum() / warr.sum())
    return macro, weighted, len(purities), total_scored


def find_true_switches(
    input_tracks: Dict[int, List[Tuple[int, np.ndarray]]],
    input_gt_map: Dict[Tuple[int, int, float, float], int],
) -> List[Tuple[int, int]]:
    """(input_track_id, frame) of every internal GT-identity change."""
    switches: List[Tuple[int, int]] = []
    for tid, frames in input_tracks.items():
        prev_gt: Optional[int] = None
        for f, b in frames:
            g = input_gt_map.get((f, tid, float(b[0]), float(b[1])), -1)
            if g == -1:
                continue
            if prev_gt is not None and g != prev_gt:
                switches.append((tid, f))
            prev_gt = g
    return switches


def find_predicted_splits(
    input_tracks: Dict[int, List[Tuple[int, np.ndarray]]],
    pred_by_frame: Dict[int, List[Tuple[int, np.ndarray]]],
    iou_threshold: float,
) -> List[Tuple[int, int]]:
    """
    For each input tracklet, record every frame at which the matched
    predicted tracklet ID changes. Detections that are absent from the
    predicted output (e.g. filtered by a short-fragment filter) are skipped;
    the 'previous predicted id' is retained across such gaps.
    """
    splits: List[Tuple[int, int]] = []
    for in_tid, frames in input_tracks.items():
        prev: Optional[int] = None
        for f, b in frames:
            cand = pred_by_frame.get(f, [])
            pid = best_match(b, cand, iou_threshold)
            if pid == -1:
                continue
            if prev is not None and pid != prev:
                splits.append((in_tid, f))
            prev = pid
    return splits


def match_switches(
    true_switches: List[Tuple[int, int]],
    predicted_splits: List[Tuple[int, int]],
    tolerance: int,
) -> Tuple[int, int, int]:
    """
    Per input tracklet, greedily match each true switch to the nearest unused
    predicted split within +/- tolerance frames. Returns (tp, fp, fn).
    """
    true_by_tid: Dict[int, List[int]] = defaultdict(list)
    pred_by_tid: Dict[int, List[int]] = defaultdict(list)
    for tid, f in true_switches:
        true_by_tid[tid].append(f)
    for tid, f in predicted_splits:
        pred_by_tid[tid].append(f)

    tp = 0
    all_tids = set(true_by_tid) | set(pred_by_tid)
    for tid in all_tids:
        t_list = sorted(true_by_tid[tid])
        p_list = sorted(pred_by_tid[tid])
        used = [False] * len(p_list)
        for tf in t_list:
            best_idx = -1
            best_d = tolerance + 1
            for i, pf in enumerate(p_list):
                if used[i]:
                    continue
                d = abs(pf - tf)
                if d <= tolerance and d < best_d:
                    best_d = d
                    best_idx = i
            if best_idx >= 0:
                used[best_idx] = True
                tp += 1
    n_true = len(true_switches)
    n_pred = len(predicted_splits)
    return tp, n_pred - tp, n_true - tp


# =====================================================================
# Orchestration
# =====================================================================

def resolve_gt_path(gt_root: Path, seq: str) -> Optional[Path]:
    for c in [gt_root / seq / "gt" / "gt.txt",
              gt_root / seq / "gt.txt",
              gt_root / f"{seq}.txt"]:
        if c.exists():
            return c
    return None


def process_sequence(
    seq: str,
    pred_path: Path,
    gt_path: Path,
    input_path: Optional[Path],
    tolerance: int,
    iou_threshold: float,
) -> SeqMetrics:
    pred_tracks = load_mot_by_track(pred_path)
    pred_by_frame = load_mot_by_frame(pred_path)
    gt_by_frame = load_mot_by_frame(gt_path)

    pred_gt_map = build_det_to_gt(pred_by_frame, gt_by_frame, iou_threshold)
    macro_p, wgt_p, n_tl, n_fr = compute_purity(pred_tracks, pred_gt_map)

    tp = fp = fn = 0
    n_true = n_pred = 0
    if input_path is not None and input_path.exists():
        input_tracks = load_mot_by_track(input_path)
        input_by_frame = load_mot_by_frame(input_path)
        input_gt_map = build_det_to_gt(input_by_frame, gt_by_frame, iou_threshold)
        true_switches = find_true_switches(input_tracks, input_gt_map)
        predicted_splits = find_predicted_splits(input_tracks, pred_by_frame, iou_threshold)
        tp, fp, fn = match_switches(true_switches, predicted_splits, tolerance)
        n_true = len(true_switches)
        n_pred = len(predicted_splits)

    return SeqMetrics(
        sequence=seq,
        num_tracklets=n_tl,
        num_frames_scored=n_fr,
        mean_purity=macro_p,
        weighted_purity=wgt_p,
        num_true_switches=n_true,
        num_predicted_splits=n_pred,
        tp=tp, fp=fp, fn=fn,
    )


def aggregate(rows: List[SeqMetrics]) -> Dict[str, float]:
    if not rows:
        return {}
    valid_p = [r.mean_purity for r in rows if not np.isnan(r.mean_purity)]
    total_tl = sum(r.num_tracklets for r in rows)
    total_fr = sum(r.num_frames_scored for r in rows)
    weighted_num = sum(r.weighted_purity * r.num_frames_scored
                       for r in rows if not np.isnan(r.weighted_purity))
    tp = sum(r.tp for r in rows)
    fp = sum(r.fp for r in rows)
    fn = sum(r.fn for r in rows)
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    if np.isnan(recall) or np.isnan(precision) or recall + precision == 0:
        f1 = float("nan")
    else:
        f1 = 2 * recall * precision / (recall + precision)
    return {
        "num_sequences": len(rows),
        "num_tracklets": total_tl,
        "mean_purity_macro": float(np.mean(valid_p)) if valid_p else float("nan"),
        "mean_purity_weighted": (weighted_num / total_fr) if total_fr else float("nan"),
        "num_true_switches": sum(r.num_true_switches for r in rows),
        "num_predicted_splits": sum(r.num_predicted_splits for r in rows),
        "tp": tp, "fp": fp, "fn": fn,
        "ids_recall": recall,
        "ids_precision": precision,
        "ids_f1": f1,
    }


def main() -> None:
    # Hardcoded defaults used when the script is invoked with no CLI flags
    # (e.g. a manual run from the IDE). The runner (run_splitter_eval.py)
    # passes --pred / --gt / --input / --seqmap / --out explicitly and those
    # values override the defaults below.
    DEFAULT_PRED = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\evaluation\SNPT\baseline\data")
    DEFAULT_GT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\soccernet-player-tracking\valid")
    DEFAULT_INPUT: Optional[Path] = None
    DEFAULT_SEQMAP = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\evaluation\seqmaps\SNPT-valid.txt")
    DEFAULT_OUT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_splitter\output\output.csv")

    ap = argparse.ArgumentParser(description="Intrinsic tracklet splitter metrics.")
    ap.add_argument("--pred",      type=Path, default=DEFAULT_PRED,
                    help="Directory with predicted (post-split) MOT files SNPT-XXX.txt.")
    ap.add_argument("--gt",        type=Path, default=DEFAULT_GT,
                    help="Ground-truth root (SNPT-XXX/gt/gt.txt or SNPT-XXX.txt).")
    ap.add_argument("--input",     type=Path, default=DEFAULT_INPUT,
                    help="Pre-splitter (baseline) MOT files; enables IDS-R/P/F1.")
    ap.add_argument("--seqmap",    type=Path, default=DEFAULT_SEQMAP,
                    help="Seqmap file listing sequence names, one per line.")
    ap.add_argument("--tolerance", type=int,  default=10,
                    help="Frame tolerance for matching predicted splits to GT switches.")
    ap.add_argument("--iou",       type=float, default=0.5,
                    help="IoU threshold for matching detections to GT.")
    ap.add_argument("--out",       type=Path, default=DEFAULT_OUT,
                    help="Output CSV with per-sequence results.")
    ap.add_argument("--quiet",     action="store_true",
                    help="Suppress per-sequence progress lines.")
    args = ap.parse_args()

    pred = args.pred
    gt = args.gt
    input_dir = args.input
    seqmap = args.seqmap
    tolerance = args.tolerance
    iou = args.iou
    out = args.out
    quiet = args.quiet

    if seqmap and seqmap.exists():
        with seqmap.open("r") as f:
            sequences = [line.strip() for line in f
                         if line.strip() and not line.strip().lower().startswith("name")]
    else:
        sequences = sorted(p.stem for p in pred.glob("SNPT-*.txt"))
    if not sequences:
        raise SystemExit(f"No SNPT-*.txt files found in {pred}")

    rows: List[SeqMetrics] = []
    for seq in sequences:
        pred_path = pred / f"{seq}.txt"
        gt_path = resolve_gt_path(gt, seq)
        input_path = (input_dir / f"{seq}.txt") if input_dir else None
        if not pred_path.exists():
            print(f"[warn] missing prediction {pred_path}; skipping")
            continue
        if gt_path is None:
            print(f"[warn] missing GT for {seq} under {gt}; skipping")
            continue
        row = process_sequence(seq, pred_path, gt_path, input_path,
                               tolerance, iou)
        rows.append(row)
        if not quiet:
            print(f"{seq}: purity={row.mean_purity:.3f}  "
                  f"IDS(R/P/F1)={row.recall:.3f}/{row.precision:.3f}/{row.f1:.3f}  "
                  f"tracklets={row.num_tracklets}  "
                  f"splits={row.num_predicted_splits}/{row.num_true_switches}")

    agg = aggregate(rows)
    print("\n== Aggregate ==")
    for k, v in agg.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sequence", "num_tracklets", "num_frames_scored",
                        "mean_purity", "weighted_purity",
                        "num_true_switches", "num_predicted_splits",
                        "tp", "fp", "fn",
                        "ids_recall", "ids_precision", "ids_f1"])
            for r in rows:
                w.writerow([r.sequence, r.num_tracklets, r.num_frames_scored,
                            f"{r.mean_purity:.4f}", f"{r.weighted_purity:.4f}",
                            r.num_true_switches, r.num_predicted_splits,
                            r.tp, r.fp, r.fn,
                            f"{r.recall:.4f}", f"{r.precision:.4f}", f"{r.f1:.4f}"])
        print(f"\nWrote per-sequence results to {out}")


if __name__ == "__main__":
    main()