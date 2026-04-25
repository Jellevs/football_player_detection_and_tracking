"""
visualize_jersey_splits.py — Dump crops around every GT identity switch in a sequence.

For each identity switch found (a tracklet whose matched GT ID changes), saves
N crops before and N crops after the switch point into a per-switch folder.
Intended for manual inspection — no TP/FP classification is done.

Usage (set SEQUENCE below, then run from repo root):
    python experiments/tracklet_splitter/visualize_jersey_splits.py
"""

from __future__ import annotations

import copy
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))

import settings
from utils.config import SplitterConfig
from utils.data_utils import load_images
from tracklets.splitters.jersey_splitter import JerseySplitter

CFG       = SplitterConfig(**settings.SPLITTER)
TOLERANCE = 15   # frames: predicted split must be within this of the GT switch to count as caught

# ── configuration ─────────────────────────────────────────────────────────────
DATA_SPLIT    = "test"
SEQUENCE      = "SNPT-120"   # ← change this to the sequence you want to inspect
GT_ROOT       = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\soccernet-player-tracking") / DATA_SPLIT
CACHE_DIR     = REPO / "output" / "cache"
OUT_DIR       = Path(__file__).parent / "output" / "figures" / SEQUENCE
IOU_THRESHOLD = 0.5

N_BEFORE = 5   # crops to save before the switch
N_AFTER  = 5   # crops to save after the switch

# ── geometry ──────────────────────────────────────────────────────────────────

def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


# ── GT loading ────────────────────────────────────────────────────────────────

def load_gt_by_frame(sequence: str) -> dict:
    for candidate in [GT_ROOT / sequence / "gt" / "gt.txt",
                      GT_ROOT / f"{sequence}.txt"]:
        if candidate.exists():
            gt_path = candidate
            break
    else:
        return {}
    by_frame: dict = defaultdict(list)
    with gt_path.open() as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            gt_id = int(float(parts[1]))
            x, y, w, h = (float(p) for p in parts[2:6])
            by_frame[frame].append((gt_id, np.array([x, y, x+w, y+h], dtype=np.float32)))
    return dict(by_frame)


def build_gt_map(tracklets: dict, gt_by_frame: dict) -> dict:
    gt_map: dict = {}
    for tid, tracklet in tracklets.items():
        for local_i, (frame, bbox) in enumerate(zip(tracklet.frames, tracklet.bboxes)):
            best_iou, best_gt = 0.0, -1
            for gt_id, gt_bbox in gt_by_frame.get(frame, []):
                iou = iou_xyxy(bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_id
            gt_map[(tid, local_i)] = best_gt if best_iou >= IOU_THRESHOLD else -1
    return gt_map


def dominant_gt(gt_sequence: list) -> int:
    """Return most common GT ID in a list, ignoring -1 (unmatched)."""
    valid = [g for g in gt_sequence if g != -1]
    if not valid:
        return -1
    return max(set(valid), key=valid.count)


def find_true_switches(tracklets: dict, gt_map: dict, window: int = 15, min_votes: int = 8) -> list:
    """
    Return list of (tid, switch_frame, switch_local_idx, gt_before, gt_after).

    A switch is only recorded when the dominant matched GT ID in a window of
    `window` frames BEFORE the candidate point differs from the dominant GT ID
    in the same-sized window AFTER it, and each side has at least `min_votes`
    valid (matched) frames. This filters out single-frame IoU blips.

    Only the first qualifying switch per tracklet is returned.
    """
    switches = []
    for tid, tracklet in tracklets.items():
        n = len(tracklet.frames)
        gt_seq = [gt_map.get((tid, i), -1) for i in range(n)]

        for local_i in range(window, n - window):
            before = gt_seq[local_i - window : local_i]
            after  = gt_seq[local_i : local_i + window]

            gt_b = dominant_gt(before)
            gt_a = dominant_gt(after)

            if gt_b == -1 or gt_a == -1 or gt_b == gt_a:
                continue

            valid_before = sum(1 for g in before if g != -1)
            valid_after  = sum(1 for g in after  if g != -1)
            if valid_before < min_votes or valid_after < min_votes:
                continue

            switches.append((tid, tracklet.frames[local_i], local_i, gt_b, gt_a))
            break   # one switch per tracklet is enough

    return switches


# ── jersey splitter ───────────────────────────────────────────────────────────

def run_jersey_splitter(tracklets: dict) -> dict:
    """Return {original_tid: split_frame_or_None} for every tracklet."""
    splitter = JerseySplitter(CFG)
    next_id  = max(tracklets.keys(), default=0) + 1
    split_frames: dict = {}
    for tid, tracklet in tracklets.items():
        tc = copy.deepcopy(tracklet)
        frags = splitter.split_tracklet(tc, next_id)
        if frags and len(frags) >= 2:
            frags.sort(key=lambda f: f.frames[0])
            split_frames[tid] = frags[1].frames[0]   # frame where second fragment starts
            next_id = max(next_id, max(f.track_id for f in frags) + 1)
        else:
            split_frames[tid] = None
    return split_frames


# ── crop loading ──────────────────────────────────────────────────────────────

def load_crop(images: list, tracklet, local_idx: int, pad: float = 1.0):
    """
    Return an RGB crop expanded by `pad` × bbox dimensions on each side.
    A green rectangle marks the tracked player's original bbox in the crop.
    """
    frame_idx = tracklet.frames[local_idx]
    if frame_idx >= len(images):
        return None
    img = cv2.imread(str(images[frame_idx]))
    if img is None:
        return None
    ox1, oy1, ox2, oy2 = map(int, tracklet.bboxes[local_idx])
    bw = ox2 - ox1
    bh = oy2 - oy1
    px = int(bw * pad)
    py = int(bh * pad)
    H, W = img.shape[:2]
    cx1 = max(0, ox1 - px)
    cy1 = max(0, oy1 - py)
    cx2 = min(W, ox2 + px)
    cy2 = min(H, oy2 + py)
    crop = img[cy1:cy2, cx1:cx2].copy()
    if crop.size == 0:
        return None
    # Draw original bbox inside the expanded crop
    rx1 = ox1 - cx1
    ry1 = oy1 - cy1
    rx2 = ox2 - cx1
    ry2 = oy2 - cy1
    thickness = max(2, bw // 40)
    cv2.rectangle(crop, (rx1, ry1), (rx2, ry2), (0, 220, 0), thickness)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


# ── crop saving ───────────────────────────────────────────────────────────────

def save_switch_crops(
    images: list,
    tracklet,
    switch_local_idx: int,
    out_dir: Path,
    n_before: int = N_BEFORE,
    n_after:  int = N_AFTER,
):
    """Save evenly-spaced crops before and after switch_local_idx."""
    n = len(tracklet.frames)
    out_dir.mkdir(parents=True, exist_ok=True)

    jerseys   = tracklet.pred_attributes.get("jerseys", [])
    entropies = tracklet.pred_attributes.get("jersey_entropies", [])

    def sample_indices(start, end, count):
        pool = list(range(start, end))
        if not pool:
            return []
        step = max(1, len(pool) // count)
        return pool[::step][:count]

    before_indices = sample_indices(max(0, switch_local_idx - n_before * 4), switch_local_idx, n_before)
    after_indices  = sample_indices(switch_local_idx, min(n, switch_local_idx + n_after * 4), n_after)

    def save_one(local_idx, filename):
        crop = load_crop(images, tracklet, local_idx)
        if crop is None:
            print(f"    [warn] no crop at local_idx={local_idx}")
            return
        j = jerseys[local_idx]   if local_idx < len(jerseys)   else None
        e = entropies[local_idx] if local_idx < len(entropies) else None
        has_j = j is not None and not (isinstance(j, float) and np.isnan(j))
        j_str = f"#{int(j)}" if has_j else "?"
        e_str = f"{e:.3f}" if e is not None and not (isinstance(e, float) and np.isnan(e)) else "?"
        print(f"    {filename}  frame={tracklet.frames[local_idx]}  jersey={j_str}  entropy={e_str}")
        bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / filename), bgr)

    for i, li in enumerate(before_indices, start=1):
        save_one(li, f"before_{i:02d}.png")
    for i, li in enumerate(after_indices, start=1):
        save_one(li, f"after_{i:02d}.png")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    seq = SEQUENCE
    print(f"Sequence : {seq}")
    print(f"Output   : {OUT_DIR}\n")

    cache_path = CACHE_DIR / f"cache_attributes_{seq}.pkl"
    if not cache_path.exists():
        print(f"[error] No attribute cache found at {cache_path}")
        sys.exit(1)

    with cache_path.open("rb") as f:
        tracklets = pickle.load(f)

    gt_by_frame = load_gt_by_frame(seq)
    if not gt_by_frame:
        print(f"[error] No GT file found for {seq}")
        sys.exit(1)

    images = load_images(GT_ROOT / seq / "img1")
    print(f"Loaded {len(tracklets)} tracklets, {len(images)} frames, {len(gt_by_frame)} GT frames\n")

    gt_map   = build_gt_map(tracklets, gt_by_frame)
    switches = find_true_switches(tracklets, gt_map)
    print(f"Found {len(switches)} GT identity switches")

    print("Running jersey splitter...")
    split_frames = run_jersey_splitter(tracklets)
    print("Done.\n")

    for switch_num, (tid, sw_frame, sw_local, gt_before, gt_after) in enumerate(switches, start=1):
        tracklet  = tracklets[tid]
        predicted = split_frames.get(tid)
        caught    = predicted is not None and abs(predicted - sw_frame) <= TOLERANCE
        label     = "SPLIT" if caught else "MISSED"

        folder_name = (f"switch_{switch_num:03d}_{label}_track{tid}"
                       f"_frame{sw_frame}_gt{gt_before}to{gt_after}")
        out_dir = OUT_DIR / folder_name
        print(f"[{switch_num}/{len(switches)}] [{label}] track={tid}  frame={sw_frame}  "
              f"GT {gt_before}→{gt_after}  tracklet_len={len(tracklet.frames)}"
              + (f"  splitter_split_at={predicted}" if predicted else ""))
        save_switch_crops(images, tracklet, sw_local, out_dir)
        print(f"  → {out_dir}\n")

    print(f"Done. {len(switches)} switches saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
