"""
browse_jersey_splits.py — Interactive browser for jersey splitter results.

For each GT identity switch in a sequence, shows N_CROPS crops directly before
and N_CROPS crops directly after the switch point, with expanded bounding boxes
and jersey/entropy annotations.

Each switch is classified as SPLIT (caught by jersey splitter) or MISSED.

Controls:
  - Right arrow / N     → next switch
  - Left arrow / P      → previous switch
  - S                   → save current figure to output directory
  - Q / Escape          → quit

Usage (from repo root):
    python experiments/tracklet_splitter/browse_jersey_splits.py
"""

from __future__ import annotations

import copy
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))

import settings
from utils.config import SplitterConfig
from utils.data_utils import load_images
from tracklets.splitters.jersey_splitter import JerseySplitter

# ── configuration (edit these) ───────────────────────────────────────────────

SEQUENCE      = "SNPT-117"       # sequence to inspect
DATA_SPLIT    = "test"           # "valid" or "test"
N_CROPS       = 5                # crops before and after the switch
BBOX_PAD      = 0.6              # expand crop by this fraction of bbox size
IOU_THRESHOLD = 0.5              # IoU for matching detections to GT
TOLERANCE     = 15               # ±frames for classifying SPLIT vs MISSED

# ── paths (derived from settings + split) ────────────────────────────────────

DATA_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development"
                 r"\data\soccernet\soccernet-player-tracking") / DATA_SPLIT
CACHE_DIR = settings.OUTPUT_ROOT / "cache"
OUT_DIR   = Path(__file__).parent / "output" / "figures"

CFG = SplitterConfig(**settings.SPLITTER)


# ── geometry ─────────────────────────────────────────────────────────────────

def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


# ── GT loading ───────────────────────────────────────────────────────────────

def load_gt_by_frame(sequence: str) -> dict:
    """Load GT as {frame_number: [(gt_id, bbox_xyxy)]}.

    GT files use MOT format with 1-indexed frames.
    """
    gt_path = DATA_ROOT / sequence / "gt" / "gt.txt"
    if not gt_path.exists():
        gt_path = DATA_ROOT / f"{sequence}.txt"
    if not gt_path.exists():
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
            by_frame[frame].append((gt_id, np.array([x, y, x + w, y + h], dtype=np.float32)))
    return dict(by_frame)


def build_gt_map(tracklets: dict, gt_by_frame: dict) -> dict:
    """Map (track_id, local_idx) -> gt_id via IoU matching.

    Cached tracklet frames are already 1-indexed (matching GT/MOT convention).
    """
    gt_map: dict = {}
    for tid, t in tracklets.items():
        for i, (frame, bbox) in enumerate(zip(t.frames, t.bboxes)):
            best_iou, best_gt = 0.0, -1
            for gt_id, gt_bbox in gt_by_frame.get(frame, []):
                iou = iou_xyxy(bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_id
            gt_map[(tid, i)] = best_gt if best_iou >= IOU_THRESHOLD else -1
    return gt_map


def dominant_gt(gt_ids: list[int]) -> int:
    valid = [g for g in gt_ids if g != -1]
    if not valid:
        return -1
    return max(set(valid), key=valid.count)


def find_true_switches(tracklets: dict, gt_map: dict,
                       window: int = 15, min_votes: int = 8) -> list[dict]:
    """Find GT identity switches using windowed dominant-ID voting.

    Returns list of dicts with switch metadata. Only the first switch per
    tracklet is returned.
    """
    switches = []
    for tid, t in tracklets.items():
        n = len(t.frames)
        gt_seq = [gt_map.get((tid, i), -1) for i in range(n)]

        for local_i in range(window, n - window):
            before = gt_seq[local_i - window : local_i]
            after  = gt_seq[local_i : local_i + window]

            gt_b = dominant_gt(before)
            gt_a = dominant_gt(after)

            if gt_b == -1 or gt_a == -1 or gt_b == gt_a:
                continue
            if sum(1 for g in before if g != -1) < min_votes:
                continue
            if sum(1 for g in after if g != -1) < min_votes:
                continue

            switches.append({
                "tid": tid,
                "frame": t.frames[local_i],
                "local_idx": local_i,
                "gt_before": gt_b,
                "gt_after": gt_a,
                "tracklet_len": n,
            })
            break   # one switch per tracklet

    return switches


# ── jersey splitter ──────────────────────────────────────────────────────────

def run_jersey_splitter(tracklets: dict) -> dict[int, list[int]]:
    """Return {original_tid: [split_frame, ...]} for every tracklet."""
    splitter = JerseySplitter(CFG)
    next_id = max(tracklets.keys(), default=0) + 1
    split_frames: dict[int, list[int]] = {}
    for tid, t in tracklets.items():
        tc = copy.deepcopy(t)
        frags = splitter.split_tracklet(tc, next_id)
        if frags and len(frags) >= 2:
            frags.sort(key=lambda f: f.frames[0])
            split_frames[tid] = [f.frames[0] for f in frags[1:]]
            next_id = max(next_id, max(f.track_id for f in frags) + 1)
        else:
            split_frames[tid] = []
    return split_frames


# ── crop extraction ──────────────────────────────────────────────────────────

def load_crop(images: list, tracklet, local_idx: int):
    """Load an expanded crop around the bbox at local_idx.

    Returns (crop_rgb, (rx, ry, rw, rh)) or (None, None).
    The rect is the bbox position relative to the crop, for drawing.
    """
    frame_idx = tracklet.frames[local_idx]
    if frame_idx < 0 or frame_idx >= len(images):
        return None, None

    img = cv2.imread(str(images[frame_idx]))
    if img is None:
        return None, None

    ox1, oy1, ox2, oy2 = map(int, tracklet.bboxes[local_idx])
    bw = ox2 - ox1
    bh = oy2 - oy1
    px = int(bw * BBOX_PAD)
    py = int(bh * BBOX_PAD)
    H, W = img.shape[:2]

    cx1 = max(0, ox1 - px)
    cy1 = max(0, oy1 - py)
    cx2 = min(W, ox2 + px)
    cy2 = min(H, oy2 + py)

    crop = img[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None, None

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rx1 = ox1 - cx1
    ry1 = oy1 - cy1
    return crop_rgb, (rx1, ry1, bw, bh)


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_switch(switch: dict, tracklets: dict, images: list,
                split_frames: dict, fig: plt.Figure):
    """Draw the before/after crop grid for one switch."""
    fig.clf()

    tid = switch["tid"]
    tracklet = tracklets[tid]
    sw_local = switch["local_idx"]
    n = switch["tracklet_len"]

    # Check if splitter caught this switch
    pred_splits = split_frames.get(tid, [])
    caught_frame = None
    for pf in pred_splits:
        if abs(pf - switch["frame"]) <= TOLERANCE:
            caught_frame = pf
            break
    label = "SPLIT" if caught_frame is not None else "MISSED"
    label_color = "green" if label == "SPLIT" else "red"

    # Get crop indices: N directly before, N directly after
    before_end = sw_local
    before_start = max(0, before_end - N_CROPS)
    before_idx = list(range(before_start, before_end))

    after_start = sw_local
    after_end = min(n, after_start + N_CROPS)
    after_idx = list(range(after_start, after_end))

    ncols = N_CROPS + N_CROPS   # always 2*N_CROPS columns

    # Attributes for annotation
    jerseys   = tracklet.pred_attributes.get("jerseys", [])
    entropies = tracklet.pred_attributes.get("jersey_entropies", [])

    def jersey_str(li):
        if li < len(jerseys):
            j = jerseys[li]
            if j is not None and not (isinstance(j, float) and np.isnan(j)):
                return f"#{int(j)}"
        return "—"

    def entropy_str(li):
        if li < len(entropies):
            e = entropies[li]
            if e is not None and not (isinstance(e, float) and np.isnan(e)):
                return f"H={e:.2f}"
        return ""

    # Title
    title = (f"[{label}]  Track {tid}  |  "
             f"GT: {switch['gt_before']} -> {switch['gt_after']}  |  "
             f"Switch frame: {switch['frame']}  |  "
             f"Tracklet len: {n}")
    if caught_frame is not None:
        title += f"  |  Splitter split at: {caught_frame}"
    fig.suptitle(title, fontsize=11, fontweight="bold", color=label_color)

    axes = fig.subplots(1, ncols, squeeze=False)[0]

    # ── BEFORE crops (lime bbox) ─────────────────────────────────────────
    for col, li in enumerate(before_idx):
        ax = axes[col]
        crop, rect = load_crop(images, tracklet, li)
        if crop is not None:
            ax.imshow(crop)
            rx, ry, rw, rh = rect
            ax.add_patch(mpatches.Rectangle(
                (rx, ry), rw, rh, linewidth=2,
                edgecolor="lime", facecolor="none"))
        else:
            ax.set_facecolor("black")

        ax.set_title(f"f{tracklet.frames[li]}\n{jersey_str(li)}  {entropy_str(li)}",
                      fontsize=8, color="white")
        ax.set_xlabel("BEFORE", fontsize=8, color="#4488ff")
        ax.set_xticks([]); ax.set_yticks([])

    for col in range(len(before_idx), N_CROPS):
        axes[col].axis("off")

    # ── AFTER crops (orange bbox) ────────────────────────────────────────
    for col_off, li in enumerate(after_idx):
        col = N_CROPS + col_off
        ax = axes[col]
        crop, rect = load_crop(images, tracklet, li)
        if crop is not None:
            ax.imshow(crop)
            rx, ry, rw, rh = rect
            ax.add_patch(mpatches.Rectangle(
                (rx, ry), rw, rh, linewidth=2,
                edgecolor="orange", facecolor="none"))
        else:
            ax.set_facecolor("black")

        ax.set_title(f"f{tracklet.frames[li]}\n{jersey_str(li)}  {entropy_str(li)}",
                      fontsize=8, color="white")
        ax.set_xlabel("AFTER", fontsize=8, color="#ff8844")
        ax.set_xticks([]); ax.set_yticks([])

    for col_off in range(len(after_idx), N_CROPS):
        axes[N_CROPS + col_off].axis("off")

    # Separator line between before and after
    mid_x = N_CROPS / ncols
    fig.add_artist(plt.Line2D(
        [mid_x, mid_x], [0.02, 0.85],
        transform=fig.transFigure,
        color="white", linewidth=2, linestyle="--"))

    fig.patch.set_facecolor("#1a1a1a")
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    seq = SEQUENCE
    save_dir = OUT_DIR / seq

    print(f"Sequence : {seq}")
    print(f"Split    : {DATA_SPLIT}")
    print(f"Crops    : {N_CROPS} before + {N_CROPS} after")
    print(f"Tolerance: +/-{TOLERANCE} frames\n")

    # ── load cached tracklets ────────────────────────────────────────────
    cache_path = CACHE_DIR / f"cache_attributes_{seq}.pkl"
    if not cache_path.exists():
        print(f"[error] No attribute cache at {cache_path}")
        sys.exit(1)

    with cache_path.open("rb") as f:
        tracklets = pickle.load(f)

    # ── load GT ──────────────────────────────────────────────────────────
    gt_by_frame = load_gt_by_frame(seq)
    if not gt_by_frame:
        print(f"[error] No GT found at {DATA_ROOT / seq / 'gt' / 'gt.txt'}")
        sys.exit(1)

    # ── load images ──────────────────────────────────────────────────────
    img_dir = DATA_ROOT / seq / "img1"
    images = load_images(img_dir)
    print(f"Loaded {len(tracklets)} tracklets, {len(images)} images, "
          f"{len(gt_by_frame)} GT frames\n")

    # ── find GT identity switches ────────────────────────────────────────
    gt_map = build_gt_map(tracklets, gt_by_frame)
    # Quick diagnostic: how many detections matched GT?
    n_matched = sum(1 for v in gt_map.values() if v != -1)
    print(f"GT matching: {n_matched}/{len(gt_map)} detections matched "
          f"({100*n_matched/max(len(gt_map),1):.1f}%)")

    switches = find_true_switches(tracklets, gt_map)
    print(f"Found {len(switches)} GT identity switches\n")

    if not switches:
        print("No identity switches found in this sequence.")
        print("This could mean:")
        print("  - No tracklets contain an identity switch")
        print("  - GT matching failed (check frame indexing)")
        print(f"  - Sample GT frames: {sorted(gt_by_frame.keys())[:5]}")
        sample_tid = next(iter(tracklets))
        print(f"  - Sample tracklet frames: {tracklets[sample_tid].frames[:5]}")
        return

    # ── run splitter ─────────────────────────────────────────────────────
    print("Running jersey splitter...")
    split_frames = run_jersey_splitter(tracklets)

    n_caught = sum(
        1 for sw in switches
        if any(abs(pf - sw["frame"]) <= TOLERANCE
               for pf in split_frames.get(sw["tid"], []))
    )
    print(f"Splitter caught {n_caught}/{len(switches)} switches\n")

    # ── print summary ────────────────────────────────────────────────────
    for i, sw in enumerate(switches):
        pred = split_frames.get(sw["tid"], [])
        caught = any(abs(pf - sw["frame"]) <= TOLERANCE for pf in pred)
        tag = "SPLIT" if caught else "MISSED"
        print(f"  [{i+1:2d}] [{tag:6s}]  track={sw['tid']:4d}  "
              f"frame={sw['frame']:5d}  GT {sw['gt_before']}->{sw['gt_after']}  "
              f"len={sw['tracklet_len']}"
              + (f"  split_at={pred}" if pred else ""))

    # ── interactive browser ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Controls: left/right or N/P to navigate, S to save, Q to quit")
    print(f"{'='*60}\n")

    current = [0]

    fig = plt.figure(figsize=(2.5 * N_CROPS * 2, 4))
    plot_switch(switches[current[0]], tracklets, images, split_frames, fig)

    def on_key(event):
        if event.key in ("right", "n"):
            current[0] = min(current[0] + 1, len(switches) - 1)
        elif event.key in ("left", "p"):
            current[0] = max(current[0] - 1, 0)
        elif event.key == "s":
            sw = switches[current[0]]
            pred = split_frames.get(sw["tid"], [])
            caught = any(abs(pf - sw["frame"]) <= TOLERANCE for pf in pred)
            tag = "SPLIT" if caught else "MISSED"
            fname = (f"{tag}_track{sw['tid']}_frame{sw['frame']}"
                     f"_gt{sw['gt_before']}to{sw['gt_after']}.png")
            save_path = save_dir / fname
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_path), dpi=150,
                        facecolor=fig.get_facecolor(), bbox_inches="tight")
            print(f"  Saved: {save_path}")
            return
        elif event.key in ("q", "escape"):
            plt.close(fig)
            return
        else:
            return

        plot_switch(switches[current[0]], tracklets, images, split_frames, fig)
        fig.canvas.draw()
        print(f"  Showing switch {current[0]+1}/{len(switches)}")

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    print("\nDone.")


if __name__ == "__main__":
    main()
