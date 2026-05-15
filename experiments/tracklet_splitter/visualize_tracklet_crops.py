"""
visualize_tracklet_crops.py — Save every frame crop for a given track ID.

For the specified sequence and track ID, reads the tracklet from the attribute
cache, draws the bounding box on an expanded crop of every frame, labels it
with the track ID, and saves PNGs to:

    experiments/tracklet_splitter/output/crops/<SEQUENCE>/track_<ID>/

Configuration
-------------
Set SEQUENCE and TRACK_ID below, then run from repo root:

    python experiments/tracklet_splitter/visualize_tracklet_crops.py

Optional: set STEP to only save every Nth frame (e.g. STEP=5 → every 5th).
Set MAX_FRAMES to cap the total number of saved crops (None = no cap).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))

import settings
from utils.data_utils import load_images

# ── configuration ─────────────────────────────────────────────────────────────

SEQUENCE  = "SNPT-138"   # ← sequence to inspect
TRACK_ID  = 48            # ← track ID to visualise
DATA_SPLIT = "test"

# Use "cache" (post-STR) or "backup_cache" (baseline, pre-STR)
CACHE_VARIANT = "cache"

EXPAND    = 0.8    # fractional padding around bbox (0.4 = 40% on each side)
STEP      = 1      # save every Nth frame (1 = every frame)
MAX_FRAMES = None  # cap total saved crops; None = all frames

GT_ROOT   = settings.DATA_ROOT.parent / DATA_SPLIT   # .../soccernet-player-tracking/test
CACHE_DIR = REPO / "output" / CACHE_VARIANT
OUT_DIR   = Path(__file__).parent / "output" / "crops" / SEQUENCE / f"track_{TRACK_ID}"

# ── label style ───────────────────────────────────────────────────────────────

BBOX_COLOR   = (0, 220, 0)     # green (BGR)
LABEL_BG     = (0, 220, 0)
LABEL_FG     = (0, 0, 0)       # black text
FONT         = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE   = 0.55
FONT_THICK   = 1


# ── helpers ───────────────────────────────────────────────────────────────────

def load_tracklet(sequence: str, track_id: int):
    import pickle
    cache_path = CACHE_DIR / f"cache_attributes_{sequence}.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    with cache_path.open("rb") as f:
        tracklets = pickle.load(f)
    if track_id not in tracklets:
        available = sorted(tracklets.keys())
        raise KeyError(f"Track ID {track_id} not found in {sequence}. "
                       f"Available IDs: {available}")
    return tracklets[track_id]


def draw_crop(img: np.ndarray, bbox_xyxy, label: str, expand: float) -> np.ndarray:
    """
    Return an expanded crop of `img` centred on `bbox_xyxy`, with the
    original bounding box drawn in green and a label above it.
    """
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    bw = x2 - x1
    bh = y2 - y1
    H, W = img.shape[:2]

    # Expand crop region
    pad_x = int(bw * expand)
    pad_y = int(bh * expand)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(W, x2 + pad_x)
    cy2 = min(H, y2 + pad_y)

    crop = img[cy1:cy2, cx1:cx2].copy()

    # bbox in crop-local coordinates
    rx1 = x1 - cx1
    ry1 = y1 - cy1
    rx2 = x2 - cx1
    ry2 = y2 - cy1

    thickness = max(2, bw // 40)
    cv2.rectangle(crop, (rx1, ry1), (rx2, ry2), BBOX_COLOR, thickness)

    # Label above bbox
    (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)
    lx1 = rx1
    ly1 = max(0, ry1 - th - baseline - 4)
    lx2 = lx1 + tw + 6
    ly2 = ly1 + th + baseline + 4

    # Clamp label box to crop bounds
    lx2 = min(crop.shape[1], lx2)
    ly2 = min(crop.shape[0], ly2)

    cv2.rectangle(crop, (lx1, ly1), (lx2, ly2), LABEL_BG, -1)
    cv2.putText(crop, label,
                (lx1 + 3, ly2 - baseline - 2),
                FONT, FONT_SCALE, LABEL_FG, FONT_THICK, cv2.LINE_AA)

    return crop


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tracklet {TRACK_ID} from {SEQUENCE} ({CACHE_VARIANT})...")
    tracklet = load_tracklet(SEQUENCE, TRACK_ID)

    img_dir = GT_ROOT / SEQUENCE / "img1"
    if not img_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {img_dir}")
    images = load_images(img_dir)  # sorted list of .jpg paths, 0-indexed

    n = len(tracklet.frames)
    label = f"ID {TRACK_ID}"
    print(f"Tracklet has {n} frames  (frames {tracklet.frames[0]}–{tracklet.frames[-1]})")

    saved = 0
    for local_i, (frame_idx, bbox) in enumerate(zip(tracklet.frames, tracklet.bboxes)):
        if local_i % STEP != 0:
            continue
        if MAX_FRAMES is not None and saved >= MAX_FRAMES:
            break

        if frame_idx >= len(images):
            print(f"  Warning: frame {frame_idx} out of range ({len(images)} images), skipping")
            continue

        img = cv2.imread(str(images[frame_idx]))
        if img is None:
            print(f"  Warning: could not read {images[frame_idx]}, skipping")
            continue

        crop = draw_crop(img, bbox, label, EXPAND)

        out_path = OUT_DIR / f"frame_{frame_idx:05d}_local_{local_i:04d}.jpg"
        cv2.imwrite(str(out_path), crop)
        saved += 1

    print(f"Saved {saved} crops to:\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
