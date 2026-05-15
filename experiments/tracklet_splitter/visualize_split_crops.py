"""
visualize_split_crops.py — Visualise post-split fragments for a given original track ID.

Loads the attribute cache for a sequence, runs the configured splitter(s),
then saves crops for every fragment that originated from the specified original
track ID.  Each fragment gets its own sub-folder so you can see how the
original tracklet was divided.

Crops show the bbox drawn on an expanded region of the frame, labelled with
the new fragment track ID (and the original parent ID for reference).

Output layout
-------------
    experiments/tracklet_splitter/output/split_crops/<SEQUENCE>/track_<ORIG_ID>/
        fragment_<NEW_ID>/
            frame_<NNNNN>_local_<MMMM>.jpg
            ...
        fragment_<NEW_ID>/
            ...

Configuration
-------------
Set SEQUENCE, ORIG_TRACK_ID, and SIGNALS below, then run from repo root:

    python experiments/tracklet_splitter/visualize_split_crops.py
"""

from __future__ import annotations

import copy
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))

import settings
from utils.config import SplitterConfig
from utils.data_utils import load_images
from tracklets.splitters.modular_splitter import ModularSplitter
from tracklets.splitters.trajectory_splitter import TrajectorySplitter
from tracklets.splitters import gta_splitter as gta

# ── configuration ─────────────────────────────────────────────────────────────

SEQUENCE      = "SNPT-138"   # ← sequence to inspect
ORIG_TRACK_ID = 17           # ← original (pre-split) track ID to follow
DATA_SPLIT    = "test"

# Which splitter signals to apply — same options as split_tracklets.py:
#   per-tracklet : 'jersey', 'team', 'bbox'
#   cross-tracklet: 'trajectory', 'gta'
SIGNALS = ['jersey']

EXPAND     = 0.8    # fractional padding around bbox (same as visualize_tracklet_crops.py)
STEP       = 1      # save every Nth frame
MAX_FRAMES = None   # cap total crops per fragment; None = all

GT_ROOT   = settings.DATA_ROOT.parent / DATA_SPLIT
CACHE_DIR = REPO / "output" / "cache"
OUT_DIR   = (Path(__file__).parent / "output" / "split_crops"
             / SEQUENCE / f"track_{ORIG_TRACK_ID}")

# ── label style ───────────────────────────────────────────────────────────────

FRAGMENT_COLORS = [
    # (0, 220, 0),     # green
    (0, 100, 255),   # orange
    (255, 80, 80),   # blue
    (0, 200, 255),   # yellow
    (200, 0, 255),   # magenta
    (255, 200, 0),   # cyan
]

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICK = 1


# ── drawing ───────────────────────────────────────────────────────────────────

def draw_crop(img: np.ndarray, bbox_xyxy, label: str,
              expand: float, color) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    bw = x2 - x1
    H, W = img.shape[:2]

    pad_x = int(bw * expand)
    pad_y = int((y2 - y1) * expand)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(W, x2 + pad_x)
    cy2 = min(H, y2 + pad_y)

    crop = img[cy1:cy2, cx1:cx2].copy()

    rx1 = x1 - cx1
    ry1 = y1 - cy1
    rx2 = x2 - cx1
    ry2 = y2 - cy1

    thickness = max(2, bw // 40)
    cv2.rectangle(crop, (rx1, ry1), (rx2, ry2), color, thickness)

    (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)
    lx1 = rx1
    ly1 = max(0, ry1 - th - baseline - 4)
    lx2 = min(crop.shape[1], lx1 + tw + 6)
    ly2 = ly1 + th + baseline + 4

    cv2.rectangle(crop, (lx1, ly1), (lx2, ly2), color, -1)
    cv2.putText(crop, label,
                (lx1 + 3, ly2 - baseline - 2),
                FONT, FONT_SCALE, (0, 0, 0), FONT_THICK, cv2.LINE_AA)

    return crop


# ── splitter pipeline ─────────────────────────────────────────────────────────

_PER_TRACKLET   = {'jersey', 'team', 'bbox'}
_CROSS_TRACKLET = {'trajectory', 'gta'}


def run_splitter(tracklets: dict, signals: list[str],
                 origin_map: dict | None = None) -> dict:
    """
    Run the configured splitter signals.
    origin_map is filled with {original_tid: [resulting_fragment_tids]}.
    Only per-tracklet signals populate origin_map; cross-tracklet signals
    (trajectory, gta) are appended with identity entries for simplicity.
    """
    cfg = SplitterConfig(**settings.SPLITTER)
    per_tracklet   = [s for s in signals if s in _PER_TRACKLET]
    cross_tracklet = [s for s in signals if s in _CROSS_TRACKLET]

    label = '+'.join(signals) if signals else 'none'
    print(f"Running splitter [{label}]...")

    if per_tracklet:
        splitter  = ModularSplitter(cfg, signals=per_tracklet)
        tracklets = _split_modular(tracklets, splitter, origin_map)
    elif origin_map is not None:
        # No per-tracklet signals — populate identity entries
        for tid in tracklets:
            origin_map[tid] = [tid]

    if 'trajectory' in cross_tracklet:
        traj = TrajectorySplitter(cfg)
        tracklets = traj.split_all_tracklets(tracklets)

    if 'gta' in cross_tracklet:
        print("  Running GTA (DBSCAN embedding clustering)...")
        tracklets = gta.split_tracklets(tracklets)

    return tracklets


def _split_modular(tracklets: dict, splitter: ModularSplitter,
                   origin_map: dict | None = None) -> dict:
    """
    Split tracklets with the modular splitter.
    origin_map, if provided, is filled with {original_tid: [new_fragment_tids]}.
    Tracklets that are not split map to [original_tid] (identity).
    """
    max_id    = max(tracklets.keys()) if tracklets else 0
    next_id   = max_id + 1
    result    = {}
    n_splits  = 0
    for tid, tracklet in tracklets.items():
        fragments = splitter.split_tracklet(tracklet, next_id)
        if fragments:
            n_splits += 1
            frag_ids = []
            for f in fragments:
                result[f.track_id] = f
                frag_ids.append(f.track_id)
                next_id = max(next_id, f.track_id + 1)
            if origin_map is not None:
                origin_map[tid] = frag_ids
        else:
            result[tid] = tracklet
            if origin_map is not None:
                origin_map[tid] = [tid]
    print(f"  {n_splits}/{len(tracklets)} tracklets split  →  {len(result)} total")
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # Load cache
    cache_path = CACHE_DIR / f"cache_attributes_{SEQUENCE}.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    with cache_path.open("rb") as f:
        tracklets = pickle.load(f)

    if ORIG_TRACK_ID not in tracklets:
        raise KeyError(f"Track {ORIG_TRACK_ID} not in {SEQUENCE}. "
                       f"Available: {sorted(tracklets.keys())}")

    print(f"Loaded {len(tracklets)} tracklets from cache.")

    # Run splitter on a deep copy, tracking which new IDs come from each original tid
    origin_map: dict[int, list[int]] = {}
    split = run_splitter(copy.deepcopy(tracklets), SIGNALS, origin_map)

    # origin_map[ORIG_TRACK_ID] gives the list of fragment IDs that resulted from it
    frag_ids = origin_map.get(ORIG_TRACK_ID)
    if not frag_ids:
        raise KeyError(f"Track {ORIG_TRACK_ID} not found in origin_map. "
                       f"Available: {sorted(origin_map.keys())}")

    fragments = {fid: split[fid] for fid in frag_ids if fid in split}

    print(f"\nFound {len(fragments)} fragment(s) from original track {ORIG_TRACK_ID}:")
    for fid, frag in sorted(fragments.items()):
        print(f"  Fragment {fid}: {len(frag.frames)} frames "
              f"(frames {frag.frames[0]}–{frag.frames[-1]})")

    # Load images
    img_dir = GT_ROOT / SEQUENCE / "img1"
    if not img_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {img_dir}")
    images = load_images(img_dir)

    # Save crops per fragment
    for frag_idx, (fid, frag) in enumerate(sorted(fragments.items())):
        color = FRAGMENT_COLORS[frag_idx % len(FRAGMENT_COLORS)]
        frag_dir = OUT_DIR / f"fragment_{fid}"
        frag_dir.mkdir(parents=True, exist_ok=True)

        if len(fragments) == 1:
            label = f"ID {fid}"
        else:
            label = f"ID {fid}"

        saved = 0
        for local_i, (frame_idx, bbox) in enumerate(zip(frag.frames, frag.bboxes)):
            if local_i % STEP != 0:
                continue
            if MAX_FRAMES is not None and saved >= MAX_FRAMES:
                break
            if frame_idx >= len(images):
                continue
            img = cv2.imread(str(images[frame_idx]))
            if img is None:
                continue

            crop = draw_crop(img, bbox, label, EXPAND, color)
            out_path = frag_dir / f"frame_{frame_idx:05d}_local_{local_i:04d}.jpg"
            cv2.imwrite(str(out_path), crop)
            saved += 1

        print(f"  Fragment {fid}: saved {saved} crops → {frag_dir}")

    print(f"\nDone. All crops under:\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
