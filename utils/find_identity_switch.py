"""
Find tracklets that contain an identity switch and export their crops.

An identity switch is a tracklet whose detections are associated with more than
one ground-truth track ID. For each qualifying tracklet we:

  1. Save one folder per GT identity with the individual crops.
  2. Save a single annotated grid image that shows the transition in order,
     with coloured borders and GT-ID labels — convenient for documentation.

Usage:
    python -m utils.find_identity_switch --sequence SNMOT-116
    python -m utils.find_identity_switch --sequence SNMOT-116 --min_per_id 10 --top 5
"""

import pickle
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


# Distinct BGR colours for GT-ID borders in the grid image
_PALETTE = [
    (0, 200, 0),     # green
    (0, 0, 220),     # red
    (220, 160, 0),   # blue
    (0, 180, 220),   # yellow
    (200, 0, 200),   # magenta
    (220, 220, 0),   # cyan
]


def _gt_sequence(tracklet):
    """Return the per-detection GT track IDs, with None/NaN filtered to -1."""
    raw = tracklet.gt_attributes.get('track_ids', []) or []
    out = []
    for v in raw:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append(-1)
        else:
            out.append(int(v))
    return out


def find_switch_tracklets(tracklets, min_per_id=5):
    """
    Return a list of (track_id, tracklet, gt_counts) for tracklets that
    contain at least two distinct GT IDs, each with >= min_per_id detections.
    Sorted by total length, longest first.
    """
    hits = []
    for tid, tr in tracklets.items():
        gts = _gt_sequence(tr)
        counts = Counter(g for g in gts if g >= 0)
        qualifying = {g: c for g, c in counts.items() if c >= min_per_id}
        if len(qualifying) >= 2:
            hits.append((tid, tr, qualifying))
    hits.sort(key=lambda x: len(x[1].frames), reverse=True)
    return hits


def _resolve_image(images, frame_idx):
    """Frames follow MOT 1-based convention; images is a 0-indexed sorted list."""
    for cand in (frame_idx - 1, frame_idx):
        if 0 <= cand < len(images):
            return images[cand]
    return None


def _load_crop(img_path, bbox):
    if img_path is None:
        return None
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    x1, y1, x2, y2 = map(int, bbox)
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    return crop if crop.size else None


def _resize_to_height(crop, h):
    ratio = h / crop.shape[0]
    new_w = max(1, int(round(crop.shape[1] * ratio)))
    return cv2.resize(crop, (new_w, h), interpolation=cv2.INTER_AREA)


def _build_grid(crops_with_meta, gt_to_color, title, row_height=180, pad=6):
    """crops_with_meta: list of (crop_bgr, gt_id, frame_idx)."""
    resized = []
    for crop, gt_id, frame_idx in crops_with_meta:
        r = _resize_to_height(crop, row_height)
        color = gt_to_color.get(gt_id, (128, 128, 128))
        r = cv2.copyMakeBorder(r, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=color)
        # Small caption strip with GT id + frame number
        caption = np.full((22, r.shape[1], 3), 30, dtype=np.uint8)
        cv2.putText(
            caption, f"gt={gt_id}  f={frame_idx}", (4, 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )
        resized.append(np.vstack([r, caption]))

    # Pack into rows so total width stays manageable
    max_row_w = 1800
    rows = [[]]
    cur_w = 0
    for tile in resized:
        tw = tile.shape[1] + pad
        if cur_w + tw > max_row_w and rows[-1]:
            rows.append([])
            cur_w = 0
        rows[-1].append(tile)
        cur_w += tw

    row_imgs = []
    for row in rows:
        if not row:
            continue
        tallest = max(t.shape[0] for t in row)
        padded = []
        for t in row:
            if t.shape[0] < tallest:
                pad_strip = np.full((tallest - t.shape[0], t.shape[1], 3), 30, dtype=np.uint8)
                t = np.vstack([t, pad_strip])
            padded.append(t)
            padded.append(np.full((tallest, pad, 3), 30, dtype=np.uint8))
        row_imgs.append(np.hstack(padded[:-1]))

    max_w = max(r.shape[1] for r in row_imgs)
    padded_rows = []
    for r in row_imgs:
        if r.shape[1] < max_w:
            pad_strip = np.full((r.shape[0], max_w - r.shape[1], 3), 30, dtype=np.uint8)
            r = np.hstack([r, pad_strip])
        padded_rows.append(r)
        padded_rows.append(np.full((pad, max_w, 3), 30, dtype=np.uint8))
    body = np.vstack(padded_rows[:-1])

    title_strip = np.full((40, body.shape[1], 3), 20, dtype=np.uint8)
    cv2.putText(
        title_strip, title, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return np.vstack([title_strip, body])


def export_switch(track_id, tracklet, gt_counts, images, out_dir, max_crops_per_id=20, label=None):
    """Write per-GT-ID crop folders and an annotated grid image."""
    folder_name = f"track_{track_id:04d}" if label is None else f"{label}_track_{track_id:04d}"
    out_dir = Path(out_dir) / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    gts = _gt_sequence(tracklet)
    frames = tracklet.frames
    bboxes = tracklet.bboxes

    gt_to_color = {
        gt_id: _PALETTE[i % len(_PALETTE)]
        for i, gt_id in enumerate(sorted(gt_counts.keys()))
    }

    # Bucket detection indices per GT id, then subsample evenly
    per_id_idx = {gt_id: [] for gt_id in gt_counts}
    for i, g in enumerate(gts):
        if g in per_id_idx:
            per_id_idx[g].append(i)
    for gt_id, idxs in per_id_idx.items():
        if len(idxs) > max_crops_per_id:
            step = len(idxs) / max_crops_per_id
            per_id_idx[gt_id] = [idxs[int(k * step)] for k in range(max_crops_per_id)]

    # Save individual crops per identity + collect tiles for the grid
    tiles = []
    for gt_id, idxs in per_id_idx.items():
        id_dir = out_dir / f"gt_{gt_id}"
        id_dir.mkdir(exist_ok=True)
        for i in idxs:
            frame_idx = frames[i]
            crop = _load_crop(_resolve_image(images, frame_idx), bboxes[i])
            if crop is None:
                continue
            cv2.imwrite(str(id_dir / f"frame_{frame_idx:06d}.jpg"), crop)
            tiles.append((crop, gt_id, frame_idx))

    if not tiles:
        return

    # Order tiles by frame so the switch reads left-to-right
    tiles.sort(key=lambda t: t[2])

    summary = ", ".join(f"gt {g}: {c}" for g, c in sorted(gt_counts.items()))
    title = f"track {track_id}  |  {len(tracklet.frames)} detections  |  {summary}"
    grid = _build_grid(tiles, gt_to_color, title)
    cv2.imwrite(str(out_dir / "grid.jpg"), grid)


def _load_images_for_sequence(sequence, split_root, default_img_path):
    from utils.data_utils import load_images
    images = load_images(img_dir=default_img_path)
    if images:
        return images
    for split in ("train", "valid", "test"):
        alt = split_root / split / sequence / "img1"
        if alt.exists():
            images = load_images(img_dir=alt)
            if images:
                return images
    return []


def main():
    min_per_id = 2          # lower → more candidate switches (borderline ones included)
    top_per_sequence = 100  # cap tracks exported per sequence
    max_crops_per_id = 500
    out = r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\report\id_switch"

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    import settings
    from utils.build import build_paths

    out_root = Path(out)
    out_root.mkdir(parents=True, exist_ok=True)

    cache_dir = settings.OUTPUT_ROOT / "cache"
    cache_files = sorted(cache_dir.glob("cache_attributes_*.pkl"))
    print(f"Found {len(cache_files)} cached sequences in {cache_dir}")

    split_root = settings.DATA_ROOT.parent
    total_hits = 0

    for cache_path in cache_files:
        sequence = cache_path.stem.replace("cache_attributes_", "")
        paths = build_paths(sequence)
        images = _load_images_for_sequence(sequence, split_root, paths.img_path)
        if not images:
            print(f"[{sequence}] skipped — no images on disk")
            continue

        with open(cache_path, "rb") as f:
            tracklets = pickle.load(f)

        hits = find_switch_tracklets(tracklets, min_per_id=min_per_id)
        if not hits:
            continue

        print(f"[{sequence}] {len(hits)} switch tracklet(s)")
        total_hits += len(hits)

        for tid, tr, counts in hits[:top_per_sequence]:
            summary = ", ".join(f"gt {g}: {c}" for g, c in sorted(counts.items()))
            print(f"    track {tid:4d}  length={len(tr.frames):5d}  {summary}")
            export_switch(
                tid, tr, counts, images, out_root,
                max_crops_per_id=max_crops_per_id, label=sequence,
            )

    print(f"\nTotal switch tracklets exported: {total_hits}")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()
