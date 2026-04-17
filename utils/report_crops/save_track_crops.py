"""
Save all crops belonging to a chosen set of track IDs from one sequence.

Used to produce report figures where the same physical player ends up with
multiple track IDs (e.g. an identity-switch fragment pair) — running this
gives one folder per track ID with every detection crop, and a side-by-side
strip showing the two tracks together.

Edit the constants at the top of `main()` and run:
    python -m utils.report_crops.save_track_crops
"""

import pickle
import sys
from pathlib import Path

import cv2
import numpy as np


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


def _resize_to_height(crop, h):
    ratio = h / crop.shape[0]
    new_w = max(1, int(round(crop.shape[1] * ratio)))
    return cv2.resize(crop, (new_w, h), interpolation=cv2.INTER_AREA)


def save_track_crops(tracklet, images, out_dir):
    """Write every detection crop of `tracklet` to `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for i, frame_idx in enumerate(tracklet.frames):
        crop = _load_crop(_resolve_image(images, frame_idx), tracklet.bboxes[i])
        if crop is None:
            continue
        path = out_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(path), crop)
        saved.append((frame_idx, crop))
    return saved


def build_strip(saved_per_track, row_height=200, pad=6, max_per_row=20):
    """Build a labelled strip: one row per track id, ordered by frame."""
    rows = []
    for track_id, saved in saved_per_track.items():
        if not saved:
            continue
        saved = sorted(saved, key=lambda x: x[0])
        tiles = []
        for frame_idx, crop in saved:
            r = _resize_to_height(crop, row_height)
            r = cv2.copyMakeBorder(r, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(40, 40, 40))
            cap = np.full((20, r.shape[1], 3), 30, dtype=np.uint8)
            cv2.putText(cap, f"f={frame_idx}", (3, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(np.vstack([r, cap]))

        # Wrap into multiple sub-rows if too many tiles
        sub_rows = [tiles[i:i + max_per_row] for i in range(0, len(tiles), max_per_row)]
        for sub in sub_rows:
            tallest = max(t.shape[0] for t in sub)
            padded = []
            for t in sub:
                if t.shape[0] < tallest:
                    pad_strip = np.full((tallest - t.shape[0], t.shape[1], 3), 30, dtype=np.uint8)
                    t = np.vstack([t, pad_strip])
                padded.append(t)
                padded.append(np.full((tallest, pad, 3), 30, dtype=np.uint8))
            row_img = np.hstack(padded[:-1])

            label = np.full((30, row_img.shape[1], 3), 20, dtype=np.uint8)
            cv2.putText(label, f"track {track_id}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            rows.append(np.vstack([label, row_img]))

    if not rows:
        return None
    max_w = max(r.shape[1] for r in rows)
    padded_rows = []
    for r in rows:
        if r.shape[1] < max_w:
            pad_strip = np.full((r.shape[0], max_w - r.shape[1], 3), 30, dtype=np.uint8)
            r = np.hstack([r, pad_strip])
        padded_rows.append(r)
        padded_rows.append(np.full((pad, max_w, 3), 30, dtype=np.uint8))
    return np.vstack(padded_rows[:-1])


def main():
    # ---- edit these for each report figure ----
    sequence = "SNPT-116"
    track_ids = [25, 49]
    out = r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\report\track_crops"
    # --------------------------------------------

    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    import settings
    from utils.build import build_paths

    paths = build_paths(sequence)
    images = _load_images_for_sequence(sequence, settings.DATA_ROOT.parent, paths.img_path)
    if not images:
        print(f"No images found for {sequence}")
        sys.exit(1)

    cache_path = paths.set_cache_path("attributes", sequence)
    if not cache_path.exists():
        print(f"Cache not found: {cache_path}")
        sys.exit(1)

    with open(cache_path, "rb") as f:
        tracklets = pickle.load(f)

    print(f"Loaded {len(tracklets)} tracklets for {sequence}")

    out_root = Path(out) / sequence
    out_root.mkdir(parents=True, exist_ok=True)

    saved_per_track = {}
    for tid in track_ids:
        if tid not in tracklets:
            print(f"  track {tid}: not present in cache")
            continue
        tr = tracklets[tid]
        track_dir = out_root / f"track_{tid:04d}"
        saved = save_track_crops(tr, images, track_dir)
        saved_per_track[tid] = saved
        print(f"  track {tid}: {len(saved)}/{len(tr.frames)} crops saved → {track_dir}")

    strip = build_strip(saved_per_track)
    if strip is not None:
        strip_name = "strip_" + "_".join(f"{t}" for t in track_ids) + ".jpg"
        strip_path = out_root / strip_name
        cv2.imwrite(str(strip_path), strip)
        print(f"\nStrip image: {strip_path}")

    print(f"\nOutput: {out_root}")


if __name__ == "__main__":
    main()
