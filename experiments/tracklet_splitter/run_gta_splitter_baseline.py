"""
run_gta_splitter_baseline.py — Run GTA splitter on BASELINE tracklets.

Loads baseline tracklets with ReID embeddings from the backup attribute cache
(output/backup_cache/), runs split_tracklets() using the same DBSCAN parameters
as the GTA repo (eps=0.7, min_samples=10, max_k=3, min_len=100), and saves
results as MOT txt files for HOTA evaluation.

Purpose
-------
Makes our GTA splitter results directly comparable to the GTA repo's 81.679
HOTA, since both now use the same raw baseline tracklets as input (rather than
post-STR tracklets).

Outputs
-------
    evaluation/SNPT/gta_baseline/data/SNPT-XXX.txt

Usage
-----
    python experiments/tracklet_splitter/run_gta_splitter_baseline.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

# ── repo root importable ──────────────────────────────────────────────────────
REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))

from tracklets.splitters.gta_splitter import split_tracklets

# ── configuration ─────────────────────────────────────────────────────────────

DATA_SPLIT = "test"

# Backup cache = attribute cache built BEFORE the STR splitter runs.
# Track IDs here match the raw baseline MOT files exactly.
BACKUP_CACHE_DIR = REPO / "output" / "backup_cache"

SEQMAP = REPO / "evaluation" / "seqmaps" / f"SNPT-{DATA_SPLIT}.txt"

OUT_DIR = REPO / "evaluation" / "SNPT" / "gta_baseline" / "data"

# GTA repo parameters (from refine_tracklets.py)
EPS         = 0.7
MIN_SAMPLES = 10
MAX_K       = 3
MIN_LEN     = 100   # frames; tracklets shorter than this are kept as-is


# ── helpers ───────────────────────────────────────────────────────────────────

def tracklets_to_mot(tracklets: dict, frame_offset: int = 1) -> list[str]:
    """
    Convert Tracklet objects (xyxy bboxes) to MOT-format lines.

    MOT format: frame, id, x, y, w, h, conf, -1, -1, -1

    The attribute cache uses 0-indexed frames while MOT files use 1-indexed
    frames, so frame_offset=1 is applied by default to align them.
    """
    lines: list[tuple[int, int, str]] = []
    for tid, tracklet in tracklets.items():
        for frame, bbox, score in zip(tracklet.frames, tracklet.bboxes, tracklet.scores):
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            w = x2 - x1
            h = y2 - y1
            conf = float(score)
            mot_frame = int(frame) + frame_offset
            lines.append((
                mot_frame,
                int(tid),
                f"{mot_frame},{int(tid)},{x1:.6f},{y1:.6f},{w:.6f},{h:.6f},{conf:.6f},-1,-1,-1"
            ))
    # Sort by frame then track id
    lines.sort(key=lambda t: (t[0], t[1]))
    return [ln[2] for ln in lines]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── discover sequences ────────────────────────────────────────────────
    if SEQMAP.exists():
        with SEQMAP.open() as f:
            sequences = [
                line.strip() for line in f
                if line.strip() and not line.strip().lower().startswith("name")
            ]
    else:
        sequences = sorted(p.stem for p in BACKUP_CACHE_DIR.glob("cache_attributes_SNPT-*.pkl"))
        sequences = [s.replace("cache_attributes_", "") for s in sequences]

    print(f"Split: {DATA_SPLIT}  |  Sequences: {len(sequences)}")
    print(f"GTA params: eps={EPS}, min_samples={MIN_SAMPLES}, max_k={MAX_K}, min_len={MIN_LEN}")
    print(f"Output dir: {OUT_DIR}\n")

    total_in = total_out = 0
    skipped = []

    for seq in sequences:
        cache_path = BACKUP_CACHE_DIR / f"cache_attributes_{seq}.pkl"

        if not cache_path.exists():
            skipped.append(seq)
            continue

        print(f"[{seq}] Loading backup cache...", end=" ", flush=True)
        with cache_path.open("rb") as f:
            tracklets: dict = pickle.load(f)

        n_in = len(tracklets)

        # Check that tracklets have embeddings
        has_emb = sum(1 for t in tracklets.values() if len(t.embeddings) > 0)
        if has_emb == 0:
            print(f"WARNING: no embeddings found, skipping")
            skipped.append(seq)
            continue

        print(f"{n_in} tracklets ({has_emb} with embeddings)  →  splitting...", end=" ", flush=True)

        split = split_tracklets(
            tracklets,
            eps=EPS,
            min_samples=MIN_SAMPLES,
            max_k=MAX_K,
            len_thres=MIN_LEN,
        )

        n_out = len(split)
        total_in  += n_in
        total_out += n_out
        print(f"{n_out} tracklets after split (+{n_out - n_in})")

        mot_lines = tracklets_to_mot(split)
        out_path = OUT_DIR / f"{seq}.txt"
        with out_path.open("w") as f:
            f.write("\n".join(mot_lines))
            if mot_lines:
                f.write("\n")

    print(f"\nDone.")
    print(f"  Sequences processed : {len(sequences) - len(skipped)}")
    print(f"  Sequences skipped   : {len(skipped)}" + (f" ({skipped})" if skipped else ""))
    print(f"  Total tracklets in  : {total_in}")
    print(f"  Total tracklets out : {total_out}  (+{total_out - total_in} from splitting)")
    print(f"\nMOT files saved to: {OUT_DIR}")
    print("Run HOTA evaluation on this directory to compare against GTA repo (81.679).")


if __name__ == "__main__":
    main()
