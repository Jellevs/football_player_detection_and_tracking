"""
Generate pair metadata for the transformer tracklet merger — balanced variant.

Identical to generate_train_data.py but uses NEGATIVE_RATIO = 2.0 for the
train split, matching the XGBoost data generation setup.  Valid and test
splits keep all negatives for unbiased evaluation.

Output:
    train_data/transformer_data_balanced/pairs_{split}.pkl
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from transformer.dataset import compute_pairwise_features

# ---------------------------------------------------------------------------
# Paths — adjust to your environment
# ---------------------------------------------------------------------------
DATA_ROOT   = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\soccernet-player-tracking")
CACHE_ROOT  = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\cache")
OUTPUT_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\train_data\transformer_data_balanced")

# ---------------------------------------------------------------------------
# Settings — run this script once per split
# ---------------------------------------------------------------------------
SPLITS_TO_PROCESS = ["train"]     # change to ["valid"] or ["test"] per run
SPLIT_OUTPUT_NAME = "train"       # change to "valid" or "test" per run

# 2:1 for train (matches XGBoost); keep None for valid/test
NEGATIVE_RATIO    = 2.0
MIN_TRACKLET_LEN  = 5
MAX_TEMPORAL_GAP  = None          # frames; None = no limit


def get_gt_id(tracklet) -> Optional[int]:
    """Extract majority ground-truth track ID from a tracklet."""
    gt_ids = tracklet.gt_attributes.get("track_ids", [])
    valid = [g for g in gt_ids
             if g is not None and not (isinstance(g, float) and np.isnan(g))]
    if not valid:
        return None
    return max(set(valid), key=valid.count)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_pairs  = []
    total_pos  = 0
    total_neg  = 0

    for split in SPLITS_TO_PROCESS:
        split_root = DATA_ROOT / split
        if not split_root.exists():
            print(f"[SKIP] Split folder not found: {split_root}")
            continue

        sequences = sorted([d.name for d in split_root.iterdir() if d.is_dir()])
        print(f"\n{'='*60}")
        print(f"Processing split: {split}  ({len(sequences)} sequences)")
        print(f"Split root: {split_root}")
        print(f"Cache root: {CACHE_ROOT}")
        if sequences:
            print(f"First sequence folder: {sequences[0]}")
            print(f"Expected cache: {CACHE_ROOT / f'cache_attributes_{sequences[0]}.pkl'}")
        # Show available caches
        available = sorted(CACHE_ROOT.glob("cache_attributes_*.pkl"))
        print(f"Available caches: {len(available)}")
        if available:
            print(f"  Example: {available[0].name}")
        print(f"{'='*60}")

        for seq in tqdm(sequences, desc=split):
            cache_path = CACHE_ROOT / f"cache_attributes_{seq}.pkl"
            if not cache_path.exists():
                tqdm.write(f"  [MISS] {cache_path.name}")
                continue

            with open(cache_path, "rb") as f:
                tracklets = pickle.load(f)

            # Filter valid tracklets
            valid_tids = [
                tid for tid, t in tracklets.items()
                if len(t.frames) >= MIN_TRACKLET_LEN and t.embeddings
            ]

            if len(valid_tids) < 2:
                continue

            # Generate all valid non-overlapping pairs
            seq_pairs = []
            for i in range(len(valid_tids)):
                for j in range(i + 1, len(valid_tids)):
                    t_a = tracklets[valid_tids[i]]
                    t_b = tracklets[valid_tids[j]]

                    # Skip temporally overlapping
                    if set(t_a.frames) & set(t_b.frames):
                        continue

                    # Temporal gap check
                    if MAX_TEMPORAL_GAP is not None:
                        end_a, start_b = t_a.frames[-1], t_b.frames[0]
                        end_b, start_a = t_b.frames[-1], t_a.frames[0]
                        if end_a <= start_b:
                            gap = start_b - end_a
                        elif end_b <= start_a:
                            gap = start_a - end_b
                        else:
                            continue
                        if gap > MAX_TEMPORAL_GAP:
                            continue

                    # Ground-truth label
                    gt_a = get_gt_id(t_a)
                    gt_b = get_gt_id(t_b)
                    label = int(gt_a is not None and gt_b is not None and gt_a == gt_b)

                    # Pairwise features
                    pw = compute_pairwise_features(t_a, t_b)

                    seq_pairs.append({
                        "sequence": seq,
                        "tid_a":    valid_tids[i],
                        "tid_b":    valid_tids[j],
                        "pairwise": pw,
                        "label":    label,
                    })

            # Negative downsampling
            if NEGATIVE_RATIO is not None and seq_pairs:
                pos = [p for p in seq_pairs if p["label"] == 1]
                neg = [p for p in seq_pairs if p["label"] == 0]
                n_keep = min(len(neg), int(len(pos) * NEGATIVE_RATIO))
                if n_keep < len(neg):
                    rng = np.random.RandomState(42)
                    idx = rng.choice(len(neg), n_keep, replace=False)
                    neg = [neg[k] for k in idx]
                seq_pairs = pos + neg

            pos_count = sum(1 for p in seq_pairs if p["label"] == 1)
            neg_count = len(seq_pairs) - pos_count
            total_pos += pos_count
            total_neg += neg_count
            tqdm.write(
                f"  {seq}: {len(valid_tids)} tracklets → "
                f"{len(seq_pairs)} pairs (pos={pos_count}, neg={neg_count})"
            )
            all_pairs.extend(seq_pairs)

    if not all_pairs:
        print("\n[ERROR] No pairs generated. Check cache paths.")
        return

    # Shuffle
    rng = np.random.RandomState(42)
    rng.shuffle(all_pairs)

    # Save pickle (used by dataset for per-frame features)
    out_path = OUTPUT_ROOT / f"pairs_{SPLIT_OUTPUT_NAME}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(all_pairs, f)

    # Save CSV summary
    import pandas as pd
    pw_names = [
        "temporal_gap", "spatial_distance", "endpoint_dx", "endpoint_dy",
        "bbox_height_ratio", "reid_cosine_sim", "siglip_cosine_sim",
        "jersey_match", "jersey_conflict", "jersey_both_confident",
        "team_match", "team_both_consistent",
    ]
    csv_rows = []
    for p in all_pairs:
        row = {
            "sequence": p["sequence"],
            "tid_a": p["tid_a"],
            "tid_b": p["tid_b"],
            "label": p["label"],
        }
        for name, val in zip(pw_names, p["pairwise"]):
            row[f"pairwise_{name}"] = float(val)
        csv_rows.append(row)
    csv_path = OUTPUT_ROOT / f"pairs_{SPLIT_OUTPUT_NAME}.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"Done.")
    print(f"  Total pairs : {len(all_pairs):,}")
    print(f"  Positive    : {total_pos:,}  ({100*total_pos/max(len(all_pairs),1):.1f}%)")
    print(f"  Negative    : {total_neg:,}  ({100*total_neg/max(len(all_pairs),1):.1f}%)")
    print(f"  Pickle      : {out_path}")
    print(f"  CSV         : {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
