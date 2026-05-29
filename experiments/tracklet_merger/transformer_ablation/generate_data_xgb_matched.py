"""
Generate transformer training data using XGBoost's optimal data config.

Matches XGBoost sweep winner: neg_ratio=3, min_tracklet_len=0, purity=0.6, split=True.
NO synthetic pairs (XGBoost doesn't use them).

Run once per split:
  1. Set SPLITS_TO_PROCESS = ["train"], SPLIT_OUTPUT_NAME = "train"  → run
  2. Set SPLITS_TO_PROCESS = ["valid"], SPLIT_OUTPUT_NAME = "valid"  → run
  3. Set SPLITS_TO_PROCESS = ["test"],  SPLIT_OUTPUT_NAME = "test"   → run

Output: train_data/transformer_xgb_matched/pairs_{split}.pkl
"""

import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

from transformer.dataset import (
    extract_frame_features, compute_pairwise_features,
    _tracklet_stats,
)
from tracklets.split_tracklets import split_tracklets
from utils.config import SplitterConfig

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT        = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\soccernet-player-tracking")
CACHE_ROOT       = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\cache")
CACHE_SPLIT_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\cache_split")
OUTPUT_ROOT      = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\train_data\transformer_xgb_matched")

# ── Settings (matching XGBoost's optimal config) ─────────────────────────────
SPLITS_TO_PROCESS = ["valid"]     # change per run: ["train"], ["valid"], ["test"]
SPLIT_OUTPUT_NAME = "valid"

NEGATIVE_RATIO    = 3             # XGBoost best: 3
MIN_TRACKLET_LEN  = 0             # XGBoost best: 0
PURITY_THRESHOLD  = 0.60          # XGBoost best: 0.6
USE_SPLIT_DATA    = True          # XGBoost best: True (post-split tracklets)

SPLITTER_CFG = SplitterConfig()


def get_gt_id(tracklet) -> Optional[int]:
    gt_ids = tracklet.gt_attributes.get("track_ids", [])
    valid = [g for g in gt_ids
             if g is not None and not (isinstance(g, float) and np.isnan(g))]
    if not valid:
        return None
    majority = max(set(valid), key=valid.count)
    purity = valid.count(majority) / len(valid)
    if purity < PURITY_THRESHOLD:
        return None
    return majority


def load_tracklets(sequence: str) -> Optional[dict]:
    """Load post-split or pre-split tracklets depending on config."""
    if USE_SPLIT_DATA:
        CACHE_SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
        split_cache = CACHE_SPLIT_ROOT / f"cache_split_{sequence}.pkl"
        if split_cache.exists():
            with open(split_cache, "rb") as f:
                return pickle.load(f)
        # Cache miss: run splitter
        attr_cache = CACHE_ROOT / f"cache_attributes_{sequence}.pkl"
        if not attr_cache.exists():
            return None
        with open(attr_cache, "rb") as f:
            tracklets = pickle.load(f)
        tracklets = split_tracklets(tracklets, SPLITTER_CFG)
        with open(split_cache, "wb") as f:
            pickle.dump(tracklets, f)
        return tracklets
    else:
        attr_cache = CACHE_ROOT / f"cache_attributes_{sequence}.pkl"
        if not attr_cache.exists():
            return None
        with open(attr_cache, "rb") as f:
            return pickle.load(f)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)

    all_pairs = []
    total_pos = 0
    total_neg = 0

    for split in SPLITS_TO_PROCESS:
        split_root = DATA_ROOT / split
        if not split_root.exists():
            print(f"[SKIP] {split_root}")
            continue

        sequences = sorted([d.name for d in split_root.iterdir() if d.is_dir()])
        print(f"\n{'='*60}")
        print(f"Split: {split}  ({len(sequences)} sequences)")
        print(f"Config: neg_ratio={NEGATIVE_RATIO}, minlen={MIN_TRACKLET_LEN}, "
              f"purity={PURITY_THRESHOLD}, split={USE_SPLIT_DATA}")
        print(f"{'='*60}")

        for seq in tqdm(sequences, desc=split):
            tracklets = load_tracklets(seq)
            if tracklets is None:
                continue

            valid_tids = [
                tid for tid, t in tracklets.items()
                if len(t.frames) >= MIN_TRACKLET_LEN and t.embeddings
            ]
            if len(valid_tids) < 2:
                continue

            seq_pairs = []
            for i in range(len(valid_tids)):
                for j in range(i + 1, len(valid_tids)):
                    t_a = tracklets[valid_tids[i]]
                    t_b = tracklets[valid_tids[j]]

                    if set(t_a.frames) & set(t_b.frames):
                        continue

                    gt_a = get_gt_id(t_a)
                    gt_b = get_gt_id(t_b)
                    if gt_a is None or gt_b is None:
                        label = 0
                    else:
                        label = int(gt_a == gt_b)

                    # Extended pairwise (12 base + 2*11 stats = 34)
                    pw_base = compute_pairwise_features(t_a, t_b)
                    feats_a, frames_a = extract_frame_features(t_a)
                    feats_b, frames_b = extract_frame_features(t_b)
                    stats_a = _tracklet_stats(feats_a, frames_a)
                    stats_b = _tracklet_stats(feats_b, frames_b)
                    extended_pw = np.concatenate([pw_base, stats_a, stats_b])

                    seq_pairs.append({
                        "sequence": seq,
                        "tid_a": valid_tids[i],
                        "tid_b": valid_tids[j],
                        "pairwise": extended_pw,
                        "label": label,
                        "is_synthetic": False,
                    })

            # Negative downsampling per sequence
            if NEGATIVE_RATIO is not None and seq_pairs:
                pos = [p for p in seq_pairs if p["label"] == 1]
                neg = [p for p in seq_pairs if p["label"] == 0]
                n_keep = min(len(neg), int(max(len(pos), 1) * NEGATIVE_RATIO))
                if n_keep < len(neg):
                    idx = rng.choice(len(neg), n_keep, replace=False)
                    neg = [neg[k] for k in idx]
                seq_pairs = pos + neg

            pos_count = sum(1 for p in seq_pairs if p["label"] == 1)
            neg_count = len(seq_pairs) - pos_count
            total_pos += pos_count
            total_neg += neg_count
            all_pairs.extend(seq_pairs)

            tqdm.write(f"  {seq}: {len(valid_tids)} tracklets -> "
                       f"{len(seq_pairs)} pairs (pos={pos_count}, neg={neg_count})")

    if not all_pairs:
        print("\n[ERROR] No pairs generated.")
        return

    rng.shuffle(all_pairs)

    out_path = OUTPUT_ROOT / f"pairs_{SPLIT_OUTPUT_NAME}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(all_pairs, f)

    print(f"\n{'='*60}")
    print(f"Done.")
    print(f"  Total pairs : {len(all_pairs):,}")
    print(f"  Positive    : {total_pos:,}  ({100*total_pos/max(len(all_pairs),1):.1f}%)")
    print(f"  Negative    : {total_neg:,}  ({100*total_neg/max(len(all_pairs),1):.1f}%)")
    print(f"  Saved to    : {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
