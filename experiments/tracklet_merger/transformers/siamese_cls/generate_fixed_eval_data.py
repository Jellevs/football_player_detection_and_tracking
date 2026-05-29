"""
Generate FIXED validation and test sets that match the inference distribution.

Unlike generate_train_data_real_synth.py, this script:
  - NO synthetic positive pairs
  - NO negative downsampling
  - ALL real pairs from the splitter output (same as inference)
  - Uses the same purity thresholds for labeling

These files should be generated ONCE and never changed. Only training data
should be varied across experiments.

Usage:
    python generate_fixed_eval_data.py

Output:
    train_data/fixed_eval/pairs_valid.pkl
    train_data/fixed_eval/pairs_test.pkl

Then point all training scripts to load val/test from train_data/fixed_eval/
instead of their own data directories.
"""

import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List
from collections import Counter

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4]))

from experiments.tracklet_merger.transformers.utils import (
    extract_frame_features, compute_pairwise_features,
)
from tracklets.split_tracklets import split_tracklets
from utils.config import SplitterConfig

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT        = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\soccernet-player-tracking")
CACHE_SPLIT_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\cache_split")
CACHE_ROOT       = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\cache")
OUTPUT_ROOT      = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\transformers\train_data\fixed_eval")

# ---------------------------------------------------------------------------
# Sequence splits (from existing data generation)
# ---------------------------------------------------------------------------
VALID_SEQUENCES = [
    'SNPT-021', 'SNPT-022', 'SNPT-023', 'SNPT-024', 'SNPT-025',
    'SNPT-026', 'SNPT-027', 'SNPT-028', 'SNPT-029', 'SNPT-030',
    'SNPT-031', 'SNPT-032', 'SNPT-033', 'SNPT-034', 'SNPT-035',
    'SNPT-036', 'SNPT-037', 'SNPT-038', 'SNPT-039', 'SNPT-040',
    'SNPT-041', 'SNPT-042', 'SNPT-043', 'SNPT-044', 'SNPT-045',
    'SNPT-046', 'SNPT-047', 'SNPT-048', 'SNPT-049', 'SNPT-050',
    'SNPT-051', 'SNPT-052', 'SNPT-053', 'SNPT-054', 'SNPT-055',
    'SNPT-056', 'SNPT-057', 'SNPT-058', 'SNPT-059', 'SNPT-078',
    'SNPT-079', 'SNPT-080', 'SNPT-081', 'SNPT-082', 'SNPT-083',
    'SNPT-084', 'SNPT-085', 'SNPT-086', 'SNPT-087', 'SNPT-088',
    'SNPT-089', 'SNPT-090', 'SNPT-091', 'SNPT-092', 'SNPT-093',
    'SNPT-094', 'SNPT-095', 'SNPT-096',
]

TEST_SEQUENCES = [
    'SNPT-116', 'SNPT-117', 'SNPT-118', 'SNPT-119', 'SNPT-120',
    'SNPT-121', 'SNPT-122', 'SNPT-123', 'SNPT-124', 'SNPT-125',
    'SNPT-126', 'SNPT-127', 'SNPT-128', 'SNPT-129', 'SNPT-130',
    'SNPT-131', 'SNPT-132', 'SNPT-133', 'SNPT-134', 'SNPT-135',
    'SNPT-136', 'SNPT-137', 'SNPT-138', 'SNPT-139', 'SNPT-140',
    'SNPT-141', 'SNPT-142', 'SNPT-143', 'SNPT-144', 'SNPT-145',
    'SNPT-146', 'SNPT-147', 'SNPT-148', 'SNPT-149', 'SNPT-150',
    'SNPT-187', 'SNPT-188', 'SNPT-189', 'SNPT-190', 'SNPT-191',
    'SNPT-192', 'SNPT-193', 'SNPT-194', 'SNPT-195', 'SNPT-196',
    'SNPT-197', 'SNPT-198', 'SNPT-199', 'SNPT-200',
]

# ---------------------------------------------------------------------------
# Settings — same labeling as real_synth, but NO curation
# ---------------------------------------------------------------------------
MIN_TRACKLET_LEN     = 0
PURITY_THRESHOLD_POS = 0.6   # strict for positives
PURITY_THRESHOLD_NEG = 0.6   # looser for negatives

SPLITTER_CFG = SplitterConfig()


# ---------------------------------------------------------------------------
# GT labeling (identical to real_synth)
# ---------------------------------------------------------------------------

def get_gt_id(tracklet, purity_threshold: float = 0.6) -> Optional[int]:
    """Extract majority GT ID if tracklet is sufficiently pure."""
    gt_ids = tracklet.gt_attributes.get("track_ids", [])
    valid = [g for g in gt_ids
             if g is not None and not (isinstance(g, float) and np.isnan(g))]
    if not valid:
        return None
    majority = max(set(valid), key=valid.count)
    purity = valid.count(majority) / len(valid)
    if purity < purity_threshold:
        return None
    return majority


# ---------------------------------------------------------------------------
# Post-split cache (same as real_synth)
# ---------------------------------------------------------------------------

def load_or_compute_split_tracklets(sequence: str) -> Optional[dict]:
    """Load post-split tracklets, running the splitter if not cached."""
    CACHE_SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
    split_cache_path = CACHE_SPLIT_ROOT / f"cache_split_{sequence}.pkl"

    if split_cache_path.exists():
        with open(split_cache_path, "rb") as f:
            return pickle.load(f)

    attr_cache = CACHE_ROOT / f"cache_attributes_{sequence}.pkl"
    if not attr_cache.exists():
        return None

    tqdm.write(f"  {sequence}: running splitter ...")
    with open(attr_cache, "rb") as f:
        tracklets = pickle.load(f)

    n_before = len(tracklets)
    tracklets = split_tracklets(tracklets, SPLITTER_CFG)
    tqdm.write(f"  {sequence}: {n_before} -> {len(tracklets)} after splitting")

    with open(split_cache_path, "wb") as f:
        pickle.dump(tracklets, f)
    return tracklets


# ---------------------------------------------------------------------------
# Pair generation — ALL pairs, NO synthetics, NO downsampling
# ---------------------------------------------------------------------------

def generate_all_pairs(sequences: List[str], split_name: str) -> List[dict]:
    """Generate ALL non-overlapping pairs from real splitter output."""
    all_pairs = []
    total_pos = 0
    total_neg = 0
    total_ambiguous = 0

    for seq in tqdm(sequences, desc=split_name):
        tracklets = load_or_compute_split_tracklets(seq)
        if tracklets is None:
            continue

        valid_tids = [
            tid for tid, t in tracklets.items()
            if len(t.frames) >= MIN_TRACKLET_LEN and t.embeddings
        ]

        if len(valid_tids) < 2:
            continue

        seq_pos = 0
        seq_neg = 0
        seq_ambiguous = 0

        for i in range(len(valid_tids)):
            for j in range(i + 1, len(valid_tids)):
                t_a = tracklets[valid_tids[i]]
                t_b = tracklets[valid_tids[j]]

                # Skip temporally overlapping (same as inference)
                if set(t_a.frames) & set(t_b.frames):
                    continue

                # Label using same dual-purity thresholds as real_synth
                gt_a_strict = get_gt_id(t_a, PURITY_THRESHOLD_POS)
                gt_b_strict = get_gt_id(t_b, PURITY_THRESHOLD_POS)
                gt_a_loose = get_gt_id(t_a, PURITY_THRESHOLD_NEG)
                gt_b_loose = get_gt_id(t_b, PURITY_THRESHOLD_NEG)

                if gt_a_strict is not None and gt_b_strict is not None and gt_a_strict == gt_b_strict:
                    label = 1
                    gt_id_a, gt_id_b = gt_a_strict, gt_b_strict
                elif gt_a_loose is not None and gt_b_loose is not None and gt_a_loose != gt_b_loose:
                    label = 0
                    gt_id_a, gt_id_b = gt_a_loose, gt_b_loose
                elif gt_a_loose is None or gt_b_loose is None:
                    label = 0
                    gt_id_a = gt_a_loose if gt_a_loose is not None else -1
                    gt_id_b = gt_b_loose if gt_b_loose is not None else -1
                else:
                    # Ambiguous: both pass loose, same ID, but one fails strict
                    seq_ambiguous += 1
                    continue

                pw = compute_pairwise_features(t_a, t_b)

                pair = {
                    "sequence": seq,
                    "tid_a": valid_tids[i],
                    "tid_b": valid_tids[j],
                    "pairwise": pw,
                    "label": label,
                    "is_synthetic": False,
                    "weight": 1.0,
                    "gt_id_a": gt_id_a,
                    "gt_id_b": gt_id_b,
                }

                all_pairs.append(pair)
                if label == 1:
                    seq_pos += 1
                else:
                    seq_neg += 1

        total_pos += seq_pos
        total_neg += seq_neg
        total_ambiguous += seq_ambiguous

        tqdm.write(
            f"  {seq}: {len(valid_tids)} tracklets -> "
            f"{seq_pos + seq_neg} pairs (pos={seq_pos}, neg={seq_neg}, "
            f"ambiguous={seq_ambiguous})"
        )

    return all_pairs, total_pos, total_neg, total_ambiguous


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for split_name, sequences in [("valid", VALID_SEQUENCES), ("test", TEST_SEQUENCES)]:
        print(f"\n{'='*60}")
        print(f"Generating FIXED {split_name} set ({len(sequences)} sequences)")
        print(f"  NO synthetic positives")
        print(f"  NO negative downsampling")
        print(f"  ALL real pairs from splitter output")
        print(f"{'='*60}")

        pairs, n_pos, n_neg, n_ambiguous = generate_all_pairs(sequences, split_name)

        if not pairs:
            print(f"  [ERROR] No pairs generated for {split_name}")
            continue

        # Shuffle (for consistency with training data format)
        rng = np.random.RandomState(42)
        rng.shuffle(pairs)

        # Save
        out_path = OUTPUT_ROOT / f"pairs_{split_name}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(pairs, f)

        # Stats
        neg_pos_ratio = n_neg / max(n_pos, 1)
        print(f"\n  {split_name} set:")
        print(f"    Total pairs     : {len(pairs):,}")
        print(f"    Positive        : {n_pos:,}  ({100*n_pos/max(len(pairs),1):.1f}%)")
        print(f"    Negative        : {n_neg:,}  ({100*n_neg/max(len(pairs),1):.1f}%)")
        print(f"    Ambiguous (skip): {n_ambiguous:,}")
        print(f"    Neg:Pos ratio   : {neg_pos_ratio:.1f}:1")
        print(f"    Saved           : {out_path}")

    print(f"\n{'='*60}")
    print(f"Done. These files should NEVER be regenerated.")
    print(f"Only modify training data generation scripts.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
