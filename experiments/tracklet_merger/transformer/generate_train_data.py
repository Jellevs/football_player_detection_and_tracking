"""
Generate training data for the Temporal Bin Transformer tracklet merger.

Key differences from the original transformer data generation:
  1. Uses **post-split** tracklets (runs the splitter first, caches results),
     matching the exact distribution the merger sees at inference time.
     The original transformer used pre-split tracklets, creating a train/test
     distribution mismatch that hurt generalisation.

  2. **Synthetic positive pair generation**: Long tracklets (>=20 frames) are
     split at random points to create additional positive pairs. Each split
     produces two fragments that are guaranteed to be from the same identity.
     This addresses the core data scarcity problem: positive pairs are rare
     (same player, non-overlapping, different tracklets) but synthetic
     splits are plentiful and perfectly labeled.

  3. Uses the same pairwise feature computation as XGBoost for consistency.

Usage:
    # Edit SPLITS_TO_PROCESS and SPLIT_OUTPUT_NAME, then:
    python generate_train_data.py

Output:
    train_data/temporal_bin_transformer/pairs_{split}.pkl
"""



import copy
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Tuple

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

from transformer.dataset import (
    extract_frame_features, compute_pairwise_features,
    _tracklet_stats, TRACKLET_STATS_DIM,
)

from tracklets.split_tracklets import split_tracklets
from utils.config import SplitterConfig

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT        = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\soccernet-player-tracking")
CACHE_ROOT       = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\cache")
CACHE_SPLIT_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\cache_split")
OUTPUT_ROOT      = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\train_data\temporal_bin_transformer")

# ---------------------------------------------------------------------------
# Settings — run this script once per split
# ---------------------------------------------------------------------------
SPLITS_TO_PROCESS = ["valid"]     # change to ["valid"] or ["test"] per run
SPLIT_OUTPUT_NAME = "valid"

NEGATIVE_RATIO          = None    # keep ALL negatives per-sequence; global rebalancing happens after combining with synthetics
MIN_TRACKLET_LEN        = 5
MAX_TEMPORAL_GAP        = None    # no limit
PURITY_THRESHOLD        = 0.80

# Synthetic positive generation
MIN_SPLIT_LEN           = 20     # only split tracklets with >= this many frames
N_SYNTHETIC_SPLITS      = 2      # random split points per eligible tracklet (capped globally to 2x real pos)
SYNTHETIC_MIN_FRAGMENT   = 8     # each fragment must have at least this many frames

SPLITTER_CFG = SplitterConfig()


# ---------------------------------------------------------------------------
# GT labeling with purity threshold
# ---------------------------------------------------------------------------

def get_gt_id(tracklet) -> Optional[int]:
    """Extract majority GT ID if tracklet is sufficiently pure."""
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


# ---------------------------------------------------------------------------
# Post-split cache (reuses XGBoost's cache if available)
# ---------------------------------------------------------------------------

def load_or_compute_split_tracklets(sequence: str) -> Optional[dict]:
    """Load post-split tracklets, running the splitter if not cached."""
    CACHE_SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
    split_cache_path = CACHE_SPLIT_ROOT / f"cache_split_{sequence}.pkl"

    if split_cache_path.exists():
        with open(split_cache_path, "rb") as f:
            return pickle.load(f)

    # Cache miss — run splitter
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
# Synthetic positive pair generation
# ---------------------------------------------------------------------------

def generate_synthetic_pairs(
    tracklets: dict,
    rng: np.random.RandomState,
) -> List[dict]:
    """
    Generate synthetic positive pairs by splitting long tracklets.

    For each tracklet with >= MIN_SPLIT_LEN frames, we pick N_SYNTHETIC_SPLITS
    random split points and create two fragments. These fragments are guaranteed
    to be from the same identity (they come from the same tracklet) and are
    non-overlapping (they're consecutive segments).

    This is the most natural and reliable form of data augmentation for this
    task: it creates realistic positive pairs that mimic what the merger would
    see if the splitter had split a tracklet at that point.

    We compute pairwise features between the fragments using the same function
    used for real pairs, but we need to create temporary Tracklet-like objects.
    """
    synthetic_pairs = []

    for tid, tracklet in tracklets.items():
        n = len(tracklet.frames)
        if n < MIN_SPLIT_LEN:
            continue

        gt_id = get_gt_id(tracklet)
        if gt_id is None:
            continue  # skip impure tracklets

        for _ in range(N_SYNTHETIC_SPLITS):
            # Pick a split point ensuring both fragments are long enough
            lo = SYNTHETIC_MIN_FRAGMENT
            hi = n - SYNTHETIC_MIN_FRAGMENT
            if lo >= hi:
                continue
            split_idx = rng.randint(lo, hi)

            # Create lightweight fragment objects with the same interface
            frag_a = _make_fragment(tracklet, 0, split_idx)
            frag_b = _make_fragment(tracklet, split_idx, n)

            if len(frag_a.frames) < MIN_TRACKLET_LEN or len(frag_b.frames) < MIN_TRACKLET_LEN:
                continue

            # Compute pairwise features
            pw = compute_pairwise_features(frag_a, frag_b)

            # Compute extended pairwise (base 12 + per-tracklet stats)
            feats_a, frames_a = extract_frame_features(frag_a)
            feats_b, frames_b = extract_frame_features(frag_b)
            stats_a = _tracklet_stats(feats_a, frames_a)
            stats_b = _tracklet_stats(feats_b, frames_b)
            extended_pw = np.concatenate([pw, stats_a, stats_b])

            synthetic_pairs.append({
                "sequence": f"synthetic_{tid}",
                "tid_a": f"{tid}_left",
                "tid_b": f"{tid}_right",
                "pairwise": extended_pw,
                "label": 1,  # guaranteed positive
                "is_synthetic": True,
                # Store pre-extracted features for the fragments
                "feats_a": feats_a,
                "frames_a": frames_a,
                "feats_b": feats_b,
                "frames_b": frames_b,
            })

    return synthetic_pairs


class _FragmentProxy:
    """Lightweight proxy that quacks like a Tracklet for feature extraction."""
    pass


def _make_fragment(tracklet, start_idx: int, end_idx: int) -> "_FragmentProxy":
    """Create a fragment proxy from a slice of a tracklet."""
    frag = _FragmentProxy()
    frag.frames = tracklet.frames[start_idx:end_idx]
    frag.bboxes = tracklet.bboxes[start_idx:end_idx]
    frag.scores = tracklet.scores[start_idx:end_idx]
    frag.embeddings = tracklet.embeddings[start_idx:end_idx] if tracklet.embeddings else []

    # Slice predicted attributes
    frag.pred_attributes = {}
    for k, v in tracklet.pred_attributes.items():
        if isinstance(v, list):
            frag.pred_attributes[k] = v[start_idx:end_idx]
        else:
            frag.pred_attributes[k] = v

    # Slice GT attributes
    frag.gt_attributes = {}
    gt_ids = tracklet.gt_attributes.get("track_ids", [])
    if gt_ids:
        frag.gt_attributes["track_ids"] = gt_ids[start_idx:end_idx]
    else:
        frag.gt_attributes["track_ids"] = []

    return frag


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)

    all_pairs = []
    all_synthetic = []
    total_pos = 0
    total_neg = 0

    for split in SPLITS_TO_PROCESS:
        split_root = DATA_ROOT / split
        if not split_root.exists():
            print(f"[SKIP] Split folder not found: {split_root}")
            continue

        sequences = sorted([d.name for d in split_root.iterdir() if d.is_dir()])
        print(f"\n{'='*60}")
        print(f"Processing split: {split}  ({len(sequences)} sequences)")
        print(f"{'='*60}")

        for seq in tqdm(sequences, desc=split):
            tracklets = load_or_compute_split_tracklets(seq)
            if tracklets is None:
                continue

            # Filter valid tracklets
            valid_tids = [
                tid for tid, t in tracklets.items()
                if len(t.frames) >= MIN_TRACKLET_LEN and t.embeddings
            ]

            if len(valid_tids) < 2:
                continue

            # ------ Real pairs (same as original, but with extended pairwise) ------
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
                    if gt_a is None or gt_b is None:
                        label = 0
                    else:
                        label = int(gt_a == gt_b)

                    # Extended pairwise features (12 base + 2*11 stats)
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

            # Negative downsampling
            if NEGATIVE_RATIO is not None and seq_pairs:
                pos = [p for p in seq_pairs if p["label"] == 1]
                neg = [p for p in seq_pairs if p["label"] == 0]
                n_keep = min(len(neg), int(len(pos) * NEGATIVE_RATIO))
                if n_keep < len(neg):
                    idx = rng.choice(len(neg), n_keep, replace=False)
                    neg = [neg[k] for k in idx]
                seq_pairs = pos + neg

            pos_count = sum(1 for p in seq_pairs if p["label"] == 1)
            neg_count = len(seq_pairs) - pos_count
            total_pos += pos_count
            total_neg += neg_count
            all_pairs.extend(seq_pairs)

            # ------ Synthetic positive pairs ------
            syn = generate_synthetic_pairs(tracklets, rng)
            all_synthetic.extend(syn)

            tqdm.write(
                f"  {seq}: {len(valid_tids)} tracklets -> "
                f"{len(seq_pairs)} real pairs (pos={pos_count}, neg={neg_count}), "
                f"{len(syn)} synthetic positives"
            )

    if not all_pairs and not all_synthetic:
        print("\n[ERROR] No pairs generated.")
        return

    # ---- Global rebalancing ----
    # Cap synthetic positives: at most 2x the number of real positives.
    # Too many synthetics would drown out real examples (synthetic pairs are
    # easier — same tracklet split in half — so they shouldn't dominate).
    max_synthetic = total_pos * 2
    if len(all_synthetic) > max_synthetic:
        rng.shuffle(all_synthetic)
        all_synthetic = all_synthetic[:max_synthetic]
        print(f"\n  Capped synthetic positives to {max_synthetic} (2x real positives)")

    all_combined = all_pairs + all_synthetic
    syn_count = len(all_synthetic)

    # Global negative downsampling: target ~3:1 neg:pos ratio.
    # With synthetics boosting positives, we may still have more negatives
    # than we need — or we may need all of them.
    GLOBAL_NEG_RATIO = 3.0
    final_pos = sum(1 for p in all_combined if p["label"] == 1)
    final_neg = sum(1 for p in all_combined if p["label"] == 0)
    target_neg = int(final_pos * GLOBAL_NEG_RATIO)

    if final_neg > target_neg:
        pos_list = [p for p in all_combined if p["label"] == 1]
        neg_list = [p for p in all_combined if p["label"] == 0]
        rng.shuffle(neg_list)
        neg_list = neg_list[:target_neg]
        all_combined = pos_list + neg_list
        final_neg = len(neg_list)
        print(f"  Downsampled negatives to {final_neg} ({GLOBAL_NEG_RATIO}:1 ratio)")

    # Shuffle
    rng.shuffle(all_combined)

    # Save
    out_path = OUTPUT_ROOT / f"pairs_{SPLIT_OUTPUT_NAME}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(all_combined, f)

    print(f"\n{'='*60}")
    print(f"Done.")
    print(f"  Real pairs      : {len(all_pairs):,}")
    print(f"    Positive (real): {total_pos:,}")
    print(f"    Negative (real): {total_neg:,}")
    print(f"  Synthetic pos   : {syn_count:,}")
    print(f"  Combined total  : {len(all_combined):,}")
    print(f"    Positive      : {final_pos:,}  ({100*final_pos/max(len(all_combined),1):.1f}%)")
    print(f"    Negative      : {final_neg:,}  ({100*final_neg/max(len(all_combined),1):.1f}%)")
    print(f"  Pickle          : {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
