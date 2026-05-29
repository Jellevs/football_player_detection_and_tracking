"""
Generate training data with SIMPLE synthetic positives.

Strategy: Same as XGBoost-matched real pairs, PLUS synthetic positive
pairs created by splitting long pure tracklets at random interior points.
This is the simplest form of data augmentation: fragments have near-identical
appearance and zero temporal gap, making them easy positives.

Uses post-split tracklets to match inference distribution.

Usage:
    Edit SPLITS_TO_PROCESS and SPLIT_OUTPUT_NAME, then:
    python generate_train_data_simple_synth.py

Output:
    train_data/siamese_cls_simple_synth/pairs_{split}.pkl
"""

import sys
import copy
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.tracklet_merger.transformers.utils import (
    extract_frame_features, compute_pairwise_features,
    REID_START, REID_END,
)
from tracklets.split_tracklets import split_tracklets
from utils.config import SplitterConfig

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT        = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\soccernet-player-tracking")
CACHE_ROOT       = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\cache")
CACHE_SPLIT_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\cache_split")
OUTPUT_ROOT      = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\transformers\train_data\siamese_cls_simple_synth")

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
SPLITS_TO_PROCESS = ["train"]     # change to ["valid"] or ["test"] per run
SPLIT_OUTPUT_NAME = "train"

NEGATIVE_RATIO      = 3
MIN_TRACKLET_LEN    = 0
PURITY_THRESHOLD    = 0.6
MAX_TEMPORAL_GAP    = None

# Synthetic positive generation
MIN_SPLIT_LEN          = 20    # only split tracklets with >= this many frames
N_SYNTHETIC_SPLITS     = 2     # split points per eligible tracklet
SYNTHETIC_MIN_FRAGMENT = 10    # each fragment must have at least this many frames
USE_REID_DRIFT_CUTS    = True  # bias cut points toward high ReID drift frames

SPLITTER_CFG = SplitterConfig()


# ---------------------------------------------------------------------------
# Helpers
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


def _compute_reid_drift(feats: np.ndarray) -> np.ndarray:
    """
    Compute per-frame ReID embedding cosine distance to the next frame.
    Returns array of length n-1.
    """
    reid = feats[:, REID_START:REID_END].astype(np.float32)
    norms = np.linalg.norm(reid, axis=1, keepdims=True) + 1e-8
    reid_normed = reid / norms
    cos_sim = np.sum(reid_normed[:-1] * reid_normed[1:], axis=1)
    return 1.0 - cos_sim  # distance: higher = more drift


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

    frag.pred_attributes = {}
    for k, v in tracklet.pred_attributes.items():
        if isinstance(v, list):
            frag.pred_attributes[k] = v[start_idx:end_idx]
        else:
            frag.pred_attributes[k] = v

    frag.gt_attributes = {}
    gt_ids = tracklet.gt_attributes.get("track_ids", [])
    if gt_ids:
        frag.gt_attributes["track_ids"] = gt_ids[start_idx:end_idx]
    else:
        frag.gt_attributes["track_ids"] = []

    return frag


def generate_synthetic_pairs(
    tracklets: dict,
    rng: np.random.RandomState,
) -> List[dict]:
    """
    Generate synthetic positive pairs by splitting long tracklets.
    When USE_REID_DRIFT_CUTS is True, cut points are biased toward frames
    with high ReID drift (where the splitter would naturally cut), producing
    more realistic synthetic positives.
    """
    synthetic_pairs = []

    for tid, tracklet in tracklets.items():
        n = len(tracklet.frames)
        if n < MIN_SPLIT_LEN:
            continue

        gt_id = get_gt_id(tracklet)
        if gt_id is None:
            continue

        # Pre compute ReID drift for drift biased cuts
        if USE_REID_DRIFT_CUTS and n > SYNTHETIC_MIN_FRAGMENT * 2 + 1:
            feats, _ = extract_frame_features(tracklet)
            drift = _compute_reid_drift(feats)
        else:
            drift = None

        for _ in range(N_SYNTHETIC_SPLITS):
            lo = SYNTHETIC_MIN_FRAGMENT
            hi = n - SYNTHETIC_MIN_FRAGMENT
            if lo >= hi:
                continue

            if USE_REID_DRIFT_CUTS and drift is not None:
                valid_drift = np.clip(drift[lo-1:hi-1], 0, None)
                if len(valid_drift) > 0 and valid_drift.sum() > 0:
                    probs = valid_drift / valid_drift.sum()
                    split_idx = rng.choice(np.arange(lo, hi), p=probs)
                else:
                    split_idx = rng.randint(lo, hi)
            else:
                split_idx = rng.randint(lo, hi)

            frag_a = _make_fragment(tracklet, 0, split_idx)
            frag_b = _make_fragment(tracklet, split_idx, n)

            if len(frag_a.frames) < MIN_TRACKLET_LEN or len(frag_b.frames) < MIN_TRACKLET_LEN:
                continue

            pw = compute_pairwise_features(frag_a, frag_b)

            # Store pre-extracted features for synthetic pairs
            # (since they don't exist in the cache)
            feats_a, frames_a = extract_frame_features(frag_a)
            feats_b, frames_b = extract_frame_features(frag_b)

            synthetic_pairs.append({
                "sequence": f"synthetic_{tid}",
                "tid_a": f"{tid}_left",
                "tid_b": f"{tid}_right",
                "pairwise": pw,
                "label": 1,
                "is_synthetic": True,
                "feats_a": feats_a,
                "frames_a": frames_a,
                "feats_b": feats_b,
                "frames_b": frames_b,
            })

    return synthetic_pairs


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

            valid_tids = [
                tid for tid, t in tracklets.items()
                if len(t.frames) >= MIN_TRACKLET_LEN and t.embeddings
            ]

            if len(valid_tids) < 2:
                continue

            # ------ Real pairs ------
            seq_pairs = []
            for i in range(len(valid_tids)):
                for j in range(i + 1, len(valid_tids)):
                    t_a = tracklets[valid_tids[i]]
                    t_b = tracklets[valid_tids[j]]

                    if set(t_a.frames) & set(t_b.frames):
                        continue

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

                    gt_a = get_gt_id(t_a)
                    gt_b = get_gt_id(t_b)
                    label = int(gt_a is not None and gt_b is not None and gt_a == gt_b)

                    pw = compute_pairwise_features(t_a, t_b)

                    seq_pairs.append({
                        "sequence": seq,
                        "tid_a":    valid_tids[i],
                        "tid_b":    valid_tids[j],
                        "pairwise": pw,
                        "label":    label,
                    })

            pos_count = sum(1 for p in seq_pairs if p["label"] == 1)
            neg_count = len(seq_pairs) - pos_count
            total_pos += pos_count
            total_neg += neg_count
            all_pairs.extend(seq_pairs)

            # ------ Synthetic positives ------
            syn = generate_synthetic_pairs(tracklets, rng)
            all_synthetic.extend(syn)

            tqdm.write(
                f"  {seq}: {len(valid_tids)} tracklets -> "
                f"{len(seq_pairs)} real (pos={pos_count}, neg={neg_count}), "
                f"{len(syn)} synthetic"
            )

    if not all_pairs and not all_synthetic:
        print("\n[ERROR] No pairs generated.")
        return

    # Cap synthetic positives at 2x real positives
    max_synthetic = total_pos * 2
    if len(all_synthetic) > max_synthetic:
        rng.shuffle(all_synthetic)
        all_synthetic = all_synthetic[:max_synthetic]
        print(f"\n  Capped synthetic positives to {max_synthetic}")

    all_combined = all_pairs + all_synthetic
    syn_count = len(all_synthetic)

    # Negative downsampling: target neg_ratio:1
    final_pos = sum(1 for p in all_combined if p["label"] == 1)
    neg_list = [p for p in all_combined if p["label"] == 0]
    pos_list = [p for p in all_combined if p["label"] == 1]

    target_neg = int(final_pos * NEGATIVE_RATIO)
    if len(neg_list) > target_neg:
        rng.shuffle(neg_list)
        neg_list = neg_list[:target_neg]

    all_combined = pos_list + neg_list

    # Shuffle
    rng.shuffle(all_combined)

    # Save
    out_path = OUTPUT_ROOT / f"pairs_{SPLIT_OUTPUT_NAME}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(all_combined, f)

    final_pos = sum(1 for p in all_combined if p["label"] == 1)
    final_neg = len(all_combined) - final_pos

    print(f"\n{'='*60}")
    print(f"Done.")
    print(f"  Real positives    : {total_pos:,}")
    print(f"  Real negatives    : {total_neg:,}")
    print(f"  Synthetic pos     : {syn_count:,}")
    print(f"  Final total       : {len(all_combined):,}")
    print(f"    Positive        : {final_pos:,}  ({100*final_pos/max(len(all_combined),1):.1f}%)")
    print(f"    Negative        : {final_neg:,}  ({100*final_neg/max(len(all_combined),1):.1f}%)")
    print(f"  Pickle            : {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
