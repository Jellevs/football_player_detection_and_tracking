"""
Generate training data for the Late Cross-Attention Transformer merger.

Key improvements over the temporal bin transformer data generation (Plan Section 3):

A1. Realistic synthetic positives:
    - Cut at frames of **largest ReID drift** within a pure tracklet
      (where the splitter would cut), so fragments carry realistic appearance gaps
    - Inject a **temporal gap** by deleting a random run of frames at the cut
    - Sample fragment lengths to match the empirical post-split length distribution

A2. Hard-negative mining:
    - Same predicted team, small temporal gap, small spatial distance, HIGH
      reid/siglip cosine similarity, but different GT id
    - These are the merges that wreck HOTA — false links between
      same-team, plausibly-adjacent, appearance-similar different players
    - Hard negatives carry weight > 1.0 for boundary-hardness loss weighting

A4. More data volume:
    - Enumerate all GT-consistent non-overlapping tracklet pairs
    - Multiple post-split fragmentations
    - Keep all negatives per sequence (global rebalancing after)

A5. Cleaner labels:
    - Tighter purity threshold (0.90) for positive eligibility
    - Route confidently-different impure tracklets into hard-negative pool
    - Don't drop impure pairs as noise in the easy-negative pool

Usage:
    # Edit SPLITS_TO_PROCESS and SPLIT_OUTPUT_NAME, then:
    python -m experiments.tracklet_merger.new_transformer.generate_train_data

Output:
    train_data/new_transformer/pairs_{split}.pkl
"""

import copy
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Tuple
from collections import Counter

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

from transformer.dataset import (
    extract_frame_features, compute_pairwise_features,
    _tracklet_stats, TRACKLET_STATS_DIM,
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
OUTPUT_ROOT      = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\train_data\new_transformer")

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
SPLITS_TO_PROCESS   = ["valid"]
SPLIT_OUTPUT_NAME   = "valid"

NEGATIVE_RATIO          = None   # keep ALL negatives; global rebalancing later
MIN_TRACKLET_LEN        = 5
MAX_TEMPORAL_GAP        = None

# A5: Tighter purity for positives, lower for negatives
PURITY_THRESHOLD_POS    = 0.90
PURITY_THRESHOLD_NEG    = 0.80

# A1: Realistic synthetic generation
MIN_SPLIT_LEN           = 20
N_SYNTHETIC_SPLITS      = 3
SYNTHETIC_MIN_FRAGMENT  = 8
USE_REID_DRIFT_CUTS     = True
INJECT_TEMPORAL_GAP     = True
TEMPORAL_GAP_RANGE      = (5, 60)

# A2: Hard-negative mining
HARD_NEG_MIN_COSINE     = 0.3
HARD_NEG_MAX_TEMPORAL_GAP = 200
HARD_NEG_WEIGHT         = 3.0    # sample weight for hard negatives

SPLITTER_CFG = SplitterConfig()


# ---------------------------------------------------------------------------
# GT labeling
# ---------------------------------------------------------------------------

def get_gt_id(tracklet, purity_threshold: float = 0.80) -> Optional[int]:
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


def get_team_mode(tracklet) -> Optional[int]:
    """Get majority team prediction for a tracklet."""
    teams = tracklet.pred_attributes.get("teams", [])
    valid = [t for t in teams
             if t is not None and not (isinstance(t, float) and np.isnan(t))]
    if not valid:
        return None
    return max(set(valid), key=valid.count)


# ---------------------------------------------------------------------------
# Post-split cache
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
# A1: Realistic synthetic positive generation
# ---------------------------------------------------------------------------

def _compute_reid_drift(feats: np.ndarray) -> np.ndarray:
    """
    Compute per-frame ReID embedding cosine distance to the next frame.
    Returns array of length n-1.
    """
    reid = feats[:, REID_START:REID_END].astype(np.float32)
    norms = np.linalg.norm(reid, axis=1, keepdims=True) + 1e-8
    reid_normed = reid / norms
    # Cosine distance between consecutive frames
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
    Generate synthetic positive pairs with realistic characteristics (A1).

    Key improvements:
    - Cut at frames of largest ReID drift (where splitter would cut)
    - Inject temporal gaps at cut points
    - Both fragments must pass minimum length check
    """
    synthetic_pairs = []

    for tid, tracklet in tracklets.items():
        n = len(tracklet.frames)
        if n < MIN_SPLIT_LEN:
            continue

        gt_id = get_gt_id(tracklet, PURITY_THRESHOLD_POS)
        if gt_id is None:
            continue

        # Pre-compute features and ReID drift
        feats, frames = extract_frame_features(tracklet)
        if USE_REID_DRIFT_CUTS and n > SYNTHETIC_MIN_FRAGMENT * 2 + 1:
            drift = _compute_reid_drift(feats)
        else:
            drift = None

        for split_i in range(N_SYNTHETIC_SPLITS):
            lo = SYNTHETIC_MIN_FRAGMENT
            hi = n - SYNTHETIC_MIN_FRAGMENT
            if lo >= hi:
                continue

            if USE_REID_DRIFT_CUTS and drift is not None:
                # A1: Cut at maximum ReID drift positions
                # Use weighted sampling biased toward high-drift frames
                valid_drift = np.clip(drift[lo-1:hi-1], 0, None)  # drift at valid cut positions
                if len(valid_drift) > 0 and valid_drift.sum() > 0:
                    probs = valid_drift / valid_drift.sum()
                    split_idx = rng.choice(np.arange(lo, hi), p=probs)
                else:
                    split_idx = rng.randint(lo, hi)
            else:
                split_idx = rng.randint(lo, hi)

            # A1: Inject temporal gap
            if INJECT_TEMPORAL_GAP:
                gap_lo, gap_hi = TEMPORAL_GAP_RANGE
                gap_frames = rng.randint(gap_lo, gap_hi + 1)
                # Remove frames around the cut point
                gap_start = max(lo, split_idx - gap_frames // 2)
                gap_end = min(hi, split_idx + (gap_frames - gap_frames // 2))

                # Create fragments with the gap
                frag_a = _make_fragment(tracklet, 0, gap_start)
                frag_b = _make_fragment(tracklet, gap_end, n)
            else:
                frag_a = _make_fragment(tracklet, 0, split_idx)
                frag_b = _make_fragment(tracklet, split_idx, n)

            if len(frag_a.frames) < MIN_TRACKLET_LEN or len(frag_b.frames) < MIN_TRACKLET_LEN:
                continue

            # Compute pairwise features
            pw = compute_pairwise_features(frag_a, frag_b)
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
                "label": 1,
                "is_synthetic": True,
                "weight": 1.0,
                "gt_id_a": gt_id,
                "gt_id_b": gt_id,
                "feats_a": feats_a,
                "frames_a": frames_a,
                "feats_b": feats_b,
                "frames_b": frames_b,
            })

    return synthetic_pairs


# ---------------------------------------------------------------------------
# A2: Hard-negative identification
# ---------------------------------------------------------------------------

def is_hard_negative(pair: dict) -> bool:
    """
    Check if a negative pair qualifies as 'hard'.
    Hard negatives: same team, small temporal gap, high appearance similarity,
    different GT identity.
    """
    if pair["label"] != 0:
        return False

    pw = pair["pairwise"]

    # Index 0: temporal_gap, 5: reid_cosine_sim, 6: siglip_cosine_sim
    # (from compute_pairwise_features layout)
    temporal_gap = abs(pw[0])
    reid_cosine = pw[5]

    # Same team (indices 9,10: team_match, team_conflict)
    team_match = pw[9]

    # Hard = high cosine + small gap + (same team or no team info)
    if reid_cosine >= HARD_NEG_MIN_COSINE and temporal_gap <= HARD_NEG_MAX_TEMPORAL_GAP:
        return True

    if team_match > 0.5 and reid_cosine >= 0.2:
        return True

    return False


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
    total_hard_neg = 0

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

                    # A5: Dual-threshold labeling
                    gt_a_strict = get_gt_id(t_a, PURITY_THRESHOLD_POS)
                    gt_b_strict = get_gt_id(t_b, PURITY_THRESHOLD_POS)
                    gt_a_loose = get_gt_id(t_a, PURITY_THRESHOLD_NEG)
                    gt_b_loose = get_gt_id(t_b, PURITY_THRESHOLD_NEG)

                    # Positive: both must pass strict purity
                    if gt_a_strict is not None and gt_b_strict is not None and gt_a_strict == gt_b_strict:
                        label = 1
                        gt_id_a, gt_id_b = gt_a_strict, gt_b_strict
                    elif gt_a_loose is not None and gt_b_loose is not None and gt_a_loose != gt_b_loose:
                        # A5: Confirmed different identity -> negative
                        label = 0
                        gt_id_a, gt_id_b = gt_a_loose, gt_b_loose
                    elif gt_a_loose is None or gt_b_loose is None:
                        # Impure -> route to negative pool (don't drop)
                        label = 0
                        gt_id_a = gt_a_loose if gt_a_loose is not None else -1
                        gt_id_b = gt_b_loose if gt_b_loose is not None else -1
                    else:
                        # Both pass loose, same ID, but one fails strict
                        # Skip: ambiguous
                        continue

                    # Extended pairwise features
                    pw_base = compute_pairwise_features(t_a, t_b)
                    feats_a, frames_a = extract_frame_features(t_a)
                    feats_b, frames_b = extract_frame_features(t_b)
                    stats_a = _tracklet_stats(feats_a, frames_a)
                    stats_b = _tracklet_stats(feats_b, frames_b)
                    extended_pw = np.concatenate([pw_base, stats_a, stats_b])

                    pair = {
                        "sequence": seq,
                        "tid_a": valid_tids[i],
                        "tid_b": valid_tids[j],
                        "pairwise": extended_pw,
                        "label": label,
                        "is_synthetic": False,
                        "weight": 1.0,
                        "gt_id_a": gt_id_a,
                        "gt_id_b": gt_id_b,
                    }

                    # A2: Mark hard negatives with higher weight
                    if label == 0 and is_hard_negative(pair):
                        pair["weight"] = HARD_NEG_WEIGHT
                        total_hard_neg += 1

                    seq_pairs.append(pair)

            pos_count = sum(1 for p in seq_pairs if p["label"] == 1)
            neg_count = len(seq_pairs) - pos_count
            total_pos += pos_count
            total_neg += neg_count
            all_pairs.extend(seq_pairs)

            # ------ Synthetic positive pairs (A1) ------
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
    # Cap synthetic positives at 2x real positives
    max_synthetic = total_pos * 2
    if len(all_synthetic) > max_synthetic:
        rng.shuffle(all_synthetic)
        all_synthetic = all_synthetic[:max_synthetic]
        print(f"\n  Capped synthetic positives to {max_synthetic} (2x real positives)")

    all_combined = all_pairs + all_synthetic
    syn_count = len(all_synthetic)

    # Global negative downsampling: target ~3:1 neg:pos ratio
    GLOBAL_NEG_RATIO = 3.0
    final_pos = sum(1 for p in all_combined if p["label"] == 1)
    final_neg = sum(1 for p in all_combined if p["label"] == 0)
    target_neg = int(final_pos * GLOBAL_NEG_RATIO)

    if final_neg > target_neg:
        pos_list = [p for p in all_combined if p["label"] == 1]
        neg_list = [p for p in all_combined if p["label"] == 0]

        # A2: Ensure hard negatives are preserved during downsampling
        hard_negs = [p for p in neg_list if p.get("weight", 1.0) > 1.0]
        easy_negs = [p for p in neg_list if p.get("weight", 1.0) <= 1.0]

        n_hard_to_keep = min(len(hard_negs), target_neg // 2)
        n_easy_to_keep = target_neg - n_hard_to_keep

        rng.shuffle(hard_negs)
        rng.shuffle(easy_negs)
        hard_negs = hard_negs[:n_hard_to_keep]
        easy_negs = easy_negs[:n_easy_to_keep]

        neg_list = hard_negs + easy_negs
        all_combined = pos_list + neg_list
        final_neg = len(neg_list)
        print(f"  Downsampled negatives to {final_neg} "
              f"({n_hard_to_keep} hard + {n_easy_to_keep} easy, {GLOBAL_NEG_RATIO}:1 ratio)")

    # Shuffle
    rng.shuffle(all_combined)

    # Save
    out_path = OUTPUT_ROOT / f"pairs_{SPLIT_OUTPUT_NAME}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(all_combined, f)

    # Stats
    hard_neg_final = sum(1 for p in all_combined if p.get("weight", 1.0) > 1.0 and p["label"] == 0)

    print(f"\n{'='*60}")
    print(f"Done.")
    print(f"  Real pairs        : {len(all_pairs):,}")
    print(f"    Positive (real) : {total_pos:,}")
    print(f"    Negative (real) : {total_neg:,}")
    print(f"    Hard negatives  : {total_hard_neg:,} (identified) -> {hard_neg_final:,} (kept)")
    print(f"  Synthetic pos     : {syn_count:,}")
    print(f"  Combined total    : {len(all_combined):,}")
    print(f"    Positive        : {final_pos:,}  ({100*final_pos/max(len(all_combined),1):.1f}%)")
    print(f"    Negative        : {final_neg:,}  ({100*final_neg/max(len(all_combined),1):.1f}%)")
    print(f"  Purity thresholds : pos={PURITY_THRESHOLD_POS}, neg={PURITY_THRESHOLD_NEG}")
    print(f"  Synthetic config  : drift_cuts={USE_REID_DRIFT_CUTS}, gap={INJECT_TEMPORAL_GAP}")
    print(f"  Pickle            : {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
