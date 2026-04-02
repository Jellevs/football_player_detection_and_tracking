"""
Generate training data for the per-frame split-point transformer.

Three types of samples:
  1. Synthetic positives — two tracklets (different GT IDs) concatenated.
     Tier 1: cross-team, Tier 2: same-team (harder, more realistic).
  2. Natural positives — tracklets with genuine GT ID switches.
  3. Negatives — clean single-identity tracklets.

Target ratio ~3:1 positive:negative (recall-biased).

Usage:
    python transformer_splitter/generate_split_data.py

    Edit SPLITS_TO_PROCESS / SPLIT_OUTPUT_NAME for each split.

Output:
    train_data/splitter_data/samples_{split}.pkl
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Dict
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT   = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\soccernet-player-tracking")
CACHE_ROOT  = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\cache")
OUTPUT_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\train_data\splitter_data")

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
SPLITS_TO_PROCESS = ["valid"]
SPLIT_OUTPUT_NAME = "valid"

MIN_TRACKLET_LEN  = 20    # minimum frames for a tracklet to be usable
MIN_CROP_LEN      = 20    # minimum crop per half in synthetic samples
MAX_GAP            = 30    # max temporal gap at synthetic junction
SAMPLES_PER_PAIR   = 2    # random crop variations per synthetic pair
POS_NEG_RATIO      = 3.0  # target positive:negative ratio
SEED               = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_gt_id(tracklet) -> Optional[int]:
    """Majority ground-truth track ID."""
    gt_ids = tracklet.gt_attributes.get("track_ids", [])
    valid = [g for g in gt_ids
             if g is not None and not (isinstance(g, float) and np.isnan(g))]
    if not valid:
        return None
    return max(set(valid), key=valid.count)


def get_dominant_team(tracklet) -> Optional[int]:
    """Majority team label (ignoring NaN)."""
    teams = tracklet.pred_attributes.get("teams", [])
    valid = [int(t) for t in teams
             if t is not None and not (isinstance(t, float) and np.isnan(t))]
    if not valid:
        return None
    return max(set(valid), key=valid.count)


def find_natural_switches(tracklet) -> List[int]:
    """Find frame indices where GT track ID changes within a tracklet."""
    gt_ids = tracklet.gt_attributes.get("track_ids", [])
    switches = []
    prev = None
    for i, gid in enumerate(gt_ids):
        if gid is None or (isinstance(gid, float) and np.isnan(gid)):
            continue
        if prev is not None and gid != prev:
            switches.append(i)
        prev = gid
    return switches


def make_synthetic_sample(
    seq: str,
    tid_a: int, len_a: int,
    tid_b: int, len_b: int,
    rng: np.random.RandomState,
) -> dict:
    """Create one synthetic concatenation sample with random crops."""
    # Random crop lengths (at least MIN_CROP_LEN each)
    max_a = max(len_a, MIN_CROP_LEN + 1)
    max_b = max(len_b, MIN_CROP_LEN + 1)
    crop_len_a = rng.randint(MIN_CROP_LEN, max_a)
    crop_len_b = rng.randint(MIN_CROP_LEN, max_b)

    # Random start positions
    start_a = rng.randint(0, len_a - crop_len_a + 1)
    start_b = rng.randint(0, len_b - crop_len_b + 1)

    # Random temporal gap
    gap = rng.randint(0, MAX_GAP + 1)

    # The split index is the first frame of tracklet B within the concatenated sequence
    split_idx = crop_len_a

    return {
        "source_type": "synthetic",
        "sequence": seq,
        "tid_a": tid_a,
        "tid_b": tid_b,
        "crop_a": (start_a, start_a + crop_len_a),
        "crop_b": (start_b, start_b + crop_len_b),
        "gap_frames": gap,
        "split_indices": [split_idx],
    }


def make_natural_sample(seq: str, tid: int, switch_indices: List[int]) -> dict:
    """Create a sample from a tracklet with natural GT ID switches."""
    return {
        "source_type": "natural",
        "sequence": seq,
        "tid_a": tid,
        "tid_b": None,
        "crop_a": None,  # use full tracklet
        "crop_b": None,
        "gap_frames": 0,
        "split_indices": switch_indices,
    }


def make_negative_sample(seq: str, tid: int, length: int) -> dict:
    """Create a negative sample (clean single-identity tracklet)."""
    return {
        "source_type": "negative",
        "sequence": seq,
        "tid_a": tid,
        "tid_b": None,
        "crop_a": None,  # use full tracklet
        "crop_b": None,
        "gap_frames": 0,
        "split_indices": [],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED)

    all_samples: List[dict] = []
    counts = defaultdict(int)

    for split in SPLITS_TO_PROCESS:
        split_root = DATA_ROOT / split
        if not split_root.exists():
            print(f"[SKIP] Split folder not found: {split_root}")
            continue

        sequences = sorted([d.name for d in split_root.iterdir() if d.is_dir()])
        print(f"\n{'='*60}")
        print(f"Split: {split}  ({len(sequences)} sequences)")
        print(f"{'='*60}")

        for seq in tqdm(sequences, desc=split):
            cache_path = CACHE_ROOT / f"cache_attributes_{seq}.pkl"
            if not cache_path.exists():
                tqdm.write(f"  [MISS] {cache_path.name}")
                continue

            with open(cache_path, "rb") as f:
                tracklets = pickle.load(f)

            # Filter valid tracklets (minimum length + have embeddings)
            valid = {}
            for tid, t in tracklets.items():
                if len(t.frames) >= MIN_TRACKLET_LEN and t.embeddings:
                    gt_id = get_gt_id(t)
                    team = get_dominant_team(t)
                    valid[tid] = {"tracklet": t, "gt_id": gt_id, "team": team,
                                  "length": len(t.frames)}

            if len(valid) < 2:
                continue

            # --- Natural positives ---
            for tid, info in valid.items():
                switches = find_natural_switches(info["tracklet"])
                if switches:
                    all_samples.append(make_natural_sample(seq, tid, switches))
                    counts["natural"] += 1

            # --- Negatives: clean single-identity tracklets ---
            negatives_this_seq = []
            for tid, info in valid.items():
                if info["gt_id"] is None:
                    continue
                switches = find_natural_switches(info["tracklet"])
                if not switches:
                    negatives_this_seq.append(
                        make_negative_sample(seq, tid, info["length"])
                    )

            # --- Synthetic positives ---
            # Group tracklets by GT ID and team
            by_gt: Dict[int, list] = defaultdict(list)
            for tid, info in valid.items():
                if info["gt_id"] is not None:
                    by_gt[info["gt_id"]].append(tid)

            gt_ids = list(by_gt.keys())
            synthetics_this_seq = []

            for i in range(len(gt_ids)):
                for j in range(i + 1, len(gt_ids)):
                    gt_a, gt_b = gt_ids[i], gt_ids[j]
                    # Pick one representative tracklet per GT ID
                    tids_a = by_gt[gt_a]
                    tids_b = by_gt[gt_b]

                    for ta in tids_a:
                        for tb in tids_b:
                            info_a = valid[ta]
                            info_b = valid[tb]

                            # Determine tier
                            same_team = (info_a["team"] is not None
                                         and info_b["team"] is not None
                                         and info_a["team"] == info_b["team"])
                            tier = "same_team" if same_team else "cross_team"

                            for _ in range(SAMPLES_PER_PAIR):
                                sample = make_synthetic_sample(
                                    seq, ta, info_a["length"],
                                    tb, info_b["length"], rng,
                                )
                                sample["tier"] = tier
                                synthetics_this_seq.append(sample)

            # Limit synthetics to maintain balance
            # Target: synthetics + natural ≈ POS_NEG_RATIO * negatives
            n_neg = len(negatives_this_seq)
            n_natural = sum(1 for s in all_samples
                           if s.get("sequence") == seq and s["source_type"] == "natural")
            target_pos = int(n_neg * POS_NEG_RATIO) - n_natural
            target_pos = max(target_pos, 0)

            if len(synthetics_this_seq) > target_pos and target_pos > 0:
                # Prefer same-team (harder) samples: keep all of them, downsample cross-team
                same_team_samples = [s for s in synthetics_this_seq if s.get("tier") == "same_team"]
                cross_team_samples = [s for s in synthetics_this_seq if s.get("tier") == "cross_team"]

                if len(same_team_samples) >= target_pos:
                    # Enough hard samples — use only same-team
                    idx = rng.choice(len(same_team_samples), target_pos, replace=False)
                    synthetics_this_seq = [same_team_samples[k] for k in idx]
                else:
                    # Keep all same-team, fill rest with cross-team
                    remaining = target_pos - len(same_team_samples)
                    if remaining < len(cross_team_samples):
                        idx = rng.choice(len(cross_team_samples), remaining, replace=False)
                        cross_team_samples = [cross_team_samples[k] for k in idx]
                    synthetics_this_seq = same_team_samples + cross_team_samples
            elif target_pos == 0 and synthetics_this_seq:
                # If there are no negatives, still keep some synthetics
                n_keep = min(len(synthetics_this_seq), 50)
                idx = rng.choice(len(synthetics_this_seq), n_keep, replace=False)
                synthetics_this_seq = [synthetics_this_seq[k] for k in idx]

            for s in synthetics_this_seq:
                counts[f"synthetic_{s.get('tier', 'unknown')}"] += 1
            counts["negative"] += len(negatives_this_seq)

            all_samples.extend(synthetics_this_seq)
            all_samples.extend(negatives_this_seq)

            tqdm.write(
                f"  {seq}: {len(valid)} tracklets → "
                f"synth={len(synthetics_this_seq)}, neg={len(negatives_this_seq)}"
            )

    if not all_samples:
        print("\n[ERROR] No samples generated. Check cache paths.")
        return

    # Fix crop_a/crop_b for natural/negative samples (use full tracklet)
    for s in all_samples:
        if s["crop_a"] is None:
            key = (s["sequence"], s["tid_a"])
            # We don't have the tracklet length here, so use a sentinel
            # The dataset will handle None crop_a as "use everything"
            s["crop_a"] = None
            s["crop_b"] = None

    # Shuffle
    rng.shuffle(all_samples)

    # For natural/negative: set crop to full range
    # We need tracklet lengths — reload briefly from one cache to verify,
    # but actually the dataset handles crop_a=None gracefully.
    # Instead, encode "full tracklet" by setting crop to (0, very_large_number)
    for s in all_samples:
        if s["crop_a"] is None:
            s["crop_a"] = (0, 99999)  # dataset will clamp to actual length

    # Save
    out_path = OUTPUT_ROOT / f"samples_{SPLIT_OUTPUT_NAME}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(all_samples, f)

    print(f"\n{'='*60}")
    print(f"Done.")
    print(f"  Total samples    : {len(all_samples):,}")
    for k, v in sorted(counts.items()):
        print(f"  {k:20s}: {v:,}")
    n_pos = sum(1 for s in all_samples if s["split_indices"])
    n_neg = sum(1 for s in all_samples if not s["split_indices"])
    print(f"  Positive (has split): {n_pos:,}")
    print(f"  Negative (clean)   : {n_neg:,}")
    if n_neg > 0:
        print(f"  Pos:Neg ratio      : {n_pos/n_neg:.2f}:1")
    print(f"  Output             : {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
