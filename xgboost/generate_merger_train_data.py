"""
generate_merger_training_data.py

Generates pairwise training data for the XGBoost tracklet merger.
Reads cached attribute tracklets (the output of predict_attributes), applies
the splitter to match the exact distribution the merger sees at inference time,
then aggregates each post-split tracklet into a fixed-length feature vector and
constructs all valid pairs with a ground-truth merge label derived from gt_track_id.

Split tracklets are cached to CACHE_SPLIT_ROOT so the splitter only runs once.
Delete the split cache files to force a re-split (e.g. after changing SIGNALS in
split_tracklets.py).

No PCA / dimensionality reduction is applied.  The full raw feature set is kept
so that the XGBoost model can decide for itself which signals matter.

Output (per split, one combined CSV):
    combined_train_data.csv
    combined_valid_data.csv
    combined_test_data.csv

Each row represents one candidate pair (A, B) with columns:
    A_*           — aggregated features for tracklet A
    B_*           — aggregated features for tracklet B
    pairwise_*    — features computed from the pair jointly
    label         — 1 if A and B should be merged (same gt identity), else 0
    sequence      — sequence name (used for group-aware train/val/test split)
"""
import sys
sys.path.insert(0, r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch")

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from tracklets.split_tracklets import split_tracklets
from utils.config import SplitterConfig


# ---------------------------------------------------------------------------
# Paths — adjust to your environment
# ---------------------------------------------------------------------------

# Which SoccerNetGS split folders to include.  Each may contain many sequences.
SPLITS_TO_PROCESS = ["train"]   # do NOT include "test" until final evaluation

SPLIT_OUTPUT_NAME = "train"             # the label used in the output filename
                                         # run again with SPLITS_TO_PROCESS=["valid"] and
                                         # SPLIT_OUTPUT_NAME="valid", etc.

MAX_TEMPORAL_GAP        = None   # frames; pairs further apart are skipped
NEGATIVE_RATIO          = 5   # neg/pos sampling ratio (None = keep all, 2.0 = old default)
MIN_TRACKLET_LEN        = 5      # frames; shorter tracklets are discarded
REID_DIM                = 512    # OSNet x1_0 embedding dimension
SIGLIP_DIM              = 768    # SigLIP embedding dimension
TEAM_CONFIDENCE_THRESHOLD = 0.6  # min KMeans confidence to count a team prediction as reliable

# ---------------------------------------------------------------------------
# Purity threshold for GT labeling
#
# Our splitter does not work perfectly, so a tracklet can contain frames from
# more than one real player. If we naively take the majority GT ID and label
# the pair based on that, we risk training on noisy examples:
#   - A tracklet that is 55% player A and 45% player B gets called "player A",
#     but its aggregated features (mean ReID, jersey mode, etc.) are a confusing
#     mix of two people.
#   - Two such impure tracklets may be labeled "should merge" even though
#     merging them would make an identity-mixed tracklet even worse.
#
# Fix: only trust the majority GT ID when it is dominant enough. If fewer than
# TRACKLET_PURITY_THRESHOLD of a tracklet's frames agree on the majority ID,
# aggregate_tracklet() sets gt_id to None, and any pair involving that tracklet
# is skipped during pair generation. This reduces dataset size slightly but
# removes the most misleading training examples.
# ---------------------------------------------------------------------------
TRACKLET_PURITY_THRESHOLD = 0.80  # at least 80% of frames must share the majority GT ID

DATA_ROOT        = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\soccernet-player-tracking")
OUTPUT_ROOT      = Path(fr"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\train_data\xgboost_{5}_neg_ratio_no_main_subj_filt_SPLITTED")
CACHE_ROOT       = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\cache")
CACHE_SPLIT_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\cache_split")

# Splitter config — must match the settings used during inference (settings.py SPLITTER dict)
SPLITTER_CFG = SplitterConfig()


# ---------------------------------------------------------------------------
# 1.  Tracklet aggregation
# ---------------------------------------------------------------------------

def aggregate_tracklet(tracklet, track_id: int) -> Optional[dict]:
    """
    Collapse a variable-length tracklet into a fixed-size feature dict.

    Raw embedding arrays are NOT stored as individual columns — XGBoost cannot
    exploit high-dimensional embedding vectors meaningfully. Instead, pairwise
    cosine similarities are computed at pair-generation time and stored as
    pairwise_reid_cosine_sim / pairwise_siglip_cosine_sim. The raw mean vectors
    are kept in the aggregated dict under 'reid_mean_vec' and 'siglip_mean_vec'
    for use during pair generation, but are never written to CSV columns.

    Jersey number    → mode, entropy_mean, confidence_mean, coverage
    Team             → mode (int), consistency, coverage
    Temporal/spatial → start_frame, end_frame, duration,
                       start_x, start_y, end_x, end_y, mean_bbox_height
    """
    frames     = tracklet.frames
    bboxes     = tracklet.bboxes
    embeddings = tracklet.embeddings

    if len(frames) < MIN_TRACKLET_LEN:
        return None
    if not embeddings:
        return None

    feats = {}

    # ---- ReID (kept as vector for pairwise cosine sim, not written to CSV) ----
    emb_arr  = np.stack(embeddings).astype(np.float32)
    norms    = np.linalg.norm(emb_arr, axis=1, keepdims=True) + 1e-6
    emb_norm = emb_arr / norms
    reid_mean_vec = emb_norm.mean(axis=0)   # stored on agg dict, not in feats

    # ---- SigLIP (same — vector for cosine sim only) ----
    siglip_all   = tracklet.pred_attributes.get("siglip_embeddings", [])
    valid_siglip = [s for s in siglip_all if np.any(np.array(s) != 0)]
    if valid_siglip:
        siglip_arr      = np.stack(valid_siglip).astype(np.float32)
        siglip_mean_vec = siglip_arr.mean(axis=0)
    else:
        siglip_mean_vec = np.zeros(SIGLIP_DIM, dtype=np.float32)

    # ---- Jersey ----
    jerseys    = tracklet.pred_attributes.get("jerseys", [])
    entropies  = tracklet.pred_attributes.get("jersey_entropies", [1.0] * len(frames))
    confs      = tracklet.pred_attributes.get("jersey_confs_mean", [0.0] * len(frames))

    valid_j = [(j, entropies[i] if i < len(entropies) else 1.0,
                   confs[i]     if i < len(confs)     else 0.0)
               for i, j in enumerate(jerseys) if not np.isnan(j)]

    if valid_j:
        js, es, cs   = zip(*valid_j)
        mode_jersey  = max(set(js), key=js.count)
        mode_es      = [e for j, e in zip(js, es) if j == mode_jersey]
        feats["jersey_mode"]        = float(mode_jersey)
        feats["jersey_entropy_mean"]= float(np.mean(mode_es))
        feats["jersey_conf_mean"]   = float(np.mean(cs))
        feats["jersey_coverage"]    = len(valid_j) / max(len(frames), 1)
    else:
        feats["jersey_mode"]        = np.nan
        feats["jersey_entropy_mean"]= 1.0
        feats["jersey_conf_mean"]   = 0.0
        feats["jersey_coverage"]    = 0.0

    # ---- Team ----
    # Compute mode/consistency on confident predictions only (clean signal),
    # but also expose mean confidence so the model can learn to weight it.
    teams_raw  = tracklet.pred_attributes.get("teams", [])
    team_confs = tracklet.pred_attributes.get("team_confs", [])

    valid_team_confs = [
        team_confs[i] for i in range(len(teams_raw))
        if teams_raw[i] is not None
        and not (isinstance(teams_raw[i], float) and np.isnan(teams_raw[i]))
        and i < len(team_confs)
        and team_confs[i] is not None
        and not (isinstance(team_confs[i], float) and np.isnan(team_confs[i]))
    ]
    teams_confident = [
        teams_raw[i] for i in range(len(teams_raw))
        if teams_raw[i] is not None
        and not (isinstance(teams_raw[i], float) and np.isnan(teams_raw[i]))
        and i < len(team_confs)
        and team_confs[i] is not None
        and not (isinstance(team_confs[i], float) and np.isnan(team_confs[i]))
        and team_confs[i] >= TEAM_CONFIDENCE_THRESHOLD
    ]

    if teams_confident:
        mode_team   = max(set(teams_confident), key=teams_confident.count)
        consistency = teams_confident.count(mode_team) / len(teams_confident)
        feats["team_mode"]        = float(mode_team)
        feats["team_consistency"] = float(consistency)
        feats["team_coverage"]    = len(teams_confident) / max(len(frames), 1)
    else:
        feats["team_mode"]        = np.nan
        feats["team_consistency"] = 0.0
        feats["team_coverage"]    = 0.0

    feats["team_conf_mean"] = float(np.mean(valid_team_confs)) if valid_team_confs else 0.0

    # ---- Temporal / spatial ----
    feats["start_frame"] = float(frames[0])
    feats["end_frame"]   = float(frames[-1])
    feats["duration"]    = float(frames[-1] - frames[0] + 1)
    feats["n_frames"]    = float(len(frames))

    # bounding box centres at start and end
    def bbox_centre(bbox):
        return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

    sx, sy = bbox_centre(bboxes[0])
    ex, ey = bbox_centre(bboxes[-1])
    feats["start_x"] = float(sx)
    feats["start_y"] = float(sy)
    feats["end_x"]   = float(ex)
    feats["end_y"]   = float(ey)

    heights = [b[3] - b[1] for b in bboxes]
    feats["mean_bbox_height"] = float(np.mean(heights))

    # ---- GT label (only for pair generation — NOT passed to model) ----
    # We require the tracklet to be sufficiently pure before trusting its GT ID.
    # A tracklet containing frames from multiple real players (because the splitter
    # missed an identity switch) would produce a misleading label if we naively
    # took the majority. See TRACKLET_PURITY_THRESHOLD at the top of this file.
    gt_ids = tracklet.gt_attributes.get("track_ids", [])
    valid_gt = [g for g in gt_ids if g is not None and not (isinstance(g, float) and np.isnan(g))]
    if valid_gt:
        majority_gt = max(set(valid_gt), key=valid_gt.count)
        purity = valid_gt.count(majority_gt) / len(valid_gt)
        gt_mode = majority_gt if purity >= TRACKLET_PURITY_THRESHOLD else None
    else:
        gt_mode = None

    return {
        "track_id":        track_id,
        "gt_id":           gt_mode,
        "features":        feats,
        "reid_mean_vec":   reid_mean_vec,
        "siglip_mean_vec": siglip_mean_vec,
    }


# ---------------------------------------------------------------------------
# 2.  Pairwise feature computation
# ---------------------------------------------------------------------------

def compute_pairwise_features(agg_a: dict, agg_b: dict) -> dict:
    """
    Features that are defined on the pair rather than each tracklet in isolation.
    These are prefixed with 'pairwise_'.
    """
    fa = agg_a["features"]
    fb = agg_b["features"]
    pw = {}

    # Which tracklet ends first?
    if fa["end_frame"] <= fb["start_frame"]:
        exit_feat, entry_feat = fa, fb
    elif fb["end_frame"] <= fa["start_frame"]:
        exit_feat, entry_feat = fb, fa
    else:
        exit_feat, entry_feat = None, None   # temporal overlap

    # Temporal gap
    pw["temporal_gap"] = (
        float(entry_feat["start_frame"] - exit_feat["end_frame"])
        if exit_feat is not None else 0.0
    )

    # Spatial distance at the hand-off point
    if exit_feat is not None:
        dx = entry_feat["start_x"] - exit_feat["end_x"]
        dy = entry_feat["start_y"] - exit_feat["end_y"]
        pw["spatial_distance"] = float(np.sqrt(dx**2 + dy**2))
        pw["endpoint_dx"]      = float(dx)
        pw["endpoint_dy"]      = float(dy)
    else:
        pw["spatial_distance"] = 0.0
        pw["endpoint_dx"]      = 0.0
        pw["endpoint_dy"]      = 0.0

    # BBox height ratio (A / B)
    h_a = fa["mean_bbox_height"]
    h_b = fb["mean_bbox_height"]
    pw["bbox_height_ratio"] = float(h_a / h_b) if h_b > 0 else 1.0

    # ReID cosine similarity — read vectors from agg dict, not from feats columns
    reid_a = agg_a["reid_mean_vec"]
    reid_b = agg_b["reid_mean_vec"]
    na, nb = np.linalg.norm(reid_a) + 1e-6, np.linalg.norm(reid_b) + 1e-6
    pw["reid_cosine_sim"] = float(np.dot(reid_a / na, reid_b / nb))

    # SigLIP cosine similarity
    sig_a = agg_a["siglip_mean_vec"]
    sig_b = agg_b["siglip_mean_vec"]
    na_s, nb_s = np.linalg.norm(sig_a) + 1e-6, np.linalg.norm(sig_b) + 1e-6
    pw["siglip_cosine_sim"] = float(np.dot(sig_a / na_s, sig_b / nb_s))

    # Jersey agreement
    j_a, j_b = fa["jersey_mode"], fb["jersey_mode"]
    both_have_jersey = not (np.isnan(j_a) if isinstance(j_a, float) else False) \
                    and not (np.isnan(j_b) if isinstance(j_b, float) else False)
    pw["jersey_match"]     = float(both_have_jersey and j_a == j_b)
    pw["jersey_conflict"]  = float(both_have_jersey and j_a != j_b)
    pw["jersey_both_confident"] = float(
        both_have_jersey
        and fa["jersey_entropy_mean"] < 0.15
        and fb["jersey_entropy_mean"] < 0.15
    )

    # Team agreement
    t_a = fa["team_mode"]
    t_b = fb["team_mode"]
    both_have_team = not (np.isnan(t_a) if isinstance(t_a, float) else False) \
                  and not (np.isnan(t_b) if isinstance(t_b, float) else False)
    pw["team_match"]    = float(both_have_team and t_a == t_b)
    pw["team_conflict"] = float(both_have_team and t_a != t_b)
    pw["team_both_consistent"] = float(
        both_have_team
        and fa["team_consistency"] > 0.9
        and fb["team_consistency"] > 0.9
    )

    return pw


# ---------------------------------------------------------------------------
# 3.  Pair generation for one sequence
# ---------------------------------------------------------------------------

def generate_pairs_for_sequence(
    aggregated: List[dict],
    sequence: str,
    max_temporal_gap: int = MAX_TEMPORAL_GAP,
    negative_ratio: Optional[float] = NEGATIVE_RATIO,
) -> pd.DataFrame:
    """
    For a list of aggregated tracklet dicts, produce a DataFrame of candidate
    pairs. A pair is a candidate if the tracklets do not fully overlap in time
    and the temporal gap between them is within max_temporal_gap.

    Label = 1 if both tracklets share the same non-None gt_id (same player).
    """
    rows = []
    n    = len(aggregated)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = aggregated[i], aggregated[j]

            fa, fb = a["features"], b["features"]

            # Skip if one completely contains the other (overlap)
            frames_end_a   = fa["end_frame"]
            frames_start_a = fa["start_frame"]
            frames_end_b   = fb["end_frame"]
            frames_start_b = fb["start_frame"]

            # Determine gap
            if frames_end_a <= frames_start_b:
                gap = frames_start_b - frames_end_a
            elif frames_end_b <= frames_start_a:
                gap = frames_start_a - frames_end_b
            else:
                # Overlapping — cannot merge, skip
                continue

            if max_temporal_gap is not None and gap > max_temporal_gap:
                continue

            # Ground-truth label
            gt_a = a["gt_id"]
            gt_b = b["gt_id"]
            if gt_a is None or gt_b is None:
                label = 0   # unknown — treat as negative
            else:
                label = int(gt_a == gt_b)

            # Build the row
            pw = compute_pairwise_features(a, b)
            row = {"sequence": sequence, "label": label}

            for k, v in fa.items():
                row[f"A_{k}"] = v
            for k, v in fb.items():
                row[f"B_{k}"] = v
            for k, v in pw.items():
                row[f"pairwise_{k}"] = v

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Optional negative downsampling to balance dataset
    if negative_ratio is not None:
        pos = df[df["label"] == 1]
        neg = df[df["label"] == 0]
        n_keep = min(len(neg), int(len(pos) * negative_ratio))
        if n_keep < len(neg):
            neg = neg.sample(n=n_keep, random_state=42)
        df = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=42)

    return df


# ---------------------------------------------------------------------------
# 4.  Cache loading (pre-split attributes) + splitting + split-cache
# ---------------------------------------------------------------------------

def load_presplit_tracklets(sequence: str) -> Optional[dict]:
    """
    Load the attributes-enriched tracklets produced by predict_attributes().
    These are the raw (pre-split) tracklets.
    """
    path = CACHE_ROOT / f"cache_attributes_{sequence}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    print(f"  [WARN] No attributes cache found for {sequence} (tried: {path})")
    return None


def load_or_compute_split_tracklets(sequence: str) -> Optional[dict]:
    """
    Return post-split tracklets for *sequence*, using a cache so the splitter
    only runs once per sequence.

    Cache hit  → load CACHE_SPLIT_ROOT/cache_split_{sequence}.pkl
    Cache miss → load pre-split attributes cache, run split_tracklets(),
                 save result to split cache, return it.

    Delete the split cache files to force a re-split (e.g. after changing
    SIGNALS in tracklets/split_tracklets.py).
    """
    CACHE_SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
    split_cache_path = CACHE_SPLIT_ROOT / f"cache_split_{sequence}.pkl"

    if split_cache_path.exists():
        tqdm.write(f"  {sequence}: loading split cache")
        with open(split_cache_path, "rb") as f:
            return pickle.load(f)

    # Cache miss — load pre-split tracklets and run the splitter
    tqdm.write(f"  {sequence}: no split cache found, running splitter …")
    tracklets = load_presplit_tracklets(sequence)
    if tracklets is None:
        return None

    n_before   = len(tracklets)
    tracklets  = split_tracklets(tracklets, SPLITTER_CFG)
    n_after    = len(tracklets)
    tqdm.write(f"  {sequence}: {n_before} → {n_after} tracklets after splitting")

    with open(split_cache_path, "wb") as f:
        pickle.dump(tracklets, f)
    tqdm.write(f"  {sequence}: split cache saved → {split_cache_path}")

    return tracklets


# ---------------------------------------------------------------------------
# 5.  Main
# ---------------------------------------------------------------------------

def main():
    output_dir = OUTPUT_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dfs    = []
    total_pos  = 0
    total_neg  = 0

    for split in SPLITS_TO_PROCESS:
        split_root = DATA_ROOT / split
        if not split_root.exists():
            print(f"[SKIP] Split folder not found: {split_root}")
            continue

        sequences = sorted([d.name for d in split_root.iterdir() if d.is_dir()])
        print(f"\n{'='*70}")
        print(f"Processing split: {split}  ({len(sequences)} sequences)")
        print(f"{'='*70}")

        for seq in tqdm(sequences, desc=split):
            tracklets = load_or_compute_split_tracklets(seq)
            if tracklets is None:
                continue

            # Aggregate each tracklet
            aggregated = []
            for tid, t in tracklets.items():
                agg = aggregate_tracklet(t, tid)
                if agg is not None:
                    aggregated.append(agg)

            if len(aggregated) < 2:
                tqdm.write(f"  {seq}: too few tracklets ({len(aggregated)}), skipping")
                continue

            df_seq = generate_pairs_for_sequence(aggregated, sequence=seq)
            if df_seq.empty:
                tqdm.write(f"  {seq}: 0 pairs generated")
                continue

            pos = int(df_seq["label"].sum())
            neg = len(df_seq) - pos
            total_pos += pos
            total_neg += neg
            tqdm.write(f"  {seq}: {len(aggregated)} tracklets → {len(df_seq)} pairs  (pos={pos}, neg={neg})")
            all_dfs.append(df_seq)

    if not all_dfs:
        print("\n[ERROR] No data generated. Check your cache paths.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    # Drop the internal-only gt_id columns (they were only used for labeling)
    gt_cols = [c for c in combined.columns if c.endswith("_gt_id") or c == "gt_id"]
    combined.drop(columns=gt_cols, inplace=True, errors="ignore")

    out_path = output_dir / f"combined_{SPLIT_OUTPUT_NAME}_data.csv"
    combined.to_csv(out_path, index=False)

    print(f"\n{'='*70}")
    print(f"Done.")
    print(f"  Total pairs  : {len(combined):,}")
    print(f"  Positive     : {total_pos:,}  ({100*total_pos/max(len(combined),1):.1f}%)")
    print(f"  Negative     : {total_neg:,}  ({100*total_neg/max(len(combined),1):.1f}%)")
    print(f"  Feature cols : {len(combined.columns) - 2}")  # excl. label + sequence
    print(f"  Saved to     : {out_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

