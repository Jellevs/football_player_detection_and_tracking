"""
Generate FIXED validation and test sets for XGBoost that match inference distribution.

Unlike generate_merger_train_data.py, this script:
  - NO negative downsampling (keeps ALL pairs)
  - ALL real pairs from the splitter output (same as inference)
  - Uses the same aggregation, pairwise features, and purity threshold as training

These files should be generated ONCE and never changed. Only training data
should be varied across experiments.

Usage:
    python generate_fixed_eval_data.py

Output:
    training_data/fixed_eval/combined_valid_data.csv
    training_data/fixed_eval/combined_test_data.csv

Then point train_xgboost_merger.py to load val/test from training_data/fixed_eval/
instead of the strategy specific data directory.
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
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT        = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\soccernet-player-tracking")
CACHE_ROOT       = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\cache")
CACHE_SPLIT_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\cache_split")
OUTPUT_ROOT      = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\xgboost\training_data\fixed_eval")

# ---------------------------------------------------------------------------
# Settings — same as the best training config, but NO downsampling
# ---------------------------------------------------------------------------
MIN_TRACKLET_LEN          = 0
TRACKLET_PURITY_THRESHOLD = 0.6
REID_DIM                  = 512
TEAM_CONFIDENCE_THRESHOLD = 0.6
MAX_TEMPORAL_GAP          = None
NEGATIVE_RATIO            = None   # keep ALL negatives — no downsampling

SPLITTER_CFG = SplitterConfig()


# ---------------------------------------------------------------------------
# 1. Tracklet aggregation (identical to generate_merger_train_data.py)
# ---------------------------------------------------------------------------

def aggregate_tracklet(tracklet, track_id: int) -> Optional[dict]:
    """
    Collapse a variable length tracklet into a fixed size feature dict.
    Identical to the training data generator so columns match exactly.
    """
    frames     = tracklet.frames
    bboxes     = tracklet.bboxes
    embeddings = tracklet.embeddings

    if len(frames) < MIN_TRACKLET_LEN:
        return None
    if not embeddings:
        return None

    feats = {}

    # ---- ReID ----
    emb_arr  = np.stack(embeddings).astype(np.float32)
    norms    = np.linalg.norm(emb_arr, axis=1, keepdims=True) + 1e-6
    emb_norm = emb_arr / norms
    reid_mean_vec = emb_norm.mean(axis=0)

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

    # ---- GT label ----
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
    }


# ---------------------------------------------------------------------------
# 2. Pairwise feature computation (identical to generate_merger_train_data.py)
# ---------------------------------------------------------------------------

def compute_pairwise_features(agg_a: dict, agg_b: dict) -> dict:
    """Pairwise features, identical to the training data generator."""
    fa = agg_a["features"]
    fb = agg_b["features"]
    pw = {}

    if fa["end_frame"] <= fb["start_frame"]:
        exit_feat, entry_feat = fa, fb
    elif fb["end_frame"] <= fa["start_frame"]:
        exit_feat, entry_feat = fb, fa
    else:
        exit_feat, entry_feat = None, None

    pw["temporal_gap"] = (
        float(entry_feat["start_frame"] - exit_feat["end_frame"])
        if exit_feat is not None else 0.0
    )

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

    h_a = fa["mean_bbox_height"]
    h_b = fb["mean_bbox_height"]
    pw["bbox_height_ratio"] = float(h_a / h_b) if h_b > 0 else 1.0

    reid_a = agg_a["reid_mean_vec"]
    reid_b = agg_b["reid_mean_vec"]
    na, nb = np.linalg.norm(reid_a) + 1e-6, np.linalg.norm(reid_b) + 1e-6
    pw["reid_cosine_sim"] = float(np.dot(reid_a / na, reid_b / nb))

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
# 3. Post split cache loading
# ---------------------------------------------------------------------------

def load_or_compute_split_tracklets(sequence: str) -> Optional[dict]:
    """Load post split tracklets, running the splitter if not cached."""
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
# 4. Pair generation — ALL pairs, NO downsampling
# ---------------------------------------------------------------------------

def generate_all_pairs(
    aggregated: List[dict],
    sequence: str,
) -> pd.DataFrame:
    """
    Generate ALL non overlapping pairs. No negative downsampling.
    Same pairing logic as generate_merger_train_data.py but without ratio filtering.
    """
    rows = []

    for i in range(len(aggregated)):
        for j in range(i + 1, len(aggregated)):
            a, b = aggregated[i], aggregated[j]
            fa, fb = a["features"], b["features"]

            # Determine gap (skip overlapping)
            if fa["end_frame"] <= fb["start_frame"]:
                gap = fb["start_frame"] - fa["end_frame"]
            elif fb["end_frame"] <= fa["start_frame"]:
                gap = fa["start_frame"] - fb["end_frame"]
            else:
                continue

            if MAX_TEMPORAL_GAP is not None and gap > MAX_TEMPORAL_GAP:
                continue

            # Ground truth label
            gt_a = a["gt_id"]
            gt_b = b["gt_id"]
            if gt_a is None or gt_b is None:
                label = 0
            else:
                label = int(gt_a == gt_b)

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

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for split in ["valid", "test"]:
        split_root = DATA_ROOT / split
        if not split_root.exists():
            print(f"[SKIP] Split folder not found: {split_root}")
            continue

        sequences = sorted([d.name for d in split_root.iterdir() if d.is_dir()])
        print(f"\n{'='*60}")
        print(f"Generating FIXED {split} set ({len(sequences)} sequences)")
        print(f"  NO negative downsampling")
        print(f"  ALL real pairs from splitter output")
        print(f"  Purity threshold: {TRACKLET_PURITY_THRESHOLD}")
        print(f"{'='*60}")

        all_dfs   = []
        total_pos = 0
        total_neg = 0

        for seq in tqdm(sequences, desc=split):
            tracklets = load_or_compute_split_tracklets(seq)
            if tracklets is None:
                continue

            aggregated = []
            for tid, t in tracklets.items():
                agg = aggregate_tracklet(t, tid)
                if agg is not None:
                    aggregated.append(agg)

            if len(aggregated) < 2:
                tqdm.write(f"  {seq}: too few tracklets ({len(aggregated)}), skipping")
                continue

            df_seq = generate_all_pairs(aggregated, sequence=seq)
            if df_seq.empty:
                tqdm.write(f"  {seq}: 0 pairs generated")
                continue

            pos = int(df_seq["label"].sum())
            neg = len(df_seq) - pos
            total_pos += pos
            total_neg += neg
            tqdm.write(
                f"  {seq}: {len(aggregated)} tracklets -> "
                f"{len(df_seq)} pairs (pos={pos}, neg={neg})"
            )
            all_dfs.append(df_seq)

        if not all_dfs:
            print(f"\n[ERROR] No data generated for {split}.")
            continue

        combined = pd.concat(all_dfs, ignore_index=True)

        gt_cols = [c for c in combined.columns if c.endswith("_gt_id") or c == "gt_id"]
        combined.drop(columns=gt_cols, inplace=True, errors="ignore")

        out_path = OUTPUT_ROOT / f"combined_{split}_data.csv"
        combined.to_csv(out_path, index=False)

        neg_pos_ratio = total_neg / max(total_pos, 1)
        print(f"\n  {split} set:")
        print(f"    Total pairs     : {len(combined):,}")
        print(f"    Positive        : {total_pos:,}  ({100*total_pos/max(len(combined),1):.1f}%)")
        print(f"    Negative        : {total_neg:,}  ({100*total_neg/max(len(combined),1):.1f}%)")
        print(f"    Neg:Pos ratio   : {neg_pos_ratio:.1f}:1")
        print(f"    Saved           : {out_path}")

    print(f"\n{'='*60}")
    print(f"Done. These files should NEVER be regenerated.")
    print(f"Only modify training data generation scripts.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
