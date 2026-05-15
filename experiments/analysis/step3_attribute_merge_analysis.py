"""
step3_attribute_merge_analysis.py — Attribute-conditioned merge decision analysis.

For each pair of split tracklets (A ends before B starts, within a sequence),
computes pairwise features from the attribute cache, then checks:
  - Which pairs were merged by each method (via MOT output files)
  - Which pairs SHOULD have been merged (via GT MOT files)

Outputs per-method accuracy breakdowns stratified by:
  - jersey_conf  : mean jersey confidence of the weaker tracklet in the pair
  - jersey_agree : fraction of frames where jersey number agrees between the pair
  - team_agree   : fraction of frames where team label agrees (on overlap window)
  - gap_frames   : temporal gap between A's last frame and B's first frame
  - spatial_dist : Euclidean distance between A's exit centroid and B's entry centroid

Configuration
-------------
Set the GT_ROOT path below.  The script expects:
    GT_ROOT / SEQUENCE / gt / gt.txt
in standard MOT format (frame, id, x, y, w, h, conf, cx, cy, cz).

Set METHODS:
    key   = display label used in output tables
    value = folder path relative to evaluation/SNPT/
            (can be nested, e.g. "mergers/xgboost/test/merger_jersey_team_bbox")

The script reads MOT files from:
    <repo>/evaluation/SNPT/<folder_path>/data/<SEQUENCE>.txt

Run from the repo root:
    python experiments/analysis/step3_attribute_merge_analysis.py
"""

from __future__ import annotations

import sys
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))

import settings

# ── configuration ─────────────────────────────────────────────────────────────

# Path to the folder that contains SEQUENCE/gt/gt.txt files
GT_ROOT: Path = settings.DATA_ROOT   # same as pipeline's DATA_ROOT

CACHE_DIR: Path = REPO / "output" / "cache"

# key   = display label
# value = folder path relative to evaluation/SNPT/
METHODS: dict[str, str] = {
    "Baseline":  "baseline",
    "XGBoost":   "mergers/xgboost/test/merger_jersey_team_bbox",
    "GTA":       "mergers/gta_split+merge",
}

EVAL_ROOT = REPO / "evaluation" / "SNPT"
OUT_DIR   = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Maximum temporal gap to consider a pair as a potential merge candidate.
# Pairs with a gap larger than this are very unlikely to be merges.
MAX_GAP_FRAMES: int = 200

# Maximum spatial distance (pixels) to consider — filters out obvious non-merges
MAX_SPATIAL_DIST: float = 400.0

# Minimum tracklet length (frames) — ignore tiny fragments
MIN_TRACKLET_LEN: int = 5

# IoU threshold for matching a predicted/GT detection
IOU_MATCH_THRESH: float = 0.3

# Feature bins for the stratified analysis
BINS = {
    "gap_frames":    [0, 10, 30, 60, 120, 200],
    "spatial_dist":  [0, 50, 100, 200, 400],
    "jersey_conf":   [0.0, 0.5, 0.7, 0.9, 1.0],
    "jersey_agree":  [0.0, 0.25, 0.5, 0.75, 1.01],
    "team_agree":    [0.0, 0.5, 0.75, 0.9, 1.01],
}

# ── MOT I/O ──────────────────────────────────────────────────────────────────

def load_mot(path: Path) -> dict[int, list[tuple]]:
    """
    Load a MOT file into {frame_1indexed: [(track_id, x, y, w, h), ...]}.
    Returns empty dict if path does not exist.
    """
    if not path.exists():
        return {}
    result: dict[int, list] = defaultdict(list)
    with path.open() as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            fr, tid, x, y, w, h = int(float(parts[0])), int(float(parts[1])), \
                float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            result[fr].append((tid, x, y, w, h))
    return result


def bbox_iou(ax, ay, aw, ah, bx, by, bw, bh) -> float:
    """Compute IoU between two TLWH bboxes."""
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def xyxy_to_tlwh(bbox_xyxy) -> tuple:
    x1, y1, x2, y2 = bbox_xyxy
    return x1, y1, x2 - x1, y2 - y1


def match_tracklet_to_ids(
    tracklet,
    mot: dict[int, list],
    frame_offset: int = 1,
) -> int | None:
    """
    For a given tracklet (0-indexed frames, xyxy bboxes), find the track_id in
    *mot* that best explains the tracklet's detections.

    We sample up to 20 frames, find the best-IoU match in the MOT file for each,
    and return the plurality track_id.  Returns None if no clear match is found.
    """
    votes: dict[int, int] = defaultdict(int)
    step = max(1, len(tracklet.frames) // 20)

    for i in range(0, len(tracklet.frames), step):
        fr_0 = tracklet.frames[i]
        fr_1 = fr_0 + frame_offset         # MOT files are 1-indexed
        bbox = tracklet.bboxes[i]
        tx, ty, tw, th = xyxy_to_tlwh(bbox)

        candidates = mot.get(fr_1, [])
        best_iou, best_tid = 0.0, None
        for tid, x, y, w, h in candidates:
            iou = bbox_iou(tx, ty, tw, th, x, y, w, h)
            if iou > best_iou:
                best_iou, best_tid = iou, tid
        if best_iou >= IOU_MATCH_THRESH and best_tid is not None:
            votes[best_tid] += 1

    if not votes:
        return None
    return max(votes, key=votes.__getitem__)


# ── feature extraction ────────────────────────────────────────────────────────

def centroid(bbox_xyxy) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def tracklet_exit_entry(t) -> tuple[np.ndarray, np.ndarray]:
    """Return (exit_centroid, entry_centroid) from last / first bbox."""
    return centroid(t.bboxes[-1]), centroid(t.bboxes[0])


def compute_jersey_agree(ta, tb) -> float:
    """
    Whether the dominant jersey number agrees between the two tracklets.
    Ignores nan values.
    """
    def jersey_mode(t):
        attrs = getattr(t, "pred_attributes", None)
        if attrs is None:
            return np.nan
        jerseys = [j for j in attrs.get("jerseys", [])
                   if not (isinstance(j, float) and np.isnan(j))]
        if not jerseys:
            return np.nan
        vals, counts = np.unique(jerseys, return_counts=True)
        return vals[np.argmax(counts)]

    j_a = jersey_mode(ta)
    j_b = jersey_mode(tb)
    if np.isnan(j_a) or np.isnan(j_b):
        return np.nan
    return 1.0 if j_a == j_b else 0.0


def compute_team_agree(ta, tb) -> float:
    """
    Whether the dominant team label agrees between the two tracklets.
    """
    def team_mode(t):
        attrs = getattr(t, "pred_attributes", None)
        if attrs is None:
            return None
        teams = attrs.get("teams", [])
        if not teams:
            return None
        vals, counts = np.unique(teams, return_counts=True)
        return int(vals[np.argmax(counts)])

    ta_team = team_mode(ta)
    tb_team = team_mode(tb)
    if ta_team is None or tb_team is None:
        return np.nan
    return 1.0 if ta_team == tb_team else 0.0


def compute_jersey_conf(ta, tb) -> float:
    """Mean jersey_confs_mean for the weaker (lower confidence) tracklet."""
    def mean_conf(t):
        attrs = getattr(t, "pred_attributes", None)
        if attrs is None:
            return np.nan
        jcm = attrs.get("jersey_confs_mean", [])
        valid = [v for v in jcm if not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(valid)) if valid else np.nan

    ca, cb = mean_conf(ta), mean_conf(tb)
    vals = [v for v in [ca, cb] if not np.isnan(v)]
    return float(min(vals)) if vals else np.nan


def compute_pair_features(ta, tb) -> dict:
    """Compute all pairwise features for a (ta comes before tb) pair."""
    gap_frames   = tb.frames[0] - ta.frames[-1]
    exit_a, _   = tracklet_exit_entry(ta)
    _, entry_b  = tracklet_exit_entry(tb)
    spatial_dist = float(np.linalg.norm(exit_a - entry_b))

    return {
        "gap_frames":   gap_frames,
        "spatial_dist": spatial_dist,
        "jersey_conf":  compute_jersey_conf(ta, tb),
        "jersey_agree": compute_jersey_agree(ta, tb),
        "team_agree":   compute_team_agree(ta, tb),
    }


# ── analysis helpers ──────────────────────────────────────────────────────────

def bin_feature(val, bins: list) -> str:
    """Return the bin label for *val* using *bins* as edges."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "nan"
    for i in range(len(bins) - 1):
        if bins[i] <= val < bins[i + 1]:
            return f"[{bins[i]}, {bins[i+1]})"
    return f">={bins[-1]}"


def analyse_pairs(pairs_df: pd.DataFrame, methods: list[str]) -> None:
    """Print accuracy tables for each feature x method."""

    for feat, bins in BINS.items():
        if feat not in pairs_df.columns:
            continue
        print(f"\n-- Stratified by: {feat} --")
        pairs_df["_bin"] = pairs_df[feat].apply(lambda v: bin_feature(v, bins))
        bin_order = [f"[{bins[i]}, {bins[i+1]})" for i in range(len(bins) - 1)] + ["nan"]

        header = f"{'Bin':<22}"
        for m in methods:
            header += f"  {m[:10]+' Prec':>14}  {'Rec':>7}  {'F1':>6}  {'n':>5}"
        print(header)
        print("-" * len(header))

        for b in bin_order:
            grp = pairs_df[pairs_df["_bin"] == b]
            if len(grp) == 0:
                continue
            row_str = f"{b:<22}"
            for m in methods:
                col_pred = f"merged_{m}"
                if col_pred not in grp.columns:
                    row_str += " " * 38
                    continue
                tp = ((grp[col_pred] == 1) & (grp["should_merge"] == 1)).sum()
                fp = ((grp[col_pred] == 1) & (grp["should_merge"] == 0)).sum()
                fn = ((grp[col_pred] == 0) & (grp["should_merge"] == 1)).sum()
                prec   = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
                recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
                f1     = (2 * prec * recall / (prec + recall)
                          if not (np.isnan(prec) or np.isnan(recall))
                          and (prec + recall) > 0
                          else float("nan"))
                row_str += f"  {prec:>14.3f}  {recall:>7.3f}  {f1:>6.3f}  {len(grp):>5}"
            print(row_str)

        pairs_df.drop(columns=["_bin"], inplace=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Step 3 -- Attribute-conditioned merge analysis ===\n")

    # Load sequence list from cache
    cache_files = sorted(CACHE_DIR.glob("cache_attributes_SNPT-*.pkl"))
    if not cache_files:
        print(f"No attribute caches found in {CACHE_DIR}. Run main.py first.")
        sys.exit(1)

    sequences = [p.stem.replace("cache_attributes_", "") for p in cache_files]
    print(f"Found {len(sequences)} sequences with attribute caches.\n")

    # Verify which methods have MOT files
    available_methods = []
    for label, folder_path in METHODS.items():
        sample_seq = sequences[0]
        if (EVAL_ROOT / folder_path / "data" / f"{sample_seq}.txt").exists():
            available_methods.append(label)
        else:
            print(f"  [WARN] Method '{label}' ({folder_path}) missing MOT data -- skipping.")
    print(f"Methods available: {available_methods}\n")

    # Check GT availability
    sample_gt = GT_ROOT / sequences[0] / "gt" / "gt.txt"
    has_gt = sample_gt.exists()
    if not has_gt:
        print(f"  [WARN] GT not found at {sample_gt.parent}.")
        print("  Running in NO-GT mode: will report per-method merge rates only.\n")
    else:
        print(f"GT root: {GT_ROOT}\n")

    # ── collect pair records ──────────────────────────────────────────────────
    all_records = []

    for seq in sequences:
        cache_path = CACHE_DIR / f"cache_attributes_{seq}.pkl"
        with cache_path.open("rb") as f:
            tracklets: dict = pickle.load(f)

        # Filter short tracklets
        tracklets = {tid: t for tid, t in tracklets.items()
                     if len(t.frames) >= MIN_TRACKLET_LEN}

        if len(tracklets) < 2:
            continue

        # Load MOT predictions for each method
        method_mots: dict[str, dict] = {}
        for m in available_methods:
            mot_path = EVAL_ROOT / METHODS[m] / "data" / f"{seq}.txt"
            method_mots[m] = load_mot(mot_path)

        # Load GT
        gt_mot: dict[int, list] | None = None
        if has_gt:
            gt_mot = load_mot(GT_ROOT / seq / "gt" / "gt.txt")

        # Map each tracklet to its output track_id per method
        tracklet_pred_id: dict[str, dict[int, int | None]] = {m: {} for m in available_methods}
        for m in available_methods:
            for tid, t in tracklets.items():
                tracklet_pred_id[m][tid] = match_tracklet_to_ids(t, method_mots[m])

        # Map each tracklet to its GT track_id
        tracklet_gt_id: dict[int, int | None] = {}
        if gt_mot is not None:
            for tid, t in tracklets.items():
                tracklet_gt_id[tid] = match_tracklet_to_ids(t, gt_mot)

        # Sort tracklets by first frame for efficient pairing
        sorted_tracklets = sorted(tracklets.values(), key=lambda t: t.frames[0])

        # Generate pairs: A ends before B starts, gap <= MAX_GAP_FRAMES
        for i, ta in enumerate(sorted_tracklets):
            for tb in sorted_tracklets[i + 1:]:
                if ta.frames[-1] >= tb.frames[0]:
                    continue
                gap = tb.frames[0] - ta.frames[-1]
                if gap > MAX_GAP_FRAMES:
                    break   # sorted by start, so all further tb are also far

                # Spatial filter
                exit_a = centroid(ta.bboxes[-1])
                entry_b = centroid(tb.bboxes[0])
                dist = float(np.linalg.norm(exit_a - entry_b))
                if dist > MAX_SPATIAL_DIST:
                    continue

                record: dict = {"seq": seq, "tid_a": ta.track_id, "tid_b": tb.track_id}

                # Pairwise features
                feats = compute_pair_features(ta, tb)
                record.update(feats)

                # Ground truth merge decision
                if gt_mot is not None:
                    gt_a = tracklet_gt_id.get(ta.track_id)
                    gt_b = tracklet_gt_id.get(tb.track_id)
                    if gt_a is not None and gt_b is not None:
                        record["should_merge"] = int(gt_a == gt_b)
                    else:
                        record["should_merge"] = np.nan
                else:
                    record["should_merge"] = np.nan

                # Per-method merge decision
                for m in available_methods:
                    pid_a = tracklet_pred_id[m].get(ta.track_id)
                    pid_b = tracklet_pred_id[m].get(tb.track_id)
                    if pid_a is not None and pid_b is not None:
                        record[f"merged_{m}"] = int(pid_a == pid_b)
                    else:
                        record[f"merged_{m}"] = np.nan

                all_records.append(record)

    if not all_records:
        print("No pairs found. Check MAX_GAP_FRAMES and cache contents.")
        sys.exit(1)

    pairs_df = pd.DataFrame(all_records)
    print(f"Total pairs collected: {len(pairs_df)}\n")

    # Save raw pairs table
    out_csv = OUT_DIR / "step3_merge_pairs.csv"
    pairs_df.to_csv(out_csv, index=False)
    print(f"Raw pairs table -> {out_csv}\n")

    # ── overall statistics ────────────────────────────────────────────────────
    print("=== Overall merge statistics ===")
    if "should_merge" in pairs_df and not pairs_df["should_merge"].isna().all():
        gt_valid = pairs_df.dropna(subset=["should_merge"])
        pos = int(gt_valid["should_merge"].sum())
        neg = len(gt_valid) - pos
        print(f"  GT positives (should merge): {pos}")
        print(f"  GT negatives (should NOT):   {neg}")
        print(f"  Total with GT labels:         {len(gt_valid)}")
        print()

        print(f"  {'Method':<30}  {'Merged':>7}  {'TP':>5}  {'FP':>5}  {'FN':>5}  "
              f"{'Prec':>6}  {'Rec':>6}  {'F1':>6}")
        for m in available_methods:
            col = f"merged_{m}"
            if col not in gt_valid.columns:
                continue
            sub = gt_valid.dropna(subset=[col])
            tp = int(((sub[col] == 1) & (sub["should_merge"] == 1)).sum())
            fp = int(((sub[col] == 1) & (sub["should_merge"] == 0)).sum())
            fn = int(((sub[col] == 0) & (sub["should_merge"] == 1)).sum())
            merged = int((sub[col] == 1).sum())
            prec   = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            f1 = (2 * prec * recall / (prec + recall)
                  if not (np.isnan(prec) or np.isnan(recall)) and (prec + recall) > 0
                  else float("nan"))
            print(
                f"  {m:<30}  {merged:>7}  {tp:>5}  {fp:>5}  {fn:>5}  "
                f"{prec:>6.3f}  {recall:>6.3f}  {f1:>6.3f}"
            )
    else:
        print("  No GT labels available -- reporting merge counts only.")
        print(f"\n  {'Method':<30}  {'Pairs merged':>12}  {'% merged':>9}")
        for m in available_methods:
            col = f"merged_{m}"
            if col not in pairs_df.columns:
                continue
            sub = pairs_df.dropna(subset=[col])
            n_merged = int((sub[col] == 1).sum())
            pct = 100 * n_merged / len(sub) if len(sub) > 0 else float("nan")
            print(f"  {m:<30}  {n_merged:>12}  {pct:>8.1f}%")

    # ── stratified analysis ───────────────────────────────────────────────────
    print("\n\n=== Stratified analysis (precision / recall / F1 per feature bin) ===")

    if "should_merge" in pairs_df and not pairs_df["should_merge"].isna().all():
        print("(Only pairs with GT labels included)\n")
        gt_valid = pairs_df.dropna(subset=["should_merge"]).copy()
        analyse_pairs(gt_valid, available_methods)
    else:
        print("No GT labels -- showing merge rate per feature bin.\n")
        pairs_copy = pairs_df.copy()
        for feat, bins in BINS.items():
            if feat not in pairs_copy.columns:
                continue
            print(f"\n-- {feat} --")
            pairs_copy["_bin"] = pairs_copy[feat].apply(lambda v: bin_feature(v, bins))
            bin_order = [f"[{bins[i]}, {bins[i+1]})" for i in range(len(bins) - 1)] + ["nan"]
            for b in bin_order:
                grp = pairs_copy[pairs_copy["_bin"] == b]
                if len(grp) == 0:
                    continue
                row = f"  {b:<22}"
                for m in available_methods:
                    col = f"merged_{m}"
                    sub = grp.dropna(subset=[col])
                    if len(sub) == 0:
                        continue
                    rate = (sub[col] == 1).mean()
                    row += f"  {m[:12]}: {rate:.2f} (n={len(sub)})"
                print(row)

    print(f"\nDone. Raw pairs table at: {out_csv}")


if __name__ == "__main__":
    main()
