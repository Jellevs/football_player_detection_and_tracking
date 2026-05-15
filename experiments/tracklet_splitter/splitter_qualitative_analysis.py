"""
splitter_qualitative_analysis.py — Characterize caught vs uncaught identity switches.

For each of the 1048 ground-truth identity switches in the baseline tracklets,
this script classifies them along several dimensions and compares caught vs
uncaught switches for a given splitter configuration.

Analyses:
  1. Same-team vs cross-team switches
  2. Switch duration (how many frames the "wrong" identity persists)
  3. Whether a temporal gap exists near the switch point
  4. Spatial proximity (bbox IoU) between the two players at the switch frame
  5. Per-sequence breakdown (which sequences are hardest)

Requirements:
  - Ground-truth MOT files:  DATA_ROOT / SNPT-XXX / gt / gt.txt
  - Baseline tracker MOT:    evaluation/SNPT/baseline/data/SNPT-XXX.txt
  - Splitter output MOT:     (configurable, e.g. the recommended config)
  - Original GSR JSON:       GSR_ROOT / test / SNGS-XXX / Labels-GameState.json
    (for team identity labels)

Usage:
    python experiments/tracklet_splitter/splitter_qualitative_analysis.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Config — edit these paths to match your setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import settings
GT_ROOT = settings.DATA_ROOT  # .../soccernet-player-tracking/test
BASELINE_DIR = REPO_ROOT / "evaluation" / "SNPT" / "baseline" / "data"
SEQMAP = REPO_ROOT / "evaluation" / "seqmaps" / "SNPT-test.txt"

# Path to the original SoccerNet GSR dataset (contains Labels-GameState.json)
# Adjust this if your GSR data is in a different location.
GSR_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\data\SoccerNetGS\test")

# Splitter output to evaluate (the recommended config)
# Set to the same as BASELINE_DIR to analyse the baseline (0 catches).
# Or point to a specific splitter output.
SPLITTER_OUTPUT_DIR = REPO_ROOT / "evaluation" / "SNPT" / "splitters" / "test" / "wo_reid_splitter" / "splitter_temporalreid_jersey_team_bbox" / "data"

# If you want to evaluate the full recommended config (STR + Jersey + Team + Bbox),
# you may need to point this to the appropriate output folder. Check what's available:
#   ls evaluation/SNPT/
# Common options:
#   SPLITTER_OUTPUT_DIR = REPO_ROOT / "evaluation" / "SNPT" / "merger_temporalreid_jersey_team_bbox_traj" / "data"
# NOTE: The merger output already includes merging. If you have a splitter-only output
# (before merging), use that instead for cleaner analysis.

TOLERANCE = 10   # frame tolerance for matching splits to GT switches
IOU_THRESH = 0.5 # IoU threshold for matching detections to GT
GAP_WINDOW = 5   # frames: if a tracklet has no detection within ±GAP_WINDOW of the
                  # switch, we consider a temporal gap present

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# MOT file IO  (adapted from intrinsic_metrics.py)
# ---------------------------------------------------------------------------

def load_mot_by_track(path: Path) -> Dict[int, List[Tuple[int, np.ndarray]]]:
    """track_id -> list of (frame, bbox_xywh) sorted by frame."""
    tracks: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            track_id = int(float(parts[1]))
            box = np.array([float(parts[2]), float(parts[3]),
                            float(parts[4]), float(parts[5])], dtype=np.float32)
            tracks[track_id].append((frame, box))
    for tid in tracks:
        tracks[tid].sort(key=lambda fb: fb[0])
    return tracks


def load_mot_by_frame(path: Path) -> Dict[int, List[Tuple[int, np.ndarray]]]:
    """frame -> list of (track_id, bbox_xywh)."""
    by_frame: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            track_id = int(float(parts[1]))
            box = np.array([float(parts[2]), float(parts[3]),
                            float(parts[4]), float(parts[5])], dtype=np.float32)
            by_frame[frame].append((track_id, box))
    return by_frame


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def bbox_center_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between bbox centers (xywh format)."""
    cx_a = a[0] + a[2] / 2
    cy_a = a[1] + a[3] / 2
    cx_b = b[0] + b[2] / 2
    cy_b = b[1] + b[3] / 2
    return float(np.sqrt((cx_a - cx_b)**2 + (cy_a - cy_b)**2))


def best_match(box: np.ndarray, candidates: List[Tuple[int, np.ndarray]],
               iou_threshold: float) -> int:
    best_iou = 0.0
    best_id = -1
    for cid, cbox in candidates:
        i = iou_xywh(box, cbox)
        if i > best_iou:
            best_iou = i
            best_id = cid
    return best_id if best_iou >= iou_threshold else -1


# ---------------------------------------------------------------------------
# GT assignment (same as intrinsic_metrics.py)
# ---------------------------------------------------------------------------

def build_det_to_gt(
    det_by_frame: Dict[int, List[Tuple[int, np.ndarray]]],
    gt_by_frame: Dict[int, List[Tuple[int, np.ndarray]]],
    iou_threshold: float,
) -> Dict[Tuple[int, int], int]:
    """Map (frame, det_track_id) -> gt_track_id. Returns -1 for unmatched."""
    out: Dict[Tuple[int, int], int] = {}
    for frame, dets in det_by_frame.items():
        gts = gt_by_frame.get(frame, [])
        for tid, box in dets:
            out[(frame, tid)] = best_match(box, gts, iou_threshold)
    return out


def build_det_to_gt_with_coords(
    det_by_frame: Dict[int, List[Tuple[int, np.ndarray]]],
    gt_by_frame: Dict[int, List[Tuple[int, np.ndarray]]],
    iou_threshold: float,
) -> Dict[Tuple[int, int, float, float], int]:
    """Map (frame, det_track_id, x, y) -> gt_track_id (for disambiguation)."""
    out = {}
    for frame, dets in det_by_frame.items():
        gts = gt_by_frame.get(frame, [])
        for tid, box in dets:
            key = (frame, tid, float(box[0]), float(box[1]))
            out[key] = best_match(box, gts, iou_threshold)
    return out


# ---------------------------------------------------------------------------
# GSR team label extraction
# ---------------------------------------------------------------------------

def load_gsr_team_labels(gsr_json_path: Path) -> Dict[int, str]:
    """
    Read Labels-GameState.json and return {gt_track_id: team_label}.
    Team label is typically "left" or "right" (or the actual team name).
    If a track has multiple team labels across frames, use majority vote.
    """
    with open(gsr_json_path, "r") as f:
        data = json.load(f)

    # Build image_id -> frame lookup
    img_to_frame = {}
    for img in data.get("images", []):
        img_to_frame[img["image_id"]] = int(Path(img["file_name"]).stem)

    # Collect team labels per track_id
    track_teams: Dict[int, List[str]] = defaultdict(list)
    for ann in data.get("annotations", []):
        cat_id = ann.get("category_id", 0)
        if cat_id not in {1, 2}:  # players and goalkeepers only
            continue
        track_id = ann.get("track_id")
        attrs = ann.get("attributes", {})
        team = attrs.get("team", None)
        if track_id is not None and team is not None:
            track_teams[track_id].append(str(team))

    # Majority vote per track
    result = {}
    for tid, teams in track_teams.items():
        from collections import Counter
        counts = Counter(teams)
        result[tid] = counts.most_common(1)[0][0]

    return result


def snpt_to_sngs(seq_name: str) -> str:
    """Convert SNPT-116 -> SNGS-116."""
    return seq_name.replace("SNPT-", "SNGS-")


# ---------------------------------------------------------------------------
# Identity switch data structure
# ---------------------------------------------------------------------------

@dataclass
class IdentitySwitch:
    sequence: str
    baseline_track_id: int
    frame: int                    # frame where the GT identity changes
    gt_id_before: int             # GT track ID before the switch
    gt_id_after: int              # GT track ID after the switch
    team_before: Optional[str] = None
    team_after: Optional[str] = None
    same_team: Optional[bool] = None
    duration_frames: int = 0      # how long the new identity persists
    has_temporal_gap: bool = False # is there a temporal gap near the switch
    bbox_distance: float = 0.0    # center distance between the 2 players at switch
    bbox_iou: float = 0.0        # IoU between the 2 players at switch
    caught: bool = False          # did the splitter catch this switch


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def find_identity_switches_detailed(
    baseline_tracks: Dict[int, List[Tuple[int, np.ndarray]]],
    gt_by_frame: Dict[int, List[Tuple[int, np.ndarray]]],
    det_gt_map: Dict[Tuple[int, int, float, float], int],
    sequence: str,
) -> List[IdentitySwitch]:
    """Find all identity switches in baseline tracklets with detailed info."""
    switches = []

    for tid, frames_boxes in baseline_tracks.items():
        prev_gt = None
        prev_frame = None

        for i, (frame, box) in enumerate(frames_boxes):
            key = (frame, tid, float(box[0]), float(box[1]))
            gt_id = det_gt_map.get(key, -1)
            if gt_id == -1:
                continue

            if prev_gt is not None and gt_id != prev_gt:
                # Identity switch detected!
                sw = IdentitySwitch(
                    sequence=sequence,
                    baseline_track_id=tid,
                    frame=frame,
                    gt_id_before=prev_gt,
                    gt_id_after=gt_id,
                )

                # Compute switch duration: how long does gt_id persist
                duration = 0
                for j in range(i, len(frames_boxes)):
                    f_j, b_j = frames_boxes[j]
                    key_j = (f_j, tid, float(b_j[0]), float(b_j[1]))
                    gt_j = det_gt_map.get(key_j, -1)
                    if gt_j == gt_id:
                        duration += 1
                    elif gt_j != -1:
                        break
                sw.duration_frames = duration

                # Check for temporal gap near the switch
                if prev_frame is not None:
                    frame_gap = frame - prev_frame
                    sw.has_temporal_gap = frame_gap > GAP_WINDOW

                # Compute spatial proximity: find both GT players at the switch frame
                gt_dets_at_frame = gt_by_frame.get(frame, [])
                bbox_before = None
                bbox_after = None
                for gt_tid, gt_box in gt_dets_at_frame:
                    if gt_tid == prev_gt:
                        bbox_before = gt_box
                    if gt_tid == gt_id:
                        bbox_after = gt_box
                if bbox_before is not None and bbox_after is not None:
                    sw.bbox_distance = bbox_center_distance(bbox_before, bbox_after)
                    sw.bbox_iou = iou_xywh(bbox_before, bbox_after)

                switches.append(sw)

            if gt_id != -1:
                prev_gt = gt_id
                prev_frame = frame

    return switches


def find_predicted_splits(
    baseline_tracks: Dict[int, List[Tuple[int, np.ndarray]]],
    splitter_by_frame: Dict[int, List[Tuple[int, np.ndarray]]],
    iou_threshold: float,
) -> List[Tuple[int, int]]:
    """Return (baseline_track_id, frame) for each predicted split."""
    splits = []
    for in_tid, frames_boxes in baseline_tracks.items():
        prev_pid = None
        for frame, box in frames_boxes:
            cand = splitter_by_frame.get(frame, [])
            pid = best_match(box, cand, iou_threshold)
            if pid == -1:
                continue
            if prev_pid is not None and pid != prev_pid:
                splits.append((in_tid, frame))
            prev_pid = pid
    return splits


def mark_caught_switches(
    switches: List[IdentitySwitch],
    predicted_splits: List[Tuple[int, int]],
    tolerance: int,
) -> None:
    """Mark each switch as caught or not, using greedy matching."""
    pred_by_tid: Dict[int, List[int]] = defaultdict(list)
    for tid, frame in predicted_splits:
        pred_by_tid[tid].append(frame)

    switch_by_tid: Dict[int, List[IdentitySwitch]] = defaultdict(list)
    for sw in switches:
        switch_by_tid[sw.baseline_track_id].append(sw)

    for tid in set(list(pred_by_tid.keys()) + list(switch_by_tid.keys())):
        sw_list = sorted(switch_by_tid.get(tid, []), key=lambda s: s.frame)
        p_list = sorted(pred_by_tid.get(tid, []))
        used = [False] * len(p_list)

        for sw in sw_list:
            best_idx = -1
            best_d = tolerance + 1
            for i, pf in enumerate(p_list):
                if used[i]:
                    continue
                d = abs(pf - sw.frame)
                if d <= tolerance and d < best_d:
                    best_d = d
                    best_idx = i
            if best_idx >= 0:
                used[best_idx] = True
                sw.caught = True


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_analysis(switches: List[IdentitySwitch]) -> None:
    total = len(switches)
    caught = [s for s in switches if s.caught]
    uncaught = [s for s in switches if not s.caught]

    print(f"\n{'='*70}")
    print(f"SPLITTER QUALITATIVE ANALYSIS")
    print(f"{'='*70}")
    print(f"Total identity switches: {total}")
    print(f"Caught: {len(caught)} ({100*len(caught)/total:.1f}%)")
    print(f"Uncaught: {len(uncaught)} ({100*len(uncaught)/total:.1f}%)")

    # ── 1. Same-team vs cross-team ────────────────────────────────────────
    has_team = [s for s in switches if s.same_team is not None]
    if has_team:
        print(f"\n--- SAME-TEAM vs CROSS-TEAM ---")
        same_all = [s for s in has_team if s.same_team]
        cross_all = [s for s in has_team if not s.same_team]
        same_caught = [s for s in same_all if s.caught]
        cross_caught = [s for s in cross_all if s.caught]
        same_uncaught = [s for s in same_all if not s.caught]
        cross_uncaught = [s for s in cross_all if not s.caught]

        print(f"  {'':30s} {'All':>8} {'Caught':>8} {'Uncaught':>8} {'Catch%':>8}")
        print(f"  {'Same team':30s} {len(same_all):>8} {len(same_caught):>8} {len(same_uncaught):>8} {100*len(same_caught)/max(1,len(same_all)):>7.1f}%")
        print(f"  {'Cross team':30s} {len(cross_all):>8} {len(cross_caught):>8} {len(cross_uncaught):>8} {100*len(cross_caught)/max(1,len(cross_all)):>7.1f}%")
        print(f"  {'Unknown team':30s} {total - len(has_team):>8}")

        if uncaught:
            same_pct = 100 * len(same_uncaught) / max(1, len(uncaught))
            print(f"\n  Of uncaught switches: {same_pct:.1f}% are same-team, {100-same_pct:.1f}% are cross-team")
        if caught:
            same_pct_c = 100 * len(same_caught) / max(1, len(caught))
            print(f"  Of caught switches:   {same_pct_c:.1f}% are same-team, {100-same_pct_c:.1f}% are cross-team")

    # ── 2. Switch duration ────────────────────────────────────────────────
    print(f"\n--- SWITCH DURATION (frames the wrong ID persists) ---")
    bins = [(0, 5, "<5"), (5, 25, "5-25"), (25, 100, "25-100"),
            (100, 300, "100-300"), (300, 99999, "300+")]

    print(f"  {'Duration':20s} {'All':>6} {'Caught':>8} {'Uncaught':>8} {'Catch%':>8}")
    for lo, hi, label in bins:
        all_bin = [s for s in switches if lo <= s.duration_frames < hi]
        caught_bin = [s for s in all_bin if s.caught]
        uncaught_bin = [s for s in all_bin if not s.caught]
        pct = 100 * len(caught_bin) / max(1, len(all_bin))
        print(f"  {label:20s} {len(all_bin):>6} {len(caught_bin):>8} {len(uncaught_bin):>8} {pct:>7.1f}%")

    durations_caught = [s.duration_frames for s in caught]
    durations_uncaught = [s.duration_frames for s in uncaught]
    if durations_caught:
        print(f"\n  Mean duration (caught):   {np.mean(durations_caught):.1f} frames")
    if durations_uncaught:
        print(f"  Mean duration (uncaught): {np.mean(durations_uncaught):.1f} frames")

    # ── 3. Temporal gap ───────────────────────────────────────────────────
    print(f"\n--- TEMPORAL GAP (>{GAP_WINDOW} frame gap near switch point) ---")
    gap_all = [s for s in switches if s.has_temporal_gap]
    nogap_all = [s for s in switches if not s.has_temporal_gap]
    gap_caught = [s for s in gap_all if s.caught]
    nogap_caught = [s for s in nogap_all if s.caught]

    print(f"  {'':20s} {'All':>6} {'Caught':>8} {'Uncaught':>8} {'Catch%':>8}")
    print(f"  {'With gap':20s} {len(gap_all):>6} {len(gap_caught):>8} {len(gap_all)-len(gap_caught):>8} {100*len(gap_caught)/max(1,len(gap_all)):>7.1f}%")
    print(f"  {'Without gap':20s} {len(nogap_all):>6} {len(nogap_caught):>8} {len(nogap_all)-len(nogap_caught):>8} {100*len(nogap_caught)/max(1,len(nogap_all)):>7.1f}%")

    # ── 4. Spatial proximity ──────────────────────────────────────────────
    has_dist = [s for s in switches if s.bbox_distance > 0]
    if has_dist:
        print(f"\n--- SPATIAL PROXIMITY (bbox center distance at switch frame) ---")
        dist_bins = [(0, 50, "<50px"), (50, 100, "50-100px"),
                     (100, 200, "100-200px"), (200, 99999, "200+px")]
        print(f"  {'Distance':20s} {'All':>6} {'Caught':>8} {'Uncaught':>8} {'Catch%':>8}")
        for lo, hi, label in dist_bins:
            all_bin = [s for s in has_dist if lo <= s.bbox_distance < hi]
            caught_bin = [s for s in all_bin if s.caught]
            pct = 100 * len(caught_bin) / max(1, len(all_bin))
            print(f"  {label:20s} {len(all_bin):>6} {len(caught_bin):>8} {len(all_bin)-len(caught_bin):>8} {pct:>7.1f}%")

        dist_caught = [s.bbox_distance for s in has_dist if s.caught]
        dist_uncaught = [s.bbox_distance for s in has_dist if not s.caught]
        if dist_caught:
            print(f"\n  Mean distance (caught):   {np.mean(dist_caught):.1f}px")
        if dist_uncaught:
            print(f"  Mean distance (uncaught): {np.mean(dist_uncaught):.1f}px")

    # ── 5. Per-sequence breakdown ─────────────────────────────────────────
    print(f"\n--- PER-SEQUENCE BREAKDOWN (sorted by uncaught count) ---")
    seq_data: Dict[str, Dict] = defaultdict(lambda: {"total": 0, "caught": 0, "uncaught": 0,
                                                       "same_team_uncaught": 0, "cross_team_uncaught": 0})
    for s in switches:
        seq_data[s.sequence]["total"] += 1
        if s.caught:
            seq_data[s.sequence]["caught"] += 1
        else:
            seq_data[s.sequence]["uncaught"] += 1
            if s.same_team is True:
                seq_data[s.sequence]["same_team_uncaught"] += 1
            elif s.same_team is False:
                seq_data[s.sequence]["cross_team_uncaught"] += 1

    sorted_seqs = sorted(seq_data.items(), key=lambda x: x[1]["uncaught"], reverse=True)

    print(f"  {'Sequence':12s} {'Total':>6} {'Caught':>7} {'Uncaught':>9} {'Recall%':>8} {'SameTeam↓':>10} {'CrossTeam↓':>11}")
    for seq, d in sorted_seqs:
        recall = 100 * d["caught"] / max(1, d["total"])
        print(f"  {seq:12s} {d['total']:>6} {d['caught']:>7} {d['uncaught']:>9} {recall:>7.1f}% {d['same_team_uncaught']:>10} {d['cross_team_uncaught']:>11}")

    # ── Top 10 hardest sequences ──────────────────────────────────────────
    print(f"\n--- TOP 10 HARDEST SEQUENCES (most uncaught switches) ---")
    print("  These sequences should be manually inspected for common patterns")
    print("  (e.g. corner kicks, free kicks, crowded set pieces)")
    for seq, d in sorted_seqs[:10]:
        recall = 100 * d["caught"] / max(1, d["total"])
        print(f"  {seq}: {d['uncaught']} uncaught / {d['total']} total (recall {recall:.1f}%)")


def save_csv(switches: List[IdentitySwitch], path: Path) -> None:
    """Save all switches to CSV for further analysis."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "sequence", "baseline_track_id", "frame",
            "gt_id_before", "gt_id_after",
            "team_before", "team_after", "same_team",
            "duration_frames", "has_temporal_gap",
            "bbox_distance", "bbox_iou", "caught",
        ])
        for s in switches:
            w.writerow([
                s.sequence, s.baseline_track_id, s.frame,
                s.gt_id_before, s.gt_id_after,
                s.team_before or "", s.team_after or "",
                "" if s.same_team is None else int(s.same_team),
                s.duration_frames, int(s.has_temporal_gap),
                f"{s.bbox_distance:.2f}", f"{s.bbox_iou:.4f}",
                int(s.caught),
            ])
    print(f"\nDetailed CSV saved to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_gt_path(gt_root: Path, seq: str) -> Optional[Path]:
    for c in [gt_root / seq / "gt" / "gt.txt",
              gt_root / seq / "gt.txt",
              gt_root / f"{seq}.txt"]:
        if c.exists():
            return c
    return None


def main() -> None:
    # Load sequence list
    with SEQMAP.open("r") as f:
        sequences = [line.strip() for line in f
                     if line.strip() and not line.strip().lower().startswith("name")]

    print(f"Sequences: {len(sequences)}")
    print(f"GT root: {GT_ROOT}")
    print(f"Baseline dir: {BASELINE_DIR}")
    print(f"Splitter output: {SPLITTER_OUTPUT_DIR}")
    print(f"GSR root: {GSR_ROOT}")
    print(f"Tolerance: ±{TOLERANCE} frames")
    print()

    all_switches: List[IdentitySwitch] = []

    for seq in sequences:
        gt_path = resolve_gt_path(GT_ROOT, seq)
        baseline_path = BASELINE_DIR / f"{seq}.txt"
        splitter_path = SPLITTER_OUTPUT_DIR / f"{seq}.txt"

        if gt_path is None:
            print(f"[WARN] No GT for {seq}, skipping")
            continue
        if not baseline_path.exists():
            print(f"[WARN] No baseline for {seq}, skipping")
            continue
        if not splitter_path.exists():
            print(f"[WARN] No splitter output for {seq}, skipping")
            continue

        # Load MOT data
        baseline_tracks = load_mot_by_track(baseline_path)
        baseline_by_frame = load_mot_by_frame(baseline_path)
        gt_by_frame = load_mot_by_frame(gt_path)
        splitter_by_frame = load_mot_by_frame(splitter_path)

        # Build GT assignment
        det_gt_map = build_det_to_gt_with_coords(baseline_by_frame, gt_by_frame, IOU_THRESH)

        # Find all identity switches
        switches = find_identity_switches_detailed(
            baseline_tracks, gt_by_frame, det_gt_map, seq
        )

        # Find predicted splits and mark caught switches
        predicted_splits = find_predicted_splits(
            baseline_tracks, splitter_by_frame, IOU_THRESH
        )
        mark_caught_switches(switches, predicted_splits, TOLERANCE)

        # Load GSR team labels
        gsr_seq = snpt_to_sngs(seq)
        gsr_json = GSR_ROOT / gsr_seq / "Labels-GameState.json"
        team_labels: Dict[int, str] = {}
        if gsr_json.exists():
            team_labels = load_gsr_team_labels(gsr_json)

        # Annotate team info
        for sw in switches:
            sw.team_before = team_labels.get(sw.gt_id_before)
            sw.team_after = team_labels.get(sw.gt_id_after)
            if sw.team_before is not None and sw.team_after is not None:
                sw.same_team = (sw.team_before == sw.team_after)

        all_switches.extend(switches)
        n_caught = sum(1 for s in switches if s.caught)
        print(f"  {seq}: {len(switches)} switches, {n_caught} caught, "
              f"{len(predicted_splits)} predicted splits")

    # Print full analysis
    print_analysis(all_switches)

    # Save detailed CSV
    csv_path = OUTPUT_DIR / "splitter_qualitative_analysis.csv"
    save_csv(all_switches, csv_path)

    # Also save a summary CSV for the per-sequence table
    summary_path = OUTPUT_DIR / "splitter_per_sequence_summary.csv"
    seq_stats: Dict[str, Dict] = defaultdict(lambda: {"total": 0, "caught": 0,
                                                        "same_team": 0, "cross_team": 0,
                                                        "same_team_caught": 0, "cross_team_caught": 0,
                                                        "mean_duration": []})
    for s in all_switches:
        d = seq_stats[s.sequence]
        d["total"] += 1
        d["mean_duration"].append(s.duration_frames)
        if s.caught:
            d["caught"] += 1
        if s.same_team is True:
            d["same_team"] += 1
            if s.caught:
                d["same_team_caught"] += 1
        elif s.same_team is False:
            d["cross_team"] += 1
            if s.caught:
                d["cross_team_caught"] += 1

    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "total_switches", "caught", "uncaught", "recall",
                     "same_team", "cross_team", "same_team_caught", "cross_team_caught",
                     "mean_switch_duration"])
        for seq in sequences:
            d = seq_stats.get(seq, {"total": 0, "caught": 0, "same_team": 0,
                                     "cross_team": 0, "same_team_caught": 0,
                                     "cross_team_caught": 0, "mean_duration": []})
            uncaught = d["total"] - d["caught"]
            recall = d["caught"] / d["total"] if d["total"] > 0 else 0
            mean_dur = np.mean(d["mean_duration"]) if d["mean_duration"] else 0
            w.writerow([seq, d["total"], d["caught"], uncaught, f"{recall:.4f}",
                         d["same_team"], d["cross_team"],
                         d["same_team_caught"], d["cross_team_caught"],
                         f"{mean_dur:.1f}"])

    print(f"Per-sequence summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
