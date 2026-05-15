"""
jersey_funnel_analysis.py — Jersey number prediction funnel at identity switches.

For each of the 1048 ground truth identity switches, this script determines:
  1. VISIBILITY: Does the GSR ground truth contain jersey annotations for both
     players involved in the switch? (Are jersey numbers observable at all?)
  2. DETECTION: Did the OCR pipeline produce a confident jersey prediction on
     both sides of the switch? (Did we read a number?)
  3. SIGNAL: Do the two detected jersey numbers differ, giving the splitter
     a signal to act on? (Is there a detectable change?)
  4. ACTION: Did the splitter actually fire within ±tolerance frames?

This separates four failure modes: jersey never visible, OCR failed to read
a visible jersey, both jerseys matched (same number), or signal was available
but the persistence filter suppressed it.

Additionally, for the splits the jersey splitter DID predict, this script
computes the frame delay between the predicted split and the GT switch.

Requirements:
  - Attribute cache:     cache/cache_attributes_SNPT-XXX.pkl
  - Ground truth MOT:    DATA_ROOT / SNPT-XXX / gt / gt.txt
  - Baseline tracker:    evaluation/SNPT/baseline/data/SNPT-XXX.txt
  - Splitter output:     (jersey only splitter output)
  - GSR JSON:            GSR_ROOT / SNGS-XXX / Labels-GameState.json

Usage:
    python experiments/attributes/jersey_funnel_analysis.py
"""
from __future__ import annotations

import csv
import json
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths — edit these to match your setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import settings

GT_ROOT = settings.DATA_ROOT
BASELINE_DIR = REPO_ROOT / "evaluation" / "SNPT" / "baseline" / "data"
SEQMAP = REPO_ROOT / "evaluation" / "seqmaps" / "SNPT-test.txt"

GSR_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\data\SoccerNetGS\test")

# Cache directory where predict_attributes() stores its pickles
CACHE_DIR = REPO_ROOT / "output" / "cache"

# Jersey only splitter output (if available; set to None to skip split matching)
# If you only have the combined splitter output, set this to that path instead.
JERSEY_SPLITTER_DIR = REPO_ROOT / "evaluation" / "SNPT" / "splitters" / "test" / "wo_reid_splitter" / "splitter_jersey" / "data"

# Combined splitter output (STR + Jersey + Team + Bbox)
# COMBINED_SPLITTER_DIR = REPO_ROOT / "evaluation" / "SNPT" / "splitters" / "test" / "wo_reid_splitter" / "splitter_temporalreid_jersey_team_bbox" / "data"

TOLERANCE = 750
IOU_THRESH = 0.5

# Jersey prediction parameters (from settings.py SPLITTER config)
JERSEY_ENTROPY_THRESHOLD = 0.01   # max entropy to count as "confident"
LOOKAHEAD_FRAMES = 100            # frames to look ahead for persistent predictions

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# MOT file IO (shared with splitter_qualitative_analysis.py)
# ---------------------------------------------------------------------------

def load_mot_by_track(path: Path) -> Dict[int, List[Tuple[int, np.ndarray]]]:
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


def build_det_to_gt(
    det_by_frame: Dict[int, List[Tuple[int, np.ndarray]]],
    gt_by_frame: Dict[int, List[Tuple[int, np.ndarray]]],
    iou_threshold: float,
) -> Dict[Tuple[int, int, float, float], int]:
    out = {}
    for frame, dets in det_by_frame.items():
        gts = gt_by_frame.get(frame, [])
        for tid, box in dets:
            key = (frame, tid, float(box[0]), float(box[1]))
            out[key] = best_match(box, gts, iou_threshold)
    return out


# ---------------------------------------------------------------------------
# GSR jersey number extraction
# ---------------------------------------------------------------------------

def load_gsr_jersey_and_team(gsr_json_path: Path) -> Tuple[Dict[int, int], Dict[int, str]]:
    """
    Read Labels-GameState.json and return:
      - {gt_track_id: jersey_number}  (majority vote across frames)
      - {gt_track_id: team_label}     (majority vote across frames)

    Jersey numbers that are None/null in the annotations are skipped.
    """
    with open(gsr_json_path, "r") as f:
        data = json.load(f)

    track_jerseys: Dict[int, List[int]] = defaultdict(list)
    track_teams: Dict[int, List[str]] = defaultdict(list)

    for ann in data.get("annotations", []):
        cat_id = ann.get("category_id", 0)
        if cat_id not in {1, 2}:
            continue
        track_id = ann.get("track_id")
        attrs = ann.get("attributes", {})

        team = attrs.get("team", None)
        jersey = attrs.get("jersey", None)

        if track_id is not None:
            if team is not None:
                track_teams[track_id].append(str(team))
            if jersey is not None:
                try:
                    track_jerseys[track_id].append(int(jersey))
                except (ValueError, TypeError):
                    pass

    jersey_result = {}
    for tid, jerseys in track_jerseys.items():
        jersey_result[tid] = Counter(jerseys).most_common(1)[0][0]

    team_result = {}
    for tid, teams in track_teams.items():
        team_result[tid] = Counter(teams).most_common(1)[0][0]

    return jersey_result, team_result


def snpt_to_sngs(seq_name: str) -> str:
    return seq_name.replace("SNPT-", "SNGS-")


# ---------------------------------------------------------------------------
# Load attribute cache (predicted jersey numbers per frame)
# ---------------------------------------------------------------------------

def load_attribute_cache(cache_dir: Path, seq: str) -> Optional[Dict]:
    """
    Load the pickled tracklet dict from the attribute prediction cache.
    Returns {track_id: Tracklet} or None if cache not found.
    """
    cache_path = cache_dir / f"cache_attributes_{seq}.pkl"
    if not cache_path.exists():
        return None
    with open(cache_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Identity switch detection (same logic as splitter_qualitative_analysis.py)
# ---------------------------------------------------------------------------

@dataclass
class JerseySwitchInfo:
    sequence: str
    baseline_track_id: int
    frame: int
    gt_id_before: int
    gt_id_after: int
    # GSR ground truth
    gt_jersey_before: Optional[int] = None   # GT jersey of player before switch
    gt_jersey_after: Optional[int] = None    # GT jersey of player after switch
    gt_jerseys_differ: Optional[bool] = None # Do GT jerseys differ?
    gt_team_before: Optional[str] = None
    gt_team_after: Optional[str] = None
    same_team: Optional[bool] = None
    # Predicted attributes
    pred_jersey_before: Optional[int] = None  # Predicted jersey in window before switch
    pred_jersey_after: Optional[int] = None   # Predicted jersey in window after switch
    pred_conf_before: float = 0.0             # Number of confident predictions before
    pred_conf_after: float = 0.0              # Number of confident predictions after
    pred_jerseys_differ: Optional[bool] = None
    # Funnel stages
    gt_visible: bool = False                  # Stage 1: GT has jersey for both players
    pred_detected: bool = False               # Stage 2: OCR found confident jersey both sides
    signal_available: bool = False            # Stage 3: Predicted jerseys differ
    splitter_fired: bool = False              # Stage 4: Splitter actually split here
    # Extra info
    duration_frames: int = 0
    has_temporal_gap: bool = False
    split_delay: Optional[int] = None         # Frame delay of predicted split (if caught)


def find_switches_with_jersey_info(
    baseline_tracks: Dict[int, List[Tuple[int, np.ndarray]]],
    gt_by_frame: Dict[int, List[Tuple[int, np.ndarray]]],
    det_gt_map: Dict[Tuple[int, int, float, float], int],
    cached_tracklets: Optional[Dict],
    gt_jerseys: Dict[int, int],
    gt_teams: Dict[int, str],
    sequence: str,
) -> List[JerseySwitchInfo]:
    """Find all identity switches and annotate with jersey information."""
    switches = []

    for tid, frames_boxes in baseline_tracks.items():
        prev_gt = None
        prev_frame = None

        # Build a mapping from frame -> index in the tracklet for fast lookup
        frame_to_idx = {f: i for i, (f, _) in enumerate(frames_boxes)}

        for i, (frame, box) in enumerate(frames_boxes):
            key = (frame, tid, float(box[0]), float(box[1]))
            gt_id = det_gt_map.get(key, -1)
            if gt_id == -1:
                continue

            if prev_gt is not None and gt_id != prev_gt:
                sw = JerseySwitchInfo(
                    sequence=sequence,
                    baseline_track_id=tid,
                    frame=frame,
                    gt_id_before=prev_gt,
                    gt_id_after=gt_id,
                )

                # Duration
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

                # Temporal gap
                if prev_frame is not None:
                    sw.has_temporal_gap = (frame - prev_frame) > 5

                # --- Stage 1: GT jersey visibility ---
                sw.gt_jersey_before = gt_jerseys.get(prev_gt)
                sw.gt_jersey_after = gt_jerseys.get(gt_id)
                sw.gt_team_before = gt_teams.get(prev_gt)
                sw.gt_team_after = gt_teams.get(gt_id)

                if sw.gt_team_before is not None and sw.gt_team_after is not None:
                    sw.same_team = (sw.gt_team_before == sw.gt_team_after)

                if sw.gt_jersey_before is not None and sw.gt_jersey_after is not None:
                    sw.gt_visible = True
                    sw.gt_jerseys_differ = (sw.gt_jersey_before != sw.gt_jersey_after)

                # --- Stage 2 & 3: Predicted jersey detection ---
                if cached_tracklets is not None and tid in cached_tracklets:
                    tracklet = cached_tracklets[tid]
                    jerseys = tracklet.pred_attributes.get('jerseys', [])
                    entropies = tracklet.pred_attributes.get('jersey_entropies',
                                tracklet.pred_attributes.get('jersey_confs', []))

                    if jerseys and len(jerseys) == len(frames_boxes):
                        # Find dominant confident jersey BEFORE the switch
                        # Look in a window before the switch frame
                        before_jerseys = _get_confident_jerseys(
                            jerseys, entropies, 0, i, JERSEY_ENTROPY_THRESHOLD
                        )
                        # Find dominant confident jersey AFTER the switch
                        after_jerseys = _get_confident_jerseys(
                            jerseys, entropies, i, min(i + LOOKAHEAD_FRAMES, len(jerseys)),
                            JERSEY_ENTROPY_THRESHOLD
                        )

                        if before_jerseys:
                            sw.pred_jersey_before = Counter(before_jerseys).most_common(1)[0][0]
                            sw.pred_conf_before = len(before_jerseys)
                        if after_jerseys:
                            sw.pred_jersey_after = Counter(after_jerseys).most_common(1)[0][0]
                            sw.pred_conf_after = len(after_jerseys)

                        if sw.pred_jersey_before is not None and sw.pred_jersey_after is not None:
                            sw.pred_detected = True
                            sw.pred_jerseys_differ = (sw.pred_jersey_before != sw.pred_jersey_after)
                            if sw.pred_jerseys_differ:
                                sw.signal_available = True

                switches.append(sw)

            if gt_id != -1:
                prev_gt = gt_id
                prev_frame = frame

    return switches


def _get_confident_jerseys(
    jerseys: list, entropies: list, start: int, end: int,
    entropy_threshold: float
) -> List[int]:
    """Extract all confident (low entropy) jersey predictions in [start, end)."""
    result = []
    for idx in range(start, end):
        if idx >= len(jerseys) or idx >= len(entropies):
            break
        j = jerseys[idx]
        e = entropies[idx]
        # Skip invalid predictions
        if j is None or (isinstance(j, float) and np.isnan(j)):
            continue
        if j == -1:
            continue
        # Check confidence
        if e is not None and not (isinstance(e, float) and np.isnan(e)):
            if e <= entropy_threshold:
                result.append(int(j))
    return result


# ---------------------------------------------------------------------------
# Split matching (find predicted splits and compute delays)
# ---------------------------------------------------------------------------

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


def match_splits_to_switches(
    switches: List[JerseySwitchInfo],
    predicted_splits: List[Tuple[int, int]],
    tolerance: int,
) -> None:
    """Mark Stage 4 (splitter_fired) and compute split delay."""
    pred_by_tid: Dict[int, List[int]] = defaultdict(list)
    for tid, frame in predicted_splits:
        pred_by_tid[tid].append(frame)

    switch_by_tid: Dict[int, List[JerseySwitchInfo]] = defaultdict(list)
    for sw in switches:
        switch_by_tid[sw.baseline_track_id].append(sw)

    for tid in set(list(pred_by_tid.keys()) + list(switch_by_tid.keys())):
        sw_list = sorted(switch_by_tid.get(tid, []), key=lambda s: s.frame)
        p_list = sorted(pred_by_tid.get(tid, []))
        used = [False] * len(p_list)

        for sw in sw_list:
            best_idx = -1
            best_d = tolerance + 1
            for pi, pf in enumerate(p_list):
                if used[pi]:
                    continue
                d = abs(pf - sw.frame)
                if d <= tolerance and d < best_d:
                    best_d = d
                    best_idx = pi
            if best_idx >= 0:
                used[best_idx] = True
                sw.splitter_fired = True
                sw.split_delay = p_list[best_idx] - sw.frame

    # Also compute delay for splits matched with unlimited tolerance
    # to show the jersey delay distribution for ALL jersey splits
    for tid in set(list(pred_by_tid.keys()) + list(switch_by_tid.keys())):
        sw_list = sorted(switch_by_tid.get(tid, []), key=lambda s: s.frame)
        p_list = sorted(pred_by_tid.get(tid, []))
        used = [False] * len(p_list)

        for sw in sw_list:
            if sw.split_delay is not None:
                continue  # Already matched within tolerance
            best_idx = -1
            best_d = 99999
            for pi, pf in enumerate(p_list):
                if used[pi]:
                    continue
                d = abs(pf - sw.frame)
                if d < best_d:
                    best_d = d
                    best_idx = pi
            if best_idx >= 0:
                used[best_idx] = True
                # Store delay but don't mark as fired (outside tolerance)
                sw.split_delay = p_list[best_idx] - sw.frame


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_funnel(switches: List[JerseySwitchInfo]) -> None:
    total = len(switches)

    print(f"\n{'='*70}")
    print(f"JERSEY NUMBER FUNNEL ANALYSIS")
    print(f"{'='*70}")
    print(f"Total identity switches: {total}")

    # Stage 1: GT visibility
    gt_visible = [s for s in switches if s.gt_visible]
    gt_differ = [s for s in gt_visible if s.gt_jerseys_differ]
    gt_same = [s for s in gt_visible if not s.gt_jerseys_differ]

    print(f"\n--- STAGE 1: GT Jersey Visibility ---")
    print(f"  Both players have GT jersey number: {len(gt_visible)} / {total} ({100*len(gt_visible)/total:.1f}%)")
    print(f"    GT jerseys differ: {len(gt_differ)} ({100*len(gt_differ)/max(1,len(gt_visible)):.1f}%)")
    print(f"    GT jerseys same:   {len(gt_same)} ({100*len(gt_same)/max(1,len(gt_visible)):.1f}%)")
    print(f"  At least one GT jersey missing: {total - len(gt_visible)} ({100*(total-len(gt_visible))/total:.1f}%)")

    # Stage 2: Predicted detection
    pred_detected = [s for s in switches if s.pred_detected]
    pred_detected_and_visible = [s for s in gt_differ if s.pred_detected]

    print(f"\n--- STAGE 2: OCR Detection ---")
    print(f"  Confident prediction on both sides: {len(pred_detected)} / {total} ({100*len(pred_detected)/total:.1f}%)")
    print(f"  Among switches with differing GT jerseys:")
    print(f"    OCR detected both: {len(pred_detected_and_visible)} / {len(gt_differ)} ({100*len(pred_detected_and_visible)/max(1,len(gt_differ)):.1f}%)")

    # Stage 3: Signal available
    signal_available = [s for s in switches if s.signal_available]

    print(f"\n--- STAGE 3: Signal Available (predicted jerseys differ) ---")
    print(f"  Predicted jerseys differ: {len(signal_available)} / {total} ({100*len(signal_available)/total:.1f}%)")
    print(f"  Among OCR detected switches: {len(signal_available)} / {len(pred_detected)} ({100*len(signal_available)/max(1,len(pred_detected)):.1f}%)")

    # Check: how many switches have signal but GT says same jersey?
    signal_but_gt_same = [s for s in signal_available if s.gt_jerseys_differ is False]
    signal_and_gt_differ = [s for s in signal_available if s.gt_jerseys_differ is True]
    print(f"    Signal + GT jerseys differ: {len(signal_and_gt_differ)}")
    print(f"    Signal but GT jerseys same (OCR error): {len(signal_but_gt_same)}")

    # Stage 4: Splitter fired
    splitter_fired = [s for s in switches if s.splitter_fired]

    print(f"\n--- STAGE 4: Splitter Fired (within ±{TOLERANCE} frames) ---")
    print(f"  Splitter caught: {len(splitter_fired)} / {total} ({100*len(splitter_fired)/total:.1f}%)")
    print(f"  Among signal available: {len(splitter_fired)} / {len(signal_available)} ({100*len(splitter_fired)/max(1,len(signal_available)):.1f}%)")

    # Funnel summary
    print(f"\n--- FUNNEL SUMMARY ---")
    print(f"  {'Stage':40s} {'Count':>6} {'% of total':>10} {'% of prev':>10}")
    print(f"  {'Total identity switches':40s} {total:>6} {'100.0%':>10} {'':>10}")
    print(f"  {'GT jerseys differ (opportunity)':40s} {len(gt_differ):>6} {100*len(gt_differ)/total:>9.1f}% {'':>10}")
    print(f"  {'OCR detected both sides':40s} {len(pred_detected_and_visible):>6} {100*len(pred_detected_and_visible)/total:>9.1f}% {100*len(pred_detected_and_visible)/max(1,len(gt_differ)):>9.1f}%")
    signal_in_gt_differ = [s for s in signal_and_gt_differ]
    print(f"  {'Predicted jerseys differ':40s} {len(signal_in_gt_differ):>6} {100*len(signal_in_gt_differ)/total:>9.1f}% {100*len(signal_in_gt_differ)/max(1,len(pred_detected_and_visible)):>9.1f}%")
    fired_in_gt_differ = [s for s in splitter_fired if s.gt_jerseys_differ is True or s.gt_jerseys_differ is None]
    print(f"  {'Splitter fired (±{0} frames)'.format(TOLERANCE):40s} {len(splitter_fired):>6} {100*len(splitter_fired)/total:>9.1f}% {100*len(splitter_fired)/max(1,len(signal_in_gt_differ)):>9.1f}%")

    # Split delay analysis
    print(f"\n--- JERSEY SPLIT DELAY ANALYSIS ---")
    delays = [(s.split_delay, s.splitter_fired) for s in switches
              if s.split_delay is not None]
    if delays:
        print(f"  Total predicted jersey splits matched to GT switches: {len(delays)}")
        for delay, within_tol in sorted(delays, key=lambda x: abs(x[0])):
            status = "TP (within tolerance)" if within_tol else "FP (outside tolerance)"
            print(f"    Delay: {delay:+4d} frames ({delay/25:.2f}s) — {status}")


def save_csv(switches: List[JerseySwitchInfo], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "sequence", "baseline_track_id", "frame",
            "gt_id_before", "gt_id_after",
            "gt_jersey_before", "gt_jersey_after", "gt_jerseys_differ",
            "gt_team_before", "gt_team_after", "same_team",
            "pred_jersey_before", "pred_jersey_after",
            "pred_conf_before", "pred_conf_after", "pred_jerseys_differ",
            "gt_visible", "pred_detected", "signal_available", "splitter_fired",
            "duration_frames", "has_temporal_gap", "split_delay",
        ])
        for s in switches:
            w.writerow([
                s.sequence, s.baseline_track_id, s.frame,
                s.gt_id_before, s.gt_id_after,
                s.gt_jersey_before if s.gt_jersey_before is not None else "",
                s.gt_jersey_after if s.gt_jersey_after is not None else "",
                "" if s.gt_jerseys_differ is None else int(s.gt_jerseys_differ),
                s.gt_team_before or "", s.gt_team_after or "",
                "" if s.same_team is None else int(s.same_team),
                s.pred_jersey_before if s.pred_jersey_before is not None else "",
                s.pred_jersey_after if s.pred_jersey_after is not None else "",
                f"{s.pred_conf_before:.0f}", f"{s.pred_conf_after:.0f}",
                "" if s.pred_jerseys_differ is None else int(s.pred_jerseys_differ),
                int(s.gt_visible), int(s.pred_detected),
                int(s.signal_available), int(s.splitter_fired),
                s.duration_frames, int(s.has_temporal_gap),
                s.split_delay if s.split_delay is not None else "",
            ])
    print(f"\nCSV saved to: {path}")


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
    with SEQMAP.open("r") as f:
        sequences = [line.strip() for line in f
                     if line.strip() and not line.strip().lower().startswith("name")]

    print(f"Sequences: {len(sequences)}")
    print(f"GT root: {GT_ROOT}")
    print(f"Baseline dir: {BASELINE_DIR}")
    print(f"Cache dir: {CACHE_DIR}")
    print(f"Jersey splitter: {JERSEY_SPLITTER_DIR}")
    print(f"GSR root: {GSR_ROOT}")
    print(f"Tolerance: ±{TOLERANCE} frames")
    print()

    all_switches: List[JerseySwitchInfo] = []

    for seq in sequences:
        gt_path = resolve_gt_path(GT_ROOT, seq)
        baseline_path = BASELINE_DIR / f"{seq}.txt"

        if gt_path is None:
            print(f"[WARN] No GT for {seq}, skipping")
            continue
        if not baseline_path.exists():
            print(f"[WARN] No baseline for {seq}, skipping")
            continue

        # Load MOT data
        baseline_tracks = load_mot_by_track(baseline_path)
        baseline_by_frame = load_mot_by_frame(baseline_path)
        gt_by_frame = load_mot_by_frame(gt_path)

        # Build GT assignment
        det_gt_map = build_det_to_gt(baseline_by_frame, gt_by_frame, IOU_THRESH)

        # Load GSR annotations
        gsr_seq = snpt_to_sngs(seq)
        gsr_json = GSR_ROOT / gsr_seq / "Labels-GameState.json"
        gt_jerseys: Dict[int, int] = {}
        gt_teams: Dict[int, str] = {}
        if gsr_json.exists():
            gt_jerseys, gt_teams = load_gsr_jersey_and_team(gsr_json)

        # Load attribute cache (predicted jerseys)
        cached_tracklets = load_attribute_cache(CACHE_DIR, seq)
        if cached_tracklets is None:
            print(f"[WARN] No attribute cache for {seq}, predictions unavailable")

        # Find switches with jersey info
        switches = find_switches_with_jersey_info(
            baseline_tracks, gt_by_frame, det_gt_map,
            cached_tracklets, gt_jerseys, gt_teams, seq
        )

        # Match against jersey splitter output (if available)
        jersey_splitter_path = JERSEY_SPLITTER_DIR / f"{seq}.txt" if JERSEY_SPLITTER_DIR else None
        if jersey_splitter_path and jersey_splitter_path.exists():
            splitter_by_frame = load_mot_by_frame(jersey_splitter_path)
            predicted_splits = find_predicted_splits(
                baseline_tracks, splitter_by_frame, IOU_THRESH
            )
            match_splits_to_switches(switches, predicted_splits, TOLERANCE)

        all_switches.extend(switches)
        n_visible = sum(1 for s in switches if s.gt_visible)
        n_detected = sum(1 for s in switches if s.pred_detected)
        n_signal = sum(1 for s in switches if s.signal_available)
        n_fired = sum(1 for s in switches if s.splitter_fired)
        print(f"  {seq}: {len(switches)} switches | "
              f"visible={n_visible} detected={n_detected} signal={n_signal} fired={n_fired}")

    print_funnel(all_switches)
    save_csv(all_switches, OUTPUT_DIR / "jersey_funnel_analysis.csv")


if __name__ == "__main__":
    main()
