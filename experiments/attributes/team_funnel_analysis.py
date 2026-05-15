"""
team_funnel_analysis.py — Team identity prediction funnel at identity switches.

For each of the 1048 ground truth identity switches, this script determines:
  1. OPPORTUNITY: Is this a cross team switch? (Same team switches are invisible
     to the team splitter by design.)
  2. DETECTION: Did the team classifier produce a confident prediction on both
     sides of the switch? (Did we classify team identity?)
  3. SIGNAL: Do the two team predictions differ, giving the splitter a signal
     to act on? (Did the classifier get it right?)
  4. ACTION: Did the team splitter actually fire within ±tolerance frames?

This separates four failure modes: same team switch (no opportunity), classifier
had no confident prediction, classifier predicted the same team for both sides
(wrong prediction), or signal was available but the persistence filter
suppressed it.

Additionally, this script computes team prediction accuracy on frames near
identity switches: how often does the team classifier agree with the GSR
ground truth team label?

Requirements:
  - Attribute cache:     cache/cache_attributes_SNPT-XXX.pkl
  - Ground truth MOT:    DATA_ROOT / SNPT-XXX / gt / gt.txt
  - Baseline tracker:    evaluation/SNPT/baseline/data/SNPT-XXX.txt
  - Team splitter output: (team only splitter output)
  - GSR JSON:            GSR_ROOT / SNGS-XXX / Labels-GameState.json

Usage:
    python experiments/attributes/team_funnel_analysis.py
"""
from __future__ import annotations

import csv
import json
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import settings

GT_ROOT = settings.DATA_ROOT
BASELINE_DIR = REPO_ROOT / "evaluation" / "SNPT" / "baseline" / "data"
SEQMAP = REPO_ROOT / "evaluation" / "seqmaps" / "SNPT-test.txt"

GSR_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\data\SoccerNetGS\test")

CACHE_DIR = REPO_ROOT / "output" / "cache"

# Team only splitter output (if available)
TEAM_SPLITTER_DIR = REPO_ROOT / "evaluation" / "SNPT" / "splitters" / "test" / "wo_reid_splitter" / "splitter_team" / "data"

# Combined splitter output (STR + Jersey + Team + Bbox)
# COMBINED_SPLITTER_DIR = REPO_ROOT / "evaluation" / "SNPT" / "splitters" / "test" / "wo_reid_splitter" / "splitter_temporalreid_jersey_team_bbox" / "data"

TOLERANCE = 750
IOU_THRESH = 0.5

# Team prediction parameters (from settings.py SPLITTER config)
TEAM_CONFIDENCE_THRESHOLD = 0.6
TEAM_LOOKAHEAD = 100
TEAM_MIN_PERSISTENCE = 5
TEAM_MIN_PERSISTENCE_RATIO = 0.8

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# MOT file IO (shared across analysis scripts)
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
# GSR team label extraction
# ---------------------------------------------------------------------------

def load_gsr_team_labels(gsr_json_path: Path) -> Dict[int, str]:
    with open(gsr_json_path, "r") as f:
        data = json.load(f)

    track_teams: Dict[int, List[str]] = defaultdict(list)
    for ann in data.get("annotations", []):
        cat_id = ann.get("category_id", 0)
        if cat_id not in {1, 2}:
            continue
        track_id = ann.get("track_id")
        attrs = ann.get("attributes", {})
        team = attrs.get("team", None)
        if track_id is not None and team is not None:
            track_teams[track_id].append(str(team))

    result = {}
    for tid, teams in track_teams.items():
        result[tid] = Counter(teams).most_common(1)[0][0]
    return result


def snpt_to_sngs(seq_name: str) -> str:
    return seq_name.replace("SNPT-", "SNGS-")


# ---------------------------------------------------------------------------
# Load attribute cache
# ---------------------------------------------------------------------------

def load_attribute_cache(cache_dir: Path, seq: str) -> Optional[Dict]:
    cache_path = cache_dir / f"cache_attributes_{seq}.pkl"
    if not cache_path.exists():
        return None
    with open(cache_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TeamSwitchInfo:
    sequence: str
    baseline_track_id: int
    frame: int
    gt_id_before: int
    gt_id_after: int
    # GSR ground truth
    gt_team_before: Optional[str] = None
    gt_team_after: Optional[str] = None
    same_team: Optional[bool] = None
    is_cross_team: bool = False           # Stage 1: opportunity
    # Predicted attributes
    pred_team_before: Optional[int] = None   # Dominant predicted team before switch
    pred_team_after: Optional[int] = None    # Dominant predicted team after switch
    pred_conf_before: int = 0                # Number of confident predictions before
    pred_conf_after: int = 0                 # Number of confident predictions after
    pred_correct_before: Optional[bool] = None  # Does prediction match GT before?
    pred_correct_after: Optional[bool] = None   # Does prediction match GT after?
    pred_teams_differ: Optional[bool] = None
    # Funnel stages
    pred_detected: bool = False           # Stage 2: confident prediction both sides
    signal_available: bool = False        # Stage 3: predicted teams differ
    splitter_fired: bool = False          # Stage 4: splitter split here
    # Extra
    duration_frames: int = 0
    has_temporal_gap: bool = False


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def find_switches_with_team_info(
    baseline_tracks: Dict[int, List[Tuple[int, np.ndarray]]],
    gt_by_frame: Dict[int, List[Tuple[int, np.ndarray]]],
    det_gt_map: Dict[Tuple[int, int, float, float], int],
    cached_tracklets: Optional[Dict],
    gt_teams: Dict[int, str],
    sequence: str,
    # We need to map GT team labels ("left"/"right") to predicted team labels (0/1)
    # This mapping is determined per sequence by majority vote
    team_label_map: Optional[Dict[str, int]] = None,
) -> List[TeamSwitchInfo]:
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
                sw = TeamSwitchInfo(
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

                if prev_frame is not None:
                    sw.has_temporal_gap = (frame - prev_frame) > 5

                # GT team info
                sw.gt_team_before = gt_teams.get(prev_gt)
                sw.gt_team_after = gt_teams.get(gt_id)
                if sw.gt_team_before is not None and sw.gt_team_after is not None:
                    sw.same_team = (sw.gt_team_before == sw.gt_team_after)
                    sw.is_cross_team = not sw.same_team

                # Predicted team info
                if cached_tracklets is not None and tid in cached_tracklets:
                    tracklet = cached_tracklets[tid]
                    teams = tracklet.pred_attributes.get('teams', [])
                    confs = tracklet.pred_attributes.get('team_confs', [])

                    if teams and len(teams) == len(frames_boxes):
                        # Get confident team predictions before the switch
                        before_teams = _get_confident_teams(
                            teams, confs, max(0, i - TEAM_LOOKAHEAD), i,
                            TEAM_CONFIDENCE_THRESHOLD
                        )
                        # Get confident team predictions after the switch
                        after_teams = _get_confident_teams(
                            teams, confs, i, min(i + TEAM_LOOKAHEAD, len(teams)),
                            TEAM_CONFIDENCE_THRESHOLD
                        )

                        if before_teams:
                            sw.pred_team_before = Counter(before_teams).most_common(1)[0][0]
                            sw.pred_conf_before = len(before_teams)
                        if after_teams:
                            sw.pred_team_after = Counter(after_teams).most_common(1)[0][0]
                            sw.pred_conf_after = len(after_teams)

                        if sw.pred_team_before is not None and sw.pred_team_after is not None:
                            sw.pred_detected = True
                            sw.pred_teams_differ = (sw.pred_team_before != sw.pred_team_after)
                            if sw.pred_teams_differ:
                                sw.signal_available = True

                        # Check prediction correctness against GT
                        if team_label_map and sw.gt_team_before is not None:
                            expected_before = team_label_map.get(sw.gt_team_before)
                            if expected_before is not None and sw.pred_team_before is not None:
                                sw.pred_correct_before = (sw.pred_team_before == expected_before)
                        if team_label_map and sw.gt_team_after is not None:
                            expected_after = team_label_map.get(sw.gt_team_after)
                            if expected_after is not None and sw.pred_team_after is not None:
                                sw.pred_correct_after = (sw.pred_team_after == expected_after)

                switches.append(sw)

            if gt_id != -1:
                prev_gt = gt_id
                prev_frame = frame

    return switches


def _get_confident_teams(
    teams: list, confs: list, start: int, end: int,
    conf_threshold: float
) -> List[int]:
    result = []
    for idx in range(start, end):
        if idx >= len(teams) or idx >= len(confs):
            break
        t = teams[idx]
        c = confs[idx]
        if t is None or (isinstance(t, float) and np.isnan(t)):
            continue
        if c is not None and not (isinstance(c, float) and np.isnan(c)):
            if c >= conf_threshold:
                result.append(int(t))
    return result


def build_team_label_map(
    cached_tracklets: Dict,
    baseline_tracks: Dict[int, List[Tuple[int, np.ndarray]]],
    det_gt_map: Dict[Tuple[int, int, float, float], int],
    gt_teams: Dict[int, str],
) -> Dict[str, int]:
    """
    Build a mapping from GT team label ("left"/"right") to predicted team label (0/1).
    Uses majority vote across all frames where both GT and prediction are available.
    """
    votes: Dict[str, List[int]] = defaultdict(list)

    for tid, frames_boxes in baseline_tracks.items():
        if tid not in cached_tracklets:
            continue
        tracklet = cached_tracklets[tid]
        teams = tracklet.pred_attributes.get('teams', [])
        confs = tracklet.pred_attributes.get('team_confs', [])

        if not teams or len(teams) != len(frames_boxes):
            continue

        for i, (frame, box) in enumerate(frames_boxes):
            key = (frame, tid, float(box[0]), float(box[1]))
            gt_id = det_gt_map.get(key, -1)
            if gt_id == -1:
                continue

            gt_team = gt_teams.get(gt_id)
            if gt_team is None:
                continue

            pred_team = teams[i]
            pred_conf = confs[i] if i < len(confs) else None

            if pred_team is None or (isinstance(pred_team, float) and np.isnan(pred_team)):
                continue
            if pred_conf is not None and not (isinstance(pred_conf, float) and np.isnan(pred_conf)):
                if pred_conf >= TEAM_CONFIDENCE_THRESHOLD:
                    votes[gt_team].append(int(pred_team))

    result = {}
    for gt_label, pred_labels in votes.items():
        if pred_labels:
            result[gt_label] = Counter(pred_labels).most_common(1)[0][0]

    return result


# ---------------------------------------------------------------------------
# Split matching
# ---------------------------------------------------------------------------

def find_predicted_splits(
    baseline_tracks: Dict[int, List[Tuple[int, np.ndarray]]],
    splitter_by_frame: Dict[int, List[Tuple[int, np.ndarray]]],
    iou_threshold: float,
) -> List[Tuple[int, int]]:
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
    switches: List[TeamSwitchInfo],
    predicted_splits: List[Tuple[int, int]],
    tolerance: int,
) -> None:
    pred_by_tid: Dict[int, List[int]] = defaultdict(list)
    for tid, frame in predicted_splits:
        pred_by_tid[tid].append(frame)

    switch_by_tid: Dict[int, List[TeamSwitchInfo]] = defaultdict(list)
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


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_funnel(switches: List[TeamSwitchInfo]) -> None:
    total = len(switches)

    print(f"\n{'='*70}")
    print(f"TEAM IDENTITY FUNNEL ANALYSIS")
    print(f"{'='*70}")
    print(f"Total identity switches: {total}")

    has_team = [s for s in switches if s.same_team is not None]
    cross_team = [s for s in has_team if s.is_cross_team]
    same_team = [s for s in has_team if not s.is_cross_team]

    # Stage 1: Opportunity
    print(f"\n--- STAGE 1: Opportunity (cross team switches) ---")
    print(f"  Cross team switches: {len(cross_team)} / {len(has_team)} ({100*len(cross_team)/max(1,len(has_team)):.1f}%)")
    print(f"  Same team switches:  {len(same_team)} / {len(has_team)} ({100*len(same_team)/max(1,len(has_team)):.1f}%)")
    print(f"  Team info missing:   {total - len(has_team)}")

    # Stage 2: Detection (among cross team)
    ct_detected = [s for s in cross_team if s.pred_detected]
    ct_not_detected = [s for s in cross_team if not s.pred_detected]

    print(f"\n--- STAGE 2: Detection (among cross team switches) ---")
    print(f"  Confident prediction both sides: {len(ct_detected)} / {len(cross_team)} ({100*len(ct_detected)/max(1,len(cross_team)):.1f}%)")
    print(f"  Missing confident prediction:    {len(ct_not_detected)} / {len(cross_team)} ({100*len(ct_not_detected)/max(1,len(cross_team)):.1f}%)")

    # Stage 3: Signal (among detected cross team)
    ct_signal = [s for s in ct_detected if s.signal_available]
    ct_no_signal = [s for s in ct_detected if not s.signal_available]

    print(f"\n--- STAGE 3: Signal (predicted teams differ) ---")
    print(f"  Predicted teams differ: {len(ct_signal)} / {len(ct_detected)} ({100*len(ct_signal)/max(1,len(ct_detected)):.1f}%)")
    print(f"  Predicted teams same (misclassification): {len(ct_no_signal)} / {len(ct_detected)} ({100*len(ct_no_signal)/max(1,len(ct_detected)):.1f}%)")

    # Check prediction correctness
    correct_before = [s for s in ct_detected if s.pred_correct_before is True]
    correct_after = [s for s in ct_detected if s.pred_correct_after is True]
    wrong_before = [s for s in ct_detected if s.pred_correct_before is False]
    wrong_after = [s for s in ct_detected if s.pred_correct_after is False]

    print(f"\n  Team prediction accuracy at switch points:")
    has_verdict_before = [s for s in ct_detected if s.pred_correct_before is not None]
    has_verdict_after = [s for s in ct_detected if s.pred_correct_after is not None]
    if has_verdict_before:
        print(f"    Before switch: {len(correct_before)}/{len(has_verdict_before)} correct ({100*len(correct_before)/len(has_verdict_before):.1f}%)")
    if has_verdict_after:
        print(f"    After switch:  {len(correct_after)}/{len(has_verdict_after)} correct ({100*len(correct_after)/len(has_verdict_after):.1f}%)")

    # Stage 4: Splitter fired
    ct_fired = [s for s in cross_team if s.splitter_fired]
    ct_signal_fired = [s for s in ct_signal if s.splitter_fired]

    print(f"\n--- STAGE 4: Splitter Fired (within ±{TOLERANCE} frames) ---")
    print(f"  Among cross team: {len(ct_fired)} / {len(cross_team)} ({100*len(ct_fired)/max(1,len(cross_team)):.1f}%)")
    print(f"  Among signal available: {len(ct_signal_fired)} / {len(ct_signal)} ({100*len(ct_signal_fired)/max(1,len(ct_signal)):.1f}%)")

    # Also check same team switches (should be 0 or very few)
    st_fired = [s for s in same_team if s.splitter_fired]
    st_signal = [s for s in same_team if s.signal_available]
    print(f"\n  Same team switches with signal (false signal): {len(st_signal)} / {len(same_team)}")
    print(f"  Same team switches fired: {len(st_fired)} / {len(same_team)}")

    # Funnel summary
    print(f"\n--- FUNNEL SUMMARY (cross team switches only) ---")
    print(f"  {'Stage':45s} {'Count':>6} {'% of total':>10} {'% of prev':>10}")
    print(f"  {'Total identity switches':45s} {total:>6} {'100.0%':>10} {'':>10}")
    print(f"  {'Cross team (opportunity)':45s} {len(cross_team):>6} {100*len(cross_team)/total:>9.1f}% {'':>10}")
    print(f"  {'Confident prediction both sides':45s} {len(ct_detected):>6} {100*len(ct_detected)/total:>9.1f}% {100*len(ct_detected)/max(1,len(cross_team)):>9.1f}%")
    print(f"  {'Predicted teams differ (signal)':45s} {len(ct_signal):>6} {100*len(ct_signal)/total:>9.1f}% {100*len(ct_signal)/max(1,len(ct_detected)):>9.1f}%")
    print(f"  {'Splitter fired (±{0} frames)'.format(TOLERANCE):45s} {len(ct_signal_fired):>6} {100*len(ct_signal_fired)/total:>9.1f}% {100*len(ct_signal_fired)/max(1,len(ct_signal)):>9.1f}%")

    # Duration breakdown for cross team
    print(f"\n--- CROSS TEAM: SIGNAL AVAILABILITY BY DURATION ---")
    bins = [(0, 5, "<5"), (5, 25, "5-25"), (25, 100, "25-100"),
            (100, 300, "100-300"), (300, 99999, "300+")]
    print(f"  {'Duration':12s} {'Cross-team':>10} {'Detected':>10} {'Signal':>10} {'Fired':>8}")
    for lo, hi, label in bins:
        ct_bin = [s for s in cross_team if lo <= s.duration_frames < hi]
        det_bin = [s for s in ct_bin if s.pred_detected]
        sig_bin = [s for s in ct_bin if s.signal_available]
        fire_bin = [s for s in ct_bin if s.splitter_fired]
        print(f"  {label:12s} {len(ct_bin):>10} {len(det_bin):>10} {len(sig_bin):>10} {len(fire_bin):>8}")


def save_csv(switches: List[TeamSwitchInfo], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "sequence", "baseline_track_id", "frame",
            "gt_id_before", "gt_id_after",
            "gt_team_before", "gt_team_after", "same_team", "is_cross_team",
            "pred_team_before", "pred_team_after",
            "pred_conf_before", "pred_conf_after",
            "pred_correct_before", "pred_correct_after",
            "pred_teams_differ",
            "pred_detected", "signal_available", "splitter_fired",
            "duration_frames", "has_temporal_gap",
        ])
        for s in switches:
            w.writerow([
                s.sequence, s.baseline_track_id, s.frame,
                s.gt_id_before, s.gt_id_after,
                s.gt_team_before or "", s.gt_team_after or "",
                "" if s.same_team is None else int(s.same_team),
                int(s.is_cross_team),
                s.pred_team_before if s.pred_team_before is not None else "",
                s.pred_team_after if s.pred_team_after is not None else "",
                s.pred_conf_before, s.pred_conf_after,
                "" if s.pred_correct_before is None else int(s.pred_correct_before),
                "" if s.pred_correct_after is None else int(s.pred_correct_after),
                "" if s.pred_teams_differ is None else int(s.pred_teams_differ),
                int(s.pred_detected), int(s.signal_available), int(s.splitter_fired),
                s.duration_frames, int(s.has_temporal_gap),
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
    print(f"Team splitter: {TEAM_SPLITTER_DIR}")
    print(f"GSR root: {GSR_ROOT}")
    print(f"Tolerance: ±{TOLERANCE} frames")
    print()

    all_switches: List[TeamSwitchInfo] = []

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

        det_gt_map = build_det_to_gt(baseline_by_frame, gt_by_frame, IOU_THRESH)

        # Load GSR team labels
        gsr_seq = snpt_to_sngs(seq)
        gsr_json = GSR_ROOT / gsr_seq / "Labels-GameState.json"
        gt_teams: Dict[int, str] = {}
        if gsr_json.exists():
            gt_teams = load_gsr_team_labels(gsr_json)

        # Load attribute cache
        cached_tracklets = load_attribute_cache(CACHE_DIR, seq)
        if cached_tracklets is None:
            print(f"[WARN] No attribute cache for {seq}, predictions unavailable")

        # Build team label mapping (GT "left"/"right" -> predicted 0/1)
        team_label_map = None
        if cached_tracklets and gt_teams:
            team_label_map = build_team_label_map(
                cached_tracklets, baseline_tracks, det_gt_map, gt_teams
            )

        # Find switches with team info
        switches = find_switches_with_team_info(
            baseline_tracks, gt_by_frame, det_gt_map,
            cached_tracklets, gt_teams, seq, team_label_map
        )

        # Match against team splitter output
        team_splitter_path = TEAM_SPLITTER_DIR / f"{seq}.txt" if TEAM_SPLITTER_DIR else None
        if team_splitter_path and team_splitter_path.exists():
            splitter_by_frame = load_mot_by_frame(team_splitter_path)
            predicted_splits = find_predicted_splits(
                baseline_tracks, splitter_by_frame, IOU_THRESH
            )
            match_splits_to_switches(switches, predicted_splits, TOLERANCE)

        all_switches.extend(switches)
        n_cross = sum(1 for s in switches if s.is_cross_team)
        n_signal = sum(1 for s in switches if s.signal_available)
        n_fired = sum(1 for s in switches if s.splitter_fired)
        print(f"  {seq}: {len(switches)} switches | "
              f"cross_team={n_cross} signal={n_signal} fired={n_fired}")

    print_funnel(all_switches)
    save_csv(all_switches, OUTPUT_DIR / "team_funnel_analysis.csv")


if __name__ == "__main__":
    main()
