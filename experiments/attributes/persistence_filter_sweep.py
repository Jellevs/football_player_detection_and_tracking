"""
persistence_filter_sweep.py — Parameter sensitivity analysis for jersey and team persistence filters.

For each parameter configuration, this script simulates the persistence filter
logic on cached attribute predictions and counts:
  - True splits:  splits at identity switches where the GT attribute differs
  - False splits: splits at identity switches where the GT attribute is the same
  - Total splits: all splits the filter would produce (including on non-switch segments)

This avoids re-running the full splitter pipeline by directly applying the
persistence logic from UnifiedSplitter to the cached predictions.

Usage:
    python experiments/attributes/persistence_filter_sweep.py
"""
from __future__ import annotations

import csv
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product
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
CACHE_DIR = REPO_ROOT / "output" / "cache"

GSR_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\data\SoccerNetGS\test")

IOU_THRESH = 0.5

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------
JERSEY_GRID = {
    "min_persistence": [10, 15, 20, 30],
    "entropy_threshold": [0.005, 0.01, 0.02],
    # Fixed parameters
    "lookahead": [50, 100, 150],
    "min_persistence_ratio": [0.7,0.8,0.9],
}

TEAM_GRID = {
    "min_persistence": [10],
    "confidence_threshold": [0.5, 0.6, 0.7],
    # Fixed parameters
    "lookahead": [50, 100, 150],
    "min_persistence_ratio": [0.7,0.8,0.9],
}


# ---------------------------------------------------------------------------
# MOT / matching helpers (shared with funnel scripts)
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
# GSR loading
# ---------------------------------------------------------------------------

import json

def load_gsr_jersey_and_team(gsr_json_path: Path):
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
# Load attribute cache
# ---------------------------------------------------------------------------

def load_attribute_cache(cache_dir: Path, seq: str):
    cache_path = cache_dir / f"cache_attributes_{seq}.pkl"
    if not cache_path.exists():
        return None
    with open(cache_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Identity switch detection
# ---------------------------------------------------------------------------

@dataclass
class SwitchInfo:
    """Minimal info about an identity switch for parameter sweeping."""
    sequence: str
    baseline_track_id: int
    switch_index: int       # index within the tracklet's frame list
    gt_id_before: int
    gt_id_after: int
    gt_jersey_before: Optional[int] = None
    gt_jersey_after: Optional[int] = None
    gt_team_before: Optional[str] = None
    gt_team_after: Optional[str] = None


def find_switches(
    baseline_tracks, gt_by_frame, det_gt_map, gt_jerseys, gt_teams, sequence
) -> List[SwitchInfo]:
    """Find all identity switches with GT attribute info."""
    switches = []
    for tid, frames_boxes in baseline_tracks.items():
        prev_gt = None
        for i, (frame, box) in enumerate(frames_boxes):
            key = (frame, tid, float(box[0]), float(box[1]))
            gt_id = det_gt_map.get(key, -1)
            if gt_id == -1:
                continue
            if prev_gt is not None and gt_id != prev_gt:
                sw = SwitchInfo(
                    sequence=sequence,
                    baseline_track_id=tid,
                    switch_index=i,
                    gt_id_before=prev_gt,
                    gt_id_after=gt_id,
                    gt_jersey_before=gt_jerseys.get(prev_gt),
                    gt_jersey_after=gt_jerseys.get(gt_id),
                    gt_team_before=gt_teams.get(prev_gt),
                    gt_team_after=gt_teams.get(gt_id),
                )
                switches.append(sw)
            if gt_id != -1:
                prev_gt = gt_id
    return switches


# ---------------------------------------------------------------------------
# Persistence filter simulation (mirrors unified_splitter.py logic)
# ---------------------------------------------------------------------------

def digit_compatible(a, b):
    s_a, s_b = str(int(a)), str(int(b))
    if len(s_a) == 1 and len(s_b) == 2:
        return s_a in s_b
    if len(s_a) == 2 and len(s_b) == 1:
        return s_b in s_a
    return False


def simulate_jersey_splits(
    jerseys: list,
    entropies: list,
    min_persistence: int,
    entropy_threshold: float,
    lookahead: int,
    min_persistence_ratio: float,
    min_fragment_length: int = 20,
) -> List[int]:
    """Simulate jersey persistence filter, return list of split indices."""
    n = len(jerseys)

    def valid(idx):
        v = jerseys[idx]
        if v is None:
            return False
        if isinstance(v, float) and np.isnan(v):
            return False
        return entropies[idx] <= entropy_threshold

    def persistent(start, candidate):
        window_end = min(start + lookahead, n)
        reliable = []
        for k in range(start, window_end):
            if valid(k):
                reliable.append(jerseys[k])
            if len(reliable) >= min_persistence:
                break
        if len(reliable) < min_persistence:
            return False
        ratio = sum(1 for j in reliable if j == candidate) / len(reliable)
        return ratio >= min_persistence_ratio

    # Establish reference
    ref = None
    for i in range(n):
        if valid(i) and persistent(i, jerseys[i]):
            ref = jerseys[i]
            break

    if ref is None:
        return []

    splits = []
    for i in range(n):
        if not valid(i):
            continue
        candidate = jerseys[i]
        if candidate == ref:
            continue
        if digit_compatible(candidate, ref):
            continue
        if persistent(i, candidate):
            splits.append(i)
            ref = candidate

    # Filter by min_fragment_length
    return _filter_fragments(splits, n, min_fragment_length)


def simulate_team_splits(
    teams: list,
    team_confs: list,
    min_persistence: int,
    confidence_threshold: float,
    lookahead: int,
    min_persistence_ratio: float,
    min_fragment_length: int = 20,
) -> List[int]:
    """Simulate team persistence filter, return list of split indices."""
    n = len(teams)

    def valid(idx):
        v = teams[idx]
        if v is None:
            return False
        if isinstance(v, float) and np.isnan(v):
            return False
        c = team_confs[idx]
        if isinstance(c, float) and np.isnan(c):
            return False
        return c >= confidence_threshold

    def persistent(start, candidate):
        window_end = min(start + lookahead, n)
        window = [teams[k] for k in range(start, window_end) if valid(k)]
        if not window:
            return False
        count = sum(1 for t in window if t == candidate)
        if count / len(window) < min_persistence_ratio:
            return False
        return True

    # Establish reference
    ref = None
    for i in range(n):
        if valid(i) and persistent(i, teams[i]):
            ref = teams[i]
            break

    if ref is None:
        return []

    splits = []
    for i in range(n):
        if not valid(i):
            continue
        candidate = teams[i]
        if candidate == ref:
            continue
        if persistent(i, candidate):
            splits.append(i)
            ref = candidate

    return _filter_fragments(splits, n, min_fragment_length)


def _filter_fragments(split_indices: List[int], n: int, min_fragment_length: int) -> List[int]:
    """Remove splits that would create fragments shorter than min_fragment_length."""
    if not split_indices:
        return []
    # Check which splits survive the fragment length filter
    boundaries = [0] + split_indices + [n]
    surviving = []
    for i in range(len(boundaries) - 1):
        frag_len = boundaries[i + 1] - boundaries[i]
        if frag_len < min_fragment_length:
            # This fragment is too short; the split creating it should be removed
            # But this is complex with cascading effects, so we use a simpler approach:
            # just check if the split itself creates two fragments >= min_fragment_length
            pass

    # Simplified: keep splits where both adjacent fragments are long enough
    for idx in split_indices:
        # Find position of this split in the boundary list
        left_boundary = 0
        right_boundary = n
        for s in split_indices:
            if s < idx and s > left_boundary:
                left_boundary = s
            if s > idx and s < right_boundary:
                right_boundary = s
        left_len = idx - left_boundary
        right_len = right_boundary - idx
        if left_len >= min_fragment_length and right_len >= min_fragment_length:
            surviving.append(idx)

    return surviving


# ---------------------------------------------------------------------------
# Sweep logic
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    # Parameters
    param_name: str       # e.g. "jersey" or "team"
    min_persistence: int
    threshold: float      # entropy for jersey, confidence for team
    lookahead: int
    min_persistence_ratio: float
    # Counts
    total_splits: int           # all splits across all tracklets
    true_splits: int            # splits near switches where GT attribute differs
    false_splits_at_switch: int # splits near switches where GT attribute is the same
    spurious_splits: int        # splits NOT near any GT switch (on pure segments)
    missed_switches: int        # switches with different GT attribute but no split
    total_differ_switches: int  # total switches where GT attribute differs
    total_same_switches: int    # total switches where GT attribute is the same

    @property
    def all_false_splits(self) -> int:
        """Total false splits = at same-attribute switches + on pure segments."""
        return self.false_splits_at_switch + self.spurious_splits


def _classify_splits(
    splits: List[int],
    tracklet_switches: List[SwitchInfo],
    attr_type: str,
    tolerance: int = 750,
) -> Tuple[int, int, int]:
    """
    Classify each split as true, false-at-switch, or spurious.

    For each split, find the nearest GT switch in this tracklet.
    - If within tolerance and GT attribute differs → true
    - If within tolerance and GT attribute is the same → false at switch
    - If no switch within tolerance → spurious (split on pure segment)

    Returns (true_count, false_at_switch_count, spurious_count).
    """
    true_count = 0
    false_at_switch = 0
    spurious = 0

    for sp in splits:
        best_dist = tolerance + 1
        best_sw = None
        for sw in tracklet_switches:
            d = abs(sp - sw.switch_index)
            if d < best_dist:
                best_dist = d
                best_sw = sw

        if best_sw is None or best_dist > tolerance:
            spurious += 1
            continue

        # Check if GT attribute differs
        if attr_type == "jersey":
            gt_before = best_sw.gt_jersey_before
            gt_after = best_sw.gt_jersey_after
            if gt_before is not None and gt_after is not None:
                if gt_before != gt_after:
                    true_count += 1
                else:
                    false_at_switch += 1
            else:
                # GT jersey unknown for one side; count as spurious
                spurious += 1
        else:  # team
            gt_before = best_sw.gt_team_before
            gt_after = best_sw.gt_team_after
            if gt_before is not None and gt_after is not None:
                if gt_before != gt_after:
                    true_count += 1
                else:
                    false_at_switch += 1
            else:
                spurious += 1

    return true_count, false_at_switch, spurious


def run_jersey_sweep(
    all_tracklet_data: List[dict],
    all_switches: List[SwitchInfo],
) -> List[SweepResult]:
    """Run jersey parameter sweep across all sequences."""

    # Pre-index switches by (seq, tid) for fast lookup
    switches_by_key: Dict[Tuple[str, int], List[SwitchInfo]] = defaultdict(list)
    for sw in all_switches:
        switches_by_key[(sw.sequence, sw.baseline_track_id)].append(sw)

    differ_switches = [sw for sw in all_switches
                       if sw.gt_jersey_before is not None
                       and sw.gt_jersey_after is not None
                       and sw.gt_jersey_before != sw.gt_jersey_after]
    same_switches = [sw for sw in all_switches
                     if sw.gt_jersey_before is not None
                     and sw.gt_jersey_after is not None
                     and sw.gt_jersey_before == sw.gt_jersey_after]

    keys = list(JERSEY_GRID.keys())
    values = [JERSEY_GRID[k] for k in keys]
    results = []

    for combo in product(*values):
        params = dict(zip(keys, combo))
        mp = params["min_persistence"]
        et = params["entropy_threshold"]
        la = params["lookahead"]
        mpr = params["min_persistence_ratio"]

        total_splits = 0
        total_true = 0
        total_false_at_switch = 0
        total_spurious = 0

        # Also track which differ-switches get hit (for recall)
        switch_hit = {id(sw): False for sw in differ_switches}

        for td in all_tracklet_data:
            jerseys = td["jerseys"]
            entropies = td["entropies"]
            tid = td["tid"]
            seq = td["seq"]

            splits = simulate_jersey_splits(
                jerseys, entropies, mp, et, la, mpr
            )
            total_splits += len(splits)

            tracklet_switches = switches_by_key.get((seq, tid), [])

            # Classify each split
            t, f, s = _classify_splits(splits, tracklet_switches, "jersey")
            total_true += t
            total_false_at_switch += f
            total_spurious += s

            # Track which differ-switches are hit (for recall)
            for sw in tracklet_switches:
                if sw.gt_jersey_before is not None and sw.gt_jersey_after is not None \
                   and sw.gt_jersey_before != sw.gt_jersey_after:
                    for sp in splits:
                        if abs(sp - sw.switch_index) <= 750:
                            switch_hit[id(sw)] = True
                            break

        recall_hits = sum(1 for v in switch_hit.values() if v)

        results.append(SweepResult(
            param_name="jersey",
            min_persistence=mp,
            threshold=et,
            lookahead=la,
            min_persistence_ratio=mpr,
            total_splits=total_splits,
            true_splits=total_true,
            false_splits_at_switch=total_false_at_switch,
            spurious_splits=total_spurious,
            missed_switches=len(differ_switches) - recall_hits,
            total_differ_switches=len(differ_switches),
            total_same_switches=len(same_switches),
        ))

        print(f"  Jersey mp={mp:3d} et={et:.3f} la={la:3d} | "
              f"splits={total_splits:4d} "
              f"true={total_true:3d} false@sw={total_false_at_switch:2d} "
              f"spurious={total_spurious:3d} "
              f"recall={recall_hits:3d}/{len(differ_switches)}")

    return results


def build_team_label_map(
    all_tracklet_data: List[dict],
    all_switches: List[SwitchInfo],
    gt_teams_by_seq: Dict[str, Dict[int, str]],
) -> Dict[str, Dict]:
    """
    For each sequence, figure out which predicted cluster (0 or 1) maps to
    which GT team label ("left" or "right") by majority vote.
    """
    seq_mapping = {}

    for seq in set(td["seq"] for td in all_tracklet_data):
        gt_teams = gt_teams_by_seq.get(seq, {})
        if not gt_teams:
            continue

        # Collect (predicted_team, gt_team) pairs from all frames
        votes: Dict[int, List[str]] = defaultdict(list)

        for td in all_tracklet_data:
            if td["seq"] != seq:
                continue
            teams = td["teams"]
            gt_ids = td["gt_ids"]  # list of (index, gt_id) pairs

            for idx, gt_id in gt_ids:
                if idx >= len(teams):
                    continue
                pred = teams[idx]
                if pred is None or (isinstance(pred, float) and np.isnan(pred)):
                    continue
                gt_label = gt_teams.get(gt_id)
                if gt_label is not None:
                    votes[int(pred)].append(gt_label)

        if not votes:
            continue

        # For each predicted cluster, find the majority GT label
        mapping = {}
        for pred_cluster, gt_labels in votes.items():
            majority = Counter(gt_labels).most_common(1)[0][0]
            mapping[pred_cluster] = majority

        seq_mapping[seq] = mapping

    return seq_mapping


def run_team_sweep(
    all_tracklet_data: List[dict],
    all_switches: List[SwitchInfo],
) -> List[SweepResult]:
    """Run team parameter sweep across all sequences."""

    # Pre-index switches by (seq, tid)
    switches_by_key: Dict[Tuple[str, int], List[SwitchInfo]] = defaultdict(list)
    for sw in all_switches:
        switches_by_key[(sw.sequence, sw.baseline_track_id)].append(sw)

    cross_team = [sw for sw in all_switches
                  if sw.gt_team_before is not None
                  and sw.gt_team_after is not None
                  and sw.gt_team_before != sw.gt_team_after]
    same_team = [sw for sw in all_switches
                 if sw.gt_team_before is not None
                 and sw.gt_team_after is not None
                 and sw.gt_team_before == sw.gt_team_after]

    keys = list(TEAM_GRID.keys())
    values = [TEAM_GRID[k] for k in keys]
    results = []

    for combo in product(*values):
        params = dict(zip(keys, combo))
        mp = params["min_persistence"]
        ct = params["confidence_threshold"]
        la = params["lookahead"]
        mpr = params["min_persistence_ratio"]

        total_splits = 0
        total_true = 0
        total_false_at_switch = 0
        total_spurious = 0

        # Track which cross-team switches get hit (for recall)
        switch_hit = {id(sw): False for sw in cross_team}

        for td in all_tracklet_data:
            teams = td["teams"]
            team_confs = td["team_confs"]
            tid = td["tid"]
            seq = td["seq"]

            splits = simulate_team_splits(
                teams, team_confs, mp, ct, la, mpr
            )
            total_splits += len(splits)

            tracklet_switches = switches_by_key.get((seq, tid), [])

            # Classify each split
            t, f, s = _classify_splits(splits, tracklet_switches, "team")
            total_true += t
            total_false_at_switch += f
            total_spurious += s

            # Track which cross-team switches are hit (for recall)
            for sw in tracklet_switches:
                if sw.gt_team_before is not None and sw.gt_team_after is not None \
                   and sw.gt_team_before != sw.gt_team_after:
                    for sp in splits:
                        if abs(sp - sw.switch_index) <= 750:
                            switch_hit[id(sw)] = True
                            break

        recall_hits = sum(1 for v in switch_hit.values() if v)

        results.append(SweepResult(
            param_name="team",
            min_persistence=mp,
            threshold=ct,
            lookahead=la,
            min_persistence_ratio=mpr,
            total_splits=total_splits,
            true_splits=total_true,
            false_splits_at_switch=total_false_at_switch,
            spurious_splits=total_spurious,
            missed_switches=len(cross_team) - recall_hits,
            total_differ_switches=len(cross_team),
            total_same_switches=len(same_team),
        ))

        print(f"  Team   mp={mp:3d} ct={ct:.2f} la={la:3d} | "
              f"splits={total_splits:4d} "
              f"true={total_true:3d} false@sw={total_false_at_switch:3d} "
              f"spurious={total_spurious:3d} "
              f"recall={recall_hits:3d}/{len(cross_team)}")

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(results: List[SweepResult], name: str):
    print(f"\n{'='*90}")
    print(f"{name.upper()} PERSISTENCE FILTER PARAMETER SWEEP")
    print(f"{'='*90}")

    # Sort by true_splits descending, then total false ascending
    results_sorted = sorted(results, key=lambda r: (-r.true_splits, r.all_false_splits))

    header = (f"{'mp':>4} {'thresh':>6} {'la':>4} {'total':>6} "
              f"{'true':>5} {'f@sw':>5} {'spur':>5} {'f_tot':>5} "
              f"{'prec':>7} {'recall':>7}")
    print(header)
    print("-" * len(header))

    for r in results_sorted:
        tp = r.true_splits
        fp_total = r.all_false_splits
        precision = tp / (tp + fp_total) if (tp + fp_total) > 0 else 0.0
        recall = tp / r.total_differ_switches if r.total_differ_switches > 0 else 0.0

        print(f"{r.min_persistence:>4d} {r.threshold:>6.3f} {r.lookahead:>4d} "
              f"{r.total_splits:>6d} "
              f"{tp:>5d} {r.false_splits_at_switch:>5d} {r.spurious_splits:>5d} "
              f"{fp_total:>5d} "
              f"{precision:>6.1%} {recall:>6.1%}")

    # Highlight current settings
    print(f"\n  Current settings marked with *")
    if name == "jersey":
        current_mp, current_t, current_la = 20, 0.01, 100
    else:
        current_mp, current_t, current_la = 5, 0.6, 100

    for r in results_sorted:
        if r.min_persistence == current_mp and abs(r.threshold - current_t) < 1e-6 \
           and r.lookahead == current_la:
            tp = r.true_splits
            fp_total = r.all_false_splits
            precision = tp / (tp + fp_total) if (tp + fp_total) > 0 else 0.0
            recall = tp / r.total_differ_switches if r.total_differ_switches > 0 else 0.0
            print(f"  * mp={r.min_persistence} threshold={r.threshold:.3f} la={r.lookahead}: "
                  f"true={tp} false@sw={r.false_splits_at_switch} spurious={r.spurious_splits} "
                  f"precision={precision:.1%} recall={recall:.1%}")


def save_results_csv(results: List[SweepResult], path: Path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "type", "min_persistence", "threshold", "lookahead",
            "min_persistence_ratio", "total_splits",
            "true_splits", "false_splits_at_switch", "spurious_splits",
            "total_false_splits",
            "total_differ_switches", "total_same_switches",
            "precision", "recall",
        ])
        for r in results:
            tp = r.true_splits
            fp_total = r.all_false_splits
            precision = tp / (tp + fp_total) if (tp + fp_total) > 0 else 0.0
            recall = tp / r.total_differ_switches if r.total_differ_switches > 0 else 0.0
            w.writerow([
                r.param_name, r.min_persistence, r.threshold, r.lookahead,
                r.min_persistence_ratio, r.total_splits,
                r.true_splits, r.false_splits_at_switch, r.spurious_splits,
                fp_total,
                r.total_differ_switches, r.total_same_switches,
                f"{precision:.4f}", f"{recall:.4f}",
            ])
    print(f"\nResults saved to {path}")


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


def main():
    with SEQMAP.open("r") as f:
        sequences = [line.strip() for line in f
                     if line.strip() and not line.strip().lower().startswith("name")]

    print(f"Sequences: {len(sequences)}")
    print(f"Loading data...")

    all_tracklet_data = []
    all_switches = []
    gt_teams_by_seq = {}

    for seq in sequences:
        gt_path = resolve_gt_path(GT_ROOT, seq)
        baseline_path = BASELINE_DIR / f"{seq}.txt"

        if gt_path is None or not baseline_path.exists():
            print(f"  [WARN] Skipping {seq}")
            continue

        baseline_tracks = load_mot_by_track(baseline_path)
        baseline_by_frame = load_mot_by_frame(baseline_path)
        gt_by_frame = load_mot_by_frame(gt_path)
        det_gt_map = build_det_to_gt(baseline_by_frame, gt_by_frame, IOU_THRESH)

        # GSR annotations
        gsr_seq = snpt_to_sngs(seq)
        gsr_json = GSR_ROOT / gsr_seq / "Labels-GameState.json"
        gt_jerseys, gt_teams = {}, {}
        if gsr_json.exists():
            gt_jerseys, gt_teams = load_gsr_jersey_and_team(gsr_json)
        gt_teams_by_seq[seq] = gt_teams

        # Attribute cache
        cached_tracklets = load_attribute_cache(CACHE_DIR, seq)
        if cached_tracklets is None:
            print(f"  [WARN] No cache for {seq}")
            continue

        # Find switches
        switches = find_switches(
            baseline_tracks, gt_by_frame, det_gt_map,
            gt_jerseys, gt_teams, seq
        )
        all_switches.extend(switches)

        # Collect tracklet prediction data for sweep
        for tid, frames_boxes in baseline_tracks.items():
            if tid not in cached_tracklets:
                continue
            tracklet = cached_tracklets[tid]
            jerseys = tracklet.pred_attributes.get("jerseys", [])
            entropies = tracklet.pred_attributes.get("jersey_entropies",
                        tracklet.pred_attributes.get("jersey_confs", []))
            teams = tracklet.pred_attributes.get("teams", [])
            team_confs = tracklet.pred_attributes.get("team_confs", [])

            n = len(frames_boxes)
            jerseys = list(jerseys) + [None] * max(0, n - len(jerseys))
            entropies = list(entropies) + [1.0] * max(0, n - len(entropies))
            teams = list(teams) + [None] * max(0, n - len(teams))
            team_confs = list(team_confs) + [0.0] * max(0, n - len(team_confs))

            # Build GT ID mapping for this tracklet (for team label mapping)
            gt_ids = []
            for i, (frame, box) in enumerate(frames_boxes):
                key = (frame, tid, float(box[0]), float(box[1]))
                gt_id = det_gt_map.get(key, -1)
                if gt_id != -1:
                    gt_ids.append((i, gt_id))

            all_tracklet_data.append({
                "tid": tid,
                "seq": seq,
                "jerseys": jerseys,
                "entropies": entropies,
                "teams": teams,
                "team_confs": team_confs,
                "gt_ids": gt_ids,
                "n_frames": n,
            })

        print(f"  {seq}: {len(switches)} switches, "
              f"{sum(1 for t in baseline_tracks if t in cached_tracklets)} tracklets")

    print(f"\nTotal: {len(all_switches)} switches, {len(all_tracklet_data)} tracklets")

    # --- Jersey sweep ---
    print(f"\n{'='*80}")
    print("JERSEY PARAMETER SWEEP")
    print(f"{'='*80}")
    jersey_results = run_jersey_sweep(all_tracklet_data, all_switches)
    print_results(jersey_results, "jersey")
    save_results_csv(jersey_results, OUTPUT_DIR / "jersey_sweep.csv")

    # --- Team sweep ---
    print(f"\n{'='*80}")
    print("TEAM PARAMETER SWEEP")
    print(f"{'='*80}")
    team_results = run_team_sweep(all_tracklet_data, all_switches)
    print_results(team_results, "team")
    save_results_csv(team_results, OUTPUT_DIR / "team_sweep.csv")


if __name__ == "__main__":
    main()
