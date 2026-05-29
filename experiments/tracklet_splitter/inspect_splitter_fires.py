#!/usr/bin/env python
"""
inspect_splitter_fires.py — Locate where each splitter fires for qualitative
analysis.

Runs the team, bbox, and trajectory splitters individually on the cached
attribute tracklets and records every (sequence, parent tracklet id, split
frame) where each splitter triggers a split. The output is intended to help
locate qualitative examples for the thesis — in particular, tracklets where
one splitter fires but the others do not.

Outputs (written to settings.OUTPUT_ROOT):
    splitter_fires_all.csv       — every fire from team, bbox, trajectory.
    splitter_fires_unique.csv    — fires unique to one splitter (no fire
                                   from any other inspected splitter within
                                   TOLERANCE_FRAMES of the same parent
                                   tracklet).
    splitter_fires_summary.csv   — per-splitter counts of total and unique
                                   fires.

Usage:
    python inspect_splitter_fires.py [--split valid|test]
                                     [--tolerance 50]

Inspection columns:
    sequence           sequence folder name (matches SoccerNet-Tracking layout)
    splitter           one of {team, bbox, trajectory}
    parent_track_id    original tracklet id before splitting
    split_frame        absolute frame number where the split was placed
    tracklet_start     first frame of the original tracklet
    tracklet_end       last frame of the original tracklet
    tracklet_length    number of frames in the original tracklet
    unique_to_splitter True if no other splitter fired within tolerance
                       in the same parent tracklet
"""

import argparse
import copy
import csv
import pickle
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import settings
from utils.config import SplitterConfig
from tracklets.splitters.modular_splitter import ModularSplitter
from tracklets.splitters.trajectory_splitter import TrajectorySplitter


SPLITTERS_TO_INSPECT = ["team", "bbox", "trajectory"]
DEFAULT_TOLERANCE_FRAMES = 50


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def set_eval_split(split: str) -> None:
    assert split in ("train", "valid", "test"), f"Unknown split: {split}"
    settings.EVAL_SPLIT = split
    settings.DATA_ROOT = Path(
        rf"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\soccernet-player-tracking\{split}"
    )


def make_splitter_config() -> SplitterConfig:
    known_fields = set(SplitterConfig.__dataclass_fields__.keys())
    known = {k: v for k, v in settings.SPLITTER.items() if k in known_fields}
    extra = {k: v for k, v in settings.SPLITTER.items() if k not in known_fields}
    cfg = SplitterConfig(**known)
    for k, v in extra.items():
        setattr(cfg, k, v)
    return cfg


def load_attribute_cache(sequence: str, cache_dir: Path) -> dict:
    cache_path = cache_dir / f"cache_attributes_{sequence}.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Attribute cache missing: {cache_path}\n"
            "Run main.py at least once to generate it."
        )
    with open(cache_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Individual splitter runners
# ---------------------------------------------------------------------------

def run_modular_splitter(tracklets: dict, signal_name: str, cfg: SplitterConfig) -> list:
    """Run ModularSplitter with a single signal and return per-fire records.

    Each fire is dict with keys:
        parent_track_id, split_frame, tracklet_start, tracklet_end,
        tracklet_length.
    """
    splitter = ModularSplitter(cfg, signals=[signal_name])

    fires = []
    next_id = (max(tracklets.keys()) if tracklets else 0) + 1

    for parent_id, tracklet in tracklets.items():
        fragments = splitter.split_tracklet(tracklet, next_id)
        if not fragments or len(fragments) < 2:
            continue

        # Sort fragments by their first frame to be safe.
        fragments_sorted = sorted(fragments, key=lambda f: f.frames[0])

        for frag in fragments_sorted[1:]:
            fires.append({
                "parent_track_id": parent_id,
                "split_frame": int(frag.frames[0]),
                "tracklet_start": int(tracklet.frames[0]),
                "tracklet_end": int(tracklet.frames[-1]),
                "tracklet_length": len(tracklet.frames),
            })

        # Keep id allocation moving so we never clash with input ids.
        next_id = max(next_id, max(f.track_id for f in fragments) + 1)

    return fires


def run_trajectory_splitter(tracklets: dict, cfg: SplitterConfig) -> list:
    """Run TrajectorySplitter (cross-tracklet) and return per-fire records."""
    splitter = TrajectorySplitter(cfg)

    # Trajectory splitter operates on the whole dict at once. Keep a deep
    # copy so we have the originals for span lookup.
    originals = copy.deepcopy(tracklets)
    result = splitter.split_all_tracklets(copy.deepcopy(tracklets))

    # Group result tracklets by parent_id. Tracklets that were not split
    # keep their original track_id and have parent_id == track_id (or None).
    by_parent = defaultdict(list)
    for tid, t in result.items():
        parent = getattr(t, "parent_id", None)
        if parent is None or parent not in originals:
            # Unsplit tracklet — its key matches an original id.
            if tid in originals:
                by_parent[tid].append(t)
        else:
            by_parent[parent].append(t)

    fires = []
    for parent_id, frags in by_parent.items():
        if len(frags) < 2 or parent_id not in originals:
            continue
        original = originals[parent_id]
        frags_sorted = sorted(frags, key=lambda t: t.frames[0])
        for frag in frags_sorted[1:]:
            fires.append({
                "parent_track_id": parent_id,
                "split_frame": int(frag.frames[0]),
                "tracklet_start": int(original.frames[0]),
                "tracklet_end": int(original.frames[-1]),
                "tracklet_length": len(original.frames),
            })

    return fires


# ---------------------------------------------------------------------------
# Uniqueness check
# ---------------------------------------------------------------------------

def is_unique_fire(
    fire: dict,
    other_fires_by_splitter: dict,
    current_splitter: str,
    tolerance: int,
) -> bool:
    """Return True if no other splitter fired within tolerance frames inside
    the same parent tracklet."""
    parent = fire["parent_track_id"]
    frame = fire["split_frame"]
    for other_name, other_fires in other_fires_by_splitter.items():
        if other_name == current_splitter:
            continue
        for other_fire in other_fires:
            if other_fire["parent_track_id"] != parent:
                continue
            if abs(other_fire["split_frame"] - frame) <= tolerance:
                return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Inspect where each splitter fires.")
    ap.add_argument("--split", default=settings.EVAL_SPLIT,
                    choices=["train", "valid", "test"])
    ap.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE_FRAMES,
                    help="Max frame distance for two fires to be considered "
                         "co-located (default: 50).")
    args = ap.parse_args()

    set_eval_split(args.split)
    cfg = make_splitter_config()

    cache_dir = settings.OUTPUT_ROOT / "cache"
    out_dir = settings.OUTPUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences = sorted([d.name for d in settings.DATA_ROOT.iterdir() if d.is_dir()])
    print(f"Split: {args.split}")
    print(f"Sequences: {len(sequences)}")
    print(f"Splitters: {SPLITTERS_TO_INSPECT}")
    print(f"Tolerance (frames): {args.tolerance}\n")

    all_records = []

    for seq in sequences:
        tracklets = load_attribute_cache(seq, cache_dir)

        # Run each inspected splitter on a deep copy so they don't interfere.
        per_splitter_fires = {}
        for splitter_name in SPLITTERS_TO_INSPECT:
            tracklets_copy = copy.deepcopy(tracklets)
            if splitter_name == "trajectory":
                fires = run_trajectory_splitter(tracklets_copy, cfg)
            else:
                fires = run_modular_splitter(tracklets_copy, splitter_name, cfg)
            per_splitter_fires[splitter_name] = fires

        # Pretty-print one line per sequence with counts.
        counts = " | ".join(
            f"{n}={len(per_splitter_fires[n]):>3d}" for n in SPLITTERS_TO_INSPECT
        )
        print(f"  {seq}  {counts}")

        # Build records with uniqueness check.
        for splitter_name, fires in per_splitter_fires.items():
            for fire in fires:
                unique = is_unique_fire(
                    fire, per_splitter_fires, splitter_name, args.tolerance
                )
                all_records.append({
                    "sequence": seq,
                    "splitter": splitter_name,
                    "parent_track_id": fire["parent_track_id"],
                    "split_frame": fire["split_frame"],
                    "tracklet_start": fire["tracklet_start"],
                    "tracklet_end": fire["tracklet_end"],
                    "tracklet_length": fire["tracklet_length"],
                    "unique_to_splitter": unique,
                })

    if not all_records:
        print("\nNo splitter fires recorded. Nothing to write.")
        return

    # --- Write all fires -----------------------------------------------------
    fieldnames = list(all_records[0].keys())
    all_csv = out_dir / "splitter_fires_all.csv"
    with open(all_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    # --- Write unique fires only --------------------------------------------
    unique_records = [r for r in all_records if r["unique_to_splitter"]]
    unique_csv = out_dir / "splitter_fires_unique.csv"
    with open(unique_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_records)

    # --- Write summary -------------------------------------------------------
    summary = defaultdict(lambda: {"total": 0, "unique": 0})
    for r in all_records:
        summary[r["splitter"]]["total"] += 1
        if r["unique_to_splitter"]:
            summary[r["splitter"]]["unique"] += 1

    summary_csv = out_dir / "splitter_fires_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["splitter", "total_fires", "unique_fires"])
        writer.writeheader()
        for name in SPLITTERS_TO_INSPECT:
            writer.writerow({
                "splitter": name,
                "total_fires": summary[name]["total"],
                "unique_fires": summary[name]["unique"],
            })

    # --- Console summary -----------------------------------------------------
    print(f"\n=== Summary ===")
    print(f"  {'splitter':<12} {'total':>6}  {'unique':>6}")
    for name in SPLITTERS_TO_INSPECT:
        print(f"  {name:<12} {summary[name]['total']:>6}  {summary[name]['unique']:>6}")

    print(f"\nAll fires:    {all_csv} ({len(all_records)} rows)")
    print(f"Unique fires: {unique_csv} ({len(unique_records)} rows)")
    print(f"Summary:      {summary_csv}")
    print(f"\nFor qualitative inspection, sort splitter_fires_unique.csv by "
          f"splitter to find candidates where one splitter caught a switch "
          f"the others missed.")


if __name__ == "__main__":
    main()
