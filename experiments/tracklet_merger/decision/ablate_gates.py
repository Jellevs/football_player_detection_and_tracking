#!/usr/bin/env python
"""
ablate_gates.py — Gate ablation for the rule-based DecisionMerger.

Disables one gate at a time (and a few combined conditions) to quantify
how much each rule contributes to the merger's performance. Mirrors the
ablation pattern used for the XGBoost hard-constraint section.

Gates in DecisionMerger (decision_merger.py):
    1. temporal_overlap        — block when tracklets share frames
    2. jersey_conflict_block   — block when both jerseys confident and differ
    3. jersey_team_merge       — auto-merge when jersey+team both agree
    4. team_conflict_block     — block when jersey agrees but teams disagree
    5. reid_fallback           — accept ReID match below threshold

Conditions evaluated:
    - baseline                   all gates ON  (= current DecisionMerger)
    - no_jersey_conflict_block   jersey mismatch no longer blocks; falls back to ReID
    - no_jersey_team_merge       confident jersey+team match no longer auto-merges; uses ReID
    - no_team_conflict_block     team mismatch (after jersey match) no longer blocks
    - no_reid_fallback           only confident jersey+team can produce a merge
    - reid_only                  disable all attribute gates (temporal overlap kept);
                                 pure ReID + hierarchical merging
    - no_jersey_logic            disable both jersey gates (no_conflict + no_auto_merge)
    - no_temporal_overlap        sanity check: allow temporally overlapping merges

We do NOT disable the temporal-overlap gate by default because it is a
physical impossibility, not a heuristic; we include it as a sanity-check
row so reviewers can see the catastrophic effect.

Prerequisites:
    Attribute caches must exist (output/cache/cache_attributes_*.pkl).

Usage:
    python ablate_gates.py                     # validation split
    python ablate_gates.py --split test        # test set
    python ablate_gates.py --reid 0.4          # override ReID threshold

Outputs:
    output/decision_merger_gate_ablation.csv
"""

import argparse
import copy
import csv
import pickle
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import settings
from utils.build import build_paths
from utils.config import SplitterConfig
from utils.evaluation_utils import save_mot_file_for_sn_trackeval
from utils.run_evaluation import run_evaluation
from tracklets.split_tracklets import split_tracklets
from tracklets.decision_merger import DecisionMerger


# ---------------------------------------------------------------------------
# Gate-toggleable merger
# ---------------------------------------------------------------------------

class AblatableDecisionMerger(DecisionMerger):
    """DecisionMerger with each gate individually disable-able.

    All gates default to ON, so a default-constructed instance behaves
    identically to the parent class.
    """

    def __init__(
        self,
        *args,
        temporal_overlap: bool = True,
        jersey_conflict_block: bool = True,
        jersey_team_merge: bool = True,
        team_conflict_block: bool = True,
        reid_fallback: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gate_temporal_overlap      = temporal_overlap
        self.gate_jersey_conflict_block = jersey_conflict_block
        self.gate_jersey_team_merge     = jersey_team_merge
        self.gate_team_conflict_block   = team_conflict_block
        self.gate_reid_fallback         = reid_fallback

    def _pair_decision(self, t1, t2) -> str:
        # 1. Temporal overlap.
        if self.gate_temporal_overlap and (set(t1.frames) & set(t2.frames)):
            return "block"

        j1 = self._get_jersey(t1)
        j2 = self._get_jersey(t2)

        if j1 is not None and j2 is not None:
            # 2. Jersey conflict.
            if j1 != j2:
                if self.gate_jersey_conflict_block:
                    return "block"
                return "reid" if self.gate_reid_fallback else "block"

            # Jerseys agree — examine team.
            team1, cons1 = self._get_team(t1)
            team2, cons2 = self._get_team(t2)
            t1_conf = team1 is not None and cons1 >= self.team_consistency_threshold
            t2_conf = team2 is not None and cons2 >= self.team_consistency_threshold

            if t1_conf and t2_conf:
                if team1 == team2:
                    # 3. Jersey+team auto-merge.
                    if self.gate_jersey_team_merge:
                        return "merge"
                    return "reid" if self.gate_reid_fallback else "block"
                # 4. Team conflict.
                if self.gate_team_conflict_block:
                    return "block"
                return "reid" if self.gate_reid_fallback else "block"

            # Jersey agrees, team not confident for both -> ReID.
            return "reid" if self.gate_reid_fallback else "block"

        # 5. At least one jersey not confident -> ReID.
        return "reid" if self.gate_reid_fallback else "block"


# ---------------------------------------------------------------------------
# Ablation conditions
# ---------------------------------------------------------------------------

ABLATIONS = [
    # name, gate-overrides
    ("baseline",                   {}),
    ("no_jersey_conflict_block",   {"jersey_conflict_block": False}),
    ("no_jersey_team_merge",       {"jersey_team_merge":     False}),
    ("no_team_conflict_block",     {"team_conflict_block":   False}),
    ("no_jersey_logic",            {"jersey_conflict_block": False,
                                    "jersey_team_merge":     False}),
    ("no_reid_fallback",           {"reid_fallback":         False}),
    ("reid_only",                  {"jersey_conflict_block": False,
                                    "jersey_team_merge":     False,
                                    "team_conflict_block":   False}),
    ("no_temporal_overlap",        {"temporal_overlap":      False}),
    ("no_conflict_blocks",         {"jersey_conflict_block": False,
                                "team_conflict_block":   False}),
]

PRIMARY_METRIC = "HOTA"
CACHE_DIR = settings.OUTPUT_ROOT / "cache"
EVAL_DIR  = settings.PROJECT_ROOT / "evaluation"


# ---------------------------------------------------------------------------
# Helpers (lifted from tune_decision_merger.py to keep this script standalone)
# ---------------------------------------------------------------------------

def load_attribute_cache(sequence: str) -> dict:
    cache_path = CACHE_DIR / f"cache_attributes_{sequence}.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Attribute cache missing: {cache_path}\n"
            "Run main.py at least once to generate it."
        )
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def make_splitter_config() -> SplitterConfig:
    known_fields = set(SplitterConfig.__dataclass_fields__.keys())
    known = {k: v for k, v in settings.SPLITTER.items() if k in known_fields}
    extra = {k: v for k, v in settings.SPLITTER.items() if k not in known_fields}
    cfg = SplitterConfig(**known)
    for k, v in extra.items():
        setattr(cfg, k, v)
    return cfg


def split_all_sequences(sequences: list) -> dict:
    splitter_cfg = make_splitter_config()
    split_cache = {}
    print("=== Splitting all sequences (one-time) ===")
    for seq in sequences:
        tracklets = load_attribute_cache(seq)
        split_cache[seq] = split_tracklets(tracklets, splitter_cfg)
    print(f"=== Splitting done — {len(split_cache)} sequences cached ===\n")
    return split_cache


def parse_summary(method_name: str) -> dict:
    summary_path = EVAL_DIR / "SNPT" / method_name / "pedestrian_summary.txt"
    if not summary_path.exists():
        return {}
    lines = summary_path.read_text().strip().splitlines()
    if len(lines) < 2:
        return {}
    headers = lines[0].split()
    values  = lines[1].split()
    return {h: float(v) for h, v in zip(headers, values)}


def run_condition(sequences, split_cache, name, gate_overrides, reid_threshold, eval_split):
    method_name = f"dm_ablate_{name}"

    merger_kwargs = dict(
        reid_threshold              = reid_threshold,
        jersey_entropy_threshold    = 0.01,
        min_jersey_predictions      = 20,
        team_confidence_threshold   = 0.6,
        team_consistency_threshold  = 0.9,
    )
    merger = AblatableDecisionMerger(**merger_kwargs, **gate_overrides)

    for sequence in sequences:
        paths = build_paths(sequence)
        tracklets = copy.deepcopy(split_cache[sequence])
        merged = merger.merge(tracklets)
        save_mot_file_for_sn_trackeval(
            tracklets_dict=merged,
            output_path=paths.evaluation_path,
            sequence_name=sequence,
            method_name=method_name,
        )

    run_evaluation(method_name, eval_split)
    return method_name, parse_summary(method_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Gate ablation for DecisionMerger.")
    ap.add_argument("--split", default=settings.EVAL_SPLIT, choices=["train", "val", "test"],
                    help="Evaluation split (default: settings.EVAL_SPLIT).")
    ap.add_argument("--reid", type=float, default=0.4,
                    help="ReID distance threshold (default: 0.4).")
    args = ap.parse_args()

    sequences = sorted([d.name for d in settings.DATA_ROOT.iterdir() if d.is_dir()])
    print(f"Sequences ({len(sequences)}): {sequences[0]} ... {sequences[-1]}")
    print(f"Eval split: {args.split}   ReID threshold: {args.reid}\n")

    split_cache = split_all_sequences(sequences)

    results = []
    for idx, (name, overrides) in enumerate(ABLATIONS, 1):
        print(f"\n[{idx}/{len(ABLATIONS)}] {name}")
        print(f"  disabled: {', '.join(k for k in overrides) or '(none)'}")
        try:
            method_name, metrics = run_condition(
                sequences, split_cache, name, overrides, args.reid, args.split
            )
            score = metrics.get(PRIMARY_METRIC, float("nan"))
            print(f"  -> {PRIMARY_METRIC} = {score:.4f}")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            metrics = {}
            method_name = f"dm_ablate_{name}"

        row = {
            "condition":   name,
            "method_name": method_name,
            "disabled":    ";".join(k for k in overrides),
            **metrics,
        }
        results.append(row)

    # ----- Save CSV -----
    out_path = settings.OUTPUT_ROOT / "decision_merger_gate_ablation.csv"
    if results:
        # Union of keys across rows so missing metrics don't break the writer.
        fieldnames = []
        seen = set()
        for r in results:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {out_path}")

    # ----- Pretty print with deltas relative to baseline -----
    baseline = next((r for r in results if r["condition"] == "baseline"), None)
    base_hota = baseline.get(PRIMARY_METRIC) if baseline else None

    print(f"\n{'=' * 80}")
    print(f"Gate ablation summary (sorted as defined; deltas vs. baseline {PRIMARY_METRIC})")
    print(f"{'=' * 80}")
    print(f"  {'condition':<28} {'HOTA':>8} {'AssA':>8} {'DetA':>8} {'IDF1':>8} {'ΔHOTA':>8}")
    for r in results:
        hota = r.get("HOTA", float("nan"))
        assa = r.get("AssA", float("nan"))
        deta = r.get("DetA", float("nan"))
        idf1 = r.get("IDF1", float("nan"))
        delta = (hota - base_hota) if (base_hota is not None and hota == hota) else float("nan")
        print(f"  {r['condition']:<28} {hota:>8.4f} {assa:>8.4f} {deta:>8.4f}"
              f" {idf1:>8.4f} {delta:>+8.4f}")


if __name__ == "__main__":
    main()
