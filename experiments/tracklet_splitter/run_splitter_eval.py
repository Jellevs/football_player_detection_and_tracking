#!/usr/bin/env python3
"""
Run intrinsic_metrics.py for every splitter variant and write a combined
comparison table to evaluation/splitter_comparison.csv.

Usage (from repo root):
    python run_splitter_eval.py

GT is read from SoccerNet test split; baseline (pre-split) tracklets are used
as the --input so that IDS-Recall and IDS-Precision can be computed.
"""

import subprocess
import sys
import csv
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────

DATA_SPLIT = "test"
REPO = Path(__file__).parent.parent.parent  # tracklet_splitter_scratch/
GT_ROOT = Path(fr"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\soccernet-player-tracking\{DATA_SPLIT}")
EVAL_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\evaluation\SNPT")
INPUT_DIR = EVAL_ROOT / "baseline" / "data"         # pre-split baseline tracklets
# INPUT_DIR = EVAL_ROOT / "splitters" / "test" / "splitter_temporalreid" / "data"         # spatio temporal reid splitted tracklets

SEQMAP   = REPO / "evaluation" / "seqmaps" / f"SNPT-{DATA_SPLIT}.txt"
SCRIPT   = REPO / "experiments" / "tracklet_splitter" / "intrinsic_metrics.py"
OUT_CSV  = REPO / "experiments" / "tracklet_splitter" / "output" / "gta_contiguous.csv"

# ── splitters to evaluate ──────────────────────────────────────────────────────
# Each tuple is (display_name, subdir relative to EVAL_ROOT).
SPLITTERS = [
    ('gta', "gta/DeepEIoU_soccernet_Split_contiguous_eps0.6_minSamples5_K3_minLen100")
    # ("baseline",   "baseline/data" ),
    # ("gta",        "splitters/test/splitter_gta/data"),
    # ("temporalreid",        "splitters/test/splitter_temporalreid/data"),

    # ("jersey",        "splitters/test/wo_reid_splitter/splitter_jersey/data"),
    # ("team",        "splitters/test/wo_reid_splitter/splitter_team/data"),
    # ("bbox",        "splitters/test/wo_reid_splitter/splitter_bbox/data"),
    # ("trajectory",        "splitters/test/wo_reid_splitter/splitter_traj/data"),
    # ("jersey+team",        "splitters/test/wo_reid_splitter/splitter_jersey_team/data"),
    # ("jersey+team+bbox",        "splitters/test/wo_reid_splitter/splitter_jersey_team_bbox/data"),
    # ("temporalreid+jersey+team+bbox",        "splitters/test/wo_reid_splitter/splitter_temporalreid_jersey_team_bbox/data"),
    # ("jersey+team+bbox+trajectory",        "splitters/test/wo_reid_splitter/splitter_jersey_team_bbox_traj/data"),
    # ("temporalreid_jersey_team_bbox_trajectory",   "splitters/test/wo_reid_splitter/splitter_temporalreid_jersey_team_bbox_traj/data" )



    # ("jersey",        "splitters/test/w_reid_splitter/splitter_jersey/data"),
    # ("team",        "splitters/test/w_reid_splitter/splitter_team/data"),
    # ("bbox",        "splitters/test/w_reid_splitter/splitter_bbox/data"),
    # ("trajectory",        "splitters/test/w_reid_splitter/splitter_traj/data"),
    # ("jersey+team",        "splitters/test/w_reid_splitter/splitter_jersey_team/data"),
    # ("jersey+team+bbox",        "splitters/test/w_reid_splitter/splitter_jersey_team_bbox/data"),
    # ("jersey+team+bbox+trajectory",        "splitters/test/w_reid_splitter/splitter_jersey_team_bbox_traj/data"),

]

TOLERANCE = 10   # frames
IOU       = 0.5


def run_one(name: str, pred_dir: Path) -> dict | None:
    per_seq_csv = EVAL_ROOT / f"_tmp_{name.replace(' ', '_')}.csv"
    cmd = [
        sys.executable, str(SCRIPT),
        "--pred",      str(pred_dir),
        "--gt",        str(GT_ROOT),
        "--input",     str(INPUT_DIR),
        "--seqmap",    str(SEQMAP),
        "--tolerance", str(TOLERANCE),
        "--iou",       str(IOU),
        "--out",       str(per_seq_csv),
        "--quiet",
    ]
    print(f"  evaluating: {name} ...", end=" ", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("FAILED")
        print(result.stderr[-800:])
        return None

    # Parse aggregate block from stdout. intrinsic_metrics.py prints each
    # aggregate entry indented by two spaces ("  key: value"); anything that
    # is not indented (e.g. the trailing "Wrote per-sequence results to ...")
    # terminates the block. We also whitelist the expected keys so any future
    # stray log line can't contaminate the CSV.
    AGG_KEYS = {
        "num_sequences", "num_tracklets",
        "mean_purity_macro", "mean_purity_weighted",
        "num_true_switches", "num_predicted_splits",
        "tp", "fp", "fn",
        "ids_recall", "ids_precision", "ids_f1",
    }
    agg = {"splitter": name}
    in_agg = False
    for line in result.stdout.splitlines():
        if "== Aggregate ==" in line:
            in_agg = True
            continue
        if not in_agg:
            continue
        # Aggregate entries are indented; any other line ends the block.
        if not line.startswith("  "):
            in_agg = False
            continue
        stripped = line.strip()
        if not stripped or ":" not in stripped:
            continue
        k, _, v = stripped.partition(":")
        k = k.strip(); v = v.strip()
        if k not in AGG_KEYS:
            continue
        try:
            agg[k] = float(v)
        except ValueError:
            agg[k] = v
    print("done")
    return agg


def main():
    all_rows = []
    for display_name, subdir in SPLITTERS:
        pred_dir = EVAL_ROOT / subdir
        if not pred_dir.exists():
            print(f"  [skip] {display_name} — {pred_dir} not found")
            continue
        row = run_one(display_name, pred_dir)
        if row:
            all_rows.append(row)

    if not all_rows:
        print("No results collected.")
        return

    # ── print table ──────────────────────────────────────────────────────────
    cols = ["splitter", "mean_purity_macro", "mean_purity_weighted",
            "ids_precision", "ids_recall", "ids_f1",
            "num_predicted_splits", "num_true_switches", "num_tracklets"]
    present = [c for c in cols if any(c in r for r in all_rows)]

    header_fmt = "{:<22}" + "{:>12}" * (len(present) - 1)
    row_fmt    = "{:<22}" + "{:>12}" * (len(present) - 1)

    print("\n" + "=" * (22 + 12 * (len(present) - 1)))
    print(header_fmt.format(*present))
    print("-" * (22 + 12 * (len(present) - 1)))
    for r in all_rows:
        vals = []
        for c in present:
            v = r.get(c, "—")
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        print(row_fmt.format(*vals))
    print("=" * (22 + 12 * (len(present) - 1)))

    # ── write CSV ─────────────────────────────────────────────────────────────
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    all_keys = list(dict.fromkeys(k for r in all_rows for k in r))
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved comparison to {OUT_CSV}")


if __name__ == "__main__":
    main()
