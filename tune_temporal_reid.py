"""
Grid search for TemporalReIDSplitter hyperparameters.

Runs the pipeline up to temporal_splitter.split_all() (no attribute prediction),
evaluates HOTA for each parameter combination, and prints a summary table.

Tunes on the validation set. Run on test only for final evaluation.

Usage:
    python tune_temporal_reid.py
"""

import re
import subprocess
import sys
from itertools import product
from pathlib import Path
from tqdm import tqdm

import settings
from utils.build import build_paths, build_configs
from utils.data_utils import load_images, organize_detections_by_track
from utils.evaluation_utils import save_mot_file_for_sn_trackeval
from detect_and_track.detect_and_track import detect_and_track
from detect_and_track.trackers.Deep_EIoU import DeepEIOUTracker
from tracklets.splitters.temporal_reid_splitter import TemporalReIDSplitter

# ---------------------------------------------------------------------------
# Grid search parameters
# ---------------------------------------------------------------------------
MIN_GAP_FRAMES_VALUES  = [5, 10, 15, 20, 30]
REID_THRESHOLD_VALUES  = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


def run_evaluation_capture(method_name: str) -> dict:
    """Run evaluation and parse HOTA/DetA/AssA/IDF1 from stdout."""
    eval_root = settings.PROJECT_ROOT / "evaluation"

    cmd = [
        sys.executable,
        str(settings.PROJECT_ROOT / "sn-trackeval" / "scripts" / "run_mot_challenge.py"),
        "--BENCHMARK", "SNMOT",
        "--SPLIT_TO_EVAL", f"SNMOT-{settings.EVAL_SPLIT}",
        "--GT_FOLDER", str(settings.DATA_ROOT),
        "--TRACKERS_FOLDER", str(eval_root / "SNPT"),
        "--TRACKERS_TO_EVAL", method_name,
        "--SEQMAP_FILE", str(eval_root / "seqmaps" / f"SNPT-{settings.EVAL_SPLIT}.txt"),
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--DO_PREPROC", "False",
        "--USE_PARALLEL", "False",
        "--TRACKER_SUB_FOLDER", "data",
        "--SKIP_SPLIT_FOL", "True",
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        print(f"  [EVAL OUTPUT] {result.stdout[:500]}")
        return {}
    output = result.stdout

    # Output format (tab-separated columns after header):
    # HOTA: method-pedestrian   HOTA   DetA   AssA   DetRe  DetPr  AssRe  AssPr  LocA ...
    # COMBINED                  83.62  93.60  74.73  ...
    # CLEAR: method-pedestrian  MOTA   MOTP   ...
    # COMBINED                  99.16  94.24  ...
    # Identity: method-ped      IDF1   IDR    IDP    ...
    # COMBINED                  83.80  ...
    metrics = {}
    lines = output.split("\n")
    for i, line in enumerate(lines):
        # Find header lines, then parse the COMBINED line right after
        if line.strip().startswith("HOTA:") and "HOTA" in line and "DetA" in line:
            # Next line with COMBINED has the values
            for j in range(i + 1, min(i + 3, len(lines))):
                if "COMBINED" in lines[j]:
                    vals = lines[j].split()
                    # vals = ["COMBINED", hota, deta, assa, ...]
                    if len(vals) >= 4:
                        metrics["HOTA"] = float(vals[1])
                        metrics["DetA"] = float(vals[2])
                        metrics["AssA"] = float(vals[3])
                    break

        if line.strip().startswith("CLEAR:") and "MOTA" in line:
            for j in range(i + 1, min(i + 3, len(lines))):
                if "COMBINED" in lines[j]:
                    vals = lines[j].split()
                    if len(vals) >= 2:
                        metrics["MOTA"] = float(vals[1])
                    # IDSW is at index 13 (0-indexed)
                    if len(vals) >= 14:
                        metrics["IDSW"] = float(vals[13])
                    break

        if line.strip().startswith("Identity:") and "IDF1" in line:
            for j in range(i + 1, min(i + 3, len(lines))):
                if "COMBINED" in lines[j]:
                    vals = lines[j].split()
                    if len(vals) >= 2:
                        metrics["IDF1"] = float(vals[1])
                    break

    return metrics


def main():
    device, tracker_cfg, jersey_cfg, splitter_cfg, merger_cfg = build_configs()
    sequences = sorted([d.name for d in settings.DATA_ROOT.iterdir() if d.is_dir()])

    # Pre-run detect_and_track for all sequences (cached, so this is fast)
    print("Pre-loading tracklets for all sequences...")
    seq_tracklets = {}
    for sequence in tqdm(sequences, desc="Loading"):
        paths = build_paths(sequence)
        tracker = DeepEIOUTracker(
            track_thresh=tracker_cfg.track_thresh,
            track_low_thresh=tracker_cfg.track_low_thresh,
            new_track_thresh=tracker_cfg.new_track_thresh,
            track_buffer=tracker_cfg.track_buffer,
            match_thresh=tracker_cfg.match_thresh,
            proximity_thresh=tracker_cfg.proximity_thresh,
            appearance_thresh=tracker_cfg.appearance_thresh,
            with_reid=tracker_cfg.with_reid,
            reid_model_name=tracker_cfg.reid_model_name,
            reid_model_path=str(paths.reid_model_path),
            frame_rate=tracker_cfg.frame_rate,
        )
        images = load_images(img_dir=paths.img_path)
        tracked_detections = detect_and_track(images, tracker, paths)
        tracklets = organize_detections_by_track(tracked_detections)
        seq_tracklets[sequence] = tracklets

    # Grid search
    param_grid = list(product(MIN_GAP_FRAMES_VALUES, REID_THRESHOLD_VALUES))
    results = []

    print(f"\nGrid search: {len(param_grid)} combinations")
    print(f"{'='*70}")

    for min_gap, reid_thresh in param_grid:
        method_name = f"tune_gap{min_gap}_thresh{reid_thresh:.2f}"

        # Run splitter for all sequences with these params
        for sequence in sequences:
            paths = build_paths(sequence)
            tracklets = seq_tracklets[sequence]

            splitter = TemporalReIDSplitter(
                min_gap_frames=min_gap,
                reid_threshold=reid_thresh,
            )
            split_result = splitter.split_all(tracklets)

            save_mot_file_for_sn_trackeval(
                tracklets_dict=split_result,
                output_path=paths.evaluation_path,
                sequence_name=sequence,
                method_name=method_name,
            )

        # Evaluate
        try:
            metrics = run_evaluation_capture(method_name)
            print(metrics)
            hota = metrics.get("HOTA", 0.0)
            print(hota)
            deta = metrics.get("DetA", 0.0)
            assa = metrics.get("AssA", 0.0)
            idf1 = metrics.get("IDF1", 0.0)
            idsw = metrics.get("IDSW", 0.0)
        except Exception as e:
            print(f"  [ERROR] {method_name}: {e}")
            hota = deta = assa = idf1 = idsw = 0.0

        results.append({
            "min_gap": min_gap,
            "reid_threshold": reid_thresh,
            "HOTA": hota,
            "DetA": deta,
            "AssA": assa,
            "IDF1": idf1,
            "IDSW": idsw,
        })

        print(f"  gap={min_gap:3d}  thresh={reid_thresh:.2f}  →  "
              f"HOTA={hota:.3f}  AssA={assa:.3f}  IDF1={idf1:.3f}  IDSW={idsw:.0f}")

    # Summary table sorted by HOTA
    print(f"\n{'='*70}")
    print(f"  RESULTS (sorted by HOTA)")
    print(f"{'='*70}")
    print(f"  {'gap':>4}  {'thresh':>6}  {'HOTA':>7}  {'DetA':>7}  {'AssA':>7}  {'IDF1':>7}  {'IDSW':>6}")
    print(f"  {'-'*52}")

    for r in sorted(results, key=lambda x: x["HOTA"], reverse=True):
        print(f"  {r['min_gap']:4d}  {r['reid_threshold']:6.2f}  "
              f"{r['HOTA']:7.3f}  {r['DetA']:7.3f}  {r['AssA']:7.3f}  "
              f"{r['IDF1']:7.3f}  {r['IDSW']:6.0f}")

    # Save results to CSV
    csv_path = Path("tune_temporal_reid_results.csv")
    with open(csv_path, "w") as f:
        f.write("min_gap,reid_threshold,HOTA,DetA,AssA,IDF1,IDSW\n")
        for r in sorted(results, key=lambda x: x["HOTA"], reverse=True):
            f.write(f"{r['min_gap']},{r['reid_threshold']},{r['HOTA']},"
                    f"{r['DetA']},{r['AssA']},{r['IDF1']},{r['IDSW']}\n")
    print(f"\nResults saved → {csv_path}")


if __name__ == "__main__":
    main()
