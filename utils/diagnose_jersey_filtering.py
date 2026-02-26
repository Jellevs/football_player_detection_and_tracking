"""
Diagnose jersey prediction filtering.

For each tracklet, saves crops at each filtering stage:
  output/<seq>/jersey_filter_debug/track_XXXX/
    stage0_all/         — all extracted full crops (before any filtering)
    stage1_reid_kept/   — kept after ReID filter
    stage1_reid_removed/— removed by ReID filter
    stage2_leg_kept/    — kept after legibility (these go to PARSeq)
    stage2_leg_removed/ — removed by legibility

Filename format: frame{XXXXX}_leg{0.XX}.jpg
                  so you can see the legibility score at a glance.

Usage from main.py (add after predict_attributes or instead of it):

    from utils.diagnose_jersey_filtering import diagnose_filtering
    diagnose_filtering(images, tracklets, paths, jersey_cfg, device, max_tracklets=10)

Or standalone (uses cache):
    python -m utils.diagnose_jersey_filtering --sequence SNMOT-116 --max_tracklets 10
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image


def diagnose_filtering(images, tracklets, paths, jersey_cfg, device, max_tracklets=10):
    """
    Run the jersey prediction pipeline with full diagnostic output.
    Saves crops at each filtering stage with legibility scores in filenames.
    """
    from attributes.jersey_number.jersey_number_predictor_parseq import JerseyNumberPredictorParseq

    # Build predictor (loads all models)
    jersey_predictor = JerseyNumberPredictorParseq(
        paths=paths,
        jersey_cfg=jersey_cfg,
        device=device
    )

    output_dir = Path(paths.output_path) / paths.sequence / "jersey_filter_debug"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort by length, take longest tracklets
    sorted_tracklets = sorted(tracklets.items(), key=lambda x: len(x[1].frames), reverse=True)
    if max_tracklets:
        sorted_tracklets = sorted_tracklets[:max_tracklets]

    summary_lines = []
    summary_lines.append(f"{'Track':>6} {'Frames':>7} {'Crops':>6} {'ReID':>5} {'Legib':>6} {'PARSeq':>7}  Notes")
    summary_lines.append("-" * 80)

    for track_id, tracklet in tqdm(sorted_tracklets, desc="Diagnosing filtering"):
        track_dir = output_dir / f"track_{track_id:04d}"

        # ── Stage 0: Extract all crops ──
        full_crops, torso_crops, indices = jersey_predictor.extract_crops(images, tracklet)
        n_extracted = len(full_crops)

        _save_crops(track_dir / "stage0_all", full_crops, torso_crops, indices, tracklet)

        if not full_crops:
            summary_lines.append(f"{track_id:>6} {len(tracklet.frames):>7} {0:>6} {'-':>5} {'-':>6} {'-':>7}  No crops extracted")
            continue

        # ── Stage 1: ReID filtering ──
        pre_reid = len(full_crops)
        if jersey_predictor.reid_filter and tracklet.embeddings:
            # Get embeddings for these indices
            embeddings = [tracklet.embeddings[i] for i in indices]
            
            # Run filter but also capture what was removed
            kept_mask = jersey_predictor.reid_filter.iterative_outlier_removal(
                jersey_predictor.reid_filter.extract_embeddings(full_crops)
            )
            
            reid_kept_full = [c for c, k in zip(full_crops, kept_mask) if k]
            reid_kept_torso = [c for c, k in zip(torso_crops, kept_mask) if k]
            reid_kept_indices = [i for i, k in zip(indices, kept_mask) if k]
            reid_removed_full = [c for c, k in zip(full_crops, kept_mask) if not k]
            reid_removed_torso = [c for c, k in zip(torso_crops, kept_mask) if not k]
            reid_removed_indices = [i for i, k in zip(indices, kept_mask) if not k]
            
            _save_crops(track_dir / "stage1_reid_kept", reid_kept_full, reid_kept_torso, reid_kept_indices, tracklet)
            _save_crops(track_dir / "stage1_reid_removed", reid_removed_full, reid_removed_torso, reid_removed_indices, tracklet)
            
            full_crops = reid_kept_full
            torso_crops = reid_kept_torso
            indices = reid_kept_indices
        
        n_after_reid = len(full_crops)

        # ── Stage 2: Legibility filtering (with scores) ──
        n_after_legibility = n_after_reid
        if jersey_predictor.legibility_predictor and full_crops:
            legible_flags, legibility_scores = jersey_predictor.legibility_predictor.predict_batch(full_crops)
            
            leg_kept_full = []
            leg_kept_torso = []
            leg_kept_indices = []
            leg_removed_full = []
            leg_removed_torso = []
            leg_removed_indices = []
            leg_kept_scores = []
            leg_removed_scores = []
            
            for j, (flag, score) in enumerate(zip(legible_flags, legibility_scores)):
                if flag:
                    leg_kept_full.append(full_crops[j])
                    leg_kept_torso.append(torso_crops[j])
                    leg_kept_indices.append(indices[j])
                    leg_kept_scores.append(score)
                else:
                    leg_removed_full.append(full_crops[j])
                    leg_removed_torso.append(torso_crops[j])
                    leg_removed_indices.append(indices[j])
                    leg_removed_scores.append(score)
            
            _save_crops(track_dir / "stage2_legibility_kept", leg_kept_full, leg_kept_torso, 
                       leg_kept_indices, tracklet, scores=leg_kept_scores, score_name="leg")
            _save_crops(track_dir / "stage2_legibility_removed", leg_removed_full, leg_removed_torso,
                       leg_removed_indices, tracklet, scores=leg_removed_scores, score_name="leg")
            
            n_after_legibility = len(leg_kept_full)
            full_crops = leg_kept_full
            torso_crops = leg_kept_torso
            indices = leg_kept_indices

        # ── Stage 3: PARSeq prediction ──
        n_parseq = 0
        if torso_crops:
            jersey_preds, confs, entropies, raw_probs = jersey_predictor.predict_jersey(torso_crops)
            n_parseq = sum(1 for j in jersey_preds if not (isinstance(j, float) and np.isnan(j)))
            
            # Save PARSeq results on the kept crops
            _save_crops_with_predictions(
                track_dir / "stage3_parseq_results",
                full_crops, torso_crops, indices, tracklet,
                jersey_preds, confs, entropies
            )

        # Notes
        notes = []
        if n_extracted == 0:
            notes.append("no pose")
        if n_after_reid < n_extracted * 0.5:
            notes.append(f"ReID aggressive ({n_extracted - n_after_reid} removed)")
        if n_after_legibility < n_after_reid * 0.3:
            notes.append(f"Legibility aggressive ({n_after_reid - n_after_legibility} removed)")
        if n_after_legibility > 0 and n_parseq == 0:
            notes.append("PARSeq decoded nothing")

        summary_lines.append(
            f"{track_id:>6} {len(tracklet.frames):>7} {n_extracted:>6} "
            f"{n_after_reid:>5} {n_after_legibility:>6} {n_parseq:>7}  {'; '.join(notes)}"
        )

    # Save summary
    summary_path = output_dir / "filtering_summary.txt"
    with open(summary_path, 'w') as f:
        for line in summary_lines:
            f.write(line + "\n")
            print(line)
    
    print(f"\nDiagnostics saved to: {output_dir}")


def _save_crops(save_dir, full_crops, torso_crops, indices, tracklet, scores=None, score_name="score"):
    """Save full and torso crops to disk."""
    if not full_crops:
        return
    
    save_dir = Path(save_dir)
    full_dir = save_dir / "full"
    torso_dir = save_dir / "torso"
    full_dir.mkdir(parents=True, exist_ok=True)
    torso_dir.mkdir(parents=True, exist_ok=True)

    for j, idx in enumerate(indices):
        frame_idx = tracklet.frames[idx]
        
        if scores is not None and j < len(scores):
            score_str = f"_{score_name}{scores[j]:.3f}"
        else:
            score_str = ""
        
        filename = f"frame{frame_idx:05d}{score_str}.jpg"
        
        if j < len(full_crops):
            crop_bgr = cv2.cvtColor(full_crops[j], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(full_dir / filename), crop_bgr)
        
        if j < len(torso_crops):
            torso_bgr = cv2.cvtColor(torso_crops[j], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(torso_dir / filename), torso_bgr)


def _save_crops_with_predictions(save_dir, full_crops, torso_crops, indices, tracklet,
                                  jerseys, confs, entropies):
    """Save crops with jersey prediction info in filename."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for j, idx in enumerate(indices):
        frame_idx = tracklet.frames[idx]
        
        jersey = jerseys[j] if j < len(jerseys) else np.nan
        conf = confs[j] if j < len(confs) else 0.0
        entropy = entropies[j] if j < len(entropies) else 1.0
        
        if isinstance(jersey, float) and np.isnan(jersey):
            jersey_str = "X"
        else:
            jersey_str = str(int(jersey))
        
        filename = f"frame{frame_idx:05d}_nr{jersey_str}_conf{conf:.3f}_ent{entropy:.3f}.jpg"
        
        if j < len(full_crops):
            crop_bgr = cv2.cvtColor(full_crops[j], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_dir / filename), crop_bgr)


# ============================================================================
# Standalone usage
# ============================================================================
if __name__ == "__main__":
    import argparse
    import pickle
    import sys

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    import settings
    from utils.build import build_paths, build_configs
    from utils.data_utils import load_images

    parser = argparse.ArgumentParser(description="Diagnose jersey prediction filtering")
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--max_tracklets", type=int, default=10)
    args = parser.parse_args()

    device, _, jersey_cfg, _, _ = build_configs()
    paths = build_paths(args.sequence)
    images = load_images(img_dir=paths.img_path)

    # Load tracklets from detection cache (before attributes)
    cache_path = paths.set_cache_path("attributes", args.sequence)
    det_cache = paths.set_cache_path("detections", args.sequence)
    
    # Try to load raw tracklets (pre-attribute) 
    if det_cache.exists():
        print(f"Loading tracklets from {det_cache}")
        with open(det_cache, 'rb') as f:
            tracklets = pickle.load(f)
    elif cache_path.exists():
        print(f"Loading attributed tracklets from {cache_path}")
        with open(cache_path, 'rb') as f:
            tracklets = pickle.load(f)
    else:
        # Build from scratch
        from utils.data_utils import organize_detections_by_track
        from detect_and_track.detect_and_track import detect_and_track
        from detect_and_track.trackers.Deep_EIoU import DeepEIOUTracker
        
        tracker_cfg = build_configs()[1]
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
        tracked = detect_and_track(images, tracker, paths)
        tracklets = organize_detections_by_track(tracked)

    print(f"Loaded {len(tracklets)} tracklets")

    diagnose_filtering(
        images=images,
        tracklets=tracklets,
        paths=paths,
        jersey_cfg=jersey_cfg,
        device=device,
        max_tracklets=args.max_tracklets,
    )