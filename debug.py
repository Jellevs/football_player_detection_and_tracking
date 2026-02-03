import torch
import pickle
import numpy as np
from pathlib import Path
from collections import Counter

from utils.config import Paths, TrackerConfig, JerseyPredictorConfig, SplitterConfig
from utils.data_utils import load_images, organize_detections_by_track
from detect_and_track.detect_and_track import detect_and_track
from detect_and_track.trackers.Deep_EIoU import DeepEIOUTracker
from attributes.attributes import predict_attributes
from tracklets.split_tracklets import split_tracklets


def debug_jersey_numbers(sequence, data_root, output_root, weights_root, device):
    """Debug jersey number predictions in tracklets"""
    
    # Same setup as main.py
    tracker_cfg = TrackerConfig(
        track_thresh=0.6,
        track_low_thresh=0.3,
        new_track_thresh=0.4,
        track_buffer=60,
        match_thresh=0.8,
        proximity_thresh=0.5,
        appearance_thresh=0.2,
        with_reid=True,
        reid_model_name="osnet_x0_25",
        frame_rate=25
    )
    
    paths = Paths(
        img_path=data_root / sequence / "img1",
        gt_detections_path=data_root / sequence / "Labels-GameState.json",
        output_path=output_root,
        cache_path=output_root / "cache",
        legibility_model_path=weights_root / "jersey_weights" / "legibility" / "output.pth",
        reid_model_path=weights_root / "jersey_weights" / "reid" / "osnet_x0_25_msmt17.pt",
        parseq_model_path=weights_root / "jersey_weights" / "parseq" / "parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt",
        centroid_reid_path=weights_root / "jersey_weights" / "centroid_reid" / "market1501_resnet50_256_128_epoch_120.ckpt",
        siglip_model_path=weights_root / "team_weights" / "siglip",
        vitpose_model_path=weights_root / "jersey_weights" / "vitpose",
        sequence=sequence
    )
    
    jersey_cfg = JerseyPredictorConfig(
        use_legibility=True,
        use_reid_filter=True,
        use_pose_cropper=True,
        legibility_arch="resnet34",
        legibility_threshold=0.6,
        reid_threshold_std=0.5,
    )
    
    splitter_cfg = SplitterConfig(
        jersey_min_fragment=20,
        jersey_min_persistence=5,
        jersey_lookahead=20,
        jersey_lookback=300,
        jersey_min_pixel_jump=10,
        jersey_entropy_threshold=0.2,
        team_min_persistence=5,
        team_min_fragment=10,
        team_lookahead=50,
    )
    
    # Run pipeline
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
    attributes_tracklets = predict_attributes(images, tracklets, paths, jersey_cfg, device)
    splitted_tracklets = split_tracklets(attributes_tracklets, splitter_cfg)
    
    # ========== ANALYZE JERSEY NUMBERS ==========
    
    print("\n" + "="*80)
    print("JERSEY NUMBER ANALYSIS")
    print("="*80)
    
    # Replace the BEFORE SPLITTING section with:
    print(f"\n{'='*80}")
    print(f"BEFORE SPLITTING ({len(attributes_tracklets)} tracklets)")
    print(f"{'='*80}")

    tracklets_with_jersey = 0
    for track_id, tracklet in attributes_tracklets.items():
        jerseys = tracklet.pred_attributes.get('jerseys', [])
        entropies = tracklet.pred_attributes.get('jersey_entropies', [])
        
        # Count valid predictions
        valid_jerseys = [j for j in jerseys if not (isinstance(j, float) and np.isnan(j))]
        
        has_jersey = len(valid_jerseys) > 0
        if has_jersey:
            tracklets_with_jersey += 1
        
        # Get jersey mode if available
        if valid_jerseys:
            jersey_mode = max(set(valid_jerseys), key=valid_jerseys.count)
            consistency = sum(1 for j in valid_jerseys if j == jersey_mode) / len(valid_jerseys)
            avg_entropy = np.mean([e for j, e in zip(jerseys, entropies) if not (isinstance(j, float) and np.isnan(j))])
        else:
            jersey_mode = -1
            consistency = 0.0
            avg_entropy = 1.0
        
        pct = (len(valid_jerseys)/len(jerseys)*100) if len(jerseys) > 0 else 0.0
        
        print(f"Tracklet {track_id}: {len(tracklet.frames)} frames | "
            f"Valid: {len(valid_jerseys)}/{len(jerseys)} ({pct:.1f}%) | "
            f"Jersey: {jersey_mode} (conf={consistency:.2f}, entropy={avg_entropy:.4f})")

    print(f"\nSummary: {tracklets_with_jersey}/{len(attributes_tracklets)} tracklets "
        f"have at least one jersey prediction ({tracklets_with_jersey/len(attributes_tracklets)*100:.1f}%)")

    # Replace the AFTER SPLITTING section with:
    print(f"\n{'='*80}")
    print(f"AFTER SPLITTING ({len(splitted_tracklets)} fragments)")
    print(f"{'='*80}")

    fragments_with_jersey = 0
    for track_id, tracklet in splitted_tracklets.items():
        jerseys = tracklet.pred_attributes.get('jerseys', [])
        entropies = tracklet.pred_attributes.get('jersey_entropies', [])
        
        valid_jerseys = [j for j in jerseys if not (isinstance(j, float) and np.isnan(j))]
        
        has_jersey = len(valid_jerseys) > 0
        if has_jersey:
            fragments_with_jersey += 1
        
        if valid_jerseys:
            jersey_mode = max(set(valid_jerseys), key=valid_jerseys.count)
            consistency = sum(1 for j in valid_jerseys if j == jersey_mode) / len(valid_jerseys)
            avg_entropy = np.mean([e for j, e in zip(jerseys, entropies) if not (isinstance(j, float) and np.isnan(j))])
        else:
            jersey_mode = -1
            consistency = 0.0
            avg_entropy = 1.0
        
        pct = (len(valid_jerseys)/len(jerseys)*100) if len(jerseys) > 0 else 0.0
        
        print(f"Fragment {track_id} (parent={tracklet.parent_id}): {len(tracklet.frames)} frames | "
            f"Valid: {len(valid_jerseys)}/{len(jerseys)} ({pct:.1f}%) | "
            f"Jersey: {jersey_mode} (conf={consistency:.2f}, entropy={avg_entropy:.4f})")

    print(f"\nSummary: {fragments_with_jersey}/{len(splitted_tracklets)} fragments "
        f"have at least one jersey prediction ({fragments_with_jersey/len(splitted_tracklets)*100:.1f}%)")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    SEQUENCE = "SNGS-123"
    DATA_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\data\SoccerNetGS\test")
    OUTPUT_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output")
    WEIGHTS_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\weights")
    
    debug_jersey_numbers(SEQUENCE, DATA_ROOT, OUTPUT_ROOT, WEIGHTS_ROOT, device)