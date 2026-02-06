# run_full_evaluation.py
"""
Run the full tracking pipeline on ALL sequences and evaluate with sn-trackeval.

This script:
1. Processes all sequences in a split (train/test)
2. Runs: detection -> tracking -> attributes -> splitting -> merging
3. Exports results in MOT format
4. Runs sn-trackeval to get HOTA, CLEAR, Identity metrics
5. Compares baseline vs after_split vs after_merge

Usage:
    python run_full_evaluation.py --split train
    python run_full_evaluation.py --split test --max_sequences 5  # Test on 5 sequences first

Prerequisites:
    git clone https://github.com/SoccerNet/sn-trackeval.git
    pip install -e sn-trackeval/
"""

import torch
from pathlib import Path
from tqdm import tqdm
import argparse
import json

from utils.data_utils import load_images, organize_detections_by_track
from utils.config import Paths, TrackerConfig, JerseyPredictorConfig, SplitterConfig, MergerConfig
from utils.evaluate_soccernet_mot import setup_trackeval_structure, run_sn_trackeval
from detect_and_track.detect_and_track import detect_and_track
from detect_and_track.trackers.Deep_EIoU import DeepEIOUTracker
from attributes.attributes import predict_attributes
from tracklets.split_tracklets import split_tracklets
from tracklets.tracklet_merger import TrackletMerger


def process_single_sequence(sequence, data_root, output_root, weights_root, 
                            tracker_cfg, jersey_cfg, splitter_cfg, merger, device):
    """
    Process a single sequence through the full pipeline.
    
    Returns:
        Tuple of (original_tracklets, split_tracklets, merged_tracklets)
    """
    
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
    
    # Initialize tracker
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
    
    # Run pipeline
    images = load_images(img_dir=paths.img_path)
    tracked_detections = detect_and_track(images, tracker, paths)
    tracklets = organize_detections_by_track(tracked_detections)
    
    # Predict attributes
    attributes_tracklets = predict_attributes(images, tracklets, paths, jersey_cfg, device)
    
    # Split tracklets
    splitted_tracklets = split_tracklets(attributes_tracklets, splitter_cfg)
    
    # Merge tracklets
    if merger is not None:
        merged_tracklets = merger.merge_tracklets(splitted_tracklets)
    else:
        merged_tracklets = splitted_tracklets
    
    return tracklets, splitted_tracklets, merged_tracklets


def main(args):
    """Main function to process all sequences and evaluate."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Paths - EDIT THESE TO MATCH YOUR SETUP
    DATA_ROOT = Path(args.data_root)
    OUTPUT_ROOT = Path(args.output_root)
    WEIGHTS_ROOT = Path(args.weights_root)
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
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
    
    merger_cfg = MergerConfig(
        xgboost_model_path=str(WEIGHTS_ROOT / "xgboost" / "xgboost_baseline.json"),
        pca_model_path=str(OUTPUT_ROOT / "training_data" / "pca_models"),
        merge_threshold=0.5,
        linkage_method='single',
        max_temporal_gap = 100
    )
    
    # =========================================================================
    # GET ALL SEQUENCES
    # =========================================================================
    
    all_sequences = sorted([d.name for d in DATA_ROOT.iterdir() if d.is_dir()])
    
    if args.max_sequences is not None:
        all_sequences = all_sequences[:args.max_sequences]
        print(f"\nTEST MODE: Processing only {len(all_sequences)} sequences")
    
    print(f"\n{'='*80}")
    print(f"PROCESSING {len(all_sequences)} SEQUENCES FROM {args.split.upper()} SPLIT")
    print(f"{'='*80}")
    print(f"Data root: {DATA_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Sequences: {all_sequences[:5]}{'...' if len(all_sequences) > 5 else ''}")
    
    # =========================================================================
    # INITIALIZE MERGER (shared across sequences)
    # =========================================================================
    

    merger = TrackletMerger(merger_cfg=merger_cfg)

    
    # =========================================================================
    # PROCESS ALL SEQUENCES
    # =========================================================================
    
    all_baseline = {}
    all_split = {}
    all_merged = {}
    
    failed_sequences = []
    
    for i, sequence in enumerate(tqdm(all_sequences, desc="Processing sequences")):
        print(f"\n[{i+1}/{len(all_sequences)}] {sequence}")
        
        try:
            baseline, split, merged = process_single_sequence(
                sequence=sequence,
                data_root=DATA_ROOT,
                output_root=OUTPUT_ROOT,
                weights_root=WEIGHTS_ROOT,
                tracker_cfg=tracker_cfg,
                jersey_cfg=jersey_cfg,
                splitter_cfg=splitter_cfg,
                merger=merger,
                device=device
            )
            
            all_baseline[sequence] = baseline
            all_split[sequence] = split
            all_merged[sequence] = merged
            
            print(f"  ✓ {len(baseline)} -> {len(split)} -> {len(merged)} tracklets")
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed_sequences.append(sequence)
            continue
    
    # =========================================================================
    # SETUP TRACKEVAL STRUCTURE
    # =========================================================================
    
    print(f"\n{'='*80}")
    print("SETTING UP EVALUATION")
    print(f"{'='*80}")
    
    if len(all_baseline) == 0:
        print("ERROR: No sequences processed successfully!")
        return
    
    all_tracklets_by_stage = {
        "baseline": all_baseline,
        "after_split": all_split,
        "after_merge": all_merged,
    }
    
    data_dir = setup_trackeval_structure(
        all_tracklets_by_stage=all_tracklets_by_stage,
        output_dir=OUTPUT_ROOT,
        split=args.split
    )
    
    # =========================================================================
    # RUN SN-TRACKEVAL
    # =========================================================================
    
    sn_trackeval_path = Path(args.sn_trackeval_path) if args.sn_trackeval_path else None
    
    results = run_sn_trackeval(
        data_dir=data_dir,
        split=args.split,
        trackers_to_eval=["baseline", "after_split", "after_merge"],
        sn_trackeval_path=sn_trackeval_path
    )
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Processed: {len(all_baseline)}/{len(all_sequences)} sequences")
    
    if failed_sequences:
        print(f"Failed: {failed_sequences}")
    
    print(f"\nResults saved to: {data_dir / 'results'}")
    print(f"\nCheck the following files for detailed metrics:")
    print(f"  - {data_dir / 'results' / f'SoccerNet-{args.split}' / 'baseline' / 'pedestrian_summary.txt'}")
    print(f"  - {data_dir / 'results' / f'SoccerNet-{args.split}' / 'after_split' / 'pedestrian_summary.txt'}")
    print(f"  - {data_dir / 'results' / f'SoccerNet-{args.split}' / 'after_merge' / 'pedestrian_summary.txt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full evaluation pipeline")
    
    # Required paths - EDIT THESE DEFAULTS TO MATCH YOUR SETUP
    parser.add_argument("--data_root", type=str, 
                        default=r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\data\SoccerNetGS\test",
                        help="Path to data folder containing sequences")
    parser.add_argument("--output_root", type=str,
                        default=r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output",
                        help="Path to save outputs")
    parser.add_argument("--weights_root", type=str,
                        default=r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\weights",
                        help="Path to model weights")
    
    # Evaluation settings
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split: train, test, or challenge")
    parser.add_argument("--sn_trackeval_path", type=str, default=None,
                        help="Path to sn-trackeval repo (auto-detected if not specified)")
    
    # Debug/test options
    parser.add_argument("--max_sequences", type=int, default=None,
                        help="Limit number of sequences (for testing)")
    
    args = parser.parse_args()
    
    main(args)