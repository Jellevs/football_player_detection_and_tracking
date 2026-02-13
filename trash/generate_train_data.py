# # generate_all_training_data.py

# import torch
# from pathlib import Path
# import pandas as pd
# from tqdm import tqdm

# from utils.data_utils import load_images, organize_detections_by_track
# from utils.config import Paths, TrackerConfig, JerseyPredictorConfig, SplitterConfig
# from detect_and_track.detect_and_track import detect_and_track
# from detect_and_track.trackers.Deep_EIoU import DeepEIOUTracker
# from attributes.attributes import predict_attributes
# from tracklets.split_tracklets import split_tracklets

# from training_data.fragment_aggregator import FragmentAggregator
# from training_data.pair_generator import PairGenerator
# from training_data.data_format import DataSaver
# from training_data.pca_reducer import PCAReducer


# def process_single_sequence(sequence, data_root, output_root, weights_root, device):
#     """
#     Process a single sequence and return aggregated fragments (WITHOUT PCA).
#     PCA will be applied later on ALL fragments from ALL sequences.
#     """
    
#     # Setup configs
#     tracker_cfg = TrackerConfig(
#         track_thresh=0.6,
#         track_low_thresh=0.3,
#         new_track_thresh=0.4,
#         track_buffer=60,
#         match_thresh=0.8,
#         proximity_thresh=0.5,
#         appearance_thresh=0.2,
#         with_reid=True,
#         reid_model_name="osnet_x0_25",
#         frame_rate=25
#     )
    
#     paths = Paths(
#         img_path=data_root / sequence / "img1",
#         gt_detections_path=data_root / sequence / "Labels-GameState.json",
#         output_path=output_root,
#         cache_path=output_root / "cache",
#         legibility_model_path=weights_root / "jersey_weights" / "legibility" / "output.pth",
#         reid_model_path=weights_root / "jersey_weights" / "reid" / "osnet_x0_25_msmt17.pt",
#         parseq_model_path=weights_root / "jersey_weights" / "parseq" / "parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt",
#         centroid_reid_path=weights_root / "jersey_weights" / "centroid_reid" / "market1501_resnet50_256_128_epoch_120.ckpt",
#         siglip_model_path=weights_root / "team_weights" / "siglip",
#         vitpose_model_path=weights_root / "jersey_weights" / "vitpose",
#         sequence=sequence
#     )
    
#     jersey_cfg = JerseyPredictorConfig(
#         use_legibility=True,
#         use_reid_filter=True,
#         use_pose_cropper=True,
#         legibility_arch="resnet34",
#         legibility_threshold=0.6,
#         reid_threshold_std=0.5,
#     )
    
#     splitter_cfg = SplitterConfig(
#         jersey_min_fragment=20,
#         jersey_min_persistence=5,
#         jersey_lookahead=20,
#         jersey_lookback=300,
#         jersey_min_pixel_jump=10,
#         jersey_entropy_threshold=0.2,
#         team_min_persistence=5,
#         team_min_fragment=10,
#         team_lookahead=50,
#     )
    
#     # Initialize tracker
#     tracker = DeepEIOUTracker(
#         track_thresh=tracker_cfg.track_thresh,
#         track_low_thresh=tracker_cfg.track_low_thresh,
#         new_track_thresh=tracker_cfg.new_track_thresh,
#         track_buffer=tracker_cfg.track_buffer,
#         match_thresh=tracker_cfg.match_thresh,
#         proximity_thresh=tracker_cfg.proximity_thresh,
#         appearance_thresh=tracker_cfg.appearance_thresh,
#         with_reid=tracker_cfg.with_reid,
#         reid_model_name=tracker_cfg.reid_model_name,
#         reid_model_path=str(paths.reid_model_path),
#         frame_rate=tracker_cfg.frame_rate,
#     )
    
#     # Run pipeline
#     images = load_images(img_dir=paths.img_path)
#     tracked_detections = detect_and_track(images, tracker, paths)
#     tracklets = organize_detections_by_track(tracked_detections)
#     attributes_tracklets = predict_attributes(images, tracklets, paths, jersey_cfg, device)
#     splitted_tracklets = split_tracklets(attributes_tracklets, splitter_cfg)
    
#     # Aggregate fragments (WITHOUT PCA)
#     aggregator = FragmentAggregator()
#     aggregated_fragments = []
    
#     for track_id, tracklet in splitted_tracklets.items():
#         agg = aggregator.aggregate_tracklet(tracklet)
#         if agg is not None:
#             # Add sequence name to metadata for tracking
#             agg['metadata']['sequence'] = sequence
#             aggregated_fragments.append(agg)
    
#     return aggregated_fragments


# def generate_all_training_data(sequences, data_root, output_root, weights_root, device):
#     """
#     Process all sequences, fit PCA on ALL data, then generate training pairs.
    
#     This is the CORRECT way to do it:
#     1. Process all sequences → collect all fragments
#     2. Fit PCA on ALL fragments from ALL sequences
#     3. Transform all fragments with fitted PCA
#     4. Generate pairs per sequence
#     5. Combine all pairs into one dataset
#     """
    
#     print(f"\n{'='*80}")
#     print(f"PROCESSING {len(sequences)} SEQUENCES")
#     print(f"{'='*80}\n")
    
#     # ========== PHASE 1: Process all sequences ==========
#     print("Phase 1: Processing all sequences and collecting fragments...")
#     print("(This will take a while - tracking, jersey detection, team classification)")
#     print()
    
#     all_fragments_by_sequence = {}
#     total_fragments = 0
    
#     for sequence in tqdm(sequences, desc="Processing sequences"):
#         try:
#             fragments = process_single_sequence(sequence, data_root, output_root, weights_root, device)
#             all_fragments_by_sequence[sequence] = fragments
#             total_fragments += len(fragments)
#             tqdm.write(f"  ✓ {sequence}: {len(fragments)} fragments")
#         except Exception as e:
#             tqdm.write(f"  ✗ {sequence}: ERROR - {e}")
#             all_fragments_by_sequence[sequence] = []
    
#     print(f"\nTotal fragments collected: {total_fragments}")
    
#     # ========== PHASE 2: Fit PCA on ALL fragments ==========
#     print(f"\n{'='*80}")
#     print("Phase 2: Fitting PCA on ALL fragments from ALL sequences")
#     print(f"{'='*80}")
    
#     # Flatten all fragments into single list
#     all_fragments = []
#     for sequence, fragments in all_fragments_by_sequence.items():
#         all_fragments.extend(fragments)
    
#     print(f"Fitting PCA on {len(all_fragments)} total fragments...")
    
#     # Fit PCA on everything
#     pca_reducer = PCAReducer(siglip_components=16, reid_components=8)
#     pca_reducer.fit(all_fragments)
    
#     # Save PCA models
#     pca_output_dir = output_root / "training_data" / "pca_models"
#     pca_reducer.save(pca_output_dir)
    
#     # ========== PHASE 3: Transform all fragments and generate pairs ==========
#     print(f"\n{'='*80}")
#     print("Phase 3: Transforming fragments and generating pairs")
#     print(f"{'='*80}\n")
    
#     pair_generator = PairGenerator(
#         max_temporal_gap=100,
#         include_all_positives=True,
#         negative_sampling_ratio=2.0
#     )
    
#     saver = DataSaver(output_root / "training_data")
    
#     all_pairs_dfs = []
#     total_pairs = 0
    
#     for sequence in tqdm(sequences, desc="Generating pairs"):
#         fragments = all_fragments_by_sequence.get(sequence, [])
        
#         if not fragments:
#             continue
        
#         # Transform this sequence's fragments
#         transformed_fragments = pca_reducer.transform(fragments.copy())
        
#         # Generate pairs for this sequence
#         pairs = pair_generator.generate_pairs(transformed_fragments)
        
#         if not pairs:
#             tqdm.write(f"  {sequence}: 0 pairs (skipped)")
#             continue
        
#         # Save individual CSV
#         csv_path = saver.save_pairs_to_csv(pairs, filename=f"{sequence}_pairs.csv")
        
#         # Also load for combining later
#         df = pd.read_csv(csv_path)
#         all_pairs_dfs.append(df)
#         total_pairs += len(df)
        
#         tqdm.write(f"  ✓ {sequence}: {len(df)} pairs")
    
#     # ========== PHASE 4: Combine all pairs ==========
#     print(f"\n{'='*80}")
#     print("Phase 4: Combining all pairs into single dataset")
#     print(f"{'='*80}")
    
#     combined_df = pd.concat(all_pairs_dfs, ignore_index=True)
    
#     print(f"\nCombined dataset:")
#     print(f"  Total pairs: {len(combined_df):,}")
#     print(f"  Total features: {len(combined_df.columns) - 3}")
#     print(f"  Positive (merge): {combined_df['label'].sum():,} ({combined_df['label'].mean()*100:.1f}%)")
#     print(f"  Negative (don't merge): {(combined_df['label']==0).sum():,} ({(1-combined_df['label'].mean())*100:.1f}%)")
    
#     # Save combined dataset
#     combined_path = output_root / "training_data" / "combined_train_data.csv"
#     combined_df.to_csv(combined_path, index=False)
    
#     print(f"\n✓ Saved combined dataset to: {combined_path}")
#     print(f"  Size: {combined_path.stat().st_size / 1024 / 1024:.1f} MB")
    
#     # Calculate final statistics
#     num_features = len(combined_df.columns) - 3
#     samples_per_feature = len(combined_df) / num_features
    
#     print(f"\n{'='*80}")
#     print("FINAL STATISTICS")
#     print(f"{'='*80}")
#     print(f"Sequences processed: {len(sequences)}")
#     print(f"Total fragments: {total_fragments}")
#     print(f"Total pairs: {len(combined_df):,}")
#     print(f"Features per pair: {num_features}")
#     print(f"Samples per feature: {samples_per_feature:.1f}")
#     print(f"Status: {'✅ EXCELLENT' if samples_per_feature >= 100 else '✅ GOOD' if samples_per_feature >= 50 else '⚠️ RISKY'}")
    
#     return combined_path


# if __name__ == "__main__":
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # ========== CONFIGURATION ==========
#     DATA_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\data\SoccerNetGS\train")
#     OUTPUT_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output")
#     WEIGHTS_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\weights")
    
#     # Get all sequence directories
#     all_sequences = sorted([d.name for d in DATA_ROOT.iterdir() if d.is_dir()])
    
#     print(f"Found {len(all_sequences)} sequences in {DATA_ROOT}")
#     print(f"First few: {all_sequences[:5]}")
#     print(f"Last few: {all_sequences[-5:]}")
    
#     # Option to test on subset first
#     TEST_MODE = False  # Set to True to test on just a few sequences
    
#     if TEST_MODE:
#         sequences = all_sequences[:3]  # Test with first 3 sequences
#         print(f"\n⚠️ TEST MODE: Processing only {len(sequences)} sequences")
#     else:
#         sequences = all_sequences
#         print(f"\n✓ FULL MODE: Processing all {len(sequences)} sequences")
    
#     # Generate training data
#     combined_path = generate_all_training_data(
#         sequences, DATA_ROOT, OUTPUT_ROOT, WEIGHTS_ROOT, device
#     )
    
#     print(f"\n{'='*80}")
#     print("✓ COMPLETE!")
#     print(f"{'='*80}")
#     print(f"Training data ready at: {combined_path}")
#     print(f"PCA models saved at: {OUTPUT_ROOT / 'training_data' / 'pca_models'}")


# generate_all_training_data.py

import torch
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from utils.data_utils import load_images, organize_detections_by_track
from utils.config import Paths, TrackerConfig, JerseyPredictorConfig, SplitterConfig
from detect_and_track.detect_and_track import detect_and_track
from detect_and_track.trackers.Deep_EIoU import DeepEIOUTracker
from attributes.attributes import predict_attributes
from tracklets.split_tracklets import split_tracklets

from training_data.fragment_aggregator import FragmentAggregator
from training_data.pair_generator import PairGenerator
from training_data.data_format import DataSaver
from training_data.pca_reducer import PCAReducer


def process_single_sequence(sequence, data_root, output_root, weights_root, device):
    """
    Process a single sequence and return aggregated fragments (WITHOUT PCA).
    PCA will be applied later on ALL fragments from ALL sequences.
    """
    
    # Setup configs
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
    attributes_tracklets = predict_attributes(images, tracklets, paths, jersey_cfg, device)
    splitted_tracklets = split_tracklets(attributes_tracklets, splitter_cfg)
    
    # Aggregate fragments (WITHOUT PCA)
    aggregator = FragmentAggregator()
    aggregated_fragments = []
    
    for track_id, tracklet in splitted_tracklets.items():
        agg = aggregator.aggregate_tracklet(tracklet)
        if agg is not None:
            # Add sequence name to metadata for tracking
            agg['metadata']['sequence'] = sequence
            aggregated_fragments.append(agg)
    
    return aggregated_fragments


def generate_all_training_data(sequences, data_root, output_root, weights_root, device, split="train", fit_pca=True):
    """
    Process all sequences, fit PCA on ALL data, then generate training pairs.
    
    This is the CORRECT way to do it:
    1. Process all sequences → collect all fragments
    2. Fit PCA on ALL fragments from ALL sequences
    3. Transform all fragments with fitted PCA
    4. Generate pairs per sequence
    5. Combine all pairs into one dataset
    """
    
    print(f"\n{'='*80}")
    print(f"PROCESSING {len(sequences)} SEQUENCES")
    print(f"{'='*80}\n")
    
    # ========== PHASE 1: Process all sequences ==========
    print("Phase 1: Processing all sequences and collecting fragments...")
    print("(This will take a while - tracking, jersey detection, team classification)")
    print()
    
    all_fragments_by_sequence = {}
    total_fragments = 0
    
    for sequence in tqdm(sequences, desc="Processing sequences"):
        try:
            fragments = process_single_sequence(sequence, data_root, output_root, weights_root, device)
            all_fragments_by_sequence[sequence] = fragments
            total_fragments += len(fragments)
            tqdm.write(f"  ✓ {sequence}: {len(fragments)} fragments")
        except Exception as e:
            tqdm.write(f"  ✗ {sequence}: ERROR - {e}")
            all_fragments_by_sequence[sequence] = []
    
    print(f"\nTotal fragments collected: {total_fragments}")
    
    # ========== PHASE 2: Fit or load PCA ==========
    pca_output_dir = weights_root / "pca_models"
    pca_reducer = PCAReducer(siglip_components=16, reid_components=8)

    if fit_pca:
        print(f"\n{'='*80}")
        print("Phase 2: Fitting PCA on fragments and saving model")
        print(f"{'='*80}")

        all_fragments = []
        for sequence, fragments in all_fragments_by_sequence.items():
            all_fragments.extend(fragments)

        print(f"Fitting PCA on {len(all_fragments)} total fragments...")
        pca_reducer.fit(all_fragments)
        pca_reducer.save(pca_output_dir)
        print(f"✓ PCA model saved to {pca_output_dir}")
    else:
        print(f"\n{'='*80}")
        print(f"Phase 2: Loading PCA model from train run")
        print(f"{'='*80}")

        if not pca_output_dir.exists():
            raise FileNotFoundError(
                f"PCA model not found at {pca_output_dir}. "
                f"Run with FIT_PCA=True on train first."
            )
        pca_reducer = PCAReducer.load(pca_output_dir)
        print(f"✓ PCA model loaded from {pca_output_dir}")
    
    # ========== PHASE 3: Transform all fragments and generate pairs ==========
    print(f"\n{'='*80}")
    print("Phase 3: Transforming fragments and generating pairs")
    print(f"{'='*80}\n")
    
    pair_generator = PairGenerator(
        max_temporal_gap=100,
        include_all_positives=True,
        negative_sampling_ratio=2.0
    )
    
    saver = DataSaver(output_root / "training_data")
    
    all_pairs_dfs = []
    total_pairs = 0
    
    for sequence in tqdm(sequences, desc="Generating pairs"):
        fragments = all_fragments_by_sequence.get(sequence, [])
        
        if not fragments:
            continue
        
        # Transform this sequence's fragments
        transformed_fragments = pca_reducer.transform(fragments.copy())
        
        # Generate pairs for this sequence
        pairs = pair_generator.generate_pairs(transformed_fragments)
        
        if not pairs:
            tqdm.write(f"  {sequence}: 0 pairs (skipped)")
            continue
        
        # Save individual CSV
        csv_path = saver.save_pairs_to_csv(pairs, filename=f"{sequence}_pairs.csv")
        
        # Also load for combining later
        df = pd.read_csv(csv_path)
        all_pairs_dfs.append(df)
        total_pairs += len(df)
        
        tqdm.write(f"  ✓ {sequence}: {len(df)} pairs")
    
    # ========== PHASE 4: Combine all pairs ==========
    print(f"\n{'='*80}")
    print("Phase 4: Combining all pairs into single dataset")
    print(f"{'='*80}")
    
    combined_df = pd.concat(all_pairs_dfs, ignore_index=True)
    
    print(f"\nCombined dataset:")
    print(f"  Total pairs: {len(combined_df):,}")
    print(f"  Total features: {len(combined_df.columns) - 3}")
    print(f"  Positive (merge): {combined_df['label'].sum():,} ({combined_df['label'].mean()*100:.1f}%)")
    print(f"  Negative (don't merge): {(combined_df['label']==0).sum():,} ({(1-combined_df['label'].mean())*100:.1f}%)")
    
    # Save combined dataset
    combined_path = output_root / "training_data" / f"combined_{split}_data.csv"
    combined_df.to_csv(combined_path, index=False)
    
    print(f"\n✓ Saved combined dataset to: {combined_path}")
    print(f"  Size: {combined_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Calculate final statistics
    num_features = len(combined_df.columns) - 3
    samples_per_feature = len(combined_df) / num_features
    
    print(f"\n{'='*80}")
    print("FINAL STATISTICS")
    print(f"{'='*80}")
    print(f"Sequences processed: {len(sequences)}")
    print(f"Total fragments: {total_fragments}")
    print(f"Total pairs: {len(combined_df):,}")
    print(f"Features per pair: {num_features}")
    print(f"Samples per feature: {samples_per_feature:.1f}")
    print(f"Status: {'✅ EXCELLENT' if samples_per_feature >= 100 else '✅ GOOD' if samples_per_feature >= 50 else '⚠️ RISKY'}")
    
    return combined_path


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ========== CHANGE THESE between runs ==========
    SPLIT = "valid"       # "train", "valid", or "test" — only used for the output filename
    FIT_PCA = False        # True for train run, False for valid/test runs
    DATA_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\data\SoccerNetGS\valid")  # swap to \valid or \test
    OUTPUT_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output")
    WEIGHTS_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\weights")
    
    # Get all sequence directories
    all_sequences = sorted([d.name for d in DATA_ROOT.iterdir() if d.is_dir()])
    
    print(f"Found {len(all_sequences)} sequences in {DATA_ROOT}")
    print(f"First few: {all_sequences[:5]}")
    print(f"Last few: {all_sequences[-5:]}")
    
    # Option to test on subset first
    TEST_MODE = False  # Set to True to test on just a few sequences
    
    if TEST_MODE:
        sequences = all_sequences[:3]  # Test with first 3 sequences
        print(f"\n⚠️ TEST MODE: Processing only {len(sequences)} sequences")
    else:
        sequences = all_sequences
        print(f"\n✓ FULL MODE: Processing all {len(sequences)} sequences")
    
    # Generate training data
    combined_path = generate_all_training_data(
        sequences, DATA_ROOT, OUTPUT_ROOT, WEIGHTS_ROOT, device, split=SPLIT, fit_pca=FIT_PCA
    )
    
    print(f"\n{'='*80}")
    print("✓ COMPLETE!")
    print(f"{'='*80}")
    print(f"Training data ready at: {combined_path}")
    print(f"PCA models saved at: {OUTPUT_ROOT / 'training_data' / 'pca_models'}")