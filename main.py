import torch
from pathlib import Path

from utils.data_utils import load_images, organize_detections_by_track
from utils.config import Paths, TrackerConfig, JerseyPredictorConfig, SplitterConfig
from utils.visualization_utils import visualize_tracklets 
from detect_and_track.detect_and_track import detect_and_track
from detect_and_track.trackers.Deep_EIoU import DeepEIOUTracker
from attributes.attributes import predict_attributes
from tracklets.split_tracklets import split_tracklets
from training_data.fragment_aggregator import FragmentAggregator
from training_data.pair_generator import PairGenerator
from training_data.data_format import DataSaver
from training_data.pca_reducer import PCAReducer


def main(tracker_cfg, paths, device):

    # Initialize tracker
    tracker = DeepEIOUTracker(
        track_thresh = tracker_cfg.track_thresh,
        track_low_thresh = tracker_cfg.track_low_thresh,
        new_track_thresh = tracker_cfg.new_track_thresh,
        track_buffer = tracker_cfg.track_buffer,
        match_thresh = tracker_cfg.match_thresh,
        proximity_thresh = tracker_cfg.proximity_thresh,
        appearance_thresh = tracker_cfg.appearance_thresh,
        with_reid = tracker_cfg.with_reid,
        reid_model_name = tracker_cfg.reid_model_name,
        reid_model_path = str(paths.reid_model_path),
        frame_rate = tracker_cfg.frame_rate,
    )
        
    # Load images
    images = load_images(img_dir=paths.img_path)
    
    # Load ground truth detections and perform tracking
    tracked_detections = detect_and_track(images, tracker, paths)

    # Organize detections by frame into tracklets
    tracklets = organize_detections_by_track(tracked_detections)
    visualize_tracklets(
        images,
        tracklets,
        paths.output_path / "videos" / f"{SEQUENCE}_original.mp4",
        title="PARSeq Jersey Detection",
    )
    # Predict attributes for each tracklet
    attributes_tracklets = predict_attributes(images, tracklets, paths, jersey_cfg, device)

    # # Split tracklets
    splitted_tracklets = split_tracklets(attributes_tracklets, splitter_cfg)


    # aggregator = FragmentAggregator()
    # aggregated_fragments = []
    # for track_id, tracklet in splitted_tracklets.items():
    #     agg = aggregator.aggregate_tracklet(tracklet)
    #     if agg is not None:
    #         aggregated_fragments.append(agg)
    
    # print(f"Aggregated {len(aggregated_fragments)} fragments")
    
    # # ========== APPLY PCA (NEW!) ==========
    # pca_reducer = PCAReducer(siglip_components=16, reid_components=8)
    # aggregated_fragments = pca_reducer.fit_transform(aggregated_fragments)
    
    # # Save PCA models
    # pca_output_dir = paths.output_path / "training_data" / "pca_models"
    # pca_reducer.save(pca_output_dir)
    
    # # ========== GENERATE PAIRS ==========
    # pair_generator = PairGenerator(
    #     max_temporal_gap=100,
    #     include_all_positives=True,
    #     negative_sampling_ratio=2.0
    # )
    
    # pairs = pair_generator.generate_pairs(aggregated_fragments)
    
    # # Save to CSV
    # saver = DataSaver(paths.output_path / "training_data")
    # csv_path = saver.save_pairs_to_csv(pairs, filename=f"{paths.sequence}_pairs.csv")
    
    # return csv_path
    # Visualize tracklets
    visualize_tracklets(
        images,
        splitted_tracklets,
        paths.output_path / "videos" / f"{SEQUENCE}_all_splitter.mp4",
        title="PARSeq Jersey Detection",
    )


if __name__ == "__main__":

    tracker_cfg = TrackerConfig(
        track_thresh = 0.6,
        track_low_thresh = 0.3,
        new_track_thresh = 0.4,
        track_buffer = 60,
        match_thresh = 0.8,
        proximity_thresh = 0.5,
        appearance_thresh = 0.2,
        with_reid = True,
        reid_model_name = "osnet_x0_25",
        frame_rate = 25
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    SEQUENCE = "SNGS-123"
    DATA_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\data\SoccerNetGS\test")
    OUTPUT_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output")
    WEIGHTS_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\weights")
    
    paths = Paths(
        img_path = DATA_ROOT / SEQUENCE / "img1",
        gt_detections_path = DATA_ROOT / SEQUENCE / "Labels-GameState.json",
        output_path = OUTPUT_ROOT,
        cache_path = OUTPUT_ROOT / "cache",
        # legibility_model_path = WEIGHTS_ROOT / "jersey_weights" /"legibility" / "legibility_resnet34_soccer_20240215.pth",
        legibility_model_path = WEIGHTS_ROOT / "jersey_weights" /"legibility" / "output.pth",
        reid_model_path = WEIGHTS_ROOT / "jersey_weights" / "reid" / "osnet_x0_25_msmt17.pt",
        parseq_model_path = WEIGHTS_ROOT / "jersey_weights" / "parseq" / "parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt",
        centroid_reid_path = WEIGHTS_ROOT / "jersey_weights" / "centroid_reid" / "market1501_resnet50_256_128_epoch_120.ckpt",
        siglip_model_path = WEIGHTS_ROOT / "team_weights" / "siglip",        
        vitpose_model_path = WEIGHTS_ROOT / "jersey_weights" / "vitpose",
        sequence = SEQUENCE
    )

    jersey_cfg = JerseyPredictorConfig(
        use_legibility = True,
        use_reid_filter = True,
        use_pose_cropper = True,
        legibility_arch = "resnet34",
        legibility_threshold = 0.6,
        reid_threshold_std = 0.5,
        debug_tracklet_id = 1,
        debug_dir=r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\debug"
    )

    splitter_cfg = SplitterConfig(
        # Jersey splitter
        jersey_min_fragment = 20,
        jersey_min_persistence = 5 ,
        jersey_lookahead = 20,
        jersey_lookback = 300,
        jersey_min_pixel_jump = 10,
        jersey_entropy_threshold = 0.2,

        # Team splitter
        team_min_persistence = 5,
        team_min_fragment = 10,
        team_lookahead = 50,
    )

    main(tracker_cfg, paths, device)


# import torch
# from pathlib import Path

# from utils.data_utils import load_images, organize_detections_by_track
# from utils.config import Paths, TrackerConfig, JerseyPredictorConfig, SplitterConfig
# from utils.visualization_utils import visualize_tracklets 
# from detect_and_track.detect_and_track import detect_and_track
# from detect_and_track.trackers.Deep_EIoU import DeepEIOUTracker
# from attributes.attributes import predict_attributes
# from tracklets.split_tracklets import split_tracklets
# from training_data.fragment_aggregator import FragmentAggregator
# from training_data.pair_generator import PairGenerator
# from training_data.data_format import DataSaver
# from training_data.pca_reducer import PCAReducer


# def main(tracker_cfg, paths, device, jersey_cfg, splitter_cfg, SEQUENCE):

#     # Initialize tracker
#     tracker = DeepEIOUTracker(
#         track_thresh = tracker_cfg.track_thresh,
#         track_low_thresh = tracker_cfg.track_low_thresh,
#         new_track_thresh = tracker_cfg.new_track_thresh,
#         track_buffer = tracker_cfg.track_buffer,
#         match_thresh = tracker_cfg.match_thresh,
#         proximity_thresh = tracker_cfg.proximity_thresh,
#         appearance_thresh = tracker_cfg.appearance_thresh,
#         with_reid = tracker_cfg.with_reid,
#         reid_model_name = tracker_cfg.reid_model_name,
#         reid_model_path = str(paths.reid_model_path),
#         frame_rate = tracker_cfg.frame_rate,
#     )
        
#     # Load images
#     images = load_images(img_dir=paths.img_path)
    
#     # Load ground truth detections and perform tracking
#     tracked_detections = detect_and_track(images, tracker, paths)

#     # Organize detections by frame into tracklets
#     tracklets = organize_detections_by_track(tracked_detections)

#     # Predict attributes for each tracklet
#     attributes_tracklets = predict_attributes(images, tracklets, paths, jersey_cfg, device)

#     # # Split tracklets
#     # splitted_tracklets = split_tracklets(attributes_tracklets, splitter_cfg)


#     # aggregator = FragmentAggregator()
#     # aggregated_fragments = []
#     # for track_id, tracklet in splitted_tracklets.items():
#     #     agg = aggregator.aggregate_tracklet(tracklet)
#     #     if agg is not None:
#     #         aggregated_fragments.append(agg)
    
#     # print(f"Aggregated {len(aggregated_fragments)} fragments")
    
#     # # ========== APPLY PCA (NEW!) ==========
#     # pca_reducer = PCAReducer(siglip_components=16, reid_components=8)
#     # aggregated_fragments = pca_reducer.fit_transform(aggregated_fragments)
    
#     # # Save PCA models
#     # pca_output_dir = paths.output_path / "training_data" / "pca_models"
#     # pca_reducer.save(pca_output_dir)
    
#     # # ========== GENERATE PAIRS ==========
#     # pair_generator = PairGenerator(
#     #     max_temporal_gap=100,
#     #     include_all_positives=True,
#     #     negative_sampling_ratio=2.0
#     # )
    
#     # pairs = pair_generator.generate_pairs(aggregated_fragments)
    
#     # # Save to CSV
#     # saver = DataSaver(paths.output_path / "training_data")
#     # csv_path = saver.save_pairs_to_csv(pairs, filename=f"{paths.sequence}_pairs.csv")
    
#     # return csv_path



# if __name__ == "__main__":

#     tracker_cfg = TrackerConfig(
#         track_thresh = 0.6,
#         track_low_thresh = 0.3,
#         new_track_thresh = 0.4,
#         track_buffer = 60,
#         match_thresh = 0.8,
#         proximity_thresh = 0.5,
#         appearance_thresh = 0.2,
#         with_reid = True,
#         reid_model_name = "osnet_x0_25",
#         frame_rate = 25
#     )

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     DATA_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\data\SoccerNetGS\valid")
#     OUTPUT_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output")
#     WEIGHTS_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\weights")
    
#     jersey_cfg = JerseyPredictorConfig(
#         use_legibility = True,
#         use_reid_filter = True,
#         use_pose_cropper = True,
#         legibility_arch = "resnet34",
#         legibility_threshold = 0.6,
#         reid_threshold_std = 0.5,
#         debug_tracklet_id = 1,
#         debug_dir=r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\debug"
#     )

#     splitter_cfg = SplitterConfig(
#         # Jersey splitter
#         jersey_min_fragment = 20,
#         jersey_min_persistence = 5 ,
#         jersey_lookahead = 20,
#         jersey_lookback = 300,
#         jersey_min_pixel_jump = 10,
#         jersey_entropy_threshold = 0.2,

#         # Team splitter
#         team_min_persistence = 5,
#         team_min_fragment = 10,
#         team_lookahead = 50,
#     )

#     # Get all sequences in the training folder
#     all_sequences = sorted([d.name for d in DATA_ROOT.iterdir() if d.is_dir()])
    
#     print(f"Found {len(all_sequences)} sequences in {DATA_ROOT}")
#     print(f"First few: {all_sequences[:5]}")
    
#     # Loop through all sequences
#     for i, SEQUENCE in enumerate(all_sequences, 1):
#         print(f"\n{'='*80}")
#         print(f"Processing {i}/{len(all_sequences)}: {SEQUENCE}")
#         print(f"{'='*80}\n")
        
#         try:
#             paths = Paths(
#                 img_path = DATA_ROOT / SEQUENCE / "img1",
#                 gt_detections_path = DATA_ROOT / SEQUENCE / "Labels-GameState.json",
#                 output_path = OUTPUT_ROOT,
#                 cache_path = OUTPUT_ROOT / "cache",
#                 legibility_model_path = WEIGHTS_ROOT / "jersey_weights" /"legibility" / "output.pth",
#                 reid_model_path = WEIGHTS_ROOT / "jersey_weights" / "reid" / "osnet_x0_25_msmt17.pt",
#                 parseq_model_path = WEIGHTS_ROOT / "jersey_weights" / "parseq" / "parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt",
#                 centroid_reid_path = WEIGHTS_ROOT / "jersey_weights" / "centroid_reid" / "market1501_resnet50_256_128_epoch_120.ckpt",
#                 siglip_model_path = WEIGHTS_ROOT / "team_weights" / "siglip",        
#                 vitpose_model_path = WEIGHTS_ROOT / "jersey_weights" / "vitpose",
#                 sequence = SEQUENCE
#             )
            
#             main(tracker_cfg, paths, device, jersey_cfg, splitter_cfg, SEQUENCE)
#             print(f"✓ Successfully processed {SEQUENCE}")
            
#         except Exception as e:
#             print(f"✗ Error processing {SEQUENCE}: {e}")
#             continue
    
#     print(f"\n{'='*80}")
#     print(f"✓ COMPLETE! Processed all sequences")
#     print(f"{'='*80}")