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
from utils.evaluation_utils import save_mot_files_for_sn_trackeval  # Updated import

from gta_link.refine_tracklets import split_tracklets as split_gta, merge_tracklets as merge_gta, get_distance_matrix, get_spatial_constraints
# from gta import split_tracklets as split_gta, merge_tracklets as merge_gta, get_distance_matrix, get_spatial_constraints

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

    # Predict attributes for each tracklet
    attributes_tracklets = predict_attributes(images, tracklets, paths, jersey_cfg, device)

    # Post-process tracklets attribute style
    splitted_tracklets = split_tracklets(attributes_tracklets, splitter_cfg)

    merger = TrackletMerger(merger_cfg=merger_cfg)
    merged_tracklets = merger.merge_tracklets(splitted_tracklets)


    # Post-process tracklets GTA
    max_x_range, max_y_range = get_spatial_constraints(tracklets, 1)

    splitted_gta_tracklets = split_gta(
        tracklets,
        eps=0.7,
        max_k=3,
        min_samples=10,
        len_thres=100
        )
    
    distance_matrix = get_distance_matrix(splitted_gta_tracklets)

    merged_gta_tracklets = merge_gta(
        splitted_gta_tracklets,
        {},
        distance_matrix,
        paths.sequence,
        max_x_range,
        max_y_range,
        0.4
    )



    # Save MOT files in sn-trackeval format
    save_mot_files_for_sn_trackeval(
        tracklets_original=tracklets,
        tracklets_gta=merged_gta_tracklets,
        tracklets_merged=merged_tracklets,
        output_dir=paths.evaluation_path,
        sequence_name=paths.sequence
    )


    # Visualize tracklets
    # visualize_tracklets(
    #     images,
    #     splitted_tracklets,
    #     paths.output_path / "videos" / f"{SEQUENCE}_all_splitter.mp4",
    #     title="PARSeq Jersey Detection",
    # )

    # visualize_tracklets(
    #     images,
    #     tracklets,
    #     paths.output_path / "videos" / f"{SEQUENCE}_original.mp4",
    #     title="PARSeq Jersey Detection",
    # )


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
    
    SEQUENCE = "SNGS-125"
    DATA_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\data\SoccerNetGS\train")
    OUTPUT_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output")
    WEIGHTS_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\weights")
    PARENT_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch")

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
        evaluation_path=PARENT_ROOT / "evaluation",
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
