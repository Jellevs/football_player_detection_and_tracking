import torch
from pathlib import Path
from tqdm import tqdm
import copy

from utils.data_utils import load_images, organize_detections_by_track
from utils.config import Paths, TrackerConfig, JerseyPredictorConfig, SplitterConfig, MergerConfig
from utils.visualization_utils import visualize_tracklets 
from utils.evaluation_utils import save_mot_files_for_sn_trackeval, save_mot_file_for_sn_trackeval  # Updated import
from detect_and_track.detect_and_track import detect_and_track
from detect_and_track.trackers.Deep_EIoU import DeepEIOUTracker
from attributes.attributes import predict_attributes
from tracklets.split_tracklets import split_tracklets
from tracklets.tracklet_merger import TrackletMerger
from tracklets.soccer_aware_merger import SoccerAwareMerger

# Import GTA
from tracklets.splitters.gta_splitter import split_tracklets as split_tracklets_gta, merge_tracklets as merge_tracklets_gta

def main(sequence, tracker_cfg, jersey_cfg, splitter_cfg, merger_cfg, device):

    # Paths config
    paths = Paths(
        img_path=DATA_ROOT / sequence / "img1",
        gt_detections_path=DATA_ROOT / sequence / "Labels-GameState.json",
        output_path=OUTPUT_ROOT,
        cache_path=OUTPUT_ROOT / "cache",
        legibility_model_path=WEIGHTS_ROOT / "jersey_weights" / "legibility" / "output.pth",
        # reid_model_path=WEIGHTS_ROOT / "jersey_weights" / "reid" / "osnet_x0_25_msmt17.pt",
        reid_model_path = PARENT_ROOT / "gta_link" / "reid_checkpoints" / "sports_model.pth.tar-60",
        parseq_model_path=WEIGHTS_ROOT / "jersey_weights" / "parseq" / "parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt",
        centroid_reid_path=WEIGHTS_ROOT / "jersey_weights" / "centroid_reid" / "market1501_resnet50_256_128_epoch_120.ckpt",
        siglip_model_path=WEIGHTS_ROOT / "team_weights" / "siglip",
        vitpose_model_path=WEIGHTS_ROOT / "jersey_weights" / "vitpose",
        evaluation_path=PARENT_ROOT / "evaluation",
        sequence=sequence
    )


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

    splitted_tracklets = split_tracklets_gta(attributes_tracklets, eps=0.6, max_k=3, min_samples=5, len_thres=100)

    # soccer_merger = SoccerAwareMerger()
    # soccer_merged_tracklets = soccer_merger.merge(splitted_tracklets)

    merged_tracklets = merge_tracklets_gta(
        splitted_tracklets, dict(), seq_name=sequence, merge_dist_thres=0.4)

    save_mot_file_for_sn_trackeval(
        tracklets_dict=merged_tracklets,
        output_path=paths.evaluation_path,
        sequence_name=sequence,
        stage_name="tracklets_gta_merged"
    )

    # visualize_tracklets(
    #     images,
    #     tracklets,
    #     paths.output_path / "videos" / f"{sequence}_original.mp4",
    #     title="PARSeq Jersey Detection",
    # )

if __name__ == "__main__":
    
    # ===== Configuration =====
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    DATA_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\data\SoccerNetGS\test")
    OUTPUT_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output")
    WEIGHTS_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\weights")
    PARENT_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch")

    # Tracker config
    tracker_cfg = TrackerConfig(
        track_thresh=0.6,
        track_low_thresh=0.3,
        new_track_thresh=0.4,
        track_buffer=60,
        match_thresh=0.8,
        proximity_thresh=0.5,
        appearance_thresh=0.2,
        with_reid=True,
        # reid_model_name="osnet_x0_25",
        reid_model_name="osnet_x1_0",

        frame_rate=25
    )
    
    # Jersey prediction config
    jersey_cfg = JerseyPredictorConfig(
        use_legibility=True,
        use_reid_filter=True,
        use_pose_cropper=True,
        legibility_arch="resnet34",
        legibility_threshold=0.6,
        reid_threshold_std=0.5,
    )
    
    # Splitter config
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
        xgboost_model_path = r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\weights\xgboost\xgboost_baseline.json",
        pca_model_path = r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\training_data\pca_models",
        merge_threshold = 0.5,
        linkage_method = 'average',
        max_temporal_gap = 100,
    )
    
    # Run pipeline on all sequences
    sequences = sorted([d.name for d in DATA_ROOT.iterdir() if d.is_dir()])
    
    for sequence in tqdm(sequences, desc="Processing sequences"):
        main(
            sequence,
            tracker_cfg=tracker_cfg,
            jersey_cfg=jersey_cfg,
            splitter_cfg=splitter_cfg,
            merger_cfg=merger_cfg,
            device=device,
        )