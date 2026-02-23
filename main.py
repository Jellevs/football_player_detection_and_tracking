from tqdm import tqdm

import settings
from utils.run_evaluation import run_evaluation
from utils.build import build_paths, build_configs
from utils.data_utils import load_images, organize_detections_by_track
from utils.visualization_utils import visualize_tracklets 
from utils.evaluation_utils import save_mot_file_for_sn_trackeval
from detect_and_track.detect_and_track import detect_and_track
from detect_and_track.trackers.Deep_EIoU import DeepEIOUTracker
from attributes.attributes import predict_attributes
from tracklets.split_tracklets import split_tracklets
from tracklets.tracklet_merger import TrackletMerger
from tracklets.soccer_aware_merger import SoccerAwareMerger
from tracklets.splitters.gta_splitter import split_tracklets as split_tracklets_gta, merge_tracklets as merge_tracklets_gta


def main(sequence, tracker_cfg, jersey_cfg, splitter_cfg, merger_cfg, device, method_name):

    # Paths config
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

    # Organize detections into tracklets
    tracklets = organize_detections_by_track(tracked_detections)

    # # Predict attributes for each tracklet
    # attributes_tracklets = predict_attributes(images, tracklets, paths, jersey_cfg, device)

    # # Split tracklets based using GTA
    # splitted_tracklets = split_tracklets_gta(attributes_tracklets, eps=0.6, max_k=3, min_samples=5, len_thres=100)

    # # Merge tracklets using GTA
    # merged_tracklets = merge_tracklets_gta(
    #     splitted_tracklets, dict(), seq_name=sequence, merge_dist_thres=0.4)

    # Save tracklets in MOT format
    save_mot_file_for_sn_trackeval(
        tracklets_dict=tracklets,
        output_path=paths.evaluation_path,
        sequence_name=sequence,
        method_name=method_name
    )


if __name__ == "__main__":
    device, tracker_cfg, jersey_cfg, splitter_cfg, merger_cfg = build_configs()

    sequences = sorted([d.name for d in settings.DATA_ROOT.iterdir() if d.is_dir()])
    
    # Run pipeline on all sequences
    for sequence in tqdm(sequences, desc="Processing sequences"):
        main(
            sequence,
            tracker_cfg=tracker_cfg,
            jersey_cfg=jersey_cfg,
            splitter_cfg=splitter_cfg,
            merger_cfg=merger_cfg,
            device=device,
            method_name=settings.METHOD_NAME
        )

    run_evaluation(settings.METHOD_NAME, settings.EVAL_SPLIT)
