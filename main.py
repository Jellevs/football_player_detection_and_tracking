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
from tracklets.splitters.gta_splitter import split_tracklets as split_tracklets_gta, merge_tracklets as merge_tracklets_gta
from tracklets.splitters.temporal_reid_splitter import TemporalReIDSplitter
from tracklets.simple_tracklet_merger import  SimpleTrackletMerger
from tracklets.xgboost_merger import XGBoostMerger
from tracklets.transformer_merger import TransformerMerger
from tracklets.transformer_merger_extended_pw import TransformerMergerExtendedPW


# TODO: remove team_min_persistence, only keep the ratio
# TODO: add kmeans confidence
# from sklearn.metrics import pairwise_distances

# centers = cluster_model.cluster_centers_  # (2, 3)
# dists = pairwise_distances(all_projections, centers)  # (N, 2)

# # Distance to assigned cluster and other cluster
# assigned_dist = dists[np.arange(len(predictions)), predictions]
# other_dist    = dists[np.arange(len(predictions)), 1 - predictions]

# # Confidence: how much closer are we to assigned vs other
# confidence = other_dist / (assigned_dist + other_dist + 1e-6)


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
    
    
    # Split at temporal gaps where ReID embeddings indicate different identities
    # TODO: if no cache run this, if cache don't coz already incorporated in attributes tracklets
    temporal_splitter = TemporalReIDSplitter(min_gap_frames=5, reid_threshold=0.15)
    pre_split_tracklets = temporal_splitter.split_all(tracklets)

    # Predict attributes for each tracklet
    # attributes_tracklets = predict_attributes(images, pre_split_tracklets, paths, jersey_cfg, device)

    # # Split tracklets based 
    # splitted_tracklets = split_tracklets(attributes_tracklets, splitter_cfg)

    # tracklet_merger = XGBoostMerger(
    #     model_path=r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\weights\xgboost\xgboost_merger.json",
    #     meta_path=r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\weights\xgboost\xgboost_merger_meta.json",
    #     merge_threshold=0.5,
    # )

    # merged_tracklets = tracklet_merger.merge(splitted_tracklets)

    
    # transformer_merger = TransformerMerger(
    #     model_path=r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\weights\transformer\best_model.pt",
    #     meta_path=r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\weights\transformer\transformer_merger_meta.json",
    #     merge_threshold=0.5,
    #     linkage_method="average",
    #     jersey_entropy_threshold=0.15,
    #     team_consistency_threshold=0.9,
    #     device=None)
    
    # merged_tracklets = transformer_merger.merge(splitted_tracklets)


    # tracklet_merger = SimpleTrackletMerger()
    # merged_tracklets = tracklet_merger.merge(splitted_tracklets)


    # Save tracklets in MOT format
    save_mot_file_for_sn_trackeval(
        tracklets_dict=pre_split_tracklets,
        output_path=paths.evaluation_path,
        sequence_name=sequence,
        method_name=method_name
    )

    # visualize_tracklets(
    #     images=images,
    #     tracklets_dict=merged_tracklets,
    #     output_path=paths.output_path / "videos" / f"{sequence}_merged.mp4",
    #     title="blablabbla",
    # )

    # visualize_tracklets(
    #     images=images,
    #     tracklets_dict=splitted_tracklets,
    #     output_path=paths.output_path / "videos" / f"{sequence}_split.mp4",
    #     title="blablabbla",
    # )   
    
    # visualize_tracklets(
    #     images=images,
    #     tracklets_dict=attributes_tracklets,
    #     output_path=paths.output_path / "videos" / f"{sequence}_baseline.mp4",
    #     title="blablabbla",
    # )
    

    # from utils.diagnose_jersey_filtering import diagnose_filtering
    # diagnose_filtering(images, tracklets, paths, jersey_cfg, device, max_tracklets=100)

    # from utils.visualize_jerseys import visualize_jersey_predictions
    # visualize_jersey_predictions(images, attributes_tracklets, paths.output_path / sequence, max_tracklets=30)


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
