import pickle
from pathlib import Path
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
from tracklets.splitters.bbox_anomaly_splitter import BboxAnomalySplitter
from tracklets.simple_tracklet_merger import  SimpleTrackletMerger
from tracklets.xgboost_merger import XGBoostMerger
from tracklets.transformer_merger import TransformerMerger
from tracklets.cross_attention_merger import CrossAttentionMerger
from tracklets.hybrid_merger import HybridMerger
from tracklets.oracle_merger import OracleMerger
from tracklets.pairwise_mlp_merger import PairwiseMLPMerger


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

    # temporal_splitter = TemporalReIDSplitter(min_gap_frames=5, reid_threshold=0.15)
    # pre_split_tracklets = temporal_splitter.split_all(tracklets)

    attributes_tracklets = predict_attributes(images, tracklets, paths, jersey_cfg, device)

    # Split tracklets (with caching)
    split_cache_dir = Path(settings.OUTPUT_ROOT) / "cache_split"
    split_cache_path = split_cache_dir / f"cache_split_{sequence}.pkl"
    if split_cache_path.exists():
        with open(split_cache_path, "rb") as f:
            splitted_tracklets = pickle.load(f)
        print(f"  Loaded cached split tracklets for {sequence} ({len(splitted_tracklets)} tracklets)")
    else:
        splitted_tracklets = split_tracklets(attributes_tracklets, splitter_cfg)
        split_cache_dir.mkdir(parents=True, exist_ok=True)
        with open(split_cache_path, "wb") as f:
            pickle.dump(splitted_tracklets, f)
        print(f"  Cached split tracklets for {sequence} ({len(splitted_tracklets)} tracklets)")


    # ---- Merger selection ----
    # XGBoost merger (best: HOTA 90.75) 90.33
    # test 0.8 -> 89.68
    # 0.75 -> 89.835
    # 0.7 -> 89.881
    # purity 0.6 ->
    # tracklet_merger = XGBoostMerger(
    #     model_path=r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\xgboost\ablations\sweep_results\neg3_minlen0_purity0.8_splitTrue\xgboost_merger.json",
    #     meta_path=r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\xgboost\ablations\sweep_results\neg3_minlen0_purity0.8_splitTrue\xgboost_merger_meta.json",
    #     merge_threshold=0.7,
    # )

    # tracklet_merger = XGBoostMerger( # 90.003
    #     model_path=r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\xgboost\feature_ablation_models\remove_siglip\xgboost_merger.json",
    #     meta_path=r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\xgboost\feature_ablation_models\remove_siglip\xgboost_merger_meta.json",
    #     merge_threshold=0.7,
    # )

    # merged_tracklets = tracklet_merger.merge(splitted_tracklets)

    # Siamese CLS Transformer merger (real_synth data, test AUC 0.974)
    # _siamese_cls_root = r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\transformers\siamese_cls\output"
    # transformer_merger = TransformerMerger(
    #     model_path=_siamese_cls_root + r"\siamese_cls_real_synth\best_model.pt",
    #     meta_path=_siamese_cls_root + r"\siamese_cls_real_synth\transformer_merger_meta.json",
    #     merge_threshold=0.7,
    # )

    # Cross Attention Transformer merger (test AUC 0.9764, F1 0.888)
    # original data test thresh = 0.7 -> 87.952
    # augmented data thresh 0.7 -> 87.584
    # cross_attn_merger = CrossAttentionMerger(
    #     model_path=r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\transformers\cross_attention\output\cross_attention_simple_synth\best_model.pt",
    #     meta_path=r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\transformers\cross_attention\output\cross_attention_simple_synth\transformer_merger_meta.json",
    #     merge_threshold=0.7,
    # )
    # merged_tracklets = cross_attn_merger.merge(splitted_tracklets)

    # Hybrid (self-attn -> cross-attn) Transformer merger (test AUC 0.9731, F1 0.881) 87.636
    # hybrid_merger = HybridMerger(
    #     model_path=r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\transformers\hybrid\output\hybrid_real_synth\best_model.pt",
    #     meta_path=r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\transformers\hybrid\output\hybrid_real_synth\transformer_merger_meta.json",
    #     merge_threshold=0.75,
    # )
    # merged_tracklets = hybrid_merger.merge(merged_tracklets)

    # Pairwise MLP merger (12 pairwise features only, 3k params)
    # test 0.7 -> 87.839
    # 0.8 -> 88.254
    mlp_merger = PairwiseMLPMerger(
        model_path=r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\transformers\pairwise_mlp\output\pairwise_mlp_xgb_matched\best_model.pt",
        merge_threshold=0.8,
    )
    merged_tracklets = mlp_merger.merge(splitted_tracklets)

    # Oracle merger (uses GT annotations for perfect merge decisions — ceiling experiment)
    # oracle_merger = OracleMerger(gt_root=settings.DATA_ROOT)
    # merged_tracklets = oracle_merger.merge(splitted_tracklets, sequence_name=sequence)



    # Save tracklets in MOT format
    save_mot_file_for_sn_trackeval(
        tracklets_dict=merged_tracklets,
        output_path=paths.evaluation_path,
        sequence_name=sequence,
        method_name=method_name
    )

    # visualize_tracklets(
    #     images=images,
    #     tracklets_dict=attributes_tracklets,
    #     output_path=paths.output_path / "videos" / f"{sequence}_ATTRIBUTES.mp4",
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
    


if __name__ == "__main__":
    device, tracker_cfg, jersey_cfg, splitter_cfg, merger_cfg = build_configs()

    sequences = sorted([d.name for d in settings.DATA_ROOT.iterdir() if d.is_dir()])

    for sequence in tqdm(sequences, desc="Processing sequences"):
        main(
            sequence,
            tracker_cfg=tracker_cfg,
            jersey_cfg=jersey_cfg,
            splitter_cfg=splitter_cfg,
            merger_cfg=merger_cfg,
            device=device,
            method_name=settings.METHOD_NAME,
        )

    run_evaluation(settings.METHOD_NAME, settings.EVAL_SPLIT)
