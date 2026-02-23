import torch
import settings

from utils.config import Paths, TrackerConfig, JerseyPredictorConfig, SplitterConfig, MergerConfig


def build_configs():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tracker_cfg  = TrackerConfig(**settings.TRACKER)
    jersey_cfg   = JerseyPredictorConfig(**settings.JERSEY)
    splitter_cfg = SplitterConfig(**settings.SPLITTER)
    merger_cfg   = MergerConfig(
        xgboost_model_path=settings.WEIGHTS_ROOT / "xgboost" / "xgboost_baseline.json",
        pca_model_path=settings.OUTPUT_ROOT / "training_data" / "pca_models",
        **settings.MERGER,
    )
    return device, tracker_cfg, jersey_cfg, splitter_cfg, merger_cfg


def build_paths(sequence):
    return Paths(
        img_path=settings.DATA_ROOT / sequence / "img1",
        gt_detections_path=settings.DATA_ROOT / sequence / "Labels-GameState.json",
        output_path=settings.OUTPUT_ROOT,
        cache_path=settings.OUTPUT_ROOT / "cache",
        legibility_model_path=settings.WEIGHTS_ROOT / "jersey_weights" / "legibility" / "output.pth",
        reid_model_path=settings.WEIGHTS_ROOT / "reid_weights" / "sports_model.pth.tar-60",
        parseq_model_path=settings.WEIGHTS_ROOT / "jersey_weights" / "parseq" / "parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt",
        centroid_reid_path=settings.WEIGHTS_ROOT / "jersey_weights" / "centroid_reid" / "market1501_resnet50_256_128_epoch_120.ckpt",
        siglip_model_path=settings.WEIGHTS_ROOT / "team_weights" / "siglip",
        vitpose_model_path=settings.WEIGHTS_ROOT / "jersey_weights" / "vitpose",
        evaluation_path=settings.PROJECT_ROOT / "evaluation",
        sequence=sequence,
    )
