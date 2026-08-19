import torch

import settings
from utils.config import Paths, TrackerConfig, JerseyPredictorConfig, SplitterConfig
from tracklets.connectors.xgboost_merger import XGBoostMerger
from tracklets.connectors.transformer_merger import TransformerMerger
from tracklets.connectors.cross_attention_merger import CrossAttentionMerger
from tracklets.connectors.hybrid_merger import HybridMerger
from tracklets.connectors.pairwise_mlp_merger import PairwiseMLPMerger
from tracklets.connectors.decision_merger import DecisionMerger
from tracklets.connectors.oracle_merger import OracleMerger


CONNECTOR_CLASSES = {
    "xgboost": XGBoostMerger,
    "transformer": TransformerMerger,
    "cross_attention": CrossAttentionMerger,
    "hybrid": HybridMerger,
    "pairwise_mlp": PairwiseMLPMerger,
    "decision": DecisionMerger,
    "oracle": OracleMerger,
}


def build_configs():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tracker_config = TrackerConfig(**settings.TRACKER)
    jersey_config = JerseyPredictorConfig(**settings.JERSEY)
    splitter_config = SplitterConfig(**settings.SPLITTER)
    return device, tracker_config, jersey_config, splitter_config


def build_connector(name):
    if name not in CONNECTOR_CLASSES:
        raise ValueError(f"Unknown connector {name!r}, expected one of {sorted(CONNECTOR_CLASSES)}")
    arguments = settings.CONNECTORS[name]
    log_path = arguments.get("log_path")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    return CONNECTOR_CLASSES[name](**arguments)


def build_paths(sequence):
    return Paths(
        img_path=settings.DATA_ROOT / sequence / "img1",
        gt_detections_path=settings.DATA_ROOT / sequence / "gt" / "gt.txt",
        output_path=settings.OUTPUT_ROOT,
        cache_path=settings.OUTPUT_ROOT / "cache",
        legibility_model_path=settings.WEIGHTS_ROOT / "jersey_weights" / "legibility" / "legibility_resnet34_soccer_20240215.pth",
        reid_model_path=settings.WEIGHTS_ROOT / "reid_weights" / "sports_model.pth.tar-60",
        parseq_model_path=settings.WEIGHTS_ROOT / "jersey_weights" / "parseq" / "parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt",
        centroid_reid_path=settings.WEIGHTS_ROOT / "jersey_weights" / "centroid_reid" / "market1501_resnet50_256_128_epoch_120.ckpt",
        siglip_model_path=settings.WEIGHTS_ROOT / "team_weights" / "siglip",
        vitpose_model_path=settings.WEIGHTS_ROOT / "jersey_weights" / "vitpose",
        evaluation_path=settings.PROJECT_ROOT / "evaluation",
        sequence=sequence,
    )
