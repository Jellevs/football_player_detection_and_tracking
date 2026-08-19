from pathlib import Path

METHOD_NAME = "xgboost_threshold_0.8"
EVAL_SPLIT = "valid"

PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\soccernet-player-tracking") / EVAL_SPLIT
OUTPUT_ROOT = PROJECT_ROOT / "output"
WEIGHTS_ROOT = PROJECT_ROOT / "weights"

SAVE_VISUALIZATION = False

TRACKER = dict(
    track_thresh=0.6,
    track_low_thresh=0.3,
    new_track_thresh=0.4,
    track_buffer=60,
    match_thresh=0.8,
    proximity_thresh=0.5,
    appearance_thresh=0.2,
    with_reid=True,
    reid_model_name="osnet_x1_0",
    frame_rate=25,
)

JERSEY = dict(
    use_legibility=True,
    use_reid_filter=True,
    use_pose_cropper=True,
    legibility_arch="resnet34",
    legibility_threshold=0.5,
    reid_threshold=3.5,
)

SPLITTER = dict(
    jersey_min_persistence=20,
    jersey_lookahead=100,
    jersey_min_persistence_ratio=0.9,
    jersey_entropy_threshold=0.01,

    team_min_persistence_ratio=0.9,
    team_lookahead=100,
    team_confidence_threshold=0.6,

    temporal_reid_min_gap_frames=5,
    temporal_reid_threshold=0.1,
    temporal_reid_min_segment_frames=5,
    temporal_reid_n_samples=20,

    bbox_lookback_window=20,
    bbox_lookahead_window=10,
    bbox_std_threshold=3,
    bbox_min_spike_velocity=35.0,
    bbox_max_spike_duration=3,

    proximity_distance=50,
    velocity_window=10,
    min_velocity_change=30.0,
    direction_change_threshold=120,
    swap_similarity_threshold=0.9,

    min_fragment_length=10,
)

CONNECTOR = "xgboost"

CONNECTOR_LOG_ROOT = OUTPUT_ROOT / "connector_logs"

# Thresholds come from the validation sweep in the thesis, one per connector.
CONNECTORS = dict(
    xgboost=dict(
        model_path=WEIGHTS_ROOT / "connectors" / "xgboost" / "xgboost_merger.json",
        meta_path=WEIGHTS_ROOT / "connectors" / "xgboost" / "xgboost_merger_meta.json",
        merge_threshold=0.8,
        log_path=CONNECTOR_LOG_ROOT / "xgboost.csv",
    ),
    transformer=dict(
        model_path=WEIGHTS_ROOT / "connectors" / "siamese_cls" / "best_model.pt",
        meta_path=WEIGHTS_ROOT / "connectors" / "siamese_cls" / "transformer_merger_meta.json",
        merge_threshold=0.7,
        log_path=CONNECTOR_LOG_ROOT / "transformer.csv",
    ),
    cross_attention=dict(
        model_path=WEIGHTS_ROOT / "connectors" / "cross_attention" / "best_model.pt",
        meta_path=WEIGHTS_ROOT / "connectors" / "cross_attention" / "transformer_merger_meta.json",
        merge_threshold=0.7,
    ),
    hybrid=dict(
        model_path=WEIGHTS_ROOT / "connectors" / "hybrid" / "best_model.pt",
        meta_path=WEIGHTS_ROOT / "connectors" / "hybrid" / "transformer_merger_meta.json",
        merge_threshold=0.75,
    ),
    pairwise_mlp=dict(
        model_path=WEIGHTS_ROOT / "connectors" / "pairwise_mlp" / "best_model.pt",
        merge_threshold=0.8,
    ),
    decision=dict(
        log_path=CONNECTOR_LOG_ROOT / "decision.csv",
    ),
    oracle=dict(
        gt_root=DATA_ROOT,
        log_path=CONNECTOR_LOG_ROOT / "oracle.csv",
    ),
)
