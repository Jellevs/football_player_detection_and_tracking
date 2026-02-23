from pathlib import Path


# Paths
GT_ROOT = Path(r"D:\TUE\Thesis\Code\data\tracking-2023\test\test")
DATA_ROOT = Path(r"D:\TUE\Thesis\Code\data\gamestate-2024\test")
OUTPUT_ROOT = Path(r"D:\TUE\Thesis\Code\tracklet_splitter\output")
WEIGHTS_ROOT = Path(r"D:\TUE\Thesis\Code\tracklet_splitter\weights")
EVALUATION_ROOT = Path(r"D:\TUE\Thesis\Code\tracklet_splitter\evaluation")
PROJECT_ROOT = Path(__file__).parent


# Evaluation
METHOD_NAME = "baseline"
EVAL_SPLIT = "test"  # "train" or "test"


# Configurations
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
    legibility_threshold=0.6,
    reid_threshold_std=0.5,
    referee_threshold=0.5

)


SPLITTER = dict(
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


MERGER = dict(
    merge_threshold=0.5,
    linkage_method="average",
    max_temporal_gap=100,
)