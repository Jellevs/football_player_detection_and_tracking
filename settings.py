from pathlib import Path

# Evaluation
METHOD_NAME = "reid_split_5_15"
EVAL_SPLIT = "valid"  # "train" or "test"

# Paths
DATA_ROOT = Path(rf"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\soccernet-player-tracking\{EVAL_SPLIT}")  # Path to the folder with the images

PROJECT_ROOT = Path(__file__).parent
OUTPUT_ROOT = Path(PROJECT_ROOT / "output") 
WEIGHTS_ROOT =  Path(PROJECT_ROOT / "weights")




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
    legibility_threshold=0.5,
    reid_threshold=3.5,
)


# 86.715 HOTA
# SPLITTER = dict(
#     jersey_min_fragment=20,
#     jersey_min_persistence=20,
#     jersey_lookahead=100,
#     jersey_entropy_threshold=0.01,
#     team_min_persistence=5,
#     team_min_fragment=10,
#     team_lookahead=50,
# )

SPLITTER = dict(
    jersey_min_fragment=20,
    jersey_min_persistence=20,
    jersey_lookahead=100,
    jersey_min_persistence_ratio=0.8,
    jersey_entropy_threshold=0.01,
    team_min_persistence=5,
    team_min_persistence_ratio=0.8,
    team_min_fragment=10,
    team_lookahead=100,
)


MERGER = dict(
    merge_threshold=0.5,
    linkage_method="average",
    max_temporal_gap=100,
)