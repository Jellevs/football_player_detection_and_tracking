from pathlib import Path

# Evaluation
METHOD_NAME = "bbox_std_3"
EVAL_SPLIT = "test"  # "train" or "test"

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

SPLITTER = dict(
    # Jersey
    jersey_min_persistence=20,
    jersey_lookahead=100,
    jersey_min_persistence_ratio=0.9,
    jersey_entropy_threshold=0.01,

    # Team
    team_min_persistence_ratio=0.9,
    team_lookahead=100,
    team_confidence_threshold=0.6,


    # STR
    temporal_reid_min_gap_frames=5,
    temporal_reid_threshold=0.1,
    temporal_reid_min_segment_frames=5,
    temporal_reid_n_samples=20,

    # Bounding Box
    bbox_lookback_window = 20,
    bbox_lookahead_window = 10,
    bbox_std_threshold = 3,
    bbox_min_spike_velocity = 35.0,
    bbox_max_spike_duration = 3,

    # Trajectory
    proximity_distance = 50,  # pixels - bboxes closer than this (was 50)
    velocity_window = 10,       # frames before/after to calculate velocity
    min_velocity_change = 30.0, # pixels/frame - minimum speed change (was 20.0)
    direction_change_threshold = 120,  # degrees - minimum direction change
    swap_similarity_threshold = 0.9, 

    # Min fragment length
    min_fragment_length=10,
)


# METHOD_NAME = f"lookback_{SPLITTER['bbox_lookback_window']}_lookahead_{SPLITTER['bbox_lookahead_window']}_std{SPLITTER['bbox_std_threshold']}_minspike{SPLITTER['bbox_min_spike_velocity']}_maxspike{SPLITTER['bbox_max_spike_duration']}"
# METHOD_NAME = f"mingapframes{SPLITTER['temporal_reid_min_gap_frames']}_thresh{SPLITTER['temporal_reid_threshold']}_minsegframes{SPLITTER['temporal_reid_min_segment_frames']}_nduration{SPLITTER['temporal_reid_n_samples']}"
# METHOD_NAME = f"persistenceratio{SPLITTER['team_min_persistence_ratio']}_lookahead{SPLITTER['team_lookahead']}_confthresh{SPLITTER['team_confidence_threshold']}"
# METHOD_NAME = f"minpersistence{SPLITTER['jersey_min_persistence']}_lookahead{SPLITTER['jersey_lookahead']}_persistenceratio{SPLITTER['jersey_min_persistence_ratio']}_entropy{SPLITTER['jersey_entropy_threshold']}"
METHOD_NAME = f"proximity_distance{SPLITTER['proximity_distance']}_velocity_window{SPLITTER['velocity_window']}_min_velocity_change{SPLITTER['min_velocity_change']}_direction_change_threshold{SPLITTER['direction_change_threshold']}_swap_similarity_threshold{SPLITTER['swap_similarity_threshold']}"


# SPLITTER = dict( 88.849
#     jersey_min_persistence=20,
#     jersey_lookahead=150,
#     jersey_min_persistence_ratio=0.9,
#     jersey_entropy_threshold=0.02,
#     team_min_persistence=5,
#     team_min_persistence_ratio=0.8,
#     team_lookahead=100,
#     team_confidence_threshold=0.6,
#     min_fragment_length=20,
#     temporal_reid_min_gap_frames=5,
#     temporal_reid_threshold=0.15,
#     temporal_reid_min_segment_frames=5,
#     temporal_reid_n_samples=20,
# )

# SPLITTER = dict( 88.698
#     jersey_min_persistence=20,
#     jersey_lookahead=150,
#     jersey_min_persistence_ratio=0.9,
#     jersey_entropy_threshold=0.02,
#     team_min_persistence=10,
#     team_min_persistence_ratio=0.9,
#     team_lookahead=150,
#     team_confidence_threshold=0.7,
#     min_fragment_length=20,
#     temporal_reid_min_gap_frames=5,
#     temporal_reid_threshold=0.15,
#     temporal_reid_min_segment_frames=5,
#     temporal_reid_n_samples=20,
# )

# SPLITTER = dict( 88.948
#     jersey_min_persistence=20,
#     jersey_lookahead=150,
#     jersey_min_persistence_ratio=0.9,
#     jersey_entropy_threshold=0.02,
#     team_min_persistence=10,
#     team_min_persistence_ratio=0.9,
#     team_lookahead=100,
#     team_confidence_threshold=0.7,
#     min_fragment_length=20,
#     temporal_reid_min_gap_frames=5,
#     temporal_reid_threshold=0.15,
#     temporal_reid_min_segment_frames=5,
#     temporal_reid_n_samples=20,
# )



















# SPLITTER = dict( 88.278
#     jersey_min_persistence=20,
#     jersey_lookahead=100,
#     jersey_min_persistence_ratio=0.8,
#     jersey_entropy_threshold=0.01,
#     team_min_persistence=5,
#     team_min_persistence_ratio=0.8,
#     team_lookahead=100,
#     team_confidence_threshold=0.6,
#     min_fragment_length=20,
#     temporal_reid_min_gap_frames=5,
#     temporal_reid_threshold=0.15,
#     temporal_reid_min_segment_frames=5,
#     temporal_reid_n_samples=20,
# )




MERGER = dict(
    merge_threshold=0.5,
    linkage_method="average",
    max_temporal_gap=100,
)

# Pre-attribute splitting flags
# Run these splitters BEFORE predict_attributes() so the team classifier
# sees purer (more identity-homogeneous) tracklets.
# Order: bbox → STR → attribute prediction
PRE_ATTRIBUTE_BBOX_SPLIT = True   # ← toggle bbox split before attribute prediction
PRE_ATTRIBUTE_STR_SPLIT  = True   # ← toggle spatio-temporal ReID split before attribute prediction