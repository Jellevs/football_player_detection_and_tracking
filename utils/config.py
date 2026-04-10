# from dataclasses import dataclass
# from pathlib import Path



# @dataclass
# class Roots:
#     DATA_ROOT: Path
#     OUTPUT_ROOT: Path
#     WEIGHTS_ROOT: Path

# @dataclass
# class Paths:
#     img_path: Path
#     gt_detections_path: Path
#     output_path: Path
#     cache_path: Path
#     legibility_model_path: Path
#     reid_model_path: Path
#     parseq_model_path: Path
#     centroid_reid_path: Path
#     siglip_model_path: Path
#     vitpose_model_path: Path
#     evaluation_path: Path
#     sequence: str

#     def set_cache_path(self, name, sequence):
#         return self.cache_path / f"cache_{name}_{sequence}.pkl"
    

# @dataclass
# class TrackerConfig:
#     track_thresh: float = 0.6
#     track_low_thresh: float = 0.3
#     new_track_thresh: float = 0.4
#     track_buffer: int = 60
#     match_thresh: float = 0.8
#     proximity_thresh: float = 0.5
#     appearance_thresh: float = 0.2
#     with_reid: bool = True
#     reid_model_name: str = "osnet_x0_25"
#     frame_rate: int = 25


# @dataclass
# class JerseyPredictorConfig:
#     use_legibility: bool = True
#     legibility_arch: str = "resnet34"
#     legibility_threshold: float = 0.3
#     use_pose_cropper: bool = True
#     use_reid_filter: bool = True
#     reid_threshold_std: float = 2.0
#     debug_tracklet_id: int = None
#     debug_dir: Path = None
#     referee_threshold: float = 0.5


# @dataclass
# class SplitterConfig:
#     # Jersey splitter
#     jersey_min_persistence: int = 5 
#     jersey_lookahead: int = 20
#     jersey_lookback: int = 50
#     jersey_min_pixel_jump: int = 10
#     jersey_entropy_threshold: float = 0.2

#     # Team splitter
#     team_min_persistence: int = 5
#     team_min_fragment: int = 10
#     team_lookahead: int = 50


# @dataclass
# class MergerConfig:
#     xgboost_model_path: Path = None
#     pca_model_path: Path = None
#     merge_threshold: float = 0.5
#     linkage_method: str = 'average'
#     max_temporal_gap: int = 100


from dataclasses import dataclass
from pathlib import Path



@dataclass
class Roots:
    DATA_ROOT: Path
    OUTPUT_ROOT: Path
    WEIGHTS_ROOT: Path

@dataclass
class Paths:
    img_path: Path
    gt_detections_path: Path
    output_path: Path
    cache_path: Path
    legibility_model_path: Path
    reid_model_path: Path
    parseq_model_path: Path
    centroid_reid_path: Path
    siglip_model_path: Path
    vitpose_model_path: Path
    evaluation_path: Path
    sequence: str

    def set_cache_path(self, name, sequence):
        return self.cache_path / f"cache_{name}_{sequence}.pkl"
    

@dataclass
class TrackerConfig:
    track_thresh: float = 0.6
    track_low_thresh: float = 0.3
    new_track_thresh: float = 0.4
    track_buffer: int = 60
    match_thresh: float = 0.8
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.2
    with_reid: bool = True
    reid_model_name: str = "osnet_x0_25"
    frame_rate: int = 25


@dataclass
class JerseyPredictorConfig:
    use_legibility: bool = True
    legibility_arch: str = "resnet34"
    legibility_threshold: float = 0.3
    use_pose_cropper: bool = True
    use_reid_filter: bool = True
    reid_threshold: float = 3.5     # Changed from 2.0 — Paper: N=3.5 (gaussian_outliers.py default)
    reid_rounds: int = 3                # Added — Paper: K=3 (gaussian_outliers.py default)
    debug_tracklet_id: int = None
    debug_dir: Path = None


@dataclass
class SplitterConfig:
    # Jersey splitter
    jersey_min_persistence: int = 5
    jersey_lookahead: int = 20
    jersey_entropy_threshold: float = 0.2
    jersey_min_persistence_ratio: float = 0.8

    # Team splitter
    team_min_persistence: int = 5
    team_min_persistence_ratio: float = 0.8
    team_lookahead: int = 50
    team_confidence_threshold: float = 0.6  # ignore team predictions below this confidence

    min_fragment_length: int = 20




@dataclass
class MergerConfig:
    xgboost_model_path: Path = None
    pca_model_path: Path = None
    merge_threshold: float = 0.5
    linkage_method: str = 'average'
    max_temporal_gap: int = 100