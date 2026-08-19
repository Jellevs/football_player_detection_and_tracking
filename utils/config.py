from dataclasses import dataclass
from pathlib import Path


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
    reid_threshold: float = 3.5   # N in the centroid-ReID outlier filter, following the paper default
    reid_rounds: int = 3          # K in the same filter
    debug_tracklet_id: int = None
    debug_dir: Path = None


@dataclass
class SplitterConfig:
    jersey_min_persistence: int = 20
    jersey_lookahead: int = 100
    jersey_entropy_threshold: float = 0.01
    jersey_min_persistence_ratio: float = 0.9

    team_min_persistence: int = 10
    team_min_persistence_ratio: float = 0.9
    team_lookahead: int = 100
    team_confidence_threshold: float = 0.6

    temporal_reid_min_gap_frames: int = 5
    temporal_reid_threshold: float = 0.15
    temporal_reid_min_segment_frames: int = 5
    temporal_reid_n_samples: int = 20

    bbox_lookback_window: int = 20
    bbox_lookahead_window: int = 10
    bbox_std_threshold: float = 4.0
    bbox_min_spike_velocity: float = 35.0
    bbox_max_spike_duration: int = 3

    proximity_distance: float = 50.0
    min_overlap_frames: int = 3
    velocity_window: int = 5
    min_velocity_change: float = 30.0
    direction_change_threshold: float = 120.0
    swap_similarity_threshold: float = 0.95

    min_fragment_length: int = 20
