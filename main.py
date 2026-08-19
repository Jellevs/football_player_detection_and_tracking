import pickle
from pathlib import Path

from tqdm import tqdm

import settings
from utils.run_evaluation import run_evaluation
from utils.build import build_paths, build_configs, build_connector
from utils.data_utils import load_images, organize_detections_by_track
from utils.visualization_utils import visualize_tracklets
from utils.evaluation_utils import save_mot_file_for_sn_trackeval
from detect_and_track.detect_and_track import detect_and_track
from detect_and_track.trackers.Deep_EIoU import DeepEIOUTracker
from attributes.attributes import predict_attributes
from tracklets.split_tracklets import split_tracklets


def reset_connector_logs():
    """Start each full evaluation with one clean set of connector decisions."""
    for connector_arguments in settings.CONNECTORS.values():
        log_path = connector_arguments.get("log_path")
        if log_path is not None:
            log_path.unlink(missing_ok=True)


def split_with_cache(tracklets, splitter_config, sequence):
    """Splitting dominates the runtime, so the result is reused across connectors."""
    cache_path = settings.OUTPUT_ROOT / "cache_split" / f"cache_split_{sequence}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as cache_file:
            return pickle.load(cache_file)

    split = split_tracklets(tracklets, splitter_config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as cache_file:
        pickle.dump(split, cache_file)
    return split


def process_sequence(sequence, tracker_config, jersey_config, splitter_config, connector, device, method_name):
    paths = build_paths(sequence)

    tracker = DeepEIOUTracker(
        track_thresh=tracker_config.track_thresh,
        track_low_thresh=tracker_config.track_low_thresh,
        new_track_thresh=tracker_config.new_track_thresh,
        track_buffer=tracker_config.track_buffer,
        match_thresh=tracker_config.match_thresh,
        proximity_thresh=tracker_config.proximity_thresh,
        appearance_thresh=tracker_config.appearance_thresh,
        with_reid=tracker_config.with_reid,
        reid_model_name=tracker_config.reid_model_name,
        reid_model_path=str(paths.reid_model_path),
        frame_rate=tracker_config.frame_rate,
    )

    images = load_images(img_dir=paths.img_path)
    tracked_detections = detect_and_track(images, tracker, paths)
    tracklets = organize_detections_by_track(tracked_detections)

    tracklets = predict_attributes(images, tracklets, paths, jersey_config, device)
    tracklets = split_with_cache(tracklets, splitter_config, sequence)
    tracklets = connector.merge(tracklets, sequence_name=sequence)

    save_mot_file_for_sn_trackeval(
        tracklets_dict=tracklets,
        output_path=paths.evaluation_path,
        sequence_name=sequence,
        method_name=method_name,
    )

    if settings.SAVE_VISUALIZATION:
        visualize_tracklets(
            images=images,
            tracklets_dict=tracklets,
            output_path=paths.output_path / "videos" / f"{sequence}_{settings.CONNECTOR}.mp4",
            title=f"{sequence} ({settings.CONNECTOR} connector)",
        )


if __name__ == "__main__":
    device, tracker_config, jersey_config, splitter_config = build_configs()
    connector = build_connector(settings.CONNECTOR)

    reset_connector_logs()
    sequences = sorted(directory.name for directory in settings.DATA_ROOT.iterdir() if directory.is_dir())
    for sequence in tqdm(sequences, desc="Processing sequences"):
        process_sequence(
            sequence,
            tracker_config=tracker_config,
            jersey_config=jersey_config,
            splitter_config=splitter_config,
            connector=connector,
            device=device,
            method_name=settings.METHOD_NAME,
        )

    run_evaluation(settings.METHOD_NAME, settings.EVAL_SPLIT)
