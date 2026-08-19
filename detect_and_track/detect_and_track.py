from tqdm import tqdm
import cv2
import pickle
import numpy as np
import supervision as sv
from collections import defaultdict


def load_annotations_by_frame(gt_txt_path):
    """
    Load MOT-format gt.txt and organize by frame index (0-based).

    MOT format per line:
        frame_id, track_id, x, y, w, h, conf, -1, -1, -1

    frame_id is 1-indexed in MOT format. We convert to 0-based frame index
    to match how images are indexed in the pipeline (images[0] = frame 1).
    """
    frame_annotations = defaultdict(list)

    with open(gt_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 6:
                continue

            frame_id = int(parts[0])
            track_id = int(parts[1])
            x        = float(parts[2])
            y        = float(parts[3])
            w        = float(parts[4])
            h        = float(parts[5])

            # Skip degenerate boxes
            if w < 10 or h < 10:
                continue
            if w * h < 200:
                continue
            if h > 0 and (w / h < 0.1 or w / h > 5.0):
                continue

            # Convert MOT 1-based frame_id to 0-based frame index
            frame_idx = frame_id - 1

            frame_annotations[frame_idx].append({
                'track_id': track_id,
                'bbox': [x, y, x + w, y + h],  # xywh -> xyxy
            })

    return frame_annotations


def create_detections(annotations):
    """
    Convert list of annotation dicts to supervision.Detections object.
    Each annotation has 'track_id' and 'bbox' (xyxy).
    """
    if not annotations:
        return sv.Detections.empty()

    bboxes    = []
    track_ids = []

    for ann in annotations:
        bboxes.append(ann['bbox'])
        track_ids.append(ann['track_id'])

    detections = sv.Detections(
        xyxy=np.array(bboxes, dtype=np.float32),
        confidence=np.ones(len(bboxes), dtype=np.float32),
    )

    # gt_track_id is kept for fragment pair labeling in training data generation
    detections.data = {
        'gt_track_id': np.array(track_ids),
    }

    return detections


def detect_and_track(images, tracker, paths):
    """ Run tracker on ground truth detections from MOT-format gt.txt """
    cache_path = paths.set_cache_path("tracked_detections", paths.sequence)

    if cache_path and cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    frame_annotations = load_annotations_by_frame(paths.gt_detections_path)

    all_tracked_detections = []
    for frame_idx, image_path in tqdm(enumerate(images), total=len(images), desc="Tracking"):
        image = cv2.imread(str(image_path))

        annotations = frame_annotations.get(frame_idx, [])
        detections  = create_detections(annotations)

        tracked_detections = tracker.update(detections, image)

        all_tracked_detections.append({
            'frame_idx': frame_idx,
            'filename':  image_path.name,
            'tracked_detections': tracked_detections,
        })

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(all_tracked_detections, f)

    return all_tracked_detections