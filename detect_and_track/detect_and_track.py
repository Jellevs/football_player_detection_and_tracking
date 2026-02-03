from tqdm import tqdm
import cv2
import pickle
import json
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm


def load_annotations_by_frame(json_path):
    """ Load annotations organized by frame filename """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    id_to_filename = {
        img['image_id']: img['file_name'] 
        for img in data['images']
    }
    
    frame_annotations = {}
    for ann in data['annotations']:
        # Only keep players, goalkeepers and referees
        if ann.get('category_id') not in [1, 2, 3]:
            continue
            
        filename = id_to_filename[ann['image_id']]
        
        if filename not in frame_annotations:
            frame_annotations[filename] = []
        frame_annotations[filename].append(ann)
    
    return frame_annotations


def create_detections(annotations):
    """ Convert annotations list to supervision.Detections object """
    if not annotations:
        return sv.Detections.empty()
    
    bboxes = []
    track_ids = []
    jerseys = []
    teams = []
    roles = []
    class_ids = []  # ← ADD THIS
    
    for annotation in annotations:
        # Only keep players, goalkeepers and referees
        if annotation.get('category_id') not in [1, 2, 3]:
            continue
        
        b = annotation['bbox_image']
        
        # ========== MINIMAL FILTERING ==========
        # Only filter truly broken annotations
        
        # Filter 1: Absurdly small (annotation errors)
        if b['w'] < 10 or b['h'] < 10:
            continue
        
        # Filter 2: Impossible aspect ratios (lines/artifacts)
        if b['h'] > 0:
            aspect_ratio = b['w'] / b['h']
            if aspect_ratio < 0.1 or aspect_ratio > 5.0:
                continue
        
        # Filter 3: Area check for complete garbage
        area = b['w'] * b['h']
        if area < 200:
            continue
        
        # =======================================
        # If we get here, keep this detection
        
        bboxes.append([b['x'], b['y'], b['x'] + b['w'], b['y'] + b['h']])
        track_ids.append(annotation.get('track_id', -1))
        jerseys.append(annotation.get('attributes', {}).get('jersey', -1))
        teams.append(annotation.get('attributes', {}).get('team', -1))
        roles.append(annotation.get('attributes', {}).get('role', -1))
        class_ids.append(annotation['category_id'])  # ← ADD THIS
    
    if not bboxes:
        return sv.Detections.empty()
    
    detections = sv.Detections(
        xyxy=np.array(bboxes, dtype=np.float32),
        confidence=np.ones(len(bboxes), dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),  # ← USE THIS instead of list comprehension
    )
    
    detections.data = {
        'gt_track_id': np.array(track_ids),
        'gt_jersey': np.array(jerseys),
        'gt_team': np.array(teams),
        'gt_role': np.array(roles),
    }
    
    return detections


def detect_and_track(images, tracker, paths):
    """ Run tracker on ground truth detections """
    cache_path = paths.set_cache_path("tracked_detections", paths.sequence)

    if cache_path and cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    frame_annotations = load_annotations_by_frame(paths.gt_detections_path)
    
    all_tracked_detections = []
    for frame_idx, image_path in tqdm(enumerate(images), total=len(images), desc="Loading detections and tracks"):
        image = cv2.imread(str(image_path))
        filename = image_path.name
        
        # Get detections for this frame
        annotations = frame_annotations.get(filename, [])
        detections = create_detections(annotations)
        
        # Track
        tracked_detections = tracker.update(detections, image)
        
        all_tracked_detections.append({
            'frame_idx': frame_idx,
            'filename': filename,
            'tracked_detections': tracked_detections,
        })
    
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(all_tracked_detections, f)
    
    return all_tracked_detections