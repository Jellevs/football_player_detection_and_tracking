import cv2
import supervision as sv
import numpy as np 
from tqdm import tqdm


def visualize_tracklets(images, tracklets_dict, output_path, title="tracklets", gt_track_ids=None):
    """ Visualize tracklets with predictions. """

    frame_to_detections = tracklets_to_frame_detections(tracklets_dict)
    
    # Create video
    first_frame = cv2.imread(str(images[0]))
    height, width = first_frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 25
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    GREEN = sv.Color(r=0, g=255, b=0)

    bbox_annotator = sv.BoxAnnotator(
        color=GREEN,
        color_lookup=sv.ColorLookup.INDEX,
        thickness=2
    )
    
    label_annotator = sv.LabelAnnotator(
        color=GREEN,
        color_lookup=sv.ColorLookup.INDEX,
        text_color=sv.Color(r=0, g=0, b=0),
        text_scale=0.4,
        text_thickness=1,
        text_padding=1,
        border_radius=1
    )

    for frame_idx, image_path in tqdm(enumerate(images), total=len(images), desc=f"Visualizing {title}"):
        image = cv2.imread(str(image_path))
        
        detections_list = frame_to_detections.get(frame_idx, [])
        
        if not detections_list:
            out.write(image)
            continue

        xyxy = np.array([d['bbox'] for d in detections_list])
        confidence = np.array([d['score'] for d in detections_list])
        tracker_id = np.array([d['tracker_id'] for d in detections_list])

        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            tracker_id=tracker_id
        )
        
        labels = []
        for detection in detections_list:

            tracker_id = detection['tracker_id']
            parent_id = detection.get('parent_id')
            
            # Ground Truth
            gt_jersey = detection.get('gt_jersey', 'X')
            gt_team = detection.get('gt_team', 'X')
            gt_role = detection.get('gt_role', 'X')
            
            # Per-frame predictions
            pred_jersey_frame = detection.get('pred_jersey')
            pred_jersey_frame = str(pred_jersey_frame) if pred_jersey_frame is not None else "X"
            
            pred_team_frame = detection.get('pred_team')
            pred_team_frame = str(pred_team_frame) if pred_team_frame is not None else "X"

            pred_team_confidence = detection.get('pred_jersey_entropies')
            pred_team_confidence = f"{pred_team_confidence:.4g}" if pred_team_confidence is not None else "X"
            
            gt_id = detection.get('gt_track_id')
            gt_str = f" | GT:{gt_id}" if gt_id is not None else ""
            label = f"ID:{tracker_id}"

            labels.append(label)

        annotated = bbox_annotator.annotate(image.copy(), detections)
        annotated = label_annotator.annotate(annotated, detections, labels=labels)
        
        out.write(annotated)
    
    out.release()
    print(f"Video saved to: {output_path}")


def tracklets_to_frame_detections(tracklets_dict):
    """ Convert tracklets to frame-by-frame detections. """
    frame_to_detections = {}

    for tracker_id, tracklet in tracklets_dict.items():
        gt_track_ids_per_frame = tracklet.gt_attributes.get('track_ids', [])
        for i in range(len(tracklet.frames)):
            frame_idx = tracklet.frames[i]

            if frame_idx not in frame_to_detections:
                frame_to_detections[frame_idx] = []

            detection_data = {
                'tracker_id': tracker_id,
                'bbox': tracklet.bboxes[i],
                'score': tracklet.scores[i],
            }

            if i < len(gt_track_ids_per_frame) and gt_track_ids_per_frame[i]:
                detection_data['gt_track_id'] = gt_track_ids_per_frame[i]
            

            # Ground Truth attributes
            gt_jerseys = tracklet.gt_attributes.get('jerseys')
            if gt_jerseys is not None and len(gt_jerseys) > 0 and i < len(gt_jerseys):
                gt_jersey = gt_jerseys[i]
                if gt_jersey and gt_jersey != 'unknown':
                    detection_data['gt_jersey'] = str(gt_jersey)

            gt_teams = tracklet.gt_attributes.get('teams')
            if gt_teams is not None and len(gt_teams) > 0 and i < len(gt_teams):
                gt_team = gt_teams[i]
                if gt_team and gt_team != 'unknown':
                    detection_data['gt_team'] = str(gt_team)
            
            gt_roles = tracklet.gt_attributes.get('roles')
            if gt_roles is not None and len(gt_roles) > 0 and i < len(gt_roles):
                gt_role = gt_roles[i]
                if gt_role and gt_role != 'unknown':
                    detection_data['gt_role'] = str(gt_role)


            # Per-frame predictions
            jerseys = tracklet.pred_attributes.get('jerseys', [])
            entropies = tracklet.pred_attributes.get('jersey_entropies', [])
            if jerseys is not None and len(jerseys) > 0 and i < len(jerseys):
                jersey = jerseys[i]
                entropy = entropies[i]
                if not (isinstance(jersey, float) and np.isnan(jersey)):
                    detection_data['pred_jersey'] = int(jersey)
                    detection_data['pred_jersey_entropies'] = float(entropy)

            teams = tracklet.pred_attributes.get('teams', [])
            if teams is not None and len(teams) > 0 and i < len(teams):
                team = teams[i]
                if not (isinstance(team, float) and np.isnan(team)):
                    detection_data['pred_team'] = int(team)
            
            
            frame_to_detections[frame_idx].append(detection_data)

    return frame_to_detections
