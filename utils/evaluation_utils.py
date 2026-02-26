from pathlib import Path
from typing import Dict


def save_mot_file_for_sn_trackeval(
    tracklets_dict: Dict,
    output_path: Path,
    sequence_name: str,
    method_name: str = "baseline"
):
    """
    Save tracklets to a single MOT format file for SoccerNet TrackEval.
    
    Args:
        tracklets_dict: Dictionary of tracklets {track_id: Tracklet}
        output_path: Base output directory path
        sequence_name: Name of the sequence (e.g., 'SNGS-123')
        stage_name: Pipeline stage name (e.g., 'baseline', 'after_split', 'after_merge')
    
    Returns:
        Path to the saved MOT file
    """
    # Create output directory structure
    # evaluation/SNMOT/{stage_name}/data/{sequence_name}.txt
    mot_dir = output_path / "SNPT" / method_name / "data"
    mot_dir.mkdir(parents=True, exist_ok=True)
    
    seq_number = sequence_name.split('-')[1]
    snmot_name = f"SNPT-{seq_number}"

    mot_file_path = mot_dir / f"{snmot_name}.txt"


    # Collect all detections across tracklets
    all_detections = []
    
    for track_id, tracklet in tracklets_dict.items():
        for i in range(len(tracklet.frames)):
            frame_idx = tracklet.frames[i]
            bbox = tracklet.bboxes[i]  # [x1, y1, x2, y2]
            score = tracklet.scores[i]
            
            # Convert from [x1, y1, x2, y2] to [left, top, width, height]
            bb_left = bbox[0]
            bb_top = bbox[1]
            bb_width = bbox[2] - bbox[0]
            bb_height = bbox[3] - bbox[1]
            
            # MOT format: frame (1-indexed), id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
            # Frame is 1-indexed in MOT format
            detection = [
                frame_idx + 1,  # Convert from 0-indexed to 1-indexed
                track_id,
                bb_left,
                bb_top,
                bb_width,
                bb_height,
                score,
                -1,  # x (3D position, unused)
                -1,  # y (3D position, unused)
                -1   # z (3D position, unused)
            ]
            
            all_detections.append(detection)
    
    # Sort by frame, then by track_id for consistency
    all_detections.sort(key=lambda x: (x[0], x[1]))
    
    # Write to file
    with open(mot_file_path, 'w') as f:
        for det in all_detections:
            # Format: frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z
            line = ','.join([f"{val:.6f}" if isinstance(val, float) else str(int(val)) 
                           for val in det])
            f.write(line + '\n')
    
    print(f"Saved MOT file: {mot_file_path}")
    print(f"  Total detections: {len(all_detections)}")
    print(f"  Total tracks: {len(tracklets_dict)}")
    
    return mot_file_path
