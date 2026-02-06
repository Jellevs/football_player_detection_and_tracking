import sys
from pathlib import Path
from typing import Dict, Optional


def tracklets_to_mot_format(tracklets, output_path):
    """
    Convert tracklets to MOT Challenge format.
    
    Format: <frame_id>,<track_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>
    
    Args:
        tracklets: Dict of {track_id: Tracklet}
        output_path: Path to save the MOT format file
        
    Returns:
        seq_length: Number of frames in sequence
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = []
    max_frame = 0
    
    for track_id, tracklet in tracklets.items():
        for i, frame_idx in enumerate(tracklet.frames):
            bbox = tracklet.bboxes[i]
            
            # Convert from xyxy to xywh
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            # MOT format is 1-indexed for frames
            frame_1indexed = frame_idx + 1
            
            # Format: frame,id,bb_left,bb_top,bb_width,bb_height
            line = f"{frame_1indexed},{track_id},{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},1,-1,-1,-1"
            lines.append((frame_1indexed, track_id, line))
            max_frame = max(max_frame, frame_1indexed)
    
    # Sort by frame, then by track_id
    lines.sort(key=lambda x: (x[0], x[1]))
    
    # Write to file
    with open(output_path, 'w') as f:
        for _, _, line in lines:
            f.write(line + '\n')
    
    return max_frame


def gt_tracklets_to_mot_format(tracklets, output_path):
    """
    Convert ground truth from tracklets to MOT format.
    
    Uses gt_attributes['track_ids'] as the ground truth track IDs.
    Ensures each (frame, gt_id) pair appears only once.
    
    Args:
        tracklets: Dict of {track_id: Tracklet} with gt_attributes
        output_path: Path to save the MOT format file
        
    Returns:
        seq_length: Number of frames in sequence
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use dict to ensure unique (frame, gt_id) pairs
    gt_detections = {}
    max_frame = 0
    
    for track_id, tracklet in tracklets.items():
        gt_track_ids = tracklet.gt_attributes.get('track_ids', [])
        
        for i, frame_idx in enumerate(tracklet.frames):
            # Get GT track ID for this detection
            if i < len(gt_track_ids) and gt_track_ids[i] != -1:
                gt_id = int(gt_track_ids[i])
            else:
                continue  # Skip detections without valid GT
            
            bbox = tracklet.bboxes[i]
            
            # Convert from xyxy to xywh
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            # MOT format is 1-indexed for frames
            frame_1indexed = frame_idx + 1
            
            # Only keep first occurrence of each (frame, gt_id) pair
            key = (frame_1indexed, gt_id)
            if key not in gt_detections:
                gt_detections[key] = (x1, y1, width, height)
            
            max_frame = max(max_frame, frame_1indexed)
    
    # Convert to sorted list of lines
    lines = []
    for (frame_1indexed, gt_id), (x1, y1, w, h) in gt_detections.items():
        line = f"{frame_1indexed},{gt_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,1,-1,-1"
        lines.append((frame_1indexed, gt_id, line))
    
    # Sort by frame, then by track_id
    lines.sort(key=lambda x: (x[0], x[1]))
    
    # Write to file
    with open(output_path, 'w') as f:
        for _, _, line in lines:
            f.write(line + '\n')
    
    return max_frame


def create_seqinfo(seq_name, num_frames, output_path):
    """Create seqinfo.ini file for a sequence."""
    content = f"""[Sequence]
name={seq_name}
imDir=img1
frameRate=25
seqLength={num_frames}
imWidth=1920
imHeight=1080
imExt=.jpg
"""
    with open(output_path, 'w') as f:
        f.write(content)


def save_mot_files_for_sn_trackeval(
    tracklets_original,
    tracklets_split, 
    tracklets_merged,
    output_dir,
    sequence_name
):
    """
    Save MOT files in sn-trackeval format (matching official SoccerNet test data).
    
    Structure created:
        output_dir/
            gt/
                SNMOT-test/
                    SNMOT-XXX/
                        gt/
                            gt.txt
                        seqinfo.ini
            trackers/
                baseline/
                    data/
                        SNMOT-XXX.txt
                after_split/
                    data/
                        SNMOT-XXX.txt
                after_merge/
                    data/
                        SNMOT-XXX.txt
    
    Args:
        tracklets_original: Original tracker output
        tracklets_split: After splitting
        tracklets_merged: After merging
        output_dir: Root directory (e.g., .../tracking-2023/)
        sequence_name: Name of the sequence (e.g., 'SNGS-123')
    """
    output_dir = Path(output_dir)
    
    # Convert SNGS-123 to SNMOT-123 for consistent naming
    seq_number = sequence_name.split('-')[1]
    snmot_name = f"SNMOT-{seq_number}"
    
    # ========== GT Structure ==========
    # gt/SNMOT-test/SNMOT-XXX/gt/gt.txt + seqinfo.ini
    gt_seq_dir = output_dir / "gt" / "SNMOT-test" / snmot_name
    gt_subdir = gt_seq_dir / "gt"
    gt_subdir.mkdir(parents=True, exist_ok=True)
    
    # Save GT
    gt_path = gt_subdir / "gt.txt"
    seq_length = gt_tracklets_to_mot_format(tracklets_original, gt_path)
    
    # Create seqinfo.ini for GT (use SNMOT name)
    seqinfo_path = gt_seq_dir / "seqinfo.ini"
    create_seqinfo(snmot_name, seq_length, seqinfo_path)
    
    # ========== Tracker Structures ==========
    # Flat files: trackers/baseline/data/SNMOT-XXX.txt
    stages = [
        ("baseline", tracklets_original),
        ("after_split", tracklets_split),
        ("after_merge", tracklets_merged),
    ]
    
    for stage_name, tracklets in stages:
        # Create trackers/baseline/data/ directory
        tracker_data_dir = output_dir / "trackers" / stage_name / "data"
        tracker_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save tracker file as SNMOT-XXX.txt
        tracker_path = tracker_data_dir / f"{snmot_name}.txt"
        tracklets_to_mot_format(tracklets, tracker_path)
    
    return output_dir