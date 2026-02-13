# import sys
# from pathlib import Path
# from typing import Dict, Optional


# def tracklets_to_mot_format(tracklets, output_path):
#     """
#     Convert tracklets to MOT Challenge format.
    
#     Format: <frame_id>,<track_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>
    
#     Args:
#         tracklets: Dict of {track_id: Tracklet}
#         output_path: Path to save the MOT format file
        
#     Returns:
#         seq_length: Number of frames in sequence
#     """
#     output_path = Path(output_path)
#     output_path.parent.mkdir(parents=True, exist_ok=True)
    
#     lines = []
#     max_frame = 0
    
    
#     for track_id, tracklet in tracklets.items():
#         for i, frame_idx in enumerate(tracklet.frames):
#             bbox = tracklet.bboxes[i]
            
#             # Convert from xyxy to xywh
#             x1, y1, x2, y2 = bbox
#             width = x2 - x1
#             height = y2 - y1
            
#             # MOT format is 1-indexed for frames
#             frame_1indexed = frame_idx + 1
            
#             # Format: frame,id,bb_left,bb_top,bb_width,bb_height
#             line = f"{frame_1indexed},{track_id},{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},1,-1,-1,-1"
#             lines.append((frame_1indexed, track_id, line))
#             max_frame = max(max_frame, frame_1indexed)
    
#     # Sort by frame, then by track_id
#     lines.sort(key=lambda x: (x[0], x[1]))
    
#     # Write to file
#     with open(output_path, 'w') as f:
#         for _, _, line in lines:
#             f.write(line + '\n')
    
#     return max_frame


# def gt_tracklets_to_mot_format(tracklets, output_path):
#     """
#     Convert ground truth from tracklets to MOT format.
    
#     Uses gt_attributes['track_ids'] as the ground truth track IDs.
#     Ensures each (frame, gt_id) pair appears only once.
    
#     Args:
#         tracklets: Dict of {track_id: Tracklet} with gt_attributes
#         output_path: Path to save the MOT format file
        
#     Returns:
#         seq_length: Number of frames in sequence
#     """
#     output_path = Path(output_path)
#     output_path.parent.mkdir(parents=True, exist_ok=True)
    
#     # Use dict to ensure unique (frame, gt_id) pairs
#     gt_detections = {}
#     max_frame = 0
    
#     for track_id, tracklet in tracklets.items():
#         gt_track_ids = tracklet.gt_attributes.get('track_ids', [])
        
#         for i, frame_idx in enumerate(tracklet.frames):
#             # Get GT track ID for this detection
#             if i < len(gt_track_ids) and gt_track_ids[i] != -1:
#                 gt_id = int(gt_track_ids[i])
#             else:
#                 continue  # Skip detections without valid GT
            
#             bbox = tracklet.bboxes[i]
            
#             # Convert from xyxy to xywh
#             x1, y1, x2, y2 = bbox
#             width = x2 - x1
#             height = y2 - y1
            
#             # MOT format is 1-indexed for frames
#             frame_1indexed = frame_idx + 1
            
#             # Only keep first occurrence of each (frame, gt_id) pair
#             key = (frame_1indexed, gt_id)
#             if key not in gt_detections:
#                 gt_detections[key] = (x1, y1, width, height)
            
#             max_frame = max(max_frame, frame_1indexed)
    
#     # Convert to sorted list of lines
#     lines = []
#     for (frame_1indexed, gt_id), (x1, y1, w, h) in gt_detections.items():
#         line = f"{frame_1indexed},{gt_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,1,-1,-1"
#         lines.append((frame_1indexed, gt_id, line))
    
#     # Sort by frame, then by track_id
#     lines.sort(key=lambda x: (x[0], x[1]))
    
#     # Write to file
#     with open(output_path, 'w') as f:
#         for _, _, line in lines:
#             f.write(line + '\n')
    
#     return max_frame


# def create_seqinfo(seq_name, num_frames, output_path):
#     """Create seqinfo.ini file for a sequence."""
#     content = f"""[Sequence]
# name={seq_name}
# imDir=img1
# frameRate=25
# seqLength={num_frames}
# imWidth=1920
# imHeight=1080
# imExt=.jpg
# """
#     with open(output_path, 'w') as f:
#         f.write(content)


# def save_mot_files_for_sn_trackeval(
#     tracklets_original,
#     tracklets_gta, 
#     tracklets_xgb,
#     tracklets_simple_merge,
#     output_dir,
#     sequence_name
# ):
#     output_dir = Path(output_dir)
    
#     # Convert SNGS-123 to SNMOT-123 for consistent naming
#     seq_number = sequence_name.split('-')[1]
#     snmot_name = f"SNMOT-{seq_number}"
    
#     # ========== GT Structure ==========
#     # gt/SNMOT-test/SNMOT-XXX/gt/gt.txt + seqinfo.ini
#     gt_seq_dir = output_dir / "gt" / "SNMOT-test" / snmot_name
#     gt_subdir = gt_seq_dir / "gt"
#     gt_subdir.mkdir(parents=True, exist_ok=True)
    
#     # Save GT
#     gt_path = gt_subdir / "gt.txt"
#     seq_length = gt_tracklets_to_mot_format(tracklets_original, gt_path)
    
#     # Create seqinfo.ini for GT (use SNMOT name)
#     seqinfo_path = gt_seq_dir / "seqinfo.ini"
#     create_seqinfo(snmot_name, seq_length, seqinfo_path)
    
#     # ========== Tracker Structures ==========
#     # Flat files: trackers/baseline/data/SNMOT-XXX.txt
#     stages = [
#         ("baseline", tracklets_original),
#         ("after_gta", tracklets_gta),
#         ("after_xgb", tracklets_xgb),
#         ("after_simple_merge", tracklets_simple_merge)
#     ]
    
#     for stage_name, tracklets in stages:
#         # Create trackers/baseline/data/ directory
#         tracker_data_dir = output_dir / "trackers" / stage_name / "data"
#         tracker_data_dir.mkdir(parents=True, exist_ok=True)
        
#         # Save tracker file as SNMOT-XXX.txt
#         tracker_path = tracker_data_dir / f"{snmot_name}.txt"
#         tracklets_to_mot_format(tracklets, tracker_path)
    
#     return output_dir


# def save_mot_file_for_sn_trackeval(
#     tracklets_original,
#     tracklets_eval,
#     output_dir,
#     sequence_name
# ):
    
#     output_dir = Path(output_dir)
    
#     # Convert SNGS-123 to SNMOT-123 for consistent naming
#     seq_number = sequence_name.split('-')[1]
#     snmot_name = f"SNMOT-{seq_number}"
    
#     # ========== GT Structure ==========
#     # gt/SNMOT-test/SNMOT-XXX/gt/gt.txt + seqinfo.ini
#     gt_seq_dir = output_dir / "gt" / "SNMOT-test" / snmot_name
#     gt_subdir = gt_seq_dir / "gt"
#     gt_subdir.mkdir(parents=True, exist_ok=True)
    
#     # Save GT
#     gt_path = gt_subdir / "gt.txt"
#     seq_length = gt_tracklets_to_mot_format(tracklets_original, gt_path)
    
#     # Create seqinfo.ini for GT (use SNMOT name)
#     seqinfo_path = gt_seq_dir / "seqinfo.ini"
#     create_seqinfo(snmot_name, seq_length, seqinfo_path)
    
#     # ========== Tracker Structures ==========
#     # Flat files: trackers/baseline/data/SNMOT-XXX.txt
#     stages = [
#         ('baseline', tracklets_original),
#         ('tracklet_eval', tracklets_eval),
#     ]
    
#     for stage_name, stage_tracklets in stages:
#         # Create trackers/baseline/data/ directory
#         tracker_data_dir = output_dir / "trackers" / stage_name / "data"
#         tracker_data_dir.mkdir(parents=True, exist_ok=True)
        
#         # Save tracker file as SNMOT-XXX.txt
#         tracker_path = tracker_data_dir / f"{snmot_name}.txt"
#         tracklets_to_mot_format(stage_tracklets, tracker_path)
    
#     return output_dir


"""
Utilities for exporting tracking results in MOT Challenge format for SoccerNet TrackEval.

MOT Challenge format specification:
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

Where:
- frame: Frame number (1-indexed)
- id: Track ID
- bb_left, bb_top, bb_width, bb_height: Bounding box in pixels
- conf: Detection confidence score
- x, y, z: 3D position (set to -1 for 2D tracking)
"""

from pathlib import Path
from typing import Dict, List
import numpy as np


def save_mot_file_for_sn_trackeval(
    tracklets_dict: Dict,
    output_path: Path,
    sequence_name: str,
    stage_name: str = "baseline"
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
    mot_dir = output_path / "SNMOT" / stage_name / "data"
    mot_dir.mkdir(parents=True, exist_ok=True)
    
    seq_number = sequence_name.split('-')[1]
    snmot_name = f"SNMOT-{seq_number}"

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


def save_mot_files_for_sn_trackeval(
    tracklets_dict_stages: Dict[str, Dict],
    output_path: Path,
    sequence_name: str
):
    """
    Save multiple pipeline stages to MOT format for SoccerNet TrackEval.
    
    Args:
        tracklets_dict_stages: Dictionary mapping stage names to tracklets
                              e.g., {'baseline': tracklets1, 'after_split': tracklets2, 'after_merge': tracklets3}
        output_path: Base output directory path
        sequence_name: Name of the sequence (e.g., 'SNGS-123')
    
    Returns:
        Dictionary mapping stage names to saved file paths
    """
    saved_files = {}
    
    for stage_name, tracklets_dict in tracklets_dict_stages.items():
        mot_file_path = save_mot_file_for_sn_trackeval(
            tracklets_dict=tracklets_dict,
            output_path=output_path,
            sequence_name=sequence_name,
            stage_name=stage_name
        )
        saved_files[stage_name] = mot_file_path
    
    print(f"\nSaved {len(saved_files)} MOT files for sequence {sequence_name}:")
    for stage_name, path in saved_files.items():
        print(f"  {stage_name}: {path}")
    
    return saved_files


def create_seqinfo_ini(
    output_path: Path,
    sequence_name: str,
    num_frames: int,
    img_width: int = 1920,
    img_height: int = 1080,
    frame_rate: int = 25
):
    """
    Create seqinfo.ini file required by TrackEval for sequence metadata.
    
    Args:
        output_path: Base output directory path
        sequence_name: Name of the sequence
        num_frames: Total number of frames in the sequence
        img_width: Image width in pixels
        img_height: Image height in pixels
        frame_rate: Frame rate of the video
    
    Returns:
        Path to the created seqinfo.ini file
    """
    # Create seqinfo in the sequence directory
    # evaluation/SNMOT/seqmaps/{sequence_name}/seqinfo.ini
    seq_dir = output_path / "SNMOT" / "seqmaps" / sequence_name
    seq_dir.mkdir(parents=True, exist_ok=True)
    
    seqinfo_path = seq_dir / "seqinfo.ini"
    
    content = f"""[Sequence]
name={sequence_name}
imDir=img1
frameRate={frame_rate}
seqLength={num_frames}
imWidth={img_width}
imHeight={img_height}
imExt=.jpg
"""
    
    with open(seqinfo_path, 'w') as f:
        f.write(content)
    
    print(f"Created seqinfo.ini: {seqinfo_path}")
    
    return seqinfo_path


def create_seqmap_file(
    output_path: Path,
    sequence_names: List[str],
    split_name: str = "test"
):
    """
    Create seqmap file listing all sequences for TrackEval.
    
    Args:
        output_path: Base output directory path
        sequence_names: List of sequence names to include
        split_name: Split name (e.g., 'train', 'val', 'test')
    
    Returns:
        Path to the created seqmap file
    """
    seqmap_dir = output_path / "SNMOT" / "seqmaps"
    seqmap_dir.mkdir(parents=True, exist_ok=True)
    
    seqmap_path = seqmap_dir / f"{split_name}.txt"
    
    with open(seqmap_path, 'w') as f:
        f.write("name\n")
        for seq in sequence_names:
            f.write(f"{seq}\n")
    
    print(f"Created seqmap file: {seqmap_path}")
    print(f"  Listed {len(sequence_names)} sequences")
    
    return seqmap_path


def get_tracklets_statistics(tracklets_dict: Dict) -> Dict:
    """
    Get statistics about the tracklets for validation.
    
    Args:
        tracklets_dict: Dictionary of tracklets {track_id: Tracklet}
    
    Returns:
        Dictionary with statistics
    """
    if not tracklets_dict:
        return {
            'num_tracklets': 0,
            'total_detections': 0,
            'avg_tracklet_length': 0,
            'min_tracklet_length': 0,
            'max_tracklet_length': 0,
            'frame_range': (0, 0)
        }
    
    tracklet_lengths = [len(tracklet.frames) for tracklet in tracklets_dict.values()]
    all_frames = []
    total_detections = 0
    
    for tracklet in tracklets_dict.values():
        all_frames.extend(tracklet.frames)
        total_detections += len(tracklet.frames)
    
    stats = {
        'num_tracklets': len(tracklets_dict),
        'total_detections': total_detections,
        'avg_tracklet_length': np.mean(tracklet_lengths),
        'min_tracklet_length': min(tracklet_lengths),
        'max_tracklet_length': max(tracklet_lengths),
        'frame_range': (min(all_frames), max(all_frames)) if all_frames else (0, 0)
    }
    
    return stats


def validate_mot_file(mot_file_path: Path) -> bool:
    """
    Validate that a MOT file is properly formatted.
    
    Args:
        mot_file_path: Path to the MOT file
    
    Returns:
        True if valid, False otherwise
    """
    try:
        with open(mot_file_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            print(f"ERROR: MOT file is empty: {mot_file_path}")
            return False
        
        for i, line in enumerate(lines[:10], 1):  # Check first 10 lines
            parts = line.strip().split(',')
            if len(parts) != 10:
                print(f"ERROR: Line {i} has {len(parts)} fields, expected 10")
                return False
            
            try:
                frame = int(parts[0])
                track_id = int(parts[1])
                conf = float(parts[6])
                
                if frame < 1:
                    print(f"ERROR: Frame number must be >= 1, got {frame}")
                    return False
                
            except ValueError as e:
                print(f"ERROR: Invalid value on line {i}: {e}")
                return False
        
        print(f"✓ MOT file validation passed: {mot_file_path}")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to validate MOT file: {e}")
        return False


def print_mot_summary(tracklets_dict_stages: Dict[str, Dict]):
    """
    Print a summary of tracklets across different pipeline stages.
    
    Args:
        tracklets_dict_stages: Dictionary mapping stage names to tracklets
    """
    print("\n" + "="*80)
    print("MOT EXPORT SUMMARY")
    print("="*80)
    
    for stage_name, tracklets_dict in tracklets_dict_stages.items():
        stats = get_tracklets_statistics(tracklets_dict)
        print(f"\n{stage_name.upper()}:")
        print(f"  Tracklets: {stats['num_tracklets']}")
        print(f"  Total detections: {stats['total_detections']}")
        print(f"  Avg tracklet length: {stats['avg_tracklet_length']:.1f}")
        print(f"  Length range: [{stats['min_tracklet_length']}, {stats['max_tracklet_length']}]")
        print(f"  Frame range: [{stats['frame_range'][0]}, {stats['frame_range'][1]}]")
    
    print("="*80 + "\n")