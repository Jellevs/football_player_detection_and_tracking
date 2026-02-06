# utils/evaluate_soccernet_mot.py
"""
Evaluation utilities for SoccerNet MOT using sn-trackeval.

SoccerNet MOT uses standard MOT Challenge format:
    frame, track_id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1

Usage:
    1. Clone sn-trackeval: git clone https://github.com/SoccerNet/sn-trackeval.git
    2. Install: pip install -e sn-trackeval/
    3. Use functions below to export and evaluate
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional


def tracklets_to_mot_format(tracklets, output_path: Path) -> int:
    """
    Convert tracklets to MOT Challenge format.
    
    Format: frame, track_id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1
    
    Args:
        tracklets: Dict of {track_id: Tracklet}
        output_path: Path to save the MOT format file
        
    Returns:
        max_frame: Maximum frame number (1-indexed)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = []
    max_frame = 0
    
    for track_id, tracklet in tracklets.items():
        for i, frame_idx in enumerate(tracklet.frames):
            bbox = tracklet.bboxes[i]
            conf = tracklet.scores[i] if i < len(tracklet.scores) else 1.0
            
            # Convert from xyxy to xywh
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            # MOT format is 1-indexed for frames
            frame_1indexed = frame_idx + 1
            
            # Format: frame,id,bb_left,bb_top,bb_width,bb_height,conf,-1,-1,-1
            line = f"{frame_1indexed},{track_id},{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},{conf:.4f},-1,-1,-1"
            lines.append((frame_1indexed, track_id, line))
            
            max_frame = max(max_frame, frame_1indexed)
    
    # Sort by frame, then by track_id
    lines.sort(key=lambda x: (x[0], x[1]))
    
    # Write to file
    with open(output_path, 'w') as f:
        for _, _, line in lines:
            f.write(line + '\n')
    
    return max_frame


def gt_tracklets_to_mot_format(tracklets, output_path: Path) -> int:
    """
    Convert ground truth from tracklets to MOT format.
    Uses gt_attributes['track_ids'] as the ground truth track IDs.
    
    Ensures each (frame, gt_id) pair appears only once.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use dict to ensure unique (frame, gt_id) pairs
    gt_detections = {}
    max_frame = 0
    
    for track_id, tracklet in tracklets.items():
        gt_track_ids = tracklet.gt_attributes.get('track_ids', [])
        
        for i, frame_idx in enumerate(tracklet.frames):
            if i < len(gt_track_ids) and gt_track_ids[i] not in [None, -1, 'unknown']:
                gt_id = int(gt_track_ids[i])
            else:
                continue
            
            bbox = tracklet.bboxes[i]
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            frame_1indexed = frame_idx + 1
            
            key = (frame_1indexed, gt_id)
            if key not in gt_detections:
                gt_detections[key] = (x1, y1, width, height)
            
            max_frame = max(max_frame, frame_1indexed)
    
    lines = []
    for (frame_1indexed, gt_id), (x1, y1, w, h) in gt_detections.items():
        line = f"{frame_1indexed},{gt_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1"
        lines.append((frame_1indexed, gt_id, line))
    
    lines.sort(key=lambda x: (x[0], x[1]))
    
    with open(output_path, 'w') as f:
        for _, _, line in lines:
            f.write(line + '\n')
    
    return max_frame


def create_seqinfo(output_path: Path, seq_name: str, seq_length: int, 
                   img_width: int = 1920, img_height: int = 1080, fps: int = 25):
    """Create seqinfo.ini file required by TrackEval."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    content = f"""[Sequence]
name={seq_name}
imDir=img1
frameRate={fps}
seqLength={seq_length}
imWidth={img_width}
imHeight={img_height}
imExt=.jpg
"""
    with open(output_path, 'w') as f:
        f.write(content)


def create_seqmap(output_path: Path, sequences: List[str]):
    """Create seqmap file listing all sequences."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("name\n")
        for seq in sorted(sequences):
            f.write(f"{seq}\n")


def setup_trackeval_structure(
    all_tracklets_by_stage: Dict[str, Dict[str, dict]],
    output_dir: Path,
    split: str = "train"
) -> Path:
    """
    Setup the full TrackEval folder structure for multiple trackers/stages.
    
    Args:
        all_tracklets_by_stage: Dict like:
            {
                "baseline": {"SNGS-001": tracklets, "SNGS-002": tracklets, ...},
                "after_split": {"SNGS-001": tracklets, ...},
                "after_merge": {"SNGS-001": tracklets, ...},
            }
        output_dir: Base output directory
        split: "train", "test", or "challenge"
        
    Returns:
        Path to the data folder
    """
    output_dir = Path(output_dir)
    data_dir = output_dir / "trackeval_data"
    
    benchmark_name = f"SoccerNet-{split}"
    
    # Get all sequence names from the first stage
    first_stage = list(all_tracklets_by_stage.keys())[0]
    sequences = list(all_tracklets_by_stage[first_stage].keys())
    
    print(f"\nSetting up TrackEval structure...")
    print(f"  Benchmark: {benchmark_name}")
    print(f"  Sequences: {len(sequences)}")
    print(f"  Stages: {list(all_tracklets_by_stage.keys())}")
    
    # GT folder structure
    gt_base = data_dir / "gt" / "mot_challenge" / benchmark_name
    
    # Setup GT for each sequence (use baseline tracklets for GT)
    baseline_tracklets = all_tracklets_by_stage[first_stage]
    
    for seq_name in sequences:
        tracklets = baseline_tracklets[seq_name]
        
        # Create GT folder
        seq_gt_dir = gt_base / seq_name / "gt"
        seq_gt_dir.mkdir(parents=True, exist_ok=True)
        
        # Export GT
        gt_path = seq_gt_dir / "gt.txt"
        max_frame = gt_tracklets_to_mot_format(tracklets, gt_path)
        
        # Create seqinfo.ini
        seqinfo_path = gt_base / seq_name / "seqinfo.ini"
        create_seqinfo(seqinfo_path, seq_name, max_frame)
    
    # Create seqmap
    seqmap_dir = gt_base / "seqmaps"
    seqmap_path = seqmap_dir / f"{benchmark_name}.txt"
    create_seqmap(seqmap_path, sequences)
    
    # Setup tracker results for each stage
    for stage_name, stage_tracklets in all_tracklets_by_stage.items():
        tracker_base = data_dir / "trackers" / "mot_challenge" / benchmark_name / stage_name / "data"
        tracker_base.mkdir(parents=True, exist_ok=True)
        
        for seq_name, tracklets in stage_tracklets.items():
            tracker_path = tracker_base / f"{seq_name}.txt"
            tracklets_to_mot_format(tracklets, tracker_path)
    
    print(f"  Output: {data_dir}")
    
    return data_dir


def run_sn_trackeval(
    data_dir: Path,
    split: str = "train",
    trackers_to_eval: List[str] = None,
    sn_trackeval_path: Optional[Path] = None
) -> Dict:
    """
    Run sn-trackeval on the prepared data.
    
    Args:
        data_dir: Path to trackeval_data folder
        split: "train", "test", or "challenge"
        trackers_to_eval: List of tracker names (stages) to evaluate
        sn_trackeval_path: Path to sn-trackeval repo
        
    Returns:
        Dict with evaluation results
    """
    data_dir = Path(data_dir)
    
    if trackers_to_eval is None:
        trackers_to_eval = ["baseline", "after_split", "after_merge"]
    
    # Find sn-trackeval
    if sn_trackeval_path is None:
        possible_paths = [
            Path.cwd() / 'sn-trackeval',
            Path(__file__).parent.parent / 'sn-trackeval',
            Path(__file__).parent / 'sn-trackeval',
        ]
        for p in possible_paths:
            if p.exists() and (p / 'trackeval').exists():
                sn_trackeval_path = p
                break
    
    if sn_trackeval_path is None:
        print("\n" + "="*60)
        print("ERROR: sn-trackeval not found!")
        print("="*60)
        print("\nPlease install it:")
        print("  git clone https://github.com/SoccerNet/sn-trackeval.git")
        print("  pip install -e sn-trackeval/")
        print("\nOr specify the path with --sn_trackeval_path")
        return None
    
    sn_trackeval_path = Path(sn_trackeval_path)
    
    # Add to path
    if str(sn_trackeval_path) not in sys.path:
        sys.path.insert(0, str(sn_trackeval_path))
    
    benchmark_name = f"SoccerNet-{split}"
    
    print(f"\n{'='*60}")
    print("RUNNING SN-TRACKEVAL")
    print(f"{'='*60}")
    print(f"  Benchmark: {benchmark_name}")
    print(f"  Trackers: {trackers_to_eval}")
    
    try:
        import trackeval
        
        # Setup config
        eval_config = trackeval.Evaluator.get_default_eval_config()
        eval_config['DISPLAY_LESS_PROGRESS'] = False
        eval_config['PRINT_RESULTS'] = True
        eval_config['PRINT_ONLY_COMBINED'] = True
        eval_config['OUTPUT_SUMMARY'] = True
        eval_config['OUTPUT_DETAILED'] = True
        eval_config['PLOT_CURVES'] = False
        
        dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dataset_config['GT_FOLDER'] = str(data_dir / "gt" / "mot_challenge")
        dataset_config['TRACKERS_FOLDER'] = str(data_dir / "trackers" / "mot_challenge")
        dataset_config['BENCHMARK'] = benchmark_name
        dataset_config['SPLIT_TO_EVAL'] = split
        dataset_config['TRACKERS_TO_EVAL'] = trackers_to_eval
        dataset_config['DO_PREPROC'] = False
        dataset_config['OUTPUT_FOLDER'] = str(data_dir / "results")
        
        # CRITICAL: Skip the split folder to avoid double naming (SoccerNet-train-train.txt)
        dataset_config['SKIP_SPLIT_FOL'] = True
        
        # Point directly to the seqmap file we created
        seqmap_file = data_dir / "gt" / "mot_challenge" / benchmark_name / "seqmaps" / f"{benchmark_name}.txt"
        dataset_config['SEQMAP_FILE'] = str(seqmap_file)
        
        metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
        
        # Create evaluator
        evaluator = trackeval.Evaluator(eval_config)
        dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
        
        metrics_list = [
            trackeval.metrics.HOTA(metrics_config),
            trackeval.metrics.CLEAR(metrics_config),
            trackeval.metrics.Identity(metrics_config),
        ]
        
        # Run evaluation
        output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)
        
        print(f"\nResults saved to: {data_dir / 'results'}")
        
        return output_res
        
    except Exception as e:
        print(f"\nERROR running sn-trackeval: {e}")
        import traceback
        traceback.print_exc()
        return None