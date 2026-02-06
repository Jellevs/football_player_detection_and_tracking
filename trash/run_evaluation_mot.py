"""
Run sn-trackeval on MOT format files

This script evaluates tracking results in MOT format (.txt files) using sn-trackeval.

Usage:
    python run_evaluation_mot.py \
        --gt_folder output/mot_files/gt \
        --trackers_folder output/mot_files \
        --trackers baseline after_split after_merge

Directory structure expected:
    gt_folder/
        SNGS-XXX/
            gt.txt
    
    trackers_folder/
        baseline/
            SNGS-XXX/
                SNGS-XXX.txt
        after_split/
            SNGS-XXX/
                SNGS-XXX.txt
        after_merge/
            SNGS-XXX/
                SNGS-XXX.txt
"""

import sys
import os
import argparse
from pathlib import Path

# Add sn-trackeval to path
# MODIFY THIS to point to your sn-trackeval directory
TRACKEVAL_PATH = Path(r"C:\path\to\sn-trackeval")
sys.path.insert(0, str(TRACKEVAL_PATH))

import trackeval


def main():
    parser = argparse.ArgumentParser(
        description="Run sn-trackeval on MOT format tracking results"
    )
    parser.add_argument(
        '--gt_folder',
        type=str,
        required=True,
        help='Path to ground truth folder (e.g., output/mot_files/gt)'
    )
    parser.add_argument(
        '--trackers_folder',
        type=str,
        required=True,
        help='Path to trackers folder (e.g., output/mot_files)'
    )
    parser.add_argument(
        '--trackers',
        nargs='+',
        default=['baseline', 'after_split', 'after_merge'],
        help='List of tracker names to evaluate'
    )
    parser.add_argument(
        '--metrics',
        nargs='+',
        default=['HOTA', 'CLEAR', 'Identity'],
        help='Metrics to compute'
    )
    parser.add_argument(
        '--use_parallel',
        action='store_true',
        help='Use parallel processing'
    )
    parser.add_argument(
        '--num_cores',
        type=int,
        default=8,
        help='Number of cores for parallel processing'
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        default=None,
        help='Where to save results (default: trackers_folder)'
    )
    args = parser.parse_args()
    
    # Convert to absolute paths
    gt_folder = Path(args.gt_folder).absolute()
    trackers_folder = Path(args.trackers_folder).absolute()
    
    if not gt_folder.exists():
        print(f"❌ Ground truth folder not found: {gt_folder}")
        return
    
    if not trackers_folder.exists():
        print(f"❌ Trackers folder not found: {trackers_folder}")
        return
    
    print(f"\n{'='*80}")
    print("sn-trackeval Evaluation")
    print(f"{'='*80}")
    print(f"GT folder:       {gt_folder}")
    print(f"Trackers folder: {trackers_folder}")
    print(f"Trackers:        {', '.join(args.trackers)}")
    print(f"Metrics:         {', '.join(args.metrics)}")
    print(f"Parallel:        {args.use_parallel}")
    print(f"{'='*80}\n")
    
    # Get list of sequences from GT folder
    sequences = sorted([d.name for d in gt_folder.iterdir() if d.is_dir()])
    
    if not sequences:
        print(f"❌ No sequence folders found in {gt_folder}")
        return
    
    print(f"Found {len(sequences)} sequences")
    print(f"First few: {sequences[:5]}")
    print()
    
    # Configure evaluator
    eval_config = {
        'USE_PARALLEL': args.use_parallel,
        'NUM_PARALLEL_CORES': args.num_cores,
        'BREAK_ON_ERROR': False,  # Continue even if one tracker fails
        'PRINT_RESULTS': True,
        'PRINT_ONLY_COMBINED': True,  # Only print combined results
        'PRINT_CONFIG': True,
        'TIME_PROGRESS': True,
        'DISPLAY_LESS_PROGRESS': False,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_DETAILED': True,
        'OUTPUT_EMPTY_CLASSES': True,
        'PLOT_CURVES': False,  # Disable plotting to avoid matplotlib issues
    }
    
    # Configure dataset
    dataset_config = {
        'GT_FOLDER': str(gt_folder.parent),  # Parent of gt/
        'TRACKERS_FOLDER': str(trackers_folder),
        'OUTPUT_FOLDER': args.output_folder if args.output_folder else str(trackers_folder),
        'TRACKERS_TO_EVAL': args.trackers,
        'CLASSES_TO_EVAL': ['pedestrian'],  # Standard class name for players
        'BENCHMARK': 'SoccerNetGS',  # Custom benchmark name
        'SPLIT_TO_EVAL': 'test',  # Or 'valid', 'train'
        'SEQ_INFO': {seq: None for seq in sequences},  # List all sequences
        'DO_PREPROC': False,  # Don't preprocess
        'TRACKER_SUB_FOLDER': '',  # Tracker files directly in tracker folder
        'OUTPUT_SUB_FOLDER': '',
        'TRACKER_DISPLAY_NAMES': None,
    }
    
    # Configure metrics
    metrics_config = {
        'METRICS': args.metrics,
        'THRESHOLD': 0.5,
    }
    
    try:
        # Create evaluator
        evaluator = trackeval.Evaluator(eval_config)
        
        # Create dataset
        # Use MotChallenge2DBox since it handles .txt MOT format
        dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
        
        # Create metrics
        metrics_list = []
        for metric_class in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, 
                            trackeval.metrics.Identity, trackeval.metrics.VACE]:
            if metric_class.get_name() in metrics_config['METRICS']:
                metrics_list.append(metric_class(metrics_config))
        
        if not metrics_list:
            print("❌ No valid metrics selected")
            return
        
        # Run evaluation
        print("\nStarting evaluation...\n")
        raw_results, messages = evaluator.evaluate(dataset_list, metrics_list, show_progressbar=True)
        
        # Print summary
        print(f"\n{'='*80}")
        print("EVALUATION COMPLETE")
        print(f"{'='*80}")
        
        for tracker in args.trackers:
            if tracker in messages['MotChallenge2DBox']:
                msg = messages['MotChallenge2DBox'][tracker]
                if msg == 'Success':
                    print(f"✓ {tracker}: {msg}")
                else:
                    print(f"✗ {tracker}: {msg}")
        
        print(f"\nResults saved to: {dataset_config['OUTPUT_FOLDER']}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()