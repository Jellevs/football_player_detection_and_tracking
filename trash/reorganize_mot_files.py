"""
Reorganize MOT Files to sn-trackeval Format

This script takes your current structure:
    evaluation/
        SNGS-XXX/
            baseline.txt
            after_split.txt
            after_merge.txt
            gt.txt

And reorganizes it to:
    evaluation/
        gt/
            SNGS-XXX/
                gt.txt
        baseline/
            SNGS-XXX/
                SNGS-XXX.txt
        after_split/
            SNGS-XXX/
                SNGS-XXX.txt
        after_merge/
            SNGS-XXX/
                SNGS-XXX.txt

Usage:
    python reorganize_mot_files.py --input evaluation --output mot_files
"""

import argparse
import shutil
from pathlib import Path


def reorganize_mot_files(input_dir, output_dir, dry_run=False):
    """
    Reorganize MOT files from per-sequence structure to per-tracker structure.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"Reorganizing MOT Files")
    print(f"{'='*80}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Mode:   {'DRY RUN (no files modified)' if dry_run else 'WRITING FILES'}")
    print(f"{'='*80}\n")
    
    # Find all sequence directories
    sequence_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    if not sequence_dirs:
        print("❌ No sequence directories found")
        return
    
    print(f"Found {len(sequence_dirs)} sequences\n")
    
    # Track statistics
    stats = {
        'gt': 0,
        'baseline': 0,
        'after_split': 0,
        'after_merge': 0,
        'errors': []
    }
    
    # Process each sequence
    for seq_dir in sorted(sequence_dirs):
        seq_name = seq_dir.name
        print(f"Processing {seq_name}...")
        
        # Check for expected files
        gt_file = seq_dir / "gt.txt"
        baseline_file = seq_dir / "baseline.txt"
        split_file = seq_dir / "after_split.txt"
        merge_file = seq_dir / "after_merge.txt"
        
        # Process GT file
        if gt_file.exists():
            dest_dir = output_dir / "gt" / seq_name
            dest_file = dest_dir / "gt.txt"
            
            if not dry_run:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(gt_file, dest_file)
            
            print(f"  ✓ gt.txt → gt/{seq_name}/gt.txt")
            stats['gt'] += 1
        else:
            msg = f"  ⚠️  Missing gt.txt"
            print(msg)
            stats['errors'].append(f"{seq_name}: {msg}")
        
        # Process baseline file
        if baseline_file.exists():
            dest_dir = output_dir / "baseline" / seq_name
            dest_file = dest_dir / f"{seq_name}.txt"
            
            if not dry_run:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(baseline_file, dest_file)
            
            print(f"  ✓ baseline.txt → baseline/{seq_name}/{seq_name}.txt")
            stats['baseline'] += 1
        else:
            msg = f"  ⚠️  Missing baseline.txt"
            print(msg)
            stats['errors'].append(f"{seq_name}: {msg}")
        
        # Process split file
        if split_file.exists():
            dest_dir = output_dir / "after_split" / seq_name
            dest_file = dest_dir / f"{seq_name}.txt"
            
            if not dry_run:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(split_file, dest_file)
            
            print(f"  ✓ after_split.txt → after_split/{seq_name}/{seq_name}.txt")
            stats['after_split'] += 1
        else:
            msg = f"  ⚠️  Missing after_split.txt"
            print(msg)
            stats['errors'].append(f"{seq_name}: {msg}")
        
        # Process merge file
        if merge_file.exists():
            dest_dir = output_dir / "after_merge" / seq_name
            dest_file = dest_dir / f"{seq_name}.txt"
            
            if not dry_run:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(merge_file, dest_file)
            
            print(f"  ✓ after_merge.txt → after_merge/{seq_name}/{seq_name}.txt")
            stats['after_merge'] += 1
        else:
            msg = f"  ⚠️  Missing after_merge.txt"
            print(msg)
            stats['errors'].append(f"{seq_name}: {msg}")
        
        print()
    
    # Print summary
    print(f"{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Files copied:")
    print(f"  GT:          {stats['gt']}")
    print(f"  Baseline:    {stats['baseline']}")
    print(f"  After split: {stats['after_split']}")
    print(f"  After merge: {stats['after_merge']}")
    
    if stats['errors']:
        print(f"\n⚠️  {len(stats['errors'])} warnings:")
        for error in stats['errors'][:10]:  # Show first 10
            print(f"  {error}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more")
    
    if dry_run:
        print(f"\n⚠️  DRY RUN: No files were actually modified")
        print(f"   Run without --dry-run to perform the reorganization")
    else:
        print(f"\n✓ Files written to: {output_dir}")
        print(f"\nNext steps:")
        print(f"  1. Verify structure: python mot_utils.py verify --mot_root {output_dir}")
        print(f"  2. Run sn-trackeval:")
        print(f"     python evaluate.py \\")
        print(f"       --gt_folder {output_dir}/gt \\")
        print(f"       --trackers_folder {output_dir} \\")
        print(f"       --trackers baseline after_split after_merge")
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Reorganize MOT files to sn-trackeval format"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory with per-sequence structure (e.g., evaluation/)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for reorganized files (e.g., mot_files/)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually copying files'
    )
    args = parser.parse_args()
    
    reorganize_mot_files(
        input_dir=args.input,
        output_dir=args.output,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()