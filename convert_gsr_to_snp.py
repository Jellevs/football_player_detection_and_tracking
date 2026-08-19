"""
convert_gsr_to_snp.py

Converts SoccerNet Game State Reconstruction (GSR-2024) annotations to
MOT-format ground truth files, filtering to players and goalkeepers only.

Output dataset: SoccerNet-Players (SNP)
- Only category_id 1 (player) and 2 (goalkeeper) are kept
- Referees, staff, ball and other categories are excluded
- Output format matches MOT17 / SoccerNet tracking-2023 conventions

Usage:
    python convert_gsr_to_snp.py \
        --gsr_root  C:/path/to/SoccerNetGS \
        --output_root C:/path/to/SoccerNet-Players \
        --split train          # or val, test

Output structure:
    SoccerNet-Players/
    ├── train/
    │   ├── SNPT-001/
    │   │   ├── gt/
    │   │   │   └── gt.txt
    │   │   ├── img1/          (if --copy_images)
    │   │   │   ├── 000001.jpg
    │   │   │   └── ...
    │   │   └── seqinfo.ini
    │   └── ...
    ├── val/
    └── test/
"""

import json
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


# GSR category_id mapping (from SoccerNet-GSR annotation spec)
PLAYER_CATEGORIES     = {1, 2}   # 1=player, 2=goalkeeper
FRAME_RATE            = 25       # SoccerNet sequences are 25fps
IMAGE_WIDTH           = 1920
IMAGE_HEIGHT          = 1080


def get_frame_number(filename):
    """
    Extract 1-based frame number from filename like '000001.jpg'.
    GSR uses 1-indexed filenames so no offset needed.
    """
    return int(Path(filename).stem)


def convert_sequence(gsr_seq_dir, output_seq_dir, snp_name=None, copy_images=False):
    """
    Convert a single GSR sequence to MOT format.

    Args:
        gsr_seq_dir:    Path to e.g. SoccerNetGS/train/SNGS-001
        output_seq_dir: Path to e.g. SoccerNet-Players/train/SNPT-001
        snp_name:       Output sequence name e.g. SNPT-001
        copy_images:    If True, copy img1 folder into output sequence dir
    """
    if snp_name is None:
        snp_name = gsr_seq_dir.name.replace("SNGS-", "SNPT-")
    json_path = gsr_seq_dir / "Labels-GameState.json"
    if not json_path.exists():
        print(f"SKIP {gsr_seq_dir.name}: no Labels-GameState.json found")
        return 0

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Build image_id -> frame_number lookup
    image_id_to_frame = {}
    for img in data['images']:
        frame_num = get_frame_number(img['file_name'])
        image_id_to_frame[img['image_id']] = frame_num

    # Count total frames for seqinfo
    total_frames = max(image_id_to_frame.values()) if image_id_to_frame else 0

    # Filter and collect annotations
    mot_rows = []
    skipped = 0

    for ann in data['annotations']:
        # Only keep players and goalkeepers
        if ann.get('category_id') not in PLAYER_CATEGORIES:
            skipped += 1
            continue

        track_id = ann.get('track_id', -1)
        if track_id < 0:
            skipped += 1
            continue

        frame_num = image_id_to_frame.get(ann['image_id'])
        if frame_num is None:
            skipped += 1
            continue

        b = ann.get('bbox_image', {})
        x = b.get('x', 0)
        y = b.get('y', 0)
        w = b.get('w', 0)
        h = b.get('h', 0)

        # Skip degenerate boxes
        if w <= 0 or h <= 0:
            skipped += 1
            continue

        # MOT format: frame, id, x, y, w, h, conf, -1, -1, -1
        mot_rows.append((frame_num, track_id, x, y, w, h))

    if not mot_rows:
        print(f"SKIP {gsr_seq_dir.name}: no valid player/goalkeeper annotations")
        return 0

    # Sort by frame then track_id (MOT convention)
    mot_rows.sort(key=lambda r: (r[0], r[1]))

    # Write gt.txt
    gt_dir = output_seq_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_path = gt_dir / "gt.txt"

    with open(gt_path, 'w') as f:
        for frame, tid, x, y, w, h in mot_rows:
            f.write(f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

    # Write seqinfo.ini (required by sn-trackeval and most MOT eval frameworks)
    img_dir = gsr_seq_dir / "img1"
    seq_length = total_frames

    seqinfo_path = output_seq_dir / "seqinfo.ini"
    with open(seqinfo_path, 'w') as f:
        f.write(f"[Sequence]\n")
        f.write(f"name={snp_name}\n")
        f.write(f"imDir=img1\n")
        f.write(f"frameRate={FRAME_RATE}\n")
        f.write(f"seqLength={seq_length}\n")
        f.write(f"imWidth={IMAGE_WIDTH}\n")
        f.write(f"imHeight={IMAGE_HEIGHT}\n")
        f.write(f"imExt=.jpg\n")

    # Copy images if requested
    if copy_images:
        src_img_dir = gsr_seq_dir / "img1"
        dst_img_dir = output_seq_dir / "img1"
        if src_img_dir.exists():
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            images = sorted(src_img_dir.glob("*.jpg"))
            for img in tqdm(images, desc=f"  Copying {snp_name} images", leave=False):
                shutil.copy2(img, dst_img_dir / img.name)
        else:
            print(f"WARNING: img1 not found at {src_img_dir}")

    return len(mot_rows)


def convert_split(gsr_root, output_root, split, copy_images=False):
    """
    Convert all sequences in a given split.

    Args:
        gsr_root:    Root of SoccerNetGS dataset (contains train/valid/test folders)
        output_root: Root of output SoccerNet-Players dataset
        split:       One of 'train', 'valid', 'test'
        copy_images: If True, copy img1 folders into output
    """
    # GSR uses 'valid', MOT convention uses 'val', we standardize to 'val'
    output_split_name = 'val' if split == 'valid' else split
    
    gsr_split_dir    = Path(gsr_root) / split
    output_split_dir = Path(output_root) / output_split_name

    if not gsr_split_dir.exists():
        raise FileNotFoundError(f"GSR split directory not found: {gsr_split_dir}")

    sequences = sorted([d for d in gsr_split_dir.iterdir() if d.is_dir()])

    if not sequences:
        print(f"No sequences found in {gsr_split_dir}")
        return

    print(f"\nConverting split '{split}' -> '{output_split_name}'")
    print(f"Source:      {gsr_split_dir}")
    print(f"Destination: {output_split_dir}")
    print(f"Sequences:   {len(sequences)}")

    total_annotations = 0
    converted = 0

    for seq_dir in sequences:
        snp_name = seq_dir.name.replace("SNGS-", "SNPT-")
        output_seq_dir = output_split_dir / snp_name
        n = convert_sequence(seq_dir, output_seq_dir, snp_name, copy_images=copy_images)
        if n > 0:
            print(f"OK  {seq_dir.name} -> {snp_name}: {n} annotations")
            total_annotations += n
            converted += 1
        
    print(f"\nDone: {converted}/{len(sequences)} sequences converted")
    print(f"Total player+goalkeeper annotations: {total_annotations}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SoccerNet-GSR annotations to SoccerNet-Players MOT format"
    )
    parser.add_argument(
        "--gsr_root",
        default=r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\data\SoccerNetGS",
        help="Path to SoccerNetGS root directory (containing train/valid/test folders)"
    )
    parser.add_argument(
        "--output_root",
        default=r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\soccernet-player-tracking",
        help="Path to output SoccerNet-Players root directory"
    )
    parser.add_argument(
        "--copy_images",
        action="store_true",
        default=True,
        help="Copy img1 image folders into the output dataset (requires significant disk space)"
    )
    parser.add_argument(
        "--split",
        choices=["train", "valid", "test", "all"],
        default="valid",
        help="Which split to convert (default: all)"
    )
    args = parser.parse_args()

    splits = ["train", "valid", "test"] if args.split == "all" else [args.split]

    print("=" * 60)
    print("SoccerNet-GSR -> SoccerNet-Players (SNP) Converter")
    print("=" * 60)
    print(f"GSR root:    {args.gsr_root}")
    print(f"Output root: {args.output_root}")
    print(f"Splits:      {splits}")

    for split in splits:
        convert_split(args.gsr_root, args.output_root, split, copy_images=args.copy_images)

    print("\n" + "=" * 60)
    print("Conversion complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()