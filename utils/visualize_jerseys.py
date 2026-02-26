"""
Visualize jersey number predictions for debugging.

Generates per-tracklet image grids showing:
- Torso crops with per-frame greedy predictions + confidence
- Bayesian consolidated prediction for the whole tracklet
- Color-coded borders (green=confident, red=low confidence, gray=no prediction)

Usage:
    # From main.py, after predict_attributes:
    from utils.visualize_jerseys import visualize_jersey_predictions
    visualize_jersey_predictions(images, tracklets, paths, jersey_cfg, device, max_tracklets=30)

    # Or standalone:
    python -m utils.visualize_jerseys --sequence SNMOT-116 --max_tracklets 30
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def visualize_jersey_predictions(images, tracklets, output_dir, max_tracklets=30, crops_per_tracklet=16):
    """
    Generate jersey prediction debug images.
    
    Args:
        images: list of image paths
        tracklets: dict of {track_id: Tracklet} with pred_attributes already filled
        output_dir: Path to save debug images
        max_tracklets: max number of tracklets to visualize
        crops_per_tracklet: max crops to show per tracklet grid
    """
    from attributes.jersey_number.jersey_number_predictor_parseq import JerseyNumberPredictorParseq

    output_dir = Path(output_dir) / "jersey_debug"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort tracklets by length (longest first — most interesting)
    sorted_tracklets = sorted(tracklets.items(), key=lambda x: len(x[1].frames), reverse=True)
    
    if max_tracklets:
        sorted_tracklets = sorted_tracklets[:max_tracklets]

    summary_rows = []

    for track_id, tracklet in tqdm(sorted_tracklets, desc="Visualizing jersey predictions"):
        jerseys = tracklet.pred_attributes.get('jerseys', [])
        confs = tracklet.pred_attributes.get('jersey_confs_mean', [])
        entropies = tracklet.pred_attributes.get('jersey_entropies', [])
        raw_probs = tracklet.pred_attributes.get('jersey_raw_probs', [])

        if not jerseys:
            continue

        # Bayesian consolidated prediction
        if raw_probs:
            consolidated_jersey, consolidated_conf = JerseyNumberPredictorParseq.consolidate_tracklet_bayesian(
                raw_probs, use_bias=True
            )
            # Also get heuristic for comparison
            valid_jerseys = [j for j in jerseys if not (isinstance(j, float) and np.isnan(j))]
            valid_confs = [c for j, c in zip(jerseys, confs) if not (isinstance(j, float) and np.isnan(j))]
            heuristic_jersey, heuristic_conf = JerseyNumberPredictorParseq.consolidate_tracklet_heuristic(
                valid_jerseys, valid_confs, use_bias=True
            )
        else:
            consolidated_jersey, consolidated_conf = -1, 0.0
            heuristic_jersey, heuristic_conf = -1, 0.0

        # Collect frames that have predictions
        crop_data = []
        for i, frame_idx in enumerate(tracklet.frames):
            if i >= len(jerseys):
                break
            jersey = jerseys[i]
            conf = confs[i] if i < len(confs) else 0.0
            entropy = entropies[i] if i < len(entropies) else 1.0
            
            crop_data.append({
                'frame_idx': frame_idx,
                'tracklet_idx': i,
                'jersey': jersey,
                'conf': conf,
                'entropy': entropy,
            })

        # Sample evenly if too many
        if len(crop_data) > crops_per_tracklet:
            step = len(crop_data) / crops_per_tracklet
            crop_data = [crop_data[int(i * step)] for i in range(crops_per_tracklet)]

        # Extract actual crops from images
        crop_images = []
        for cd in crop_data:
            frame_idx = cd['frame_idx']
            tidx = cd['tracklet_idx']
            
            img = cv2.imread(str(images[frame_idx]))
            if img is None:
                continue
            
            bbox = tracklet.bboxes[tidx]
            x1, y1, x2, y2 = map(int, bbox)
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            crop_images.append((crop, cd))

        if not crop_images:
            continue

        # Build grid image
        grid = _build_tracklet_grid(
            track_id, crop_images,
            consolidated_jersey, consolidated_conf,
            heuristic_jersey, heuristic_conf,
            len(tracklet.frames)
        )

        # Save
        save_path = output_dir / f"track_{track_id:04d}.jpg"
        cv2.imwrite(str(save_path), grid)

        # Summary row
        pred_counts = {}
        for j in jerseys:
            if isinstance(j, float) and np.isnan(j):
                continue
            j = int(j)
            pred_counts[j] = pred_counts.get(j, 0) + 1
        
        summary_rows.append({
            'track_id': track_id,
            'n_frames': len(tracklet.frames),
            'n_predicted': sum(1 for j in jerseys if not (isinstance(j, float) and np.isnan(j))),
            'bayesian': consolidated_jersey,
            'heuristic': heuristic_jersey,
            'pred_counts': pred_counts,
        })

    # Save summary text
    _save_summary(output_dir / "summary.txt", summary_rows)
    print(f"Jersey debug images saved to: {output_dir}")


def _build_tracklet_grid(track_id, crop_images, bayes_jersey, bayes_conf,
                          heur_jersey, heur_conf, total_frames):
    """Build a grid image for one tracklet."""

    CELL_W, CELL_H = 80, 120  # crop display size
    HEADER_H = 60
    COLS = min(8, len(crop_images))
    ROWS = (len(crop_images) + COLS - 1) // COLS

    grid_w = COLS * CELL_W + 10
    grid_h = HEADER_H + ROWS * CELL_H + 10

    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 40  # dark gray background

    # Header
    bayes_str = str(bayes_jersey) if bayes_jersey != -1 else "?"
    heur_str = str(heur_jersey) if heur_jersey != -1 else "?"
    
    cv2.putText(grid, f"Track {track_id}  ({total_frames} frames)",
                (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(grid, f"Bayesian: #{bayes_str} ({bayes_conf:.1f})",
                (5, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
    cv2.putText(grid, f"Heuristic: #{heur_str} ({heur_conf:.2f})",
                (5 + grid_w // 2, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)

    # Divider line
    cv2.line(grid, (0, HEADER_H - 5), (grid_w, HEADER_H - 5), (100, 100, 100), 1)

    # Crops
    for idx, (crop, cd) in enumerate(crop_images):
        row = idx // COLS
        col = idx % COLS
        
        x_off = col * CELL_W + 5
        y_off = HEADER_H + row * CELL_H

        # Resize crop to fit cell (preserve aspect ratio)
        ch, cw = crop.shape[:2]
        scale = min((CELL_W - 4) / cw, (CELL_H - 25) / ch)
        new_w = int(cw * scale)
        new_h = int(ch * scale)
        resized = cv2.resize(crop, (new_w, new_h))

        # Border color based on confidence
        jersey = cd['jersey']
        conf = cd['conf']
        is_valid = not (isinstance(jersey, float) and np.isnan(jersey))
        
        if is_valid and conf >= 0.5:
            border_color = (0, 200, 0)      # green — confident
        elif is_valid:
            border_color = (0, 180, 255)     # orange — low confidence
        else:
            border_color = (100, 100, 100)   # gray — no prediction

        # Draw border
        cv2.rectangle(grid, (x_off - 1, y_off - 1),
                      (x_off + new_w + 1, y_off + new_h + 1), border_color, 1)

        # Place crop
        grid[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        # Label below crop
        if is_valid:
            label = f"#{int(jersey)} c:{conf:.2f}"
        else:
            label = "X"
        
        cv2.putText(grid, label, (x_off, y_off + new_h + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
        # Frame number
        cv2.putText(grid, f"f{cd['frame_idx']}", (x_off, y_off + new_h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (150, 150, 150), 1)

    return grid


def _save_summary(path, rows):
    """Save a text summary of all tracklet predictions."""
    with open(path, 'w') as f:
        f.write(f"{'Track':>6} {'Frames':>7} {'Pred':>5} {'Bayesian':>9} {'Heuristic':>10}  Per-frame counts\n")
        f.write("-" * 75 + "\n")
        for r in sorted(rows, key=lambda x: x['track_id']):
            counts_str = ", ".join(f"#{k}:{v}" for k, v in sorted(r['pred_counts'].items(), key=lambda x: -x[1]))
            f.write(f"{r['track_id']:>6} {r['n_frames']:>7} {r['n_predicted']:>5} "
                    f"{'#' + str(r['bayesian']):>9} {'#' + str(r['heuristic']):>10}  {counts_str}\n")
    print(f"Summary saved to: {path}")


# ============================================================================
# Standalone usage
# ============================================================================
if __name__ == "__main__":
    import argparse
    import pickle
    import sys

    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    import settings
    from utils.build import build_paths, build_configs
    from utils.data_utils import load_images

    parser = argparse.ArgumentParser(description="Visualize jersey number predictions")
    parser.add_argument("--sequence", type=str, required=True, help="Sequence name (e.g. SNMOT-116)")
    parser.add_argument("--max_tracklets", type=int, default=30, help="Max tracklets to visualize")
    parser.add_argument("--crops_per_tracklet", type=int, default=16, help="Max crops per tracklet")
    args = parser.parse_args()

    paths = build_paths(args.sequence)
    images = load_images(img_dir=paths.img_path)

    # Load cached attributes
    cache_path = paths.set_cache_path("attributes", args.sequence)
    if not cache_path.exists():
        print(f"Cache not found: {cache_path}")
        print("Run main.py first to generate attributes, then re-run this script.")
        sys.exit(1)

    with open(cache_path, 'rb') as f:
        tracklets = pickle.load(f)

    print(f"Loaded {len(tracklets)} tracklets for {args.sequence}")

    visualize_jersey_predictions(
        images=images,
        tracklets=tracklets,
        output_dir=paths.output_path / args.sequence,
        max_tracklets=args.max_tracklets,
        crops_per_tracklet=args.crops_per_tracklet,
    )