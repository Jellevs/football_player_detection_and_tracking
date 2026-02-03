import json
from pathlib import Path

def analyze_bbox_sizes(sequence_path):
    with open(sequence_path / "Labels-GameState.json", 'r') as f:
        data = json.load(f)
    
    widths = []
    heights = []
    
    for ann in data['annotations']:
        if ann['category_id'] == 1:  # Players only
            bbox = ann['bbox_image']
            widths.append(bbox['w'])
            heights.append(bbox['h'])
    
    import numpy as np
    print(f"Width:  mean={np.mean(widths):.1f}, min={np.min(widths)}, max={np.max(widths)}")
    print(f"Height: mean={np.mean(heights):.1f}, min={np.min(heights)}, max={np.max(heights)}")

# Compare
print("SNGS-200:")
analyze_bbox_sizes(Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\data\SoccerNetGS\test\SNGS-200"))

print("\nSNGS-123:")
analyze_bbox_sizes(Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\data\SoccerNetGS\test\SNGS-123"))