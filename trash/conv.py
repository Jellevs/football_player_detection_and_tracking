import os
import shutil

# Update these paths to your actual local paths
src_root = r'C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\evaluation'
dst_root = r'C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\evaluation'

# The different versions you want to evaluate as separate "trackers"
versions = {
    'baseline.txt': 'baseline',
    'after_merge.txt': 'after_merge'
}

for filename, tracker_name in versions.items():
    # Create the directory: eval_ready/trackers/tracker_name/data
    dest_dir = os.path.join(dst_root, tracker_name, 'data')
    os.makedirs(dest_dir, exist_ok=True)
    
    print(f"Organizing {tracker_name}...")
    
    # Iterate through each sequence folder (SNGS-116, SNGS-117, etc.)
    for seq in os.listdir(src_root):
        seq_path = os.path.join(src_root, seq)
        
        if os.path.isdir(seq_path):
            src_file = os.path.join(seq_path, filename)
            
            if os.path.exists(src_file):
                # Standard MOT format: data folder contains {sequence_name}.txt
                dest_file = os.path.join(dest_dir, f"{seq}.txt")
                shutil.copy2(src_file, dest_file)

print("Done! Your trackers are ready in the 'eval_ready' folder.")