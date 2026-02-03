# training_data/data_format.py

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict


class DataSaver:
    """Save training data to CSV format"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    
    def save_pairs_to_csv(self, pairs: List[Dict], filename: str = "training_pairs.csv"):
        """
        Save pairwise data to CSV format.
        
        Each row contains:
        - Fragment A features (flattened)
        - Fragment B features (flattened)
        - Pairwise features
        - Label
        """
        
        rows = []
        
        for pair in pairs:
            row = {}
            
            # Add metadata
            row['fragment_A_id'] = pair['fragment_A_id']
            row['fragment_B_id'] = pair['fragment_B_id']
            row['label'] = pair['label']
            
            # Flatten fragment A features
            for key, value in pair['features']['fragment_A'].items():
                if isinstance(value, np.ndarray):
                    # For arrays (like reid_mean), create separate columns
                    for i, v in enumerate(value):
                        row[f'A_{key}_{i}'] = v
                else:
                    row[f'A_{key}'] = value
            
            # Flatten fragment B features
            for key, value in pair['features']['fragment_B'].items():
                if isinstance(value, np.ndarray):
                    for i, v in enumerate(value):
                        row[f'B_{key}_{i}'] = v
                else:
                    row[f'B_{key}'] = value
            
            # Add pairwise features
            for key, value in pair['features']['pairwise'].items():
                row[f'pairwise_{key}'] = value
            
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Save to CSV
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        
        print(f"Saved {len(df)} pairs to {output_path}")
        print(f"Columns: {len(df.columns)}")
        print(f"Positive samples: {df['label'].sum()}")
        print(f"Negative samples: {len(df) - df['label'].sum()}")
        
        return output_path