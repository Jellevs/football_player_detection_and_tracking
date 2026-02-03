# training_data/pca_reducer.py

import numpy as np
from sklearn.decomposition import PCA
import joblib
from pathlib import Path


class PCAReducer:
    """
    Reduce high-dimensional embeddings using PCA while preserving most variance.
    """
    
    def __init__(self, siglip_components=16, reid_components=8):
        """
        Args:
            siglip_components: Number of PCA components for SigLIP (768 → 16)
            reid_components: Number of PCA components for ReID (512 → 8)
        """
        self.siglip_components = siglip_components
        self.reid_components = reid_components
        
        self.pca_siglip = None
        self.pca_reid = None
    
    
    def fit(self, aggregated_fragments):
        """
        Fit PCA on all embeddings from aggregated fragments.
        
        Args:
            aggregated_fragments: List of fragment dicts from FragmentAggregator
        """
        print("\n" + "="*60)
        print("Fitting PCA on embeddings")
        print("="*60)
        
        # Collect all embeddings
        siglip_embeddings = []
        reid_embeddings = []
        
        for frag in aggregated_fragments:
            siglip_embeddings.append(frag['aggregated']['siglip_mean'])
            reid_embeddings.append(frag['aggregated']['reid_mean'])
        
        siglip_embeddings = np.array(siglip_embeddings)
        reid_embeddings = np.array(reid_embeddings)
        
        print(f"Collected {len(siglip_embeddings)} SigLIP embeddings ({siglip_embeddings.shape[1]}-dim)")
        print(f"Collected {len(reid_embeddings)} ReID embeddings ({reid_embeddings.shape[1]}-dim)")
        
        # Fit PCA
        self.pca_siglip = PCA(n_components=self.siglip_components, random_state=42)
        self.pca_reid = PCA(n_components=self.reid_components, random_state=42)
        
        self.pca_siglip.fit(siglip_embeddings)
        self.pca_reid.fit(reid_embeddings)
        
        # Report variance retained
        siglip_variance = self.pca_siglip.explained_variance_ratio_.sum()
        reid_variance = self.pca_reid.explained_variance_ratio_.sum()
        
        print(f"\nSigLIP PCA: 768 → {self.siglip_components} dims")
        print(f"  Variance retained: {siglip_variance:.2%}")
        
        print(f"\nReID PCA: 512 → {self.reid_components} dims")
        print(f"  Variance retained: {reid_variance:.2%}")
        
        return self
    
    
    def transform(self, aggregated_fragments):
        """
        Transform embeddings in all fragments using fitted PCA.
        
        Args:
            aggregated_fragments: List of fragment dicts
            
        Returns:
            List of fragments with reduced embeddings
        """
        if self.pca_siglip is None or self.pca_reid is None:
            raise ValueError("PCA not fitted! Call fit() first.")
        
        print("\nTransforming embeddings with PCA...")
        
        for frag in aggregated_fragments:
            # Transform SigLIP
            siglip_original = frag['aggregated']['siglip_mean']
            siglip_reduced = self.pca_siglip.transform([siglip_original])[0]
            frag['aggregated']['siglip_mean'] = siglip_reduced
            
            # Transform ReID
            reid_original = frag['aggregated']['reid_mean']
            reid_reduced = self.pca_reid.transform([reid_original])[0]
            frag['aggregated']['reid_mean'] = reid_reduced
        
        print(f"Transformed {len(aggregated_fragments)} fragments")
        
        return aggregated_fragments
    
    
    def fit_transform(self, aggregated_fragments):
        """Fit PCA and transform in one step."""
        self.fit(aggregated_fragments)
        return self.transform(aggregated_fragments)
    
    
    def save(self, output_dir):
        """
        Save fitted PCA models to disk for inference.
        
        Args:
            output_dir: Directory to save PCA models
        """
        if self.pca_siglip is None or self.pca_reid is None:
            raise ValueError("PCA not fitted! Call fit() first.")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        siglip_path = output_dir / 'pca_siglip.pkl'
        reid_path = output_dir / 'pca_reid.pkl'
        
        joblib.dump(self.pca_siglip, siglip_path)
        joblib.dump(self.pca_reid, reid_path)
        
        print(f"\nPCA models saved:")
        print(f"  SigLIP: {siglip_path}")
        print(f"  ReID: {reid_path}")
        
        return siglip_path, reid_path
    
    
    @classmethod
    def load(cls, output_dir):
        """
        Load fitted PCA models from disk.
        
        Args:
            output_dir: Directory containing PCA models
            
        Returns:
            PCAReducer instance with loaded models
        """
        output_dir = Path(output_dir)
        
        siglip_path = output_dir / 'pca_siglip.pkl'
        reid_path = output_dir / 'pca_reid.pkl'
        
        if not siglip_path.exists() or not reid_path.exists():
            raise FileNotFoundError(f"PCA models not found in {output_dir}")
        
        reducer = cls()
        reducer.pca_siglip = joblib.load(siglip_path)
        reducer.pca_reid = joblib.load(reid_path)
        
        print(f"Loaded PCA models from {output_dir}")
        
        return reducer