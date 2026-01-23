
import numpy as np
import cv2
from .torchreid.utils import FeatureExtractor



class SimpleReIDExtractor:
    """
    Wrapper around Deep-EIoU's built-in FeatureExtractor
    Makes it easy to extract features from detections
    """
    
    def __init__(self, model_path, device='cuda', model_name='osnext_x1_0'):
        """
        Initialize using the built-in FeatureExtractor
        
        Args:
            model_path: Path to model.osnet.pth.tar-10
            device: 'cuda' or 'cpu'
        """
        print("🔧 Initializing OSNet using built-in FeatureExtractor...")
        
        # Use the built-in FeatureExtractor - it does everything!
        self.extractor = FeatureExtractor(
            model_name=model_name,
            model_path=model_path,
            image_size=(256, 128),
            pixel_mean=[0.485, 0.456, 0.406],
            pixel_std=[0.229, 0.224, 0.225],
            pixel_norm=True,
            device=device,
            verbose=True  # Shows model details
        )
        
        print("✅ OSNet ready!")
    
    def extract_features_from_detections(self, image, detections):
        """
        Extract ReID features from detections
        
        Args:
            image: Full frame (H, W, 3) in BGR format (OpenCV)
            detections: numpy array (N, 5) with [x1, y1, x2, y2, score]
        
        Returns:
            numpy array (N, 512) - L2 normalized embeddings
        """
        if len(detections) == 0:
            return np.array([]).reshape(0, 512)
        
        # Crop all players from image
        crops = []
        h, w = image.shape[:2]
        
        for det in detections:
            x1, y1, x2, y2 = det[:4].astype(int)
            
            # Clip to boundaries
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                # Add black crop as placeholder
                crops.append(np.zeros((64, 32, 3), dtype=np.uint8))
                continue
            
            # Extract crop
            crop = image[y1:y2, x1:x2]
            
            # Convert BGR (OpenCV) to RGB (FeatureExtractor expects RGB)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(crop_rgb)
        
        # Extract features using built-in extractor
        # It handles all preprocessing automatically!
        features = self.extractor(crops)  # That's it!
        
        # Convert to numpy and normalize
        features = features.cpu().numpy()
        
        # L2 normalize (unit vectors for cosine similarity)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / (norms + 1e-12)
        
        return features
    



def load_reid_extractor(model_path, device='cuda', model_name='osnet_x1_0'):
    """
    Load the ReID extractor (uses built-in FeatureExtractor)
    
    Args:
        model_path: Path to model.osnet.pth.tar-10
        device: 'cuda' or 'cpu'
    
    Returns:
        SimpleReIDExtractor instance
    
    Example:
        >>> extractor = load_reid_extractor('model.osnet.pth.tar-10')
        >>> features = extractor.extract_features_from_detections(image, detections)
    """
    return SimpleReIDExtractor(model_path, device, model_name)
