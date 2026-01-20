import sys
from pathlib import Path

CENTROIDS_REID_PATH = Path(__file__).parent / 'reid' / 'centroids_reid'
sys.path.insert(0, str(CENTROIDS_REID_PATH))

import types

# Fixing imports, imcompatibility issues etc
try:
    from pytorch_lightning.callbacks import Callback
    base_module = types.ModuleType('base')
    base_module.Callback = Callback
    sys.modules['pytorch_lightning.callbacks.base'] = base_module
except ImportError:
    pass

import torch
import numpy as np
from PIL import Image
import warnings

import pytorch_lightning as pl

# Store original __setattr__
_original_setattr = pl.LightningModule.__setattr__

def _patched_setattr(self, name, value):
    """ Handle hparams assignment for old code """
    if name == 'hparams':
        try:
            if hasattr(self, 'hparams'):
                if hasattr(value, 'items'):
                    self.hparams.update(value)
                elif hasattr(value, '__dict__'):
                    self.hparams.update(vars(value))
                else:
                    self.hparams.update({'value': value})
                return
        except AttributeError:
            pass
    # Normal attribute setting
    _original_setattr(self, name, value)

# Apply the patch to the errors
pl.LightningModule.__setattr__ = _patched_setattr

from train_ctl_model import CTLModel
from datasets.transforms import ReidTransforms
from config import cfg


class CentroidReIDFilter:
    """ ReID-based outlier filter using Centroid-ReID """
    
    def __init__(self, checkpoint_path, threshold_std=3.5, rounds=5, min_samples=3, device='cuda'):
        self.threshold_std = threshold_std
        self.rounds = rounds
        self.min_samples = min_samples
        self.device = device
        
        self.model = self._load_model(checkpoint_path)
        self.transforms = self._get_transforms()
            
    
    def _load_model(self, checkpoint_path):
        """ Load Centroid-ReID model with all compatibility fixes """
        
        config_file = CENTROIDS_REID_PATH / 'configs' / '256_resnet50.yml'
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        # Load config
        cfg.merge_from_file(str(config_file))
        opts = [
            "MODEL.PRETRAIN_PATH", str(checkpoint_path),
            "MODEL.PRETRAINED", True,
            "TEST.ONLY_TEST", True,
            "MODEL.RESUME_TRAINING", False
        ]
        cfg.merge_from_list(opts)
        
        # PyTorch giving errors: Add safe globals
        try:
            safe_globals_list = []
            
            try:
                from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
                safe_globals_list.append(ModelCheckpoint)
            except ImportError:
                pass
            
            try:
                from yacs.config import CfgNode
                safe_globals_list.append(CfgNode)
            except ImportError:
                pass
            
            from collections import OrderedDict
            safe_globals_list.append(OrderedDict)
            
            if safe_globals_list:
                torch.serialization.add_safe_globals(safe_globals_list)
        except (ImportError, AttributeError):
            pass
        
        use_cuda = self.device == 'cuda' and torch.cuda.is_available()
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            try:
                # Try normal loading first
                model = CTLModel.load_from_checkpoint(
                    str(checkpoint_path), 
                    cfg=cfg
                )
            except AttributeError as e:
                if "'hparams'" in str(e) or "can't set attribute" in str(e):
                    # hparams issue - load manually
                    print("Using manual checkpoint loading (hparams compatibility)")
                    model = self._load_checkpoint_manually(checkpoint_path, cfg)
                else:
                    raise
        
        if use_cuda:
            model.to('cuda')
        
        model.eval()
        return model
    
    
    def _load_checkpoint_manually(self, checkpoint_path, cfg):
        """ Manual checkpoint loading to bypass hparams issues """
        
        # Load checkpoint dict
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model = CTLModel(cfg)
        
        # Load state dict
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        return model
    
    
    def _get_transforms(self):
        """ Get image transforms """
        transforms_base = ReidTransforms(cfg)
        return transforms_base.build_transforms(is_train=False)
    
    
    def extract_embeddings(self, crops):
        """ Extract Centroid-ReID embeddings. """
        embeddings = []
        use_cuda = self.device == 'cuda' and torch.cuda.is_available()
        
        for crop in crops:
            # Convert to PIL
            if crop.dtype == np.uint8:
                img = Image.fromarray(crop)
            else:
                img = Image.fromarray((crop * 255).astype(np.uint8))
            
            img_tensor = torch.stack([self.transforms(img)])
            
            # Extract features
            with torch.no_grad():
                if use_cuda:
                    img_tensor = img_tensor.cuda()
                
                _, global_feat = self.model.backbone(img_tensor)
                global_feat = self.model.bn(global_feat)
            
            embeddings.append(global_feat.cpu().numpy().flatten())
        
        return np.array(embeddings)
    
    
    def filter(self, crops_list, indices, original_embeddings=None):
        """ Filter crops by removing ReID outliers """
        full_crops = crops_list[0]
        
        if len(full_crops) < self.min_samples:
            return crops_list, indices
        
        # Extract embeddings
        embeddings = self.extract_embeddings(full_crops)
        
        # Apply Gaussian filtering
        kept_mask = self.iterative_outlier_removal(embeddings)
        
        # Filter all crop lists
        filtered_crops_list = [
            [crop for crop, keep in zip(crops, kept_mask) if keep]
            for crops in crops_list
        ]
        filtered_indices = [idx for idx, keep in zip(indices, kept_mask) if keep]
        
        return filtered_crops_list, filtered_indices
    
    
    def iterative_outlier_removal(self, embeddings):
        """ Iterative Gaussian outlier removal """
        current_mask = np.ones(len(embeddings), dtype=bool)
        
        for round_idx in range(self.rounds):
            current_embeddings = embeddings[current_mask]
            mu = np.mean(current_embeddings, axis=0, keepdims=True)
            euclidean_distances = np.linalg.norm(embeddings - mu, axis=1)
            
            mean_distance = np.mean(euclidean_distances)
            std_distance = np.std(euclidean_distances)
            threshold = self.threshold_std * std_distance
            
            new_mask = (euclidean_distances - mean_distance) <= threshold
            current_mask = current_mask & new_mask
            
            if np.sum(new_mask) == np.sum(current_mask):
                break
        
        return current_mask

