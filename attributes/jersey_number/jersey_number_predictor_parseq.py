from pathlib import Path
import torch
import numpy as np
from PIL import Image
import cv2

from .parseq.strhub.data.module import SceneTextDataModule
from .legibility_classifier import LegibilityPredictor
from .pose_cropper import PoseCropper
from .centroid_reid_filter import CentroidReIDFilter


class JerseyNumberPredictorParseq:
    """ Jersey number detection pipeline Based on mkoshkina's jersey-number-pipeline (https://github.com/mkoshkina/jersey-number-pipeline) """
    def __init__(self, paths, jersey_cfg, device='cpu'):
        self.jersey_cfg = jersey_cfg
        self.device = device
        self.debug_dir = Path(jersey_cfg.debug_dir) if jersey_cfg.debug_dir else None
        self.paths = paths

        self.parseq_model = self.load_parseq_model(paths.parseq_model_path)
        self.img_transform = SceneTextDataModule.get_transform(self.parseq_model.hparams.img_size)

        self.legibility_predictor = self.load_legibility_model(paths.legibility_model_path) if jersey_cfg.use_legibility else None
        self.pose_cropper = self.load_pose_cropper_model(paths.vitpose_model_path) if jersey_cfg.use_pose_cropper else None
        self.reid_filter = self.load_reid_filter_model(paths.centroid_reid_path) if jersey_cfg.use_reid_filter else None


    def predict(self, images, tracklet):
        """ Predict jersey numbers for a tracklet """
        num_frames = len(tracklet.frames)
        
        # Extract all crops
        full_crops, torso_crops, indices = self.extract_crops(images, tracklet)
        
        # Keep unfiltered torso crops for team classification
        torso_crops_for_teams = torso_crops.copy()
        indices_for_teams = indices.copy()

        if not full_crops:
            return [], [], np.full(num_frames, np.nan), np.zeros(num_frames)
        
        # Stage 1: ReID outlier filtering
        if self.reid_filter and tracklet.embeddings:
            embeddings = [tracklet.embeddings[i] for i in indices]
            [full_crops, torso_crops], indices = self.reid_filter.filter(
                [full_crops, torso_crops], indices, embeddings
            )
                
        # Stage 2: Legibility filtering
        if self.legibility_predictor and full_crops:
            [full_crops, torso_crops], indices = self.legibility_predictor.filter(
                [full_crops, torso_crops], indices
            )

        # Stage 3: Predict jersey numbers on filtered torso crops
        if torso_crops:
            jersey_predictions, confs_mean, entropies = self.predict_jersey(torso_crops)
        else:
            jersey_predictions, confs_mean, entropies = [], [], []
        
        jerseys, confs_mean, entropies = self.map_to_frames(jersey_predictions, confs_mean, entropies, indices, num_frames)


        # self.save_torso_crops(torso_crops, indices_for_teams, tracklet)

        return torso_crops_for_teams, indices_for_teams, jerseys, confs_mean, entropies


    def extract_crops(self, images, tracklet):
        """ Extract full and torso crops for all frames in tracklet """
        full_crops = []
        torso_crops = []
        indices = []

        for i, (frame_idx, bbox) in enumerate(zip(tracklet.frames, tracklet.bboxes)):
            image = cv2.imread(str(images[frame_idx]))
            if image is None:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            full_crop = image[y1:y2, x1:x2]
            if full_crop.size == 0:
                continue
                
            full_crop_rgb = cv2.cvtColor(full_crop, cv2.COLOR_BGR2RGB)

            # Get torso crop using pose estimation or fallback to heuristic
            if self.pose_cropper:
                torso_crop = self.pose_cropper.get_torso_crop(image, bbox)
                if torso_crop is None:
                    torso_crop = self.simple_torso_crop(full_crop_rgb)
            else:
                torso_crop = self.simple_torso_crop(full_crop_rgb)

            if torso_crop is None or torso_crop.size == 0:
                continue

            full_crops.append(full_crop_rgb)
            torso_crops.append(torso_crop)
            indices.append(i)

        return full_crops, torso_crops, indices
        

    def load_parseq_model(self, parseq_model_path):
        """ Load PARSeq OCR model """
        model = torch.load(parseq_model_path, map_location='cpu', weights_only=False)

        if 'state_dict' in model:
            state_dict = model['state_dict']
        elif 'model' in model:
            state_dict = model['model']
        else:
            state_dict = model

        new_state_dict = {f'model.{k}': v for k, v in state_dict.items()}

        model = torch.hub.load('baudm/parseq', 'parseq', pretrained=False)
        model.load_state_dict(new_state_dict, strict=True)
        model = model.eval().to(self.device)
        
        print("PARSeq model loaded")
        return model
    

    def load_legibility_model(self, legibility_model_path):
        """ Load legibility classifier """
        return LegibilityPredictor(
            model_path=str(legibility_model_path),
            device=self.device,
            threshold=self.jersey_cfg.legibility_threshold,
            arch=self.jersey_cfg.legibility_arch
        )
    
    
    def load_pose_cropper_model(self, vitpose_model_path):
        """ Load ViTPose model for torso cropping """
        # Use local path if available, otherwise HuggingFace
        return PoseCropper(
            device=self.device,
            model_path=str(vitpose_model_path)
        )

    
    def load_reid_filter_model(self, centroid_reid_path):
        """ Load Centroid-ReID model for outlier filtering """
        return CentroidReIDFilter(
            checkpoint_path=centroid_reid_path,
            threshold_std=self.jersey_cfg.reid_threshold_std,
            rounds=5,
            min_samples=3
        )
    

    def map_to_frames(self, jersey_predictions, confs_mean, entropies, indices, num_frames):
        """ Map predictions back to all tracklet frames """
        result_map = dict(zip(indices, zip(jersey_predictions, confs_mean, entropies)))

        jerseys = np.array([result_map.get(i, (np.nan, 0.0, 1.0))[0] for i in range(num_frames)])
        confs_mean = np.array([result_map.get(i, (np.nan, 0.0, 1.0))[1] for i in range(num_frames)])
        entropies = np.array([result_map.get(i, (np.nan, 0.0, 1.0))[2] for i in range(num_frames)])

        return jerseys, confs_mean, entropies


    def predict_jersey(self, crops, batch_size=32):
        """ Run PARSeq on crops to predict jersey numbers """
        jerseys = []
        confs_mean = []
        entropies = []
        
        for i in range(0, len(crops), batch_size):
            batch = crops[i:i + batch_size]
            tensors = [self.img_transform(Image.fromarray(c)) for c in batch]
            batch_tensor = torch.stack(tensors).to(self.device)
            
            with torch.no_grad():
                logits = self.parseq_model(batch_tensor)
                probs = logits.softmax(-1)
            
            labels, raw_confs = self.parseq_model.tokenizer.decode(probs)
            
            for label, conf_tensor in zip(labels, raw_confs):
                label = label.strip()
                conf_values = conf_tensor.cpu().numpy()

                if len(conf_values) > 1:
                    conf_values = conf_values[:-1]
                
                if label.isdigit() and 0 <= int(label) <= 99:
                    jerseys.append(int(label))

                    confs_mean.append(float(conf_values.mean()))

                    probs_clipped = np.clip(conf_values.mean())
                    entropy = float(-np.sum(probs_clipped * np.log(probs_clipped)))
                    entropies.append(entropy)
                else:
                    jerseys.append(np.nan)
                    confs_mean.append(0.0)
                    entropies.append(1.0)
        
        return jerseys, confs_mean, entropies
    

    def simple_torso_crop(self, crop):
        """ Simple heuristic torso crop """
        h = crop.shape[0]
        return crop[int(h * 0.2):int(h * 0.8), :, :]
    

    def save_torso_crops(self, torso_crops, indices, tracklet, max_crops=10):
        """ Save torso crops to debug directory for testing """
        if not self.debug_dir:
            return
        
        debug_dir = Path(self.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample crops if too many
        if len(torso_crops) > max_crops:
            step = len(torso_crops) // max_crops
            crops_to_save = torso_crops[::step][:max_crops]
            indices_to_save = indices[::step][:max_crops]
        else:
            crops_to_save = torso_crops
            indices_to_save = indices
        
        # Save each crop
        for i, (crop, tracklet_idx) in enumerate(zip(crops_to_save, indices_to_save)):
            frame_idx = tracklet.frames[tracklet_idx]
            
            # Get GT jersey if available
            gt_jerseys = tracklet.gt_attributes.get('jerseys', [])
            gt_jersey = gt_jerseys[tracklet_idx] if tracklet_idx < len(gt_jerseys) else "X"
            
            filename = f"track{tracklet.track_id:04d}_frame{frame_idx:05d}_gt{gt_jersey}.jpg"
            save_path = debug_dir / filename
            
            # Convert RGB to BGR for cv2
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_path), crop_bgr)
