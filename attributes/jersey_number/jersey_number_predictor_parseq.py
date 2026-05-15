from pathlib import Path
import torch
import numpy as np
from PIL import Image
import cv2
from scipy.special import softmax

from .parseq.strhub.data.module import SceneTextDataModule
from .legibility_classifier import LegibilityPredictor
from .pose_cropper import PoseCropper
from .centroid_reid_filter import CentroidReIDFilter


# ============================================================================
# Constants from Koshkina helpers.py — exact values from the paper's code
# ============================================================================
TOKEN_LIST = 'E0123456789'   # E = end-of-string, then digits 0-9 (11 tokens)
TS = 2.367                    # Temperature scaling factor from the paper
BIAS_FOR_DIGITS = [0.06, 0.094, 0.094, 0.094, 0.094, 0.094, 0.094, 0.094, 0.094, 0.094, 0.094]


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

        if not full_crops:
            return [], [], np.full(num_frames, np.nan), np.zeros(num_frames), np.ones(num_frames)

        torso_crops_for_teams = torso_crops.copy()
        indices_for_teams = indices.copy()
                
        # Stage 1: Legibility filtering
        if self.legibility_predictor and full_crops:
            [full_crops, torso_crops], indices = self.legibility_predictor.filter(
                [full_crops, torso_crops], indices
            )

        if torso_crops:
            jersey_predictions, confs_mean, entropies, raw_probs_list = self.predict_jersey(torso_crops)
        else:
            jersey_predictions, confs_mean, entropies, raw_probs_list = [], [], [], []
        
        jerseys, confs_mean, entropies = self.map_to_frames(jersey_predictions, confs_mean, entropies, indices, num_frames)

        # # Store raw probs on the tracklet for tracklet-level consolidation later
        tracklet.pred_attributes['jersey_raw_probs'] = raw_probs_list
        tracklet.pred_attributes['jersey_raw_probs_indices'] = indices

        return torso_crops_for_teams, indices_for_teams, jerseys, confs_mean, entropies

    def extract_crops(self, images, tracklet):
        """ Extract full and torso crops for all frames in tracklet """
        full_crops = []
        torso_crops = []
        indices = []

        all_keypoints = []
        all_scores = []

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

            # Get torso crop using pose estimation; discard frames where pose
            # confidence is too low, since these crops tend to be unreliable
            # (occlusion, non-torso-visible pose) and the legibility classifier
            # may have slipped a bad crop through.
            if self.pose_cropper:
                result = self.pose_cropper.get_torso_crop(image, bbox, return_keypoints=True)
                if result[0] is None:
                    continue
                torso_crop, (keypoints, scores) = result
                all_keypoints.append(keypoints)
                all_scores.append(scores)
            else:
                torso_crop = self.simple_torso_crop(full_crop_rgb)
                all_keypoints.append(None)
                all_scores.append(None)

            # ========== ROBUST VALIDATION ==========
            if torso_crop is None:
                continue
            if not isinstance(torso_crop, np.ndarray):
                continue
            if torso_crop.size == 0:
                continue
            if len(torso_crop.shape) != 3:
                continue
            
            height, width, channels = torso_crop.shape
            if channels != 3:
                continue
            if height < 5 or width < 5:
                continue
            if height > h or width > w:
                continue

            aspect_ratio = width / height
            if aspect_ratio < 0.1 or aspect_ratio > 10.0:
                continue
            # ==========================================

            # Extra validation just before append
            try:
                test_pil = Image.fromarray(torso_crop)
                if test_pil.size[0] < 5 or test_pil.size[1] < 5:
                    continue
            except Exception as e:
                continue
            
            full_crops.append(full_crop_rgb)
            torso_crops.append(torso_crop)
            indices.append(i)

        tracklet.pose_keypoints = all_keypoints
        tracklet.pose_scores = all_scores
        
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
        return PoseCropper(
            device=self.device,
            model_path=str(vitpose_model_path)
        )

    
    def load_reid_filter_model(self, centroid_reid_path):
        """ Load Centroid-ReID model for outlier filtering """
        return CentroidReIDFilter(
            checkpoint_path=centroid_reid_path,
            threshold=getattr(self.jersey_cfg, 'reid_threshold', 3.5),
            rounds=getattr(self.jersey_cfg, 'reid_rounds', 3),
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
        """
        Run PARSeq on crops to predict jersey numbers.
        
        Matches Koshkina str.py run_inference():
            logits = model.forward(image.to(model.device))
            probs_full = logits[:,:3,:11].softmax(-1)
            preds, probs = model.tokenizer.decode(probs_full)
            logits = logits[:,:3,:11]
            results[filename] = {'label':preds[0], 'confidence':confidence, 'raw': probs_full, 'logits':logits}
        
        Returns:
            jerseys: list of int or np.nan per crop (greedy decoded, for splitting)
            confs_mean: list of float per crop
            entropies: list of float per crop (proper entropy from full distribution)
            raw_probs_list: list of np.array shape (2, 11) per crop — for Bayesian consolidation
        """
        jerseys = []
        confs_mean = []
        entropies = []
        raw_probs_list = []     # NEW: raw probability vectors for Bayesian consolidation
        
        for i in range(0, len(crops), batch_size):
            batch = crops[i:i + batch_size]
            tensors = [self.img_transform(Image.fromarray(c)) for c in batch]
            batch_tensor = torch.stack(tensors).to(self.device)
            
            with torch.no_grad():
                logits = self.parseq_model(batch_tensor)
                
                # ================================================================
                # CRITICAL FIX: Match Koshkina str.py exactly
                # 
                # Koshkina slices logits to [:, :3, :11] before softmax.
                # This extracts only:
                #   - First 3 positions (tens digit, units digit, EOS)
                #   - First 11 vocab entries (maps to TOKEN_LIST = 'E0123456789')
                #
                # PARSeq full vocab is 95+ chars (all ASCII). We only care about
                # the digit tokens. In PARSeq's default charset, the first entries
                # of the vocab are: [EOS, 0, 1, 2, ..., 9, ...more chars]
                # which maps exactly to TOKEN_LIST = 'E0123456789'
                # ================================================================
                logits_sliced = logits[:, :3, :11]                  # [batch, 3, 11]
                probs_full = logits_sliced.softmax(-1)              # [batch, 3, 11]
            
            # Decode using PARSeq tokenizer on the sliced probs
            labels, raw_confs = self.parseq_model.tokenizer.decode(probs_full)
            
            for j, (label, conf_tensor) in enumerate(zip(labels, raw_confs)):
                label = label.strip()
                
                # ================================================================
                # Extract raw probability vectors for Bayesian consolidation
                # Matches what Koshkina saves as 'raw' in the JSON:
                #   probs_full = probs_full.cpu().detach().numpy()[0].tolist()
                #   results[filename] = {..., 'raw': probs_full, ...}
                #
                # raw_probs shape: [3, 11] — we take first 2 positions (tens, units)
                # This is what helpers.py process_jersey_id_predictions_bayesian reads:
                #   raw_result = results_dict[name]['raw']
                #   raw_result = np.array([np.array(xi) for xi in raw_result])
                # ================================================================
                frame_raw_probs = probs_full[j, :2, :].cpu().numpy()   # shape (2, 11)
                raw_probs_list.append(frame_raw_probs)
                
                # Also extract logits for temperature scaling path
                # (matches 'logits' field in Koshkina JSON)
                frame_logits = logits_sliced[j, :2, :].cpu().numpy()   # shape (2, 11)
                
                # Confidence values from tokenizer decode
                conf_values = conf_tensor.cpu().numpy()
                
                # Strip the EOS confidence (last value) — same as original
                if len(conf_values) > 1:
                    char_confs = conf_values[:-1]
                else:
                    char_confs = conf_values

                if label.isdigit() and 0 <= int(label) <= 99:
                    jerseys.append(int(label))
                    # Match Koshkina helpers.py: use product of per-char confidences
                    # (not mean) — product penalizes uncertain characters more aggressively
                    confs_mean.append(float(char_confs.prod()))

                    # ================================================================
                    # FIX: Proper entropy calculation
                    # Old code: probs_clipped = np.clip(conf_values.mean())  ← BROKEN
                    # 
                    # Now: compute entropy from the full 11-class distribution at
                    # each digit position, matching the paper's approach.
                    # ================================================================
                    entropy = self._compute_entropy_from_raw(frame_raw_probs, label)
                    entropies.append(entropy)
                else:
                    jerseys.append(np.nan)
                    confs_mean.append(0.0)
                    entropies.append(1.0)
                    
        return jerseys, confs_mean, entropies, raw_probs_list


    @staticmethod
    def _compute_entropy_from_raw(raw_probs, label):
        """
        Compute Shannon entropy from the full 11-class probability distributions.
        
        raw_probs: shape (2, 11) — softmax probabilities at tens and units positions
        label: decoded string like "14" or "7"
        
        Returns: float entropy summed over the digit positions that matter.
        """
        n_digits = len(label)
        total_entropy = 0.0
        for pos in range(min(n_digits, 2)):
            probs = np.clip(raw_probs[pos], 1e-10, 1.0)
            # Normalize in case clipping changed the sum
            probs = probs / probs.sum()
            total_entropy += float(-np.sum(probs * np.log(probs)))
        return total_entropy


    # ========================================================================
    # Tracklet-level Bayesian Consolidation
    # Exact port of Koshkina helpers.py: predict_jersey_number()
    # Called AFTER all per-frame predictions are done.
    # ========================================================================

    @staticmethod
    def consolidate_tracklet_bayesian(raw_probs_list, use_bias=True, use_ts=False, logits_list=None):
        """
        Bayesian tracklet-level prediction consolidation.
        
        Exact port of Koshkina helpers.py:
            - initialize_priors()
            - split_predictions_by_digit() with update_posteriors()
            - predict_jersey_number()
        
        Args:
            raw_probs_list: list of np.array shape (2, 11) per legible frame
                           These are softmax probs (the 'raw' field in Koshkina JSON)
            use_bias: apply 1-digit vs 2-digit prior (paper uses True)
            use_ts: if True, apply temperature scaling to logits_list instead
            logits_list: list of np.array shape (2, 11) per legible frame (pre-softmax)
        
        Returns:
            jersey_number: int or -1
            confidence: float (sum of log-likelihoods at best digits)
        """
        if not raw_probs_list or len(raw_probs_list) == 0:
            return -1, 0.0
        
        # Build the image_predictions list — each entry is [tens_probs(11,), units_probs(11,)]
        image_predictions = []
        for idx in range(len(raw_probs_list)):
            if use_ts and logits_list is not None:
                # Matches helpers.py apply_ts():
                #   raw = np.array(logits) / TS
                #   conf0 = softmax(raw[0])
                #   conf1 = softmax(raw[1])
                raw_logits = np.array(logits_list[idx]) / TS
                tens_probs = softmax(raw_logits[0])
                units_probs = softmax(raw_logits[1])
            else:
                tens_probs = np.array(raw_probs_list[idx][0])
                units_probs = np.array(raw_probs_list[idx][1])
            
            image_predictions.append(np.array([tens_probs, units_probs]))

        image_predictions = np.array(image_predictions)

        # ---- Exact port of helpers.py predict_jersey_number() ----

        # Initialize priors — matches helpers.py initialize_priors()
        num_digits = 11
        tens_priors = np.full(num_digits, 1.0 / num_digits)
        if use_bias:
            units_priors = np.array(BIAS_FOR_DIGITS)
        else:
            units_priors = np.full(num_digits, 1.0 / num_digits)

        # Split and apply priors — matches helpers.py split_predictions_by_digit()
        tens_likelihoods = []
        units_likelihoods = []
        for entry in image_predictions:
            e0 = entry[0]  # tens position probs (11,)
            e1 = entry[1]  # units position probs (11,)

            # Update posteriors — matches helpers.py update_posteriors()
            tens_posterior = tens_priors * e0
            tens_posterior /= np.sum(tens_posterior)
            
            units_posterior = units_priors * e1
            units_posterior /= np.sum(units_posterior)

            tens_likelihoods.append(tens_posterior)
            units_likelihoods.append(units_posterior)

        tens_likelihoods = np.array(tens_likelihoods)
        units_likelihoods = np.array(units_likelihoods)

        # Sum of log-likelihoods — matches helpers.py predict_jersey_number()
        log_likelihoods_tens = np.log(np.clip(tens_likelihoods, 1e-10, None))
        sum_logl_tens = np.sum(log_likelihoods_tens, axis=0)
        
        log_likelihoods_units = np.log(np.clip(units_likelihoods, 1e-10, None))
        sum_logl_units = np.sum(log_likelihoods_units, axis=0)

        tens_digit = np.argmax(sum_logl_tens)
        units_digit = np.argmax(sum_logl_units)

        prob_tens = sum_logl_tens[tens_digit]
        prob_units = sum_logl_units[units_digit]

        # Map back to token string — matches helpers.py predict_jersey_number()
        batch_tokens = TOKEN_LIST[tens_digit] + TOKEN_LIST[units_digit]
        batch_probs = [prob_tens, prob_units]
        
        # Remove trailing E (end-of-string)
        for i in range(2):
            if batch_tokens[i] == 'E':
                batch_tokens = batch_tokens[:i]
                batch_probs = batch_probs[:i]
                break
        
        if batch_tokens and batch_tokens.isdigit():
            confidence = batch_probs[0] if len(batch_probs) == 1 else batch_probs[0] + batch_probs[1]
            return int(batch_tokens), float(confidence)
        else:
            return -1, 0.0


    @staticmethod
    def consolidate_tracklet_heuristic(jerseys, confs_mean, use_bias=True):
        """
        Heuristic tracklet-level prediction consolidation.
        
        Exact port of Koshkina helpers.py: process_jersey_id_predictions() 
        with find_best_prediction().
        
        Uses confidence-weighted majority vote with optional 1-digit/2-digit bias.
        
        Args:
            jerseys: list of int (per-frame jersey predictions, NaN filtered out)
            confs_mean: list of float (per-frame confidences)
            use_bias: weight 2-digit numbers at 0.61, 1-digit at 0.39
        
        Returns:
            jersey_number: int or -1
            confidence: float
        """
        FILTER_THRESHOLD = 0.2
        SUM_THRESHOLD = 1.0

        # Build results array [value, confidence]
        results = []
        for jersey, conf in zip(jerseys, confs_mean):
            if isinstance(jersey, float) and np.isnan(jersey):
                continue
            results.append([int(jersey), conf])
        
        if not results:
            return -1, 0.0
        
        results = np.array(results)

        # Filter low confidence — matches helpers.py find_best_prediction()
        if FILTER_THRESHOLD > 0:
            for entry in results:
                if entry[1] < FILTER_THRESHOLD:
                    entry[1] = 0

        unique_predictions = np.unique(results[:, 0])
        weights = []
        for value in unique_predictions:
            rows_with_value = results[np.where(results[:, 0] == value)]
            # Bias — matches helpers.py get_bias()
            if use_bias:
                b = 0.61 if int(value) > 9 else 0.39
            else:
                b = 1.0
            adjusted_prob = rows_with_value[:, 1] * b
            sum_weights = np.sum(adjusted_prob)
            weights.append(sum_weights)

        best_weight = np.max(weights)
        index_of_best = np.argmax(weights)
        best_prediction = unique_predictions[index_of_best] if best_weight > SUM_THRESHOLD else -1

        return int(best_prediction), float(best_weight)


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
        
        if len(torso_crops) > max_crops:
            step = len(torso_crops) // max_crops
            crops_to_save = torso_crops[::step][:max_crops]
            indices_to_save = indices[::step][:max_crops]
        else:
            crops_to_save = torso_crops
            indices_to_save = indices
        
        for i, (crop, tracklet_idx) in enumerate(zip(crops_to_save, indices_to_save)):
            frame_idx = tracklet.frames[tracklet_idx]
            
            filename = f"track{tracklet.track_id:04d}_frame{frame_idx:05d}.jpg"
            save_path = debug_dir / filename
            
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_path), crop_bgr)