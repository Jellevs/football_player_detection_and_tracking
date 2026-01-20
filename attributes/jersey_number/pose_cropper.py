import torch
import cv2
from PIL import Image
from transformers import VitPoseForPoseEstimation, AutoProcessor

# COCO keypoint indices
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12


class PoseCropper:
    """ Pose-based torso cropping using ViTPose from huggingface instead of MMPose because of import/version problems"""
    
    def __init__(self, device='cuda', model_path='usyd-community/vitpose-base-simple'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Check if local path or HuggingFace model name
        local_only = self.is_local_path(model_path)

        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=local_only)
        self.model = VitPoseForPoseEstimation.from_pretrained(model_path, local_files_only=local_only)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.pad_left = 5
        self.pad_right = 5
        self.pad_bottom = 0
        self.pad_top = 5

        self.min_keypoint_conf = 0.3
        
        print(f"PoseCropper loaded on {self.device}")
    
    
    @staticmethod
    def is_local_path(path):
        """ Check if path is a local directory (not a HuggingFace model name) """
        import os
        return os.path.isdir(path)


    def get_torso_crop(self, image, bbox, return_keypoints=False):
        """ Extract torso crop from image using pose estimation """

        x1, y1, x2, y2 = map(int, bbox)

        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        player_crop = image[y1:y2, x1:x2]
        if player_crop.size == 0:
            return (None, None) if return_keypoints else None
        
        if len(player_crop.shape) == 3 and player_crop.shape[2] == 3:
            player_rgb = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
        else:
            player_rgb = player_crop
        pil_image = Image.fromarray(player_rgb)
        
        keypoints, scores = self.detect_pose(pil_image)
        
        if keypoints is None:
            return (None, None) if return_keypoints else None
        
        torso_bbox = self.get_torso_bbox(keypoints, scores, player_crop.shape)
        
        if torso_bbox is None:
            # Fallback to heuristic
            crop_h = player_crop.shape[0]
            torso_y1 = int(crop_h * 0.2)
            torso_y2 = int(crop_h * 0.8)
            torso_crop = player_rgb[torso_y1:torso_y2, :, :]
        else:
            tx1, ty1, tx2, ty2 = torso_bbox
            torso_crop = player_rgb[ty1:ty2, tx1:tx2]
        
        if torso_crop.size == 0:
            return (None, None) if return_keypoints else None
        
        if return_keypoints:
            return torso_crop, (keypoints, scores)
        return torso_crop
    

    def detect_pose(self, pil_image):
        """ Run ViTPose on a single PIL image """

        width, height = pil_image.size
        boxes = [[[0, 0, width, height]]]
        
        inputs = self.processor(
            images=[pil_image],
            boxes=boxes,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        pose_results = self.processor.post_process_pose_estimation(
            outputs,
            boxes=boxes,
        )
        
        if pose_results and len(pose_results) > 0:
            if len(pose_results[0]) > 0:
                person_result = pose_results[0][0]
                keypoints = person_result['keypoints'].cpu().numpy()
                scores = person_result['scores'].cpu().numpy()
                return keypoints, scores
                
        return None, None
            

    def get_torso_bbox(self, keypoints, scores, crop_shape):
        """ Get torso bounding box from keypoints """
        h, w = crop_shape[:2]
        
        left_shoulder = keypoints[LEFT_SHOULDER]
        right_shoulder = keypoints[RIGHT_SHOULDER]
        left_hip = keypoints[LEFT_HIP]
        right_hip = keypoints[RIGHT_HIP]
        
        shoulder_scores = [scores[LEFT_SHOULDER], scores[RIGHT_SHOULDER]]
        hip_scores = [scores[LEFT_HIP], scores[RIGHT_HIP]]
        
        valid_shoulders = sum(s >= self.min_keypoint_conf for s in shoulder_scores)
        valid_hips = sum(s >= self.min_keypoint_conf for s in hip_scores)
        
        if valid_shoulders < 1 or valid_hips < 1:
            return None
        
        x_coords = []
        y_coords = []
        
        if scores[LEFT_SHOULDER] >= self.min_keypoint_conf:
            x_coords.append(left_shoulder[0])
            y_coords.append(left_shoulder[1])
        if scores[RIGHT_SHOULDER] >= self.min_keypoint_conf:
            x_coords.append(right_shoulder[0])
            y_coords.append(right_shoulder[1])
        if scores[LEFT_HIP] >= self.min_keypoint_conf:
            x_coords.append(left_hip[0])
            y_coords.append(left_hip[1])
        if scores[RIGHT_HIP] >= self.min_keypoint_conf:
            x_coords.append(right_hip[0])
            y_coords.append(right_hip[1])
        
        if not x_coords or not y_coords:
            return None
        
        x1 = int(min(x_coords) - self.pad_left)
        x2 = int(max(x_coords) + self.pad_right)
        y1 = int(min(y_coords) - self.pad_top)
        y2 = int(max(y_coords) + self.pad_bottom)
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        return (x1, y1, x2, y2)