import numpy as np
import supervision as sv
from types import SimpleNamespace
from typing import Optional

from .core.Deep_EIoU import Deep_EIoU
from .reid.extractor import load_reid_extractor

class DeepEIOUTracker:
    """ Deep-EIoU tracker interface using supervision Detections object """
    def __init__(
        self,
        track_thresh: float = 0.6,
        track_low_thresh: float = 0.3,
        new_track_thresh: float = 0.4,
        track_buffer: int = 60,
        match_thresh: float = 0.8,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.2,
        with_reid: bool = False,
        reid_model_path: Optional[str] = None,
        reid_model_name: str = 'osnet_x0_25',
        device: str = 'cuda',
        frame_rate: int = 25
    ):        
        if with_reid and reid_model_path is None:
            raise ValueError("reid_model_path must be provided when with_reid=True")
        
        self.args = SimpleNamespace(
            track_high_thresh=track_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            proximity_thresh=proximity_thresh,
            appearance_thresh=appearance_thresh,
            with_reid=with_reid
        )
        
        self.tracker = Deep_EIoU(self.args, frame_rate=frame_rate)
        
        # Initialize ReID if enabled
        self.reid_extractor = None
        if with_reid:
            self.reid_extractor = load_reid_extractor(
                reid_model_path,
                device=device,
                model_name=reid_model_name
            )
        
        self.with_reid = with_reid
        
        # Store last embeddings for team clustering
        self.last_embeddings = None
        self.last_detections_array = None


    def update(self, detections, frame):
        """ Update tracker with new detections """
        # Convert supervision format to Deep-EIoU format
        if len(detections) == 0:
            detections_array = np.empty((0, 5))
        else:
            detections_array = np.column_stack([
                detections.xyxy,
                detections.confidence
            ])
        
        # Extract ReID features if enabled
        if self.with_reid and len(detections_array) > 0:
            embeddings = self.reid_extractor.extract_features_from_detections(
                frame, detections_array
            )
        else:
            embeddings = np.array([]).reshape(0, 512)
        
        # Store embeddings and detections
        self.last_embeddings = embeddings
        self.last_detections_array = detections_array
        
        # Update tracker
        tracked_objects = self.tracker.update(detections_array, embeddings)
        
        # Convert back to supervision format
        if len(tracked_objects) == 0:
            return sv.Detections.empty()
        
        # Extract tracking results
        tracked_xyxy = []
        tracked_confidence = []
        tracked_class_id = []
        tracked_ids = []
        tracked_embeddings = []  
        
        for track in tracked_objects:
            tlbr = track.tlbr
            tracked_xyxy.append(tlbr)
            tracked_confidence.append(track.score)
            tracked_ids.append(track.track_id)

            if self.with_reid and hasattr(track, 'smooth_feat') and track.smooth_feat is not None:
                tracked_embeddings.append(track.smooth_feat)
            else:
                tracked_embeddings.append(np.zeros(512))
            
            # Preserve class_id
            if hasattr(detections, 'class_id') and detections.class_id is not None:
                best_iou = 0
                best_class_id = 0
                
                for i, det_box in enumerate(detections.xyxy):
                    iou = self.compute_iou(tlbr, det_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_class_id = detections.class_id[i]
                
                if best_iou > 0.5:
                    tracked_class_id.append(best_class_id)
                else:
                    tracked_class_id.append(0)
            else:
                tracked_class_id.append(0)
        
        # Create supervision Detections
        tracked_detections = sv.Detections(
            xyxy=np.array(tracked_xyxy),
            confidence=np.array(tracked_confidence),
            class_id=np.array(tracked_class_id) if tracked_class_id else None,
            tracker_id=np.array(tracked_ids)
        )
        
        tracked_detections.data = {}

        if self.with_reid:
            tracked_detections.data['reid_embedding'] = np.array(tracked_embeddings)

        # Copy attributes
        if hasattr(detections, 'data') and detections.data:
            gt_track_ids = []
            gt_jerseys = []
            gt_teams = []
            gt_roles = []

            for tracked_box in tracked_xyxy:
                best_iou = 0
                best_idx = -1

                for j, det_box in enumerate(detections.xyxy):
                    iou = self.compute_iou(tracked_box, det_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j
                        

                if best_iou > 0.5 and best_idx != -1:
                    gt_track_ids.append(detections.data.get('gt_track_id', [-1])[best_idx])
                    gt_jerseys.append(detections.data.get('gt_jersey', ['unknown'])[best_idx])
                    gt_teams.append(detections.data.get('gt_team', ['unknown'])[best_idx])
                    gt_roles.append(detections.data.get('gt_role', ['unknown'])[best_idx])

                else:
                    gt_track_ids.append(-1)
                    gt_jerseys.append('unknown')
                    gt_teams.append('unknown')
                    gt_roles.append('unknown')

            tracked_detections.data['gt_track_id'] = np.array(gt_track_ids)
            tracked_detections.data['gt_jersey'] = np.array(gt_jerseys)
            tracked_detections.data['gt_team'] = np.array(gt_teams)
            tracked_detections.data['gt_role'] = np.array(gt_roles)

        return tracked_detections


    def get_last_embeddings(self, tracked_detections: sv.Detections) -> np.ndarray:
        """
        Get ReID embeddings for the tracked detections from the last update() call.
        This matches embeddings to tracked objects by finding the best IoU match.
        """
        if not self.with_reid or self.last_embeddings is None:
            return np.array([]).reshape(len(tracked_detections), 512)
        
        if len(tracked_detections) == 0:
            return np.array([]).reshape(0, 512)
        
        # Match tracked detections to original detections using IoU
        matched_embeddings = []
        
        for tracked_box in tracked_detections.xyxy:
            best_iou = 0
            best_idx = 0
            
            # Find the detection with highest IoU
            for i, det_box in enumerate(self.last_detections_array[:, :4]):
                iou = self.compute_iou(tracked_box, det_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            # Use the embedding from the best matching detection
            if best_iou > 0.5:
                matched_embeddings.append(self.last_embeddings[best_idx])
            else:
                # No good match, use zero embedding
                matched_embeddings.append(np.zeros(512))
        
        return np.array(matched_embeddings)
    
    
    @staticmethod
    def compute_iou(box1, box2):
        """ Compute IoU between two boxes [x1, y1, x2, y2] """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)