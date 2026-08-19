import numpy as np
import supervision as sv


def sv_to_xyxy(detections: sv.Detections) -> np.ndarray:
    """
    Convert supervision Detections to xyxy format with scores
    
    Args:
        detections: supervision Detections object
    
    Returns:
        numpy array (N, 5) with [x1, y1, x2, y2, score]
    """
    if len(detections) == 0:
        return np.empty((0, 5))
    
    return np.column_stack([
        detections.xyxy,
        detections.confidence
    ])


def tracks_to_sv(tracked_objects, original_detections=None) -> sv.Detections:
    """
    Convert Deep-EIoU tracked objects to supervision Detections
    
    Args:
        tracked_objects: List of STrack objects
        original_detections: Optional original detections to preserve class_id
    
    Returns:
        sv.Detections with tracker_id
    """
    if len(tracked_objects) == 0:
        return sv.Detections.empty()
    
    xyxy = np.array([track.tlbr for track in tracked_objects])
    confidence = np.array([track.score for track in tracked_objects])
    tracker_id = np.array([track.track_id for track in tracked_objects])
    
    # Try to preserve class_id if available
    class_id = None
    if original_detections is not None and hasattr(original_detections, 'class_id'):
        if original_detections.class_id is not None:
            class_id = np.zeros(len(tracked_objects), dtype=int)
    
    return sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        tracker_id=tracker_id
    )
