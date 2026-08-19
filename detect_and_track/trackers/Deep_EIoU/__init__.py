"""Deep-EIoU Tracker - Simple multi-object tracking with ReID"""

from .tracker import DeepEIOUTracker
from .reid.extractor import load_reid_extractor

__version__ = "0.1.0"
__all__ = ["DeepEIOUTracker", "load_reid_extractor"]