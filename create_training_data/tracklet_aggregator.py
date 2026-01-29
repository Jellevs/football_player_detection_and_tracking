# training_data/fragment_aggregator.py

import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path


class FragmentAggregator:
    """
    Aggregates frame-level tracklet data into fixed-size feature vectors
    for training neural networks.
    """
    
    def __init__(self):
        pass
    
    
    def aggregate_tracklet(self, tracklet) -> Dict:
        """
        Convert a tracklet into aggregated features.
        
        Returns dict with:
        - metadata: tracklet_id, start_frame, end_frame, duration, etc.
        - aggregated_features: fixed-size feature vector
        - gt_info: ground truth for labeling
        """
        
        num_frames = len(tracklet.frames)
        
        if num_frames == 0:
            return None
        
        # ==== Metadata ====
        metadata = {
            'tracklet_id': tracklet.track_id,
            'parent_id': tracklet.parent_id,
            'start_frame': tracklet.frames[0],
            'end_frame': tracklet.frames[-1],
            'duration': num_frames,
        }
        
        # ==== Jersey Features ====
        jersey_features = self._aggregate_jersey(tracklet)
        
        # ==== Team Features (SigLIP embeddings) ====
        team_features = self._aggregate_team_siglip(tracklet)
        
        # ==== ReID Features ====
        reid_features = self._aggregate_reid(tracklet)
        
        # ==== Spatial Features ====
        spatial_features = self._aggregate_spatial(tracklet)
        
        # ==== Temporal Features ====
        temporal_features = self._aggregate_temporal(tracklet)
        
        # ==== Velocity Features ====
        velocity_features = self._aggregate_velocity(tracklet)
        
        # ==== Ground Truth Info ====
        gt_info = self._get_gt_info(tracklet)
        
        # Combine all features
        aggregated = {
            **jersey_features,
            **team_features,
            **reid_features,
            **spatial_features,
            **temporal_features,
            **velocity_features,
        }
        
        return {
            'metadata': metadata,
            'aggregated': aggregated,
            'gt_info': gt_info,
        }
    
    
    def _aggregate_jersey(self, tracklet) -> Dict:
        """Aggregate jersey number predictions and entropies"""
        jerseys = tracklet.pred_attributes.get('jerseys', [])
        entropies = tracklet.pred_attributes.get('jersey_entropies', [])
        
        if not jerseys or len(jerseys) == 0:
            return {
                'jersey_mode': -1,
                'jersey_entropy_mean': 1.0,
                'jersey_entropy_std': 0.0,
                'jersey_entropy_min': 1.0,
                'jersey_entropy_max': 1.0,
                'jersey_confidence': 0.0,
                'jersey_consistency': 0.0,
            }
        
        # Filter valid predictions (not NaN)
        valid_mask = [not (isinstance(j, float) and np.isnan(j)) for j in jerseys]
        valid_jerseys = [j for j, valid in zip(jerseys, valid_mask) if valid]
        valid_entropies = [e for e, valid in zip(entropies, valid_mask) if valid]
        
        if not valid_jerseys:
            return {
                'jersey_mode': -1,
                'jersey_entropy_mean': 1.0,
                'jersey_entropy_std': 0.0,
                'jersey_entropy_min': 1.0,
                'jersey_entropy_max': 1.0,
                'jersey_confidence': 0.0,
                'jersey_consistency': 0.0,
            }
        
        # Mode (most common jersey)
        jersey_mode = int(max(set(valid_jerseys), key=valid_jerseys.count))
        
        # Entropy statistics
        entropies_array = np.array(valid_entropies)
        
        # Jersey consistency: what % of frames agree with mode?
        jersey_consistency = sum(1 for j in valid_jerseys if j == jersey_mode) / len(valid_jerseys)
        
        return {
            'jersey_mode': jersey_mode,
            'jersey_entropy_mean': float(np.mean(entropies_array)),
            'jersey_entropy_std': float(np.std(entropies_array)),
            'jersey_entropy_min': float(np.min(entropies_array)),
            'jersey_entropy_max': float(np.max(entropies_array)),
            'jersey_confidence': 1.0 - float(np.mean(entropies_array)),  # Lower entropy = higher confidence
            'jersey_consistency': jersey_consistency,
        }
    
    
    def _aggregate_team_siglip(self, tracklet) -> Dict:
        """
        Aggregate team features using SigLIP embeddings.
        
        Note: We're using the MEAN of SigLIP embeddings across frames,
        not the UMAP projections or K-means labels.
        """
        teams = tracklet.pred_attributes.get('teams', [])
        
        # We need to extract SigLIP embeddings from crops
        # This is tricky because your current pipeline doesn't store per-frame SigLIP embeddings
        # They're computed in batch and only K-means labels are stored back to tracklets
        
        # OPTION 1: Store team_mode from K-means as auxiliary feature
        # (We'll need to modify attributes.py to also store SigLIP embeddings per frame)
        
        # For now, let's use what we have: K-means labels
        valid_teams = [t for t in teams if not (isinstance(t, float) and np.isnan(t))]
        
        if not valid_teams:
            return {
                'team_mode': -1,
                'team_consistency': 0.0,
            }
        
        team_mode = int(max(set(valid_teams), key=valid_teams.count))
        team_consistency = sum(1 for t in valid_teams if t == team_mode) / len(valid_teams)
        
        # TODO: Add mean SigLIP embedding (768-dim) once we store it
        return {
            'team_mode': team_mode,
            'team_consistency': team_consistency,
            # 'siglip_mean': np.zeros(768),  # Placeholder - need to add this!
            # 'siglip_std': np.zeros(768),
        }
    
    
    def _aggregate_reid(self, tracklet) -> Dict:
        """Aggregate ReID embeddings (OSNet)"""
        if not tracklet.embeddings or len(tracklet.embeddings) == 0:
            return {
                'reid_mean': np.zeros(512),
                'reid_std': np.zeros(512),
            }
        
        embeddings = np.array(tracklet.embeddings)
        
        return {
            'reid_mean': np.mean(embeddings, axis=0),  # [512]
            'reid_std': np.std(embeddings, axis=0),    # [512]
        }
    
    
    def _aggregate_spatial(self, tracklet) -> Dict:
        """Aggregate spatial bbox features"""
        bboxes = np.array(tracklet.bboxes)  # [N, 4] (x1, y1, x2, y2)
        
        # Compute centers
        centers_x = (bboxes[:, 0] + bboxes[:, 2]) / 2
        centers_y = (bboxes[:, 1] + bboxes[:, 3]) / 2
        
        # Compute bbox sizes
        widths = bboxes[:, 2] - bboxes[:, 0]
        heights = bboxes[:, 3] - bboxes[:, 1]
        
        return {
            'center_x_mean': float(np.mean(centers_x)),
            'center_y_mean': float(np.mean(centers_y)),
            'center_x_std': float(np.std(centers_x)),
            'center_y_std': float(np.std(centers_y)),
            'bbox_width_mean': float(np.mean(widths)),
            'bbox_height_mean': float(np.mean(heights)),
            'bbox_width_std': float(np.std(widths)),
            'bbox_height_std': float(np.std(heights)),
            'spatial_extent_x': float(np.max(centers_x) - np.min(centers_x)),
            'spatial_extent_y': float(np.max(centers_y) - np.min(centers_y)),
        }
    
    
    def _aggregate_temporal(self, tracklet) -> Dict:
        """Aggregate temporal features"""
        frames = tracklet.frames
        
        # Frame density: frames per second (assuming 25 fps)
        frame_range = frames[-1] - frames[0] + 1
        frame_density = len(frames) / frame_range if frame_range > 0 else 1.0
        
        return {
            'start_frame': frames[0],
            'end_frame': frames[-1],
            'duration_frames': len(frames),
            'frame_range': frame_range,
            'frame_density': frame_density,
        }
    
    
    def _aggregate_velocity(self, tracklet) -> Dict:
        """Aggregate velocity features"""
        bboxes = np.array(tracklet.bboxes)
        
        # Compute centers
        centers = np.column_stack([
            (bboxes[:, 0] + bboxes[:, 2]) / 2,
            (bboxes[:, 1] + bboxes[:, 3]) / 2
        ])
        
        # Compute frame-to-frame velocities
        if len(centers) < 2:
            return {
                'velocity_x_mean': 0.0,
                'velocity_y_mean': 0.0,
                'velocity_mag_mean': 0.0,
                'velocity_mag_std': 0.0,
                'velocity_direction_mean': 0.0,
            }
        
        velocities = np.diff(centers, axis=0)  # [N-1, 2]
        
        velocity_mag = np.linalg.norm(velocities, axis=1)
        velocity_directions = np.arctan2(velocities[:, 1], velocities[:, 0])  # radians
        
        return {
            'velocity_x_mean': float(np.mean(velocities[:, 0])),
            'velocity_y_mean': float(np.mean(velocities[:, 1])),
            'velocity_mag_mean': float(np.mean(velocity_mag)),
            'velocity_mag_std': float(np.std(velocity_mag)),
            'velocity_direction_mean': float(np.mean(velocity_directions)),
        }
    
    
    def _get_gt_info(self, tracklet) -> Dict:
        """Extract ground truth information for labeling"""
        gt_track_ids = tracklet.gt_attributes.get('track_ids', [])
        
        if not gt_track_ids or len(gt_track_ids) == 0:
            return {
                'has_gt': False,
                'gt_track_id_mode': -1,
                'purity': 0.0,
                'is_pure': False,
            }
        
        # Find most common gt_track_id
        from collections import Counter
        counter = Counter(gt_track_ids)
        gt_mode, mode_count = counter.most_common(1)[0]
        
        purity = mode_count / len(gt_track_ids)
        is_pure = purity >= 0.95  # At least 95% same player
        
        return {
            'has_gt': True,
            'gt_track_id_mode': int(gt_mode),
            'purity': float(purity),
            'is_pure': bool(is_pure),
        }