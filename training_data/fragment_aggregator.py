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
        """
        
        num_frames = len(tracklet.frames)
        
        if num_frames == 0:
            return None
        
        # ==== Metadata ====
        metadata = {
            'tracklet_id': tracklet.track_id,
            'start_frame': tracklet.frames[0],
            'end_frame': tracklet.frames[-1],
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
            'gt_info': gt_info,  # Used for labeling
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
                'jersey_consistency': 0.0,
            }
        
        # Filter valid predictions
        valid_mask = [not (isinstance(j, float) and np.isnan(j)) for j in jerseys]
        valid_jerseys = [j for j, valid in zip(jerseys, valid_mask) if valid]
        valid_entropies = [e for e, valid in zip(entropies, valid_mask) if valid]
        
        if not valid_jerseys:
            return {
                'jersey_mode': -1,
                'jersey_entropy_mean': 1.0,
                'jersey_entropy_std': 0.0,
                'jersey_consistency': 0.0,
            }
        
        jersey_mode = int(max(set(valid_jerseys), key=valid_jerseys.count))
        entropies_array = np.array(valid_entropies)
        jersey_consistency = sum(1 for j in valid_jerseys if j == jersey_mode) / len(valid_jerseys)
        
        return {
            'jersey_mode': jersey_mode,
            'jersey_entropy_mean': float(np.mean(entropies_array)),
            'jersey_entropy_std': float(np.std(entropies_array)),
            'jersey_consistency': jersey_consistency,
        }


    def _aggregate_team_siglip(self, tracklet) -> Dict:
        """Aggregate SigLIP embeddings"""
        siglip_embeddings = tracklet.pred_attributes.get('siglip_embeddings', [])
        teams = tracklet.pred_attributes.get('teams', [])
        
        # Filter valid embeddings
        valid_embeddings = [emb for emb in siglip_embeddings 
                        if isinstance(emb, np.ndarray) and not np.allclose(emb, 0)]
        
        valid_teams = [t for t in teams if not (isinstance(t, float) and np.isnan(t))]
        
        if valid_teams:
            team_mode = int(max(set(valid_teams), key=valid_teams.count))
            team_consistency = sum(1 for t in valid_teams if t == team_mode) / len(valid_teams)
        else:
            team_mode = -1
            team_consistency = 0.0
        
        if valid_embeddings:
            embeddings_array = np.array(valid_embeddings)
            siglip_mean = np.mean(embeddings_array, axis=0)
        else:
            siglip_mean = np.zeros(768)
        
        return {
            'team_mode': team_mode,
            'team_consistency': team_consistency,
            'siglip_mean': siglip_mean,
        }


    def _aggregate_reid(self, tracklet) -> Dict:
        """Aggregate ReID embeddings"""
        if not tracklet.embeddings or len(tracklet.embeddings) == 0:
            return {'reid_mean': np.zeros(512)}
        
        embeddings = np.array(tracklet.embeddings)
        return {'reid_mean': np.mean(embeddings, axis=0)}


    def _aggregate_spatial(self, tracklet) -> Dict:
        """Aggregate spatial bbox features"""
        bboxes = np.array(tracklet.bboxes)
        
        centers_x = (bboxes[:, 0] + bboxes[:, 2]) / 2
        centers_y = (bboxes[:, 1] + bboxes[:, 3]) / 2
        heights = bboxes[:, 3] - bboxes[:, 1]
        
        return {
            # Trajectory endpoints
            'start_x': float(centers_x[0]),
            'start_y': float(centers_y[0]),
            'end_x': float(centers_x[-1]),
            'end_y': float(centers_y[-1]),
            # Zone information
            'center_x_mean': float(np.mean(centers_x)),
            'center_y_mean': float(np.mean(centers_y)),
            # Player size
            'bbox_height_mean': float(np.mean(heights)),
        }


    def _aggregate_temporal(self, tracklet) -> Dict:
        """Aggregate temporal features"""
        return {
            'duration_frames': len(tracklet.frames),
        }


    def _aggregate_velocity(self, tracklet) -> Dict:
        """Aggregate velocity features"""
        bboxes = np.array(tracklet.bboxes)
        centers = np.column_stack([
            (bboxes[:, 0] + bboxes[:, 2]) / 2,
            (bboxes[:, 1] + bboxes[:, 3]) / 2
        ])
        
        if len(centers) < 2:
            return {
                'velocity_x_mean': 0.0,
                'velocity_y_mean': 0.0,
                'velocity_mag_std': 0.0,
            }
        
        velocities = np.diff(centers, axis=0)
        velocity_mag = np.linalg.norm(velocities, axis=1)
        
        return {
            'velocity_x_mean': float(np.mean(velocities[:, 0])),
            'velocity_y_mean': float(np.mean(velocities[:, 1])),
            'velocity_mag_std': float(np.std(velocity_mag)),
        }
    

    def _get_gt_info(self, tracklet) -> Dict:
        """Extract ground truth information for labeling"""
        gt_track_ids = tracklet.gt_attributes.get('track_ids', [])
        
        if not gt_track_ids or len(gt_track_ids) == 0:
            return {
                'gt_track_id_mode': -1,
                'is_pure': False,
            }
        
        # Find most common gt_track_id
        from collections import Counter
        counter = Counter(gt_track_ids)
        gt_mode, mode_count = counter.most_common(1)[0]
        
        purity = mode_count / len(gt_track_ids)
        is_pure = purity >= 0.95  # At least 95% same player
        
        return {
            'gt_track_id_mode': int(gt_mode),
            'is_pure': bool(is_pure),
        }