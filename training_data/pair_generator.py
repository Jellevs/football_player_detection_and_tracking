# training_data/pair_generator.py

import numpy as np
from typing import Dict, List, Tuple
from itertools import combinations


class PairGenerator:
    """
    Generate pairwise comparisons between tracklet fragments
    for training a merge classifier.
    """
    
    def __init__(self, 
                 max_temporal_gap=100,
                 include_all_positives=True,
                 negative_sampling_ratio=2.0):
        """
        Args:
            max_temporal_gap: Maximum frames between tracklets to consider (reduces O(N²))
            include_all_positives: Include all positive pairs (same player)
            negative_sampling_ratio: Ratio of negatives to positives to sample
        """
        self.max_temporal_gap = max_temporal_gap
        self.include_all_positives = include_all_positives
        self.negative_sampling_ratio = negative_sampling_ratio
    
    
    def generate_pairs(self, aggregated_fragments: List[Dict]) -> List[Dict]:
        """
        Generate pairwise comparisons between fragments.
        
        Args:
            aggregated_fragments: List of dicts from FragmentAggregator
            
        Returns:
            List of pair dicts with features and labels
        """
        
        pairs = []
        
        # Filter to only pure fragments for training
        pure_fragments = [
            f for f in aggregated_fragments 
            if f['gt_info']['is_pure']
        ]
        
        print(f"Generating pairs from {len(pure_fragments)} pure fragments "
              f"(out of {len(aggregated_fragments)} total)")
        
        # Collect positive and negative candidates
        positive_pairs = []
        negative_pairs = []
        
        # Generate all valid pairs
        for i, frag_A in enumerate(pure_fragments):
            for frag_B in pure_fragments[i+1:]:
                
                # Check temporal overlap (impossible to be same player)
                if self._has_temporal_overlap(frag_A, frag_B):
                    continue
                
                # Check temporal proximity (reduce search space)
                temporal_gap = self._compute_temporal_gap(frag_A, frag_B)
                if temporal_gap > self.max_temporal_gap:
                    continue
                
                # Compute pair features
                pair_features = self._compute_pair_features(frag_A, frag_B)
                
                # Determine label
                label = self._get_label(frag_A, frag_B)
                
                pair_data = {
                    'fragment_A_id': frag_A['metadata']['tracklet_id'],
                    'fragment_B_id': frag_B['metadata']['tracklet_id'],
                    'features': pair_features,
                    'label': label,
                }
                
                if label == 1:
                    positive_pairs.append(pair_data)
                else:
                    negative_pairs.append(pair_data)
        
        # Sample pairs
        if self.include_all_positives:
            pairs.extend(positive_pairs)
        
        # Sample negatives
        num_negatives = int(len(positive_pairs) * self.negative_sampling_ratio)
        if num_negatives < len(negative_pairs):
            sampled_negatives = np.random.choice(
                len(negative_pairs), 
                size=num_negatives, 
                replace=False
            )
            pairs.extend([negative_pairs[i] for i in sampled_negatives])
        else:
            pairs.extend(negative_pairs)
        
        print(f"Generated {len(positive_pairs)} positive pairs, "
              f"{min(num_negatives, len(negative_pairs))} negative pairs")
        
        return pairs
    
    
    def _has_temporal_overlap(self, frag_A: Dict, frag_B: Dict) -> bool:
        """Check if two fragments exist at the same time"""
        A_start = frag_A['metadata']['start_frame']
        A_end = frag_A['metadata']['end_frame']
        B_start = frag_B['metadata']['start_frame']
        B_end = frag_B['metadata']['end_frame']
        
        # No overlap if one ends before the other starts
        if A_end < B_start or B_end < A_start:
            return False
        return True
    
    
    def _compute_temporal_gap(self, frag_A: Dict, frag_B: Dict) -> int:
        """Compute frames between two fragments"""
        A_end = frag_A['metadata']['end_frame']
        B_start = frag_B['metadata']['start_frame']
        
        if B_start > A_end:
            return B_start - A_end
        else:
            A_start = frag_A['metadata']['start_frame']
            B_end = frag_B['metadata']['end_frame']
            return A_start - B_end
    
    
    def _compute_pair_features(self, frag_A: Dict, frag_B: Dict) -> Dict:
        """Compute pairwise features between two fragments"""
        
        A_agg = frag_A['aggregated']
        B_agg = frag_B['aggregated']
        
        # Temporal
        temporal_gap = self._compute_temporal_gap(frag_A, frag_B)
        
        # Spatial
        endpoint_distance = np.sqrt(
            (A_agg['end_x'] - B_agg['start_x'])**2 +
            (A_agg['end_y'] - B_agg['start_y'])**2
        )
        
        spatial_distance = np.sqrt(
            (A_agg['center_x_mean'] - B_agg['center_x_mean'])**2 +
            (A_agg['center_y_mean'] - B_agg['center_y_mean'])**2
        )
        
        # Jersey
        jersey_match = int(A_agg['jersey_mode'] == B_agg['jersey_mode'])
        jersey_entropy_diff = abs(A_agg['jersey_entropy_mean'] - B_agg['jersey_entropy_mean'])
        
        # Team
        team_match = int(A_agg['team_mode'] == B_agg['team_mode'])
        
        # SigLIP
        siglip_cosine_sim = self._cosine_similarity(A_agg['siglip_mean'], B_agg['siglip_mean'])
        siglip_euclidean_dist = np.linalg.norm(A_agg['siglip_mean'] - B_agg['siglip_mean'])
        
        # ReID
        reid_cosine_sim = self._cosine_similarity(A_agg['reid_mean'], B_agg['reid_mean'])
        reid_euclidean_dist = np.linalg.norm(A_agg['reid_mean'] - B_agg['reid_mean'])
        
        # Bbox size
        bbox_height_ratio = A_agg['bbox_height_mean'] / (B_agg['bbox_height_mean'] + 1e-6)
        
        # Velocity
        velocity_similarity = np.sqrt(
            (A_agg['velocity_x_mean'] - B_agg['velocity_x_mean'])**2 +
            (A_agg['velocity_y_mean'] - B_agg['velocity_y_mean'])**2
        )
        
        pairwise_features = {
            'temporal_gap': temporal_gap,
            'endpoint_distance': endpoint_distance,
            'spatial_distance': spatial_distance,
            'jersey_match': jersey_match,
            'jersey_entropy_diff': jersey_entropy_diff,
            'team_match': team_match,
            'siglip_cosine_sim': siglip_cosine_sim,
            'siglip_euclidean_dist': siglip_euclidean_dist,
            'reid_cosine_sim': reid_cosine_sim,
            'reid_euclidean_dist': reid_euclidean_dist,
            'bbox_height_ratio': bbox_height_ratio,
            'velocity_similarity': velocity_similarity,
        }
        
        return {
            'fragment_A': A_agg,
            'fragment_B': B_agg,
            'pairwise': pairwise_features,
        }
        
    
    def _cosine_similarity(self, vec_A: np.ndarray, vec_B: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot_product = np.dot(vec_A, vec_B)
        norm_A = np.linalg.norm(vec_A)
        norm_B = np.linalg.norm(vec_B)
        
        if norm_A < 1e-6 or norm_B < 1e-6:
            return 0.0
        
        return float(dot_product / (norm_A * norm_B))
    
    
    def _get_label(self, frag_A: Dict, frag_B: Dict) -> int:
        """
        Determine if two fragments should be merged (same player).
        
        Returns:
            1 if same player (should merge)
            0 if different players (should not merge)
        """
        gt_A = frag_A['gt_info']['gt_track_id_mode']
        gt_B = frag_B['gt_info']['gt_track_id_mode']
        
        return 1 if gt_A == gt_B else 0