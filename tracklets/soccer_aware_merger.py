import numpy as np
import torch
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from typing import Dict

class SoccerAwareMerger:
    def __init__(self, merge_threshold=0.4, spatial_factor=1.0):
        self.merge_threshold = merge_threshold
        self.spatial_factor = spatial_factor

    def merge(self, tracklets: Dict):
        """
        Main entry point for GTA-style merging using hierarchical clustering.
        """
        tracklet_ids = sorted(tracklets.keys())
        n = len(tracklet_ids)
        if n < 2: return tracklets

        # 1. Build Distance Matrix
        dist_matrix = self.compute_distance_matrix(tracklets, tracklet_ids)

        # 2. Hierarchical Clustering (GTA uses Average Linkage)
        condensed_dist = squareform(dist_matrix)
        Z = linkage(condensed_dist, method='average')
        
        # Cut tree at threshold alpha
        cluster_labels = fcluster(Z, t=self.merge_threshold, criterion='distance')
        
        # 3. Map clusters and execute merges
        return self.apply_cluster_merges(tracklets, tracklet_ids, cluster_labels)

    def compute_distance_matrix(self, tracklets, tracklet_ids):
        n = len(tracklet_ids)
        dist_matrix = np.ones((n, n))
        np.fill_diagonal(dist_matrix, 0)

        # Pre-calculate spatial constraints for the whole sequence
        # (Using your existing function from refine_tracklets.py)
        # max_x, max_y = get_spatial_constraints(tracklets, self.spatial_factor)

        for i in range(n):
            for j in range(i + 1, n):
                t1, t2 = tracklets[tracklet_ids[i]], tracklets[tracklet_ids[j]]
                
                # HARD CONSTRAINT: Temporal Overlap
                if set(t1.frames) & set(t2.frames):
                    dist_matrix[i, j] = dist_matrix[j, i] = 9999
                    continue # Distance remains 1.0

                # CALC BASE DISTANCE (Cosine similarity of ReID)
                dist = self.get_soccer_distance(t1, t2)
                
                dist_matrix[i, j] = dist_matrix[j, i] = dist

        return dist_matrix

    def get_soccer_distance(self, t1, t2):
        """
        Calculates distance with soft team constraints to handle occlusions.
        """
        # A. ReID Base (Average Cosine Distance)
        feat1 = np.mean(t1.embeddings, axis=0)
        feat2 = np.mean(t2.embeddings, axis=0)
        cos_sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
        dist = 1.0 - cos_sim

        # B. Soft Team Constraint
        # Handle the 'occlusion trap': only penalize if consistency is high
        # Pulling 'mode' and 'consistency' from your existing pred_attributes logic
        team1, cons1 = self.get_team_stats(t1)
        team2, cons2 = self.get_team_stats(t2)

        if team1 is not None and team2 is not None and team1 != team2:
            # If both are very consistent (no occlusion), it's a different person
            if cons1 > 0.9 and cons2 > 0.9:
                return 1.0 
            # If consistency is low, it might be an occlusion; apply smaller penalty
            dist *= 1.2 

        # C. Jersey Bonus (Entropy-based)
        # Entropy < 0.15 is your 'very confident' threshold
        j1, e1 = self.get_jersey_stats(t1)
        j2, e2 = self.get_jersey_stats(t2)

        if j1 == j2 and j1 is not None and e1 < 0.15 and e2 < 0.15:
            dist -= 0.3 # Matching confident jerseys bridges the ReID gap

        return np.clip(dist, 0, 1)

    def get_team_stats(self, tracklet):
        """Helper to extract mode and consistency from pred_attributes['teams']"""
        teams = [t for t in tracklet.pred_attributes.get('teams', []) if not np.isnan(t)]
        if not teams: return None, 0.0
        mode = max(set(teams), key=teams.count)
        consistency = teams.count(mode) / len(tracklet.frames)
        return mode, consistency

    def get_jersey_stats(self, tracklet):
        """Helper to extract jersey info and mean entropy"""
        jerseys = [j for j in tracklet.pred_attributes.get('jerseys', []) if not np.isnan(j)]
        entropies = tracklet.pred_attributes.get('jersey_entropies', [1.0])
        if not jerseys: return None, 1.0
        mode = max(set(jerseys), key=jerseys.count)
        return mode, np.mean(entropies)

    def apply_cluster_merges(self, tracklets, tracklet_ids, cluster_labels):
        """Consolidates clusters into final tracklets"""
        clusters = {}
        for idx, cluster_id in enumerate(cluster_labels):
            clusters.setdefault(cluster_id, []).append(tracklet_ids[idx])

        merged_results = {}
        for cluster_id, member_ids in clusters.items():
            # Sort by start frame to maintain chronological order
            member_ids.sort(key=lambda x: tracklets[x].frames[0])
            
            # Use your existing merge_tracklet_list logic
            base_track = tracklets[member_ids[0]]
            for next_id in member_ids[1:]:
                other = tracklets[next_id]
                # Combine all lists as per tracklet.py structure
                base_track.frames.extend(other.frames)
                base_track.bboxes.extend(other.bboxes)
                base_track.scores.extend(other.scores)
                base_track.embeddings.extend(other.embeddings)
                
                for key in base_track.pred_attributes:
                    base_track.pred_attributes[key].extend(other.pred_attributes.get(key, []))

            merged_results[cluster_id] = base_track

        return merged_results
    

    