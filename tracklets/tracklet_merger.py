import numpy as np
import xgboost
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

from training_data.fragment_aggregator import FragmentAggregator
from training_data.pca_reducer import PCAReducer


class TrackletMerger:
    def __init__(self, merger_cfg):
        self.linkage_method = merger_cfg.linkage_method # TODO: test single linking and average, increase threshold such that transitive tracklets also match
        self.merge_threshold = merger_cfg.merge_threshold

        self.aggregator = FragmentAggregator()
        self.pca_reducer = PCAReducer.load(merger_cfg.pca_model_path)
        self.xgb_model = self.load_xgboost_model(merger_cfg.xgboost_model_path)


    def load_xgboost_model(self, model_path: str):
        """Load trained XGBoost model."""
        # Load saved model
        model = xgboost.Booster()  
        model.load_model(model_path)
        return model
    

    def merge_tracklets(self, tracklets):
        fragments, track_id_map = self.aggregate_tracklets(tracklets)

        # Apply PCA TODO: Test without PCA
        fragments = self.pca_reducer.transform(fragments)

        similarity_matrix = self.compute_similarity_matrix(fragments)

        cluster_labels = self.hierarchical_clustering(similarity_matrix)

        merged_tracklets = self.apply_merges(tracklets, track_id_map, cluster_labels, fragments)

        return merged_tracklets


    def hierarchical_clustering(self, similarity_matrix):
        """
        Apply hierarchical clustering to similarity matrix.
        
        Returns:
            cluster_labels: Array of cluster assignments for each fragment
        """
        n = len(similarity_matrix)
        
        if n < 2:
            return np.array([0])
        
        # Convert similarity to distance (1 - similarity)
        distance_matrix = 1.0 - similarity_matrix
        
        # Ensure distance matrix is valid
        np.fill_diagonal(distance_matrix, 0)
        distance_matrix = np.clip(distance_matrix, 0, 1)
        
        # Make symmetric (handle floating point issues)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        
        # Convert to condensed form for scipy
        condensed_dist = squareform(distance_matrix, checks=False)
        
        # Perform hierarchical clustering
        Z = linkage(condensed_dist, method=self.linkage_method)
        
        # Cut tree at threshold (convert threshold to distance)
        distance_threshold = 1.0 - self.merge_threshold
        cluster_labels = fcluster(Z, t=distance_threshold, criterion='distance')
        
        # Adjust to 0-indexed
        cluster_labels = cluster_labels - 1
        
        n_clusters = len(np.unique(cluster_labels))
        print(f"  Formed {n_clusters} clusters from {n} fragments")
        
        return cluster_labels
    

    def aggregate_tracklets(self, tracklets):
        """ Aggregate Attributes from Tracklets """
        fragments = []
        track_id_map = {}

        for track_id, tracklet in tracklets.items():
            agg = self.aggregator.aggregate_tracklet(tracklet)

            if agg is not None:
                idx = len(fragments)
                track_id_map[idx] = track_id
                fragments.append(agg)

        return fragments, track_id_map
    

    def compute_similarity_matrix(self, fragments):
        n = len(fragments)
        similarity_matrix = np.eye(n)

        feature_rows = []
        pair_indices = []
        for i in range(n):
            for j in range(i+1, n):
                frag_A = fragments[i]
                frag_B = fragments[j]

                # Fragments that overlap can never be the same player
                if self.has_temporal_overlap(frag_A, frag_B):
                    continue

                features = self.compute_pair_features(frag_A, frag_B)
                feature_rows.append(features)
                pair_indices.append((i,j))

        
        feature_matrix = np.array(feature_rows)
        dmatrix = xgboost.DMatrix(feature_matrix)
        similarities = self.xgb_model.predict(dmatrix)
        for (i, j), sim in zip(pair_indices, similarities):
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim  # Make symmetric
        
        return similarity_matrix 
    

    def apply_merges(self, tracklets, track_id_map, cluster_labels, fragments):
        clusters = {}

        for fragment_idx, cluster_id in enumerate(cluster_labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            track_id = track_id_map[fragment_idx]
            clusters[cluster_id].append((track_id, fragments[fragment_idx]))

        merged_tracklets = {}
        next_id = 0

        for cluster_id, members in clusters.items():
            # ========== ADD THIS CHECK ==========
            # Split cluster into non-overlapping groups
            non_overlapping_groups = self.split_overlapping_fragments(members, fragments, track_id_map)
            
            for group in non_overlapping_groups:
                if len(group) == 1:
                    track_id = group[0][0]
                    merged_tracklets[next_id] = tracklets[track_id]
                    merged_tracklets[next_id].track_id = next_id
                else:
                    members_sorted = sorted(group, key=lambda x: x[1]['metadata']['start_frame'])
                    merged = self.merge_tracklet_list(
                        [tracklets[m[0]] for m in members_sorted],
                        new_track_id=next_id
                    )
                    merged_tracklets[next_id] = merged
                
                next_id += 1

        return merged_tracklets
    
    def split_overlapping_fragments(self, members, fragments, track_id_map):
        """
        Split a cluster into groups where no fragments within a group overlap.
        
        Returns list of groups, where each group can be safely merged.
        """
        if len(members) == 1:
            return [members]
        
        # Build overlap graph
        n = len(members)
        overlaps = np.zeros((n, n), dtype=bool)
        
        for i in range(n):
            for j in range(i+1, n):
                frag_A = members[i][1]
                frag_B = members[j][1]
                if self.has_temporal_overlap(frag_A, frag_B):
                    overlaps[i, j] = True
                    overlaps[j, i] = True
        
        # Greedy grouping: assign each fragment to first compatible group
        groups = []
        assigned = [False] * n
        
        for i in range(n):
            if assigned[i]:
                continue
                
            # Start new group with this fragment
            current_group = [members[i]]
            assigned[i] = True
            
            # Try to add other fragments that don't overlap with ANY in current group
            for j in range(i+1, n):
                if assigned[j]:
                    continue
                    
                # Check if j overlaps with any fragment in current group
                can_add = True
                for member_idx in range(n):
                    if assigned[member_idx] and members[member_idx] in current_group:
                        if overlaps[j, member_idx]:
                            can_add = False
                            break
                
                if can_add:
                    current_group.append(members[j])
                    assigned[j] = True
            
            groups.append(current_group)
        
        if len(groups) > 1:
            print(f"  Warning: Cluster had {len(members)} fragments with overlaps - split into {len(groups)} groups")
        
        return groups
    
    def merge_tracklet_list(self, tracklet_list, new_track_id):
        """
        Merge multiple tracklets into one.
        
        Assumes tracklets are sorted by start frame and don't overlap.
        """
        from tracklets.tracklet import Tracklet
        
        # Create new merged tracklet
        merged = Tracklet(track_id=new_track_id)
        merged.parent_id = tracklet_list[0].parent_id
        
        # Concatenate all data
        for tracklet in tracklet_list:
            merged.frames.extend(tracklet.frames)
            merged.bboxes.extend(tracklet.bboxes)
            merged.scores.extend(tracklet.scores)
            merged.embeddings.extend(tracklet.embeddings)
            
            # GT attributes
            for key in tracklet.gt_attributes:
                if key not in merged.gt_attributes:
                    merged.gt_attributes[key] = []
                merged.gt_attributes[key].extend(tracklet.gt_attributes.get(key, []))
            
            # Pred attributes
            for key in tracklet.pred_attributes:
                if key not in merged.pred_attributes:
                    merged.pred_attributes[key] = []
                attr_val = tracklet.pred_attributes.get(key, [])
                if isinstance(attr_val, list):
                    merged.pred_attributes[key].extend(attr_val)
        
        return merged
    

    def compute_pair_features(self, frag_A, frag_B):
        """
        Compute feature vector for a pair of fragments.
        
        Returns flat numpy array matching training feature order.
        """
        A_agg = frag_A['aggregated']
        B_agg = frag_B['aggregated']
        
        features = []
        
        # ===== Fragment A features =====
        features.append(A_agg['jersey_mode'])
        features.append(A_agg['jersey_entropy_mean'])
        features.append(A_agg['jersey_entropy_std'])
        features.append(A_agg['jersey_consistency'])
        features.append(A_agg['team_mode'])
        features.append(A_agg['team_consistency'])
        
        # SigLIP embedding (after PCA: 16 dims)
        features.extend(A_agg['siglip_mean'])
        
        # ReID embedding (after PCA: 8 dims)
        features.extend(A_agg['reid_mean'])
        
        # Spatial features
        features.append(A_agg['start_x'])
        features.append(A_agg['start_y'])
        features.append(A_agg['end_x'])
        features.append(A_agg['end_y'])
        features.append(A_agg['center_x_mean'])
        features.append(A_agg['center_y_mean'])
        features.append(A_agg['bbox_height_mean'])
        
        # Temporal features
        features.append(A_agg['duration_frames'])
        
        # Velocity features
        features.append(A_agg['velocity_x_mean'])
        features.append(A_agg['velocity_y_mean'])
        features.append(A_agg['velocity_mag_std'])
        
        # ===== Fragment B features =====
        features.append(B_agg['jersey_mode'])
        features.append(B_agg['jersey_entropy_mean'])
        features.append(B_agg['jersey_entropy_std'])
        features.append(B_agg['jersey_consistency'])
        features.append(B_agg['team_mode'])
        features.append(B_agg['team_consistency'])
        
        # SigLIP embedding (after PCA: 16 dims)
        features.extend(B_agg['siglip_mean'])
        
        # ReID embedding (after PCA: 8 dims)
        features.extend(B_agg['reid_mean'])
        
        # Spatial features
        features.append(B_agg['start_x'])
        features.append(B_agg['start_y'])
        features.append(B_agg['end_x'])
        features.append(B_agg['end_y'])
        features.append(B_agg['center_x_mean'])
        features.append(B_agg['center_y_mean'])
        features.append(B_agg['bbox_height_mean'])
        
        # Temporal features
        features.append(B_agg['duration_frames'])
        
        # Velocity features
        features.append(B_agg['velocity_x_mean'])
        features.append(B_agg['velocity_y_mean'])
        features.append(B_agg['velocity_mag_std'])
        
        # ===== Pairwise features =====
        temporal_gap = self.compute_temporal_gap(frag_A, frag_B)
        
        endpoint_distance = np.sqrt(
            (A_agg['end_x'] - B_agg['start_x'])**2 +
            (A_agg['end_y'] - B_agg['start_y'])**2
        )
        
        spatial_distance = np.sqrt(
            (A_agg['center_x_mean'] - B_agg['center_x_mean'])**2 +
            (A_agg['center_y_mean'] - B_agg['center_y_mean'])**2
        )
        
        jersey_match = int(A_agg['jersey_mode'] == B_agg['jersey_mode'])
        jersey_entropy_diff = abs(A_agg['jersey_entropy_mean'] - B_agg['jersey_entropy_mean'])
        team_match = int(A_agg['team_mode'] == B_agg['team_mode'])
        
        siglip_cosine_sim = self.cosine_similarity(A_agg['siglip_mean'], B_agg['siglip_mean'])
        siglip_euclidean_dist = np.linalg.norm(
            np.array(A_agg['siglip_mean']) - np.array(B_agg['siglip_mean'])
        )
        
        reid_cosine_sim = self.cosine_similarity(A_agg['reid_mean'], B_agg['reid_mean'])
        reid_euclidean_dist = np.linalg.norm(
            np.array(A_agg['reid_mean']) - np.array(B_agg['reid_mean'])
        )
        
        bbox_height_ratio = A_agg['bbox_height_mean'] / (B_agg['bbox_height_mean'] + 1e-6)
        
        velocity_similarity = np.sqrt(
            (A_agg['velocity_x_mean'] - B_agg['velocity_x_mean'])**2 +
            (A_agg['velocity_y_mean'] - B_agg['velocity_y_mean'])**2
        )
        
        features.extend([
            temporal_gap,
            endpoint_distance,
            spatial_distance,
            jersey_match,
            jersey_entropy_diff,
            team_match,
            siglip_cosine_sim,
            siglip_euclidean_dist,
            reid_cosine_sim,
            reid_euclidean_dist,
            bbox_height_ratio,
            velocity_similarity,
        ])
        
        return np.array(features, dtype=np.float32)
    

    def compute_temporal_gap(self, frag_A, frag_B):
        """Compute frames between two fragments"""
        A_end = frag_A['metadata']['end_frame']
        B_start = frag_B['metadata']['start_frame']
        
        if B_start > A_end:
            return B_start - A_end
        else:
            A_start = frag_A['metadata']['start_frame']
            B_end = frag_B['metadata']['end_frame']
            return A_start - B_end
        

    def cosine_similarity(self, vec_A, vec_B):
        """Compute cosine similarity between two vectors"""
        dot_product = np.dot(vec_A, vec_B)
        norm_A = np.linalg.norm(vec_A)
        norm_B = np.linalg.norm(vec_B)
        
        if norm_A < 1e-6 or norm_B < 1e-6:
            return 0.0
        
        return float(dot_product / (norm_A * norm_B))
        
    def has_temporal_overlap(self, frag_A, frag_B):
        """Check if two fragments exist at the same time"""
        A_start = frag_A['metadata']['start_frame']
        A_end = frag_A['metadata']['end_frame']
        B_start = frag_B['metadata']['start_frame']
        B_end = frag_B['metadata']['end_frame']
        
        # No overlap if one ends before the other starts
        if A_end < B_start or B_end < A_start:
            return False
        return True

