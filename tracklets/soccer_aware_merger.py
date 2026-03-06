import numpy as np
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
        max_x, max_y = self.get_spatial_constraints(tracklets, self.spatial_factor)

        DEBUG_IDS = {2, 22}  # TODO: remove after debugging

        for i in range(n):
            for j in range(i + 1, n):
                t1, t2 = tracklets[tracklet_ids[i]], tracklets[tracklet_ids[j]]
                debug = {tracklet_ids[i], tracklet_ids[j]} == DEBUG_IDS

                # HARD CONSTRAINT: Temporal Overlap.
                # Use 2.0 (> any valid cosine distance). The true hard
                # enforcement is in apply_cluster_merges via the safeguard.
                if set(t1.frames) & set(t2.frames):
                    if debug: print(f"[DEBUG {tracklet_ids[i]}&{tracklet_ids[j]}] BLOCKED: temporal overlap")
                    dist_matrix[i, j] = dist_matrix[j, i] = 2.0
                    continue

                # HARD CONSTRAINT: Spatial plausibility
                if not self.check_spatial_plausibility(t1, t2, max_x, max_y):
                    if debug: print(f"[DEBUG {tracklet_ids[i]}&{tracklet_ids[j]}] BLOCKED: spatial (max_x={max_x:.1f}, max_y={max_y:.1f})")
                    dist_matrix[i, j] = dist_matrix[j, i] = 2.0
                    continue

                # CALC BASE DISTANCE (Cosine similarity of ReID)
                dist = self.get_soccer_distance(t1, t2)

                if debug:
                    j1, e1 = self.get_jersey_stats(t1)
                    j2, e2 = self.get_jersey_stats(t2)
                    team1, cons1 = self.get_team_stats(t1)
                    team2, cons2 = self.get_team_stats(t2)
                    print(f"[DEBUG {tracklet_ids[i]}&{tracklet_ids[j]}] dist={dist:.4f} | jersey=({j1},e={e1:.4f}) vs ({j2},e={e2:.4f}) | team=({team1},c={cons1:.2f}) vs ({team2},c={cons2:.2f})")

                dist_matrix[i, j] = dist_matrix[j, i] = dist

        return dist_matrix

    @staticmethod
    def get_spatial_constraints(tracklets, factor):
        """Calculate max spatial range across all tracklets."""
        min_x, max_x_val = float('inf'), -float('inf')
        min_y, max_y_val = float('inf'), -float('inf')
        for t in tracklets.values():
            for bbox in t.bboxes:
                cx = bbox[0] + (bbox[2] - bbox[0]) / 2
                cy = bbox[1] + (bbox[3] - bbox[1]) / 2
                min_x, max_x_val = min(min_x, cx), max(max_x_val, cx)
                min_y, max_y_val = min(min_y, cy), max(max_y_val, cy)
        return abs(max_x_val - min_x) * factor, abs(max_y_val - min_y) * factor

    @staticmethod
    def check_spatial_plausibility(t1, t2, max_x, max_y):
        """Check if the exit/entry positions of two tracklets are spatially plausible."""
        # Determine which tracklet ends first
        if t1.frames[-1] <= t2.frames[0]:
            exit_bbox, entry_bbox = t1.bboxes[-1], t2.bboxes[0]
        elif t2.frames[-1] <= t1.frames[0]:
            exit_bbox, entry_bbox = t2.bboxes[-1], t1.bboxes[0]
        else:
            return True  # Interleaved — can't check simply, allow it

        cx1 = exit_bbox[0] + (exit_bbox[2] - exit_bbox[0]) / 2
        cy1 = exit_bbox[1] + (exit_bbox[3] - exit_bbox[1]) / 2
        cx2 = entry_bbox[0] + (entry_bbox[2] - entry_bbox[0]) / 2
        cy2 = entry_bbox[1] + (entry_bbox[3] - entry_bbox[1]) / 2

        return abs(cx1 - cx2) <= max_x and abs(cy1 - cy2) <= max_y

    def get_soccer_distance(self, t1, t2):
        """
        Calculates distance with soft team constraints to handle occlusions.
        """
        # A. ReID Base (All-pairs Average Cosine Distance, matching GTA)
        feat1 = np.stack(t1.embeddings).astype(np.float32)
        feat2 = np.stack(t2.embeddings).astype(np.float32)
        norms1 = np.linalg.norm(feat1, axis=1, keepdims=True) + 1e-6
        norms2 = np.linalg.norm(feat2, axis=1, keepdims=True) + 1e-6
        cos_sim_matrix = (feat1 @ feat2.T) / (norms1 @ norms2.T)
        dist = 1.0 - cos_sim_matrix.mean()

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

        # C. Jersey + Team identity check
        j1, e1 = self.get_jersey_stats(t1)
        j2, e2 = self.get_jersey_stats(t2)

        both_confident = j1 is not None and j2 is not None and e1 < 0.15 and e2 < 0.15

        if both_confident and j1 != j2:
            # Different confident jersey numbers = definitively different players.
            # Hard block to prevent average-linkage chaining across jersey boundaries.
            return 1.0

        jerseys_match = both_confident and j1 == j2

        if jerseys_match:
            if team1 is not None and team2 is not None and team1 == team2:
                # Same jersey + same team = unique player identity in soccer.
                # Short-circuit: guarantee a merge regardless of ReID distance.
                return 0.05
            else:
                # Same jersey but team unknown/different — still a strong signal,
                # but not conclusive (could be mirrored jersey numbers across teams).
                dist -= 0.3

        return np.clip(dist, 0, 1)

    def get_team_stats(self, tracklet):
        """Helper to extract mode and consistency from pred_attributes['teams']"""
        teams = [t for t in tracklet.pred_attributes.get('teams', []) if not np.isnan(t)]
        if not teams: return None, 0.0
        mode = max(set(teams), key=teams.count)
        consistency = teams.count(mode) / len(teams)  # use valid count, not total frames
        return mode, consistency

    def get_jersey_stats(self, tracklet):
        """Helper to extract jersey info and mean entropy"""
        all_jerseys = tracklet.pred_attributes.get('jerseys', [])
        all_entropies = tracklet.pred_attributes.get('jersey_entropies', [1.0])
        valid = [(j, all_entropies[i]) for i, j in enumerate(all_jerseys) if not np.isnan(j)]
        if not valid: return None, 1.0
        jerseys, entropies = zip(*valid)
        mode = max(set(jerseys), key=jerseys.count)
        # Use entropy only from frames that predicted the mode — confused frames
        # that predicted a different number shouldn't penalize the confident ones.
        mode_entropies = [e for j, e in zip(jerseys, entropies) if j == mode]
        return mode, np.mean(mode_entropies)

    def apply_cluster_merges(self, tracklets, tracklet_ids, cluster_labels):
        """Consolidates clusters into final tracklets"""
        clusters = {}
        for idx, cluster_id in enumerate(cluster_labels):
            clusters.setdefault(cluster_id, []).append(tracklet_ids[idx])

        DEBUG_IDS = {2, 22}  # TODO: remove after debugging
        for cluster_id, member_ids in clusters.items():
            if DEBUG_IDS.issubset(set(member_ids)):
                print(f"[DEBUG apply] tracklets {DEBUG_IDS} are in cluster {cluster_id} with members {sorted(member_ids)}")
                break
        else:
            print(f"[DEBUG apply] tracklets {DEBUG_IDS} are in DIFFERENT clusters")
            for cluster_id, member_ids in clusters.items():
                if DEBUG_IDS & set(member_ids):
                    print(f"  cluster {cluster_id}: {sorted(member_ids)}")

        merged_results = {}
        # Overflow IDs start after all scipy cluster IDs (1..K)
        next_overflow_id = max(clusters.keys()) + 1

        for cluster_id, member_ids in clusters.items():
            # Sort by start frame to maintain chronological order
            member_ids.sort(key=lambda x: tracklets[x].frames[0])

            base_track = tracklets[member_ids[0]]
            current_frames = set(base_track.frames)

            for next_id in member_ids[1:]:
                other = tracklets[next_id]
                other_frames = set(other.frames)

                # Hard safeguard: never merge tracklets with overlapping frames.
                # Average-linkage chains can cause overlapping pairs to land in
                # the same cluster despite the 2.0 distance penalty. When that
                # happens, keep the overlapping tracklet as a standalone entry
                # instead of silently dropping it.
                if current_frames & other_frames:
                    if {member_ids[0], next_id} & DEBUG_IDS:
                        print(f"[DEBUG apply] BLOCKED merge of {next_id} into base {member_ids[0]} (frame overlap). current_frames has {len(current_frames)} frames, overlap={len(current_frames & other_frames)}")
                    merged_results[next_overflow_id] = other
                    next_overflow_id += 1
                    continue

                if {member_ids[0], next_id} & DEBUG_IDS:
                    print(f"[DEBUG apply] MERGED {next_id} into base {member_ids[0]}")

                base_track.frames.extend(other.frames)
                base_track.bboxes.extend(other.bboxes)
                base_track.scores.extend(other.scores)
                base_track.embeddings.extend(other.embeddings)

                for key in base_track.pred_attributes:
                    base_track.pred_attributes[key].extend(other.pred_attributes.get(key, []))

                current_frames |= other_frames

            merged_results[cluster_id] = base_track

        return merged_results
    

    