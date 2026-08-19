from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

from tracklets.tracklet import Tracklet


def split_tracklets(tmp_trklets, eps=0.7, max_k=3, min_samples=10, len_thres=100):
    """
    Splits each tracklet into temporally contiguous fragments using DBSCAN.

    After DBSCAN assigns cluster labels to frames based on ReID embedding
    similarity, the tracklet is split at every point where the label changes
    in temporal order. Each output fragment is a maximal contiguous run of
    frames with the same cluster label.

    Args:
        tmp_trklets (dict): Dictionary of tracklets to be processed.
        eps (float): DBSCAN neighborhood radius (cosine distance).
        min_samples (int): DBSCAN minimum samples for a core point.
        len_thres (int): Minimum tracklet length to attempt splitting.
        max_k (int): Maximum number of clusters to consider.

    Returns:
        dict: New dictionary of tracklets after splitting.
    """
    new_id = max(tmp_trklets.keys()) + 1
    tracklets = defaultdict()

    for tid in tqdm(sorted(list(tmp_trklets.keys())), total=len(tmp_trklets), desc="Splitting tracklets"):
        trklet = tmp_trklets[tid]
        if len(trklet.frames) < len_thres:
            tracklets[tid] = trklet
            continue

        embs = np.stack(trklet.embeddings)
        frames = np.array(trklet.frames)
        bboxes = np.stack(trklet.bboxes)
        scores = np.array(trklet.scores)

        id_switch_detected, clusters = detect_id_switch(embs, eps=eps, min_samples=min_samples, max_clusters=max_k)
        if not id_switch_detected:
            tracklets[tid] = trklet
            continue

        # Sort by frame number (should already be sorted, but be safe)
        sort_idx = np.argsort(frames)
        frames   = frames[sort_idx]
        clusters = clusters[sort_idx]
        bboxes   = bboxes[sort_idx]
        embs     = embs[sort_idx]
        scores   = scores[sort_idx]

        # Split at every temporal label transition
        transitions  = np.where(np.diff(clusters) != 0)[0] + 1
        split_points = [0] + transitions.tolist() + [len(frames)]

        generated_ids = []
        for k in range(len(split_points) - 1):
            start, end = split_points[k], split_points[k + 1]
            assert new_id not in tmp_trklets

            tracklets[new_id] = Tracklet(
                new_id,
                frames[start:end].tolist(),
                scores[start:end].tolist(),
                bboxes[start:end].tolist(),
                embeddings=embs[start:end].tolist(),
            )
            generated_ids.append(new_id)
            new_id += 1

        print(f"-> Split Tracklet {tid} into {len(generated_ids)} fragments: {generated_ids}")

    assert len(tracklets) >= len(tmp_trklets)
    return tracklets


def merge_tracklets(tracklets, seq2Dist, seq_name=None, merge_dist_thres=None):
    max_x_range, max_y_range = get_spatial_constraints(tracklets, 1)

    
    Dist = get_distance_matrix(tracklets)

    idx2tid = {idx: tid for idx, tid in enumerate(tracklets.keys())}
    
    # Hierarchical Clustering
    # While there are still values (exclude diagonal) in distance matrix lower than merging distance threshold
    #   Step 1: find minimal distance for tracklet pair
    #   Step 2: merge tracklet pair
    #   Step 3: update distance matrix
    diagonal_mask = np.eye(Dist.shape[0], dtype=bool)
    non_diagonal_mask = ~diagonal_mask
    while (np.any(Dist[non_diagonal_mask] < merge_dist_thres)):
        # Get the indices of the minimum value considering the mask
        min_index = np.argmin(Dist[non_diagonal_mask])
        min_value = np.min(Dist[non_diagonal_mask])
        # Translate this index to the original array's indices
        masked_indices = np.where(non_diagonal_mask)
        track1_idx, track2_idx = masked_indices[0][min_index], masked_indices[1][min_index]

        assert min_value == Dist[track1_idx, track2_idx] == Dist[track2_idx, track1_idx], "Values should match!"

        track1 = tracklets[idx2tid[track1_idx]]
        track2 = tracklets[idx2tid[track2_idx]]

        inSpatialRange = check_spatial_constraints(track1, track2, max_x_range, max_y_range)
        if inSpatialRange:
            track1.embeddings += track2.embeddings      # Note: currently we merge track 2 to track 1 without creating a new track
            track1.frames += track2.frames
            track1.bboxes += track2.bboxes
            track1.scores += track2.scores
            
            # update tracklets dictionary
            tracklets[idx2tid[track1_idx]] = track1
            tracklets.pop(idx2tid[track2_idx])

            # Remove the merged tracklet (track2) from the distance matrix
            Dist = np.delete(Dist, track2_idx, axis=0)  # Remove row for track2
            Dist = np.delete(Dist, track2_idx, axis=1)  # Remove column for track2
            # update idx2tid
            idx2tid = {idx: tid for idx, tid in enumerate(tracklets.keys())}
            
            # Update distance matrix only for the merged tracklet's row and column
            for idx in range(Dist.shape[0]):
                Dist[track1_idx, idx] = get_distance(idx2tid[track1_idx], idx2tid[idx], tracklets[idx2tid[track1_idx]], tracklets[idx2tid[idx]])
                Dist[idx, track1_idx] = Dist[track1_idx, idx]  # Ensure symmetry
            
            seq2Dist[seq_name] = Dist                   # used to display Dist
            
            # update mask
            diagonal_mask = np.eye(Dist.shape[0], dtype=bool)
            non_diagonal_mask = ~diagonal_mask
        else:
            # change distance between track pair to threshold
            Dist[track1_idx, track2_idx], Dist[track2_idx, track1_idx] = merge_dist_thres, merge_dist_thres
    return tracklets

def detect_id_switch(embs, eps=None, min_samples=None, max_clusters=None):
    """
    Detects identity switches within a tracklet using clustering.

    Args:
        embs (list of numpy arrays): A list where each element is a numpy array representing an embedding.
                                     Each embedding has the same dimensionality.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
        bool: True if an identity switch is detected, otherwise False.
    """
    if len(embs) > 15000:
        embs = embs[1::2]

    embs = np.stack(embs)

    # Standardize the embeddings
    scaler = StandardScaler()
    embs_scaled = scaler.fit_transform(embs)

    # Apply DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embs_scaled)
    labels = db.labels_

    # Count the number of clusters (excluding noise)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]

    if -1 in labels and len(unique_labels) > 1:
        # Find the cluster centers
        cluster_centers = np.array([embs_scaled[labels == label].mean(axis=0) for label in unique_labels])
        
        # Assign noise points to the nearest cluster
        noise_indices = np.where(labels == -1)[0]
        for idx in noise_indices:
            distances = cdist([embs_scaled[idx]], cluster_centers, metric='cosine')
            nearest_cluster = np.argmin(distances)
            labels[idx] = list(unique_labels)[nearest_cluster]
    
    n_clusters = len(unique_labels)

    if max_clusters and n_clusters > max_clusters:
        # Merge clusters to ensure the number of clusters does not exceed max_clusters
        while n_clusters > max_clusters:
            cluster_centers = np.array([embs_scaled[labels == label].mean(axis=0) for label in unique_labels])
            distance_matrix = cdist(cluster_centers, cluster_centers, metric='cosine')
            np.fill_diagonal(distance_matrix, np.inf)  # Ignore self-distances
            
            # Find the closest pair of clusters
            min_dist_idx = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
            cluster_to_merge_1, cluster_to_merge_2 = unique_labels[min_dist_idx[0]], unique_labels[min_dist_idx[1]]

            # Merge the clusters
            labels[labels == cluster_to_merge_2] = cluster_to_merge_1
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels != -1]
            n_clusters = len(unique_labels)

    return n_clusters > 1, labels


def get_distance(track1_id, track2_id, track1, track2):
    """
    Calculates the cosine distance between two tracks using CPU for memory efficiency.
    """
    assert track1_id == track1.track_id and track2_id == track2.track_id   # debug line
    
    # 1. Check for temporal overlap (Hard Constraint)
    doesOverlap = False
    if (track1_id != track2_id):
        # Use times or frames depending on your specific class attribute
        doesOverlap = set(track1.frames) & set(track2.frames)
    
    if doesOverlap:
        return 1.0 # Maximum distance for overlapping tracks
    
    # 2. Perform Cosine Similarity on CPU
    # We explicitly use 'cpu' to avoid CUDA memory allocation failures
    device = torch.device("cpu")
    
    # Use np.stack to ensure features are a proper numeric array
    track1_features_tensor = torch.tensor(np.stack(track1.embeddings), dtype=torch.float32).to(device)
    track2_features_tensor = torch.tensor(np.stack(track2.embeddings), dtype=torch.float32).to(device)
    
    count1 = len(track1_features_tensor)
    count2 = len(track2_features_tensor)

    # Cosine Similarity Formula: (A . B) / (||A|| * ||B||)
    cos_sim_Numerator = torch.matmul(track1_features_tensor, track2_features_tensor.T)
    
    # Calculate norms for denominator
    track1_features_dist = torch.norm(track1_features_tensor, p=2, dim=1, keepdim=True)
    track2_features_dist = torch.norm(track2_features_tensor, p=2, dim=1, keepdim=True)
    cos_sim_Denominator = torch.matmul(track1_features_dist, track2_features_dist.T)
    
    # Add epsilon to avoid division by zero if a feature vector is all zeros
    cos_Dist = 1 - (cos_sim_Numerator / (cos_sim_Denominator + 1e-6))
    
    # Return average distance between all feature pairs
    total_cos_Dist = cos_Dist.sum()
    result = total_cos_Dist / (count1 * count2)
    
    return result.item() # Return as standard float


def check_spatial_constraints(trk_1, trk_2, max_x_range, max_y_range):
    """
    Checks if two tracklets meet spatial constraints for potential merging.

    Args:
        trk_1 (Tracklet): The first tracklet object containing times and bounding boxes.
        trk_2 (Tracklet): The second tracklet object containing times and bounding boxes, to be evaluated
                        against trk_1 for merging possibility.
        max_x_range (float): The maximum allowed distance in the x-coordinate between the end of trk_1 and
                             the start of trk_2 for them to be considered for merging.
        max_y_range (float): The maximum allowed distance in the y-coordinate under the same conditions as
                             the x-coordinate.

    Returns:
        bool: True if the spatial constraints are met (the tracklets are close enough to consider merging),
              False otherwise.
    """
    inSpatialRange = True
    seg_1 = find_consecutive_segments(trk_1.frames)
    seg_2 = find_consecutive_segments(trk_2.frames)
    '''Debug
    '''
    
    subtracks = query_subtracks(seg_1, seg_2, trk_1, trk_2)
    subtrack_1st = subtracks.pop(0)
    while subtracks:
        subtrack_2nd = subtracks.pop(0)
        if subtrack_1st.parent_id == subtrack_2nd.parent_id:
            subtrack_1st = subtrack_2nd
            continue
        x_1, y_1, w_1, h_1 = subtrack_1st.bboxes[-1][0 : 4]
        x_2, y_2, w_2, h_2 = subtrack_2nd.bboxes[0][0 : 4]
        x_1 += w_1 / 2
        y_1 += h_1 / 2
        x_2 += w_2 / 2
        y_2 += h_2 / 2
        dx = abs(x_1 - x_2)
        dy = abs(y_1 - y_2)
        
        # check the distance between exit location of track_1 and enter location of track_2
        if dx > max_x_range or dy > max_y_range:
            inSpatialRange = False
            break
        else:
            subtrack_1st = subtrack_2nd
    return inSpatialRange


def find_consecutive_segments(track_times):
    """
    Identifies and returns the start and end indices of consecutive segments in a list of times.

    Args:
        track_times (list): A list of frame times (integers) representing when a tracklet was detected.

    Returns:
        list of tuples: Each tuple contains two integers (start_index, end_index) representing the start and end of a consecutive segment.
    """
    segments = []
    start_index = 0
    end_index = 0
    for i in range(1, len(track_times)):
        if track_times[i] == track_times[end_index] + 1:
            end_index = i
        else:
            segments.append((start_index, end_index))
            start_index = i
            end_index = i
    segments.append((start_index, end_index))
    return segments


def query_subtracks(seg1, seg2, track1, track2):
    """
    Processes and pairs up segments from two different tracks to form valid subtracks based on their temporal alignment.

    Args:
        seg1 (list of tuples): List of segments from the first track where each segment is a tuple of start and end indices.
        seg2 (list of tuples): List of segments from the second track similar to seg1.
        track1 (Tracklet): First track object containing times and bounding boxes.
        track2 (Tracklet): Second track object similar to track1.

    Returns:
        list: Returns a list of subtracks which are either segments of track1 or track2 sorted by time.
    """
    subtracks = []  # List to store valid subtracks
    while seg1 and seg2:  # Continue until seg1 or seg1 is empty
        s1_start, s1_end = seg1[0]  # Get the start and end indices of the first segment in seg1
        s2_start, s2_end = seg2[0]  # Get the start and end indices of the first segment in seg2
        '''Optionally eliminate false positive subtracks
        if (s1_end - s1_start + 1) < 30:
            seg1.pop(0)  # Remove the first element from seg1
            continue
        if (s2_end - s2_start + 1) < 30:
            seg2.pop(0)  # Remove the first element from seg2
            continue
        '''

        subtrack_1 = track1.extract(s1_start, s1_end)
        subtrack_2 = track2.extract(s2_start, s2_end)

        s1_startFrame = track1.frames[s1_start]  # Get the starting frame of subtrack 1
        s2_startFrame = track2.frames[s2_start]  # Get the starting frame of subtrack 2


        if s1_startFrame < s2_startFrame:  # Compare the starting frames of the two subtracks
            assert track1.frames[s1_end] <= s2_startFrame
            subtracks.append(subtrack_1)
            subtracks.append(subtrack_2)
        else:
            assert s1_startFrame >= track2.frames[s2_end]
            subtracks.append(subtrack_2)
            subtracks.append(subtrack_1)
        seg1.pop(0)
        seg2.pop(0)
    
    seg_remain = seg1 if seg1 else seg2
    track_remain = track1 if seg1 else track2
    while seg_remain:
        s_start, s_end = seg_remain[0]
        if(s_end - s_start) < 30:
            seg_remain.pop(0)
            continue
        subtracks.append(track_remain.extract(s_start, s_end))
        seg_remain.pop(0)
    
    return subtracks  # Return the list of valid subtracks sorted ascending temporally


def get_distance_matrix(tid2track):
    """
    Constructs and returns a distance matrix between all tracklets based on overlapping times and feature similarities.

    Args:
        tid2track (dict): Dictionary mapping track IDs to their respective track objects.

    Returns:
        ndarray: A square matrix where each element (i, j) represents the calculated distance between track i and track j.
    """
    Dist = np.zeros((len(tid2track), len(tid2track)))

    for i, (track1_id, track1) in enumerate(tid2track.items()):
        assert len(track1.frames) == len(track1.bboxes)
        for j, (track2_id, track2) in enumerate(tid2track.items()):
            if j < i:
                Dist[i][j] = Dist[j][i]
            else:
                Dist[i][j] = get_distance(track1_id, track2_id, track1, track2)
    return Dist


def get_spatial_constraints(tid2track, factor):
    """
    Calculates and returns the maximal spatial constraints for bounding boxes across all tracks.

    Args:
        tid2track (dict): Dictionary mapping track IDs to their respective track objects.
        factor (float): Factor by which to scale the calculated x and y ranges.

    Returns:
        tuple: Maximal x and y range scaled by the given factor.
    """

    min_x = float('inf')
    max_x = -float('inf')
    min_y = float('inf')
    max_y = -float('inf')

    for track in tid2track.values():
        for bbox in track.bboxes:
            assert len(bbox) == 4
            x, y, w, h = bbox[0:4]  # x, y is coordinate of top-left point of bounding box
            x += w / 2  # get center point
            y += h / 2  # get center point
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

    x_range = abs(max_x - min_x) * factor
    y_range = abs(max_y - min_y) * factor

    return x_range, y_range