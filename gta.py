"""
GTA Integration with Property Compatibility Fix

The issue: GTA's merge function does:
    track1.features += track2.features
    
With properties, this doesn't work as expected because:
1. Gets track1.features (returns track1.embeddings)
2. Concatenates with track2.features
3. Tries to set the result back via the setter
4. But the reference is lost

Solution: Patch the merge to explicitly use the setter.
"""

import sys
from pathlib import Path

# Import your modified GTA functions
try:
    from gta_link.refine_tracklets import (
        split_tracklets as gta_split_original,
        get_distance_matrix,
        get_spatial_constraints,
    )
    GTA_AVAILABLE = True
except ImportError:
    print("Warning: GTA functions not found")
    GTA_AVAILABLE = False


def merge_tracklets_fixed(
    tracklets,
    seq2Dist,
    dist_matrix,
    seq_name,
    max_x_range,
    max_y_range,
    merge_dist_thres
):
    """
    Fixed version of GTA merge that works with property-based tracklets.
    
    The key fix: Instead of track1.features += track2.features,
    we do: track1.features = track1.features + track2.features
    
    This explicitly uses the setter.
    """
    import numpy as np
    from copy import deepcopy
    
    # Build distance matrix if needed
    if dist_matrix is None:
        dist_matrix = get_distance_matrix(tracklets)
    
    # Create id-to-index mapping for distance matrix
    id_list = list(tracklets.keys())
    id_to_idx = {track_id: idx for idx, track_id in enumerate(id_list)}
    
    # Iterative merging
    merge_count = 0
    max_merges = len(tracklets) * 2  # Safety limit: shouldn't need more merges than 2x tracklets
    failed_pairs = set()  # Track pairs that can't be merged (temporal overlap, etc.)
    
    while merge_count < max_merges:
        # Find closest pair
        min_dist = float('inf')
        merge_pair = None
        
        ids = list(tracklets.keys())
        for i, id1 in enumerate(ids):
            for id2 in ids[i+1:]:
                # Skip if we already tried and failed this pair
                if (id1, id2) in failed_pairs or (id2, id1) in failed_pairs:
                    continue
                
                track1 = tracklets[id1]
                track2 = tracklets[id2]
                
                # Check temporal overlap
                if set(track1.times) & set(track2.times):
                    # Mark as failed and skip
                    failed_pairs.add((id1, id2))
                    continue
                
                # Check spatial constraints
                track1_x = [(bbox[0] + bbox[2])/2 for bbox in track1.bboxes]
                track1_y = [(bbox[1] + bbox[3])/2 for bbox in track1.bboxes]
                track2_x = [(bbox[0] + bbox[2])/2 for bbox in track2.bboxes]
                track2_y = [(bbox[1] + bbox[3])/2 for bbox in track2.bboxes]
                
                x_overlap = not (max(track1_x) + max_x_range < min(track2_x) or 
                                max(track2_x) + max_x_range < min(track1_x))
                y_overlap = not (max(track1_y) + max_y_range < min(track2_y) or 
                                max(track2_y) + max_y_range < min(track1_y))
                
                if not (x_overlap and y_overlap):
                    continue
                
                # Get distance from matrix
                if id1 in id_to_idx and id2 in id_to_idx:
                    idx1 = id_to_idx[id1]
                    idx2 = id_to_idx[id2]
                    
                    # Handle both dict and numpy array
                    if isinstance(dist_matrix, np.ndarray):
                        dist = dist_matrix[idx1, idx2]
                    else:
                        dist = dist_matrix.get((id1, id2), dist_matrix.get((id2, id1), float('inf')))
                else:
                    # New tracklet not in original matrix, recompute
                    if not track1.features or not track2.features:
                        dist = float('inf')
                    else:
                        embs1 = np.array(track1.features)
                        embs2 = np.array(track2.features)
                        
                        # Normalize
                        embs1_norm = embs1 / (np.linalg.norm(embs1, axis=1, keepdims=True) + 1e-6)
                        embs2_norm = embs2 / (np.linalg.norm(embs2, axis=1, keepdims=True) + 1e-6)
                        
                        # Cosine distance
                        cos_sim = np.dot(embs1_norm, embs2_norm.T)
                        cos_dist = 1 - cos_sim
                        dist = np.mean(cos_dist)
                
                if dist < min_dist and dist < merge_dist_thres:
                    min_dist = dist
                    merge_pair = (id1, id2)
        
        # If no valid merge found, stop
        if merge_pair is None:
            break
        
        # Merge the pair
        id1, id2 = merge_pair
        track1 = tracklets[id1]
        track2 = tracklets[id2]
        
        merge_count += 1
        print(f"  Merge {merge_count}: {id1} + {id2} (dist={min_dist:.3f})")
        
        # Determine order and which ID to keep
        # IMPORTANT: Keep track of original IDs for dict updates
        keep_id = id1
        delete_id = id2
        
        if track1.times[-1] < track2.times[0]:
            # track1 comes before track2 - order is correct
            first_track = track1
            second_track = track2
        elif track2.times[-1] < track1.times[0]:
            # track2 comes before track1 - need to reverse
            first_track = track2
            second_track = track1
        else:
            # This should never happen since we filtered overlaps above
            print(f"    ERROR: Unexpected temporal overlap!")
            failed_pairs.add((id1, id2))
            continue
        
        # THE KEY FIX: Use direct attribute access instead of properties
        # Merge into the first track (chronologically)
        first_track.frames = first_track.frames + second_track.frames
        first_track.bboxes = first_track.bboxes + second_track.bboxes
        first_track.scores = first_track.scores + second_track.scores
        
        # Merge embeddings if both have them
        if first_track.embeddings and second_track.embeddings:
            first_track.embeddings = first_track.embeddings + second_track.embeddings
        
        # Update tracklets dict: keep keep_id, delete delete_id
        tracklets[keep_id] = first_track
        del tracklets[delete_id]
        
        # Update id_to_idx mapping
        if delete_id in id_to_idx:
            del id_to_idx[delete_id]
        
        # Clean up failed_pairs: remove any pairs involving the deleted tracklet
        failed_pairs = {(a, b) for a, b in failed_pairs if a != delete_id and b != delete_id}
    
    if merge_count > 0:
        print(f"✓ Merge complete: {merge_count} merges performed")
        if merge_count >= max_merges:
            print(f"  ⚠️  Warning: Hit safety limit of {max_merges} merges")
            print(f"  This may indicate an infinite loop bug")
        if len(failed_pairs) > 0:
            print(f"  Skipped {len(failed_pairs)} pairs due to temporal overlap")
    else:
        print(f"✓ Merge complete: No merges performed (all tracklets distinct)")
    
    print(f"  Final: {len(tracklets)} tracklets")
    
    return tracklets


def split_tracklets(tracklets, eps=0.7, max_k=3, min_samples=10, len_thres=100):
    """
    Wrapper for GTA split that works with properties.
    Split doesn't modify features with +=, so it should work fine.
    """
    if not GTA_AVAILABLE:
        raise ImportError("GTA functions not available")
    
    return gta_split_original(tracklets, eps, max_k, min_samples, len_thres)


def merge_tracklets(
    tracklets,
    seq2Dist,
    dist_matrix,
    seq_name,
    max_x_range,
    max_y_range,
    merge_dist_thres
):
    """Use the fixed merge function"""
    return merge_tracklets_fixed(
        tracklets,
        seq2Dist,
        dist_matrix,
        seq_name,
        max_x_range,
        max_y_range,
        merge_dist_thres
    )


# Export the fixed functions
__all__ = [
    'split_tracklets',
    'merge_tracklets',
    'get_distance_matrix',
    'get_spatial_constraints',
]


if __name__ == "__main__":
    print("GTA Integration with Property Fix")
    print("="*60)
    print("\nUsage:")
    print("  from gta_fixed import split_tracklets, merge_tracklets")
    print("  from gta_fixed import get_distance_matrix, get_spatial_constraints")
    print("\nThis version works with Tracklet properties!")