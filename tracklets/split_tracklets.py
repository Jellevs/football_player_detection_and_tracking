from .splitters.unified_splitter import UnifiedSplitter
from .splitters.trajectory_splitter import TrajectorySplitter
from .splitters.bbox_anomaly_splitter import BboxAnomalySplitter
from .splitters.jersey_splitter import JerseySplitter
from .splitters.team_splitter import TeamSplitter
from .splitters import gta_splitter


def split_tracklets(tracklets, splitter_cfg):
    """
    Split tracklets in a single simultaneous pass evaluating jersey and team
    signals together. Either signal can trigger a split independently using
    the same persistence logic as the individual splitters.
    """
    print("\n=== Splitting by unified jersey+team signal ===")
    splitter = UnifiedSplitter(splitter_cfg)
    return split_by_unified(tracklets, splitter)


# def split_tracklets(tracklets, splitter_cfg):
#     """
#     Split tracklets in a single simultaneous pass evaluating jersey and team
#     signals together. Either signal can trigger a split independently using
#     the same persistence logic as the individual splitters.
#     """
#     print("\n=== Splitting by unified jersey+team signal ===")
    
#     jersey_splitted = split_by_jersey(tracklets, JerseySplitter(splitter_cfg))
#     team_splitted = split_by_team(jersey_splitted, TeamSplitter(splitter_cfg))
#     # bbox_splitted =split_by_bbox_velocity(team_splitted, BboxAnomalySplitter(splitter_cfg))
    
#     # traj_splitted = split_by_trajectory(team_splitted, TrajectorySplitter(splitter_cfg))
#     gta_splitted = split_by_gta(team_splitted)


#     return gta_splitted



def split_by_unified(tracklets, splitter):
    """Split tracklets using the unified jersey+team splitter."""
    max_existing_id  = max(tracklets.keys()) if tracklets else 0
    next_available_id = max_existing_id + 1

    new_tracklets = {}
    split_count   = 0

    for track_id, tracklet in tracklets.items():
        fragments = splitter.split_tracklet(tracklet, next_available_id)

        if fragments:
            split_count += 1
            print(f"  Tracklet {track_id} split into {len(fragments)} fragments (unified)")
            for fragment in fragments:
                new_tracklets[fragment.track_id] = fragment
                next_available_id = max(next_available_id, fragment.track_id + 1)
        else:
            new_tracklets[tracklet.track_id] = tracklet

    print(f"Unified splitting: {split_count}/{len(tracklets)} tracklets split")
    return new_tracklets


def split_by_bbox_velocity(tracklets, bbox_splitter):
    """ Split tracklets based on bbox velocity anomalies """
    max_existing_id = max(tracklets.keys()) if tracklets else 0
    next_available_id = max_existing_id + 1
    
    new_tracklets = {}
    split_count = 0
    
    for track_id, tracklet in tracklets.items():
        fragments = bbox_splitter.split_tracklet(tracklet, next_available_id)
        
        if fragments:
            split_count += 1
            print(f"  Tracklet {track_id} split into {len(fragments)} fragments (bbox velocity)")
            
            for fragment in fragments:
                new_tracklets[fragment.track_id] = fragment
                next_available_id = max(next_available_id, fragment.track_id + 1)
        else:
            # No split
            new_tracklets[tracklet.track_id] = tracklet
    
    print(f"Bbox velocity splitting: {split_count}/{len(tracklets)} tracklets split")
    return new_tracklets


def split_by_jersey(tracklets, jersey_splitter):
    """ Split tracklets based on jersey number changes """
    max_existing_id = max(tracklets.keys()) if tracklets else 0
    next_available_id = max_existing_id + 1
    
    new_tracklets = {}
    split_count = 0
    
    for track_id, tracklet in tracklets.items():
        fragments = jersey_splitter.split_tracklet(tracklet, next_available_id)
        
        if fragments:
            split_count += 1
            print(f"  Tracklet {track_id} split into {len(fragments)} fragments (jersey)")
            
            for fragment in fragments:
                new_tracklets[fragment.track_id] = fragment
                next_available_id = max(next_available_id, fragment.track_id + 1)
        else:
            # No split
            new_tracklets[tracklet.track_id] = tracklet
    
    print(f"Jersey splitting: {split_count}/{len(tracklets)} tracklets split")
    return new_tracklets


def split_by_team(tracklets, team_splitter):
    """ Split tracklets based on team changes """
    max_existing_id = max(tracklets.keys()) if tracklets else 0
    next_available_id = max_existing_id + 1

    new_tracklets = {}
    split_count = 0

    for track_id, tracklet in tracklets.items():
        fragments = team_splitter.split_tracklet(tracklet, next_available_id)

        if fragments:
            split_count += 1
            print(f"  Tracklet {track_id} split into {len(fragments)} fragments (team)")

            for fragment in fragments:
                new_tracklets[fragment.track_id] = fragment
                next_available_id = max(next_available_id, fragment.track_id + 1)
        else:
            # No split
            new_tracklets[tracklet.track_id] = tracklet

    print(f"Team splitting: {split_count}/{len(tracklets)} tracklets split")
    return new_tracklets


def split_by_trajectory(tracklets, trajectory_splitter):
    """ Split tracklets based on trajectory proximity and velocity analysis """
    return trajectory_splitter.split_all_tracklets(tracklets)


def split_by_gta(tracklets):
    """ Split tracklets using GTA DBSCAN-based embedding clustering """
    print("\n=== Splitting by GTA (embedding clustering) ===")
    n_before = len(tracklets)
    new_tracklets = gta_splitter.split_tracklets(tracklets)
    print(f"GTA splitting: {n_before} -> {len(new_tracklets)} tracklets ({len(new_tracklets) - n_before} new)")
    return new_tracklets