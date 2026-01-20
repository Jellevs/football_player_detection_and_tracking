from .splitters.jersey_splitter import JerseySplitter
from .splitters.team_splitter import TeamSplitter


def split_tracklets(tracklets, splitter_cfg):
    """
    Split tracklets in two stages:
    1. First by jersey number changes
    2. Then by team changes
    """
    # Stage 1: Split by jersey numbers
    print("\n=== Stage 1: Splitting by jersey numbers ===")
    jersey_splitter = JerseySplitter(splitter_cfg)
    tracklets_after_jersey = split_by_jersey(tracklets, jersey_splitter)
    
    # Stage 2: Split by team changes
    print("\n=== Stage 2: Splitting by team changes ===")
    team_splitter = TeamSplitter(splitter_cfg)
    tracklets_after_team = split_by_team(tracklets_after_jersey, team_splitter)
    
    return tracklets_after_team


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