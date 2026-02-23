# from tqdm import tqdm
# import pickle
# import numpy as np

# from .jersey_number.jersey_number_predictor_parseq import JerseyNumberPredictorParseq
# from .teamclassifier import TeamClassifier
# from .referee_classifier import RefereeClassifier


# def predict_attributes(images, tracklets, paths, jersey_cfg, device):
#     """ Predict jersey numbers and team id for all tracklets """
#     cache_path = paths.set_cache_path("attributes", paths.sequence)

#     if cache_path and cache_path.exists():
#         print(f"Loading cached attributes from {cache_path}")
#         with open(cache_path, 'rb') as f:
#             return pickle.load(f)

#     print(f"cahce not found for {cache_path}")
#     jersey_predictor = JerseyNumberPredictorParseq(
#         paths=paths,
#         jersey_cfg=jersey_cfg,
#         device=device
#     )

#     team_classifier = TeamClassifier(
#         device=device,
#         batch_size=32,
#         paths=paths
#     )

#     referee_classifier = RefereeClassifier(
#         model_path=paths.referee_classifier_path,
#         device=device,
#         threshold=jersey_cfg.referee_threshold
#     )

#     # Phase 1: Extract crops and predict jersey numbers
#     print("Phase 1: Extracting crops and predicting jersey numbers...")
    
#     tracklet_torso_crops = {}
#     tracklet_crop_indices = {}
#     tracklet_crop_roles = {}
    
#     all_torso_crops = []
#     all_crop_info = []
#     player_mask = []
#     gt_teams = []
    
#     for track_id, tracklet in tqdm(tracklets.items(), desc="Processing tracklets"):
#         # Get unfiltered torso crops for teams + filtered jersey predictions
#         torso_crops, indices, jerseys, jersey_confs_mean, jersey_entropies = jersey_predictor.predict(images, tracklet)

#         # Store jersey predictions
#         tracklet.pred_attributes['jerseys'] = jerseys.tolist()
#         tracklet.pred_attributes['jersey_confs_mean'] = jersey_confs_mean.tolist()
#         tracklet.pred_attributes['jersey_entropies'] = jersey_entropies.tolist()

#         # Store torso crops for team classification
#         tracklet_torso_crops[track_id] = torso_crops
#         tracklet_crop_indices[track_id] = indices
        
#         roles = tracklet.gt_attributes.get('roles', [])
#         teams = tracklet.gt_attributes.get('teams', [])
        
#         # Build crop info for team classification
#         crop_roles = []
#         for i, crop in enumerate(torso_crops):
#             tracklet_idx = indices[i]
            
#             # Get role for this crop
#             role = roles[tracklet_idx] if tracklet_idx < len(roles) else "unknown"
#             crop_roles.append(role)
            
#             all_torso_crops.append(crop)
#             all_crop_info.append((track_id, i))
            
#             # Only players should be used for clustering (not GK/referee)
#             is_player = (role == "player")
#             player_mask.append(is_player)
            
#             # Store GT team for evaluation
#             gt_team = teams[tracklet_idx] if tracklet_idx < len(teams) else "unknown"
#             gt_teams.append(gt_team)
        
#         tracklet_crop_roles[track_id] = crop_roles


#     # Phase 2: Team classification using torso crops
#     player_mask = np.array(player_mask)
#     team_predictions, siglip_embeddings = team_classifier.fit_predict_all(all_torso_crops, player_mask)
    


#     # Phase 3: Map predictions back to tracklets
#     # Only assign teams to PLAYERS (not referees or goalkeepers)
#     print("Phase 3: Mapping predictions to tracklets...")
    
#     prediction_lookup = {}
#     for global_idx, (track_id, local_idx) in enumerate(all_crop_info):
#         prediction_lookup[(track_id, local_idx)] = team_predictions[global_idx]
    
#     for track_id, tracklet in tracklets.items():

#         indices = tracklet_crop_indices[track_id]
#         crop_roles = tracklet_crop_roles[track_id]
#         num_frames = len(tracklet.frames)

#         teams_full = [np.nan] * num_frames
#         siglip_full = [np.zeros(768)] * num_frames
        
#         n_crops = len(tracklet_torso_crops[track_id])
#         for local_idx in range(n_crops):
#             tracklet_idx = indices[local_idx]

#             global_idx = all_crop_info.index((track_id, local_idx))
#             siglip_embedding = siglip_embeddings[global_idx]
#             siglip_full[tracklet_idx] = siglip_embedding

#             # Only assign team to players, skip referees and goalkeepers
#             role = crop_roles[local_idx]
#             if role != "player":
#                 continue  # Leave as np.nan for non-players
            
#             pred = prediction_lookup.get((track_id, local_idx))
#             if pred is not None:
#                 teams_full[tracklet_idx] = pred
        
#         tracklet.pred_attributes['teams'] = teams_full
#         tracklet.pred_attributes['siglip_embeddings'] = siglip_full
        
#     print("Attribute prediction finished!")


#     # Save cache
#     if cache_path:
#         cache_path.parent.mkdir(parents=True, exist_ok=True)
#         with open(cache_path, 'wb') as f:
#             pickle.dump(tracklets, f)
#         print(f"Cached attributes to {cache_path}")

#     return tracklets


from tqdm import tqdm
import pickle
import numpy as np

from .jersey_number.jersey_number_predictor_parseq import JerseyNumberPredictorParseq
from .teamclassifier import TeamClassifier
from .referee_classifier import RefereeClassifier


def predict_attributes(images, tracklets, paths, jersey_cfg, device):
    """ Predict jersey numbers and team id for all tracklets """
    cache_path = paths.set_cache_path("attributes", paths.sequence)

    if cache_path and cache_path.exists():
        print(f"Loading cached attributes from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f"Cache not found for {cache_path}")

    jersey_predictor = JerseyNumberPredictorParseq(
        paths=paths,
        jersey_cfg=jersey_cfg,
        device=device
    )

    team_classifier = TeamClassifier(
        device=device,
        batch_size=32,
        paths=paths
    )

    # Load referee classifier if configured
    referee_classifier = RefereeClassifier(
        model_path=paths.referee_classifier_path,
        device=device,
        threshold=jersey_cfg.referee_threshold
    )
 

    # Phase 1: Extract crops and predict jersey numbers
    print("Phase 1: Extracting crops and predicting jersey numbers...")

    tracklet_torso_crops = {}
    tracklet_crop_indices = {}
    tracklet_is_referee = {}   # tracklet-level referee flag (one bool per tracklet)

    all_torso_crops = []
    all_crop_info = []
    player_mask = []

    for track_id, tracklet in tqdm(tracklets.items(), desc="Processing tracklets"):
        # Get unfiltered torso crops for team classification + filtered jersey predictions
        torso_crops, indices, jerseys, jersey_confs_mean, jersey_entropies = jersey_predictor.predict(images, tracklet)

        # Store jersey predictions
        tracklet.pred_attributes['jerseys'] = jerseys.tolist()
        tracklet.pred_attributes['jersey_confs_mean'] = jersey_confs_mean.tolist()
        tracklet.pred_attributes['jersey_entropies'] = jersey_entropies.tolist()

        # Store torso crops for team classification
        tracklet_torso_crops[track_id] = torso_crops
        tracklet_crop_indices[track_id] = indices

        # ---- Predict referee role (tracklet-level majority vote) ----
        if referee_classifier is not None and len(torso_crops) > 0:
            is_ref, ref_confidence = referee_classifier.predict_tracklet(torso_crops)
            print(is_ref, ref_confidence)
        else:
            # Fallback: treat everything as a player (no classifier available)
            is_ref = False
            ref_confidence = 0.0

        tracklet_is_referee[track_id] = is_ref
        tracklet.pred_attributes['is_referee'] = is_ref
        tracklet.pred_attributes['referee_confidence'] = ref_confidence

        # Build crop info for team classification
        # All crops from this tracklet share the same role prediction
        for i, crop in enumerate(torso_crops):
            all_torso_crops.append(crop)
            all_crop_info.append((track_id, i))

            # Only players (not referees) should be used to fit the KMeans clustering
            player_mask.append(not is_ref)


    # Phase 2: Team classification using torso crops
    print("Phase 2: Team classification...")
    player_mask = np.array(player_mask)
    team_predictions, siglip_embeddings = team_classifier.fit_predict_all(all_torso_crops, player_mask)


    # Phase 3: Map predictions back to tracklets
    # Only assign teams to non-referees
    print("Phase 3: Mapping predictions to tracklets...")

    prediction_lookup = {}
    for global_idx, (track_id, local_idx) in enumerate(all_crop_info):
        prediction_lookup[(track_id, local_idx)] = team_predictions[global_idx]

    for track_id, tracklet in tracklets.items():

        indices = tracklet_crop_indices[track_id]
        is_ref = tracklet_is_referee[track_id]
        num_frames = len(tracklet.frames)

        teams_full = [np.nan] * num_frames
        siglip_full = [np.zeros(768)] * num_frames

        n_crops = len(tracklet_torso_crops[track_id])
        for local_idx in range(n_crops):
            tracklet_idx = indices[local_idx]

            global_idx = all_crop_info.index((track_id, local_idx))
            siglip_embedding = siglip_embeddings[global_idx]
            siglip_full[tracklet_idx] = siglip_embedding

            # Only assign team prediction to non-referees
            if is_ref:
                continue  # Leave as np.nan for referees

            pred = prediction_lookup.get((track_id, local_idx))
            if pred is not None:
                teams_full[tracklet_idx] = pred

        tracklet.pred_attributes['teams'] = teams_full
        tracklet.pred_attributes['siglip_embeddings'] = siglip_full

    print("Attribute prediction finished!")

    # Save cache
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(tracklets, f)
        print(f"Cached attributes to {cache_path}")

    return tracklets