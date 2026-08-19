from tqdm import tqdm
import pickle
import numpy as np

from .jersey_number.jersey_number_predictor_parseq import JerseyNumberPredictorParseq
from .teamclassifier import TeamClassifier


def predict_attributes(images, tracklets, paths, jersey_config, device):
    """ Predict jersey numbers and team id for all tracklets """
    cache_path = paths.set_cache_path("attributes", paths.sequence)

    if cache_path and cache_path.exists():
        print(f"Loading cached attributes from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f"Cache not found for {cache_path}")
    jersey_predictor = JerseyNumberPredictorParseq(
        paths=paths,
        jersey_config=jersey_config,
        device=device
    )

    team_classifier = TeamClassifier(
        device=device,
        batch_size=32,
        paths=paths
    )

    print("Phase 1: Extracting crops and predicting jersey numbers...")

    tracklet_team_crops = {}
    tracklet_team_indices = {}
    all_team_crops = []
    all_team_crop_info = []

    for track_id, tracklet in tqdm(tracklets.items(), desc="Jersey prediction"):
        team_crops, team_indices, jerseys, jersey_confs_mean, jersey_entropies = jersey_predictor.predict(images, tracklet)

        tracklet.pred_attributes['jerseys'] = jerseys.tolist()
        tracklet.pred_attributes['jersey_confs_mean'] = jersey_confs_mean.tolist()
        tracklet.pred_attributes['jersey_entropies'] = jersey_entropies.tolist()

        # Collect ReID-filtered torso crops for team classification
        tracklet_team_crops[track_id] = team_crops
        tracklet_team_indices[track_id] = team_indices

        for local_idx in range(len(team_crops)):
            all_team_crops.append(team_crops[local_idx])
            all_team_crop_info.append((track_id, local_idx))

    # Phase 2: Team classification, fit on all ReID-filtered torso crops
    print(f"Phase 2: Team classification ({len(all_team_crops)} crops)...")
    team_predictions, siglip_embeddings, team_confidences = team_classifier.fit_predict_all(all_team_crops)

 


    # Phase 3: Map predictions back to tracklets
    print("Phase 3: Mapping predictions to tracklets...")

    global_idx_lookup = {}
    for global_idx, (track_id, local_idx) in enumerate(all_team_crop_info):
        global_idx_lookup[(track_id, local_idx)] = global_idx

    for track_id, tracklet in tracklets.items():
        indices = tracklet_team_indices[track_id]
        num_frames = len(tracklet.frames)

        teams_full      = [np.nan] * num_frames
        team_confs_full = [np.nan] * num_frames
        siglip_full     = [np.zeros(768)] * num_frames

        n_crops = len(tracklet_team_crops[track_id])
        for local_idx in range(n_crops):
            tracklet_idx = indices[local_idx]

            global_idx = global_idx_lookup[(track_id, local_idx)]
            siglip_full[tracklet_idx]     = siglip_embeddings[global_idx]
            teams_full[tracklet_idx]      = team_predictions[global_idx]
            team_confs_full[tracklet_idx] = float(team_confidences[global_idx])

        tracklet.pred_attributes['teams']      = teams_full
        tracklet.pred_attributes['team_confs'] = team_confs_full
        tracklet.pred_attributes['siglip_embeddings'] = siglip_full

    print("Attribute prediction finished!")

    # Save cache
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(tracklets, f)
        print(f"Cached attributes to {cache_path}")

    return tracklets
