from tqdm import tqdm
import pickle
import numpy as np

from .jersey_number.jersey_number_predictor_parseq import JerseyNumberPredictorParseq
from .teamclassifier import TeamClassifier


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

    # Phase 1: Extract crops and predict jersey numbers
    print("Phase 1: Extracting crops and predicting jersey numbers...")

    tracklet_torso_crops = {}
    tracklet_crop_indices = {}

    all_torso_crops = []
    all_crop_info = []

    for track_id, tracklet in tqdm(tracklets.items(), desc="Processing tracklets"):
        # Get torso crops for team classification + jersey predictions
        torso_crops, indices, jerseys, jersey_confs_mean, jersey_entropies = jersey_predictor.predict(images, tracklet)

        # Store jersey predictions
        tracklet.pred_attributes['jerseys'] = jerseys.tolist()
        tracklet.pred_attributes['jersey_confs_mean'] = jersey_confs_mean.tolist()
        tracklet.pred_attributes['jersey_entropies'] = jersey_entropies.tolist()

        # Store torso crops for team classification
        tracklet_torso_crops[track_id] = torso_crops
        tracklet_crop_indices[track_id] = indices

        # All tracklets are players or goalkeepers — no GT-based filtering needed
        for i in range(len(torso_crops)):
            all_torso_crops.append(torso_crops[i])
            all_crop_info.append((track_id, i))

    # Phase 2: Team classification — fit on all crops, no player_mask needed
    print("Phase 2: Team classification...")
    team_predictions, siglip_embeddings = team_classifier.fit_predict_all(all_torso_crops)

    # Phase 3: Map predictions back to tracklets
    print("Phase 3: Mapping predictions to tracklets...")

    prediction_lookup = {}
    for global_idx, (track_id, local_idx) in enumerate(all_crop_info):
        prediction_lookup[(track_id, local_idx)] = team_predictions[global_idx]

    for track_id, tracklet in tracklets.items():

        indices = tracklet_crop_indices[track_id]
        num_frames = len(tracklet.frames)

        teams_full = [np.nan] * num_frames
        siglip_full = [np.zeros(768)] * num_frames

        n_crops = len(tracklet_torso_crops[track_id])
        for local_idx in range(n_crops):
            tracklet_idx = indices[local_idx]

            global_idx = all_crop_info.index((track_id, local_idx))
            siglip_full[tracklet_idx] = siglip_embeddings[global_idx]

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