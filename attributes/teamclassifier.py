from transformers import AutoProcessor, SiglipVisionModel
import umap
from sklearn.cluster import KMeans
import supervision as sv
import torch
from tqdm import tqdm
import numpy as np


def create_batches(sequence, batch_size):
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    """ Team classifier using SigLIP embeddings, UMAP, and KMeans clustering """
    
    def __init__(self, device, batch_size, paths):
        self.device = device
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(
            pretrained_model_name_or_path=paths.siglip_model_path
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(paths.siglip_model_path)
        

    def extract_features(self, crops):
        """ Extract SigLIP embeddings from crops """
        if len(crops) == 0:
            return np.array([]).reshape(0, 768)
            
        crops_pil = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = list(create_batches(crops_pil, self.batch_size))
        data = []
        
        with torch.no_grad():
            for batch in tqdm(batches, desc="Extracting features", leave=False):
                inputs = self.processor(
                    images=batch, return_tensors='pt'
                ).to(self.device)
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data)
    

    def fit_predict_all(self, all_crops, player_mask):
        """ Fit UMAP on all crops and cluster based on player crops only """
        if len(all_crops) == 0:
            return np.array([])
        
        player_mask = np.array(player_mask)
        
        # Extract features for ALL crops
        all_features = self.extract_features(all_crops)
        
        # UMAP fit_transform on ALL data
        reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=30, min_dist=0.0, metric='cosine')
        all_projections = reducer.fit_transform(all_features)
        
        player_projections = all_projections[player_mask]
        
        if len(player_projections) < 2:
            return np.zeros(len(all_crops), dtype=int)
        
        # Fit KMeans on player projections only
        cluster_model = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_model.fit(player_projections)
        
        # Predict on ALL projections and filter non  players out later
        predictions = cluster_model.predict(all_projections)
        
        return predictions