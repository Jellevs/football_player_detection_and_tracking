import torch
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import umap

from transformers import AutoProcessor, AutoModel


class TeamClassifier:
    def __init__(self, device='cpu', batch_size=32, paths=None):
        self.device = device
        self.batch_size = batch_size

        model_path = str(paths.siglip_model_path) if paths and paths.siglip_model_path else "google/siglip-base-patch16-224"
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.model.eval()

        # Detect whether we have the full SiglipModel (with get_image_features)
        # or just SiglipVisionModel (loaded from local vision-only weights)
        self._has_get_image_features = hasattr(self.model, 'get_image_features')
        if not self._has_get_image_features:
            print("SigLIP: Using vision model forward() (no get_image_features)")


    def extract_features(self, crops):
        """
        Extract SigLIP embeddings for a list of crops (numpy arrays, RGB).
        Returns array of shape (N, embed_dim).
        """
        if not crops:
            return np.array([])

        all_embeddings = []
        valid_indices = []

        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0:
                continue
            valid_indices.append(i)

        valid_crops = [crops[i] for i in valid_indices]

        for batch_start in range(0, len(valid_crops), self.batch_size):
            batch = valid_crops[batch_start:batch_start + self.batch_size]
            pil_images = [Image.fromarray(c) for c in batch]

            inputs = self.processor(images=pil_images, return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                if self._has_get_image_features:
                    # Full SiglipModel, use the convenience method
                    embeddings = self.model.get_image_features(**inputs)
                else:
                    # SiglipVisionModel, call forward directly, use pooler_output
                    outputs = self.model(**inputs)
                    # pooler_output is the [CLS] token embedding after projection
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        embeddings = outputs.pooler_output
                    else:
                        # Fallback: mean pool the last hidden state
                        embeddings = outputs.last_hidden_state.mean(dim=1)

            embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)

        all_embeddings = np.concatenate(all_embeddings)

        # Map back to original indices (fill skipped crops with zeros)
        embed_dim = all_embeddings.shape[1]
        full_embeddings = np.zeros((len(crops), embed_dim))
        for i, idx in enumerate(valid_indices):
            full_embeddings[idx] = all_embeddings[i]

        return full_embeddings


    def fit_predict_all(self, all_crops):
        """
        Extract SigLIP features for all crops, reduce with UMAP,
        and cluster into 2 teams with KMeans.

        No player_mask needed, dataset only contains players and goalkeepers.
        """
        if len(all_crops) == 0:
            print("no crops")
            return np.array([]), np.array([]), np.array([])

        # Extract features for all crops
        all_features = self.extract_features(all_crops)

        # UMAP dimensionality reduction
        reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=30, min_dist=0.0, metric='cosine')
        all_projections = reducer.fit_transform(all_features)

        if len(all_projections) < 2:
            return np.zeros(len(all_crops), dtype=int), all_features, np.zeros(len(all_crops))

        # KMeans clustering into 2 teams
        cluster_model = KMeans(n_clusters=2, random_state=42, n_init=10)
        predictions = cluster_model.fit_predict(all_projections)

        # Distance-based confidence: how much closer is the point to its own
        # centroid vs the other centroid.  Range [0, 1]; 0.5 = equidistant.
        distances = cluster_model.transform(all_projections)   # (N, 2)
        own_dist   = distances[np.arange(len(predictions)), predictions]
        other_dist = distances[np.arange(len(predictions)), 1 - predictions]
        confidences = 1.0 - own_dist / (own_dist + other_dist + 1e-8)



        return predictions, all_features, confidences
