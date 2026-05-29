"""
cross_attention_merger.py

Drop-in replacement for TransformerMerger using the CrossAttentionTransformer.

Unlike the Siamese CLS model, cross-attention requires both tracklets to be
present during the forward pass (frames attend to each other), so there is no
two-stage encode/classify shortcut.  Instead we batch all valid pairs through
the full model.

Usage in main.py:
    from tracklets.cross_attention_merger import CrossAttentionMerger

    merger = CrossAttentionMerger(
        model_path  = "weights/cross_attention/best_model.pt",
        meta_path   = "weights/cross_attention/transformer_merger_meta.json",
        merge_threshold = 0.5,
    )
    merged_tracklets = merger.merge(splitted_tracklets)
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from experiments.tracklet_merger.transformers.cross_attention.model import CrossAttentionTransformer
from experiments.tracklet_merger.transformers.utils import extract_frame_features, compute_pairwise_features


class CrossAttentionMerger:
    def __init__(
        self,
        model_path: Path,
        meta_path: Path,
        merge_threshold: float = 0.5,
        linkage_method: str = "average",
        jersey_entropy_threshold: float = 0.15,
        team_consistency_threshold: float = 0.9,
        team_confidence_threshold: float = 0.6,
        device: str = None,
    ):
        self.merge_threshold = merge_threshold
        self.linkage_method = linkage_method
        self.jersey_entropy_threshold = jersey_entropy_threshold
        self.team_consistency_threshold = team_consistency_threshold
        self.team_confidence_threshold = team_confidence_threshold

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model
        checkpoint = torch.load(str(model_path), map_location=self.device, weights_only=False)
        self.config = checkpoint["config"]
        self.model = CrossAttentionTransformer(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Load metadata
        with open(meta_path) as f:
            meta = json.load(f)
        self.t_max = meta.get("t_max", self.config.t_max)
        self.max_frame_value = meta.get("max_frame_value", self.config.max_frame_value)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def merge(self, tracklets: Dict) -> Dict:
        tracklet_ids = sorted(tracklets.keys())
        n = len(tracklet_ids)
        if n < 2:
            return tracklets

        dist_matrix = self._build_distance_matrix(tracklets, tracklet_ids)

        condensed = squareform(dist_matrix)
        Z = linkage(condensed, method=self.linkage_method)
        labels = fcluster(Z, t=self.merge_threshold, criterion="distance")

        return self._apply_merges(tracklets, tracklet_ids, labels)

    # ------------------------------------------------------------------
    # Distance matrix
    # ------------------------------------------------------------------

    def _build_distance_matrix(self, tracklets, tracklet_ids):
        n = len(tracklet_ids)
        dist_matrix = np.ones((n, n))
        np.fill_diagonal(dist_matrix, 0.0)

        # Pre-extract and prepare all tracklet features
        prepared = {}
        for idx, tid in enumerate(tracklet_ids):
            prepared[idx] = self._prepare_tracklet(tracklets[tid])

        # Collect valid pairs
        pair_indices = []
        pair_data = []

        for i in range(n):
            for j in range(i + 1, n):
                t_a = tracklets[tracklet_ids[i]]
                t_b = tracklets[tracklet_ids[j]]

                # Hard constraint: temporal overlap
                if set(t_a.frames) & set(t_b.frames):
                    dist_matrix[i, j] = dist_matrix[j, i] = 2.0
                    continue

                # Hard constraint: confident jersey mismatch
                if self._jersey_conflict(t_a, t_b):
                    dist_matrix[i, j] = dist_matrix[j, i] = 2.0
                    continue

                # Hard constraint: consistent team mismatch
                if self._team_conflict(t_a, t_b):
                    dist_matrix[i, j] = dist_matrix[j, i] = 2.0
                    continue

                pw = compute_pairwise_features(t_a, t_b)
                pair_indices.append((i, j))
                pair_data.append((prepared[i], prepared[j], pw))

        # Batch score all valid pairs
        if pair_indices:
            probs = self._batch_score(pair_data)
            for (i, j), p in zip(pair_indices, probs):
                d = float(1.0 - p)
                dist_matrix[i, j] = dist_matrix[j, i] = d

        return dist_matrix

    # ------------------------------------------------------------------
    # Feature preparation & scoring
    # ------------------------------------------------------------------

    def _prepare_tracklet(self, tracklet):
        """Extract and pad features for a single tracklet."""
        features, frames = extract_frame_features(tracklet)

        # Truncate feature dim if checkpoint was trained on fewer features
        expected_dim = self.config.raw_token_dim
        if features.shape[1] > expected_dim:
            features = features[:, :expected_dim]

        n = len(frames)

        # Temporal subsampling if too long
        if n > self.t_max:
            indices = np.linspace(0, n - 1, self.t_max, dtype=int)
            features = features[indices]
            frames = frames[indices]
            n = self.t_max

        # Pad to t_max
        feat_dim = features.shape[1]
        padded = np.zeros((self.t_max, feat_dim), dtype=np.float32)
        padded[:n] = features

        mask = np.ones(self.t_max, dtype=bool)
        mask[:n] = False

        positions = np.zeros(self.t_max, dtype=np.float32)
        if n > 1:
            positions[:n] = np.arange(n, dtype=np.float32) / (n - 1)

        frame_nums = np.zeros(self.t_max, dtype=np.float32)
        frame_nums[:n] = frames / self.max_frame_value

        return (
            torch.from_numpy(padded),
            torch.from_numpy(mask),
            torch.from_numpy(positions),
            torch.from_numpy(frame_nums),
        )

    @torch.no_grad()
    def _batch_score(self, pair_data):
        """Run full forward pass on batches of tracklet pairs."""
        batch_size = 64
        all_probs = []

        for start in range(0, len(pair_data), batch_size):
            end = min(start + batch_size, len(pair_data))
            batch = pair_data[start:end]

            tokens_a = torch.stack([d[0][0] for d in batch]).to(self.device)
            mask_a   = torch.stack([d[0][1] for d in batch]).to(self.device)
            pos_a    = torch.stack([d[0][2] for d in batch]).to(self.device)
            frames_a = torch.stack([d[0][3] for d in batch]).to(self.device)

            tokens_b = torch.stack([d[1][0] for d in batch]).to(self.device)
            mask_b   = torch.stack([d[1][1] for d in batch]).to(self.device)
            pos_b    = torch.stack([d[1][2] for d in batch]).to(self.device)
            frames_b = torch.stack([d[1][3] for d in batch]).to(self.device)

            pw = torch.tensor(
                np.stack([d[2] for d in batch]), dtype=torch.float32
            ).to(self.device)

            logits = self.model(
                tokens_a, mask_a, pos_a, frames_a,
                tokens_b, mask_b, pos_b, frames_b,
                pw,
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())

        return all_probs

    # ------------------------------------------------------------------
    # Hard-constraint helpers (identical to TransformerMerger)
    # ------------------------------------------------------------------

    def _jersey_conflict(self, t1, t2) -> bool:
        j1, e1 = self._jersey_stats(t1)
        j2, e2 = self._jersey_stats(t2)
        both_conf = (j1 is not None and j2 is not None
                     and e1 < self.jersey_entropy_threshold
                     and e2 < self.jersey_entropy_threshold)
        return bool(both_conf and j1 != j2)

    def _team_conflict(self, t1, t2) -> bool:
        tm1, c1 = self._team_stats(t1)
        tm2, c2 = self._team_stats(t2)
        both_cons = (tm1 is not None and tm2 is not None
                     and c1 >= self.team_consistency_threshold
                     and c2 >= self.team_consistency_threshold)
        return bool(both_cons and tm1 != tm2)

    @staticmethod
    def _jersey_stats(tracklet):
        jerseys = tracklet.pred_attributes.get("jerseys", [])
        entropies = tracklet.pred_attributes.get("jersey_entropies", [])
        valid = [(j, entropies[i] if i < len(entropies) else 1.0)
                 for i, j in enumerate(jerseys)
                 if not (isinstance(j, float) and np.isnan(j))]
        if not valid:
            return None, 1.0
        js, es = zip(*valid)
        mode = max(set(js), key=js.count)
        return mode, float(np.mean([e for j, e in zip(js, es) if j == mode]))

    def _team_stats(self, tracklet):
        teams      = tracklet.pred_attributes.get("teams", [])
        team_confs = tracklet.pred_attributes.get("team_confs", [])
        valid = [
            teams[i] for i in range(len(teams))
            if not (isinstance(teams[i], float) and np.isnan(teams[i]))
            and i < len(team_confs)
            and team_confs[i] is not None
            and not (isinstance(team_confs[i], float) and np.isnan(team_confs[i]))
            and float(team_confs[i]) >= self.team_confidence_threshold
        ]
        if not valid:
            return None, 0.0
        mode = max(set(valid), key=valid.count)
        return mode, valid.count(mode) / len(valid)

    # ------------------------------------------------------------------
    # Merge application (identical to TransformerMerger)
    # ------------------------------------------------------------------

    def _apply_merges(self, tracklets, tracklet_ids, cluster_labels):
        clusters = {}
        for idx, cid in enumerate(cluster_labels):
            clusters.setdefault(cid, []).append(tracklet_ids[idx])

        result = {}
        overflow_id = max(clusters.keys()) + 1

        for cid, members in clusters.items():
            members.sort(key=lambda x: tracklets[x].frames[0])
            base = tracklets[members[0]]
            current_frames = set(base.frames)

            for next_id in members[1:]:
                other = tracklets[next_id]
                if current_frames & set(other.frames):
                    result[overflow_id] = other
                    overflow_id += 1
                    continue
                base.frames.extend(other.frames)
                base.bboxes.extend(other.bboxes)
                base.scores.extend(other.scores)
                base.embeddings.extend(other.embeddings)
                for k in base.pred_attributes:
                    base.pred_attributes[k].extend(other.pred_attributes.get(k, []))
                current_frames |= set(other.frames)

            result[cid] = base

        return result
