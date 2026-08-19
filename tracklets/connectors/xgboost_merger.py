"""
xgboost_merger.py

Drop-in replacement for SoccerAwareMerger / SimpleTrackletMerger.

Usage in main.py:
    from tracklets.connectors.xgboost_merger import XGBoostMerger

    merger = XGBoostMerger(
        model_path = settings.WEIGHTS_ROOT / "xgboost" / "xgboost_merger.json",
        meta_path  = settings.WEIGHTS_ROOT / "xgboost" / "xgboost_merger_meta.json",
        merge_threshold = 0.5,          # distance threshold for hierarchical clustering
        linkage_method  = "average",
    )
    merged_tracklets = merger.merge(splitted_tracklets)

The model outputs a merge-probability for each pair; this is converted to a
distance (1 - p) and fed into the same average-linkage hierarchical clustering
used by SoccerAwareMerger, so the clustering behaviour is identical.

Temporal overlap is enforced as a hard constraint by default. Jersey and team
conflicts can optionally be enforced at distance = 2.0 for ablation studies.
"""

import csv
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from xgboost import XGBClassifier


class XGBoostMerger:
    def __init__(
        self,
        model_path: Union[Path, str, List[Union[Path, str]]],
        meta_path: Path,
        merge_threshold: float = 0.5,
        linkage_method: str = "average",
        jersey_entropy_threshold: float = 0.15,
        team_consistency_threshold: float = 0.9,
        team_confidence_threshold: float = 0.6,
        disable_temporal_constraint: bool = False,
        disable_jersey_constraint: bool = True,
        disable_team_constraint: bool = True,
        log_path: Optional[Union[Path, str]] = None,
    ):
        self.merge_threshold            = merge_threshold
        self.linkage_method             = linkage_method
        self.jersey_entropy_threshold   = jersey_entropy_threshold
        self.team_consistency_threshold = team_consistency_threshold
        self.team_confidence_threshold  = team_confidence_threshold
        self.disable_temporal_constraint = disable_temporal_constraint
        self.disable_jersey_constraint   = disable_jersey_constraint
        self.disable_team_constraint     = disable_team_constraint
        self.log_path = Path(log_path) if log_path is not None else None

        # Load model(s), single path or list of paths for ensemble averaging
        if isinstance(model_path, (list, tuple)):
            paths = list(model_path)
        else:
            paths = [model_path]
        self.models = []
        for p in paths:
            m = XGBClassifier()
            m.load_model(str(p))
            self.models.append(m)
        # Back-compat alias for any caller that still references .model
        self.model = self.models[0]

        # Load metadata (column order + threshold)
        with open(meta_path) as f:
            meta = json.load(f)
        self.feature_cols = meta["feature_cols"]
        # Override threshold from meta if not set explicitly by the caller
        # (caller's merge_threshold takes precedence as it controls clustering)

    # Public interface

    def merge(self, tracklets: Dict, sequence_name: str = "") -> Dict:
        tracklet_ids = sorted(tracklets.keys())
        n = len(tracklet_ids)
        if n < 2:
            return tracklets

        dist_matrix = self._build_distance_matrix(tracklets, tracklet_ids)

        condensed  = squareform(dist_matrix)
        Z          = linkage(condensed, method=self.linkage_method)
        labels     = fcluster(Z, t=self.merge_threshold, criterion="distance")

        return self._apply_merges(tracklets, tracklet_ids, labels, sequence_name)

    # Distance matrix

    def _build_distance_matrix(self, tracklets, tracklet_ids):
        n            = len(tracklet_ids)
        dist_matrix  = np.ones((n, n))
        np.fill_diagonal(dist_matrix, 0.0)

        # Build one feature row per pair, then batch-predict
        pair_indices = []
        pair_rows    = []

        for i in range(n):
            for j in range(i + 1, n):
                t_a = tracklets[tracklet_ids[i]]
                t_b = tracklets[tracklet_ids[j]]

                # Hard constraint: temporal overlap
                if not self.disable_temporal_constraint and set(t_a.frames) & set(t_b.frames):
                    dist_matrix[i, j] = dist_matrix[j, i] = 2.0
                    continue

                # Hard constraint: confident jersey mismatch
                if not self.disable_jersey_constraint and self._jersey_conflict(t_a, t_b):
                    dist_matrix[i, j] = dist_matrix[j, i] = 2.0
                    continue

                # Hard constraint: consistent team mismatch
                if not self.disable_team_constraint and self._team_conflict(t_a, t_b):
                    dist_matrix[i, j] = dist_matrix[j, i] = 2.0
                    continue

                agg_a = self._aggregate(t_a)
                agg_b = self._aggregate(t_b)
                row   = self._build_feature_row(agg_a, agg_b)

                pair_indices.append((i, j))
                pair_rows.append(row)

        if pair_rows:
            import pandas as pd
            X      = pd.DataFrame(pair_rows, columns=self.feature_cols).fillna(0).values
            probs_per_model = np.stack(
                [m.predict_proba(X)[:, 1] for m in self.models], axis=0
            )
            probs = probs_per_model.mean(axis=0)
            for (i, j), p in zip(pair_indices, probs):
                d = float(1.0 - p)
                dist_matrix[i, j] = dist_matrix[j, i] = d

        return dist_matrix

    # Feature construction  (mirrors generate_merger_training_data.py)

    def _aggregate(self, tracklet) -> dict:
        """Aggregate a tracklet into the same feature dict used during training."""
        frames     = tracklet.frames
        bboxes     = tracklet.bboxes
        embeddings = tracklet.embeddings
        fa = {}

        # ReID
        if embeddings:
            emb = np.stack(embeddings).astype(np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-6
            emb   = emb / norms
            mean  = emb.mean(axis=0)
            std   = emb.std(axis=0)
        else:
            mean = np.zeros(512, dtype=np.float32)
            std  = np.zeros(512, dtype=np.float32)
        for i, v in enumerate(mean): fa[f"reid_mean_{i}"] = v
        for i, v in enumerate(std):  fa[f"reid_std_{i}"]  = v

        # Retained only for the pairwise SigLIP cosine feature.
        siglip_all = tracklet.pred_attributes.get("siglip_embeddings", [])
        valid_siglip = [
            np.asarray(siglip, dtype=np.float32)
            for siglip in siglip_all
            if np.any(np.asarray(siglip) != 0)
        ]
        fa["_siglip_mean_vec"] = (
            np.stack(valid_siglip).mean(axis=0)
            if valid_siglip
            else np.zeros(768, dtype=np.float32)
        )

        # Jersey
        jerseys   = tracklet.pred_attributes.get("jerseys", [])
        entropies = tracklet.pred_attributes.get("jersey_entropies", [1.0] * len(frames))
        confs     = tracklet.pred_attributes.get("jersey_confs_mean", [0.0] * len(frames))
        valid_j   = [(j, entropies[i] if i < len(entropies) else 1.0,
                         confs[i]     if i < len(confs)     else 0.0)
                     for i, j in enumerate(jerseys) if not (isinstance(j, float) and np.isnan(j))]
        if valid_j:
            js, es, cs = zip(*valid_j)
            mode = max(set(js), key=js.count)
            fa["jersey_mode"]         = float(mode)
            fa["jersey_entropy_mean"] = float(np.mean([e for j_, e in zip(js, es) if j_ == mode]))
            fa["jersey_conf_mean"]    = float(np.mean(cs))
            fa["jersey_coverage"]     = len(valid_j) / max(len(frames), 1)
        else:
            fa["jersey_mode"]         = np.nan
            fa["jersey_entropy_mean"] = 1.0
            fa["jersey_conf_mean"]    = 0.0
            fa["jersey_coverage"]     = 0.0

        # Team
        teams_raw  = tracklet.pred_attributes.get("teams", [])
        team_confs = tracklet.pred_attributes.get("team_confs", [])

        valid_team_confs = [
            team_confs[i] for i in range(len(teams_raw))
            if teams_raw[i] is not None
            and not (isinstance(teams_raw[i], float) and np.isnan(teams_raw[i]))
            and i < len(team_confs)
            and team_confs[i] is not None
            and not (isinstance(team_confs[i], float) and np.isnan(team_confs[i]))
        ]
        teams_confident = [
            teams_raw[i] for i in range(len(teams_raw))
            if teams_raw[i] is not None
            and not (isinstance(teams_raw[i], float) and np.isnan(teams_raw[i]))
            and i < len(team_confs)
            and team_confs[i] is not None
            and not (isinstance(team_confs[i], float) and np.isnan(team_confs[i]))
            and team_confs[i] >= self.team_confidence_threshold
        ]

        if teams_confident:
            mode = max(set(teams_confident), key=teams_confident.count)
            fa["team_mode"]        = float(mode)
            fa["team_consistency"] = teams_confident.count(mode) / len(teams_confident)
            fa["team_coverage"]    = len(teams_confident) / max(len(frames), 1)
        else:
            fa["team_mode"]        = np.nan
            fa["team_consistency"] = 0.0
            fa["team_coverage"]    = 0.0

        fa["team_conf_mean"] = float(np.mean(valid_team_confs)) if valid_team_confs else 0.0

        # Temporal / spatial
        def cx_cy(bbox):
            return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

        sx, sy = cx_cy(bboxes[0])
        ex, ey = cx_cy(bboxes[-1])
        fa["start_frame"]      = float(frames[0])
        fa["end_frame"]        = float(frames[-1])
        fa["duration"]         = float(frames[-1] - frames[0] + 1)
        fa["n_frames"]         = float(len(frames))
        fa["start_x"]          = float(sx)
        fa["start_y"]          = float(sy)
        fa["end_x"]            = float(ex)
        fa["end_y"]            = float(ey)
        fa["mean_bbox_height"] = float(np.mean([b[3] - b[1] for b in bboxes]))

        return fa

    def _build_feature_row(self, fa: dict, fb: dict) -> dict:
        """Build the flat feature row matching training column names."""
        row = {}
        for k, v in fa.items():
            if not k.startswith("_"):
                row[f"A_{k}"] = v
        for k, v in fb.items():
            if not k.startswith("_"):
                row[f"B_{k}"] = v

        # Pairwise
        if fa["end_frame"] <= fb["start_frame"]:
            exit_f, entry_f = fa, fb
        elif fb["end_frame"] <= fa["start_frame"]:
            exit_f, entry_f = fb, fa
        else:
            exit_f, entry_f = None, None

        row["pairwise_temporal_gap"] = (
            float(entry_f["start_frame"] - exit_f["end_frame"]) if exit_f else 0.0
        )
        if exit_f:
            dx = entry_f["start_x"] - exit_f["end_x"]
            dy = entry_f["start_y"] - exit_f["end_y"]
            row["pairwise_spatial_distance"] = float(np.sqrt(dx**2 + dy**2))
            row["pairwise_endpoint_dx"]      = float(dx)
            row["pairwise_endpoint_dy"]      = float(dy)
        else:
            row["pairwise_spatial_distance"] = 0.0
            row["pairwise_endpoint_dx"]      = 0.0
            row["pairwise_endpoint_dy"]      = 0.0

        h_a, h_b = fa["mean_bbox_height"], fb["mean_bbox_height"]
        row["pairwise_bbox_height_ratio"] = float(h_a / h_b) if h_b > 0 else 1.0

        reid_a = np.array([fa[k] for k in sorted(fa) if k.startswith("reid_mean_")], dtype=np.float32)
        reid_b = np.array([fb[k] for k in sorted(fb) if k.startswith("reid_mean_")], dtype=np.float32)
        na, nb = np.linalg.norm(reid_a) + 1e-6, np.linalg.norm(reid_b) + 1e-6
        row["pairwise_reid_cosine_sim"] = float(np.dot(reid_a / na, reid_b / nb))

        siglip_a = fa["_siglip_mean_vec"]
        siglip_b = fb["_siglip_mean_vec"]
        na_s = np.linalg.norm(siglip_a) + 1e-6
        nb_s = np.linalg.norm(siglip_b) + 1e-6
        row["pairwise_siglip_cosine_sim"] = float(
            np.dot(siglip_a / na_s, siglip_b / nb_s)
        )

        j_a, j_b = fa["jersey_mode"], fb["jersey_mode"]
        nan_a = isinstance(j_a, float) and np.isnan(j_a)
        nan_b = isinstance(j_b, float) and np.isnan(j_b)
        both  = not nan_a and not nan_b
        row["pairwise_jersey_match"]          = float(both and j_a == j_b)
        row["pairwise_jersey_conflict"]       = float(both and j_a != j_b)
        row["pairwise_jersey_both_confident"] = float(
            both and fa["jersey_entropy_mean"] < 0.15 and fb["jersey_entropy_mean"] < 0.15
        )

        t_a, t_b = fa["team_mode"], fb["team_mode"]
        nan_ta = isinstance(t_a, float) and np.isnan(t_a)
        nan_tb = isinstance(t_b, float) and np.isnan(t_b)
        both_t = not nan_ta and not nan_tb
        row["pairwise_team_match"]             = float(both_t and t_a == t_b)
        row["pairwise_team_conflict"]          = float(both_t and t_a != t_b)
        row["pairwise_team_both_consistent"]   = float(
            both_t and fa["team_consistency"] > 0.9 and fb["team_consistency"] > 0.9
        )

        return row

    # Hard-constraint helpers

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
        jerseys   = tracklet.pred_attributes.get("jerseys", [])
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
        teams_raw  = tracklet.pred_attributes.get("teams", [])
        team_confs = tracklet.pred_attributes.get("team_confs", [])
        teams = [
            teams_raw[i] for i in range(len(teams_raw))
            if teams_raw[i] is not None
            and not (isinstance(teams_raw[i], float) and np.isnan(teams_raw[i]))
            and i < len(team_confs)
            and team_confs[i] is not None
            and not (isinstance(team_confs[i], float) and np.isnan(team_confs[i]))
            and team_confs[i] >= self.team_confidence_threshold
        ]
        if not teams:
            return None, 0.0
        mode = max(set(teams), key=teams.count)
        return mode, teams.count(mode) / len(teams)

    # Merge application  (identical logic to SoccerAwareMerger)

    def _apply_merges(self, tracklets, tracklet_ids, cluster_labels, sequence_name=""):
        clusters = {}
        for idx, cid in enumerate(cluster_labels):
            clusters.setdefault(cid, []).append(tracklet_ids[idx])

        result       = {}
        overflow_id  = max(clusters.keys()) + 1

        for cid, members in clusters.items():
            members.sort(key=lambda x: tracklets[x].frames[0])
            base          = tracklets[members[0]]
            current_frames = set(base.frames)

            for next_id in members[1:]:
                other = tracklets[next_id]
                if current_frames & set(other.frames):
                    result[overflow_id] = other
                    overflow_id += 1
                    continue
                self._log_merge(sequence_name, members[0], next_id, base, other)
                base.frames.extend(other.frames)
                base.bboxes.extend(other.bboxes)
                base.scores.extend(other.scores)
                base.embeddings.extend(other.embeddings)
                for k in base.pred_attributes:
                    base.pred_attributes[k].extend(other.pred_attributes.get(k, []))
                for k in base.gt_attributes:
                    base.gt_attributes[k].extend(other.gt_attributes.get(k, []))
                current_frames |= set(other.frames)

            result[cid] = base

        return result

    def _log_merge(self, sequence_name, base_id, absorbed_id, base, absorbed):
        if self.log_path is None:
            return
        write_header = not self.log_path.exists()
        with self.log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "merger_type", "sequence",
                    "base_id", "absorbed_id",
                    "base_start_frame", "base_end_frame",
                    "absorbed_start_frame", "absorbed_end_frame",
                ])
            writer.writerow([
                "XGBoostMerger", sequence_name,
                base_id, absorbed_id,
                base.frames[0], base.frames[-1],
                absorbed.frames[0], absorbed.frames[-1],
            ])
