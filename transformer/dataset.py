"""
Dataset and feature extraction for the Siamese CLS Transformer.

The dataset loads pre-generated pair metadata and extracts per-frame
features from the original pickle caches on initialization.
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---- Feature layout constants ----
REID_START    = 0
REID_END      = 512
SIGLIP_START  = 512
SIGLIP_END    = 1280
BBOX_START    = 1280
BBOX_END      = 1287
SCALAR_START  = 1287
SCALAR_END    = 1292   # jersey, entropy, jersey_conf, team, team_conf
SCORE_START   = 1292
SCORE_END     = 1293
RAW_DIM       = 1293

# Pairwise feature indices that need adjustment on A/B swap
PW_ENDPOINT_DX_IDX         = 2
PW_ENDPOINT_DY_IDX         = 3
PW_BBOX_HEIGHT_RATIO_IDX   = 4


# ---------------------------------------------------------------------------
# Feature extraction from Tracklet objects
# ---------------------------------------------------------------------------

def extract_frame_features(tracklet) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract per-frame features from a Tracklet object.

    Returns:
        features: (n_frames, 1293) float32 array
        frames:   (n_frames,) float32 array of frame indices
    """
    n = len(tracklet.frames)
    features = np.zeros((n, RAW_DIM), dtype=np.float32)

    jerseys    = tracklet.pred_attributes.get("jerseys", [])
    entropies  = tracklet.pred_attributes.get("jersey_entropies", [])
    confs      = tracklet.pred_attributes.get("jersey_confs_mean", [])
    teams      = tracklet.pred_attributes.get("teams", [])
    team_confs = tracklet.pred_attributes.get("team_confs", [])
    siglip_all = tracklet.pred_attributes.get("siglip_embeddings", [])

    for i in range(n):
        # ReID embedding (512-dim, L2-normalized)
        if tracklet.embeddings and i < len(tracklet.embeddings):
            emb = np.array(tracklet.embeddings[i], dtype=np.float32)
            norm = np.linalg.norm(emb) + 1e-6
            features[i, REID_START:REID_END] = emb / norm

        # SigLIP embedding (768-dim)
        if i < len(siglip_all):
            sig = np.array(siglip_all[i], dtype=np.float32)
            if np.any(sig != 0):
                features[i, SIGLIP_START:SIGLIP_END] = sig

        # BBox derived features (7-dim)
        if i < len(tracklet.bboxes):
            b = tracklet.bboxes[i]
            x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w  = x2 - x1
            h  = y2 - y1
            features[i, BBOX_START:BBOX_END] = [
                cx, cy, w, h,
                w / (h + 1e-6),   # aspect ratio
                w * h,            # area
                cy / 1080.0,      # normalized y (assume 1080p)
            ]

        # Scalar features (5-dim): jersey, entropy, jersey_conf, team, team_conf
        jersey = float(jerseys[i]) if i < len(jerseys) else 0.0
        if isinstance(jersey, float) and np.isnan(jersey):
            jersey = 0.0
        entropy   = float(entropies[i])  if i < len(entropies)  else 1.0
        conf      = float(confs[i])      if i < len(confs)      else 0.0
        team      = float(teams[i])      if i < len(teams)      else 0.0
        if isinstance(team, float) and np.isnan(team):
            team = 0.0
        team_conf = float(team_confs[i]) if i < len(team_confs) else 0.0
        if isinstance(team_conf, float) and np.isnan(team_conf):
            team_conf = 0.0
        features[i, SCALAR_START:SCALAR_END] = [jersey, entropy, conf, team, team_conf]

        # Detection score (1-dim)
        score = float(tracklet.scores[i]) if i < len(tracklet.scores) else 0.0
        features[i, SCORE_START:SCORE_END] = score

    return features, np.array(tracklet.frames, dtype=np.float32)


# ---------------------------------------------------------------------------
# Pairwise features (mirrors XGBoost pipeline exactly)
# ---------------------------------------------------------------------------

def compute_pairwise_features(tracklet_a, tracklet_b) -> np.ndarray:
    """
    Compute 12-dim pairwise feature vector.

    Order: temporal_gap, spatial_distance, endpoint_dx, endpoint_dy,
           bbox_height_ratio, reid_cosine_sim, siglip_cosine_sim,
           jersey_match, jersey_conflict, jersey_both_confident,
           team_match, team_both_consistent
    """
    pw = np.zeros(12, dtype=np.float32)
    fa, fb = tracklet_a, tracklet_b

    # Temporal ordering
    if fa.frames[-1] <= fb.frames[0]:
        exit_t, entry_t = fa, fb
    elif fb.frames[-1] <= fa.frames[0]:
        exit_t, entry_t = fb, fa
    else:
        exit_t, entry_t = None, None

    # 0: temporal_gap
    if exit_t is not None:
        pw[0] = float(entry_t.frames[0] - exit_t.frames[-1])

    # 1-3: spatial_distance, endpoint_dx, endpoint_dy
    if exit_t is not None:
        ex = (exit_t.bboxes[-1][0] + exit_t.bboxes[-1][2]) / 2.0
        ey = (exit_t.bboxes[-1][1] + exit_t.bboxes[-1][3]) / 2.0
        sx = (entry_t.bboxes[0][0] + entry_t.bboxes[0][2]) / 2.0
        sy = (entry_t.bboxes[0][1] + entry_t.bboxes[0][3]) / 2.0
        dx, dy = sx - ex, sy - ey
        pw[1] = float(np.sqrt(dx**2 + dy**2))
        pw[2] = float(dx)
        pw[3] = float(dy)

    # 4: bbox_height_ratio
    h_a = np.mean([b[3] - b[1] for b in fa.bboxes])
    h_b = np.mean([b[3] - b[1] for b in fb.bboxes])
    pw[4] = float(h_a / h_b) if h_b > 0 else 1.0

    # 5: reid_cosine_sim
    if fa.embeddings and fb.embeddings:
        emb_a = np.stack(fa.embeddings).astype(np.float32)
        emb_b = np.stack(fb.embeddings).astype(np.float32)
        emb_a = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-6)
        emb_b = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-6)
        mean_a = emb_a.mean(axis=0)
        mean_b = emb_b.mean(axis=0)
        pw[5] = float(np.dot(
            mean_a / (np.linalg.norm(mean_a) + 1e-6),
            mean_b / (np.linalg.norm(mean_b) + 1e-6),
        ))

    # 6: siglip_cosine_sim
    sig_a = [s for s in fa.pred_attributes.get("siglip_embeddings", [])
             if np.any(np.array(s) != 0)]
    sig_b = [s for s in fb.pred_attributes.get("siglip_embeddings", [])
             if np.any(np.array(s) != 0)]
    if sig_a and sig_b:
        ma = np.stack(sig_a).mean(axis=0)
        mb = np.stack(sig_b).mean(axis=0)
        pw[6] = float(np.dot(
            ma / (np.linalg.norm(ma) + 1e-6),
            mb / (np.linalg.norm(mb) + 1e-6),
        ))

    # Helper: jersey stats
    def jersey_stats(t):
        js = t.pred_attributes.get("jerseys", [])
        es = t.pred_attributes.get("jersey_entropies", [])
        valid = [(j, es[i] if i < len(es) else 1.0)
                 for i, j in enumerate(js)
                 if not (isinstance(j, float) and np.isnan(j))]
        if not valid:
            return None, 1.0
        nums, ents = zip(*valid)
        mode = max(set(nums), key=nums.count)
        return mode, float(np.mean([e for n, e in zip(nums, ents) if n == mode]))

    j_a, e_a = jersey_stats(fa)
    j_b, e_b = jersey_stats(fb)
    both_j = j_a is not None and j_b is not None

    pw[7]  = float(both_j and j_a == j_b)               # jersey_match
    pw[8]  = float(both_j and j_a != j_b)               # jersey_conflict
    pw[9]  = float(both_j and e_a < 0.15 and e_b < 0.15)  # jersey_both_confident

    # Helper: team stats
    def team_stats(t):
        ts = [x for x in t.pred_attributes.get("teams", [])
              if not (isinstance(x, float) and np.isnan(x))]
        if not ts:
            return None, 0.0
        mode = max(set(ts), key=ts.count)
        return mode, ts.count(mode) / len(ts)

    t_a, c_a = team_stats(fa)
    t_b, c_b = team_stats(fb)
    both_t = t_a is not None and t_b is not None

    pw[10] = float(both_t and t_a == t_b)                # team_match
    pw[11] = float(both_t and c_a > 0.9 and c_b > 0.9)  # team_both_consistent

    return pw


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SequencePairDataset(Dataset):
    """
    PyTorch Dataset for tracklet pair classification.

    Loads pair metadata and pre-extracts per-frame features from pickle
    caches into memory at initialization. __getitem__ performs temporal
    subsampling, frame dropout, padding, and optional A/B swap augmentation.
    """

    def __init__(
        self,
        pair_metadata: List[dict],
        cache_root: Path,
        t_max: int = 200,
        max_frame_value: float = 2000.0,
        training: bool = True,
        frame_dropout: float = 0.15,
        augment_swap: bool = True,
    ):
        """
        Args:
            pair_metadata: list of dicts with keys:
                sequence, tid_a, tid_b, pairwise (ndarray), label (int)
            cache_root: path to directory containing cache_attributes_*.pkl
            t_max: max sequence length (subsample longer tracklets)
            max_frame_value: normalization constant for absolute frame numbers
            training: enable augmentation
            frame_dropout: fraction of frames to drop during training
            augment_swap: enable A/B swap augmentation
        """
        self.t_max = t_max
        self.max_frame_value = max_frame_value
        self.training = training
        self.frame_dropout = frame_dropout
        self.augment_swap = augment_swap

        # Pre-extract features for all referenced tracklets
        self._feature_cache: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._load_features(pair_metadata, cache_root)

        # Store pair metadata (only keep what we need)
        self.pairs = []
        for p in pair_metadata:
            key_a = (p["sequence"], p["tid_a"])
            key_b = (p["sequence"], p["tid_b"])
            if key_a in self._feature_cache and key_b in self._feature_cache:
                self.pairs.append({
                    "key_a": key_a,
                    "key_b": key_b,
                    "pairwise": p["pairwise"],
                    "label": p["label"],
                })

    def _load_features(self, pair_metadata, cache_root):
        """Load pickle caches and extract features for all referenced tracklets."""
        # Collect needed (sequence, tid) keys
        needed = set()
        for p in pair_metadata:
            needed.add((p["sequence"], p["tid_a"]))
            needed.add((p["sequence"], p["tid_b"]))

        # Group by sequence to load each cache once
        by_seq: Dict[str, set] = {}
        for seq, tid in needed:
            by_seq.setdefault(seq, set()).add(tid)

        for seq, tids in by_seq.items():
            cache_path = Path(cache_root) / f"cache_attributes_{seq}.pkl"
            if not cache_path.exists():
                print(f"  [WARN] Cache not found: {cache_path}")
                continue
            with open(cache_path, "rb") as f:
                tracklets = pickle.load(f)
            for tid in tids:
                if tid in tracklets:
                    feats, frames = extract_frame_features(tracklets[tid])
                    self._feature_cache[(seq, tid)] = (feats, frames)

        print(f"  Loaded features for {len(self._feature_cache)} tracklets")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        feats_a, frames_a = self._feature_cache[p["key_a"]]
        feats_b, frames_b = self._feature_cache[p["key_b"]]

        tok_a, mask_a, pos_a, fn_a = self._prepare_sequence(feats_a, frames_a)
        tok_b, mask_b, pos_b, fn_b = self._prepare_sequence(feats_b, frames_b)

        pairwise = torch.tensor(p["pairwise"].copy(), dtype=torch.float32)
        label = torch.tensor(p["label"], dtype=torch.float32)

        # A/B swap augmentation
        if self.training and self.augment_swap and torch.rand(1).item() > 0.5:
            tok_a, tok_b = tok_b, tok_a
            mask_a, mask_b = mask_b, mask_a
            pos_a, pos_b = pos_b, pos_a
            fn_a, fn_b = fn_b, fn_a
            pairwise[PW_ENDPOINT_DX_IDX] = -pairwise[PW_ENDPOINT_DX_IDX]
            pairwise[PW_ENDPOINT_DY_IDX] = -pairwise[PW_ENDPOINT_DY_IDX]
            ratio = pairwise[PW_BBOX_HEIGHT_RATIO_IDX]
            if ratio != 0:
                pairwise[PW_BBOX_HEIGHT_RATIO_IDX] = 1.0 / ratio

        return tok_a, mask_a, pos_a, fn_a, tok_b, mask_b, pos_b, fn_b, pairwise, label

    def _prepare_sequence(
        self, features: np.ndarray, frames: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample, apply frame dropout, pad, and create position tensors."""
        n = len(frames)

        # Temporal subsampling if too long
        if n > self.t_max:
            indices = np.linspace(0, n - 1, self.t_max, dtype=int)
            features = features[indices]
            frames = frames[indices]
            n = self.t_max

        # Frame dropout during training (always keep first and last)
        if self.training and self.frame_dropout > 0 and n > 2:
            keep = np.random.rand(n) > self.frame_dropout
            keep[0] = True
            keep[-1] = True
            if keep.sum() >= 2:
                features = features[keep]
                frames = frames[keep]
                n = int(keep.sum())

        # Pad to t_max
        padded = np.zeros((self.t_max, RAW_DIM), dtype=np.float32)
        padded[:n] = features

        # Mask: True = padding (ignored by attention)
        mask = np.ones(self.t_max, dtype=bool)
        mask[:n] = False

        # Intra-tracklet positions (normalized 0..1)
        positions = np.zeros(self.t_max, dtype=np.float32)
        if n > 1:
            positions[:n] = np.arange(n, dtype=np.float32) / (n - 1)

        # Absolute frame numbers (normalized)
        frame_nums = np.zeros(self.t_max, dtype=np.float32)
        frame_nums[:n] = frames / self.max_frame_value

        return (
            torch.from_numpy(padded),
            torch.from_numpy(mask),
            torch.from_numpy(positions),
            torch.from_numpy(frame_nums),
        )


# ---------------------------------------------------------------------------
# Per-tracklet aggregate stats (mirrors XGBoost feature set)
# ---------------------------------------------------------------------------

# Indices into the per-frame feature array
_BBOX_CX   = BBOX_START + 0
_BBOX_CY   = BBOX_START + 1
_BBOX_H    = BBOX_START + 3
_ENTROPY   = SCALAR_START + 1   # jersey entropy
_TEAM      = SCALAR_START + 3   # team value
_JERSEY    = SCALAR_START + 0   # jersey value

# Entropy threshold below which a jersey prediction is considered reliable
_JERSEY_ENTROPY_THR = 0.2

# Number of per-tracklet stats (must match _tracklet_stats output length)
TRACKLET_STATS_DIM = 11


def _tracklet_stats(feats: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """
    Compute 11 per-tracklet aggregate statistics from raw per-frame features.

    Mirrors the XGBoost feature set for per-tracklet info that is not already
    captured in the 12-dim pairwise vector:
        0  duration          (end_frame - start_frame)
        1  n_frames
        2  start_cx          (normalised centre-x of first bbox)
        3  start_cy
        4  end_cx            (normalised centre-x of last bbox)
        5  end_cy
        6  mean_bbox_height
        7  jersey_coverage   (fraction of frames with reliable jersey)
        8  jersey_entropy_mean (mean entropy of reliable predictions)
        9  team_consistency  (fraction of frames matching mode team)
       10  team_coverage     (fraction of frames with valid team)
    """
    n = len(frames)
    stats = np.zeros(TRACKLET_STATS_DIM, dtype=np.float32)

    stats[0] = float(frames[-1] - frames[0])        # duration
    stats[1] = float(n)                              # n_frames
    stats[2] = float(feats[0,  _BBOX_CX])            # start_cx
    stats[3] = float(feats[0,  _BBOX_CY])            # start_cy
    stats[4] = float(feats[-1, _BBOX_CX])            # end_cx
    stats[5] = float(feats[-1, _BBOX_CY])            # end_cy
    stats[6] = float(feats[:, _BBOX_H].mean())       # mean_bbox_height

    # Jersey stats — only frames where entropy < threshold
    entropies = feats[:, _ENTROPY]
    reliable  = entropies < _JERSEY_ENTROPY_THR
    n_reliable = reliable.sum()
    stats[7] = float(n_reliable) / max(n, 1)         # jersey_coverage
    stats[8] = float(entropies[reliable].mean()) if n_reliable > 0 else 1.0

    # Team stats — frames where team != 0 (0 is the fill value for missing)
    team_vals = feats[:, _TEAM]
    valid_team = team_vals != 0.0
    n_valid = valid_team.sum()
    stats[10] = float(n_valid) / max(n, 1)           # team_coverage
    if n_valid > 0:
        tv = team_vals[valid_team]
        unique, counts = np.unique(tv, return_counts=True)
        stats[9] = float(counts.max()) / float(n_valid)  # team_consistency
    else:
        stats[9] = 0.0

    return stats


# Extended pairwise dim: original 12 + 11 stats for A + 11 stats for B
EXTENDED_PAIRWISE_DIM = 12 + 2 * TRACKLET_STATS_DIM   # = 34


class SequencePairDatasetExtended(SequencePairDataset):
    """
    Extends SequencePairDataset by appending per-tracklet aggregate statistics
    to the pairwise feature vector.

    The original 12-dim pairwise vector is kept unchanged; 11 stats for
    tracklet A and 11 for tracklet B are appended → 34-dim total.
    No data regeneration needed: stats are computed on-the-fly from the
    same feature cache used for the sequence tokens.
    """

    def __getitem__(self, idx):
        p = self.pairs[idx]
        feats_a, frames_a = self._feature_cache[p["key_a"]]
        feats_b, frames_b = self._feature_cache[p["key_b"]]

        tok_a, mask_a, pos_a, fn_a = self._prepare_sequence(feats_a, frames_a)
        tok_b, mask_b, pos_b, fn_b = self._prepare_sequence(feats_b, frames_b)

        stats_a = _tracklet_stats(feats_a, frames_a)
        stats_b = _tracklet_stats(feats_b, frames_b)
        extended_pw = np.concatenate([p["pairwise"], stats_a, stats_b])
        pairwise = torch.tensor(extended_pw, dtype=torch.float32)
        label    = torch.tensor(p["label"], dtype=torch.float32)

        # A/B swap augmentation — swap stats too
        if self.training and self.augment_swap and torch.rand(1).item() > 0.5:
            tok_a, tok_b = tok_b, tok_a
            mask_a, mask_b = mask_b, mask_a
            pos_a, pos_b = pos_b, pos_a
            fn_a, fn_b = fn_b, fn_a
            pairwise[PW_ENDPOINT_DX_IDX] = -pairwise[PW_ENDPOINT_DX_IDX]
            pairwise[PW_ENDPOINT_DY_IDX] = -pairwise[PW_ENDPOINT_DY_IDX]
            ratio = pairwise[PW_BBOX_HEIGHT_RATIO_IDX]
            if ratio != 0:
                pairwise[PW_BBOX_HEIGHT_RATIO_IDX] = 1.0 / ratio
            # Swap A/B stats blocks
            pairwise[12:23], pairwise[23:34] = pairwise[23:34].clone(), pairwise[12:23].clone()

        return tok_a, mask_a, pos_a, fn_a, tok_b, mask_b, pos_b, fn_b, pairwise, label
