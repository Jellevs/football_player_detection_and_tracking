"""
Shared feature extraction utilities for transformer tracklet mergers.

Provides per-frame feature extraction and pairwise feature computation
used by data generation scripts and dataset classes.

Feature layout (per frame, 1293 dim total):
    [0:512)      ReID embedding (L2 normalized)
    [512:1280)   SigLIP embedding (L2 normalized, 768 dim)
    [1280:1287)  BBox derived: cx, cy, w, h, aspect, area, cy_norm
    [1287:1292)  Scalars: jersey, entropy, jersey_conf, team, team_conf
    [1292:1293)  Detection score

Pairwise features (12 dim):
    0  temporal_gap
    1  spatial_distance
    2  endpoint_dx
    3  endpoint_dy
    4  bbox_height_ratio
    5  reid_cosine_sim
    6  siglip_cosine_sim
    7  jersey_match
    8  jersey_conflict
    9  jersey_both_confident
    10 team_match
    11 team_both_consistent
"""

import numpy as np
from typing import Tuple


# ---- Feature layout constants ----
REID_START    = 0
REID_END      = 512
SIGLIP_START  = 512
SIGLIP_END    = 1280   # 512 + 768
BBOX_START    = 1280
BBOX_END      = 1287
SCALAR_START  = 1287
SCALAR_END    = 1292   # jersey, entropy, jersey_conf, team, team_conf
SCORE_START   = 1292
SCORE_END     = 1293
RAW_DIM       = 1293

SIGLIP_DIM    = 768

# Pairwise feature indices that need adjustment on A/B swap
PW_ENDPOINT_DX_IDX       = 2
PW_ENDPOINT_DY_IDX       = 3
PW_BBOX_HEIGHT_RATIO_IDX = 4

# Team confidence threshold for pairwise features
_TEAM_CONF_THR = 0.6

# Number of pairwise features
PAIRWISE_DIM = 12


# ---------------------------------------------------------------------------
# Feature extraction from Tracklet objects
# ---------------------------------------------------------------------------

def extract_frame_features(tracklet) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract per frame features from a Tracklet object.

    Returns:
        features: (n_frames, 525) float32 array
        frames:   (n_frames,) float32 array of frame indices
    """
    n = len(tracklet.frames)
    features = np.zeros((n, RAW_DIM), dtype=np.float32)

    jerseys        = tracklet.pred_attributes.get("jerseys", [])
    entropies      = tracklet.pred_attributes.get("jersey_entropies", [])
    confs          = tracklet.pred_attributes.get("jersey_confs_mean", [])
    teams          = tracklet.pred_attributes.get("teams", [])
    team_confs     = tracklet.pred_attributes.get("team_confs", [])
    siglip_embeds  = tracklet.pred_attributes.get("siglip_embeddings", [])

    for i in range(n):
        # ReID embedding (512 dim, L2 normalized)
        if tracklet.embeddings and i < len(tracklet.embeddings):
            emb = np.array(tracklet.embeddings[i], dtype=np.float32)
            norm = np.linalg.norm(emb) + 1e-6
            features[i, REID_START:REID_END] = emb / norm

        # SigLIP embedding (768 dim, L2 normalized)
        if i < len(siglip_embeds):
            sig = np.array(siglip_embeds[i], dtype=np.float32)
            if np.any(sig != 0):
                sig_norm = np.linalg.norm(sig) + 1e-6
                features[i, SIGLIP_START:SIGLIP_END] = sig / sig_norm

        # BBox derived features (7 dim)
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

        # Scalar features (5 dim): jersey, entropy, jersey_conf, team, team_conf
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

        # Detection score (1 dim)
        score = float(tracklet.scores[i]) if i < len(tracklet.scores) else 0.0
        features[i, SCORE_START:SCORE_END] = score

    return features, np.array(tracklet.frames, dtype=np.float32)


# ---------------------------------------------------------------------------
# Pairwise features (11 dim, no SigLIP)
# ---------------------------------------------------------------------------

def compute_pairwise_features(tracklet_a, tracklet_b) -> np.ndarray:
    """
    Compute 12 dim pairwise feature vector.

    Order: temporal_gap, spatial_distance, endpoint_dx, endpoint_dy,
           bbox_height_ratio, reid_cosine_sim, siglip_cosine_sim,
           jersey_match, jersey_conflict, jersey_both_confident,
           team_match, team_both_consistent
    """
    pw = np.zeros(PAIRWISE_DIM, dtype=np.float32)
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

    # 1 to 3: spatial_distance, endpoint_dx, endpoint_dy
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
    sig_a_all = fa.pred_attributes.get("siglip_embeddings", [])
    sig_b_all = fb.pred_attributes.get("siglip_embeddings", [])
    valid_sig_a = [s for s in sig_a_all if np.any(np.array(s) != 0)]
    valid_sig_b = [s for s in sig_b_all if np.any(np.array(s) != 0)]
    if valid_sig_a and valid_sig_b:
        sig_mean_a = np.stack(valid_sig_a).astype(np.float32).mean(axis=0)
        sig_mean_b = np.stack(valid_sig_b).astype(np.float32).mean(axis=0)
        na_s = np.linalg.norm(sig_mean_a) + 1e-6
        nb_s = np.linalg.norm(sig_mean_b) + 1e-6
        pw[6] = float(np.dot(sig_mean_a / na_s, sig_mean_b / nb_s))

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

    pw[7]  = float(both_j and j_a == j_b)                 # jersey_match
    pw[8]  = float(both_j and j_a != j_b)                 # jersey_conflict
    pw[9]  = float(both_j and e_a < 0.15 and e_b < 0.15)  # jersey_both_confident

    # Helper: team stats (only count predictions with sufficient confidence)
    def team_stats(t):
        ts  = t.pred_attributes.get("teams", [])
        tcs = t.pred_attributes.get("team_confs", [])
        valid = [
            ts[i] for i in range(len(ts))
            if not (isinstance(ts[i], float) and np.isnan(ts[i]))
            and i < len(tcs)
            and tcs[i] is not None
            and not (isinstance(tcs[i], float) and np.isnan(tcs[i]))
            and float(tcs[i]) >= _TEAM_CONF_THR
        ]
        if not valid:
            return None, 0.0
        mode = max(set(valid), key=valid.count)
        return mode, valid.count(mode) / len(valid)

    t_a, c_a = team_stats(fa)
    t_b, c_b = team_stats(fb)
    both_t = t_a is not None and t_b is not None

    pw[10] = float(both_t and t_a == t_b)                  # team_match
    pw[11] = float(both_t and c_a > 0.9 and c_b > 0.9)    # team_both_consistent

    return pw
