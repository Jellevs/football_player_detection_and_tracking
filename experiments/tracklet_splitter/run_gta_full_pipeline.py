"""
run_gta_full_pipeline.py — Run full GTA pipeline (split + merge) on baseline tracklets.

Loads baseline tracklets from tracked_detections cache, runs the GTA DBSCAN
split followed by hierarchical cosine-distance merge, and saves MOT files for
HOTA evaluation.

This script reimplements the original GTA pipeline flow from refine_tracklets.py,
fixing the xyxy/xywh bbox issue that exists in gta_splitter.py.

Key differences from gta_splitter.py (which cause result discrepancies):
  1. Spatial constraints computed on PRE-SPLIT tracklets (as in original GTA)
  2. Bbox centres computed correctly for xyxy format
  3. Uses tracked_detections cache directly (same as main.py baseline)

Outputs
-------
    evaluation/SNPT/gta_full/data/SNPT-XXX.txt

Usage
-----
    python experiments/tracklet_splitter/run_gta_full_pipeline.py
"""

from __future__ import annotations

import copy
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

# ── repo root importable ─────────────────────────────────────────────────────
REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))

from utils.data_utils import organize_detections_by_track

# ── configuration ─────────────────────────────────────────────────────────────

DATA_SPLIT = "test"

CACHE_DIR = REPO / "output" / "cache"
SEQMAP    = REPO / "evaluation" / "seqmaps" / f"SNPT-{DATA_SPLIT}.txt"
OUT_DIR   = REPO / "evaluation" / "SNPT" / "gta_full" / "data"

# GTA default parameters (from refine_tracklets.py argparse defaults)
EPS             = 0.7
MIN_SAMPLES     = 10
MAX_K           = 3
MIN_LEN         = 100
SPATIAL_FACTOR  = 1.0
MERGE_DIST_THRES = 0.4

# Set to True to skip the merge step (split-only, like run_gta_splitter_baseline.py)
SPLIT_ONLY = False


# ══════════════════════════════════════════════════════════════════════════════
#  GTA functions — faithful to refine_tracklets.py, adapted for xyxy bboxes
# ══════════════════════════════════════════════════════════════════════════════

def bbox_centre_xyxy(bbox):
    """Compute centre (cx, cy) from an xyxy bbox."""
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return (x1 + x2) / 2, (y1 + y2) / 2


def detect_id_switch(embs, eps=None, min_samples=None, max_clusters=None):
    """DBSCAN clustering to detect identity switches (unchanged from GTA)."""
    if len(embs) > 15000:
        embs = embs[1::2]

    embs = np.stack(embs)

    scaler = StandardScaler()
    embs_scaled = scaler.fit_transform(embs)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embs_scaled)
    labels = db.labels_

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]

    if -1 in labels and len(unique_labels) > 1:
        cluster_centers = np.array([embs_scaled[labels == label].mean(axis=0) for label in unique_labels])
        noise_indices = np.where(labels == -1)[0]
        for idx in noise_indices:
            distances = cdist([embs_scaled[idx]], cluster_centers, metric='cosine')
            nearest_cluster = np.argmin(distances)
            labels[idx] = list(unique_labels)[nearest_cluster]

    n_clusters = len(unique_labels)

    if max_clusters and n_clusters > max_clusters:
        while n_clusters > max_clusters:
            cluster_centers = np.array([embs_scaled[labels == label].mean(axis=0) for label in unique_labels])
            distance_matrix = cdist(cluster_centers, cluster_centers, metric='cosine')
            np.fill_diagonal(distance_matrix, np.inf)
            min_dist_idx = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
            cluster_to_merge_1 = unique_labels[min_dist_idx[0]]
            cluster_to_merge_2 = unique_labels[min_dist_idx[1]]
            labels[labels == cluster_to_merge_2] = cluster_to_merge_1
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels != -1]
            n_clusters = len(unique_labels)

    return n_clusters > 1, labels


def split_tracklets(tmp_trklets, eps=0.7, max_k=3, min_samples=10, len_thres=100):
    """Split tracklets via DBSCAN on ReID embeddings (same as GTA)."""
    from tracklets.tracklet import Tracklet

    new_id = max(tmp_trklets.keys()) + 1
    tracklets = {}

    for tid in tqdm(sorted(tmp_trklets.keys()), total=len(tmp_trklets), desc="GTA split"):
        trklet = tmp_trklets[tid]
        if len(trklet.frames) < len_thres:
            tracklets[tid] = trklet
        else:
            embs = np.stack(trklet.embeddings)
            frames = np.array(trklet.frames)
            bboxes = np.stack(trklet.bboxes) if isinstance(trklet.bboxes[0], np.ndarray) else np.array(trklet.bboxes)
            scores = np.array(trklet.scores)

            id_switch_detected, clusters = detect_id_switch(
                embs, eps=eps, min_samples=min_samples, max_clusters=max_k
            )

            if not id_switch_detected:
                tracklets[tid] = trklet
            else:
                unique_labels = set(clusters)
                for label in unique_labels:
                    if label == -1:
                        continue
                    mask = clusters == label
                    tracklets[new_id] = Tracklet(
                        new_id,
                        frames[mask].tolist(),
                        scores[mask].tolist(),
                        bboxes[mask].tolist(),
                        embeddings=embs[mask].tolist(),
                    )
                    new_id += 1

    return tracklets


def get_spatial_constraints_xyxy(tid2track, factor):
    """
    Spatial constraints for xyxy bboxes.
    Original GTA does x += w/2 which is correct for xywh.
    We compute centre directly from xyxy.
    """
    min_x = float('inf')
    max_x = -float('inf')
    min_y = float('inf')
    max_y = -float('inf')

    for track in tid2track.values():
        for bbox in track.bboxes:
            cx, cy = bbox_centre_xyxy(bbox)
            min_x = min(min_x, cx)
            max_x = max(max_x, cx)
            min_y = min(min_y, cy)
            max_y = max(max_y, cy)

    x_range = abs(max_x - min_x) * factor
    y_range = abs(max_y - min_y) * factor
    return x_range, y_range


def get_distance(track1_id, track2_id, track1, track2):
    """Cosine distance between two tracks (same as GTA, forced CPU)."""
    assert track1_id == track1.track_id and track2_id == track2.track_id

    if track1_id != track2_id:
        if set(track1.frames) & set(track2.frames):
            return 1.0

    device = torch.device("cpu")
    t1 = torch.tensor(np.stack(track1.embeddings), dtype=torch.float32, device=device)
    t2 = torch.tensor(np.stack(track2.embeddings), dtype=torch.float32, device=device)

    cos_sim_num = torch.matmul(t1, t2.T)
    norm1 = torch.norm(t1, p=2, dim=1, keepdim=True)
    norm2 = torch.norm(t2, p=2, dim=1, keepdim=True)
    cos_sim_den = torch.matmul(norm1, norm2.T)

    cos_dist = 1 - cos_sim_num / cos_sim_den
    return (cos_dist.sum() / (len(t1) * len(t2))).item()


def get_distance_matrix(tid2track):
    """Pairwise distance matrix (same as GTA)."""
    n = len(tid2track)
    Dist = np.zeros((n, n))
    items = list(tid2track.items())
    for i, (id1, t1) in enumerate(items):
        for j, (id2, t2) in enumerate(items):
            if j < i:
                Dist[i][j] = Dist[j][i]
            else:
                Dist[i][j] = get_distance(id1, id2, t1, t2)
    return Dist


def find_consecutive_segments(track_times):
    """Same as GTA."""
    segments = []
    start_index = 0
    end_index = 0
    for i in range(1, len(track_times)):
        if track_times[i] == track_times[end_index] + 1:
            end_index = i
        else:
            segments.append((start_index, end_index))
            start_index = i
            end_index = i
    segments.append((start_index, end_index))
    return segments


def query_subtracks(seg1, seg2, track1, track2):
    """Same as GTA."""
    subtracks = []
    while seg1 and seg2:
        s1_start, s1_end = seg1[0]
        s2_start, s2_end = seg2[0]

        subtrack_1 = track1.extract(s1_start, s1_end)
        subtrack_2 = track2.extract(s2_start, s2_end)

        s1_startFrame = track1.frames[s1_start]
        s2_startFrame = track2.frames[s2_start]

        if s1_startFrame < s2_startFrame:
            assert track1.frames[s1_end] <= s2_startFrame
            subtracks.append(subtrack_1)
            subtracks.append(subtrack_2)
        else:
            assert s1_startFrame >= track2.frames[s2_end]
            subtracks.append(subtrack_2)
            subtracks.append(subtrack_1)
        seg1.pop(0)
        seg2.pop(0)

    seg_remain = seg1 if seg1 else seg2
    track_remain = track1 if seg1 else track2
    while seg_remain:
        s_start, s_end = seg_remain[0]
        if (s_end - s_start) < 30:
            seg_remain.pop(0)
            continue
        subtracks.append(track_remain.extract(s_start, s_end))
        seg_remain.pop(0)

    return subtracks


def check_spatial_constraints_xyxy(trk_1, trk_2, max_x_range, max_y_range):
    """
    Spatial constraint check adapted for xyxy bboxes.
    Original GTA uses x + w/2 (correct for xywh); we use (x1+x2)/2.
    """
    inSpatialRange = True
    seg_1 = find_consecutive_segments(trk_1.frames)
    seg_2 = find_consecutive_segments(trk_2.frames)

    subtracks = query_subtracks(seg_1, seg_2, trk_1, trk_2)
    if not subtracks:
        return True

    subtrack_1st = subtracks.pop(0)
    while subtracks:
        subtrack_2nd = subtracks.pop(0)
        if subtrack_1st.parent_id == subtrack_2nd.parent_id:
            subtrack_1st = subtrack_2nd
            continue

        cx1, cy1 = bbox_centre_xyxy(subtrack_1st.bboxes[-1])
        cx2, cy2 = bbox_centre_xyxy(subtrack_2nd.bboxes[0])
        dx = abs(cx1 - cx2)
        dy = abs(cy1 - cy2)

        if dx > max_x_range or dy > max_y_range:
            inSpatialRange = False
            break
        else:
            subtrack_1st = subtrack_2nd

    return inSpatialRange


def merge_tracklets(tracklets, max_x_range, max_y_range, merge_dist_thres):
    """
    Hierarchical merge (same logic as GTA), with xyxy bbox handling.

    NOTE: In the original GTA, spatial constraints are computed on PRE-SPLIT
    tracklets.  max_x_range / max_y_range are passed in (not recomputed here).
    """
    print(f"  Computing distance matrix for {len(tracklets)} tracklets...", flush=True)
    Dist = get_distance_matrix(tracklets)

    idx2tid = {idx: tid for idx, tid in enumerate(tracklets.keys())}

    diagonal_mask = np.eye(Dist.shape[0], dtype=bool)
    non_diagonal_mask = ~diagonal_mask

    merge_count = 0
    while np.any(Dist[non_diagonal_mask] < merge_dist_thres):
        min_index = np.argmin(Dist[non_diagonal_mask])
        min_value = np.min(Dist[non_diagonal_mask])
        masked_indices = np.where(non_diagonal_mask)
        track1_idx = masked_indices[0][min_index]
        track2_idx = masked_indices[1][min_index]

        track1 = tracklets[idx2tid[track1_idx]]
        track2 = tracklets[idx2tid[track2_idx]]

        inSpatialRange = check_spatial_constraints_xyxy(track1, track2, max_x_range, max_y_range)

        if inSpatialRange:
            merge_count += 1
            # Merge track2 into track1
            track1.embeddings = track1.embeddings + track2.embeddings
            track1.frames = track1.frames + track2.frames
            track1.bboxes = track1.bboxes + track2.bboxes
            track1.scores = track1.scores + track2.scores

            tracklets[idx2tid[track1_idx]] = track1
            tracklets.pop(idx2tid[track2_idx])

            Dist = np.delete(Dist, track2_idx, axis=0)
            Dist = np.delete(Dist, track2_idx, axis=1)
            idx2tid = {idx: tid for idx, tid in enumerate(tracklets.keys())}

            for idx in range(Dist.shape[0]):
                Dist[track1_idx, idx] = get_distance(
                    idx2tid[track1_idx], idx2tid[idx],
                    tracklets[idx2tid[track1_idx]], tracklets[idx2tid[idx]]
                )
                Dist[idx, track1_idx] = Dist[track1_idx, idx]

            diagonal_mask = np.eye(Dist.shape[0], dtype=bool)
            non_diagonal_mask = ~diagonal_mask
        else:
            Dist[track1_idx, track2_idx] = merge_dist_thres
            Dist[track2_idx, track1_idx] = merge_dist_thres

    print(f"  Merged {merge_count} tracklet pairs")
    return tracklets


# ══════════════════════════════════════════════════════════════════════════════
#  MOT output
# ══════════════════════════════════════════════════════════════════════════════

def tracklets_to_mot(tracklets: dict, frame_offset: int = 1) -> list[str]:
    """Convert xyxy tracklets to MOT-format lines (xywh, 1-indexed frames)."""
    lines = []
    for tid, tracklet in tracklets.items():
        for frame, bbox, score in zip(tracklet.frames, tracklet.bboxes, tracklet.scores):
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            w = x2 - x1
            h = y2 - y1
            mot_frame = int(frame) + frame_offset
            lines.append((
                mot_frame, int(tid),
                f"{mot_frame},{int(tid)},{x1:.6f},{y1:.6f},{w:.6f},{h:.6f},{float(score):.6f},-1,-1,-1"
            ))
    lines.sort(key=lambda t: (t[0], t[1]))
    return [ln[2] for ln in lines]


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── discover sequences ────────────────────────────────────────────────
    if SEQMAP.exists():
        with SEQMAP.open() as f:
            sequences = [
                line.strip() for line in f
                if line.strip() and not line.strip().lower().startswith("name")
            ]
    else:
        sequences = sorted(
            p.stem.replace("cache_tracked_detections_", "")
            for p in CACHE_DIR.glob("cache_tracked_detections_SNPT-*.pkl")
        )

    print(f"Split: {DATA_SPLIT}  |  Sequences: {len(sequences)}")
    print(f"GTA params: eps={EPS}, min_samples={MIN_SAMPLES}, max_k={MAX_K}, "
          f"min_len={MIN_LEN}, merge_dist={MERGE_DIST_THRES}, spatial_factor={SPATIAL_FACTOR}")
    print(f"Mode: {'split only' if SPLIT_ONLY else 'split + merge'}")
    print(f"Output: {OUT_DIR}\n")

    total_in = total_split = total_out = 0
    skipped = []
    t0 = time.time()

    for seq in sequences:
        cache_path = CACHE_DIR / f"cache_tracked_detections_{seq}.pkl"

        if not cache_path.exists():
            skipped.append((seq, "no tracked_detections cache"))
            continue

        print(f"\n[{seq}] Loading tracked_detections...", end=" ", flush=True)
        with cache_path.open("rb") as f:
            tracked_detections = pickle.load(f)

        tracklets = organize_detections_by_track(tracked_detections)
        n_in = len(tracklets)

        # Check embeddings
        has_emb = sum(1 for t in tracklets.values() if len(t.embeddings) > 0)
        if has_emb == 0:
            print(f"WARNING: no embeddings, skipping")
            skipped.append((seq, "no embeddings"))
            continue

        print(f"{n_in} tracklets ({has_emb} with embeddings)")

        # ── Spatial constraints on PRE-SPLIT tracklets (as in original GTA) ──
        max_x_range, max_y_range = get_spatial_constraints_xyxy(tracklets, SPATIAL_FACTOR)

        # ── Split ─────────────────────────────────────────────────────────
        split_result = split_tracklets(
            tracklets, eps=EPS, min_samples=MIN_SAMPLES,
            max_k=MAX_K, len_thres=MIN_LEN,
        )
        n_split = len(split_result)
        print(f"  After split: {n_split} tracklets (+{n_split - n_in})")

        # ── Merge ─────────────────────────────────────────────────────────
        if SPLIT_ONLY:
            final = split_result
        else:
            print(f"  Merging...", flush=True)
            final = merge_tracklets(
                split_result,
                max_x_range=max_x_range,
                max_y_range=max_y_range,
                merge_dist_thres=MERGE_DIST_THRES,
            )
        n_out = len(final)
        print(f"  Final: {n_out} tracklets")

        total_in    += n_in
        total_split += n_split
        total_out   += n_out

        # ── Save MOT ─────────────────────────────────────────────────────
        mot_lines = tracklets_to_mot(final)
        out_path = OUT_DIR / f"{seq}.txt"
        with out_path.open("w") as f:
            f.write("\n".join(mot_lines))
            if mot_lines:
                f.write("\n")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.1f}s")
    print(f"  Sequences processed : {len(sequences) - len(skipped)}")
    print(f"  Sequences skipped   : {len(skipped)}")
    if skipped:
        for seq, reason in skipped[:5]:
            print(f"    {seq}: {reason}")
    print(f"  Total tracklets in  : {total_in}")
    print(f"  After split         : {total_split} (+{total_split - total_in})")
    print(f"  After merge         : {total_out}")
    print(f"\nMOT files saved to: {OUT_DIR}")
    print(f"Run HOTA evaluation on 'gta_full' to get the score.")


if __name__ == "__main__":
    main()
