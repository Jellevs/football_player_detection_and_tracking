import numpy as np
from scipy.spatial.distance import cdist


class TemporalReIDSplitter:
    """
    Splits tracklets at temporal gaps where ReID embeddings indicate different identities.

    For each temporal gap (discontinuity in frame indices), compares the mean ReID embedding
    of the segment before the gap with the segment after. If the cosine distance exceeds
    a threshold, splits the tracklet at that gap.
    """

    def __init__(self, min_gap_frames=10, reid_threshold=0.3, min_segment_frames=5, n_samples=20):
        """
        Args:
            min_gap_frames: Minimum gap size (in frames) to consider as a split candidate.
            reid_threshold: Cosine distance threshold between segment means. Above this = different identity.
            min_segment_frames: Minimum number of detections in a segment to keep it.
            n_samples: Number of embeddings to sample from each side of a gap for distance computation.
        """
        self.min_gap_frames = min_gap_frames
        self.reid_threshold = reid_threshold
        self.min_segment_frames = min_segment_frames
        self.n_samples = n_samples

    def split_tracklet(self, tracklet, next_available_id):
        """
        Split a single tracklet at temporal gaps where ReID distance exceeds threshold.

        Returns:
            List of fragment Tracklets, or None if no split needed.
        """
        frames = np.array(tracklet.frames)
        tid = tracklet.track_id

        if len(frames) < 2 * self.min_segment_frames:
            if tid == 17:
                print(f"  [DEBUG] Tracklet 17 skipped: too few frames ({len(frames)} < {2 * self.min_segment_frames})")
            return None

        has_embs = tracklet.embeddings and len(tracklet.embeddings) > 0
        if tid == 17:
            print(f"  [DEBUG] Tracklet 17: {len(frames)} frames, has_embeddings={has_embs}, n_embeddings={len(tracklet.embeddings) if has_embs else 0}")

        if not tracklet.embeddings or len(tracklet.embeddings) != len(frames):
            if tid == 17:
                print(f"  [DEBUG] Tracklet 17 skipped: embeddings mismatch (embs={len(tracklet.embeddings) if tracklet.embeddings else 0} vs frames={len(frames)})")
            return None

        embs = np.stack(tracklet.embeddings)

        # Find temporal gaps
        diffs = np.diff(frames)
        gap_indices = np.where(diffs >= self.min_gap_frames)[0]

        if tid == 17:
            print(f"  [DEBUG] Tracklet 17: frame range {frames[0]}-{frames[-1]}, gaps found: {len(gap_indices)}")
            if len(gap_indices) > 0:
                for gi in gap_indices:
                    print(f"    Gap: frame {frames[gi]} -> {frames[gi+1]} (delta={diffs[gi]})")

        if len(gap_indices) == 0:
            return None

        # Check each gap: compare embeddings before and after
        split_points = []
        for gap_idx in gap_indices:
            # Sample embeddings from each side of the gap
            before_start = max(0, gap_idx + 1 - self.n_samples)
            before_embs = embs[before_start:gap_idx + 1]

            after_end = min(len(embs), gap_idx + 1 + self.n_samples)
            after_embs = embs[gap_idx + 1:after_end]

            if len(before_embs) == 0 or len(after_embs) == 0:
                continue

            # Compute cosine distance between segment means
            mean_before = before_embs.mean(axis=0, keepdims=True)
            mean_after = after_embs.mean(axis=0, keepdims=True)
            cos_dist = cdist(mean_before, mean_after, metric='cosine')[0, 0]

            if tid == 17:
                print(f"    Cosine dist at gap: {cos_dist:.4f} (threshold={self.reid_threshold}), before_samples={len(before_embs)}, after_samples={len(after_embs)}")

            if cos_dist > self.reid_threshold:
                split_points.append(gap_idx + 1)  # Split index: first frame of the new segment

        if not split_points:
            return None

        # Create fragments from split points
        boundaries = [0] + split_points + [len(frames)]
        fragments = []
        current_id = next_available_id

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1] - 1  # inclusive, as extract() expects

            if (end - start + 1) < self.min_segment_frames:
                continue

            fragment = tracklet.extract(start, end)
            fragment.track_id = current_id
            fragment.parent_id = tracklet.parent_id
            fragments.append(fragment)
            current_id += 1

        if len(fragments) <= 1:
            return None

        return fragments

    def split_all(self, tracklets):
        """
        Apply temporal gap + ReID splitting to all tracklets.

        Args:
            tracklets: Dict of {track_id: Tracklet}

        Returns:
            New dict of tracklets after splitting.
        """
        max_existing_id = max(tracklets.keys()) if tracklets else 0
        next_available_id = max_existing_id + 1

        new_tracklets = {}
        split_count = 0

        for track_id, tracklet in tracklets.items():
            fragments = self.split_tracklet(tracklet, next_available_id)

            if fragments:
                split_count += 1
                frag_ids = [f.track_id for f in fragments]
                print(f"  Tracklet {track_id} split into {len(fragments)} fragments (temporal gap + ReID): {frag_ids}")

                for fragment in fragments:
                    new_tracklets[fragment.track_id] = fragment
                    next_available_id = max(next_available_id, fragment.track_id + 1)
            else:
                new_tracklets[tracklet.track_id] = tracklet

        print(f"Temporal gap + ReID splitting: {split_count}/{len(tracklets)} tracklets split")
        return new_tracklets
