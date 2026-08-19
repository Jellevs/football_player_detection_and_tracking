"""
oracle_merger.py

Oracle (ground-truth) merger that makes perfect merge decisions.

For each split tracklet, it matches detections to ground truth annotations
via IoU, assigns the majority GT identity, and merges all tracklets that
share the same GT identity (respecting temporal overlap constraints).

This establishes the performance ceiling for the merge step:
  - Remaining HOTA errors after the oracle are caused by detection,
    tracking, or splitting mistakes, NOT by the merger.

Usage in main.py:
    from tracklets.connectors.oracle_merger import OracleMerger

    merger = OracleMerger(gt_root=settings.DATA_ROOT)
    merged_tracklets = merger.merge(splitted_tracklets, sequence_name="SNMOT-...")
"""

import csv
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, Optional, Union


class OracleMerger:
    def __init__(
        self,
        gt_root: Path,
        iou_threshold: float = 0.5,
        log_path: Optional[Union[Path, str]] = None,
    ):
        """
        Args:
            gt_root: Root directory containing GT annotations.
                     Tries: gt_root/seq/gt/gt.txt, gt_root/seq/gt.txt, gt_root/seq.txt
            iou_threshold: Minimum IoU to consider a detection matched to GT.
        """
        self.gt_root = Path(gt_root)
        self.iou_threshold = iou_threshold
        self.log_path = Path(log_path) if log_path is not None else None
        self.gt_assignments: Dict[int, int] = {}  # populated after each merge() call: tracklet_id -> gt_id

    # Public interface

    def merge(self, tracklets: Dict, sequence_name: str) -> Dict:
        """
        Merge tracklets using ground truth identity assignments.

        Args:
            tracklets: {track_id: Tracklet} from the splitter
            sequence_name: Name of the sequence (to find GT file)
        Returns:
            Merged tracklets dict
        """
        if len(tracklets) < 2:
            return tracklets

        # Load ground truth
        gt_by_frame = self._load_gt(sequence_name)
        if gt_by_frame is None:
            print(f"[OracleMerger] WARNING: No GT found for {sequence_name}, returning unmerged")
            return tracklets

        # Assign each tracklet a GT identity
        tracklet_ids = sorted(tracklets.keys())
        gt_assignments = {}
        unmatched = []

        for tid in tracklet_ids:
            gt_id = self._assign_gt_identity(tracklets[tid], gt_by_frame)
            if gt_id is not None:
                gt_assignments[tid] = gt_id
            else:
                unmatched.append(tid)

        # Group tracklets by GT identity
        gt_groups = defaultdict(list)
        for tid, gt_id in gt_assignments.items():
            gt_groups[gt_id].append(tid)

        # Stats
        n_groups_multi = sum(1 for g in gt_groups.values() if len(g) > 1)
        n_merges = sum(len(g) - 1 for g in gt_groups.values() if len(g) > 1)
        print(f"[OracleMerger] {len(tracklet_ids)} tracklets -> "
              f"{len(gt_groups)} GT identities, "
              f"{n_groups_multi} groups need merging ({n_merges} merges), "
              f"{len(unmatched)} unmatched")

        self.gt_assignments = gt_assignments

        # Apply merges
        return self._apply_merges(tracklets, gt_groups, unmatched, sequence_name)

    # GT loading

    def _load_gt(self, sequence_name: str) -> Optional[Dict]:
        """Load GT annotations as {frame: [(gt_id, bbox_xywh), ...]}."""
        gt_path = self._resolve_gt_path(sequence_name)
        if gt_path is None:
            return None

        by_frame = defaultdict(list)
        with gt_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 6:
                    continue
                frame = int(float(parts[0]))
                gt_id = int(float(parts[1]))
                bbox = np.array([float(parts[2]), float(parts[3]),
                                 float(parts[4]), float(parts[5])], dtype=np.float32)
                by_frame[frame].append((gt_id, bbox))

        return dict(by_frame)

    def _resolve_gt_path(self, seq: str) -> Optional[Path]:
        """Find GT file for a sequence."""
        for candidate in [
            self.gt_root / seq / "gt" / "gt.txt",
            self.gt_root / seq / "gt.txt",
            self.gt_root / f"{seq}.txt",
        ]:
            if candidate.exists():
                return candidate
        return None

    # GT identity assignment

    def _assign_gt_identity(self, tracklet, gt_by_frame) -> Optional[int]:
        """
        Assign a GT identity to a tracklet by majority IoU matching.

        For each frame in the tracklet, find the GT bbox with highest IoU.
        The GT ID that appears most often is the tracklet's identity.
        """
        gt_id_votes = Counter()

        for frame, bbox_xyxy in zip(tracklet.frames, tracklet.bboxes):
            if frame not in gt_by_frame:
                continue

            # Convert tracklet bbox from xyxy to xywh for IoU computation
            x1, y1, x2, y2 = bbox_xyxy
            det_xywh = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)

            # Find best matching GT
            best_iou = 0.0
            best_gt_id = None
            for gt_id, gt_bbox_xywh in gt_by_frame[frame]:
                iou = self._iou_xywh(det_xywh, gt_bbox_xywh)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_id = gt_id

            if best_gt_id is not None and best_iou >= self.iou_threshold:
                gt_id_votes[best_gt_id] += 1

        if not gt_id_votes:
            return None

        # Return the GT ID with the most votes
        return gt_id_votes.most_common(1)[0][0]

    # Merge application

    def _apply_merges(self, tracklets, gt_groups, unmatched, sequence_name=""):
        """Merge tracklets that share the same GT identity."""
        result = {}
        overflow_id = max(tracklets.keys()) + 1000  # safe offset

        # Merge each GT group
        for gt_id, members in gt_groups.items():
            members.sort(key=lambda x: tracklets[x].frames[0])

            base = tracklets[members[0]]
            current_frames = set(base.frames)

            for next_id in members[1:]:
                other = tracklets[next_id]
                # Respect temporal overlap, can't merge if they exist in same frames
                if current_frames & set(other.frames):
                    result[overflow_id] = other
                    overflow_id += 1
                    continue

                self._log_merge(sequence_name, gt_id, members[0], next_id, base, other)
                base.frames.extend(other.frames)
                base.bboxes.extend(other.bboxes)
                base.scores.extend(other.scores)
                base.embeddings.extend(other.embeddings)
                for k in base.pred_attributes:
                    base.pred_attributes[k].extend(other.pred_attributes.get(k, []))
                for k in base.gt_attributes:
                    base.gt_attributes[k].extend(other.gt_attributes.get(k, []))
                current_frames |= set(other.frames)

            result[members[0]] = base

        # Keep unmatched tracklets as-is
        for tid in unmatched:
            result[tid] = tracklets[tid]

        return result

    def _log_merge(self, sequence_name, gt_id, base_id, absorbed_id, base, absorbed):
        if self.log_path is None:
            return
        write_header = not self.log_path.exists()
        with self.log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "merger_type", "sequence", "gt_id",
                    "base_id", "absorbed_id",
                    "base_start_frame", "base_end_frame",
                    "absorbed_start_frame", "absorbed_end_frame",
                ])
            writer.writerow([
                "OracleMerger", sequence_name, gt_id,
                base_id, absorbed_id,
                base.frames[0], base.frames[-1],
                absorbed.frames[0], absorbed.frames[-1],
            ])

    # Geometry

    @staticmethod
    def _iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
        """IoU between two xywh bboxes."""
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)

        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return float(inter / union) if union > 0 else 0.0
