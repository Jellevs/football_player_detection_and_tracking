"""
majority_merger.py, Ensemble merger using majority vote across multiple connectors.

Runs each sub-merger on a deep copy of the input tracklets, extracts which
original tracklets each merger placed in the same cluster (via merge logs and
union-find), and merges a pair only when at least ceil(N/2) mergers agree.

Usage:
    from tracklets.connectors.majority_merger import MajorityMerger

    majority_merger = MajorityMerger(
        mergers=[decision_merger, xgboost_merger, transformer_merger],
        log_path=r"merger_log_majority.csv",
    )
    merged = majority_merger.merge(copy.deepcopy(splitted_tracklets), sequence_name=sequence)
"""

import copy
import csv
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union


class MajorityMerger:
    def __init__(
        self,
        mergers: List,
        log_path: Optional[Union[Path, str]] = None,
    ):
        self.mergers = mergers
        self.quorum = math.ceil(len(mergers) / 2)
        self.log_path = Path(log_path) if log_path is not None else None

    def merge(self, tracklets: Dict, sequence_name: str = "") -> Dict:
        if len(tracklets) < 2:
            return tracklets

        orig_ids = sorted(tracklets.keys())

        # Run each sub-merger, extract clusters from merge logs
        cluster_maps = []
        for idx, merger in enumerate(self.mergers):
            tc = copy.deepcopy(tracklets)

            # Temporarily redirect the merger's log to a temp file
            # so we can read back which merges it performed.
            saved_log_path = getattr(merger, 'log_path', None)
            tmp_fd, tmp_name = tempfile.mkstemp(suffix='.csv')
            os.close(tmp_fd)
            os.unlink(tmp_name)  # delete so merger creates it fresh with header
            tmp_path = Path(tmp_name)
            merger.log_path = tmp_path

            merger.merge(tc, sequence_name=sequence_name)

            # Build cluster map from the merge log using union-find
            cmap = self._cluster_map_from_log(tmp_path, orig_ids)
            cluster_maps.append(cmap)

            # Clean up: restore original log path, remove temp file
            merger.log_path = saved_log_path
            if tmp_path.exists():
                tmp_path.unlink()

        # Majority vote on pairs
        majority_pairs = set()
        for i in range(len(orig_ids)):
            for j in range(i + 1, len(orig_ids)):
                a, b = orig_ids[i], orig_ids[j]
                votes = sum(
                    1 for cm in cluster_maps
                    if cm[a] == cm[b]
                )
                if votes >= self.quorum:
                    majority_pairs.add((a, b))

        # Build majority clusters via union-find
        parent = {}

        def find(x):
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for a, b in majority_pairs:
            union(a, b)

        groups = defaultdict(list)
        for tid in orig_ids:
            groups[find(tid)].append(tid)

        # Apply merges (same logic as XGBoostMerger._apply_merges)
        result = {}
        next_id = 1

        for root in sorted(groups.keys()):
            members = groups[root]
            members.sort(key=lambda x: tracklets[x].frames[0])

            base = tracklets[members[0]]
            current_frames = set(base.frames)

            for absorbed_id in members[1:]:
                other = tracklets[absorbed_id]
                if current_frames & set(other.frames):
                    # Frame overlap, keep separate
                    result[next_id] = other
                    next_id += 1
                    continue
                self._log_merge(sequence_name, members[0], absorbed_id, base, other)
                base.frames.extend(other.frames)
                base.bboxes.extend(other.bboxes)
                base.scores.extend(other.scores)
                base.embeddings.extend(other.embeddings)
                for k in base.pred_attributes:
                    base.pred_attributes[k].extend(other.pred_attributes.get(k, []))
                for k in base.gt_attributes:
                    base.gt_attributes[k].extend(other.gt_attributes.get(k, []))
                current_frames |= set(other.frames)

            result[next_id] = base
            next_id += 1

        return result

    # Cluster extraction via merge log (same approach as
    # generate_merger_comparison.py, known correct)

    @staticmethod
    def _cluster_map_from_log(log_path: Path, orig_ids) -> Dict:
        """Build a cluster map from a merger's log file using union-find.

        Each original tracklet ID maps to its root in the union-find tree.
        Two tracklets in the same cluster share the same root.  Tracklets
        not mentioned in the log are their own root (singleton clusters).
        """
        parent = {}

        def find(x):
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        if log_path.exists():
            with open(log_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    base = int(row['base_id'])
                    absorbed = int(row['absorbed_id'])
                    union(base, absorbed)

        return {tid: find(tid) for tid in orig_ids}

    # Logging

    def _log_merge(self, sequence_name, base_id, absorbed_id, base, absorbed):
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
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
                "MajorityMerger", sequence_name,
                base_id, absorbed_id,
                base.frames[0], base.frames[-1],
                absorbed.frames[0], absorbed.frames[-1],
            ])
