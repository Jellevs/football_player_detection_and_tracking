"""
Modular splitter — combine any subset of per-tracklet signals.

Usage
-----
    splitter = ModularSplitter(config, signals=['jersey', 'team', 'bbox'])
    # or any combination: ['jersey'], ['team', 'bbox'], ['jersey', 'team'], …

All active per-tracklet signals (jersey, team, bbox) run on the ORIGINAL
unsplit tracklet and their switch indices are merged before a single split
is applied.  This avoids the window-truncation artefact of sequential
splitting (where the team persistence window gets clipped at a jersey-split
boundary).

Cross-tracklet signals (trajectory, gta) cannot be unified here because they
require global context.  They are handled as separate passes in
split_tracklets.py.

Supported signals
-----------------
  'jersey'     — jersey-number change (attribute-based, per-tracklet)
  'team'       — team-colour change   (attribute-based, per-tracklet)
  'bbox'       — bounding-box velocity spike (geometry-based, per-tracklet)
"""

from __future__ import annotations

import numpy as np
from typing import List, Sequence

from .unified_splitter import UnifiedSplitter
from .bbox_anomaly_splitter import BboxAnomalySplitter
from utils.config import SplitterConfig


VALID_SIGNALS = {'jersey', 'team', 'bbox'}


class ModularSplitter(UnifiedSplitter):
    """
    Drop-in replacement for UnifiedSplitter that accepts any combination of
    per-tracklet signals.  All signals see the full original tracklet before
    any split is applied.
    """

    def __init__(self, config: SplitterConfig = None, signals: Sequence[str] = ('jersey', 'team')):
        super().__init__(config)

        unknown = set(signals) - VALID_SIGNALS
        if unknown:
            raise ValueError(
                f"Unknown signals: {unknown}. Valid per-tracklet signals: {VALID_SIGNALS}. "
                "Cross-tracklet signals (trajectory, gta) are handled in split_tracklets.py."
            )

        self.signals = set(signals)
        self._signal_label = '+'.join(sorted(self.signals))

        if 'bbox' in self.signals:
            self._bbox = BboxAnomalySplitter(config)

    # ------------------------------------------------------------------
    # Public interface (overrides UnifiedSplitter.split_tracklet)
    # ------------------------------------------------------------------

    def split_tracklet(self, tracklet, next_available_id: int):
        n = len(tracklet.frames)
        if n == 0:
            return None

        all_switch_indices: set[int] = set()

        # ── attribute-based signals (jersey and/or team) ──────────────
        if 'jersey' in self.signals or 'team' in self.signals:
            jerseys, entropies, teams, team_confs = self._load_attributes(tracklet, n)
            indices = self._detect_switches(jerseys, entropies, teams, team_confs)
            all_switch_indices.update(indices)

        # ── bbox velocity signal ──────────────────────────────────────
        if 'bbox' in self.signals:
            indices = self._bbox_switch_indices(tracklet)
            all_switch_indices.update(indices)

        switch_indices = sorted(all_switch_indices)

        if not switch_indices:
            return None
        
        
        return self._create_fragments(tracklet, switch_indices, n, next_available_id)

    # ------------------------------------------------------------------
    # Attribute loading (nulls out inactive signals)
    # ------------------------------------------------------------------

    def _load_attributes(self, tracklet, n: int):
        attrs = tracklet.pred_attributes

        if 'jersey' in self.signals:
            jerseys   = list(attrs.get('jerseys', []))           + [None] * max(0, n - len(attrs.get('jerseys', [])))
            entropies = list(attrs.get('jersey_entropies', []))  + [1.0]  * max(0, n - len(attrs.get('jersey_entropies', [])))
        else:
            jerseys   = [None] * n
            entropies = [1.0]  * n

        if 'team' in self.signals:
            teams      = list(attrs.get('teams', []))      + [None] * max(0, n - len(attrs.get('teams', [])))
            team_confs = list(attrs.get('team_confs', [])) + [0.0]  * max(0, n - len(attrs.get('team_confs', [])))
        else:
            teams      = [None] * n
            team_confs = [0.0]  * n

        return jerseys, entropies, teams, team_confs

    # ------------------------------------------------------------------
    # Bbox velocity signal
    # ------------------------------------------------------------------

    def _bbox_switch_indices(self, tracklet) -> List[int]:
        min_frames = self._bbox.lookback_window + self._bbox.lookahead_window
        if len(tracklet.bboxes) < min_frames:
            return []
        velocities = self._bbox.calculate_velocities(tracklet.bboxes)
        return self._bbox.detect_velocity_spikes(velocities)
