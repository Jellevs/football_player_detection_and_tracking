"""
Tracklet splitting entry point.

Configure which signals to use by editing SIGNALS below.  Any combination of
per-tracklet signals is supported.  Cross-tracklet signals are opt-in via
separate flags.

Per-tracklet signals (unified, all see the full original tracklet):
    'jersey'        jersey-number change
    'team'          team-colour change
    'bbox'          bounding-box velocity spike

Cross-tracklet signals (run as separate passes):
    'temporal_reid' ReID embedding comparison across temporal gaps
    'trajectory'    proximity + velocity-swap analysis (needs all tracklets)
    'gta'           DBSCAN embedding clustering

Per-tracklet signals are always grouped into a single unified pass so they
all see the full original tracklet before any split is applied.
The position of 'temporal_reid' in the list controls whether it runs before
or after the unified per-tracklet pass:
    ['temporal_reid', 'jersey', 'team']  →  STR first, then jersey+team
    ['jersey', 'team', 'temporal_reid']  →  jersey+team first, then STR
'trajectory' and 'gta' always run last (after all other signals).

Examples
--------
    SIGNALS = ['jersey', 'team']                        # attribute splitters only
    SIGNALS = ['jersey', 'team', 'bbox']                # add bbox anomaly
    SIGNALS = ['temporal_reid', 'jersey', 'team']       # STR first, then attributes
    SIGNALS = ['jersey', 'team', 'temporal_reid']       # attributes first, then STR
    SIGNALS = ['temporal_reid']                         # STR only
    SIGNALS = ['jersey', 'team', 'trajectory']          # attributes + trajectory
"""

from .splitters.modular_splitter import ModularSplitter
from .splitters.trajectory_splitter import TrajectorySplitter
from .splitters.temporal_reid_splitter import TemporalReIDSplitter
from .splitters import gta_splitter

SIGNALS = ["temporal_reid", "jersey", "team", "bbox"]

_PER_TRACKLET   = {'jersey', 'team', 'bbox'}
_CROSS_TRACKLET = {'trajectory', 'gta', 'temporal_reid'}


def split_tracklets(tracklets, splitter_config):
    signals      = [s.lower() for s in SIGNALS]
    per_tracklet = [s for s in signals if s in _PER_TRACKLET]

    label = '+'.join(signals) if signals else 'none'
    print(f"Splitting [{label}]")

    # Determine whether temporal_reid runs before or after the per-tracklet
    # unified pass based on its position in SIGNALS relative to any
    # per-tracklet signal.  If no per-tracklet signals are present, position
    # doesn't matter, temporal_reid just runs when encountered.
    first_per_tracklet_idx = next(
        (i for i, s in enumerate(signals) if s in _PER_TRACKLET), len(signals)
    )
    temporal_reid_idx = next(
        (i for i, s in enumerate(signals) if s == 'temporal_reid'), None
    )
    temporal_reid_before = (
        temporal_reid_idx is not None and temporal_reid_idx < first_per_tracklet_idx
    )

    # ── temporal reid BEFORE per-tracklet (if placed earlier in SIGNALS) ──
    if 'temporal_reid' in signals and temporal_reid_before:
        tracklets = split_by_temporal_reid(tracklets, splitter_config)

    # ── per-tracklet signals (all unified in one pass) ────────────────
    if per_tracklet:
        splitter  = ModularSplitter(splitter_config, signals=per_tracklet)
        tracklets = _split_by_modular(tracklets, splitter)

    # ── temporal reid AFTER per-tracklet (if placed later in SIGNALS) ─
    if 'temporal_reid' in signals and not temporal_reid_before:
        tracklets = split_by_temporal_reid(tracklets, splitter_config)

    # ── trajectory and gta always run last ───────────────────────────
    if 'trajectory' in signals:
        tracklets = split_by_trajectory(tracklets, TrajectorySplitter(splitter_config))

    if 'gta' in signals:
        tracklets = split_by_gta(tracklets)

    return tracklets


# ── helpers ───────────────────────────────────────────────────────────────────

def _split_by_modular(tracklets, splitter):
    max_existing_id   = max(tracklets.keys()) if tracklets else 0
    next_available_id = max_existing_id + 1

    new_tracklets = {}
    split_count   = 0

    for track_id, tracklet in tracklets.items():
        fragments = splitter.split_tracklet(tracklet, next_available_id)

        if fragments:
            split_count += 1
            print(f"Tracklet {track_id} → {len(fragments)} fragments ({splitter._signal_label})")
            for frag in fragments:
                new_tracklets[frag.track_id] = frag
                next_available_id = max(next_available_id, frag.track_id + 1)
        else:
            new_tracklets[tracklet.track_id] = tracklet

    print(f"{split_count}/{len(tracklets)} tracklets split")
    return new_tracklets


def split_by_temporal_reid(tracklets, splitter_config):
    splitter = TemporalReIDSplitter(
        min_gap_frames=splitter_config.temporal_reid_min_gap_frames,
        reid_threshold=splitter_config.temporal_reid_threshold,
        min_segment_frames=splitter_config.temporal_reid_min_segment_frames,
        n_samples=splitter_config.temporal_reid_n_samples,
    )
    n_before = len(tracklets)
    new_tracklets = splitter.split_all(tracklets)
    print(f"{n_before} → {len(new_tracklets)} tracklets ({len(new_tracklets) - n_before} new)")
    return new_tracklets


def split_by_trajectory(tracklets, trajectory_splitter):
    return trajectory_splitter.split_all_tracklets(tracklets)


def split_by_gta(tracklets):
    print("Splitting by GTA (embedding clustering)")
    n_before = len(tracklets)
    new_tracklets = gta_splitter.split_tracklets(tracklets)
    print(f"{n_before} → {len(new_tracklets)} tracklets ({len(new_tracklets) - n_before} new)")
    return new_tracklets
