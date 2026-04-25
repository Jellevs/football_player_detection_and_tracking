"""
Tracklet splitting entry point.

Configure which signals to use by editing SIGNALS below.  Any combination of
per-tracklet signals is supported.  Cross-tracklet signals are opt-in via
separate flags.

Per-tracklet signals (unified — all see the full original tracklet):
    'jersey'      jersey-number change
    'team'        team-colour change
    'bbox'        bounding-box velocity spike

Cross-tracklet signals (run as separate passes):
    'trajectory'  proximity + velocity-swap analysis (needs all tracklets)
    'gta'         DBSCAN embedding clustering

Examples
--------
    SIGNALS = ['jersey', 'team']               # current production config
    SIGNALS = ['jersey', 'team', 'bbox']       # add bbox anomaly
    SIGNALS = ['jersey']                       # jersey only
    SIGNALS = ['team', 'bbox']                 # team + bbox, no jersey
    SIGNALS = ['jersey', 'team', 'trajectory'] # unified + cross-tracklet traj
"""

from .splitters.modular_splitter import ModularSplitter
from .splitters.trajectory_splitter import TrajectorySplitter
from .splitters import gta_splitter

# ── configure signals here ────────────────────────────────────────────────────
# SIGNALS = ['jersey', 'bbox', 'team']

SIGNALS = ['jersey']
# ─────────────────────────────────────────────────────────────────────────────

_PER_TRACKLET   = {'jersey', 'team', 'bbox'}
_CROSS_TRACKLET = {'trajectory', 'gta'}


def split_tracklets(tracklets, splitter_cfg):
    signals        = [s.lower() for s in SIGNALS]
    per_tracklet   = [s for s in signals if s in _PER_TRACKLET]
    cross_tracklet = [s for s in signals if s in _CROSS_TRACKLET]

    label = '+'.join(signals) if signals else 'none'
    print(f"\n=== Splitting [{label}] ===")

    # ── per-tracklet signals (all unified in one pass) ────────────────
    if per_tracklet:
        splitter  = ModularSplitter(splitter_cfg, signals=per_tracklet)
        tracklets = _split_by_modular(tracklets, splitter)

    # ── cross-tracklet signals ────────────────────────────────────────
    if 'trajectory' in cross_tracklet:
        tracklets = split_by_trajectory(tracklets, TrajectorySplitter(splitter_cfg))

    if 'gta' in cross_tracklet:
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
            print(f"  Tracklet {track_id} → {len(fragments)} fragments ({splitter._signal_label})")
            for frag in fragments:
                new_tracklets[frag.track_id] = frag
                next_available_id = max(next_available_id, frag.track_id + 1)
        else:
            new_tracklets[tracklet.track_id] = tracklet

    print(f"  {split_count}/{len(tracklets)} tracklets split")
    return new_tracklets


def split_by_trajectory(tracklets, trajectory_splitter):
    return trajectory_splitter.split_all_tracklets(tracklets)


def split_by_gta(tracklets):
    print("\n=== Splitting by GTA (embedding clustering) ===")
    n_before = len(tracklets)
    new_tracklets = gta_splitter.split_tracklets(tracklets)
    print(f"  {n_before} → {len(new_tracklets)} tracklets ({len(new_tracklets) - n_before} new)")
    return new_tracklets
