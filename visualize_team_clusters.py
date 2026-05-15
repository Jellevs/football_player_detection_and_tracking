"""
visualize_team_clusters.py
==========================
Visualise the SigLIP → UMAP → KMeans team-clustering pipeline for one sequence.

Loads the attribute cache (no GPU needed), then:
  1. Collects all per-frame SigLIP embeddings + team labels
  2. Runs UMAP to 2-D *for display only*
  3. Plots four panels:
       A. UMAP 2-D coloured by predicted team   (what KMeans actually decided)
       B. UMAP 2-D coloured by cluster confidence  (how sure the model is)
       C. UMAP 2-D coloured by jersey number     (sanity: same jersey = same team?)
       D. Alternative: GMM fit on the same UMAP projection
          (handles non-spherical / unequal-variance clusters better than KMeans)

Also prints:
  - Silhouette score for KMeans vs GMM
  - Per-tracklet team consistency
  - Warning if the two clusters look highly non-spherical (KMeans might be wrong)

Usage:
    python visualize_team_clusters.py --seq SNGS-060
    python visualize_team_clusters.py --seq SNGS-060 --all_sequences
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import umap
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

# REPO_ROOT = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(REPO_ROOT))
import settings

CACHE_DIR  = settings.OUTPUT_ROOT / "cache"
OUTPUT_DIR = Path(__file__).parent / "output" / "team_clusters"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEAM_COLORS = {0: "#e05c5c", 1: "#5c8de0", -1: "#aaaaaa"}   # red / blue / grey
TEAM_LABELS = {0: "Team 0", 1: "Team 1", -1: "unknown"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cache(seq: str) -> Optional[Dict]:
    path = CACHE_DIR / f"cache_attributes_{seq}.pkl"
    if not path.exists():
        print(f"[WARN] No cache for {seq}: {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def collect_embeddings(tracklets: Dict) -> Tuple[
    np.ndarray,   # embeddings  (N, 768)
    np.ndarray,   # team labels (N,)
    np.ndarray,   # confidences (N,)
    np.ndarray,   # jersey nums (N,)  nan if unknown
    np.ndarray,   # track_ids   (N,)
]:
    embs, teams, confs, jerseys, tids = [], [], [], [], []

    for tid, tracklet in tracklets.items():
        siglip  = tracklet.pred_attributes.get("siglip_embeddings", [])
        t_list  = tracklet.pred_attributes.get("teams", [])
        c_list  = tracklet.pred_attributes.get("team_confs", [])
        j_list  = tracklet.pred_attributes.get("jerseys", [])

        for i, emb in enumerate(siglip):
            emb_arr = np.asarray(emb, dtype=np.float32)
            if not np.any(emb_arr != 0):     # skip zero-padded frames
                continue

            team = t_list[i] if i < len(t_list) else np.nan
            conf = c_list[i] if i < len(c_list) else np.nan
            jnum = j_list[i] if i < len(j_list) else np.nan

            # Convert nan team to -1 for plotting
            if team is None or (isinstance(team, float) and np.isnan(team)):
                team = -1
            if conf is None or (isinstance(conf, float) and np.isnan(conf)):
                conf = 0.0

            embs.append(emb_arr)
            teams.append(int(team))
            confs.append(float(conf))
            jerseys.append(float(jnum) if not (isinstance(jnum, float) and np.isnan(jnum)) else np.nan)
            tids.append(tid)

    return (
        np.stack(embs),
        np.array(teams, dtype=int),
        np.array(confs, dtype=np.float32),
        np.array(jerseys, dtype=np.float32),
        np.array(tids, dtype=int),
    )


# ---------------------------------------------------------------------------
# Clustering helpers
# ---------------------------------------------------------------------------

def fit_gmm(projections: np.ndarray, n_components: int = 2) -> np.ndarray:
    gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=5,
                          covariance_type="full")
    return gmm.fit_predict(projections)


def cluster_quality(projections: np.ndarray, labels: np.ndarray, name: str) -> float:
    """Print + return silhouette score (skips if only 1 unique label)."""
    unique = np.unique(labels[labels >= 0])
    if len(unique) < 2:
        print(f"  {name}: only 1 cluster — silhouette N/A")
        return 0.0
    mask  = labels >= 0
    score = silhouette_score(projections[mask], labels[mask])
    print(f"  {name} silhouette score: {score:.4f}  (higher = more separated)")
    return score


def non_sphericity_warning(projections: np.ndarray, labels: np.ndarray) -> None:
    """Warn if one cluster has significantly larger variance than the other."""
    for lab in [0, 1]:
        pts = projections[labels == lab]
        if len(pts) < 5:
            continue
        cov  = np.cov(pts.T)
        eigs = np.linalg.eigvalsh(cov)
        ratio = float(eigs[-1] / (eigs[0] + 1e-8))
        if ratio > 10:
            print(f"  [WARN] Cluster {lab}: eigenvalue ratio={ratio:.1f} "
                  f"— cluster is elongated/non-spherical. "
                  "KMeans may be wrong here; check GMM panel.")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def scatter(ax, xy, c, cmap, vmin, vmax, alpha=0.55, s=12):
    return ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap=cmap,
                      vmin=vmin, vmax=vmax, alpha=alpha, s=s, linewidths=0)


def plot_sequence(
    seq:        str,
    tracklets:  Dict,
    save_path:  Optional[Path] = None,
    show:       bool = False,
) -> None:
    embs, km_teams, confs, jerseys, tids = collect_embeddings(tracklets)
    n = len(embs)
    if n < 10:
        print(f"  [SKIP] Only {n} valid embeddings for {seq}")
        return

    print(f"\n=== {seq}  ({n} crop embeddings from {len(tracklets)} tracklets) ===")

    # ── Re-run UMAP to 2-D purely for display ───────────────────────────────
    embs_norm = normalize(embs, norm="l2")
    reducer   = umap.UMAP(n_components=2, random_state=42,
                          n_neighbors=min(30, n - 1), min_dist=0.05,
                          metric="cosine")
    xy = reducer.fit_transform(embs_norm)

    # ── GMM on the same 2-D projection ──────────────────────────────────────
    gmm_teams = fit_gmm(xy, n_components=2)

    # ── Silhouette scores ────────────────────────────────────────────────────
    sil_km  = cluster_quality(xy, km_teams,  "KMeans")
    sil_gmm = cluster_quality(xy, gmm_teams, "GMM   ")
    non_sphericity_warning(xy, km_teams)

    # ── Per-tracklet team consistency ────────────────────────────────────────
    print(f"\n  Per-tracklet team consistency (KMeans):")
    for tid in sorted(set(tids)):
        mask = tids == tid
        t    = km_teams[mask]
        valid = t[t >= 0]
        if len(valid) == 0:
            print(f"    Track {tid:>4}: no valid predictions")
            continue
        mode   = int(np.bincount(valid).argmax())
        purity = valid.tolist().count(mode) / len(valid)
        j_vals = jerseys[mask]
        j_vals = j_vals[~np.isnan(j_vals)]
        j_str  = f"jersey={int(np.bincount(j_vals.astype(int)).argmax())}" if len(j_vals) > 0 else "jersey=?"
        bar    = "█" * int(purity * 20) + "░" * (20 - int(purity * 20))
        print(f"    Track {tid:>4}: team={mode} [{bar}] {purity:.0%}  {j_str}")

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"Team clustering — {seq}  (n={n} crops)", fontsize=13, y=0.98)

    # ── Panel A: KMeans teams ────────────────────────────────────────────────
    ax = axes[0, 0]
    for team_id, col in TEAM_COLORS.items():
        mask = km_teams == team_id
        if mask.sum() == 0:
            continue
        ax.scatter(xy[mask, 0], xy[mask, 1],
                   color=col, alpha=0.5, s=12, linewidths=0,
                   label=f"{TEAM_LABELS[team_id]} (n={mask.sum()})")
    ax.set_title(f"A: KMeans prediction  (sil={sil_km:.3f})", fontsize=10)
    ax.legend(fontsize=8, markerscale=2)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_aspect("equal", adjustable="datalim")

    # ── Panel B: Cluster confidence ──────────────────────────────────────────
    ax = axes[0, 1]
    sc = scatter(ax, xy, confs, cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(ScalarMappable(Normalize(0, 1), cmap="RdYlGn"), ax=ax,
                 label="confidence", shrink=0.8)
    ax.set_title("B: Cluster confidence\n(green = high, red = low)", fontsize=10)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_aspect("equal", adjustable="datalim")

    # ── Panel C: Jersey numbers ──────────────────────────────────────────────
    ax = axes[1, 0]
    valid_j = ~np.isnan(jerseys)
    ax.scatter(xy[~valid_j, 0], xy[~valid_j, 1],
               color="#cccccc", alpha=0.3, s=8, linewidths=0, label="no jersey")
    if valid_j.sum() > 0:
        j_vals     = jerseys[valid_j]
        unique_j   = np.unique(j_vals)
        cmap_j     = plt.cm.get_cmap("tab20", len(unique_j))
        for k, jn in enumerate(unique_j):
            m = valid_j & (jerseys == jn)
            ax.scatter(xy[m, 0], xy[m, 1], color=cmap_j(k),
                       alpha=0.6, s=14, linewidths=0, label=f"#{int(jn)}")
    ax.set_title("C: Jersey number\n(same jersey should cluster together)", fontsize=10)
    if len(np.unique(jerseys[valid_j])) <= 12:
        ax.legend(fontsize=6, markerscale=1.5, ncol=2)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_aspect("equal", adjustable="datalim")

    # ── Panel D: GMM alternative ─────────────────────────────────────────────
    ax = axes[1, 1]
    for team_id, col in {0: "#e05c5c", 1: "#5c8de0"}.items():
        mask = gmm_teams == team_id
        ax.scatter(xy[mask, 0], xy[mask, 1],
                   color=col, alpha=0.5, s=12, linewidths=0,
                   label=f"GMM cluster {team_id} (n={mask.sum()})")

    # Highlight points where KMeans and GMM disagree
    disagree = km_teams != gmm_teams
    if disagree.sum() > 0:
        ax.scatter(xy[disagree, 0], xy[disagree, 1],
                   facecolors="none", edgecolors="black", s=40, linewidths=0.8,
                   label=f"KMeans/GMM disagree (n={disagree.sum()})", zorder=5)

    ax.set_title(
        f"D: GMM alternative  (sil={sil_gmm:.3f})\n"
        f"circled = KMeans/GMM disagree ({disagree.sum()} pts)",
        fontsize=10
    )
    ax.legend(fontsize=8, markerscale=2)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  Summary:")
    print(f"    KMeans  — team 0: {(km_teams==0).sum():>5}  team 1: {(km_teams==1).sum():>5}  "
          f"unknown: {(km_teams==-1).sum():>5}")
    print(f"    GMM     — team 0: {(gmm_teams==0).sum():>5}  team 1: {(gmm_teams==1).sum():>5}")
    print(f"    Disagreements between KMeans and GMM: {disagree.sum()} / {n} "
          f"({100*disagree.sum()/n:.1f}%)")
    if sil_gmm > sil_km + 0.02:
        print(f"  ⚠  GMM silhouette ({sil_gmm:.3f}) notably BETTER than KMeans ({sil_km:.3f})"
              " — consider switching to GMM in teamclassifier.py")
    elif sil_km > sil_gmm + 0.02:
        print(f"  ✓  KMeans silhouette ({sil_km:.3f}) is better than GMM ({sil_gmm:.3f})")
    else:
        print(f"  ~  KMeans and GMM give similar silhouette scores")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise team clusters")
    parser.add_argument("--seq",           default=None,
                        help="Single sequence name, e.g. SNGS-060")
    parser.add_argument("--all_sequences", action="store_true",
                        help="Run for every sequence found in the cache dir")
    args = parser.parse_args()

    if args.all_sequences:
        seqs = sorted(
            p.stem.replace("cache_attributes_", "")
            for p in CACHE_DIR.glob("cache_attributes_*.pkl")
        )
    elif args.seq:
        seqs = [args.seq]
    else:
        # Default: first available sequence
        found = sorted(CACHE_DIR.glob("cache_attributes_*.pkl"))
        if not found:
            print(f"No attribute caches found in {CACHE_DIR}")
            sys.exit(1)
        seqs = [found[0].stem.replace("cache_attributes_", "")]
        print(f"No --seq given, using first available: {seqs[0]}")

    for seq in seqs:
        tracklets = load_cache(seq)
        if tracklets is None:
            continue
        save_path = OUTPUT_DIR / f"{seq}_team_clusters.png"
        plot_sequence(seq, tracklets, save_path=save_path)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()