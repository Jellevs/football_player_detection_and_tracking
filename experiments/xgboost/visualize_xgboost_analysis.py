"""
Visualize XGBoost connector analysis figures for the thesis.

Generates three thesis-ready figures:
  1. Feature importance bar chart (horizontal, color-coded by category)
  2. Merge threshold curve (threshold vs HOTA/AssA)
  3. Feature ablation chart (remove-one vs keep-only delta HOTA)

Usage:
    python experiments/xgboost/visualize_xgboost_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)

# Feature importance from the best config (neg3_minlen0_purity0.8_splitTrue)
FEAT_IMP_PATH = (
    SCRIPT_DIR.parent / "tracklet_merger" / "xgboost" / "ablations" / "sweep_results"
    / "neg3_minlen0_purity0.8_splitTrue" / "feature_importance.csv"
)

# ── style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Category colors
CAT_COLORS = {
    "ReID":     "#E24A33",
    "Team":     "#4878CF",
    "Jersey":   "#6ACC65",
    "Spatial":  "#D65F5F",
    "Temporal": "#B47CC7",
    "SigLIP":   "#C4AD66",
    "Other":    "#999999",
}


def categorize_feature(name):
    """Assign a feature to its category based on name."""
    if "reid" in name:
        return "ReID"
    if "team" in name:
        return "Team"
    if "jersey" in name:
        return "Jersey"
    if "siglip" in name:
        return "SigLIP"
    # Spatial features
    spatial_keywords = [
        "spatial_distance", "endpoint_d", "bbox_height",
        "start_x", "start_y", "end_x", "end_y", "mean_bbox",
    ]
    if any(kw in name for kw in spatial_keywords):
        return "Spatial"
    # Temporal features
    temporal_keywords = [
        "temporal_gap", "duration", "start_frame", "end_frame", "n_frames",
    ]
    if any(kw in name for kw in temporal_keywords):
        return "Temporal"
    return "Other"


def pretty_feature_name(name):
    """Convert feature names to readable labels."""
    replacements = {
        "pairwise_reid_cosine_sim": "ReID cosine similarity",
        "pairwise_team_match": "Team match",
        "pairwise_spatial_distance": "Spatial distance",
        "pairwise_jersey_conflict": "Jersey conflict",
        "pairwise_team_conflict": "Team conflict",
        "pairwise_jersey_match": "Jersey match",
        "pairwise_jersey_both_confident": "Jersey both confident",
        "pairwise_temporal_gap": "Temporal gap",
        "pairwise_endpoint_dx": "Endpoint Δx",
        "pairwise_endpoint_dy": "Endpoint Δy",
        "pairwise_team_both_consistent": "Team both consistent",
        "pairwise_siglip_cosine_sim": "SigLIP cosine similarity",
        "pairwise_bbox_height_ratio": "Bbox height ratio",
        "A_duration": "Tracklet A duration",
        "B_duration": "Tracklet B duration",
        "A_jersey_coverage": "A jersey coverage",
        "B_jersey_conf_mean": "B jersey confidence",
        "B_jersey_entropy_mean": "B jersey entropy",
        "A_end_y": "A end y",
        "B_n_frames": "B num frames",
        "A_n_frames": "A num frames",
        "A_mean_bbox_height": "A mean bbox height",
        "A_team_mode": "A team mode",
        "B_end_y": "B end y",
        "A_start_frame": "A start frame",
        "B_team_coverage": "B team coverage",
        "B_start_y": "B start y",
    }
    return replacements.get(name, name.replace("_", " ").replace("pairwise ", ""))


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
def plot_feature_importance():
    """Horizontal bar chart of XGBoost feature importance, top 20 features."""
    df = pd.read_csv(FEAT_IMP_PATH)
    df = df.sort_values("importance", ascending=False).head(20).reset_index(drop=True)

    # Reverse for horizontal bar (top feature at top)
    df = df.iloc[::-1].reset_index(drop=True)

    categories = [categorize_feature(f) for f in df["feature"]]
    colors = [CAT_COLORS[c] for c in categories]
    labels = [pretty_feature_name(f) for f in df["feature"]]

    fig, ax = plt.subplots(figsize=(8, 6.5))

    bars = ax.barh(range(len(df)), df["importance"], color=colors,
                   edgecolor="white", linewidth=0.5, height=0.7, zorder=3)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("Top 20 XGBoost Feature Importances", fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--", zorder=0)

    # Legend for categories (only those present)
    present_cats = sorted(set(categories), key=lambda c: list(CAT_COLORS.keys()).index(c))
    patches = [mpatches.Patch(color=CAT_COLORS[c], label=c) for c in present_cats]
    ax.legend(handles=patches, loc="lower right", framealpha=0.9, fontsize=9)

    # Annotate top 3 bars with values
    for i in range(len(df) - 3, len(df)):
        val = df.iloc[i]["importance"]
        ax.text(val + 0.001, i, f"{val:.3f}", va="center", fontsize=8.5,
                fontweight="bold", color="#333333")

    out_path = OUT_DIR / "xgboost_feature_importance.pdf"
    fig.savefig(out_path, format="pdf")
    out_path_png = OUT_DIR / "xgboost_feature_importance.png"
    fig.savefig(out_path_png, format="png")
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Saved: {out_path_png}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Merge Threshold Curve
# ══════════════════════════════════════════════════════════════════════════════
def plot_merge_threshold():
    """Line plot of merge threshold vs HOTA and AssA."""
    # Data from Table clustering_sweep in the thesis (average linkage)
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    hota =       [89.961, 90.054, 90.211, 90.354, 90.419, 90.613, 90.743, 90.829, 90.849, 90.795, 90.867]
    assa =       [86.050, 86.227, 86.530, 86.804, 86.932, 87.304, 87.556, 87.727, 87.765, 87.662, 87.802]

    best_idx = np.argmax(hota)

    fig, ax1 = plt.subplots(figsize=(7, 4))

    # HOTA line
    line1, = ax1.plot(thresholds, hota, "o-", color="#E24A33", linewidth=2,
                      markersize=6, label="HOTA", zorder=4)
    ax1.scatter([thresholds[best_idx]], [hota[best_idx]], s=120, color="#E24A33",
                edgecolor="black", linewidth=1.5, zorder=5, marker="*")
    ax1.set_xlabel("Merge Threshold")
    ax1.set_ylabel("HOTA", color="#E24A33")
    ax1.tick_params(axis="y", labelcolor="#E24A33")

    # AssA on secondary axis
    ax2 = ax1.twinx()
    line2, = ax2.plot(thresholds, assa, "s--", color="#4878CF", linewidth=2,
                      markersize=5, label="AssA", zorder=3)
    ax2.set_ylabel("AssA", color="#4878CF")
    ax2.tick_params(axis="y", labelcolor="#4878CF")
    ax2.spines["top"].set_visible(False)

    # Annotate best
    ax1.annotate(f"Best: {hota[best_idx]:.3f}",
                 xy=(thresholds[best_idx], hota[best_idx]),
                 xytext=(15, 12), textcoords="offset points",
                 fontsize=9, fontweight="bold", color="#E24A33",
                 arrowprops=dict(arrowstyle="->", color="#E24A33", lw=1.2))

    # Default threshold marker
    default_idx = thresholds.index(0.50)
    ax1.axvline(x=0.50, color="#999999", linestyle=":", linewidth=1, alpha=0.7, zorder=1)
    ax1.text(0.505, hota[0] + 0.05, "default", fontsize=8, color="#999999", style="italic")

    # Combined legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", framealpha=0.9)

    ax1.set_title("Effect of Merge Threshold on Validation Performance", fontweight="bold")
    ax1.grid(axis="both", alpha=0.3, linestyle="--", zorder=0)

    # Tighten y range for HOTA
    ax1.set_ylim(89.5, 91.0)

    out_path = OUT_DIR / "xgboost_merge_threshold.pdf"
    fig.savefig(out_path, format="pdf")
    out_path_png = OUT_DIR / "xgboost_merge_threshold.png"
    fig.savefig(out_path_png, format="png")
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Saved: {out_path_png}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Feature Ablation Chart
# ══════════════════════════════════════════════════════════════════════════════
def plot_feature_ablation():
    """Two-panel chart: remove-one (left) and keep-only (right) delta HOTA."""
    baseline = 90.872

    # Remove-one data (from Table feature_ablation_remove)
    remove_groups = ["SigLIP", "Temporal", "Team", "Jersey", "Spatial", "ReID"]
    remove_hota =   [90.752, 90.448, 90.343, 89.679, 89.642, 88.239]
    remove_delta =  [h - baseline for h in remove_hota]

    # Keep-only data (from Table feature_ablation_keep)
    keep_groups =   ["ReID", "Spatial", "Jersey", "Team", "Temporal", "SigLIP"]
    keep_hota =     [87.758, 83.667, 83.774, 81.622, 80.242, 79.079]
    keep_delta =    [h - baseline for h in keep_hota]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5),
                                    gridspec_kw={"width_ratios": [6, 5], "wspace": 0.35})

    # ── Left panel: Remove one group ──
    y_pos = np.arange(len(remove_groups))
    colors_remove = ["#6ACC65" if d > 0 else "#E24A33" for d in remove_delta]

    bars1 = ax1.barh(y_pos, remove_delta, color=colors_remove, edgecolor="white",
                     linewidth=0.5, height=0.6, zorder=3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(remove_groups)
    ax1.set_xlabel("Δ HOTA")
    ax1.set_title("Remove One Feature Group", fontweight="bold")
    ax1.axvline(x=0, color="black", linewidth=0.8, zorder=2)
    ax1.grid(axis="x", alpha=0.3, linestyle="--", zorder=0)

    # Annotate bars
    for i, (d, h) in enumerate(zip(remove_delta, remove_hota)):
        if d > 0:
            ax1.text(d + 0.02, i, f"{d:+.3f}", va="center", ha="left",
                     fontsize=9, fontweight="bold", color="#333333")
        else:
            # Place label inside the bar if long enough, else outside
            if abs(d) > 0.3:
                ax1.text(d / 2, i, f"{d:+.3f}", va="center", ha="center",
                         fontsize=9, fontweight="bold", color="white")
            else:
                ax1.text(d - 0.02, i, f"{d:+.3f}", va="center", ha="right",
                         fontsize=9, fontweight="bold", color="#333333")

    # ── Right panel: Keep only one group ──
    y_pos2 = np.arange(len(keep_groups))
    colors_keep = [CAT_COLORS.get(g, "#4878CF") for g in keep_groups]

    bars2 = ax2.barh(y_pos2, keep_delta, color=colors_keep, edgecolor="white",
                     linewidth=0.5, height=0.6, zorder=3)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(keep_groups)
    ax2.set_xlabel("Δ HOTA")
    ax2.set_title("Keep Only One Feature Group", fontweight="bold")
    ax2.axvline(x=0, color="black", linewidth=0.8, zorder=2)
    ax2.grid(axis="x", alpha=0.3, linestyle="--", zorder=0)

    # Annotate bars
    for i, (d, h) in enumerate(zip(keep_delta, keep_hota)):
        ax2.text(d + 0.15, i, f"{d:+.2f}", va="center", ha="left",
                 fontsize=9, fontweight="bold", color="#333333")

    fig.suptitle("Feature Group Ablation Study (Baseline HOTA = {:.3f})".format(baseline),
                 fontsize=13, fontweight="bold", y=1.02)

    out_path = OUT_DIR / "xgboost_feature_ablation.pdf"
    fig.savefig(out_path, format="pdf")
    out_path_png = OUT_DIR / "xgboost_feature_ablation.png"
    fig.savefig(out_path_png, format="png")
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Saved: {out_path_png}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("Generating XGBoost analysis visualizations")
    print("=" * 60)

    plot_feature_importance()
    plot_merge_threshold()
    plot_feature_ablation()

    print("\nDone!")
