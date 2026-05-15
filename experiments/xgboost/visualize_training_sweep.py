"""
Visualize XGBoost training data generation sweep results.

Generates two thesis-ready figures:
  1. Main effects bar chart (4 panels, one per parameter)
  2. Heatmap matrix showing interaction effects

Usage:
    python experiments/xgboost/visualize_training_sweep.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "sweep_summary.csv"
OUT_DIR = SCRIPT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ── style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "bar": "#4878CF",
    "best": "#E24A33",
    "point": "#333333",
}


def load_data():
    df = pd.read_csv(CSV_PATH)
    # Clean negative_ratio: NaN means "None" (no downsampling)
    df["negative_ratio"] = df["negative_ratio"].fillna("None")
    df["negative_ratio"] = df["negative_ratio"].apply(
        lambda x: "None" if x == "None" else str(int(float(x)))
    )
    # Clean splitted for display
    df["splitted_label"] = df["splitted"].map({True: "Pre-split", False: "No pre-split"})
    # Clean purity for display
    df["purity_label"] = df["tracklet_purity"].apply(lambda x: f"{x:.1f}")
    # Clean min_tracklet_len for display
    df["minlen_label"] = df["min_tracklet_len"].astype(str)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Main effects bar chart
# ══════════════════════════════════════════════════════════════════════════════
def plot_main_effects(df):
    """Four-panel bar chart showing the marginal effect of each parameter."""

    params = [
        {
            "col": "negative_ratio",
            "title": "Negative Ratio",
            "xlabel": "Negative Ratio",
            "order": ["None", "3", "5", "10"],
        },
        {
            "col": "min_tracklet_len",
            "title": "Min. Tracklet Length",
            "xlabel": "Min. Tracklet Length",
            "order": [0, 5, 10],
        },
        {
            "col": "tracklet_purity",
            "title": "Tracklet Purity",
            "xlabel": "Tracklet Purity Threshold",
            "order": [0.6, 0.8, 1.0],
        },
        {
            "col": "splitted",
            "title": "Pre-splitting",
            "xlabel": "Pre-split Training Data",
            "order": [True, False],
            "xlabels": ["Yes", "No"],
        },
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
    fig.subplots_adjust(wspace=0.12)

    metric = "test_auc_roc"

    for ax, p in zip(axes, params):
        order = p["order"]
        means, stds, counts = [], [], []
        individual_points = []

        for val in order:
            subset = df[df[p["col"]] == val][metric]
            means.append(subset.mean())
            stds.append(subset.std())
            counts.append(len(subset))
            individual_points.append(subset.values)

        x = np.arange(len(order))
        best_idx = np.argmax(means)

        # Bar colours: highlight the best
        colors = [COLORS["best"] if i == best_idx else COLORS["bar"] for i in range(len(order))]

        bars = ax.bar(x, means, width=0.55, color=colors, edgecolor="white",
                      linewidth=0.5, zorder=3)

        # Error bars (std)
        ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="#555555",
                     capsize=4, capthick=1.2, linewidth=1.2, zorder=4)

        # Individual config points as strip
        for i, pts in enumerate(individual_points):
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(pts))
            ax.scatter(x[i] + jitter, pts, s=12, color=COLORS["point"],
                       alpha=0.35, zorder=5, linewidths=0)

        # Labels
        xlabels = p.get("xlabels", [str(v) for v in order])
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel(p["xlabel"])
        ax.set_title(p["title"], fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)

        # Annotate best mean
        ax.annotate(f"{means[best_idx]:.4f}",
                     xy=(best_idx, means[best_idx]),
                     xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=9, fontweight="bold",
                     color=COLORS["best"])

    axes[0].set_ylabel("Test AUC-ROC")

    # Set y-axis range to zoom in on the relevant range
    all_vals = df[metric].values
    y_min = max(0.88, all_vals.min() - 0.01)
    y_max = min(1.0, all_vals.max() + 0.01)
    axes[0].set_ylim(y_min, y_max)

    fig.suptitle("Main Effect of Each Training Data Parameter on Test AUC-ROC",
                 fontsize=13, fontweight="bold", y=1.04)

    out_path = OUT_DIR / "xgboost_sweep_main_effects.pdf"
    fig.savefig(out_path, format="pdf")
    out_path_png = OUT_DIR / "xgboost_sweep_main_effects.png"
    fig.savefig(out_path_png, format="png")
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Saved: {out_path_png}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Heatmap matrix
# ══════════════════════════════════════════════════════════════════════════════
def plot_heatmap(df):
    """
    2x3 heatmap grid.
    Rows:   splitted (Pre-split / No pre-split)
    Cols:   tracklet_purity (0.6, 0.8, 1.0)
    Each cell: negative_ratio (y) x min_tracklet_len (x), colored by test AUC.
    """
    metric = "test_auc_roc"

    neg_order = ["None", "3", "5", "10"]
    minlen_order = [0, 5, 10]
    purity_order = [0.6, 0.8, 1.0]
    split_order = [True, False]
    split_labels = ["Pre-split", "No pre-split"]

    # Global colour scale
    vmin = df[metric].min()
    vmax = df[metric].max()
    # Use a perceptually uniform colourmap
    cmap = plt.cm.RdYlGn

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5),
                             gridspec_kw={"hspace": 0.35, "wspace": 0.08})

    for row_idx, (split_val, split_lbl) in enumerate(zip(split_order, split_labels)):
        for col_idx, purity_val in enumerate(purity_order):
            ax = axes[row_idx, col_idx]
            subset = df[(df["splitted"] == split_val) &
                        (df["tracklet_purity"] == purity_val)]

            # Build the matrix: neg_ratio (rows) x min_tracklet_len (cols)
            matrix = np.full((len(neg_order), len(minlen_order)), np.nan)
            for i, neg in enumerate(neg_order):
                for j, ml in enumerate(minlen_order):
                    match = subset[(subset["negative_ratio"] == neg) &
                                   (subset["min_tracklet_len"] == ml)]
                    if len(match) == 1:
                        matrix[i, j] = match[metric].values[0]

            im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax,
                           aspect="auto", origin="upper")

            # Annotate cells with values
            for i in range(len(neg_order)):
                for j in range(len(minlen_order)):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        # Pick text colour based on background
                        bg_norm = (val - vmin) / (vmax - vmin)
                        text_color = "white" if bg_norm < 0.4 or bg_norm > 0.85 else "black"
                        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                                fontsize=8.5, fontweight="bold", color=text_color)

            # Axis labels
            ax.set_xticks(range(len(minlen_order)))
            ax.set_xticklabels([str(m) for m in minlen_order])
            ax.set_yticks(range(len(neg_order)))

            if col_idx == 0:
                ax.set_yticklabels(neg_order)
                ax.set_ylabel(f"{split_lbl}\n\nNegative Ratio", fontweight="bold")
            else:
                ax.set_yticklabels([])

            if row_idx == 1:
                ax.set_xlabel("Min. Tracklet Length")

            ax.set_title(f"Purity = {purity_val:.1f}", fontsize=11)

            # Highlight the global best cell
            global_best = df[metric].max()
            if not np.isnan(matrix).all() and np.nanmax(matrix) == global_best:
                best_pos = np.unravel_index(np.nanargmax(matrix), matrix.shape)
                rect = plt.Rectangle((best_pos[1] - 0.5, best_pos[0] - 0.5),
                                     1, 1, linewidth=2.5, edgecolor="black",
                                     facecolor="none", zorder=10)
                ax.add_patch(rect)

    # Colourbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Test AUC-ROC", fontsize=11)

    fig.suptitle("Test AUC-ROC by Training Data Configuration",
                 fontsize=13, fontweight="bold", y=0.98)

    out_path = OUT_DIR / "xgboost_sweep_heatmap.pdf"
    fig.savefig(out_path, format="pdf")
    out_path_png = OUT_DIR / "xgboost_sweep_heatmap.png"
    fig.savefig(out_path_png, format="png")
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Saved: {out_path_png}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df = load_data()

    print("=" * 60)
    print("Generating XGBoost training sweep visualizations")
    print("=" * 60)

    # Print quick summary
    best_row = df.loc[df["test_auc_roc"].idxmax()]
    print(f"\nBest config: {best_row['config_name']}")
    print(f"  Test AUC: {best_row['test_auc_roc']:.4f}")
    print(f"  Test F1:  {best_row['test_f1']:.4f}")
    print()

    plot_main_effects(df)
    plot_heatmap(df)

    print("\nDone!")
