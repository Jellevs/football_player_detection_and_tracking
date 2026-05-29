"""
Generate thesis-ready training figures.

USAGE
-----
Edit the RUNS list below — one entry per log file.
Each entry needs at minimum a "path" and a "label".
"color" and "linestyle" are optional (auto-assigned if omitted).

Then run:
    python plot_thesis_training.py

Produces two PNGs in OUTPUT_DIR:
    val_auc_learning_curves.png
    generalization_gap.png
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# =======================================================================
#  CONFIGURE HERE
# =======================================================================

# Output folder for the saved figures
OUTPUT_DIR = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\figures")

# One dict per run. Required keys: "path", "label"
# Optional keys: "color", "linestyle" (defaults assigned automatically)
RUNS = [
    # Pairwise vs non pairwise
    #     {
    #     "path":  r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\transformers\siamese_cls\output\siamese_cls_xgb_matched\training_log.csv",
    #     "label": "Pairwise Real",
    #     "color": "#0072B2",
    # },
    #     {
    #     "path":  r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\transformers\siamese_cls\output\siamese_cls_no_pairwise\training_log.csv",
    #     "label": "No Pairwise Real",
    #     "color": "#D55E00",
    # },


    # Data gen synth
    #         {
    #     "path":  r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\transformers\cross_attention\output\cross_attention_simple_synth\training_log.csv",
    #     "label": "Cross Attention",
    #     "color": "#0072B2",
    # },
    #     {
    #     "path":  r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\transformers\siamese_cls\runs\ablation_base_d80_ff128\training_log.csv",
    #     "label": "Siamese CLS",
    #     "color": "#D55E00",
    # },
    #         {
    #     "path":  r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\transformers\hybrid\output\hybrid_simple_synth\training_log.csv",
    #     "label": "Hybrid",
    #     "color": "#CC79A7",
    # },

    # real
            {
        "path":  r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\transformers\cross_attention\output\cross_attention_xgb_matched\training_log.csv",
        "label": "Cross Attention",
        "color": "#0072B2",
    },
        {
        "path":  r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\transformers\siamese_cls\output\siamese_cls_xgb_matched\training_log.csv",
        "label": "Siamese CLS",
        "color": "#D55E00",
    },
            {
        "path":  r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\transformers\hybrid\output\hybrid_xgb_matched\training_log.csv",
        "label": "Hybrid",
        "color": "#CC79A7",
    },
    

]

# Smooth curves with a moving average (1 = no smoothing)
SMOOTHING_WINDOW = 3

# =======================================================================
#  END OF CONFIGURATION — no need to edit below
# =======================================================================

# Default color cycle if "color" is not set for a run
_DEFAULT_COLORS = [
    "#534AB7", "#1D9E75", "#D85A30", "#555555", "#C44E52",
    "#8C6BB1", "#88419D", "#2171B5", "#CB181D", "#238B45",
]
_DEFAULT_LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def read_log(path):
    """Read a training CSV log into a dict of numpy arrays."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    result = {}
    for key in rows[0]:
        try:
            result[key] = np.array([float(r[key]) for r in rows])
        except (ValueError, TypeError):
            result[key] = [r[key] for r in rows]
    return result


def smooth(arr, window):
    """Simple moving average, preserving array length."""
    if window <= 1:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(arr)]


def load_runs():
    """Load all log files and fill in default style values."""
    loaded = []
    for i, entry in enumerate(RUNS):
        path = Path(entry["path"])
        log = read_log(path)
        if log is None:
            print(f"WARNING: empty or missing log — {path}")
            continue
        loaded.append({
            "label":     entry["label"],
            "color":     entry.get("color",     _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]),
            "linestyle": entry.get("linestyle", _DEFAULT_LINESTYLES[i % len(_DEFAULT_LINESTYLES)]),
            "linewidth": entry.get("linewidth", 1.4),
            "alpha":     entry.get("alpha",     0.9),
            "log":       log,
        })
        best_val = max(log["val_auc"])
        print(f"  {entry['label']:40s}  {len(log['epoch']):3d} epochs,  best val AUC = {best_val:.4f}")
    return loaded


# -----------------------------------------------------------------------
# Figure 1: Validation AUC learning curves
# -----------------------------------------------------------------------

def plot_val_auc(runs):
    fig, ax = plt.subplots(figsize=(7, 3.5))

    all_aucs = np.concatenate([
        np.concatenate([r["log"]["val_auc"], r["log"]["train_auc"]]) for r in runs
    ])
    y_min = max(0.0, np.min(all_aucs) - 0.03)
    y_max = min(1.0, np.max(all_aucs) + 0.03)

    for run in runs:
        log = run["log"]
        epochs = log["epoch"]

        # Validation AUC — solid line
        ax.plot(epochs, smooth(log["val_auc"], SMOOTHING_WINDOW),
                color=run["color"],
                linestyle="-",
                linewidth=run["linewidth"],
                alpha=run["alpha"],
                label=f"{run['label']} val")

        # Train AUC — dashed line
        ax.plot(epochs, smooth(log["train_auc"], SMOOTHING_WINDOW),
                color=run["color"],
                linestyle="--",
                linewidth=run["linewidth"],
                alpha=run["alpha"],
                label=f"{run['label']} train")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_title("AUC learning curves")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_ylim(y_min, y_max)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()

    out = OUTPUT_DIR / "val_auc_learning_curves.png"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 2: Generalization gap (train AUC - val AUC)
# -----------------------------------------------------------------------

def plot_generalization_gap(runs):
    fig, ax = plt.subplots(figsize=(7, 3.5))

    for run in runs:
        log = run["log"]
        epochs = log["epoch"]
        gap = log["train_auc"] - log["val_auc"]
        gap_smooth = smooth(gap, SMOOTHING_WINDOW)

        ax.plot(epochs, gap_smooth,
                color=run["color"],
                linestyle=run["linestyle"],
                linewidth=run["linewidth"],
                alpha=run["alpha"],
                label=run["label"])

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train AUC − Val AUC")
    ax.set_title("Generalization gap")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()

    out = OUTPUT_DIR / "generalization_gap.png"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Smoothing window: {SMOOTHING_WINDOW}")
    print(f"Loading {len(RUNS)} run(s)...\n")

    runs = load_runs()

    if not runs:
        print("No valid runs found. Check your paths.")
    else:
        print()
        plot_val_auc(runs)
        plot_generalization_gap(runs)
        print("\nDone.")
