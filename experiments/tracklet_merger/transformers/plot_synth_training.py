"""
Plot training curves for Siamese CLS, Cross-Attention, and Hybrid
architectures trained on synthetic data only.

Produces two images:
    1. synth_auc_loss.png  -- AUC (top) and Loss (bottom)
    2. synth_gen_gap.png   -- Generalization gap (train AUC − val AUC)

Usage:
    python -m experiments.tracklet_merger.transformers.plot_synth_training
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path


# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------

BASE = Path(__file__).resolve().parent
OUTPUT_DIR = BASE.parents[2] / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

LOG_PATHS = {
    "siamese_cls": (
        BASE / "siamese_cls" / "runs"
        / "20260526_033128_simple_synth_L4_d80_ff128_h4_do0.2_fdo0.15_lr0.0005_wd0.001_bs32_ls0.05_sd0.1"
        / "training_log.csv"
    ),
    "cross_attention": (
        BASE / "cross_attention" / "output" / "cross_attention_simple_synth"
        / "training_log.csv"
    ),
    "hybrid": (
        BASE / "hybrid" / "output" / "hybrid_simple_synth"
        / "training_log.csv"
    ),
}


# -----------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------

COLORS = {
    "siamese_cls":     "#534AB7",
    "cross_attention": "#1D9E75",
    "hybrid":          "#D85A30",
}

LABELS = {
    "siamese_cls":     "Siamese CLS",
    "cross_attention": "Cross-Attention",
    "hybrid":          "Hybrid",
}

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

# Smoothing window (1 = no smoothing)
SMOOTHING_WINDOW = 3


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


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    # Load logs
    logs = {}
    for key, path in LOG_PATHS.items():
        log = read_log(path)
        if log is None:
            print(f"WARNING: empty or missing log for {key} at {path}")
            continue
        logs[key] = log
        best_val = max(log["val_auc"])
        print(f"  {LABELS[key]:20s}  {len(log['epoch']):3d} epochs, best val AUC = {best_val:.4f}")

    if not logs:
        print("No logs found, nothing to plot.")
        return

    # ------------------------------------------------------------------
    # Figure 1: AUC (top) and Loss (bottom)
    # ------------------------------------------------------------------
    fig1, (ax_auc, ax_loss) = plt.subplots(2, 1, figsize=(7, 6))

    for key, log in logs.items():
        ep    = log["epoch"]
        color = COLORS[key]
        label = LABELS[key]

        # AUC
        ax_auc.plot(ep, smooth(log["train_auc"], SMOOTHING_WINDOW),
                    color=color, linewidth=1.3, alpha=0.9, label=f"{label} train")
        ax_auc.plot(ep, smooth(log["val_auc"], SMOOTHING_WINDOW),
                    color=color, linewidth=1.3, alpha=0.9, linestyle="--", label=f"{label} val")
        best_i = np.argmax(log["val_auc"])
        # ax_auc.plot(ep[best_i], log["val_auc"][best_i], "o",
        #             color=color, markersize=5, zorder=5)

        # Loss
        ax_loss.plot(ep, smooth(log["train_loss"], SMOOTHING_WINDOW),
                     color=color, linewidth=1.3, alpha=0.9, label=f"{label} train")
        ax_loss.plot(ep, smooth(log["val_loss"], SMOOTHING_WINDOW),
                     color=color, linewidth=1.3, alpha=0.9, linestyle="--", label=f"{label} val")

    ax_auc.set(xlabel="Epoch", ylabel="AUC", title="AUC")
    ax_auc.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_auc.legend(fontsize=7.5, loc="lower right", framealpha=0.9)

    ax_loss.set(xlabel="Epoch", ylabel="Loss", title="Loss")
    ax_loss.legend(fontsize=7.5, loc="upper right", framealpha=0.9)

    fig1.tight_layout()
    out1 = OUTPUT_DIR / "synth_auc_loss.png"
    fig1.savefig(out1)
    print(f"Saved → {out1}")
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Figure 2: Generalization gap
    # ------------------------------------------------------------------
    fig2, ax_gap = plt.subplots(figsize=(7, 3.5))

    for key, log in logs.items():
        ep    = log["epoch"]
        color = COLORS[key]
        label = LABELS[key]

        gap = log["train_auc"] - log["val_auc"]
        ax_gap.plot(ep, smooth(gap, SMOOTHING_WINDOW),
                    color=color, linewidth=1.3, alpha=0.9, label=label)

    ax_gap.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax_gap.set(xlabel="Epoch", ylabel="Train AUC − Val AUC", title="Generalization gap")
    ax_gap.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_gap.legend(fontsize=8.5, loc="upper left", framealpha=0.9)

    fig2.tight_layout()
    out2 = OUTPUT_DIR / "synth_gen_gap.png"
    fig2.savefig(out2)
    print(f"Saved → {out2}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
