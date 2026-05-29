"""
Generate the two transformer experiment figures for the thesis.

Reads training logs directly from the output CSV files.

Outputs (saved to figures/ at the repo root):
    1. transformer_training_curves.png  -- per-model panels, both data strategies
    2. transformer_val_auc_and_gap.png  -- all models compared + generalization gap

Usage:
    python plot_thesis_figures.py
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "figures"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRANSFORMER_BASE = SCRIPT_DIR


def log_path(model, data):
    return os.path.join(
        TRANSFORMER_BASE, model, "output", f"{model}_{data}", "training_log.csv"
    )


def read_log(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    result = {}
    for key in rows[0]:
        try:
            result[key] = [float(r[key]) for r in rows]
        except (ValueError, TypeError):
            result[key] = [r[key] for r in rows]
    return result


# -----------------------------------------------------------------------
# Load all training logs
# -----------------------------------------------------------------------

logs = {}
for model in ["siamese_cls", "cross_attention", "hybrid"]:
    for data in ["simple_synth", "xgb_matched"]:
        logs[f"{model}_{data}"] = read_log(log_path(model, data))

logs["pairwise_mlp_xgb_matched"] = read_log(
    os.path.join(TRANSFORMER_BASE, "pairwise_mlp", "output",
                 "pairwise_mlp_xgb_matched", "training_log.csv")
)


def best_epoch_idx(log):
    return int(np.argmax(log["val_auc"]))


# -----------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})

COLORS = {
    "siamese_cls":     "#534AB7",
    "cross_attention":  "#1D9E75",
    "hybrid":           "#D85A30",
    "pairwise_mlp":     "#888780",
}

# Separate colors for the two data strategies within each model panel
COLOR_SYNTH = "#2A7FDB"   # blue for real+synth
COLOR_REAL  = "#D85A30"   # coral for real-only

TITLES = {
    "siamese_cls":     "Siamese CLS (299k)",
    "cross_attention":  "Cross-attention (252k)",
    "hybrid":           "Hybrid (302k)",
    "pairwise_mlp":     "Pairwise MLP (3k)",
}


# -----------------------------------------------------------------------
# Figure 1: Per-model training panels (2x2)
#
#   Each subplot shows ONE architecture with both data strategies overlaid.
#     - Solid line  = validation AUC
#     - Dashed line = validation loss (thin, secondary)
#     - Blue        = real+synth
#     - Coral       = real-only
#     - Dotted vertical = best validation epoch
# -----------------------------------------------------------------------

def plot_training_curves():
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5))
    axes = axes.flatten()

    panels = [
        ("siamese_cls",     ["simple_synth", "xgb_matched"]),
        ("cross_attention",  ["simple_synth", "xgb_matched"]),
        ("hybrid",           ["simple_synth", "xgb_matched"]),
        ("pairwise_mlp",     ["xgb_matched"]),
    ]

    data_colors = {"simple_synth": COLOR_SYNTH, "xgb_matched": COLOR_REAL}
    data_labels = {"simple_synth": "Real + synth", "xgb_matched": "Real only"}

    for idx, (model, data_keys) in enumerate(panels):
        ax = axes[idx]

        for data_key in data_keys:
            key = f"{model}_{data_key}"
            log = logs[key]
            epochs = log["epoch"]
            color = data_colors[data_key]
            best_i = best_epoch_idx(log)

            # Validation AUC (main signal, solid)
            ax.plot(epochs, log["val_auc"], color=color, linestyle="-",
                    linewidth=1.4, label=f"{data_labels[data_key]} (val AUC)")

            # Train AUC (thin dashed, same color)
            ax.plot(epochs, log["train_auc"], color=color, linestyle="--",
                    linewidth=0.8, alpha=0.5)

            # Validation loss (thin dotted, same color, secondary info)
            ax.plot(epochs, log["val_loss"], color=color, linestyle=":",
                    linewidth=0.8, alpha=0.45)

            # Best epoch marker
            ax.axvline(epochs[best_i], color=color, linestyle=":",
                       linewidth=1.0, alpha=0.6)
            ax.plot(epochs[best_i], log["val_auc"][best_i], "o",
                    color=color, markersize=4, zorder=5)

        ax.set_title(TITLES[model])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AUC / Loss")
        ax.set_ylim(0.38, 1.02)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # Shared legend below figure
    legend_elements = [
        Line2D([0], [0], color=COLOR_SYNTH, linewidth=1.5, label="Real + synth"),
        Line2D([0], [0], color=COLOR_REAL, linewidth=1.5, label="Real only"),
        Line2D([0], [0], color="gray", linestyle="-", linewidth=1.2,
               label="Val AUC"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=0.8,
               alpha=0.5, label="Train AUC"),
        Line2D([0], [0], color="gray", linestyle=":", linewidth=0.8,
               alpha=0.45, label="Val loss"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.02), fontsize=8.5)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(OUTPUT_DIR, "transformer_training_curves.png")
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 2: Cross-model comparison + generalization gap
#
#   Left panel:  Validation AUC for all models (best data strategy each
#                shown as solid; alternative as dashed for context).
#   Right panel: Generalization gap (train AUC - val AUC).
#   Color = architecture. Solid = real-synth, dashed = real-only.
# -----------------------------------------------------------------------

def plot_val_auc_and_gap():
    fig, (ax_val, ax_gap) = plt.subplots(1, 2, figsize=(7.5, 3.2))

    models = ["siamese_cls", "cross_attention", "hybrid"]
    data_styles = [
        ("simple_synth", "-",  "real-synth"),
        ("xgb_matched",  "--", "real-only"),
    ]

    for model in models:
        color = COLORS[model]
        for data_key, ls, dlabel in data_styles:
            key = f"{model}_{data_key}"
            log = logs[key]
            epochs = log["epoch"]
            val = np.array(log["val_auc"])
            train = np.array(log["train_auc"])
            gap = train - val

            label = f"{TITLES[model].split(' (')[0]} ({dlabel})"
            ax_val.plot(epochs, val, color=color, linestyle=ls,
                        linewidth=1.2, alpha=0.85, label=label)
            ax_gap.plot(epochs, gap, color=color, linestyle=ls,
                        linewidth=1.2, alpha=0.85, label=label)

    # Pairwise MLP
    mlp = logs["pairwise_mlp_xgb_matched"]
    mlp_ep = mlp["epoch"]
    mlp_val = np.array(mlp["val_auc"])
    mlp_gap = np.array(mlp["train_auc"]) - mlp_val
    ax_val.plot(mlp_ep, mlp_val, color=COLORS["pairwise_mlp"],
                linestyle="-", linewidth=1.2, alpha=0.85, label="Pairwise MLP")
    ax_gap.plot(mlp_ep, mlp_gap, color=COLORS["pairwise_mlp"],
                linestyle="-", linewidth=1.2, alpha=0.85, label="Pairwise MLP")

    # Formatting
    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("Validation AUC")
    ax_val.set_ylim(0.65, 0.92)
    ax_val.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_val.set_title("Validation AUC")

    ax_gap.set_xlabel("Epoch")
    ax_gap.set_ylabel("Train AUC - Val AUC")
    ax_gap.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.4)
    ax_gap.set_ylim(-0.05, 0.16)
    ax_gap.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_gap.set_title("Generalization gap")

    # Legend: color = architecture, linestyle = data strategy
    legend_elements = [
        Line2D([0], [0], color=COLORS["siamese_cls"], linewidth=1.5,
               label="Siamese CLS"),
        Line2D([0], [0], color=COLORS["cross_attention"], linewidth=1.5,
               label="Cross-attention"),
        Line2D([0], [0], color=COLORS["hybrid"], linewidth=1.5,
               label="Hybrid"),
        Line2D([0], [0], color=COLORS["pairwise_mlp"], linewidth=1.5,
               label="Pairwise MLP"),
        Line2D([0], [0], color="gray", linestyle="-", linewidth=1.2,
               label="Real-synth"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.2,
               label="Real-only"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=6,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.06), fontsize=8)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = os.path.join(OUTPUT_DIR, "transformer_val_auc_and_gap.png")
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    plot_training_curves()
    plot_val_auc_and_gap()
    print("All figures saved.")
