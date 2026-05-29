"""
Visualization script for transformer connector experiment results.
Generates four publication-ready figures for the thesis experiments section.

Usage:
    python plot_transformer_results.py

Outputs (saved to ./figures/):
    1. val_auc_learning_curves.pdf
    2. test_auc_ap_comparison.pdf
    3. generalization_gap.pdf
    4. precision_recall_f1.pdf
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ──────────────────────────────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

COLORS = {
    "siamese": "#534AB7",
    "cross": "#1D9E75",
    "hybrid": "#D85A30",
}

LABELS = {
    "siamese": "Siamese CLS",
    "cross": "Cross-attention",
    "hybrid": "Hybrid",
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# Training log data
# ──────────────────────────────────────────────────────────────────────

# --- Validation AUC per epoch ---

val_auc = {
    "siamese_simple_synth": [
        0.808, 0.823, 0.821, 0.826, 0.833, 0.841, 0.857, 0.863, 0.856,
        0.877, 0.875, 0.882, 0.860, 0.890, 0.884, 0.874, 0.893, 0.876,
        0.890, 0.880, 0.863, 0.869, 0.858, 0.874, 0.867, 0.858, 0.871,
        0.854, 0.856, 0.859, 0.852, 0.851,
    ],
    "cross_simple_synth": [
        0.815, 0.825, 0.825, 0.833, 0.836, 0.844, 0.846, 0.860, 0.882,
        0.858, 0.878, 0.870, 0.880, 0.881, 0.869, 0.886, 0.889, 0.871,
        0.882, 0.886, 0.890, 0.887, 0.884, 0.892, 0.888, 0.887, 0.887,
        0.893, 0.891, 0.893, 0.894, 0.874, 0.872, 0.896, 0.888, 0.878,
        0.890, 0.885, 0.887, 0.885, 0.881, 0.877, 0.889, 0.894, 0.895,
        0.897, 0.882, 0.894, 0.892, 0.897, 0.885, 0.898, 0.899, 0.900,
        0.899, 0.899, 0.893, 0.896, 0.899, 0.895, 0.896, 0.893, 0.896,
        0.896, 0.897, 0.896, 0.892, 0.898, 0.891,
    ],
    "hybrid_simple_synth": [
        0.816, 0.825, 0.826, 0.830, 0.825, 0.842, 0.846, 0.856, 0.878,
        0.874, 0.870, 0.883, 0.880, 0.885, 0.880, 0.881, 0.862, 0.874,
        0.891, 0.870, 0.882, 0.880, 0.893, 0.882, 0.884, 0.869, 0.894,
        0.890, 0.877, 0.896, 0.873, 0.880, 0.888, 0.883, 0.887, 0.875,
        0.887, 0.885, 0.893, 0.897, 0.892, 0.893, 0.887, 0.885, 0.885,
        0.883, 0.883, 0.889, 0.879, 0.885, 0.886, 0.877, 0.881, 0.885,
        0.881,
    ],
    "siamese_xgb_matched": [
        0.821, 0.820, 0.826, 0.826, 0.836, 0.835, 0.833, 0.838, 0.842,
        0.840, 0.850, 0.855, 0.839, 0.865, 0.870, 0.871, 0.871, 0.853,
        0.864, 0.844, 0.828, 0.842, 0.854, 0.850, 0.864, 0.849, 0.858,
        0.848, 0.831, 0.852, 0.847, 0.852,
    ],
    "cross_xgb_matched": [
        0.809, 0.815, 0.825, 0.829, 0.831, 0.831, 0.833, 0.835, 0.839,
        0.840, 0.842, 0.846, 0.843, 0.851, 0.851, 0.854, 0.858, 0.860,
        0.866, 0.864, 0.871, 0.880, 0.874, 0.877, 0.880, 0.878, 0.887,
        0.888, 0.889, 0.869, 0.875, 0.871, 0.884, 0.886, 0.876, 0.884,
        0.888, 0.874, 0.886, 0.889, 0.887, 0.884, 0.880, 0.882, 0.885,
        0.887, 0.881, 0.883, 0.885, 0.880, 0.882, 0.877, 0.883, 0.877,
        0.877,
    ],
    "hybrid_xgb_matched": [
        0.812, 0.821, 0.821, 0.826, 0.832, 0.833, 0.833, 0.836, 0.831,
        0.840, 0.839, 0.841, 0.840, 0.844, 0.847, 0.846, 0.849, 0.853,
        0.859, 0.859, 0.863, 0.865, 0.869, 0.871, 0.865, 0.873, 0.880,
        0.876, 0.884, 0.881, 0.872, 0.877, 0.883, 0.876, 0.878, 0.878,
        0.875, 0.883, 0.880, 0.878, 0.879, 0.873, 0.878, 0.874,
    ],
}

# --- Train AUC per epoch ---

train_auc = {
    "siamese_simple_synth": [
        0.925, 0.931, 0.933, 0.936, 0.938, 0.939, 0.942, 0.947, 0.948,
        0.951, 0.951, 0.955, 0.957, 0.961, 0.964, 0.966, 0.968, 0.969,
        0.969, 0.973, 0.972, 0.973, 0.976, 0.977, 0.978, 0.978, 0.980,
        0.982, 0.984, 0.984, 0.983, 0.984,
    ],
    "cross_simple_synth": [
        0.923, 0.933, 0.936, 0.936, 0.939, 0.942, 0.944, 0.946, 0.951,
        0.952, 0.954, 0.955, 0.955, 0.956, 0.958, 0.958, 0.958, 0.957,
        0.960, 0.959, 0.960, 0.959, 0.959, 0.960, 0.961, 0.962, 0.961,
        0.962, 0.962, 0.962, 0.963, 0.963, 0.963, 0.964, 0.963, 0.961,
        0.963, 0.965, 0.962, 0.965, 0.964, 0.963, 0.965, 0.967, 0.967,
        0.968, 0.971, 0.969, 0.969, 0.968, 0.969, 0.969, 0.969, 0.971,
        0.969, 0.971, 0.969, 0.971, 0.970, 0.971, 0.972, 0.972, 0.972,
        0.973, 0.973, 0.974, 0.975, 0.974, 0.974,
    ],
    "hybrid_simple_synth": [
        0.921, 0.932, 0.935, 0.937, 0.940, 0.941, 0.945, 0.948, 0.951,
        0.953, 0.955, 0.954, 0.958, 0.960, 0.958, 0.959, 0.958, 0.960,
        0.958, 0.961, 0.961, 0.963, 0.963, 0.961, 0.962, 0.963, 0.964,
        0.962, 0.962, 0.964, 0.963, 0.965, 0.964, 0.964, 0.966, 0.966,
        0.967, 0.967, 0.967, 0.969, 0.972, 0.971, 0.972, 0.973, 0.972,
        0.973, 0.973, 0.972, 0.973, 0.974, 0.975, 0.977, 0.976, 0.977,
        0.978,
    ],
    "siamese_xgb_matched": [
        0.784, 0.804, 0.817, 0.817, 0.816, 0.819, 0.819, 0.824, 0.827,
        0.832, 0.834, 0.841, 0.858, 0.857, 0.877, 0.896, 0.900, 0.908,
        0.905, 0.917, 0.916, 0.921, 0.926, 0.937, 0.942, 0.939, 0.949,
        0.955, 0.956, 0.962, 0.962, 0.963,
    ],
    "cross_xgb_matched": [
        0.774, 0.806, 0.811, 0.814, 0.817, 0.820, 0.820, 0.824, 0.826,
        0.828, 0.830, 0.835, 0.836, 0.840, 0.841, 0.847, 0.850, 0.854,
        0.861, 0.865, 0.869, 0.875, 0.877, 0.886, 0.885, 0.887, 0.885,
        0.894, 0.891, 0.897, 0.894, 0.894, 0.895, 0.900, 0.904, 0.904,
        0.903, 0.907, 0.910, 0.914, 0.915, 0.916, 0.917, 0.915, 0.920,
        0.922, 0.924, 0.923, 0.924, 0.926, 0.931, 0.932, 0.934, 0.934,
        0.935,
    ],
    "hybrid_xgb_matched": [
        0.787, 0.806, 0.815, 0.815, 0.818, 0.821, 0.823, 0.824, 0.828,
        0.829, 0.831, 0.834, 0.838, 0.842, 0.844, 0.851, 0.850, 0.855,
        0.857, 0.865, 0.870, 0.879, 0.879, 0.880, 0.885, 0.894, 0.890,
        0.899, 0.897, 0.903, 0.902, 0.904, 0.911, 0.909, 0.908, 0.911,
        0.912, 0.911, 0.921, 0.924, 0.931, 0.934, 0.933, 0.931,
    ],
}

# --- Test metrics ---

test_metrics = {
    "siamese_simple_synth": {
        "auc": 0.875, "ap": 0.468, "f1": 0.480,
        "precision": 0.495, "recall": 0.466,
    },
    "siamese_xgb_matched": {
        "auc": 0.843, "ap": 0.368, "f1": 0.401,
        "precision": 0.425, "recall": 0.380,
    },
    "cross_simple_synth": {
        "auc": 0.882, "ap": 0.475, "f1": 0.471,
        "precision": 0.423, "recall": 0.531,
    },
    "cross_xgb_matched": {
        "auc": 0.872, "ap": 0.471, "f1": 0.460,
        "precision": 0.451, "recall": 0.470,
    },
    "hybrid_simple_synth": {
        "auc": 0.875, "ap": 0.474, "f1": 0.466,
        "precision": 0.437, "recall": 0.499,
    },
    "hybrid_xgb_matched": {
        "auc": 0.857, "ap": 0.397, "f1": 0.429,
        "precision": 0.373, "recall": 0.506,
    },
}


# ──────────────────────────────────────────────────────────────────────
# Figure 1: Validation AUC learning curves (side by side)
# ──────────────────────────────────────────────────────────────────────

def plot_learning_curves():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

    for ax, suffix, title in [
        (ax1, "simple_synth", "Real + synthetic data"),
        (ax2, "xgb_matched", "Real only (XGB matched)"),
    ]:
        for key, label, color in [
            ("siamese", LABELS["siamese"], COLORS["siamese"]),
            ("cross", LABELS["cross"], COLORS["cross"]),
            ("hybrid", LABELS["hybrid"], COLORS["hybrid"]),
        ]:
            data = val_auc[f"{key}_{suffix}"]
            epochs = np.arange(1, len(data) + 1)
            ax.plot(epochs, data, color=color, label=label,
                    linewidth=1.2, alpha=0.85)

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0.80, 0.91)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    ax1.set_ylabel("Validation AUC")
    ax1.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "val_auc_learning_curves.png"))
    print("Saved val_auc_learning_curves.png")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Figure 2: Test AUC-ROC and Average Precision (grouped bars)
# ──────────────────────────────────────────────────────────────────────

def plot_test_auc_ap():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    models = ["siamese", "cross", "hybrid"]
    x = np.arange(len(models))
    w = 0.30

    for ax, metric, title, ylim in [
        (ax1, "auc", "Test AUC-ROC", (0.82, 0.90)),
        (ax2, "ap", "Test average precision", (0.30, 0.52)),
    ]:
        ss_vals = [test_metrics[f"{m}_simple_synth"][metric] for m in models]
        xgb_vals = [test_metrics[f"{m}_xgb_matched"][metric] for m in models]

        bars1 = ax.bar(x - w / 2, ss_vals, w, label="Real + synthetic",
                       color="#AFA9EC", edgecolor="white", linewidth=0.5)
        bars2 = ax.bar(x + w / 2, xgb_vals, w, label="Real only",
                       color="#534AB7", edgecolor="white", linewidth=0.5)

        # Value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in models])
        ax.set_ylim(ylim)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    ax1.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "test_auc_ap_comparison.png"))
    print("Saved test_auc_ap_comparison.png")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Figure 3: Generalization gap (train AUC - val AUC)
# ──────────────────────────────────────────────────────────────────────

def plot_generalization_gap():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

    for ax, suffix, title in [
        (ax1, "simple_synth", "Real + synthetic data"),
        (ax2, "xgb_matched", "Real only (XGB matched)"),
    ]:
        for key, label, color in [
            ("siamese", LABELS["siamese"], COLORS["siamese"]),
            ("cross", LABELS["cross"], COLORS["cross"]),
            ("hybrid", LABELS["hybrid"], COLORS["hybrid"]),
        ]:
            t = np.array(train_auc[f"{key}_{suffix}"])
            v = np.array(val_auc[f"{key}_{suffix}"])
            n = min(len(t), len(v))
            gap = t[:n] - v[:n]
            epochs = np.arange(1, n + 1)
            ax.plot(epochs, gap, color=color, label=label,
                    linewidth=1.2, alpha=0.85)

        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylim(-0.04, 0.16)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    ax1.set_ylabel("Train AUC − Val AUC")
    ax1.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "generalization_gap.png"))
    print("Saved generalization_gap.png")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Figure 4: Precision / Recall / F1 breakdown
# ──────────────────────────────────────────────────────────────────────

def plot_precision_recall_f1():
    fig, ax = plt.subplots(figsize=(7, 3.5))

    models = ["siamese", "cross", "hybrid"]
    metrics = ["precision", "recall", "f1"]
    metric_labels = ["Precision", "Recall", "F1"]
    n_models = len(models)
    n_metrics = len(metrics)
    n_bars = n_metrics * 2  # 2 data strategies per metric

    # Color pairs: lighter = real+synth, darker = real only
    color_pairs = [
        ("#CECBF6", "#534AB7"),  # precision: purple light/dark
        ("#9FE1CB", "#0F6E56"),  # recall: teal light/dark
        ("#F0997B", "#993C1D"),  # f1: coral light/dark
    ]

    x = np.arange(n_models)
    total_width = 0.75
    bar_width = total_width / n_bars
    offsets = np.linspace(
        -total_width / 2 + bar_width / 2,
        total_width / 2 - bar_width / 2,
        n_bars,
    )

    for i, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        ss_vals = [test_metrics[f"{m}_simple_synth"][metric] for m in models]
        xgb_vals = [test_metrics[f"{m}_xgb_matched"][metric] for m in models]
        c_light, c_dark = color_pairs[i]

        ax.bar(x + offsets[i * 2], ss_vals, bar_width,
               label=f"{mlabel} (real+synth)", color=c_light,
               edgecolor="white", linewidth=0.3)
        ax.bar(x + offsets[i * 2 + 1], xgb_vals, bar_width,
               label=f"{mlabel} (real only)", color=c_dark,
               edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in models])
    ax.set_ylabel("Score")
    ax.set_ylim(0.30, 0.58)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="upper right", ncol=2, fontsize=8, framealpha=0.9)
    ax.set_title("Test set precision, recall, and F1")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "precision_recall_f1.png"))
    print("Saved precision_recall_f1.png")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_learning_curves()
    plot_test_auc_ap()
    plot_generalization_gap()
    plot_precision_recall_f1()
    print(f"\nAll figures saved to {OUTPUT_DIR}/")
