"""
Generate thesis figures for transformer training data strategy.

This script compares the original (XGBoost-matched) training data against the
real-synthetic training data for the three transformer architectures.

Outputs:
    validation_auc_data_strategy.png
    generalization_gap_data_strategy.png
"""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


ROOT = Path(
    r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development"
    r"\tracklet_splitter_scratch"
)

OUTPUT_DIR = ROOT / "experiments" / "tracklet_merger" / "transformers" / "good_images"

SMOOTHING_WINDOW = 3

ARCHITECTURES = [
    {
        "key": "cross",
        "label": "Cross Attention",
        "color": "#0072B2",
        "original": ROOT
        / "experiments/tracklet_merger/transformers/cross_attention/output"
        / "cross_attention_xgb_matched/training_log.csv",
        "synthetic": ROOT
        / "experiments/tracklet_merger/transformers/cross_attention/output"
        / "cross_attention_simple_synth/training_log.csv",
    },
    {
        "key": "siamese",
        "label": "Siamese CLS",
        "color": "#D55E00",
        "original": ROOT
        / "experiments/tracklet_merger/transformers/siamese_cls/output"
        / "siamese_cls_xgb_matched/training_log.csv",
        # Use the finalized base-size Siamese CLS run for the synthetic setting.
        "synthetic": ROOT
        / "experiments/tracklet_merger/transformers/siamese_cls/runs"
        / "ablation_base_d80_ff128/training_log.csv",
    },
    {
        "key": "hybrid",
        "label": "Hybrid",
        "color": "#CC79A7",
        "original": ROOT
        / "experiments/tracklet_merger/transformers/hybrid/output"
        / "hybrid_xgb_matched/training_log.csv",
        "synthetic": ROOT
        / "experiments/tracklet_merger/transformers/hybrid/output"
        / "hybrid_simple_synth/training_log.csv",
    },
]


def read_log(path: Path) -> dict[str, np.ndarray]:
    """Read a training CSV log into numpy arrays."""
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Empty training log: {path}")

    result = {}
    for key in rows[0]:
        result[key] = np.array([float(row[key]) for row in rows], dtype=float)
    return result


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average while preserving length."""
    if window <= 1:
        return values
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def load_all_runs() -> dict[str, dict[str, dict]]:
    """Load all configured logs."""
    loaded = {"original": {}, "synthetic": {}}
    for arch in ARCHITECTURES:
        for strategy in ["original", "synthetic"]:
            log = read_log(arch[strategy])
            loaded[strategy][arch["key"]] = {
                "label": arch["label"],
                "color": arch["color"],
                "log": log,
            }
            print(
                f"{strategy:9s} | {arch['label']:16s} | "
                f"{len(log['epoch']):3d} epochs | "
                f"best val AUC = {max(log['val_auc']):.4f}"
            )
    return loaded


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
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
        }
    )


def plot_validation_auc(runs: dict[str, dict[str, dict]]) -> None:
    """Plot validation AUC only, split by training data strategy."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    panels = [
        ("original", "Original training data"),
        ("synthetic", "Real-synthetic training data"),
    ]

    all_val_auc = []
    for strategy, _ in panels:
        for run in runs[strategy].values():
            all_val_auc.append(run["log"]["val_auc"])
    all_val_auc = np.concatenate(all_val_auc)
    y_min = max(0.0, float(np.min(all_val_auc)) - 0.03)
    y_max = min(1.0, float(np.max(all_val_auc)) + 0.03)

    for ax, (strategy, title) in zip(axes, panels):
        for arch in ARCHITECTURES:
            run = runs[strategy][arch["key"]]
            log = run["log"]
            ax.plot(
                log["epoch"],
                smooth(log["val_auc"], SMOOTHING_WINDOW),
                color=run["color"],
                linewidth=1.5,
                alpha=0.95,
                label=run["label"],
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    axes[0].set_ylabel("Validation AUC")
    axes[1].legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()

    out = OUTPUT_DIR / "validation_auc_data_strategy.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def plot_generalization_gap(runs: dict[str, dict[str, dict]]) -> None:
    """Plot train AUC minus validation AUC, split by training data strategy."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    panels = [
        ("original", "Original training data"),
        ("synthetic", "Real-synthetic training data"),
    ]

    all_gaps = []
    for strategy, _ in panels:
        for run in runs[strategy].values():
            log = run["log"]
            all_gaps.append(log["train_auc"] - log["val_auc"])
    all_gaps = np.concatenate(all_gaps)
    y_min = min(0.0, float(np.min(all_gaps)) - 0.02)
    y_max = float(np.max(all_gaps)) + 0.02

    for ax, (strategy, title) in zip(axes, panels):
        for arch in ARCHITECTURES:
            run = runs[strategy][arch["key"]]
            log = run["log"]
            gap = log["train_auc"] - log["val_auc"]
            ax.plot(
                log["epoch"],
                smooth(gap, SMOOTHING_WINDOW),
                color=run["color"],
                linewidth=1.5,
                alpha=0.95,
                label=run["label"],
            )
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    axes[0].set_ylabel("Train AUC - Val AUC")
    axes[1].legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()

    out = OUTPUT_DIR / "generalization_gap_data_strategy.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Smoothing window: {SMOOTHING_WINDOW}\n")
    runs = load_all_runs()
    print()
    plot_validation_auc(runs)
    plot_generalization_gap(runs)


if __name__ == "__main__":
    main()
