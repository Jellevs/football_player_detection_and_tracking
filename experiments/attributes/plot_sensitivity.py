"""
plot_sensitivity.py — Visualize parameter sensitivity for splitters.

Reads sensitivity CSV files (produced by bbox_parameter_sweep.py,
str_parameter_sweep.py, trajectory_parameter_sweep.py) and creates:

  1. Line plots: HOTA vs parameter value (one subplot per parameter)
     Shows how sensitive the metric is to each parameter while others
     remain at their default values.

  2. Bar chart: Sensitivity magnitude per parameter (HOTA range)
     Quick comparison of which parameters matter most.

Usage:
    python experiments/attributes/plot_sensitivity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).parent / "output"

# Mapping of nice display names
PARAM_DISPLAY = {
    "std_threshold": "Z-score Threshold",
    "min_spike_velocity": "Min Spike Velocity (px/f)",
    "max_spike_duration": "Max Spike Duration (frames)",
    "lookback_window": "Lookback Window (frames)",
    "lookahead_window": "Lookahead Window (frames)",
    "min_fragment_length": "Min Fragment Length (frames)",
    "reid_threshold": "ReID Cosine Threshold",
    "min_gap_frames": "Min Gap Size (frames)",
    "min_segment_frames": "Min Segment Length (frames)",
    "n_samples": "N Embedding Samples",
    "proximity_distance": "Proximity Distance (px)",
    "min_velocity_change": "Min Velocity Change (px/f)",
    "direction_change_threshold": "Direction Change (degrees)",
    "min_overlap_frames": "Min Overlap Frames",
    "swap_similarity_threshold": "Swap Similarity Threshold",
}


def plot_sensitivity_lines(csv_path: Path, title: str, output_path: Path,
                           defaults: dict = None):
    """
    Create line plots showing HOTA vs each parameter value.
    One subplot per parameter, arranged in a grid.
    """
    df = pd.read_csv(csv_path)
    params = df["parameter"].unique()
    n_params = len(params)

    # Determine grid layout
    n_cols = min(3, n_params)
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows))
    if n_params == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, param in enumerate(params):
        ax = axes[idx]
        subset = df[df["parameter"] == param].sort_values("value")

        values = subset["value"].values
        hotas = subset["HOTA"].values

        ax.plot(values, hotas, "o-", color="#2563eb", linewidth=2, markersize=6)

        # Mark default value
        if defaults and param in defaults:
            default_val = defaults[param]
            ax.axvline(x=default_val, color="#dc2626", linestyle="--",
                       alpha=0.7, linewidth=1.5, label=f"Default ({default_val})")
            ax.legend(fontsize=8)

        # Formatting
        display_name = PARAM_DISPLAY.get(param, param)
        ax.set_xlabel(display_name, fontsize=9)
        ax.set_ylabel("HOTA", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        # Show HOTA range as annotation
        hota_range = hotas.max() - hotas.min()
        ax.set_title(f"Range: {hota_range:.3f}", fontsize=9, color="#666")

    # Hide unused subplots
    for idx in range(n_params, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_sensitivity_bars(csv_paths: dict, output_path: Path):
    """
    Bar chart comparing sensitivity magnitude across splitters and parameters.
    csv_paths: {"Bbox": path, "STR": path, "Trajectory": path}
    """
    all_data = []

    for splitter_name, csv_path in csv_paths.items():
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        for param in df["parameter"].unique():
            subset = df[df["parameter"] == param]
            hota_range = subset["HOTA"].max() - subset["HOTA"].min()
            display_name = PARAM_DISPLAY.get(param, param)
            all_data.append({
                "splitter": splitter_name,
                "parameter": display_name,
                "hota_range": hota_range,
            })

    if not all_data:
        print("No data to plot.")
        return

    result_df = pd.DataFrame(all_data)

    # Sort by hota_range descending
    result_df = result_df.sort_values("hota_range", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(result_df) * 0.4)))

    colors = {"Bbox": "#2563eb", "STR": "#16a34a", "Trajectory": "#dc2626"}
    labels = result_df["splitter"] + ": " + result_df["parameter"]

    bars = ax.barh(
        range(len(result_df)),
        result_df["hota_range"],
        color=[colors.get(s, "#666") for s in result_df["splitter"]],
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_yticks(range(len(result_df)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("HOTA Range (sensitivity)", fontsize=10)
    ax.set_title("Parameter Sensitivity Comparison", fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, result_df["hota_range"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8, color="#333")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Bbox sensitivity ---
    bbox_csv = OUTPUT_DIR / "bbox_sensitivity.csv"
    if bbox_csv.exists():
        bbox_defaults = {
            "std_threshold": 4.0,
            "min_spike_velocity": 35.0,
            "max_spike_duration": 3,
            "lookback_window": 20,
            "lookahead_window": 10,
            "min_fragment_length": 10,
        }
        plot_sensitivity_lines(
            bbox_csv,
            "Bbox Splitter: Parameter Sensitivity",
            OUTPUT_DIR / "bbox_sensitivity.png",
            defaults=bbox_defaults,
        )

    # --- STR sensitivity ---
    str_csv = OUTPUT_DIR / "str_sensitivity.csv"
    if str_csv.exists():
        str_defaults = {
            "reid_threshold": 0.15,
            "min_gap_frames": 5,
            "min_segment_frames": 5,
            "n_samples": 20,
        }
        plot_sensitivity_lines(
            str_csv,
            "STR Splitter: Parameter Sensitivity",
            OUTPUT_DIR / "str_sensitivity.png",
            defaults=str_defaults,
        )

    # --- Trajectory sensitivity ---
    traj_csv = OUTPUT_DIR / "trajectory_sensitivity.csv"
    if traj_csv.exists():
        traj_defaults = {
            "proximity_distance": 50,
            "min_velocity_change": 30.0,
            "direction_change_threshold": 120,
            "min_overlap_frames": 3,
        }
        plot_sensitivity_lines(
            traj_csv,
            "Trajectory Splitter: Parameter Sensitivity",
            OUTPUT_DIR / "trajectory_sensitivity.png",
            defaults=traj_defaults,
        )

    # --- Combined comparison bar chart ---
    csv_paths = {
        "Bbox": bbox_csv,
        "STR": str_csv,
        "Trajectory": traj_csv,
    }
    existing = {k: v for k, v in csv_paths.items() if v.exists()}
    if existing:
        plot_sensitivity_bars(existing, OUTPUT_DIR / "sensitivity_comparison.png")


if __name__ == "__main__":
    main()
