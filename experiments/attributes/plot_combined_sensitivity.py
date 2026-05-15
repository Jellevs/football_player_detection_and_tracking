"""
plot_combined_sensitivity.py — Combined strip plot showing HOTA spread
across all splitter hyperparameter configurations.

Each column is a splitter, each dot is one configuration from the
hyperparameter sweep. The default configuration is highlighted.
This gives an at a glance view of how sensitive each splitter is
to its hyperparameters.

Intended for the main text of the thesis; the detailed per parameter
sensitivity subplots go in the appendix.

Usage:
    python experiments/attributes/plot_combined_sensitivity.py
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def _long(p: Path) -> Path:
    """Prefix with \\\\?\\ on Windows so paths > 260 chars work."""
    s = str(p)
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        return Path("\\\\?\\" + s)
    return p

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TUNE_ROOT = REPO_ROOT / "evaluation" / "SNPT" / "hyperparameter_tune"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Splitter definitions
# ---------------------------------------------------------------------------

SPLITTERS = {
    "Jersey": {
        "dir": TUNE_ROOT / "jersey",
        "parse": lambda name: _parse_jersey(name),
        "defaults": {
            "minpersistence": 20, "lookahead": 100,
            "persistenceratio": 0.9, "entropy": 0.01,
        },
    },
    "Team": {
        "dir": TUNE_ROOT / "team",
        "parse": lambda name: _parse_team(name),
        "defaults": {
            "persistenceratio": 0.9, "lookahead": 100,
            "confthresh": 0.6,
        },
    },
    "Bbox Anomaly": {
        "dir": TUNE_ROOT / "bbox",
        "parse": lambda name: _parse_bbox(name),
        "defaults": {
            "lookback": 20, "lookahead": 10,
            "std": 3.0, "minspike": 35.0, "maxspike": 3,
        },
    },
    "Temporal ReID": {
        "dir": TUNE_ROOT / "str",
        "parse": lambda name: _parse_str(name),
        "defaults": {
            "mingap": 5, "thresh": 0.1,
            "minseg": 5, "nsamples": 20,
        },
    },
    "Trajectory": {
        "dir": TUNE_ROOT / "traj",
        "parse": lambda name: _parse_traj(name),
        "defaults": {
            "proxdist": 50, "velwindow": 5, "minvelchange": 30.0,
            "dirthresh": 180, "swapsim": 0.95,
        },
    },
}

# ---------------------------------------------------------------------------
# Folder name parsers
# ---------------------------------------------------------------------------

def _parse_jersey(name: str) -> Optional[dict]:
    m = re.match(
        r"minpersistence(\d+)_lookahead(\d+)_persistenceratio([\d.]+)_entropy([\d.]+)",
        name,
    )
    if not m:
        return None
    return {
        "minpersistence": int(m.group(1)),
        "lookahead": int(m.group(2)),
        "persistenceratio": float(m.group(3)),
        "entropy": float(m.group(4)),
    }


def _parse_team(name: str) -> Optional[dict]:
    m = re.match(
        r"persistenceratio([\d.]+)_lookahead(\d+)_confthresh([\d.]+)",
        name,
    )
    if not m:
        return None
    return {
        "persistenceratio": float(m.group(1)),
        "lookahead": int(m.group(2)),
        "confthresh": float(m.group(3)),
    }


def _parse_bbox(name: str) -> Optional[dict]:
    m = re.match(
        r"lookback_(\d+)_lookahead_(\d+)_std(\d+\.?\d*)_minspike([\d.]+)_maxspike(\d+)",
        name,
    )
    if not m:
        return None
    return {
        "lookback": int(m.group(1)),
        "lookahead": int(m.group(2)),
        "std": float(m.group(3)),
        "minspike": float(m.group(4)),
        "maxspike": int(m.group(5)),
    }


def _parse_str(name: str) -> Optional[dict]:
    m = re.match(
        r"mingapframes(\d+)_thresh([\d.]+)_minsegframes(\d+)_nduration(\d+)",
        name,
    )
    if not m:
        return None
    return {
        "mingap": int(m.group(1)),
        "thresh": float(m.group(2)),
        "minseg": int(m.group(3)),
        "nsamples": int(m.group(4)),
    }


def _parse_traj(name: str) -> Optional[dict]:
    """Parse trajectory folder names."""
    m = re.match(
        r"proximity_distance(\d+)_velocity_window(\d+)_min_velocity_change([\d.]+)_"
        r"direction_change_threshold(\d+)_swap_similarity_threshold([\d.]+)",
        name,
    )
    if not m:
        return None
    return {
        "proxdist": int(m.group(1)),
        "velwindow": int(m.group(2)),
        "minvelchange": float(m.group(3)),
        "dirthresh": int(m.group(4)),
        "swapsim": float(m.group(5)),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_summary(summary_path: Path) -> Optional[Dict[str, float]]:
    with open(_long(summary_path), "r") as f:
        lines = f.readlines()
    if len(lines) < 2:
        return None
    headers = lines[0].strip().split()
    values = lines[1].strip().split()
    metrics = {}
    for h, v in zip(headers, values):
        try:
            metrics[h] = float(v)
        except ValueError:
            pass
    return metrics


def load_splitter_results(splitter_cfg: dict) -> List[dict]:
    """Load all configs for one splitter. Returns list of {params, HOTA, is_default}."""
    results = []
    tune_dir = splitter_cfg["dir"]
    if not tune_dir.exists():
        return results

    defaults = splitter_cfg["defaults"]

    for folder in sorted(tune_dir.iterdir()):
        if not folder.is_dir():
            continue
        params = splitter_cfg["parse"](folder.name)
        if params is None:
            continue

        summary = folder / "pedestrian_summary.txt"
        try:
            metrics = parse_summary(summary)
        except (OSError, IOError):
            continue
        if metrics is None:
            continue

        # Check if this is the default configuration
        is_default = all(
            abs(float(params[k]) - float(defaults[k])) < 0.01
            for k in defaults
        )

        results.append({
            "params": params,
            "HOTA": metrics.get("HOTA", 0.0),
            "is_default": is_default,
            "folder": folder.name,
        })

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_combined(output_path: Path):
    """Strip plot: one column per splitter, each dot is a configuration."""

    data = {}
    for name, cfg in SPLITTERS.items():
        results = load_splitter_results(cfg)
        if results:
            data[name] = results

    if not data:
        print("No data found for any splitter.")
        return

    n_splitters = len(data)
    splitter_names = list(data.keys())

    fig, ax = plt.subplots(figsize=(2.5 + 1.8 * n_splitters, 5))

    colors = {
        "Jersey": "#8b5cf6",
        "Team": "#ec4899",
        "Bbox Anomaly": "#2563eb",
        "Temporal ReID": "#16a34a",
        "Trajectory": "#d97706",
    }

    for i, name in enumerate(splitter_names):
        results = data[name]
        hotas = [r["HOTA"] for r in results]
        defaults_mask = [r["is_default"] for r in results]
        color = colors.get(name, "#666")

        # Jitter x positions for visibility
        rng = np.random.RandomState(42)
        jitter = rng.uniform(-0.15, 0.15, size=len(hotas))
        x_pos = np.full(len(hotas), i) + jitter

        # Plot all non-default points
        non_def_x = [x for x, d in zip(x_pos, defaults_mask) if not d]
        non_def_h = [h for h, d in zip(hotas, defaults_mask) if not d]
        ax.scatter(non_def_x, non_def_h, c=color, s=50, alpha=0.6,
                   edgecolors="white", linewidths=0.5, zorder=3)

        # Highlight default
        # def_x = [x for x, d in zip(x_pos, defaults_mask) if d]
        # def_h = [h for h, d in zip(hotas, defaults_mask) if d]
        # if def_x:
        #     ax.scatter(def_x, def_h, c=color, s=120, alpha=1.0,
        #                edgecolors="black", linewidths=1.5, zorder=4,
        #                marker="D", label=f"Default" if i == 0 else None)

        # Show range annotation
        hota_range = max(hotas) - min(hotas)
        ax.annotate(
            f"range: {hota_range:.2f}",
            xy=(i, max(hotas)),
            xytext=(0, 12), textcoords="offset points",
            ha="center", fontsize=8, color="#666",
        )

        # Horizontal lines for min/max
        ax.hlines([min(hotas), max(hotas)], i - 0.25, i + 0.25,
                  colors=color, alpha=0.3, linewidths=1)

    ax.set_xticks(range(n_splitters))
    ax.set_xticklabels(splitter_names, fontsize=11)
    ax.set_ylabel("HOTA", fontsize=11)
    ax.set_title("Splitter Hyperparameter Sensitivity Overview", fontsize=13,
                 fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # Add legend for the default marker
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markeredgecolor="black",
               markerfacecolor="gray", markersize=8, label="Default config"),
        Line2D([0], [0], marker="o", color="w", markeredgecolor="white",
               markerfacecolor="gray", markersize=7, alpha=0.6,
               label="Sweep config"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("COMBINED SPLITTER SENSITIVITY OVERVIEW")
    print("=" * 70)

    for name, cfg in SPLITTERS.items():
        results = load_splitter_results(cfg)
        if results:
            hotas = [r["HOTA"] for r in results]
            print(f"  {name}: {len(results)} configs, "
                  f"HOTA range [{min(hotas):.3f}, {max(hotas):.3f}], "
                  f"spread = {max(hotas) - min(hotas):.3f}")
        else:
            print(f"  {name}: no results found")

    plot_combined(OUTPUT_DIR / "splitter_sensitivity_overview.png")


if __name__ == "__main__":
    main()
