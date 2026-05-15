"""
plot_str_sensitivity.py — Parse STR hyperparameter tuning results and generate
sensitivity plots + grid search summary.

Reads the pedestrian_summary.txt files from:
    evaluation/SNPT/hyperparameter_tune/str/<config_name>/pedestrian_summary.txt

Folder naming convention:
    mingapframes{GAP}_thresh{THRESH}_minsegframes{SEG}_nduration{N}

Produces:
  1. CSV with all parsed results
  2. Sensitivity line plots (one subplot per parameter, HOTA vs value)
  3. Grid search ranking table (top configs by HOTA)

Usage:
    python experiments/attributes/plot_str_sensitivity.py
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STR_TUNE_DIR = REPO_ROOT / "evaluation" / "SNPT" / "hyperparameter_tune" / "str"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Default anchor for sensitivity (one-at-a-time) analysis
DEFAULTS = {
    "mingap":    5,
    "thresh":    0.15,
    "minseg":    5,
    "nsamples":  20,
}

PARAM_DISPLAY = {
    "thresh":    "ReID cosine threshold",
    "mingap":    "Min gap size (frames)",
    "minseg":    "Min segment length (frames)",
    "nsamples":  "N embedding samples",
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_folder_name(name: str) -> Dict[str, float]:
    """Extract parameters from folder name like
    mingapframes5_thresh0.15_minsegframes5_nduration20"""
    m = re.match(
        r"mingapframes(\d+)_thresh([\d.]+)_minsegframes(\d+)_nduration(\d+)",
        name,
    )
    if not m:
        return None
    return {
        "mingap":   int(m.group(1)),
        "thresh":   float(m.group(2)),
        "minseg":   int(m.group(3)),
        "nsamples": int(m.group(4)),
    }


def parse_summary(summary_path: Path) -> Dict[str, float]:
    """Parse pedestrian_summary.txt and return dict of metrics."""
    with open(summary_path, "r") as f:
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


def load_all_results() -> List[dict]:
    """Scan all STR tuning folders and collect results."""
    results = []
    if not STR_TUNE_DIR.exists():
        print(f"[ERROR] Tuning directory not found: {STR_TUNE_DIR}")
        return results

    for folder in sorted(STR_TUNE_DIR.iterdir()):
        if not folder.is_dir():
            continue
        summary = folder / "pedestrian_summary.txt"
        if not summary.exists():
            continue

        params = parse_folder_name(folder.name)
        if params is None:
            print(f"  [WARN] Could not parse folder name: {folder.name}")
            continue

        metrics = parse_summary(summary)
        if metrics is None:
            continue

        results.append({
            **params,
            "folder": folder.name,
            "HOTA": metrics.get("HOTA", 0.0),
            "DetA": metrics.get("DetA", 0.0),
            "AssA": metrics.get("AssA", 0.0),
            "IDF1": metrics.get("IDF1", 0.0),
            "MOTA": metrics.get("MOTA", 0.0),
            "IDSW": int(metrics.get("IDSW", 0)),
        })

    return results


# ---------------------------------------------------------------------------
# Sensitivity extraction
# ---------------------------------------------------------------------------

def extract_sensitivity(results: List[dict], param_name: str) -> List[Tuple[float, float]]:
    """
    Extract (param_value, HOTA) pairs where only param_name varies
    and all other params match the defaults.
    """
    other_params = [p for p in DEFAULTS if p != param_name]
    points = []

    for r in results:
        match = True
        for p in other_params:
            if abs(float(r[p]) - float(DEFAULTS[p])) > 0.001:
                match = False
                break
        if match:
            points.append((float(r[param_name]), r["HOTA"]))

    points.sort(key=lambda x: x[0])
    return points


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sensitivity(results: List[dict], output_path: Path):
    """Create sensitivity line plots, one subplot per parameter."""
    params_to_plot = ["thresh", "mingap", "minseg", "nsamples"]

    sensitivity_data = {}
    for param in params_to_plot:
        points = extract_sensitivity(results, param)
        if len(points) >= 2:
            sensitivity_data[param] = points

    n_params = len(sensitivity_data)
    if n_params == 0:
        print("No sensitivity data to plot.")
        return

    n_cols = min(2, n_params)
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.0 * n_rows))
    if n_params == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    # Compute shared y-axis limits across all subplots
    all_hotas = []
    for points in sensitivity_data.values():
        all_hotas.extend([p[1] for p in points])
    global_min = min(all_hotas)
    global_max = max(all_hotas)
    global_pad = (global_max - global_min) * 0.15
    y_lo = global_min - global_pad
    y_hi = global_max + global_pad

    for idx, (param, points) in enumerate(sensitivity_data.items()):
        ax = axes[idx]
        values = [p[0] for p in points]
        hotas = [p[1] for p in points]

        ax.plot(values, hotas, "o-", color="#2563eb", linewidth=2, markersize=7,
                zorder=3)

        # Mark default
        default_val = DEFAULTS[param]
        ax.axvline(x=default_val, color="#dc2626", linestyle="--",
                   alpha=0.6, linewidth=1.5, label=f"Default ({default_val})")

        # Formatting
        display = PARAM_DISPLAY.get(param, param)
        ax.set_xlabel(display, fontsize=10)
        ax.set_ylabel("HOTA", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=8, loc="best")

        # Show range
        hota_range = max(hotas) - min(hotas)
        ax.set_title(f"Range: {hota_range:.3f}", fontsize=9, color="#666")

        # Shared y-axis so relative sensitivity is visually comparable
        ax.set_ylim(y_lo, y_hi)

    for idx in range(n_params, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("STR Splitter: Parameter Sensitivity (HOTA)", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_grid_ranking(results: List[dict], output_path: Path, top_n: int = 15):
    """Horizontal bar chart of top N configurations ranked by HOTA."""
    sorted_results = sorted(results, key=lambda x: x["HOTA"], reverse=True)[:top_n]

    labels = []
    hotas = []
    for r in reversed(sorted_results):
        label = (f"gap={r['mingap']} thr={r['thresh']:.2f} "
                 f"seg={r['minseg']} n={r['nsamples']}")
        labels.append(label)
        hotas.append(r["HOTA"])

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.35)))

    hota_min, hota_max = min(hotas), max(hotas)
    norm_hotas = [(h - hota_min) / (hota_max - hota_min + 1e-9) for h in hotas]
    colors = [plt.cm.RdYlGn(0.3 + 0.7 * n) for n in norm_hotas]

    bars = ax.barh(range(len(labels)), hotas, color=colors, edgecolor="white",
                   linewidth=0.5)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8, fontfamily="monospace")
    ax.set_xlabel("HOTA", fontsize=10)
    ax.set_title(f"STR Splitter: Top {top_n} Configurations", fontsize=12,
                 fontweight="bold")

    ax.set_xlim(min(hotas) - 0.1, max(hotas) + 0.15)
    ax.grid(True, axis="x", alpha=0.3)

    for bar, val in zip(bars, hotas):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8, color="#333")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_results_csv(results: List[dict], path: Path):
    sorted_results = sorted(results, key=lambda x: -x["HOTA"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "min_gap_frames", "reid_threshold", "min_segment_frames",
            "n_samples", "HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW",
            "folder",
        ])
        for r in sorted_results:
            w.writerow([
                r["mingap"], r["thresh"], r["minseg"], r["nsamples"],
                r["HOTA"], r["DetA"], r["AssA"], r["IDF1"],
                r["MOTA"], r["IDSW"], r["folder"],
            ])
    print(f"Saved: {path}")


def save_sensitivity_csv(results: List[dict], path: Path):
    params_to_plot = ["thresh", "mingap", "minseg", "nsamples"]
    param_to_csv_name = {
        "thresh":   "reid_threshold",
        "mingap":   "min_gap_frames",
        "minseg":   "min_segment_frames",
        "nsamples": "n_samples",
    }

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value", "HOTA", "n_splits"])
        for param in params_to_plot:
            points = extract_sensitivity(results, param)
            csv_name = param_to_csv_name.get(param, param)
            for val, hota in points:
                w.writerow([csv_name, val, f"{hota:.4f}", ""])
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("STR SPLITTER — HYPERPARAMETER TUNING RESULTS")
    print("=" * 70)

    results = load_all_results()
    print(f"Loaded {len(results)} configurations\n")

    if not results:
        return

    # Print summary table
    sorted_results = sorted(results, key=lambda x: -x["HOTA"])
    print(f"{'Config':55s} {'HOTA':>7} {'AssA':>7} {'IDF1':>7} {'IDSW':>5}")
    print("-" * 85)
    for r in sorted_results:
        print(f"{r['folder']:55s} {r['HOTA']:>7.3f} {r['AssA']:>7.3f} "
              f"{r['IDF1']:>7.3f} {r['IDSW']:>5d}")

    # Sensitivity summary
    print(f"\n{'='*70}")
    print("SENSITIVITY SUMMARY (one-at-a-time, default anchor: "
          f"gap={DEFAULTS['mingap']} thresh={DEFAULTS['thresh']} "
          f"seg={DEFAULTS['minseg']} n={DEFAULTS['nsamples']})")
    print(f"{'='*70}")

    for param in ["thresh", "mingap", "minseg", "nsamples"]:
        points = extract_sensitivity(results, param)
        if not points:
            print(f"\n  {param}: NO DATA")
            continue
        hotas = [h for _, h in points]
        hota_range = max(hotas) - min(hotas)
        sens = "HIGH" if hota_range > 0.5 else "LOW" if hota_range < 0.1 else "MODERATE"
        print(f"\n  {PARAM_DISPLAY.get(param, param)} (range: {hota_range:.3f}, {sens}):")
        for val, hota in points:
            marker = " *" if abs(val - DEFAULTS[param]) < 0.001 else ""
            print(f"    {val:>8} → HOTA {hota:.3f}{marker}")

    # Save outputs
    save_results_csv(results, OUTPUT_DIR / "str_gridsearch_results.csv")
    save_sensitivity_csv(results, OUTPUT_DIR / "str_sensitivity.csv")
    plot_sensitivity(results, OUTPUT_DIR / "str_sensitivity.png")
    plot_grid_ranking(results, OUTPUT_DIR / "str_ranking.png")


if __name__ == "__main__":
    main()
