"""
plot_team_sensitivity.py — Parse team hyperparameter tuning results and generate
sensitivity plots + grid search summary.

Reads the pedestrian_summary.txt files from:
    evaluation/SNPT/hyperparameter_tune/team/<config_name>/pedestrian_summary.txt

Folder naming convention:
    persistenceratio{PR}_lookahead{LA}_confthresh{CT}

Produces:
  1. CSV with all parsed results
  2. Sensitivity line plots (one subplot per parameter, HOTA vs value)
  3. Grid search ranking table (top configs by HOTA)

Usage:
    python experiments/attributes/plot_team_sensitivity.py
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
TEAM_TUNE_DIR = REPO_ROOT / "evaluation" / "SNPT" / "hyperparameter_tune" / "team"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Default anchor for sensitivity (one-at-a-time) analysis
DEFAULTS = {
    "persistenceratio": 0.9,
    "lookahead":        100,
    "confthresh":       0.7,
}

PARAM_DISPLAY = {
    "confthresh":       "Confidence threshold",
    "lookahead":        "Lookahead window (frames)",
    "persistenceratio": "Min persistence ratio",
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_folder_name(name: str) -> Dict[str, float]:
    m = re.match(
        r"persistenceratio([\d.]+)_lookahead(\d+)_confthresh([\d.]+)",
        name,
    )
    if not m:
        return None
    return {
        "persistenceratio": float(m.group(1)),
        "lookahead":        int(m.group(2)),
        "confthresh":       float(m.group(3)),
    }


def parse_summary(summary_path: Path) -> Dict[str, float]:
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
    results = []
    if not TEAM_TUNE_DIR.exists():
        print(f"[ERROR] Tuning directory not found: {TEAM_TUNE_DIR}")
        return results

    for folder in sorted(TEAM_TUNE_DIR.iterdir()):
        if not folder.is_dir():
            continue
        summary = folder / "pedestrian_summary.txt"
        if not summary.exists():
            continue

        params = parse_folder_name(folder.name)
        if params is None:
            print(f"  [WARN] Could not parse folder name: {folder.name}")
            continue

        try:
            metrics = parse_summary(summary)
        except (OSError, IOError):
            print(f"  [WARN] Could not read: {summary}")
            continue
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
    other_params = [p for p in DEFAULTS if p != param_name]
    points = []

    for r in results:
        match = True
        for p in other_params:
            if abs(float(r[p]) - float(DEFAULTS[p])) > 0.01:
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
    params_to_plot = ["confthresh", "lookahead", "persistenceratio"]

    sensitivity_data = {}
    for param in params_to_plot:
        points = extract_sensitivity(results, param)
        if len(points) >= 2:
            sensitivity_data[param] = points

    n_params = len(sensitivity_data)
    if n_params == 0:
        print("No sensitivity data to plot.")
        return

    n_cols = min(3, n_params)
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows))
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

        default_val = DEFAULTS[param]
        ax.axvline(x=default_val, color="#dc2626", linestyle="--",
                   alpha=0.6, linewidth=1.5, label=f"Default ({default_val})")

        display = PARAM_DISPLAY.get(param, param)
        ax.set_xlabel(display, fontsize=10)
        ax.set_ylabel("HOTA", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=8, loc="best")

        hota_range = max(hotas) - min(hotas)
        ax.set_title(f"Range: {hota_range:.3f}", fontsize=9, color="#666")

        # Shared y-axis so relative sensitivity is visually comparable
        ax.set_ylim(y_lo, y_hi)

    for idx in range(n_params, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Team Splitter: Parameter Sensitivity (HOTA)", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_grid_ranking(results: List[dict], output_path: Path, top_n: int = 15):
    sorted_results = sorted(results, key=lambda x: x["HOTA"], reverse=True)[:top_n]

    labels = []
    hotas = []
    for r in reversed(sorted_results):
        label = (f"pr={r['persistenceratio']:.1f} la={r['lookahead']} "
                 f"ct={r['confthresh']:.1f}")
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
    ax.set_title(f"Team Splitter: Top {top_n} Configurations", fontsize=12,
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
            "persistence_ratio", "lookahead", "confidence_threshold",
            "HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "folder",
        ])
        for r in sorted_results:
            w.writerow([
                r["persistenceratio"], r["lookahead"], r["confthresh"],
                r["HOTA"], r["DetA"], r["AssA"], r["IDF1"],
                r["MOTA"], r["IDSW"], r["folder"],
            ])
    print(f"Saved: {path}")


def save_sensitivity_csv(results: List[dict], path: Path):
    params_to_plot = ["confthresh", "lookahead", "persistenceratio"]
    param_to_csv_name = {
        "confthresh": "confidence_threshold",
        "lookahead": "lookahead_window",
        "persistenceratio": "persistence_ratio",
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
    print("TEAM SPLITTER — HYPERPARAMETER TUNING RESULTS")
    print("=" * 70)

    results = load_all_results()
    print(f"Loaded {len(results)} configurations\n")

    if not results:
        return

    sorted_results = sorted(results, key=lambda x: -x["HOTA"])
    print(f"{'Config':55s} {'HOTA':>7} {'AssA':>7} {'IDF1':>7} {'IDSW':>5}")
    print("-" * 85)
    for r in sorted_results:
        print(f"{r['folder']:55s} {r['HOTA']:>7.3f} {r['AssA']:>7.3f} "
              f"{r['IDF1']:>7.3f} {r['IDSW']:>5d}")

    print(f"\n{'='*70}")
    print("SENSITIVITY SUMMARY (one-at-a-time, default anchor: "
          f"pr={DEFAULTS['persistenceratio']} la={DEFAULTS['lookahead']} "
          f"ct={DEFAULTS['confthresh']})")
    print(f"{'='*70}")

    for param in ["confthresh", "lookahead", "persistenceratio"]:
        points = extract_sensitivity(results, param)
        if not points:
            print(f"\n  {param}: NO DATA")
            continue
        hotas = [h for _, h in points]
        hota_range = max(hotas) - min(hotas)
        sens = "HIGH" if hota_range > 0.5 else "LOW" if hota_range < 0.1 else "MODERATE"
        print(f"\n  {PARAM_DISPLAY.get(param, param)} (range: {hota_range:.3f}, {sens}):")
        for val, hota in points:
            marker = " *" if abs(val - DEFAULTS[param]) < 0.01 else ""
            print(f"    {val:>8} -> HOTA {hota:.3f}{marker}")

    # Save outputs
    save_results_csv(results, OUTPUT_DIR / "team_gridsearch_results.csv")
    save_sensitivity_csv(results, OUTPUT_DIR / "team_sensitivity.csv")
    plot_sensitivity(results, OUTPUT_DIR / "team_sensitivity.png")
    plot_grid_ranking(results, OUTPUT_DIR / "team_ranking.png")


if __name__ == "__main__":
    main()
