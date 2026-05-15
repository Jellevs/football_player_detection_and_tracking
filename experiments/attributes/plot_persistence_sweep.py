"""
plot_persistence_sweep.py — Plot precision vs recall tradeoff from parameter sweep results.

Creates a side-by-side figure showing the precision/recall tradeoff for
jersey and team persistence filter configurations, with the chosen
configuration highlighted.

Usage:
    python experiments/attributes/plot_persistence_sweep.py
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
})

OUTPUT_DIR = Path(__file__).parent / "output"
JERSEY_CSV = OUTPUT_DIR / "jersey_sweep.csv"
TEAM_CSV = OUTPUT_DIR / "team_sweep.csv"

# Current chosen settings
JERSEY_CURRENT = {"min_persistence": 20, "threshold": 0.02, "lookahead": 150, "min_persistence_ratio": 0.9}
TEAM_CURRENT = {"min_persistence": 10, "threshold": 0.7, "lookahead": 100, "min_persistence_ratio": 0.9}


def load_csv(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "min_persistence": int(row["min_persistence"]),
                "threshold": float(row["threshold"]),
                "lookahead": int(row["lookahead"]),
                "min_persistence_ratio": float(row["min_persistence_ratio"]),
                "total_splits": int(row["total_splits"]),
                "true_splits": int(row["true_splits"]),
                "false_at_switch": int(row["false_splits_at_switch"]),
                "spurious": int(row["spurious_splits"]),
                "total_false": int(row["total_false_splits"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
            })
    return rows


def is_current(row, current):
    return (row["min_persistence"] == current["min_persistence"]
            and abs(row["threshold"] - current["threshold"]) < 1e-6
            and row["lookahead"] == current["lookahead"]
            and abs(row["min_persistence_ratio"] - current["min_persistence_ratio"]) < 1e-6)


def compute_pareto_front(rows):
    """
    Compute the Pareto front: configurations not dominated on both precision and recall.
    A point is dominated if another point has both higher precision AND higher recall.
    Returns the Pareto-optimal rows sorted by recall.
    """
    pareto = []
    for r in rows:
        dominated = False
        for other in rows:
            if other is r:
                continue
            if (other["precision"] >= r["precision"] and other["recall"] >= r["recall"]
                    and (other["precision"] > r["precision"] or other["recall"] > r["recall"])):
                dominated = True
                break
        if not dominated:
            pareto.append(r)
    pareto.sort(key=lambda r: r["recall"])
    return pareto


def plot_precision_recall(ax, rows, current, title, color_by="lookahead"):
    """Plot precision vs recall with points colored by a parameter, plus Pareto front."""

    # Get unique values for coloring
    if color_by == "lookahead":
        unique_vals = sorted(set(r["lookahead"] for r in rows))
        cmap = plt.cm.viridis
    elif color_by == "threshold":
        unique_vals = sorted(set(r["threshold"] for r in rows))
        cmap = plt.cm.plasma
    else:
        unique_vals = sorted(set(r["min_persistence"] for r in rows))
        cmap = plt.cm.coolwarm

    colors = {v: cmap(i / max(1, len(unique_vals) - 1)) for i, v in enumerate(unique_vals)}

    # Plot all points
    for val in unique_vals:
        subset = [r for r in rows if r[color_by] == val]
        recalls = [r["recall"] * 100 for r in subset]
        precisions = [r["precision"] * 100 for r in subset]

        if color_by == "lookahead":
            label = f"lookahead={val}"
        elif color_by == "threshold":
            label = f"thr={val}"
        else:
            label = f"mp={val}"

        ax.scatter(recalls, precisions, c=[colors[val]], s=30, alpha=0.5,
                   label=label, edgecolors="none")

    # Compute and plot Pareto front
    pareto = compute_pareto_front(rows)
    pareto_recalls = [r["recall"] * 100 for r in pareto]
    pareto_precisions = [r["precision"] * 100 for r in pareto]

    # Draw step-style line connecting Pareto front points
    # For a proper Pareto front visualization, use a step function
    # that extends horizontally then drops vertically
    extended_r = []
    extended_p = []
    for i, (rec, prec) in enumerate(zip(pareto_recalls, pareto_precisions)):
        if i > 0:
            # Horizontal step from previous precision to current recall
            extended_r.append(rec)
            extended_p.append(pareto_precisions[i - 1])
        extended_r.append(rec)
        extended_p.append(prec)

    ax.plot(extended_r, extended_p, color="red", linewidth=1.5, alpha=0.7,
            linestyle="--", label="Pareto front", zorder=5)
    ax.scatter(pareto_recalls, pareto_precisions, c="red", s=18, alpha=0.8,
               edgecolors="darkred", linewidths=0.5, zorder=6)

    # Highlight current setting
    current_row = [r for r in rows if is_current(r, current)]
    if current_row:
        r = current_row[0]
        ax.scatter([r["recall"] * 100], [r["precision"] * 100],
                   c="red", s=150, marker="*", zorder=10, edgecolors="black",
                   linewidths=0.8, label="Selected")

    ax.set_xlabel("Recall (%)")
    ax.set_ylabel("Precision (%)")
    ax.set_title(title)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 105)


def plot_split_breakdown(ax, rows, current, title):
    """Bar chart showing true vs spurious vs false@switch for selected configs."""
    # Select a subset of interesting configurations to show
    # Sort by total_splits to get a range from conservative to aggressive
    rows_sorted = sorted(rows, key=lambda r: r["total_splits"])

    # Pick ~8 evenly spaced configs
    n = len(rows_sorted)
    if n <= 8:
        selected = rows_sorted
    else:
        indices = np.linspace(0, n - 1, 8, dtype=int)
        selected = [rows_sorted[i] for i in indices]

    # Make sure current is included
    current_row = [r for r in rows if is_current(r, current)]
    if current_row and current_row[0] not in selected:
        selected.append(current_row[0])
        selected = sorted(selected, key=lambda r: r["total_splits"])

    x = np.arange(len(selected))
    width = 0.6

    true_vals = [r["true_splits"] for r in selected]
    false_vals = [r["false_at_switch"] for r in selected]
    spur_vals = [r["spurious"] for r in selected]

    bars_true = ax.bar(x, true_vals, width, label="True splits", color="#2ecc71")
    bars_false = ax.bar(x, false_vals, width, bottom=true_vals,
                        label="False (at switch)", color="#e74c3c")
    bars_spur = ax.bar(x, spur_vals, width,
                       bottom=[t + f for t, f in zip(true_vals, false_vals)],
                       label="Spurious", color="#f39c12")

    # Labels
    labels = []
    for r in selected:
        is_cur = is_current(r, current)
        if title.startswith("Jersey"):
            lbl = f"mp={r['min_persistence']}\net={r['threshold']}\nla={r['lookahead']}"
        else:
            lbl = f"ct={r['threshold']}\nla={r['lookahead']}\nmpr={r['min_persistence_ratio']}"
        if is_cur:
            lbl += "\n*"
        labels.append(lbl)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5, ha="center")
    ax.set_ylabel("Number of splits")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")


def print_pareto(rows, name, current):
    """Print the Pareto front configurations."""
    pareto = compute_pareto_front(rows)
    print(f"\n{'='*80}")
    print(f"{name.upper()} PARETO FRONT ({len(pareto)} configurations)")
    print(f"{'='*80}")
    print(f"{'mp':>4} {'thresh':>6} {'la':>4} {'mpr':>5} | "
          f"{'true':>5} {'f@sw':>5} {'spur':>5} | "
          f"{'prec':>7} {'recall':>7}  {'note':>10}")
    print("-" * 75)
    for r in pareto:
        is_cur = is_current(r, current)
        note = " <-- selected" if is_cur else ""
        print(f"{r['min_persistence']:>4d} {r['threshold']:>6.3f} {r['lookahead']:>4d} "
              f"{r['min_persistence_ratio']:>5.2f} | "
              f"{r['true_splits']:>5d} {r['false_at_switch']:>5d} {r['spurious']:>5d} | "
              f"{r['precision']*100:>6.1f}% {r['recall']*100:>6.1f}%{note}")


def main():
    jersey_rows = load_csv(JERSEY_CSV)
    team_rows = load_csv(TEAM_CSV)

    # Print Pareto fronts
    print_pareto(jersey_rows, "jersey", JERSEY_CURRENT)
    print_pareto(team_rows, "team", TEAM_CURRENT)

    # --- Figure 1: Precision vs Recall scatter ---
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    plot_precision_recall(ax1, jersey_rows, JERSEY_CURRENT,
                          "Jersey: Precision vs Recall", color_by="lookahead")
    plot_precision_recall(ax2, team_rows, TEAM_CURRENT,
                          "Team: Precision vs Recall", color_by="lookahead")

    fig1.tight_layout()
    fig1.savefig(OUTPUT_DIR / "persistence_precision_recall.pdf", bbox_inches="tight")
    fig1.savefig(OUTPUT_DIR / "persistence_precision_recall.png", bbox_inches="tight", dpi=200)
    print(f"Saved precision-recall plot to {OUTPUT_DIR / 'persistence_precision_recall.pdf'}")

    # --- Figure 2: Split breakdown bars ---
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 4.5))

    plot_split_breakdown(ax3, jersey_rows, JERSEY_CURRENT, "Jersey: Split Breakdown")
    plot_split_breakdown(ax4, team_rows, TEAM_CURRENT, "Team: Split Breakdown")

    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "persistence_split_breakdown.pdf", bbox_inches="tight")
    fig2.savefig(OUTPUT_DIR / "persistence_split_breakdown.png", bbox_inches="tight", dpi=200)
    print(f"Saved split breakdown to {OUTPUT_DIR / 'persistence_split_breakdown.pdf'}")

    plt.show()


if __name__ == "__main__":
    main()
