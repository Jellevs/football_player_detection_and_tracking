"""
visualize_qualitative_analysis.py — Generate thesis figures from splitter analysis CSVs.

Reads:
  - splitter_qualitative_analysis.csv   (per-switch details)
  - splitter_per_sequence_summary.csv   (per-sequence aggregates)

Outputs thesis-quality figures to experiments/tracklet_splitter/output/figures/

Usage:
    python experiments/tracklet_splitter/visualize_qualitative_analysis.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_CSV = OUTPUT_DIR / "splitter_qualitative_analysis.csv"
SEQ_CSV = OUTPUT_DIR / "splitter_per_sequence_summary.csv"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLOR_CAUGHT = "#2563eb"
COLOR_UNCAUGHT = "#dc2626"
COLOR_NEUTRAL = "#6b7280"


def load_data():
    df = pd.read_csv(DETAIL_CSV)
    seq_df = pd.read_csv(SEQ_CSV)
    df["caught"] = df["caught"].astype(bool)
    df["has_temporal_gap"] = df["has_temporal_gap"].astype(bool)
    df["same_team"] = df["same_team"].map({1: True, 0: False, "1": True, "0": False})
    return df, seq_df


# =========================================================================
# Figure 1: Same-team vs cross-team catch rates
# =========================================================================
def fig_team_breakdown(df: pd.DataFrame):
    has_team = df.dropna(subset=["same_team"])
    same = has_team[has_team["same_team"] == True]
    cross = has_team[has_team["same_team"] == False]

    categories = ["Same team", "Cross team"]
    totals = [len(same), len(cross)]
    caught_counts = [same["caught"].sum(), cross["caught"].sum()]
    uncaught_counts = [t - c for t, c in zip(totals, caught_counts)]
    catch_rates = [c / t * 100 if t > 0 else 0 for c, t in zip(caught_counts, totals)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: stacked bar (counts)
    x = np.arange(len(categories))
    w = 0.5
    ax1.bar(x, caught_counts, w, label="Caught", color=COLOR_CAUGHT, alpha=0.85)
    ax1.bar(x, uncaught_counts, w, bottom=caught_counts, label="Uncaught", color=COLOR_UNCAUGHT, alpha=0.55)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.set_ylabel("Number of identity switches")
    ax1.set_title("(a) Distribution by team relationship")
    ax1.legend()

    for i, (c, u, t) in enumerate(zip(caught_counts, uncaught_counts, totals)):
        ax1.text(i, t + 5, f"n={t}", ha="center", fontsize=10, color=COLOR_NEUTRAL)

    # Right: catch rate comparison
    bars = ax2.bar(x, catch_rates, w, color=[COLOR_CAUGHT, COLOR_CAUGHT], alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel("Catch rate (%)")
    ax2.set_title("(b) Catch rate by team relationship")
    ax2.set_ylim(0, max(catch_rates) * 1.4)

    for i, (rate, c, t) in enumerate(zip(catch_rates, caught_counts, totals)):
        ax2.text(i, rate + 0.5, f"{rate:.1f}%\n({c}/{t})", ha="center", fontsize=10)

    fig.suptitle("Same-team vs cross-team identity switches", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "team_breakdown.pdf")
    fig.savefig(FIG_DIR / "team_breakdown.png")
    plt.close(fig)
    print(f"  Saved: team_breakdown.pdf/png")


# =========================================================================
# Figure 2: Switch duration distribution — caught vs uncaught
# =========================================================================
def fig_duration_distribution(df: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: histogram of durations (log scale x-axis, caught vs uncaught)
    caught = df[df["caught"]]
    uncaught = df[~df["caught"]]

    bins_log = [0, 2, 5, 10, 25, 50, 100, 200, 400, 800]
    ax1.hist(uncaught["duration_frames"].clip(upper=799), bins=bins_log,
             alpha=0.6, color=COLOR_UNCAUGHT, label=f"Uncaught (n={len(uncaught)})", edgecolor="white")
    ax1.hist(caught["duration_frames"].clip(upper=799), bins=bins_log,
             alpha=0.85, color=COLOR_CAUGHT, label=f"Caught (n={len(caught)})", edgecolor="white")
    ax1.set_xlabel("Switch duration (frames)")
    ax1.set_ylabel("Count")
    ax1.set_title("(a) Duration distribution")
    ax1.legend()
    ax1.set_xscale("symlog", linthresh=5)
    ax1.set_xticks([0, 2, 5, 10, 25, 50, 100, 200, 400, 800])
    ax1.get_xaxis().set_major_formatter(mticker.ScalarFormatter())

    # Right: catch rate per duration bin
    bin_edges = [(0, 5, "<5"), (5, 25, "5–25"), (25, 100, "25–100"),
                 (100, 300, "100–300"), (300, 99999, "300+")]
    labels_b = []
    rates = []
    counts_all = []
    counts_caught = []
    for lo, hi, label in bin_edges:
        mask = (df["duration_frames"] >= lo) & (df["duration_frames"] < hi)
        n = mask.sum()
        c = (mask & df["caught"]).sum()
        labels_b.append(label)
        rates.append(100 * c / n if n > 0 else 0)
        counts_all.append(n)
        counts_caught.append(c)

    x = np.arange(len(labels_b))
    bars = ax2.bar(x, rates, 0.6, color=COLOR_CAUGHT, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_b)
    ax2.set_xlabel("Switch duration (frames)")
    ax2.set_ylabel("Catch rate (%)")
    ax2.set_title("(b) Catch rate by duration")
    ax2.set_ylim(0, max(rates) * 1.4)

    for i, (rate, c, n) in enumerate(zip(rates, counts_caught, counts_all)):
        ax2.text(i, rate + 0.8, f"{rate:.1f}%\n({c}/{n})", ha="center", fontsize=9)

    fig.suptitle("Switch duration analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "duration_distribution.pdf")
    fig.savefig(FIG_DIR / "duration_distribution.png")
    plt.close(fig)
    print(f"  Saved: duration_distribution.pdf/png")


# =========================================================================
# Figure 3: Temporal gap and spatial distance
# =========================================================================
def fig_gap_and_distance(df: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: temporal gap catch rate
    gap = df[df["has_temporal_gap"]]
    nogap = df[~df["has_temporal_gap"]]
    categories = ["With temporal gap", "Without temporal gap"]
    totals = [len(gap), len(nogap)]
    caught_c = [gap["caught"].sum(), nogap["caught"].sum()]
    rates = [100 * c / t if t > 0 else 0 for c, t in zip(caught_c, totals)]

    x = np.arange(2)
    bars = ax1.bar(x, rates, 0.5, color=COLOR_CAUGHT, alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=10)
    ax1.set_ylabel("Catch rate (%)")
    ax1.set_title("(a) Temporal gap at switch point")
    ax1.set_ylim(0, max(rates) * 1.5)
    for i, (r, c, t) in enumerate(zip(rates, caught_c, totals)):
        ax1.text(i, r + 0.8, f"{r:.1f}%\n({c}/{t})", ha="center", fontsize=10)

    # Right: spatial distance distribution (only switches where both players visible)
    has_dist = df[df["bbox_distance"] > 0].copy()
    caught_d = has_dist[has_dist["caught"]]
    uncaught_d = has_dist[~has_dist["caught"]]

    dist_bins = np.arange(0, 550, 25)
    ax2.hist(uncaught_d["bbox_distance"].clip(upper=524), bins=dist_bins,
             alpha=0.55, color=COLOR_UNCAUGHT, label=f"Uncaught (n={len(uncaught_d)})", edgecolor="white")
    ax2.hist(caught_d["bbox_distance"].clip(upper=524), bins=dist_bins,
             alpha=0.85, color=COLOR_CAUGHT, label=f"Caught (n={len(caught_d)})", edgecolor="white")
    ax2.set_xlabel("Bbox center distance (pixels)")
    ax2.set_ylabel("Count")
    ax2.set_title("(b) Player distance at switch frame")
    ax2.legend()

    # Add mean lines
    if len(caught_d) > 0:
        ax2.axvline(caught_d["bbox_distance"].mean(), color=COLOR_CAUGHT,
                     linestyle="--", linewidth=1.5, label=f"Mean caught: {caught_d['bbox_distance'].mean():.0f}px")
    if len(uncaught_d) > 0:
        ax2.axvline(uncaught_d["bbox_distance"].mean(), color=COLOR_UNCAUGHT,
                     linestyle="--", linewidth=1.5, label=f"Mean uncaught: {uncaught_d['bbox_distance'].mean():.0f}px")
    ax2.legend(fontsize=9)

    fig.suptitle("Temporal and spatial characteristics of identity switches", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "gap_and_distance.pdf")
    fig.savefig(FIG_DIR / "gap_and_distance.png")
    plt.close(fig)
    print(f"  Saved: gap_and_distance.pdf/png")


# =========================================================================
# Figure 4: Per-sequence difficulty (top 15 hardest)
# =========================================================================
def fig_per_sequence(seq_df: pd.DataFrame):
    seq_df = seq_df.sort_values("uncaught", ascending=False)
    top = seq_df.head(15).copy()

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(top))
    w = 0.35

    ax.bar(x - w/2, top["caught"], w, label="Caught", color=COLOR_CAUGHT, alpha=0.85)
    ax.bar(x + w/2, top["uncaught"], w, label="Uncaught", color=COLOR_UNCAUGHT, alpha=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(top["sequence"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of identity switches")
    ax.set_title("Top 15 hardest sequences (most uncaught identity switches)")
    ax.legend()

    # Add total count labels
    for i, (_, row) in enumerate(top.iterrows()):
        total = row["total_switches"]
        recall = row["recall"] * 100
        ax.text(i, row["uncaught"] + 1, f"{recall:.0f}%", ha="center", fontsize=8, color=COLOR_NEUTRAL)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "per_sequence_difficulty.pdf")
    fig.savefig(FIG_DIR / "per_sequence_difficulty.png")
    plt.close(fig)
    print(f"  Saved: per_sequence_difficulty.pdf/png")


# =========================================================================
# Figure 5: Combined summary — caught vs uncaught profile table as figure
# =========================================================================
def fig_summary_table(df: pd.DataFrame):
    """Create a visual summary comparing caught vs uncaught switch profiles."""
    caught = df[df["caught"]]
    uncaught = df[~df["caught"]]

    has_team_c = caught.dropna(subset=["same_team"])
    has_team_u = uncaught.dropna(subset=["same_team"])
    has_dist_c = caught[caught["bbox_distance"] > 0]
    has_dist_u = uncaught[uncaught["bbox_distance"] > 0]

    rows_data = [
        ("Count", f"{len(caught)}", f"{len(uncaught)}"),
        ("Mean duration (frames)", f"{caught['duration_frames'].mean():.1f}", f"{uncaught['duration_frames'].mean():.1f}"),
        ("Duration < 5 frames", f"{(caught['duration_frames'] < 5).sum()} ({100*(caught['duration_frames'] < 5).mean():.1f}%)",
         f"{(uncaught['duration_frames'] < 5).sum()} ({100*(uncaught['duration_frames'] < 5).mean():.1f}%)"),
        ("Duration > 100 frames", f"{(caught['duration_frames'] >= 100).sum()} ({100*(caught['duration_frames'] >= 100).mean():.1f}%)",
         f"{(uncaught['duration_frames'] >= 100).sum()} ({100*(uncaught['duration_frames'] >= 100).mean():.1f}%)"),
        ("With temporal gap", f"{caught['has_temporal_gap'].sum()} ({100*caught['has_temporal_gap'].mean():.1f}%)",
         f"{uncaught['has_temporal_gap'].sum()} ({100*uncaught['has_temporal_gap'].mean():.1f}%)"),
        ("Same team", f"{(has_team_c['same_team']==True).sum()} ({100*(has_team_c['same_team']==True).mean():.1f}%)" if len(has_team_c)>0 else "N/A",
         f"{(has_team_u['same_team']==True).sum()} ({100*(has_team_u['same_team']==True).mean():.1f}%)" if len(has_team_u)>0 else "N/A"),
        ("Mean bbox distance (px)", f"{has_dist_c['bbox_distance'].mean():.1f}" if len(has_dist_c)>0 else "N/A",
         f"{has_dist_u['bbox_distance'].mean():.1f}" if len(has_dist_u)>0 else "N/A"),
    ]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis("off")
    col_labels = ["Property", "Caught", "Uncaught"]
    table = ax.table(
        cellText=rows_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(3):
        table[0, j].set_facecolor("#e5e7eb")
        table[0, j].set_text_props(fontweight="bold")

    # Color columns
    for i in range(1, len(rows_data) + 1):
        table[i, 1].set_facecolor("#dbeafe")  # light blue
        table[i, 2].set_facecolor("#fee2e2")  # light red

    ax.set_title("Caught vs uncaught identity switch profiles", fontsize=13, pad=20)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "summary_table.pdf")
    fig.savefig(FIG_DIR / "summary_table.png")
    plt.close(fig)
    print(f"  Saved: summary_table.pdf/png")


# =========================================================================
# Figure 6: Duration vs team interaction
# =========================================================================
def fig_duration_team_interaction(df: pd.DataFrame):
    """Show that short switches are common in both same-team and cross-team."""
    has_team = df.dropna(subset=["same_team"]).copy()
    same = has_team[has_team["same_team"] == True]
    cross = has_team[has_team["same_team"] == False]

    bin_edges = [(0, 5, "<5"), (5, 25, "5–25"), (25, 100, "25–100"),
                 (100, 300, "100–300"), (300, 99999, "300+")]

    fig, ax = plt.subplots(figsize=(9, 5))

    labels_b = [b[2] for b in bin_edges]
    x = np.arange(len(labels_b))
    w = 0.35

    same_counts = []
    cross_counts = []
    for lo, hi, _ in bin_edges:
        same_counts.append(((same["duration_frames"] >= lo) & (same["duration_frames"] < hi)).sum())
        cross_counts.append(((cross["duration_frames"] >= lo) & (cross["duration_frames"] < hi)).sum())

    ax.bar(x - w/2, same_counts, w, label=f"Same team (n={len(same)})", color="#f59e0b", alpha=0.8)
    ax.bar(x + w/2, cross_counts, w, label=f"Cross team (n={len(cross)})", color="#8b5cf6", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_b)
    ax.set_xlabel("Switch duration (frames)")
    ax.set_ylabel("Count")
    ax.set_title("Switch duration by team relationship")
    ax.legend()

    # Add count labels
    for i, (s, c) in enumerate(zip(same_counts, cross_counts)):
        ax.text(i - w/2, s + 2, str(s), ha="center", fontsize=9)
        ax.text(i + w/2, c + 2, str(c), ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "duration_team_interaction.pdf")
    fig.savefig(FIG_DIR / "duration_team_interaction.png")
    plt.close(fig)
    print(f"  Saved: duration_team_interaction.pdf/png")


# =========================================================================
# Print console summary
# =========================================================================
def print_summary(df: pd.DataFrame, seq_df: pd.DataFrame):
    total = len(df)
    caught = df["caught"].sum()
    uncaught = total - caught

    print(f"\n{'='*60}")
    print(f"SUMMARY STATISTICS FOR THESIS")
    print(f"{'='*60}")
    print(f"Total identity switches: {total}")
    print(f"Caught: {caught} ({100*caught/total:.1f}%)")
    print(f"Uncaught: {uncaught} ({100*uncaught/total:.1f}%)")

    # Key findings for the thesis text
    short = (df["duration_frames"] < 5).sum()
    short_pct = 100 * short / total
    print(f"\nShort switches (<5 frames): {short} ({short_pct:.1f}% of all)")
    short_caught = ((df["duration_frames"] < 5) & df["caught"]).sum()
    print(f"  Of which caught: {short_caught} ({100*short_caught/short:.1f}%)")

    long = (df["duration_frames"] >= 100).sum()
    long_caught = ((df["duration_frames"] >= 100) & df["caught"]).sum()
    print(f"\nLong switches (>=100 frames): {long} ({100*long/total:.1f}% of all)")
    print(f"  Of which caught: {long_caught} ({100*long_caught/long:.1f}%)")

    gap_switches = df["has_temporal_gap"].sum()
    gap_caught = (df["has_temporal_gap"] & df["caught"]).sum()
    print(f"\nSwitches with temporal gap: {gap_switches} ({100*gap_switches/total:.1f}%)")
    print(f"  Catch rate: {100*gap_caught/gap_switches:.1f}%")
    nogap_caught = (~df["has_temporal_gap"] & df["caught"]).sum()
    nogap_total = (~df["has_temporal_gap"]).sum()
    print(f"Switches without gap: {nogap_total}")
    print(f"  Catch rate: {100*nogap_caught/nogap_total:.1f}%")

    has_team = df.dropna(subset=["same_team"])
    same = has_team[has_team["same_team"] == True]
    cross = has_team[has_team["same_team"] == False]
    print(f"\nSame-team switches: {len(same)} ({100*len(same)/total:.1f}%)")
    print(f"  Catch rate: {100*same['caught'].mean():.1f}%")
    print(f"Cross-team switches: {len(cross)} ({100*len(cross)/total:.1f}%)")
    print(f"  Catch rate: {100*cross['caught'].mean():.1f}%")

    # Excluding short switches
    meaningful = df[df["duration_frames"] >= 5]
    meaningful_caught = meaningful["caught"].sum()
    print(f"\n--- EXCLUDING <5 frame switches ---")
    print(f"Meaningful switches (>=5 frames): {len(meaningful)}")
    print(f"Caught: {meaningful_caught} ({100*meaningful_caught/len(meaningful):.1f}%)")

    # Top 5 hardest sequences
    print(f"\nTop 5 hardest sequences:")
    top5 = seq_df.sort_values("uncaught", ascending=False).head(5)
    for _, row in top5.iterrows():
        print(f"  {row['sequence']}: {row['uncaught']} uncaught / {row['total_switches']} total "
              f"(recall {row['recall']*100:.1f}%), "
              f"same_team={row['same_team']}, cross_team={row['cross_team']}")


# =========================================================================
# Main
# =========================================================================
def main():
    print("Loading data...")
    df, seq_df = load_data()
    print(f"  {len(df)} switches, {len(seq_df)} sequences\n")

    print("Generating figures...")
    fig_team_breakdown(df)
    fig_duration_distribution(df)
    fig_gap_and_distance(df)
    fig_per_sequence(seq_df)
    fig_summary_table(df)
    fig_duration_team_interaction(df)

    print_summary(df, seq_df)

    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
