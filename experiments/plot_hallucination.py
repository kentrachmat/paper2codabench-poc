#!/usr/bin/env python3
"""
Generate individual hallucination experiment plots.

Produces:
  experiments/results/plots/hallucination_heatmap.png
  experiments/results/plots/hallucination_rates.png
  experiments/results/plots/hallucination_per_paper.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Colors
PROP_COLOR = "#E53935"
NOPROP_COLOR = "#43A047"
SKIP_COLOR = "#9E9E9E"

# Load data
hall = json.load(open(RESULTS_DIR / "hallucination_results.json"))
plant_types = hall["plant_types"]
completed = [r for r in hall["results"] if r["status"] == "completed"]
papers = [r["paper_id"] for r in completed]


def build_matrix():
    """Build propagation matrix: 1=propagated, 0=not, -1=N/A"""
    matrix = []
    for r in completed:
        row = []
        for pt in plant_types:
            chk = next((c for c in r["checks"] if c["plant_type"] == pt), None)
            if chk is None or not chk.get("planted"):
                row.append(-1)
            elif chk.get("propagated"):
                row.append(1)
            else:
                row.append(0)
        matrix.append(row)
    return np.array(matrix, dtype=float)


def plot_heatmap():
    """Paper x plant_type propagation matrix."""
    mat = build_matrix()
    cmap = matplotlib.colors.ListedColormap([SKIP_COLOR, NOPROP_COLOR, PROP_COLOR])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(8, max(4, len(papers) * 0.6)))
    ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(plant_types)))
    ax.set_xticklabels([pt.replace("_", "\n") for pt in plant_types], fontsize=9)
    ax.set_yticks(range(len(papers)))
    ax.set_yticklabels(papers, fontsize=9)

    for i in range(len(papers)):
        for j in range(len(plant_types)):
            v = mat[i, j]
            label = "YES" if v == 1 else ("no" if v == 0 else "N/A")
            color = "white" if v != 0 else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=9, color=color, fontweight="bold")

    ax.set_title("Hallucination Propagation\n(planted error -> LLM output)", fontweight="bold")
    patches = [
        mpatches.Patch(color=PROP_COLOR, label="Propagated"),
        mpatches.Patch(color=NOPROP_COLOR, label="Not propagated"),
        mpatches.Patch(color=SKIP_COLOR, label="N/A"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8)

    out = PLOTS_DIR / "hallucination_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_rates():
    """Bar chart of propagation rates by error type."""
    rates = []
    for pt in plant_types:
        planted = propagated = 0
        for r in hall["results"]:
            for c in r.get("checks", []):
                if c["plant_type"] == pt and c.get("planted"):
                    planted += 1
                    if c.get("propagated"):
                        propagated += 1
        rates.append(propagated / planted * 100 if planted else 0)

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(plant_types))
    bars = ax.bar(x, rates, color=[PROP_COLOR if r > 0 else NOPROP_COLOR for r in rates],
                  edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([pt.replace("_", "\n") for pt in plant_types], fontsize=9)
    ax.set_ylabel("Propagation rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Hallucination Rate by Error Type", fontweight="bold")
    ax.axhline(50, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{rate:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    out = PLOTS_DIR / "hallucination_rates.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_per_paper():
    """Per-paper propagation count."""
    paper_names = []
    propagation_counts = []

    for r in completed:
        paper_names.append(r["paper_id"])
        count = sum(1 for c in r["checks"] if c.get("propagated"))
        propagation_counts.append(count)

    fig, ax = plt.subplots(figsize=(max(6, len(paper_names) * 0.8), 5))
    x = np.arange(len(paper_names))
    colors = [PROP_COLOR if c > 0 else NOPROP_COLOR for c in propagation_counts]
    bars = ax.bar(x, propagation_counts, color=colors, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(paper_names, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Propagated errors")
    ax.set_title("Hallucination Propagation per Paper", fontweight="bold")
    ax.set_ylim(0, max(propagation_counts + [1]) + 1)

    for bar, cnt in zip(bars, propagation_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(cnt), ha="center", va="bottom", fontsize=10, fontweight="bold")

    out = PLOTS_DIR / "hallucination_per_paper.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    plot_heatmap()
    plot_rates()
    plot_per_paper()
