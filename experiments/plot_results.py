#!/usr/bin/env python3
"""
Generate summary plots for Paper2Codabench experiments.

Produces individual PNGs and optionally a combined figure.

Usage:
    python experiments/plot_results.py              # individual PNGs only
    python experiments/plot_results.py --combined   # also generate combined
"""
import argparse
import json
import math
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

# ── Load data ────────────────────────────────────────────────────────────────
hall = json.load(open(RESULTS_DIR / "hallucination_results.json"))
fitb = json.load(open(RESULTS_DIR / "fitb_results.json"))
run_all = json.load(open(RESULTS_DIR / "run_all_results.json"))

# ── Colour palette ───────────────────────────────────────────────────────────
PASS_COLOR = "#4CAF50"
FAIL_COLOR = "#F44336"
SKIP_COLOR = "#9E9E9E"
A_COLOR = "#2196F3"
B_COLOR = "#FF9800"
PROP_COLOR = "#E53935"
NOPROP_COLOR = "#43A047"

plant_types = hall["plant_types"]
completed = [r for r in hall["results"] if r["status"] == "completed"]
papers = [r["paper_id"] for r in completed]
vr = run_all["verification_results"]


def build_matrix():
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


def plot_hallucination_heatmap(ax=None):
    """Hallucination propagation heatmap."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, max(4, len(papers) * 0.6)))

    mat = build_matrix()
    cmap = matplotlib.colors.ListedColormap([SKIP_COLOR, NOPROP_COLOR, PROP_COLOR])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(plant_types)))
    ax.set_xticklabels([pt.replace("_", "\n") for pt in plant_types], fontsize=8)
    ax.set_yticks(range(len(papers)))
    ax.set_yticklabels(papers, fontsize=9)

    for i in range(len(papers)):
        for j in range(len(plant_types)):
            v = mat[i, j]
            label = "YES" if v == 1 else ("no" if v == 0 else "N/A")
            color = "white" if v != 0 else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=9, color=color, fontweight="bold")

    ax.set_title("Hallucination Propagation\n(planted error \u2192 LLM output)", fontsize=10, fontweight="bold")
    patches = [
        mpatches.Patch(color=PROP_COLOR, label="Propagated"),
        mpatches.Patch(color=NOPROP_COLOR, label="Not propagated"),
        mpatches.Patch(color=SKIP_COLOR, label="N/A"),
    ]
    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7, framealpha=0.8)

    if standalone:
        out = PLOTS_DIR / "hallucination_heatmap.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {out}")


def plot_hallucination_rates(ax=None):
    """Hallucination propagation rate by error type."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 5))

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

    x = np.arange(len(plant_types))
    bars = ax.bar(x, rates, color=[PROP_COLOR if r > 0 else NOPROP_COLOR for r in rates],
                  edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([pt.replace("_", "\n") for pt in plant_types], fontsize=8)
    ax.set_ylabel("Propagation rate (%)", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title("Hallucination Rate\nby Error Type", fontsize=10, fontweight="bold")
    ax.axhline(50, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{rate:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    if standalone:
        out = PLOTS_DIR / "hallucination_rates.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {out}")


def plot_pipeline_pass_fail(ax=None):
    """Pipeline pass/fail chart."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))

    pipelines = {}
    for pipe_key, label in [("pipeline_a", "Pipeline A"), ("pipeline_b", "Pipeline B")]:
        if pipe_key not in vr:
            continue
        results = vr[pipe_key]["results"]
        passed = sum(1 for r in results if r["status"] == "PASS")
        failed = sum(1 for r in results if r["status"] == "FAIL")
        skipped = sum(1 for r in results if r["status"] == "SKIP")
        pipelines[label] = {"PASS": passed, "FAIL": failed, "SKIP": skipped}

    pipe_labels = list(pipelines.keys())
    n = len(pipe_labels)
    bar_w = 0.25
    x = np.arange(n)

    for i, (status, color) in enumerate([("PASS", PASS_COLOR), ("FAIL", FAIL_COLOR), ("SKIP", SKIP_COLOR)]):
        vals = [pipelines[l][status] for l in pipe_labels]
        offset = (i - 1) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, label=status, color=color, edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(pipe_labels, fontsize=10)
    ax.set_ylabel("Number of papers", fontsize=9)
    ax.set_title("Bundle Verification\nPass / Fail / Skip", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(p["PASS"] + p["FAIL"] + p["SKIP"] for p in pipelines.values()) + 1.5)

    for i, label in enumerate(pipe_labels):
        total = sum(pipelines[label].values())
        passed = pipelines[label]["PASS"]
        rate = passed / total * 100 if total else 0
        ax.text(i, -0.9, f"{rate:.0f}% pass rate", ha="center", fontsize=9, color="dimgrey", style="italic")

    if standalone:
        out = PLOTS_DIR / "pipeline_pass_fail.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {out}")


def plot_fitb_comparison(ax=None):
    """FITB comparison between pipelines."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 5))

    def avg_fitb(pipe_data):
        results = pipe_data.get("results", [])
        if not results:
            return 0, 0
        arr = sum(r["fitb_array_count"] for r in results) / len(results)
        deep = sum(r["deep_search_count"] for r in results) / len(results)
        return arr, deep

    a_arr, a_deep = avg_fitb(fitb["pipeline_a"])
    b_arr, b_deep = avg_fitb(fitb["pipeline_b"])

    categories = ["Declared\n[FILL IN BLANK]", "Undeclared\n(deep search)"]
    a_vals = [a_arr, a_deep]
    b_vals = [b_arr, b_deep]
    x = np.arange(len(categories))

    ax.bar(x - 0.18, a_vals, 0.35, label="Pipeline A", color=A_COLOR, alpha=0.85, edgecolor="white")
    ax.bar(x + 0.18, b_vals, 0.35, label="Pipeline B", color=B_COLOR, alpha=0.85, edgecolor="white")
    for xi, (av, bv) in enumerate(zip(a_vals, b_vals)):
        ax.text(xi - 0.18, av + 0.05, f"{av:.1f}", ha="center", va="bottom", fontsize=9)
        ax.text(xi + 0.18, bv + 0.05, f"{bv:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("Avg missing fields per paper", fontsize=9)
    ax.set_title("Fill-in-the-Blank Missing Fields\n(avg per paper, lower = better)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(max(a_vals), max(b_vals)) + 1.5)

    if standalone:
        out = PLOTS_DIR / "fitb_comparison.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {out}")


def plot_pipeline_scores(ax=None):
    """Pipeline B primary scores for passing papers."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))

    pb_results = [r for r in vr.get("pipeline_b", {}).get("results", []) if r["status"] == "PASS"]

    paper_names = []
    primary_scores = []
    primary_metric_names = []

    for r in pb_results:
        scores = r.get("scores", {})
        if not scores:
            continue
        valid = {k: v for k, v in scores.items() if isinstance(v, (int, float)) and not math.isnan(v)}
        if not valid:
            continue
        first_key = next(iter(valid))
        paper_names.append(r["paper_id"])
        primary_scores.append(valid[first_key])
        primary_metric_names.append(first_key)

    if not paper_names:
        ax.text(0.5, 0.5, "No passing papers", ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.set_title("Pipeline B — Primary Score\nper Passing Paper", fontsize=10, fontweight="bold")
        if standalone:
            out = PLOTS_DIR / "pipeline_scores.png"
            fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"Saved: {out}")
        return

    x = np.arange(len(paper_names))
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(paper_names)))
    bars = ax.bar(x, primary_scores, color=colors, edgecolor="white", linewidth=0.8)

    for bar, score, metric in zip(bars, primary_scores, primary_metric_names):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(abs(s) for s in primary_scores) * 0.02,
                f"{score:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(paper_names, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Primary metric score", fontsize=9)
    ax.set_title("Pipeline B — Primary Score\nper Passing Paper", fontsize=10, fontweight="bold")

    for xi, metric in enumerate(primary_metric_names):
        ax.text(xi, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.08,
                metric, ha="center", fontsize=6, color="dimgrey", style="italic", rotation=15)

    if standalone:
        out = PLOTS_DIR / "pipeline_scores.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {out}")


def generate_combined():
    """Generate the combined 5-panel figure."""
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("Paper2Codabench — Experiment Summary", fontsize=16, fontweight="bold", y=0.98)

    axes = fig.subplot_mosaic(
        [["hall_heat", "hall_bar", "pipeline_pass"],
         ["fitb_bar", "scores", "pipeline_pass"]],
        gridspec_kw={"hspace": 0.45, "wspace": 0.35},
    )

    plot_hallucination_heatmap(axes["hall_heat"])
    plot_hallucination_rates(axes["hall_bar"])
    plot_pipeline_pass_fail(axes["pipeline_pass"])
    plot_fitb_comparison(axes["fitb_bar"])
    plot_pipeline_scores(axes["scores"])

    out = RESULTS_DIR / "summary_plots.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved combined: {out}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate experiment summary plots")
    parser.add_argument("--combined", action="store_true", help="Also generate combined figure")
    args = parser.parse_args()

    # Generate individual PNGs
    plot_hallucination_heatmap()
    plot_hallucination_rates()
    plot_pipeline_pass_fail()
    plot_fitb_comparison()
    plot_pipeline_scores()

    if args.combined:
        generate_combined()
