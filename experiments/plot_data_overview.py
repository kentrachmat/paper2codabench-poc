#!/usr/bin/env python3
"""
NeurIPS 2024-2025: Comprehensive Data Overview for Paper2Codabench

Generates individual plot PNGs and optionally a combined 6-panel figure.

Usage:
    python experiments/plot_data_overview.py              # individual PNGs only
    python experiments/plot_data_overview.py --combined    # also generate combined
"""
import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
PAPERS_META = BASE / "papers" / "metadata.json"
CROISSANT_A = BASE / "croissant_tasks"
CROISSANT_B = BASE / "croissant_tasks_pipelineB"
BUNDLES_A = BASE / "bundles"
BUNDLES_B = BASE / "bundles_pipelineB"
PLOTS_DIR = BASE / "experiments" / "results" / "plots"
COMBINED_PATH = BASE / "experiments" / "results" / "data_overview.png"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load helpers ───────────────────────────────────────────────────────
def load_csv(year):
    path = DATA_DIR / f"neurips_{year}_db_papers.csv"
    with open(path, "r") as f:
        return list(csv.DictReader(f))

def stats(rows):
    total = len(rows)
    croissant = sum(1 for r in rows if r.get("has_croissant", "").lower() == "true")
    dataset = sum(1 for r in rows if r.get("dataset_url", "").strip())
    return {"total": total, "has_croissant": croissant, "has_dataset_url": dataset}

def count_croissant_tasks(directory):
    if not directory.exists():
        return 0
    return len(list(directory.glob("*.croissant_task.json")))

def count_bundles(directory):
    if not directory.exists():
        return 0
    return len([d for d in directory.iterdir() if d.is_dir() and (d / "competition.yaml").exists()])

# ── Load data ──────────────────────────────────────────────────────────
data_2024 = load_csv(2024)
data_2025 = load_csv(2025)
s24 = stats(data_2024)
s25 = stats(data_2025)

with open(PAPERS_META) as f:
    codabench_papers = json.load(f)["competitions"]

codabench_2024 = [p for p in codabench_papers if "2024" in p.get("track", "")]
codabench_2025 = [p for p in codabench_papers if "2025" in p.get("track", "")]

pipeline_croissant = count_croissant_tasks(CROISSANT_B) or count_croissant_tasks(CROISSANT_A)
pipeline_bundles = count_bundles(BUNDLES_B) or count_bundles(BUNDLES_A)

# ── Constants ──────────────────────────────────────────────────────────
COMP_TRACK_TOTAL_2024 = 16
COMP_TRACK_TOTAL_2025 = 18

INTERSECTION_MAP = {
    "paper1": "TmLYHibOFT",
    "paper3": "F48NfH4NFE",
    "paper4": "yG4Fj0voJZ",
    "paper5": "BeFjjyzWOJ",
}
db_2025_ids = {r["paper_id"] for r in data_2025}
db_2025_lookup = {r["paper_id"]: r for r in data_2025}

# ── Dataset hosting platform categorisation (2025) ────────────────────
def categorise_url(url):
    d = urlparse(url).netloc.lower()
    if "huggingface" in d: return "HuggingFace"
    if "kaggle" in d: return "Kaggle"
    if "github" in d: return "GitHub"
    if "dataverse" in d: return "Dataverse"
    if "doi.org" in d: return "DOI"
    if "zenodo" in d: return "Zenodo"
    return "Other"

dataset_urls_2025 = [r["dataset_url"].strip() for r in data_2025 if r.get("dataset_url", "").strip()]
platform_counts = Counter(categorise_url(u) for u in dataset_urls_2025)

# ── UpSet combination counts ──────────────────────────────────────────
codabench_or_ids = set(INTERSECTION_MAP.values())
combo_counts = Counter()
for r in data_2025:
    pid = r["paper_id"]
    has_cr = r.get("has_croissant", "").lower() == "true"
    has_du = bool(r.get("dataset_url", "").strip())
    has_cb = pid in codabench_or_ids
    combo_counts[(has_cr, has_du, has_cb)] += 1

# ── Colors ─────────────────────────────────────────────────────────────
C_BLUE = "#3B82F6"; C_BLUE_LIGHT = "#93C5FD"
C_GREEN = "#10B981"; C_GREEN_LIGHT = "#6EE7B7"
C_ORANGE = "#F59E0B"; C_RED = "#EF4444"
C_PURPLE = "#8B5CF6"; C_GRAY = "#9CA3AF"
C_TEAL = "#14B8A6"; C_DARK = "#1F2937"


# ═══════════════════════════════════════════════════════════════════════
# Individual plot functions
# ═══════════════════════════════════════════════════════════════════════

def plot_papers_by_year(ax=None):
    """Panel 1: NeurIPS Papers by Year & Track."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))

    years = ["2024", "2025"]
    db_vals = [s24["total"], s25["total"]]
    comp_total = [COMP_TRACK_TOTAL_2024, COMP_TRACK_TOTAL_2025]
    codabench_vals = [len(codabench_2024), len(codabench_2025)]
    comp_other = [ct - cb for ct, cb in zip(comp_total, codabench_vals)]

    x = np.arange(len(years))
    w = 0.35
    bars_db = ax.bar(x - w / 2, db_vals, w, color=C_BLUE, label="D&B Track")
    bars_cb = ax.bar(x + w / 2, codabench_vals, w, color=C_ORANGE, label="Comp. Track — Codabench (ours)")
    bars_co = ax.bar(x + w / 2, comp_other, w, bottom=codabench_vals, color="#FBBF24", label="Comp. Track — other platforms")

    for bar, v in zip(bars_db, db_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 5, str(v), ha="center", va="bottom", fontweight="bold", fontsize=12)
    for bar, cb, ct in zip(bars_co, codabench_vals, comp_total):
        ax.text(bar.get_x() + bar.get_width() / 2, ct + 1, str(ct), ha="center", va="bottom", fontweight="bold", fontsize=12)
        if cb > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, cb / 2, str(cb), ha="center", va="center", fontweight="bold", fontsize=10, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("Number of Papers")
    ax.set_title("NeurIPS Papers by Year & Track", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, max(db_vals) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if standalone:
        out = PLOTS_DIR / "papers_by_year.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {out}")


def plot_croissant_adoption(ax=None):
    """Panel 2: Croissant Adoption in D&B Track."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))

    years = ["2024", "2025"]
    croissant_yes = [s24["has_croissant"], s25["has_croissant"]]
    croissant_no = [s24["total"] - s24["has_croissant"], s25["total"] - s25["has_croissant"]]

    ax.bar(years, croissant_yes, color=C_GREEN, label="Has Croissant")
    ax.bar(years, croissant_no, bottom=croissant_yes, color=C_GRAY, label="No Croissant")

    for i, (cy, cn) in enumerate(zip(croissant_yes, croissant_no)):
        total = cy + cn
        pct = cy / total * 100 if total > 0 else 0
        if cy > 15:
            ax.text(i, cy / 2, f"{cy}\n({pct:.1f}%)", ha="center", va="center", fontweight="bold", color="white", fontsize=11)
        else:
            ax.text(i, cy + 5, f"{cy} ({pct:.1f}%)", ha="center", va="bottom", fontweight="bold", color=C_GREEN, fontsize=10)
        if cn > 15:
            ax.text(i, cy + cn / 2, str(cn), ha="center", va="center", color="white", fontsize=11)
        ax.text(i, total + 5, str(total), ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.annotate("0.2% \u2192 85.7%", xy=(1, s25["has_croissant"]), xytext=(1.35, s25["total"] * 0.55),
                fontsize=11, fontweight="bold", color=C_GREEN,
                arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.5), ha="center")
    ax.set_ylabel("Number of Papers")
    ax.set_title("Croissant Adoption (D&B Track)", fontweight="bold")
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(s24["total"], s25["total"]) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if standalone:
        out = PLOTS_DIR / "croissant_adoption.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {out}")


def plot_data_availability_upset():
    """Panel 3: UpSet-style intersection (standalone only)."""
    set_names = ["Croissant", "Dataset URL", "Codabench"]
    combos = [(key, cnt) for key, cnt in combo_counts.items() if cnt > 0]
    combos.sort(key=lambda x: -x[1])
    n_combos = len(combos)
    n_sets = len(set_names)

    fig, (ax_bars, ax_dots) = plt.subplots(2, 1, figsize=(10, 7),
                                            gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05})

    x_pos = np.arange(n_combos)
    counts = [c[1] for c in combos]
    bar_colors = []
    for combo, _ in combos:
        cr, du, cb = combo
        if cb: bar_colors.append(C_ORANGE)
        elif cr and du: bar_colors.append(C_GREEN)
        elif cr: bar_colors.append(C_BLUE)
        elif du: bar_colors.append(C_TEAL)
        else: bar_colors.append(C_GRAY)

    bars = ax_bars.bar(x_pos, counts, color=bar_colors, edgecolor="white", linewidth=0.8, width=0.6)
    for bar, cnt in zip(bars, counts):
        ax_bars.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3, str(cnt),
                     ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax_bars.set_ylabel("Papers")
    ax_bars.set_title("NeurIPS 2025 D&B: Data Availability (UpSet-style)", fontweight="bold")
    ax_bars.set_xlim(-0.5, n_combos - 0.5)
    ax_bars.set_xticks([])
    ax_bars.set_ylim(0, max(counts) * 1.15)
    for sp in ["top", "right", "bottom"]:
        ax_bars.spines[sp].set_visible(False)

    for i, (combo, _) in enumerate(combos):
        active = list(combo)
        active_indices = [j for j, a in enumerate(active) if a]
        if len(active_indices) >= 2:
            ax_dots.plot([i, i], [min(active_indices), max(active_indices)], color=C_DARK, linewidth=2, zorder=1)
        for j in range(n_sets):
            if active[j]:
                ax_dots.scatter(i, j, s=180, color=C_DARK, zorder=2, edgecolors="none")
            else:
                ax_dots.scatter(i, j, s=180, facecolors="white", edgecolors=C_GRAY, linewidths=1.5, zorder=2)

    ax_dots.set_yticks(range(n_sets))
    ax_dots.set_yticklabels(set_names, fontsize=10)
    ax_dots.set_xlim(-0.5, n_combos - 0.5)
    ax_dots.set_ylim(-0.5, n_sets - 0.5)
    ax_dots.set_xticks([])
    ax_dots.invert_yaxis()
    for sp in ["top", "right", "bottom"]:
        ax_dots.spines[sp].set_visible(False)

    set_totals = [
        sum(c[1] for c in combos if c[0][0]),
        sum(c[1] for c in combos if c[0][1]),
        sum(c[1] for c in combos if c[0][2]),
    ]
    for j, tot in enumerate(set_totals):
        ax_dots.text(n_combos - 0.3, j, str(tot), ha="left", va="center", fontsize=9, color="dimgrey", style="italic")

    out = PLOTS_DIR / "data_availability_upset.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_hosting_platforms(ax=None):
    """Panel 4: Dataset Hosting Platforms."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))

    platform_order = ["HuggingFace", "Kaggle", "GitHub", "Dataverse", "DOI", "Zenodo", "Other"]
    platform_vals = [platform_counts.get(p, 0) for p in platform_order]
    platform_colors = [C_ORANGE, C_BLUE, C_DARK, C_PURPLE, C_TEAL, C_GREEN, C_GRAY]

    y_pos = np.arange(len(platform_order))
    bars = ax.barh(y_pos, platform_vals, color=platform_colors, height=0.6, edgecolor="white")
    for bar, v in zip(bars, platform_vals):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                f"{v} ({v / len(dataset_urls_2025) * 100:.1f}%)", va="center", fontweight="bold", fontsize=10)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(platform_order, fontsize=10)
    ax.set_xlabel("Number of Datasets")
    ax.set_title(f"Dataset Hosting Platforms (2025, n={len(dataset_urls_2025)})", fontweight="bold")
    ax.set_xlim(0, max(platform_vals) * 1.3)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if standalone:
        out = PLOTS_DIR / "hosting_platforms.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {out}")


def plot_competition_papers_table():
    """Panel 5: Competition Papers table."""
    fig, ax = plt.subplots(figsize=(12, max(3, len(codabench_papers) * 0.4 + 1)))
    ax.axis("off")

    header = [["#", "Competition (Competition Track)", "Year", "Task", "Also in\nD&B?"]]
    rows = []
    for i, p in enumerate(codabench_papers, 1):
        year_label = "2024" if "2024" in p["track"] else "2025"
        short_title = p["title"][:48] + ("\u2026" if len(p["title"]) > 48 else "")
        task = p.get("task_type", "?")
        or_id = INTERSECTION_MAP.get(p["paper_id"])
        in_db = "\u2713" if (or_id and or_id in db_2025_ids) else "\u2014"
        rows.append([str(i), short_title, year_label, task, in_db])

    all_rows = header + rows
    table = ax.table(cellText=all_rows, loc="center", cellLoc="left",
                     colWidths=[0.05, 0.58, 0.08, 0.14, 0.10])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    for j in range(5):
        table[0, j].set_facecolor(C_DARK)
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(all_rows)):
        yr = all_rows[i][2]
        bg = "#EDE9FE" if yr == "2024" else "#CCFBF1"
        for j in range(5):
            table[i, j].set_facecolor(bg)
    ax.set_title(f"Codabench Competition Papers ({len(codabench_papers)} total)", fontweight="bold", pad=15)

    out = PLOTS_DIR / "competition_papers_table.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_summary_statistics_table():
    """Panel 6: Summary Statistics table."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")

    both_24 = sum(1 for r in data_2024 if r.get("has_croissant", "").lower() == "true" and r.get("dataset_url", "").strip())
    both_25 = sum(1 for r in data_2025 if r.get("has_croissant", "").lower() == "true" and r.get("dataset_url", "").strip())
    intersection_count = sum(1 for oid in INTERSECTION_MAP.values() if oid in db_2025_ids)

    table_data = [
        ["", "2024", "2025", "Total"],
        ["D&B Track Papers", str(s24["total"]), str(s25["total"]), str(s24["total"] + s25["total"])],
        ["  w/ Croissant", str(s24["has_croissant"]), str(s25["has_croissant"]), str(s24["has_croissant"] + s25["has_croissant"])],
        ["  w/ Dataset URL", str(s24["has_dataset_url"]), str(s25["has_dataset_url"]), str(s24["has_dataset_url"] + s25["has_dataset_url"])],
        ["  w/ Both", str(both_24), str(both_25), str(both_24 + both_25)],
        ["", "", "", ""],
        ["Competition Track", str(COMP_TRACK_TOTAL_2024), str(COMP_TRACK_TOTAL_2025), str(COMP_TRACK_TOTAL_2024 + COMP_TRACK_TOTAL_2025)],
        ["  on Codabench (ours)", str(len(codabench_2024)), str(len(codabench_2025)), str(len(codabench_papers))],
        ["  Also in D&B 2025", "\u2014", str(intersection_count), str(intersection_count)],
        ["  Pipeline Tasks", "", "", str(pipeline_croissant)],
        ["  Pipeline Bundles", "", "", str(pipeline_bundles)],
    ]

    table = ax.table(cellText=table_data, loc="center", cellLoc="center", colWidths=[0.35, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    for j in range(4):
        table[0, j].set_facecolor(C_DARK)
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in [1, 6]:
        for j in range(4):
            table[i, j].set_text_props(fontweight="bold")
            table[i, j].set_facecolor("#F3F4F6")
    for j in range(4):
        table[5, j].set_facecolor("white")
        table[5, j].set_edgecolor("white")
    for i in [2, 3, 4, 7, 8, 9, 10]:
        for j in range(4):
            table[i, j].set_facecolor("#FAFAFA" if i % 2 == 0 else "white")
    ax.set_title("Summary Statistics", fontweight="bold", pad=20)

    out = PLOTS_DIR / "summary_statistics_table.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


def generate_combined():
    """Generate the full 6-panel combined figure."""
    fig = plt.figure(figsize=(18, 20))
    fig.suptitle("NeurIPS 2024\u20132025: Paper2Codabench Data Overview",
                 fontsize=16, fontweight="bold", y=0.98)

    axes = fig.subplot_mosaic(
        [["panel1", "panel2"],
         ["panel3_placeholder", "panel4"],
         ["panel5", "panel6"]],
        gridspec_kw={"hspace": 0.40, "wspace": 0.30, "height_ratios": [1, 1.2, 1.2]},
    )

    plot_papers_by_year(axes["panel1"])
    plot_croissant_adoption(axes["panel2"])
    plot_hosting_platforms(axes["panel4"])

    # Panel 3 (UpSet) needs custom axes — hide placeholder and draw manually
    axes["panel3_placeholder"].set_visible(False)
    bbox = axes["panel3_placeholder"].get_position()

    set_names = ["Croissant", "Dataset URL", "Codabench"]
    combos = [(key, cnt) for key, cnt in combo_counts.items() if cnt > 0]
    combos.sort(key=lambda x: -x[1])
    n_combos = len(combos)
    n_sets = len(set_names)

    ax_bars = fig.add_axes([bbox.x0, bbox.y0 + bbox.height * 0.45, bbox.width, bbox.height * 0.55])
    ax_dots = fig.add_axes([bbox.x0, bbox.y0, bbox.width, bbox.height * 0.35])

    x_pos = np.arange(n_combos)
    counts = [c[1] for c in combos]
    bar_colors = []
    for combo, _ in combos:
        cr, du, cb = combo
        if cb: bar_colors.append(C_ORANGE)
        elif cr and du: bar_colors.append(C_GREEN)
        elif cr: bar_colors.append(C_BLUE)
        elif du: bar_colors.append(C_TEAL)
        else: bar_colors.append(C_GRAY)

    bars = ax_bars.bar(x_pos, counts, color=bar_colors, edgecolor="white", linewidth=0.8, width=0.6)
    for bar, cnt in zip(bars, counts):
        ax_bars.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3, str(cnt),
                     ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax_bars.set_ylabel("Papers", fontsize=10)
    ax_bars.set_title("NeurIPS 2025 D&B: Data Availability (UpSet-style)", fontweight="bold", pad=10)
    ax_bars.set_xlim(-0.5, n_combos - 0.5)
    ax_bars.set_xticks([])
    ax_bars.set_ylim(0, max(counts) * 1.15)
    for sp in ["top", "right", "bottom"]:
        ax_bars.spines[sp].set_visible(False)

    for i, (combo, _) in enumerate(combos):
        active = list(combo)
        active_indices = [j for j, a in enumerate(active) if a]
        if len(active_indices) >= 2:
            ax_dots.plot([i, i], [min(active_indices), max(active_indices)], color=C_DARK, linewidth=2, zorder=1)
        for j in range(n_sets):
            if active[j]:
                ax_dots.scatter(i, j, s=180, color=C_DARK, zorder=2, edgecolors="none")
            else:
                ax_dots.scatter(i, j, s=180, facecolors="white", edgecolors=C_GRAY, linewidths=1.5, zorder=2)

    ax_dots.set_yticks(range(n_sets))
    ax_dots.set_yticklabels(set_names, fontsize=10)
    ax_dots.set_xlim(-0.5, n_combos - 0.5)
    ax_dots.set_ylim(-0.5, n_sets - 0.5)
    ax_dots.set_xticks([])
    ax_dots.invert_yaxis()
    for sp in ["top", "right", "bottom"]:
        ax_dots.spines[sp].set_visible(False)

    set_totals = [sum(c[1] for c in combos if c[0][0]), sum(c[1] for c in combos if c[0][1]), sum(c[1] for c in combos if c[0][2])]
    for j, tot in enumerate(set_totals):
        ax_dots.text(n_combos - 0.3, j, str(tot), ha="left", va="center", fontsize=9, color="dimgrey", style="italic")

    # Panel 5: Competition Papers table
    ax5 = axes["panel5"]
    ax5.axis("off")
    header = [["#", "Competition (Competition Track)", "Year", "Task", "Also in\nD&B?"]]
    rows = []
    for i, p in enumerate(codabench_papers, 1):
        year_label = "2024" if "2024" in p["track"] else "2025"
        short_title = p["title"][:48] + ("\u2026" if len(p["title"]) > 48 else "")
        task = p.get("task_type", "?")
        or_id = INTERSECTION_MAP.get(p["paper_id"])
        in_db = "\u2713" if (or_id and or_id in db_2025_ids) else "\u2014"
        rows.append([str(i), short_title, year_label, task, in_db])
    all_rows = header + rows
    table5 = ax5.table(cellText=all_rows, loc="center", cellLoc="left", colWidths=[0.05, 0.58, 0.08, 0.14, 0.10])
    table5.auto_set_font_size(False)
    table5.set_fontsize(8)
    table5.scale(1, 1.5)
    for j in range(5):
        table5[0, j].set_facecolor(C_DARK)
        table5[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(all_rows)):
        yr = all_rows[i][2]
        bg = "#EDE9FE" if yr == "2024" else "#CCFBF1"
        for j in range(5):
            table5[i, j].set_facecolor(bg)
    ax5.set_title(f"Codabench Competition Papers ({len(codabench_papers)} total)", fontweight="bold", pad=15)

    # Panel 6: Summary Statistics table
    ax6 = axes["panel6"]
    ax6.axis("off")
    both_24 = sum(1 for r in data_2024 if r.get("has_croissant", "").lower() == "true" and r.get("dataset_url", "").strip())
    both_25 = sum(1 for r in data_2025 if r.get("has_croissant", "").lower() == "true" and r.get("dataset_url", "").strip())
    intersection_count = sum(1 for oid in INTERSECTION_MAP.values() if oid in db_2025_ids)
    table_data = [
        ["", "2024", "2025", "Total"],
        ["D&B Track Papers", str(s24["total"]), str(s25["total"]), str(s24["total"] + s25["total"])],
        ["  w/ Croissant", str(s24["has_croissant"]), str(s25["has_croissant"]), str(s24["has_croissant"] + s25["has_croissant"])],
        ["  w/ Dataset URL", str(s24["has_dataset_url"]), str(s25["has_dataset_url"]), str(s24["has_dataset_url"] + s25["has_dataset_url"])],
        ["  w/ Both", str(both_24), str(both_25), str(both_24 + both_25)],
        ["", "", "", ""],
        ["Competition Track", str(COMP_TRACK_TOTAL_2024), str(COMP_TRACK_TOTAL_2025), str(COMP_TRACK_TOTAL_2024 + COMP_TRACK_TOTAL_2025)],
        ["  on Codabench (ours)", str(len(codabench_2024)), str(len(codabench_2025)), str(len(codabench_papers))],
        ["  Also in D&B 2025", "\u2014", str(intersection_count), str(intersection_count)],
        ["  Pipeline Tasks", "", "", str(pipeline_croissant)],
        ["  Pipeline Bundles", "", "", str(pipeline_bundles)],
    ]
    table6 = ax6.table(cellText=table_data, loc="center", cellLoc="center", colWidths=[0.35, 0.15, 0.15, 0.15])
    table6.auto_set_font_size(False)
    table6.set_fontsize(10)
    table6.scale(1, 1.5)
    for j in range(4):
        table6[0, j].set_facecolor(C_DARK)
        table6[0, j].set_text_props(color="white", fontweight="bold")
    for i in [1, 6]:
        for j in range(4):
            table6[i, j].set_text_props(fontweight="bold")
            table6[i, j].set_facecolor("#F3F4F6")
    for j in range(4):
        table6[5, j].set_facecolor("white")
        table6[5, j].set_edgecolor("white")
    for i in [2, 3, 4, 7, 8, 9, 10]:
        for j in range(4):
            table6[i, j].set_facecolor("#FAFAFA" if i % 2 == 0 else "white")
    ax6.set_title("Summary Statistics", fontweight="bold", pad=20)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.savefig(COMBINED_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved combined: {COMBINED_PATH}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data overview plots")
    parser.add_argument("--combined", action="store_true", help="Also generate combined 6-panel figure")
    args = parser.parse_args()

    # Generate individual PNGs
    plot_papers_by_year()
    plot_croissant_adoption()
    plot_data_availability_upset()
    plot_hosting_platforms()
    plot_competition_papers_table()
    plot_summary_statistics_table()

    if args.combined:
        generate_combined()
