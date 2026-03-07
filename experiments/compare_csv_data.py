#!/usr/bin/env python3
"""
Compare two NeurIPS paper data sources for consistency.

Sources:
  1. neurips2025_db_croissants.csv (root, ~1097 rows)
     Columns: paper_id, title, authors, decision, abstract, tldr, keywords, code_url, pdf_url, croissant_url
  2. data/neurips_2025_db_papers.csv (~497 rows)
     Columns: year, paper_id, title, authors, abstract, keywords, has_croissant, croissant_url, dataset_url, openreview_url, pdf_url, is_competition_paper

Compares by paper_id: overlap, unique entries, and field consistency.

Usage:
    python experiments/compare_csv_data.py
"""
import csv
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
ROOT_CSV = BASE / "neurips2025_db_croissants.csv"
DATA_CSV = BASE / "data" / "neurips_2025_db_papers.csv"
RESULTS_DIR = BASE / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compare():
    root_rows = load_csv(ROOT_CSV)
    data_rows = load_csv(DATA_CSV)

    root_by_id = {r["paper_id"]: r for r in root_rows}
    data_by_id = {r["paper_id"]: r for r in data_rows}

    root_ids = set(root_by_id.keys())
    data_ids = set(data_by_id.keys())

    only_root = root_ids - data_ids
    only_data = data_ids - root_ids
    both = root_ids & data_ids

    # Compare shared fields for papers in both
    title_mismatches = []
    croissant_url_mismatches = []

    for pid in sorted(both):
        root_r = root_by_id[pid]
        data_r = data_by_id[pid]

        # Title comparison (case-insensitive, strip whitespace)
        root_title = root_r.get("title", "").strip()
        data_title = data_r.get("title", "").strip()
        if root_title.lower() != data_title.lower():
            title_mismatches.append({
                "paper_id": pid,
                "root_title": root_title[:80],
                "data_title": data_title[:80],
            })

        # Croissant URL comparison
        root_url = root_r.get("croissant_url", "").strip()
        data_url = data_r.get("croissant_url", "").strip()
        if root_url != data_url:
            croissant_url_mismatches.append({
                "paper_id": pid,
                "root_croissant_url": root_url[:120],
                "data_croissant_url": data_url[:120],
            })

    result = {
        "root_csv": str(ROOT_CSV),
        "data_csv": str(DATA_CSV),
        "root_total": len(root_rows),
        "data_total": len(data_rows),
        "in_both": len(both),
        "only_in_root": len(only_root),
        "only_in_data": len(only_data),
        "title_mismatches": len(title_mismatches),
        "croissant_url_mismatches": len(croissant_url_mismatches),
        "details": {
            "only_in_root_ids": sorted(only_root)[:20],
            "only_in_data_ids": sorted(only_data)[:20],
            "title_mismatches": title_mismatches[:10],
            "croissant_url_mismatches": croissant_url_mismatches[:10],
        },
    }

    return result


def print_summary(result):
    print(f"\n{'='*60}")
    print("CSV Data Consistency Check")
    print(f"{'='*60}\n")

    print(f"| Source | File | Rows |")
    print(f"|--------|------|------|")
    print(f"| Root CSV | neurips2025_db_croissants.csv | {result['root_total']} |")
    print(f"| Data CSV | data/neurips_2025_db_papers.csv | {result['data_total']} |")
    print()

    print(f"### Overlap Summary")
    print(f"- Papers in both: {result['in_both']}")
    print(f"- Only in root CSV: {result['only_in_root']}")
    print(f"- Only in data CSV: {result['only_in_data']}")
    print()

    print(f"### Field Consistency (among {result['in_both']} shared papers)")
    print(f"- Title mismatches: {result['title_mismatches']}")
    print(f"- Croissant URL mismatches: {result['croissant_url_mismatches']}")
    print()

    if result["details"]["title_mismatches"]:
        print("### Sample Title Mismatches")
        for m in result["details"]["title_mismatches"][:5]:
            print(f"  {m['paper_id']}:")
            print(f"    Root: {m['root_title']}")
            print(f"    Data: {m['data_title']}")
        print()


def main():
    result = compare()

    # Save results
    output_path = RESULTS_DIR / "csv_comparison.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {output_path}")

    print_summary(result)


if __name__ == "__main__":
    main()
