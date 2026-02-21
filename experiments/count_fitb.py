#!/usr/bin/env python3
"""
Count [FILL IN THE BLANK] placeholders across all Croissant Task JSONs.

Reads croissant_tasks/*.croissant_task.json, counts FITB entries from:
  1. The fill_in_the_blank array
  2. Deep text search for literal "[FILL IN THE BLANK]" strings in all values

Outputs a markdown table to stdout and saves JSON to experiments/results/fitb_results.json.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CROISSANT_DIR = PROJECT_ROOT / "croissant_tasks"
METADATA_PATH = PROJECT_ROOT / "papers" / "metadata.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

FITB_MARKER = "[FILL IN THE BLANK]"


def load_metadata():
    """Load paper metadata for titles and task types."""
    with open(METADATA_PATH, "r") as f:
        data = json.load(f)
    return {p["paper_id"]: p for p in data["competitions"]}


def deep_search_fitb(obj, path=""):
    """Recursively search all string values for FITB_MARKER occurrences."""
    found = []
    if isinstance(obj, str):
        if FITB_MARKER in obj:
            found.append(path or "(root)")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            found.extend(deep_search_fitb(v, f"{path}.{k}" if path else k))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            found.extend(deep_search_fitb(item, f"{path}[{i}]"))
    return found


def main():
    metadata = load_metadata()

    # Find all croissant task files
    ct_files = sorted(CROISSANT_DIR.glob("*.croissant_task.json"))
    if not ct_files:
        print("No croissant task files found in", CROISSANT_DIR)
        sys.exit(1)

    results = []

    for ct_file in ct_files:
        paper_id = ct_file.name.replace(".croissant_task.json", "")
        with open(ct_file, "r") as f:
            data = json.load(f)

        # Method 1: explicit fill_in_the_blank array
        fitb_array = data.get("fill_in_the_blank", [])
        array_count = len(fitb_array)

        # Method 2: deep text search
        deep_hits = deep_search_fitb(data)
        deep_count = len(deep_hits)

        meta = metadata.get(paper_id, {})
        results.append({
            "paper_id": paper_id,
            "title": meta.get("title", "Unknown"),
            "task_type": meta.get("task_type", "Unknown"),
            "fitb_array_count": array_count,
            "fitb_array_items": fitb_array,
            "deep_search_count": deep_count,
            "deep_search_paths": deep_hits,
        })

    # Print markdown table
    print()
    print("## Fill-in-the-Blank (FITB) Summary")
    print()
    print(f"| Paper | Title | Task Type | FITB (array) | FITB (deep) |")
    print(f"|-------|-------|-----------|:------------:|:-----------:|")
    for r in results:
        short_title = r["title"][:50] + ("..." if len(r["title"]) > 50 else "")
        print(f"| {r['paper_id']} | {short_title} | {r['task_type']} | {r['fitb_array_count']} | {r['deep_search_count']} |")
    print()

    total_array = sum(r["fitb_array_count"] for r in results)
    total_deep = sum(r["deep_search_count"] for r in results)
    print(f"**Total papers:** {len(results)}")
    print(f"**Total FITB (array):** {total_array}  |  **Total FITB (deep search):** {total_deep}")
    print()

    # Print details for each paper
    for r in results:
        if r["fitb_array_items"]:
            print(f"### {r['paper_id']}: FITB items")
            for item in r["fitb_array_items"]:
                print(f"  - {item}")
            print()

    # Save JSON results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "fitb_results.json"
    with open(output_path, "w") as f:
        json.dump({"results": results, "total_papers": len(results),
                    "total_fitb_array": total_array, "total_fitb_deep": total_deep}, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
