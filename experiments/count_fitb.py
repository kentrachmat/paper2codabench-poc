#!/usr/bin/env python3
"""
Count [FILL IN THE BLANK] placeholders across all Croissant Task JSONs.

Scans both Pipeline A (croissant_tasks/) and Pipeline B (croissant_tasks_pipelineB/)
directories, counts FITB entries from:
  1. The fill_in_the_blank array
  2. Deep text search for literal "[FILL IN THE BLANK]" strings in all values

Outputs a comparison markdown table and saves JSON to experiments/results/fitb_results.json.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CROISSANT_DIR_A = PROJECT_ROOT / "croissant_tasks"
CROISSANT_DIR_B = PROJECT_ROOT / "croissant_tasks_pipelineB"
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


def scan_croissant_dir(directory: Path, metadata: dict) -> list:
    """
    Scan a directory of Croissant Task JSONs and count FITB entries.

    Args:
        directory: Path to croissant tasks directory
        metadata: Paper metadata dict keyed by paper_id

    Returns:
        List of result dicts per paper
    """
    ct_files = sorted(directory.glob("*.croissant_task.json"))
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

    return results


def main():
    metadata = load_metadata()

    # Scan Pipeline A
    results_a = scan_croissant_dir(CROISSANT_DIR_A, metadata)

    # Scan Pipeline B
    results_b = scan_croissant_dir(CROISSANT_DIR_B, metadata) if CROISSANT_DIR_B.exists() else []

    # Build lookup by paper_id
    a_by_id = {r["paper_id"]: r for r in results_a}
    b_by_id = {r["paper_id"]: r for r in results_b}

    all_paper_ids = sorted(set(list(a_by_id.keys()) + list(b_by_id.keys())))

    if not all_paper_ids:
        print("No croissant task files found in either pipeline directory.")
        sys.exit(1)

    # Print comparison table
    print()
    print("## Fill-in-the-Blank (FITB) Comparison: Pipeline A vs Pipeline B")
    print()
    print(f"| Paper | Title | A_fitb_array | A_fitb_deep | B_fitb_array | B_fitb_deep |")
    print(f"|-------|-------|:------------:|:-----------:|:------------:|:-----------:|")
    for pid in all_paper_ids:
        a = a_by_id.get(pid, {})
        b = b_by_id.get(pid, {})
        title = a.get("title", b.get("title", "Unknown"))
        short_title = title[:40] + ("..." if len(title) > 40 else "")
        a_arr = a.get("fitb_array_count", "-")
        a_deep = a.get("deep_search_count", "-")
        b_arr = b.get("fitb_array_count", "-")
        b_deep = b.get("deep_search_count", "-")
        print(f"| {pid} | {short_title} | {a_arr} | {a_deep} | {b_arr} | {b_deep} |")
    print()

    # Totals
    total_a_arr = sum(r["fitb_array_count"] for r in results_a)
    total_a_deep = sum(r["deep_search_count"] for r in results_a)
    total_b_arr = sum(r["fitb_array_count"] for r in results_b)
    total_b_deep = sum(r["deep_search_count"] for r in results_b)

    print(f"**Pipeline A:** {len(results_a)} papers, FITB array={total_a_arr}, deep={total_a_deep}")
    print(f"**Pipeline B:** {len(results_b)} papers, FITB array={total_b_arr}, deep={total_b_deep}")
    print()

    # Print details for each paper
    for r in results_a:
        if r["fitb_array_items"]:
            print(f"### {r['paper_id']} (Pipeline A): FITB items")
            for item in r["fitb_array_items"]:
                print(f"  - {item}")
            print()

    for r in results_b:
        if r["fitb_array_items"]:
            print(f"### {r['paper_id']} (Pipeline B): FITB items")
            for item in r["fitb_array_items"]:
                print(f"  - {item}")
            print()

    # Save JSON results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "fitb_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "pipeline_a": {
                "results": results_a,
                "total_papers": len(results_a),
                "total_fitb_array": total_a_arr,
                "total_fitb_deep": total_a_deep,
            },
            "pipeline_b": {
                "results": results_b,
                "total_papers": len(results_b),
                "total_fitb_array": total_b_arr,
                "total_fitb_deep": total_b_deep,
            },
        }, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
