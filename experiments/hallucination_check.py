#!/usr/bin/env python3
"""
Hallucination Detection via Error Planting.

For each paper, plants deliberate errors in the paper text, runs Croissant Task
extraction, and checks if the planted errors propagate to the LLM output.

Usage:
    python experiments/hallucination_check.py
    python experiments/hallucination_check.py --papers paper1,paper2
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import Config
from extract_croissant_task import call_azure_openai
from error_planting import PLANT_TYPES, plant_error, check_planted_error
from prompts import ERROR_PLANTING_SYSTEM, create_error_planting_extraction_prompt

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_paper_text(paper_id: str) -> str:
    """Load saved PDF text for a paper."""
    text_path = Config.CROISSANT_DIR / f"{paper_id}.pdf_text.txt"
    if not text_path.exists():
        raise FileNotFoundError(f"PDF text not found: {text_path}")
    return text_path.read_text(encoding="utf-8")


def run_extraction_with_planted_error(modified_text: str) -> dict:
    """Run Croissant Task extraction on modified paper text."""
    messages = [
        {"role": "system", "content": ERROR_PLANTING_SYSTEM},
        {"role": "user", "content": create_error_planting_extraction_prompt(modified_text)},
    ]

    raw_response = call_azure_openai(messages, max_tokens=16000)

    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        return {"_parse_error": True, "_raw": raw_response[:500]}


def run_hallucination_check(paper_ids: list) -> dict:
    """
    Run hallucination check for given papers.

    Returns:
        Dict with per-paper and per-plant-type results
    """
    all_results = []

    for paper_id in paper_ids:
        print(f"\n{'='*60}")
        print(f"Hallucination check: {paper_id}")
        print(f"{'='*60}")

        try:
            paper_text = load_paper_text(paper_id)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            all_results.append({
                "paper_id": paper_id,
                "status": "skipped",
                "error": str(e),
                "checks": [],
            })
            continue

        # Truncate long text for LLM context
        if len(paper_text) > 80000:
            paper_text = paper_text[:80000]

        paper_checks = []

        for plant_type in PLANT_TYPES:
            print(f"\n  Plant type: {plant_type}")

            # Plant error
            modified_text, plant_record = plant_error(paper_text, plant_type, seed=42)

            if not plant_record["planted"]:
                print(f"    Could not plant error (no matching target found)")
                paper_checks.append({
                    "plant_type": plant_type,
                    "planted": False,
                    "propagated": False,
                    "details": "Could not find target to modify",
                })
                continue

            print(f"    Original: {plant_record['original']}")
            print(f"    Planted:  {plant_record['replacement']}")

            # Run extraction on modified text
            print(f"    Running LLM extraction...")
            try:
                llm_output = run_extraction_with_planted_error(modified_text)
            except Exception as e:
                print(f"    LLM call failed: {e}")
                paper_checks.append({
                    "plant_type": plant_type,
                    "planted": True,
                    "propagated": False,
                    "details": f"LLM call failed: {e}",
                })
                continue

            if llm_output.get("_parse_error"):
                print(f"    LLM response could not be parsed as JSON")
                paper_checks.append({
                    "plant_type": plant_type,
                    "planted": True,
                    "propagated": False,
                    "details": "LLM response not valid JSON",
                })
                continue

            # Check propagation
            detection = check_planted_error(plant_record, llm_output)
            paper_checks.append(detection)

            status = "PROPAGATED" if detection["propagated"] else "NOT propagated"
            print(f"    Result: {status}")
            print(f"    Details: {detection['details']}")

        all_results.append({
            "paper_id": paper_id,
            "status": "completed",
            "checks": paper_checks,
        })

    return all_results


def print_summary(results: list):
    """Print a summary table of hallucination check results."""
    print(f"\n{'='*60}")
    print("Hallucination Check Summary")
    print(f"{'='*60}\n")

    # Per-paper summary
    print(f"| Paper | metric_rename | number_change | dataset_rename | column_rename |")
    print(f"|-------|:-------------:|:-------------:|:--------------:|:-------------:|")

    for r in results:
        if r["status"] == "skipped":
            print(f"| {r['paper_id']} | SKIP | SKIP | SKIP | SKIP |")
            continue

        cells = []
        for plant_type in PLANT_TYPES:
            check = next((c for c in r["checks"] if c["plant_type"] == plant_type), None)
            if check is None:
                cells.append("-")
            elif not check["planted"]:
                cells.append("N/A")
            elif check["propagated"]:
                cells.append("YES")
            else:
                cells.append("no")

        print(f"| {r['paper_id']} | {' | '.join(cells)} |")

    print()

    # Per-plant-type propagation rate
    print("### Propagation rates by plant type:")
    for plant_type in PLANT_TYPES:
        planted = 0
        propagated = 0
        for r in results:
            for c in r.get("checks", []):
                if c["plant_type"] == plant_type and c.get("planted"):
                    planted += 1
                    if c.get("propagated"):
                        propagated += 1
        if planted > 0:
            rate = propagated / planted * 100
            print(f"  {plant_type}: {propagated}/{planted} ({rate:.1f}%)")
        else:
            print(f"  {plant_type}: no valid plantings")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Hallucination detection via error planting"
    )
    parser.add_argument(
        "--papers",
        type=str,
        help="Comma-separated paper IDs (default: all papers with saved PDF text)",
    )

    args = parser.parse_args()

    if args.papers:
        paper_ids = [p.strip() for p in args.papers.split(",")]
    else:
        # Find all papers with saved PDF text
        text_files = sorted(Config.CROISSANT_DIR.glob("*.pdf_text.txt"))
        paper_ids = [f.name.replace(".pdf_text.txt", "") for f in text_files]

    if not paper_ids:
        print("No papers found. Run extract_croissant_task.py first to generate PDF text files.")
        sys.exit(1)

    print(f"Papers to check: {', '.join(paper_ids)}")
    print(f"Plant types: {', '.join(PLANT_TYPES)}")

    results = run_hallucination_check(paper_ids)

    print_summary(results)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "hallucination_results.json"
    with open(output_path, "w") as f:
        json.dump({"results": results, "plant_types": PLANT_TYPES}, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
