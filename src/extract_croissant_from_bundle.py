#!/usr/bin/env python3
"""
Pipeline B Step 2: Extract Croissant Task from a generated bundle.

Reads bundle files (competition.yaml, metrics.py, score.py, solution.py, CSV headers)
and uses LLM to produce a Croissant Task JSON-LD.

Output goes to croissant_tasks_pipelineB/<paper_id>.croissant_task.json.

Usage:
    python src/extract_croissant_from_bundle.py bundles_pipelineB/paper1
    python src/extract_croissant_from_bundle.py bundles_pipelineB/paper1 --paper-id paper1
"""
import argparse
import json
import sys
from pathlib import Path

from config import Config
from extract_croissant_task import call_azure_openai
from croissant_schema import CroissantTaskProblem
from prompts import BUNDLE_TO_CROISSANT_SYSTEM, create_bundle_to_croissant_prompt


MAX_FILE_CHARS = 3000
MAX_CSV_ROWS = 5


def read_file_truncated(file_path: Path, max_chars: int = MAX_FILE_CHARS) -> str:
    """Read a file, truncating to max_chars."""
    if not file_path.exists():
        return "(file not found)"
    text = file_path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + f"\n... (truncated at {max_chars} chars)"
    return text


def read_csv_header_and_rows(file_path: Path, max_rows: int = MAX_CSV_ROWS) -> str:
    """Read CSV header and first few rows."""
    if not file_path.exists():
        return "(file not found)"
    lines = file_path.read_text(encoding="utf-8", errors="replace").strip().split("\n")
    selected = lines[: max_rows + 1]  # header + max_rows data rows
    return "\n".join(selected)


def assemble_bundle_context(bundle_path: Path) -> str:
    """Assemble bundle files into a single context string for the LLM."""
    sections = []

    # competition.yaml
    comp_yaml = bundle_path / "competition.yaml"
    sections.append(f"=== competition.yaml ===\n{read_file_truncated(comp_yaml)}")

    # metrics.py
    metrics_py = bundle_path / "scoring_program" / "metrics.py"
    sections.append(f"=== scoring_program/metrics.py ===\n{read_file_truncated(metrics_py)}")

    # score.py
    score_py = bundle_path / "scoring_program" / "score.py"
    sections.append(f"=== scoring_program/score.py ===\n{read_file_truncated(score_py)}")

    # ingestion.py
    ingestion_py = bundle_path / "ingestion_program" / "ingestion.py"
    sections.append(f"=== ingestion_program/ingestion.py ===\n{read_file_truncated(ingestion_py)}")

    # solution.py
    solution_py = bundle_path / "examples" / "solution.py"
    sections.append(f"=== examples/solution.py ===\n{read_file_truncated(solution_py)}")

    # CSV files (headers + first few rows)
    input_csv = bundle_path / "input_data" / "input.csv"
    sections.append(f"=== input_data/input.csv (first {MAX_CSV_ROWS} rows) ===\n{read_csv_header_and_rows(input_csv)}")

    reference_csv = bundle_path / "reference_data" / "reference.csv"
    sections.append(f"=== reference_data/reference.csv (first {MAX_CSV_ROWS} rows) ===\n{read_csv_header_and_rows(reference_csv)}")

    sample_csv = bundle_path / "examples" / "sample_submission.csv"
    if sample_csv.exists():
        sections.append(f"=== examples/sample_submission.csv (first {MAX_CSV_ROWS} rows) ===\n{read_csv_header_and_rows(sample_csv)}")

    return "\n\n".join(sections)


def extract_croissant_from_bundle(bundle_path: Path, paper_id: str) -> dict:
    """
    Main Pipeline B Step 2: Bundle → Croissant Task.

    Args:
        bundle_path: Path to bundle directory
        paper_id: Identifier for the paper

    Returns:
        Extracted Croissant Task as dictionary
    """
    print(f"\n{'='*60}")
    print(f"Pipeline B Step 2: Extract Croissant Task from Bundle")
    print(f"Bundle: {bundle_path}")
    print(f"Paper ID: {paper_id}")
    print(f"{'='*60}\n")

    # Step 1: Assemble bundle context
    print("  Reading bundle files...")
    bundle_context = assemble_bundle_context(bundle_path)
    print(f"  Bundle context: {len(bundle_context)} characters")

    # Step 2: Call LLM
    print("  Calling LLM for Croissant Task extraction...")
    messages = [
        {"role": "system", "content": BUNDLE_TO_CROISSANT_SYSTEM},
        {"role": "user", "content": create_bundle_to_croissant_prompt(bundle_context)},
    ]

    raw_response = call_azure_openai(messages, max_tokens=16000)

    # Step 3: Parse JSON
    try:
        croissant_data = json.loads(raw_response)
        croissant_data["paper_id"] = paper_id
        print("  Successfully parsed JSON response")
    except json.JSONDecodeError as e:
        print(f"  Error parsing JSON: {e}")
        print(f"  Raw response:\n{raw_response[:500]}")
        raise

    # Step 4: Validate with Pydantic
    try:
        validated = CroissantTaskProblem(**croissant_data)
        croissant_dict = validated.model_dump(by_alias=True)
        print("  Croissant Task schema validation passed")
    except Exception as e:
        print(f"  Validation warning: {e}")
        print("  Continuing with unvalidated Croissant Task...")
        croissant_dict = croissant_data

    # Step 5: Log fill-in-the-blank
    fitb = croissant_dict.get("fill_in_the_blank", [])
    if fitb:
        print(f"\n  [FILL IN THE BLANK] placeholders found ({len(fitb)}):")
        for item in fitb:
            print(f"    - {item}")
    else:
        print("  No [FILL IN THE BLANK] placeholders found.")

    # Step 6: Save
    Config.CROISSANT_PIPELINE_B_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Config.CROISSANT_PIPELINE_B_DIR / f"{paper_id}.croissant_task.json"
    with open(output_path, "w") as f:
        json.dump(croissant_dict, f, indent=2)

    # Save raw response
    raw_path = Config.CROISSANT_PIPELINE_B_DIR / f"{paper_id}.raw.json"
    with open(raw_path, "w") as f:
        json.dump({"raw_response": raw_response}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Croissant Task extraction from bundle complete!")
    print(f"   Croissant Task: {output_path}")
    print(f"   Raw response: {raw_path}")
    print(f"{'='*60}\n")

    return croissant_dict


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline B Step 2: Extract Croissant Task from bundle"
    )
    parser.add_argument(
        "bundle_path",
        type=Path,
        help="Path to bundle directory (e.g., bundles_pipelineB/paper1)",
    )
    parser.add_argument(
        "--paper-id",
        type=str,
        help="Paper identifier (default: derived from bundle directory name)",
    )

    args = parser.parse_args()

    if not args.bundle_path.exists():
        print(f"  Error: Bundle directory not found: {args.bundle_path}")
        sys.exit(1)

    paper_id = args.paper_id if args.paper_id else args.bundle_path.name

    try:
        croissant_task = extract_croissant_from_bundle(args.bundle_path, paper_id)
        print(f"  Task: {croissant_task.get('name', 'N/A')}")
        evaluation = croissant_task.get("cr:evaluation", {})
        print(f"  Metric: {evaluation.get('primaryMetric', 'N/A')}")
    except Exception as e:
        print(f"\n  Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
