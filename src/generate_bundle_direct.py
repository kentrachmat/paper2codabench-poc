#!/usr/bin/env python3
"""
Pipeline B Step 1: Generate Codabench bundle directly from paper PDF.

Skips Croissant Task as intermediate format — goes straight from PDF to bundle.
Output goes to bundles_pipelineB/<paper_id>/.

Usage:
    python src/generate_bundle_direct.py papers/paper1.pdf
    python src/generate_bundle_direct.py papers/paper1.pdf --paper-id paper1
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from config import Config
from llm_client import call_azure_openai
from extract_croissant_task import extract_pdf_text, summarize_long_paper, CHUNK_PAGE_THRESHOLD
from generate_bundle import (
    create_bundle_structure,
    generate_bundle_files_with_llm,
    generate_toy_data_with_llm,
    fill_template,
    load_template,
    extract_bundle_vars,
    generate_fallback_solution,
)
from prompts import DIRECT_BUNDLE_SYSTEM, create_direct_bundle_prompt, infer_task_type


def adapt_summary_to_croissant_shape(summary: dict, paper_id: str) -> dict:
    """
    Convert the compact JSON summary from the direct bundle prompt
    into a dict shaped like a Croissant Task, so that existing
    generate_bundle_files_with_llm() and extract_bundle_vars() work.
    """
    fields = summary.get("output_schema", [
        {"name": "id", "dataType": "sc:Text", "description": "unique identifier"},
        {"name": "pred", "dataType": "sc:Float", "description": "prediction"},
    ])

    packages = summary.get("packages", ["pandas", "numpy", "scikit-learn"])

    adapted = {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "dct": "http://purl.org/dc/terms/",
            "sc": "https://schema.org/",
        },
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "@type": "cr:TaskProblem",
        "@id": f"{paper_id}-task",
        "name": summary.get("name", "Unknown Task"),
        "description": summary.get("description", ""),
        "paper_id": paper_id,
        "cr:input": [
            {
                "name": "competition_dataset",
                "description": summary.get("input_description", "[FILL IN THE BLANK: description of input data format and columns]"),
                "url": "[FILL IN THE BLANK: URL to the competition dataset]",
            }
        ],
        "cr:output": {
            "name": "predictions",
            "description": summary.get("output_description", "[FILL IN THE BLANK: description of expected output format]"),
            "cr:schema": {
                "name": "prediction_schema",
                "field": fields,
            },
        },
        "cr:implementation": {
            "cr:environment": {
                "language": "Python",
                "packages": packages,
            },
            "entryPoint": "solution.py",
            "interface": "predict(input_dir, output_dir)",
        },
        "cr:evaluation": {
            "primaryMetric": summary.get("primary_metric", "accuracy"),
            "metrics": summary.get("metrics", [summary.get("primary_metric", "accuracy")]),
            "higherIsBetter": summary.get("higher_is_better", True),
            "notes": summary.get("metric_notes", ""),
        },
        "cr:execution": {
            "runtimeLimitSec": summary.get("runtime_limit_sec", 600),
            "memoryLimitMb": summary.get("memory_limit_mb", 4096),
            "allowedExternalData": "unknown",
        },
        "open_questions": summary.get("open_questions", []),
        "fill_in_the_blank": summary.get("fill_in_the_blank", []),
    }

    return adapted


def generate_bundle_direct(pdf_path: Path, paper_id: str, output_dir: Path = None) -> Path:
    """
    Main Pipeline B Step 1: PDF → Bundle directly.

    Args:
        pdf_path: Path to PDF file
        paper_id: Identifier for the paper
        output_dir: Output directory (default: bundles_pipelineB/<paper_id>)

    Returns:
        Path to generated bundle
    """
    print(f"\n{'='*60}")
    print(f"Pipeline B: Direct Bundle Generation")
    print(f"Paper: {pdf_path.name} (ID: {paper_id})")
    print(f"{'='*60}\n")

    # Step 1: Extract PDF text
    paper_text, page_texts = extract_pdf_text(pdf_path)

    # Save extracted PDF text
    pdf_text_path = Config.BUNDLES_PIPELINE_B_DIR / f"{paper_id}.pdf_text.txt"
    pdf_text_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pdf_text_path, "w", encoding="utf-8") as f:
        f.write(paper_text)

    # Step 2: Summarize if long
    num_pages = len(page_texts)
    if num_pages > CHUNK_PAGE_THRESHOLD:
        paper_text = summarize_long_paper(page_texts)
    elif len(paper_text) > 100000:
        print(f"  Paper text is very long ({len(paper_text)} chars), truncating...")
        paper_text = paper_text[:100000]

    # Step 3: Call LLM for direct bundle summary (JSON mode)
    print("\n  Calling LLM for direct bundle specification...")
    user_prompt = create_direct_bundle_prompt(paper_text)
    messages = [
        {"role": "system", "content": DIRECT_BUNDLE_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    raw_response = call_azure_openai(messages, max_tokens=16000)

    # Parse JSON response
    try:
        summary = json.loads(raw_response)
    except json.JSONDecodeError as e:
        print(f"  Error parsing LLM response as JSON: {e}")
        print(f"  Raw response:\n{raw_response[:500]}")
        raise

    print(f"  Task: {summary.get('name', 'Unknown')}")
    print(f"  Metric: {summary.get('primary_metric', 'N/A')}")

    # Step 4: Adapt to Croissant-compatible shape
    croissant_task = adapt_summary_to_croissant_shape(summary, paper_id)

    # Save the adapted croissant task for reference
    adapted_path = Config.BUNDLES_PIPELINE_B_DIR / f"{paper_id}.adapted_croissant.json"
    with open(adapted_path, "w") as f:
        json.dump(croissant_task, f, indent=2)
    print(f"  Adapted Croissant Task saved to {adapted_path.name}")

    # Determine output directory
    if output_dir is None:
        output_dir = Config.BUNDLES_PIPELINE_B_DIR / paper_id

    bundle_path = Path(output_dir)

    if bundle_path.exists():
        print(f"\n  Bundle directory already exists: {bundle_path}")
        response = input("  Overwrite? (y/n): ")
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)
        shutil.rmtree(bundle_path)

    # Step 5: Create bundle structure
    create_bundle_structure(bundle_path)

    # Step 6: Generate bundle files with LLM (reuse existing function)
    template_vars, fallback_log = extract_bundle_vars(croissant_task)
    if fallback_log:
        print("\n  Sanitization fallbacks applied:")
        for entry in fallback_log:
            print(f"    - {entry}")
    llm_generation_successful = False

    try:
        generated_files = generate_bundle_files_with_llm(croissant_task)

        print("\n  Writing LLM-generated files...")
        (bundle_path / "ingestion_program" / "ingestion.py").write_text(generated_files["ingestion.py"])
        (bundle_path / "scoring_program" / "score.py").write_text(generated_files["score.py"])
        (bundle_path / "scoring_program" / "metrics.py").write_text(generated_files["metrics.py"])
        (bundle_path / "examples" / "solution.py").write_text(generated_files["solution.py"])

        llm_generation_successful = True
        print("  LLM generation successful!")

    except Exception as e:
        print(f"\n  LLM generation failed: {e}")
        print("  Falling back to template-based generation...")

    if not llm_generation_successful:
        print("\n  Generating with templates...")
        from generate_bundle import load_template as lt, fill_template as ft

        ingestion_py = ft(lt("ingestion.py.template"), template_vars)
        (bundle_path / "ingestion_program" / "ingestion.py").write_text(ingestion_py)

        score_py = ft(lt("score.py.template"), template_vars)
        (bundle_path / "scoring_program" / "score.py").write_text(score_py)

        metrics_py = ft(lt("metrics.py.template"), template_vars)
        (bundle_path / "scoring_program" / "metrics.py").write_text(metrics_py)

        solution_py = generate_fallback_solution(croissant_task)
        (bundle_path / "examples" / "solution.py").write_text(solution_py)

        print("  Template-based generation completed")

    # Step 7: Generate competition.yaml
    print("\n  Generating competition.yaml...")
    comp_yaml = fill_template(load_template("competition.yaml.template"), template_vars)
    (bundle_path / "competition.yaml").write_text(comp_yaml)

    # Metadata files
    (bundle_path / "ingestion_program" / "metadata").write_text("command: python ingestion.py\n")
    (bundle_path / "scoring_program" / "metadata").write_text("command: python score.py\n")

    # Generate ancillary HTML files and logo
    print("\n  Generating ancillary files...")
    overview_html = fill_template(load_template("overview.html.template"), template_vars)
    evaluation_html = fill_template(load_template("evaluation.html.template"), template_vars)
    terms_html = fill_template(load_template("terms.html.template"), template_vars)
    (bundle_path / "overview.html").write_text(overview_html)
    (bundle_path / "evaluation.html").write_text(evaluation_html)
    (bundle_path / "terms.html").write_text(terms_html)

    import shutil as _shutil
    logo_src = Config.TEMPLATES_DIR / "logo.png"
    if logo_src.exists():
        _shutil.copy(logo_src, bundle_path / "logo.png")
    print("  Ancillary files created")

    # Step 8: Generate toy data
    print("\n  Generating toy data...")
    input_df, reference_df, sample_submission_df = generate_toy_data_with_llm(croissant_task, num_samples=20)

    (bundle_path / "input_data" / "input.csv").write_text(input_df.to_csv(index=False))
    (bundle_path / "reference_data" / "reference.csv").write_text(reference_df.to_csv(index=False))
    (bundle_path / "examples" / "sample_submission.csv").write_text(sample_submission_df.to_csv(index=False))

    # Step 9: Generate README
    task_name = croissant_task.get("name", "Unknown Task")
    task_type = infer_task_type(croissant_task)
    evaluation = croissant_task.get("cr:evaluation", {})
    execution = croissant_task.get("cr:execution", {})

    readme_content = f"""# {task_name}

Auto-generated Codabench bundle (Pipeline B: direct from paper).

## Task Overview

**Type:** {task_type}
**Primary Metric:** {evaluation.get('primaryMetric', 'N/A')}

{croissant_task.get('description', 'No description provided')}

## Submission Format

**Type:** Code execution
**File:** solution.py
**Interface:** `predict(input_dir, output_dir)`

## Evaluation

Primary metric: {evaluation.get('primaryMetric', 'N/A')}
Higher is better: {evaluation.get('higherIsBetter', True)}

## Resource Limits

- Runtime: {execution.get('runtimeLimitSec', 600)} seconds
- Memory: {execution.get('memoryLimitMb', 4096)} MB

---

Generated by Paper2Codabench (Pipeline B)
"""
    (bundle_path / "README.md").write_text(readme_content)

    print(f"\n{'='*60}")
    print(f"Pipeline B bundle generation complete!")
    print(f"   Location: {bundle_path}")
    print(f"   Task: {task_name}")
    print(f"   Type: {task_type}")
    print(f"{'='*60}\n")

    # Validate bundle
    from bundle_validator import validate_bundle as _validate_bundle
    is_valid, validation_errors = _validate_bundle(bundle_path)
    if is_valid:
        print("  Bundle validation: PASSED")
    else:
        print(f"  Bundle validation: {len(validation_errors)} issues")
        for err in validation_errors:
            print(f"    - {err}")

    # Show bundle contents
    print("\nBundle contents:")
    for item in sorted(bundle_path.rglob("*")):
        if item.is_file():
            rel_path = item.relative_to(bundle_path)
            size = item.stat().st_size
            print(f"  {rel_path} ({size} bytes)")

    return bundle_path


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline B: Generate Codabench bundle directly from paper PDF"
    )
    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to PDF file (e.g., papers/paper1.pdf)",
    )
    parser.add_argument(
        "--paper-id",
        type=str,
        help="Paper identifier (default: derived from filename)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output directory for bundle (default: bundles_pipelineB/<paper_id>)",
    )

    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"  Error: PDF file not found: {args.pdf_path}")
        sys.exit(1)

    paper_id = args.paper_id if args.paper_id else args.pdf_path.stem

    try:
        bundle_path = generate_bundle_direct(args.pdf_path, paper_id, args.output)
        print(f"\n  Success! Bundle ready at: {bundle_path}")
    except Exception as e:
        print(f"\n  Error generating bundle: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
