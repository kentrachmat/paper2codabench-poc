#!/usr/bin/env python3
"""
Run all experiments: Pipeline A, Pipeline B, verification, and FITB comparison.

Usage:
    python experiments/run_all.py
    python experiments/run_all.py --pipeline a|b|both
    python experiments/run_all.py --papers paper1,paper2
    python experiments/run_all.py --skip-llm        # only verify + count (no LLM calls)
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import Config

RESULTS_DIR = Path(__file__).resolve().parent / "results"
METADATA_PATH = PROJECT_ROOT / "papers" / "metadata.json"


def load_metadata():
    """Load paper metadata."""
    with open(METADATA_PATH, "r") as f:
        data = json.load(f)
    return {p["paper_id"]: p for p in data["competitions"]}


def get_paper_ids(args_papers: str = None) -> list:
    """Get list of paper IDs to process."""
    if args_papers:
        return [p.strip() for p in args_papers.split(",")]

    # Find all PDFs in papers/
    pdf_files = sorted(Config.PAPERS_DIR.glob("*.pdf"))
    return [f.stem for f in pdf_files]


def run_pipeline_a(paper_id: str):
    """Run Pipeline A: Paper → Croissant Task → Bundle."""
    pdf_path = Config.PAPERS_DIR / f"{paper_id}.pdf"
    if not pdf_path.exists():
        print(f"  SKIP {paper_id}: PDF not found at {pdf_path}")
        return False

    # Step 1: Extract Croissant Task
    croissant_path = Config.CROISSANT_DIR / f"{paper_id}.croissant_task.json"
    if not croissant_path.exists():
        print(f"\n  [Pipeline A] Step 1: Extracting Croissant Task for {paper_id}...")
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "src" / "extract_croissant_task.py"),
             str(pdf_path), "--paper-id", paper_id],
            cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            print(f"  FAIL: Croissant extraction failed for {paper_id}")
            print(result.stderr[-500:] if result.stderr else result.stdout[-500:])
            return False
    else:
        print(f"  [Pipeline A] Step 1: Croissant Task already exists for {paper_id}")

    # Step 2: Generate Bundle
    bundle_path = Config.BUNDLES_DIR / paper_id
    if not bundle_path.exists():
        print(f"  [Pipeline A] Step 2: Generating bundle for {paper_id}...")
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "src" / "generate_bundle.py"),
             str(croissant_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=600,
            input="y\n",  # Auto-confirm overwrite
        )
        if result.returncode != 0:
            print(f"  FAIL: Bundle generation failed for {paper_id}")
            print(result.stderr[-500:] if result.stderr else result.stdout[-500:])
            return False
    else:
        print(f"  [Pipeline A] Step 2: Bundle already exists for {paper_id}")

    return True


def run_pipeline_b(paper_id: str):
    """Run Pipeline B: Paper → Bundle → Croissant Task."""
    pdf_path = Config.PAPERS_DIR / f"{paper_id}.pdf"
    if not pdf_path.exists():
        print(f"  SKIP {paper_id}: PDF not found at {pdf_path}")
        return False

    # Step 1: Generate Bundle directly
    bundle_path = Config.BUNDLES_PIPELINE_B_DIR / paper_id
    if not bundle_path.exists():
        print(f"\n  [Pipeline B] Step 1: Direct bundle generation for {paper_id}...")
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "src" / "generate_bundle_direct.py"),
             str(pdf_path), "--paper-id", paper_id],
            cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=600,
            input="y\n",  # Auto-confirm overwrite
        )
        if result.returncode != 0:
            print(f"  FAIL: Direct bundle generation failed for {paper_id}")
            print(result.stderr[-500:] if result.stderr else result.stdout[-500:])
            return False
    else:
        print(f"  [Pipeline B] Step 1: Bundle already exists for {paper_id}")

    # Step 2: Extract Croissant Task from bundle
    croissant_path = Config.CROISSANT_PIPELINE_B_DIR / f"{paper_id}.croissant_task.json"
    if not croissant_path.exists():
        print(f"  [Pipeline B] Step 2: Extracting Croissant Task from bundle for {paper_id}...")
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "src" / "extract_croissant_from_bundle.py"),
             str(bundle_path), "--paper-id", paper_id],
            cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            print(f"  FAIL: Croissant extraction from bundle failed for {paper_id}")
            print(result.stderr[-500:] if result.stderr else result.stdout[-500:])
            return False
    else:
        print(f"  [Pipeline B] Step 2: Croissant Task already exists for {paper_id}")

    return True


def run_verification():
    """Run bundle verification for both pipelines."""
    print(f"\n{'='*60}")
    print("Running bundle verification...")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "experiments" / "verify_bundles.py")],
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True, timeout=1200,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode == 0


def run_fitb_count():
    """Run FITB counting for both pipelines."""
    print(f"\n{'='*60}")
    print("Running FITB count...")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "experiments" / "count_fitb.py")],
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True, timeout=120,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode == 0


def print_unified_summary():
    """Print unified comparison from saved results."""
    print(f"\n{'='*60}")
    print("UNIFIED COMPARISON SUMMARY")
    print(f"{'='*60}\n")

    # Load verification results
    verif_path = RESULTS_DIR / "verification_results.json"
    if verif_path.exists():
        with open(verif_path) as f:
            verif = json.load(f)
        a_verif = {r["paper_id"]: r for r in verif.get("pipeline_a", {}).get("results", [])}
        b_verif = {r["paper_id"]: r for r in verif.get("pipeline_b", {}).get("results", [])}
    else:
        a_verif, b_verif = {}, {}

    # Load FITB results
    fitb_path = RESULTS_DIR / "fitb_results.json"
    if fitb_path.exists():
        with open(fitb_path) as f:
            fitb = json.load(f)
        a_fitb = {r["paper_id"]: r for r in fitb.get("pipeline_a", {}).get("results", [])}
        b_fitb = {r["paper_id"]: r for r in fitb.get("pipeline_b", {}).get("results", [])}
    else:
        a_fitb, b_fitb = {}, {}

    all_ids = sorted(set(
        list(a_verif.keys()) + list(b_verif.keys()) +
        list(a_fitb.keys()) + list(b_fitb.keys())
    ))

    if not all_ids:
        print("No results found.")
        return

    print(f"| Paper | A_status | A_FITB | B_status | B_FITB |")
    print(f"|-------|:--------:|:------:|:--------:|:------:|")
    for pid in all_ids:
        a_st = a_verif.get(pid, {}).get("status", "-")
        b_st = b_verif.get(pid, {}).get("status", "-")
        a_fb = a_fitb.get(pid, {}).get("fitb_array_count", "-")
        b_fb = b_fitb.get(pid, {}).get("fitb_array_count", "-")
        print(f"| {pid} | {a_st} | {a_fb} | {b_st} | {b_fb} |")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run all experiments for Pipeline A and B comparison"
    )
    parser.add_argument(
        "--pipeline",
        choices=["a", "b", "both"],
        default="both",
        help="Which pipeline(s) to run (default: both)",
    )
    parser.add_argument(
        "--papers",
        type=str,
        help="Comma-separated paper IDs (default: all papers in papers/)",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM calls, only run verification and FITB counting",
    )

    args = parser.parse_args()

    paper_ids = get_paper_ids(args.papers)
    if not paper_ids:
        print("No papers found in papers/ directory.")
        sys.exit(1)

    print(f"Papers: {', '.join(paper_ids)}")
    print(f"Pipeline: {args.pipeline}")
    print(f"Skip LLM: {args.skip_llm}")

    if not args.skip_llm:
        # Run pipelines
        for paper_id in paper_ids:
            if args.pipeline in ("a", "both"):
                print(f"\n--- Pipeline A: {paper_id} ---")
                run_pipeline_a(paper_id)

            if args.pipeline in ("b", "both"):
                print(f"\n--- Pipeline B: {paper_id} ---")
                run_pipeline_b(paper_id)

    # Run verification
    run_verification()

    # Run FITB count
    run_fitb_count()

    # Print unified summary
    print_unified_summary()

    # Save unified results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    unified = {
        "papers": paper_ids,
        "pipeline": args.pipeline,
        "skip_llm": args.skip_llm,
    }

    # Load sub-results if available
    for name in ("verification_results", "fitb_results"):
        path = RESULTS_DIR / f"{name}.json"
        if path.exists():
            with open(path) as f:
                unified[name] = json.load(f)

    output_path = RESULTS_DIR / "run_all_results.json"
    with open(output_path, "w") as f:
        json.dump(unified, f, indent=2)
    print(f"\nUnified results saved to {output_path}")


if __name__ == "__main__":
    main()
