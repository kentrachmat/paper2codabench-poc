#!/usr/bin/env python3
"""
LLM-emulated participant submission.

Generates a solution.py from a participant's perspective (only sees competition
description and input data), then optionally runs it locally or submits to Codabench.

Usage:
    python src/emulated_submission.py bundles_pipelineB/paper1
    python src/emulated_submission.py bundles/paper1 --competition-id 123
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from config import Config
from llm_client import call_azure_openai_for_code
from prompts import EMULATED_SOLUTION_SYSTEM, create_emulated_solution_prompt


def generate_emulated_solution(bundle_path: Path) -> str:
    """
    Read competition description + input data from a bundle and ask the LLM
    to write a participant-perspective solution.py.

    Returns:
        Generated Python source code
    """
    bundle_path = Path(bundle_path).resolve()

    # Read competition description
    readme_path = bundle_path / "README.md"
    comp_yaml_path = bundle_path / "competition.yaml"
    description = ""
    if readme_path.exists():
        description = readme_path.read_text(encoding="utf-8")[:6000]
    elif comp_yaml_path.exists():
        description = comp_yaml_path.read_text(encoding="utf-8")[:6000]

    # Read input data sample
    input_csv_path = bundle_path / "input_data" / "input.csv"
    if not input_csv_path.exists():
        raise FileNotFoundError(f"input.csv not found in {bundle_path / 'input_data'}")

    import pandas as pd
    input_df = pd.read_csv(input_csv_path)
    input_columns = list(input_df.columns)
    sample_csv = input_df.head(5).to_csv(index=False)

    # Infer target column from reference
    ref_path = bundle_path / "reference_data" / "reference.csv"
    if ref_path.exists():
        ref_df = pd.read_csv(ref_path)
        ref_cols = list(ref_df.columns)
        extra = [c for c in ref_cols if c not in input_columns]
        target_column = extra[0] if extra else ref_cols[-1]
    else:
        target_column = "pred"

    prompt = create_emulated_solution_prompt(
        description, input_columns, target_column, sample_csv
    )
    code = call_azure_openai_for_code(EMULATED_SOLUTION_SYSTEM, prompt)

    # Validate syntax
    compile(code, "<emulated_solution>", "exec")
    return code


def emulated_submit(bundle_path: Path, competition_id: int = None) -> dict:
    """
    Generate an emulated solution, then run locally or submit to Codabench.

    Args:
        bundle_path: Path to bundle directory
        competition_id: If given, submit to Codabench; otherwise run locally

    Returns:
        Scores dict (local) or submission response (Codabench)
    """
    bundle_path = Path(bundle_path).resolve()

    print(f"\nGenerating emulated solution for {bundle_path.name}...")
    code = generate_emulated_solution(bundle_path)

    # Write to a temp file inside bundle/examples/
    emulated_path = bundle_path / "examples" / "emulated_solution.py"
    emulated_path.write_text(code, encoding="utf-8")
    print(f"  Saved emulated solution: {emulated_path}")

    if competition_id is not None:
        # Submit to Codabench
        from codabench_client import CodabenchClient
        client = CodabenchClient()
        return client.submit_solution(competition_id, str(emulated_path))

    # Run locally
    print("  Running local simulation...")
    local_run = Path(__file__).resolve().parent / "local_run.py"
    result = subprocess.run(
        [sys.executable, str(local_run), str(bundle_path), str(emulated_path)],
        capture_output=True, text=True, timeout=300,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Local simulation failed")

    # Try to parse scores from output
    scores_line = [l for l in result.stdout.splitlines() if ":" in l and any(
        c.isdigit() for c in l)]
    return {"status": "completed", "output": result.stdout[-500:]}


def main():
    parser = argparse.ArgumentParser(description="LLM-emulated participant submission")
    parser.add_argument("bundle_path", type=Path, help="Path to bundle directory")
    parser.add_argument("--competition-id", type=int, help="Codabench competition ID for remote submit")
    args = parser.parse_args()

    if not args.bundle_path.exists():
        print(f"Error: Bundle not found: {args.bundle_path}")
        sys.exit(1)

    try:
        result = emulated_submit(args.bundle_path, args.competition_id)
        print(f"\nResult: {json.dumps(result, indent=2, default=str)}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
