#!/usr/bin/env python3
"""
LLM Judge: rates the quality of a generated Codabench competition bundle.

Usage:
    python src/judge_rating.py bundles_pipelineB/paper1
    python src/judge_rating.py bundles/paper1
"""
import argparse
import json
import sys
from pathlib import Path

from config import Config
from llm_client import call_azure_openai_for_code
from prompts import JUDGE_RATING_SYSTEM, create_judge_rating_prompt


def _read_file_safe(path: Path, max_chars: int = 4000) -> str:
    """Read file content, truncated for LLM context."""
    if not path.exists():
        return "[FILE NOT FOUND]"
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + "\n... [TRUNCATED]"
    return text


def judge_competition(bundle_path: Path) -> dict:
    """
    Read all bundle files and ask the LLM to evaluate quality.

    Returns:
        Dict with overall_score, criteria_scores, and feedback
    """
    bundle_path = Path(bundle_path).resolve()

    # Build context from bundle files
    files_to_read = [
        ("README.md", 4000),
        ("competition.yaml", 2000),
        ("scoring_program/metrics.py", 4000),
        ("scoring_program/score.py", 4000),
        ("ingestion_program/ingestion.py", 4000),
        ("examples/solution.py", 3000),
        ("input_data/input.csv", 2000),
        ("reference_data/reference.csv", 2000),
    ]

    context_parts = []
    for rel_path, max_chars in files_to_read:
        content = _read_file_safe(bundle_path / rel_path, max_chars)
        context_parts.append(f"=== {rel_path} ===\n{content}")

    bundle_context = "\n\n".join(context_parts)

    prompt = create_judge_rating_prompt(bundle_context)
    raw = call_azure_openai_for_code(JUDGE_RATING_SYSTEM, prompt, max_tokens=2000)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "overall_score": None,
            "criteria_scores": {},
            "feedback": raw[:500],
            "_parse_error": True,
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="LLM judge for Codabench bundle quality")
    parser.add_argument("bundle_path", type=Path, help="Path to bundle directory")
    args = parser.parse_args()

    if not args.bundle_path.exists():
        print(f"Error: Bundle not found: {args.bundle_path}")
        sys.exit(1)

    print(f"Judging bundle: {args.bundle_path.name}")
    result = judge_competition(args.bundle_path)

    print(f"\nJudge Rating:")
    print(f"  Overall Score: {result.get('overall_score', 'N/A')}")
    criteria = result.get("criteria_scores", {})
    for k, v in criteria.items():
        print(f"  {k}: {v}")
    print(f"\nFeedback: {result.get('feedback', 'N/A')}")

    # Save result alongside bundle
    output_path = args.bundle_path / "judge_rating.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
