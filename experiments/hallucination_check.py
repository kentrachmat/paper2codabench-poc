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
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import Config
from llm_client import call_azure_openai
from error_planting import PLANT_TYPES, plant_error, check_planted_error
from prompts import ERROR_PLANTING_SYSTEM, create_error_planting_extraction_prompt

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARTIFACTS_DIR = RESULTS_DIR / "hallucination_artifacts"

# ── Logging setup ───────────────────────────────────────────────────────────
LOG_PATH = RESULTS_DIR / "hallucination_check.log"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, mode="w"),
    ],
)
log = logging.getLogger("hallucination_check")

# Max chars of paper text to send per LLM call (reduces tokens dramatically).
# 15 000 chars ≈ 3 750 tokens; keeps total prompt well under 8 K.
MAX_PAPER_CHARS = 15_000
# Output tokens: a Croissant JSON is ≤ 2 K tokens; no need for 16 K.
MAX_OUTPUT_TOKENS = 4_000


def load_paper_text(paper_id: str) -> str:
    """Load saved PDF text for a paper."""
    text_path = Config.CROISSANT_DIR / f"{paper_id}.pdf_text.txt"
    if not text_path.exists():
        raise FileNotFoundError(f"PDF text not found: {text_path}")
    return text_path.read_text(encoding="utf-8")


def run_extraction_with_planted_error(modified_text: str) -> dict:
    """Run Croissant Task extraction on modified paper text."""
    user_prompt = create_error_planting_extraction_prompt(modified_text)
    total_chars = len(ERROR_PLANTING_SYSTEM) + len(user_prompt)
    log.debug(
        "LLM call — system: %d chars, user: %d chars, total: %d chars (~%d tokens), "
        "max_output_tokens: %d",
        len(ERROR_PLANTING_SYSTEM),
        len(user_prompt),
        total_chars,
        total_chars // 4,
        MAX_OUTPUT_TOKENS,
    )

    messages = [
        {"role": "system", "content": ERROR_PLANTING_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

    raw_response = call_azure_openai(messages, max_tokens=MAX_OUTPUT_TOKENS)
    log.debug("LLM response: %d chars", len(raw_response))

    try:
        return json.loads(raw_response)
    except json.JSONDecodeError as exc:
        log.warning("JSON parse error: %s — first 300 chars: %s", exc, raw_response[:300])
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
            log.warning("SKIP %s: %s", paper_id, e)
            all_results.append({
                "paper_id": paper_id,
                "status": "skipped",
                "error": str(e),
                "checks": [],
            })
            continue

        # Truncate aggressively — we only need enough context for the LLM to
        # "read" the planted value; full paper is too many tokens for S0 quota.
        original_len = len(paper_text)
        if len(paper_text) > MAX_PAPER_CHARS:
            paper_text = paper_text[:MAX_PAPER_CHARS]
        log.info(
            "%s: text %d → %d chars (~%d tokens after truncation)",
            paper_id, original_len, len(paper_text), len(paper_text) // 4,
        )

        paper_checks = []

        # Save original text artifact
        paper_artifact_dir = ARTIFACTS_DIR / paper_id
        paper_artifact_dir.mkdir(parents=True, exist_ok=True)
        (paper_artifact_dir / "original.txt").write_text(paper_text, encoding="utf-8")

        for plant_type in PLANT_TYPES:
            log.info("\n  Plant type: %s", plant_type)

            # Plant error
            modified_text, plant_record = plant_error(paper_text, plant_type, seed=42)

            if not plant_record["planted"]:
                log.warning("    Could not plant error (no matching target found)")
                paper_checks.append({
                    "plant_type": plant_type,
                    "planted": False,
                    "propagated": False,
                    "details": "Could not find target to modify",
                })
                continue

            log.info("    Original: %s", plant_record['original'])
            log.info("    Planted:  %s", plant_record['replacement'])

            # Save modified text artifact
            (paper_artifact_dir / f"{plant_type}_modified.txt").write_text(
                modified_text, encoding="utf-8"
            )

            # Run extraction on modified text
            log.info("    Running LLM extraction...")
            try:
                llm_output = run_extraction_with_planted_error(modified_text)
            except Exception as e:
                log.error("    LLM call failed (%s): %s", type(e).__name__, e)
                paper_checks.append({
                    "plant_type": plant_type,
                    "planted": True,
                    "propagated": False,
                    "details": f"LLM call failed: {e}",
                })
                continue

            # Save LLM output artifact
            with open(paper_artifact_dir / f"{plant_type}_llm_output.json", "w") as f:
                json.dump(llm_output, f, indent=2)

            if llm_output.get("_parse_error"):
                log.warning("    LLM response could not be parsed as JSON")
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

            # Save diff artifact
            diff_record = {
                "plant_record": plant_record,
                "detection": detection,
            }
            with open(paper_artifact_dir / f"{plant_type}_diff.json", "w") as f:
                json.dump(diff_record, f, indent=2, default=str)

            status = "PROPAGATED" if detection["propagated"] else "NOT propagated"
            log.info("    Result: %s", status)
            log.info("    Details: %s", detection['details'])

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

    log.info("Papers to check: %s", ', '.join(paper_ids))
    log.info("Plant types: %s", ', '.join(PLANT_TYPES))
    log.info("Max paper chars per call: %d (~%d tokens)", MAX_PAPER_CHARS, MAX_PAPER_CHARS // 4)
    log.info("Max output tokens: %d", MAX_OUTPUT_TOKENS)
    log.info("Log file: %s", LOG_PATH)

    results = run_hallucination_check(paper_ids)

    print_summary(results)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "hallucination_results.json"
    with open(output_path, "w") as f:
        json.dump({"results": results, "plant_types": PLANT_TYPES}, f, indent=2)
    log.info("Results saved to %s", output_path)
    log.info("Full debug log: %s", LOG_PATH)


if __name__ == "__main__":
    main()
