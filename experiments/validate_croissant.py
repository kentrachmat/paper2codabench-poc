#!/usr/bin/env python3
"""
Validate generated Croissant Task files against the Croissant schema.

Checks:
  - Required JSON-LD fields: @context, @type, name, description
  - Croissant Task structure: cr:input, cr:output, cr:evaluation
  - Optionally fetches real Croissant files from HuggingFace for comparison

Usage:
    python experiments/validate_croissant.py
"""
import csv
import json
import sys
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

from config import Config

RESULTS_DIR = BASE / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Required fields in a Croissant Task
REQUIRED_FIELDS = ["@context", "@type", "name", "description"]
RECOMMENDED_FIELDS = ["cr:input", "cr:output", "cr:evaluation", "cr:execution"]


def validate_croissant_file(path: Path) -> dict:
    """Validate a single Croissant Task JSON file."""
    result = {
        "file": str(path.name),
        "valid": True,
        "errors": [],
        "warnings": [],
    }

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        result["valid"] = False
        result["errors"].append(f"Invalid JSON: {e}")
        return result

    if not isinstance(data, dict):
        result["valid"] = False
        result["errors"].append("Root is not a JSON object")
        return result

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in data:
            result["errors"].append(f"Missing required field: {field}")
            result["valid"] = False

    # Check @type
    at_type = data.get("@type", "")
    if at_type and "TaskProblem" not in at_type:
        result["warnings"].append(f"@type is '{at_type}', expected 'cr:TaskProblem'")

    # Check recommended fields
    for field in RECOMMENDED_FIELDS:
        if field not in data:
            result["warnings"].append(f"Missing recommended field: {field}")

    # Check evaluation structure
    evaluation = data.get("cr:evaluation", {})
    if evaluation and "primaryMetric" not in evaluation:
        result["warnings"].append("cr:evaluation missing primaryMetric")

    # Check output schema
    output = data.get("cr:output", {})
    if output:
        schema = output.get("cr:schema", {})
        fields = schema.get("field", [])
        if not fields:
            result["warnings"].append("cr:output has no schema fields")

    return result


def fetch_huggingface_croissant(url: str, timeout: int = 10) -> dict | None:
    """Fetch a Croissant JSON-LD file from a URL."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  Failed to fetch {url}: {e}")
        return None


def compare_with_huggingface():
    """Fetch a few real Croissant files from HuggingFace and compare structure."""
    comparisons = []

    # Find croissant URLs from the data CSV
    data_csv = BASE / "data" / "neurips_2025_db_papers.csv"
    if not data_csv.exists():
        print("  data/neurips_2025_db_papers.csv not found, skipping HuggingFace comparison")
        return comparisons

    croissant_urls = []
    with open(data_csv) as f:
        for row in csv.DictReader(f):
            url = row.get("croissant_url", "").strip()
            if url and "huggingface" in url.lower():
                croissant_urls.append(url)

    # Fetch up to 3 real Croissant files
    for url in croissant_urls[:3]:
        print(f"  Fetching: {url[:80]}...")
        real_data = fetch_huggingface_croissant(url)
        if real_data is None:
            continue

        real_keys = set(real_data.keys())
        comparison = {
            "url": url[:120],
            "has_context": "@context" in real_data,
            "has_type": "@type" in real_data,
            "type_value": real_data.get("@type", "N/A"),
            "top_level_keys": sorted(real_keys)[:20],
            "total_keys": len(real_keys),
        }
        comparisons.append(comparison)

    return comparisons


def main():
    print(f"\n{'='*60}")
    print("Croissant Task Validation")
    print(f"{'='*60}\n")

    # Validate all generated Croissant Task files
    results = {"pipeline_a": [], "pipeline_b": []}

    for label, directory in [
        ("pipeline_a", Config.CROISSANT_DIR),
        ("pipeline_b", Config.CROISSANT_PIPELINE_B_DIR),
    ]:
        if not directory.exists():
            print(f"  {label}: directory not found ({directory})")
            continue

        files = sorted(directory.glob("*.croissant_task.json"))
        if not files:
            print(f"  {label}: no Croissant Task files found")
            continue

        print(f"\n  {label}: {len(files)} files")
        for path in files:
            validation = validate_croissant_file(path)
            results[label].append(validation)
            status = "VALID" if validation["valid"] else "INVALID"
            print(f"    {path.name}: {status}")
            for err in validation["errors"]:
                print(f"      ERROR: {err}")
            for warn in validation["warnings"][:3]:
                print(f"      WARN: {warn}")

    # Also check adapted croissant files from Pipeline B bundles
    adapted_files = sorted(Config.BUNDLES_PIPELINE_B_DIR.glob("*.adapted_croissant.json"))
    if adapted_files:
        print(f"\n  Pipeline B adapted files: {len(adapted_files)}")
        for path in adapted_files:
            validation = validate_croissant_file(path)
            results["pipeline_b"].append(validation)

    # Compare with HuggingFace
    print("\n  Fetching real Croissant files from HuggingFace...")
    hf_comparisons = compare_with_huggingface()
    results["huggingface_comparisons"] = hf_comparisons

    # Summary
    for label in ["pipeline_a", "pipeline_b"]:
        items = results[label]
        if items:
            valid_count = sum(1 for r in items if r["valid"])
            print(f"\n  {label}: {valid_count}/{len(items)} valid")

    # Save results
    output_path = RESULTS_DIR / "croissant_validation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
