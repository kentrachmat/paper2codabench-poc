#!/usr/bin/env python3
"""
Verify all generated bundles by running local simulation.

Scans bundles/ for all paper directories, runs local_run.py with solution.py,
captures exit code and scores, and reports pass/fail.

Outputs a markdown table to stdout and saves JSON to experiments/results/verification_results.json.
"""
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLES_DIR = PROJECT_ROOT / "bundles"
METADATA_PATH = PROJECT_ROOT / "papers" / "metadata.json"
LOCAL_RUN = PROJECT_ROOT / "src" / "local_run.py"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_metadata():
    """Load paper metadata for titles and task types."""
    with open(METADATA_PATH, "r") as f:
        data = json.load(f)
    return {p["paper_id"]: p for p in data["competitions"]}


def run_bundle(bundle_path):
    """Run local simulation for a bundle and return results."""
    submission = bundle_path / "examples" / "solution.py"
    if not submission.exists():
        return {"status": "SKIP", "error": "No solution.py found", "scores": {}}

    try:
        result = subprocess.run(
            [sys.executable, str(LOCAL_RUN), str(bundle_path), str(submission)],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(PROJECT_ROOT),
        )

        stdout = result.stdout
        stderr = result.stderr
        output = stdout + stderr

        if result.returncode == 0:
            # Try to parse scores from output
            scores = {}
            in_scores = False
            for line in stdout.split("\n"):
                line = line.strip()
                if "Scores:" in line:
                    in_scores = True
                    continue
                if in_scores and ":" in line and not line.startswith("="):
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        try:
                            val = float(parts[1].strip())
                            scores[key] = val
                        except ValueError:
                            pass
                elif in_scores and line.startswith("="):
                    break

            return {"status": "PASS", "scores": scores, "error": None}
        else:
            # Extract last meaningful error lines
            error_lines = [l for l in output.strip().split("\n") if l.strip()][-5:]
            return {"status": "FAIL", "scores": {}, "error": "\n".join(error_lines)}

    except subprocess.TimeoutExpired:
        return {"status": "FAIL", "scores": {}, "error": "Timeout (10 min)"}
    except Exception as e:
        return {"status": "FAIL", "scores": {}, "error": str(e)}


def main():
    metadata = load_metadata()

    # Find all bundle directories
    bundle_dirs = sorted(
        [d for d in BUNDLES_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")],
        key=lambda d: d.name,
    )

    if not bundle_dirs:
        print("No bundles found in", BUNDLES_DIR)
        sys.exit(1)

    results = []
    pass_count = 0
    fail_count = 0

    for bundle_dir in bundle_dirs:
        paper_id = bundle_dir.name
        meta = metadata.get(paper_id, {})
        print(f"Testing {paper_id}...", end=" ", flush=True)

        result = run_bundle(bundle_dir)
        result["paper_id"] = paper_id
        result["title"] = meta.get("title", "Unknown")
        result["task_type"] = meta.get("task_type", "Unknown")
        results.append(result)

        if result["status"] == "PASS":
            pass_count += 1
            scores_str = ", ".join(f"{k}={v:.4f}" for k, v in result["scores"].items())
            print(f"PASS  [{scores_str}]")
        else:
            fail_count += 1
            print(f"{result['status']}  [{result.get('error', '')[:80]}]")

    # Print markdown summary table
    print()
    print("## Bundle Verification Summary")
    print()
    print(f"| Paper | Title | Task Type | Status | Scores |")
    print(f"|-------|-------|-----------|:------:|--------|")
    for r in results:
        short_title = r["title"][:50] + ("..." if len(r["title"]) > 50 else "")
        status_icon = "PASS" if r["status"] == "PASS" else "FAIL"
        if r["scores"]:
            scores_str = ", ".join(f"{k}={v:.4f}" for k, v in r["scores"].items())
        else:
            scores_str = r.get("error", "")[:60]
        print(f"| {r['paper_id']} | {short_title} | {r['task_type']} | {status_icon} | {scores_str} |")
    print()
    print(f"**Results:** {pass_count} passed, {fail_count} failed out of {len(results)} bundles")
    print()

    # Save JSON results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "verification_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "results": results,
            "total": len(results),
            "passed": pass_count,
            "failed": fail_count,
        }, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
