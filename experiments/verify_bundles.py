#!/usr/bin/env python3
"""
Verify all generated bundles by running local simulation.

Scans both bundles/ (Pipeline A) and bundles_pipelineB/ (Pipeline B),
runs local_run.py with solution.py, captures exit code and scores.

Outputs a comparison markdown table and saves JSON to experiments/results/verification_results.json.
"""
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLES_DIR_A = PROJECT_ROOT / "bundles"
BUNDLES_DIR_B = PROJECT_ROOT / "bundles_pipelineB"
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


def scan_bundles_dir(directory: Path, metadata: dict) -> list:
    """
    Scan a bundles directory and verify each bundle.

    Args:
        directory: Path to bundles directory
        metadata: Paper metadata dict keyed by paper_id

    Returns:
        List of result dicts per bundle
    """
    if not directory.exists():
        return []

    bundle_dirs = sorted(
        [d for d in directory.iterdir() if d.is_dir() and not d.name.startswith(".")],
        key=lambda d: d.name,
    )

    results = []
    for bundle_dir in bundle_dirs:
        paper_id = bundle_dir.name
        meta = metadata.get(paper_id, {})
        print(f"  Testing {paper_id} ({directory.name})...", end=" ", flush=True)

        result = run_bundle(bundle_dir)
        result["paper_id"] = paper_id
        result["title"] = meta.get("title", "Unknown")
        result["task_type"] = meta.get("task_type", "Unknown")
        results.append(result)

        if result["status"] == "PASS":
            scores_str = ", ".join(f"{k}={v:.4f}" for k, v in result["scores"].items())
            print(f"PASS  [{scores_str}]")
        else:
            print(f"{result['status']}  [{result.get('error', '')[:80]}]")

    return results


def main():
    metadata = load_metadata()

    # Scan Pipeline A
    print("\n=== Pipeline A (bundles/) ===")
    results_a = scan_bundles_dir(BUNDLES_DIR_A, metadata)

    # Scan Pipeline B
    print("\n=== Pipeline B (bundles_pipelineB/) ===")
    results_b = scan_bundles_dir(BUNDLES_DIR_B, metadata)

    if not results_a and not results_b:
        print("No bundles found in either directory.")
        sys.exit(1)

    # Build lookup
    a_by_id = {r["paper_id"]: r for r in results_a}
    b_by_id = {r["paper_id"]: r for r in results_b}
    all_paper_ids = sorted(set(list(a_by_id.keys()) + list(b_by_id.keys())))

    # Print comparison table
    print()
    print("## Bundle Verification Comparison: Pipeline A vs Pipeline B")
    print()
    print(f"| Paper | Title | A_status | A_scores | B_status | B_scores |")
    print(f"|-------|-------|:--------:|----------|:--------:|----------|")
    for pid in all_paper_ids:
        a = a_by_id.get(pid, {})
        b = b_by_id.get(pid, {})
        title = a.get("title", b.get("title", "Unknown"))
        short_title = title[:40] + ("..." if len(title) > 40 else "")

        a_status = a.get("status", "-")
        b_status = b.get("status", "-")

        if a.get("scores"):
            a_scores = ", ".join(f"{k}={v:.4f}" for k, v in a["scores"].items())
        else:
            a_scores = a.get("error", "-") if a else "-"
            a_scores = a_scores[:50] if isinstance(a_scores, str) else "-"

        if b.get("scores"):
            b_scores = ", ".join(f"{k}={v:.4f}" for k, v in b["scores"].items())
        else:
            b_scores = b.get("error", "-") if b else "-"
            b_scores = b_scores[:50] if isinstance(b_scores, str) else "-"

        print(f"| {pid} | {short_title} | {a_status} | {a_scores} | {b_status} | {b_scores} |")
    print()

    pass_a = sum(1 for r in results_a if r["status"] == "PASS")
    fail_a = sum(1 for r in results_a if r["status"] != "PASS")
    pass_b = sum(1 for r in results_b if r["status"] == "PASS")
    fail_b = sum(1 for r in results_b if r["status"] != "PASS")

    print(f"**Pipeline A:** {pass_a} passed, {fail_a} failed out of {len(results_a)} bundles")
    print(f"**Pipeline B:** {pass_b} passed, {fail_b} failed out of {len(results_b)} bundles")
    print()

    # Save JSON results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "verification_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "pipeline_a": {
                "results": results_a,
                "total": len(results_a),
                "passed": pass_a,
                "failed": fail_a,
            },
            "pipeline_b": {
                "results": results_b,
                "total": len(results_b),
                "passed": pass_b,
                "failed": fail_b,
            },
        }, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
