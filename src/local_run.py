#!/usr/bin/env python3
"""
Local simulation of Codabench evaluation pipeline.

Simulates:
    Submission → Ingestion → Scoring → Results + Seal

Bundle structure:
    - input_data/: Input data for participants to make predictions on
    - reference_data/: Ground truth labels (hidden from participants)
    - examples/: Sample submission files showing the correct format
    - ingestion_program/: Validates and processes submissions
    - scoring_program/: Computes metrics

Usage:
    python src/local_run.py bundles/paper1 bundles/paper1/examples/sample_submission.csv
    python src/local_run.py bundles/paper1 bundles/paper1/examples/sample_submission.csv --verbose
    python src/local_run.py bundles/paper1 my_custom_submission.csv
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from datetime import datetime

from seal import create_evaluation_seal


def validate_bundle(bundle_path: Path) -> bool:
    """Validate bundle structure"""
    print("🔍 Validating bundle structure...")

    required_dirs = [
        "ingestion_program",
        "scoring_program",
        "reference_data"
    ]

    required_files = [
        "ingestion_program/ingestion.py",
        "scoring_program/score.py",
        "scoring_program/metrics.py",
    ]

    missing = []

    for dir_name in required_dirs:
        if not (bundle_path / dir_name).exists():
            missing.append(f"directory: {dir_name}")

    for file_path in required_files:
        if not (bundle_path / file_path).exists():
            missing.append(f"file: {file_path}")

    if missing:
        print("✗ Bundle validation failed. Missing:")
        for item in missing:
            print(f"  - {item}")
        return False

    print("✓ Bundle structure is valid")
    return True


def run_program(program_path: Path, work_dir: Path, verbose: bool = False) -> tuple:
    """
    Run a Python program and capture output.

    Args:
        program_path: Path to Python script
        work_dir: Working directory
        verbose: Print all output

    Returns:
        (success, output) tuple
    """
    try:
        result = subprocess.run(
            [sys.executable, str(program_path)],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        output = result.stdout + result.stderr

        if verbose:
            print(output)

        return result.returncode == 0, output

    except subprocess.TimeoutExpired:
        return False, "Program execution timed out (5 minutes)"
    except Exception as e:
        return False, f"Execution error: {e}"


def run_local_simulation(bundle_path: Path, submission_path: Path, verbose: bool = False) -> dict:
    """
    Run complete local simulation.

    Args:
        bundle_path: Path to bundle directory
        submission_path: Path to submission file
        verbose: Print detailed output

    Returns:
        Scores dictionary
    """
    # Resolve to absolute paths to avoid issues when changing working directories
    bundle_path = bundle_path.resolve()
    submission_path = submission_path.resolve()

    print(f"\n{'='*60}")
    print(f"Local Simulation: {bundle_path.name}")
    print(f"{'='*60}\n")

    # Validate bundle
    if not validate_bundle(bundle_path):
        raise ValueError("Invalid bundle structure")

    # Validate submission
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission file not found: {submission_path}")

    print(f"✓ Submission file: {submission_path.name}")

    # Create working directories
    print("\n📁 Setting up working directories...")
    work_dir = Path(tempfile.mkdtemp(prefix="codabench_sim_"))

    try:
        # Create Codabench-like directory structure
        submission_dir = work_dir / "submission"
        input_dir = work_dir / "input_data"
        output_dir = work_dir / "output"
        scores_dir = work_dir / "scores"

        submission_dir.mkdir()
        input_dir.mkdir()
        output_dir.mkdir()
        scores_dir.mkdir()

        print(f"✓ Working directory: {work_dir}")

        # Copy submission to submission directory with expected filename
        # Ingestion scripts expect the file to be named "submission.csv" (from TaskSpec)
        expected_submission_filename = "submission.csv"
        shutil.copy(submission_path, submission_dir / expected_submission_filename)
        print(f"✓ Copied submission as: {expected_submission_filename}")

        # Copy input data
        input_data_dir = bundle_path / "input_data"
        if input_data_dir.exists():
            shutil.copytree(input_data_dir, input_dir, dirs_exist_ok=True)

        # Copy reference data to work_dir
        reference_dir = work_dir / "reference_data"
        shutil.copytree(bundle_path / "reference_data", reference_dir)

        # Copy scoring program
        scoring_dir = work_dir / "scoring_program"
        shutil.copytree(bundle_path / "scoring_program", scoring_dir)

        print("\n" + "="*60)
        print("STEP 1: Ingestion")
        print("="*60)

        # Run ingestion program
        ingestion_script = bundle_path / "ingestion_program" / "ingestion.py"

        print(f"Running: {ingestion_script.name}")
        success, output = run_program(ingestion_script, work_dir, verbose)

        if not success:
            print("✗ Ingestion failed!")
            print(output)
            raise RuntimeError("Ingestion failed")

        print("✅ Ingestion completed successfully")

        # Check if predictions were created
        predictions_file = output_dir / "predictions.csv"
        if not predictions_file.exists():
            raise FileNotFoundError("Ingestion did not create predictions.csv")

        print(f"✓ Predictions file created: {predictions_file.name}")

        print("\n" + "="*60)
        print("STEP 2: Scoring")
        print("="*60)

        # Run scoring program
        scoring_script = scoring_dir / "score.py"

        print(f"Running: {scoring_script.name}")
        success, output = run_program(scoring_script, work_dir, verbose)

        if not success:
            print("✗ Scoring failed!")
            print(output)
            raise RuntimeError("Scoring failed")

        print("✅ Scoring completed successfully")

        # Read scores
        scores_file = scores_dir / "scores.json"
        if not scores_file.exists():
            raise FileNotFoundError("Scoring did not create scores.json")

        with open(scores_file, 'r') as f:
            scores = json.load(f)

        print(f"✓ Scores file created: {scores_file.name}")

        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)

        for metric_name, value in scores.items():
            print(f"  {metric_name:20s}: {value:.6f}")

        print("="*60)

        # Create verification seal
        print("\n🔒 Creating verification seal...")

        submission_info = {
            'filename': submission_path.name,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
        }

        seal = create_evaluation_seal(bundle_path, scores, submission_info)
        print(f"✓ Seal created: {seal.get('seal_id', 'N/A')}")

        print(f"\n✅ Simulation completed successfully!")
        print(f"   Working directory: {work_dir}")
        print(f"   (Temporary files can be deleted)\n")

        return scores

    except Exception as e:
        print(f"\n✗ Simulation failed: {e}")
        raise

    finally:
        # Optionally clean up (keep for debugging in POC)
        if not verbose:
            try:
                shutil.rmtree(work_dir)
                print(f"🧹 Cleaned up working directory")
            except:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="Run local Codabench simulation"
    )
    parser.add_argument(
        "bundle_path",
        type=Path,
        help="Path to bundle directory"
    )
    parser.add_argument(
        "submission_path",
        type=Path,
        help="Path to submission file (e.g., submission.csv)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output from programs"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.bundle_path.exists():
        print(f"✗ Error: Bundle not found: {args.bundle_path}")
        sys.exit(1)

    if not args.submission_path.exists():
        print(f"✗ Error: Submission file not found: {args.submission_path}")
        sys.exit(1)

    # Run simulation
    try:
        scores = run_local_simulation(
            args.bundle_path,
            args.submission_path,
            args.verbose
        )

        print("\n" + "="*60)
        print("Summary:")
        print("="*60)
        print(f"Bundle: {args.bundle_path.name}")
        print(f"Submission: {args.submission_path.name}")
        print(f"\nScores:")
        for metric, value in scores.items():
            print(f"  {metric}: {value:.6f}")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n✗ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
