import sys
import shutil
import importlib.util
from pathlib import Path

def main():
    # Set up paths (Codabench or local)
    submission_dir = Path('/app/ingested_program')
    input_dir = Path('/app/input_data')
    output_dir = Path('/app/output')

    if not submission_dir.exists():
        submission_dir = Path('submission')
        input_dir = Path('input_data')
        output_dir = Path('output')

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find solution.py
    solution_file = submission_dir / 'solution.py'
    if not solution_file.exists():
        py_files = list(submission_dir.glob('*.py'))
        if py_files:
            solution_file = py_files[0]
        else:
            # CSV fallback: no .py file found, check for .csv
            csv_files = list(submission_dir.glob('*.csv'))
            if csv_files:
                print(f"CSV submission detected: {csv_files[0].name}")
                shutil.copy(csv_files[0], output_dir / 'predictions.csv')
                print("Copied CSV to output/predictions.csv")
                return
            print("ERROR: No solution.py or CSV file found")
            sys.exit(1)

    # Import and run solution
    spec = importlib.util.spec_from_file_location('solution', str(solution_file))
    solution = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solution)

    if not hasattr(solution, 'predict'):
        print("ERROR: solution.py does not define a 'predict' function")
        sys.exit(1)

    try:
        solution.predict(str(input_dir), str(output_dir))
    except Exception as e:
        print(f"ERROR: Exception occurred while running predict: {e}")
        sys.exit(1)

    # Verify output was created
    predictions_file = output_dir / 'predictions.csv'
    if not predictions_file.exists():
        print("ERROR: Solution did not create predictions.csv")
        sys.exit(1)

if __name__ == "__main__":
    main()