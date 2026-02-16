import sys
import pandas as pd
from pathlib import Path

def main():
    # Set up paths
    submission_dir = Path('/app/ingested_program') if Path('/app').exists() else Path('submission')
    output_dir = Path('/app/output') if Path('/app').exists() else Path('output')
    submission_file = submission_dir / 'submission.csv'
    output_file = output_dir / 'predictions.csv'

    # Ensure submission file exists
    if not submission_file.exists():
        print(f"Error: Submission file not found at {submission_file}")
        sys.exit(1)

    # Load submission file
    try:
        submission = pd.read_csv(submission_file)
    except Exception as e:
        print(f"Error: Failed to read submission file. {e}")
        sys.exit(1)

    # Validate required columns
    required_columns = ['id', 'pred']
    if not all(col in submission.columns for col in required_columns):
        print(f"Error: Submission file must contain the columns {required_columns}")
        sys.exit(1)

    # Check for missing IDs
    if submission['id'].isnull().any():
        print("Error: Submission contains missing IDs.")
        sys.exit(1)

    # Check for duplicate IDs
    if submission['id'].duplicated().any():
        print("Error: Submission contains duplicate IDs.")
        sys.exit(1)

    # Validate predictions format
    if submission['pred'].isnull().any():
        print("Error: Submission contains NaN values in 'pred' column.")
        sys.exit(1)
    if not pd.api.types.is_numeric_dtype(submission['pred']):
        print("Error: 'pred' column must contain numeric values.")
        sys.exit(1)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write validated predictions to output file
    try:
        submission.to_csv(output_file, index=False)
        print(f"Validated predictions written to {output_file}")
    except Exception as e:
        print(f"Error: Failed to write predictions to output file. {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()