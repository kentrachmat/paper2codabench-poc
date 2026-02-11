import sys
import pandas as pd
from pathlib import Path

def main():
    # Set up paths
    submission_dir = Path('/app/ingested_program') if Path('/app').exists() else Path('submission')
    output_dir = Path('/app/output') if Path('/app').exists() else Path('output')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Find submission file
    submission_file = submission_dir / 'submission.csv'
    if not submission_file.exists():
        print(f"Error: Submission file not found at {submission_file}")
        sys.exit(1)
    
    # Load submission file
    try:
        df = pd.read_csv(submission_file)
    except Exception as e:
        print(f"Error: Failed to read submission file. {e}")
        sys.exit(1)
    
    # Validate required columns
    required_columns = ['id', 'protein', 'pred']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Submission file must contain the following columns: {required_columns}")
        sys.exit(1)
    
    # Check for missing IDs
    if df['id'].isnull().any():
        print("Error: Submission file contains missing IDs.")
        sys.exit(1)
    
    # Check for duplicate IDs
    if df['id'].duplicated().any():
        print("Error: Submission file contains duplicate IDs.")
        sys.exit(1)
    
    # Check predictions for valid format (no NaN, correct data types)
    if df['pred'].isnull().any():
        print("Error: Submission file contains NaN values in 'pred' column.")
        sys.exit(1)
    
    if not pd.api.types.is_numeric_dtype(df['pred']):
        print("Error: 'pred' column must contain numeric values.")
        sys.exit(1)
    
    # Write validated predictions to output
    output_file = output_dir / 'predictions.csv'
    try:
        df.to_csv(output_file, index=False)
        print(f"Submission successfully validated and written to {output_file}")
    except Exception as e:
        print(f"Error: Failed to write output file. {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()