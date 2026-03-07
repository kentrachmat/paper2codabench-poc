import json
import sys
import pandas as pd
from pathlib import Path
from metrics import compute_metrics

def main():
    try:
        # Set up paths
        output_dir = Path('/app/output') if Path('/app/output').exists() else Path('output')
        reference_dir = Path('/app/reference_data') if Path('/app/reference_data').exists() else Path('reference_data')
        scores_dir = Path('/app/scores') if Path('/app/scores').exists() else Path('scores')

        predictions_path = output_dir / 'predictions.csv'
        reference_path = reference_dir / 'reference.csv'
        scores_path = scores_dir / 'scores.json'

        # Validate file existence
        if not predictions_path.exists():
            raise FileNotFoundError(f"Predictions file not found at {predictions_path}")
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference file not found at {reference_path}")

        # Load data
        predictions = pd.read_csv(predictions_path)
        reference = pd.read_csv(reference_path)

        # Validate columns
        required_columns = ['id', 'pred', 'skin_tone', 'gender', 'age']
        for col in required_columns:
            if col not in predictions.columns:
                raise ValueError(f"Missing column '{col}' in predictions file")
            if col not in reference.columns:
                raise ValueError(f"Missing column '{col}' in reference file")

        # Merge data
        merged = pd.merge(reference, predictions, on=['id', 'pred', 'skin_tone', 'gender'], suffixes=['_true', '_pred'])

        # Extract true and predicted values
        y_true = merged['age_true'].values
        y_pred = merged['age_pred'].values

        # Compute metrics
        scores = compute_metrics(y_true, y_pred, task_type="classification")

        # Write scores to JSON
        with scores_path.open('w') as f:
            json.dump(scores, f)

    except Exception as e:
        # Handle errors and exit gracefully
        sys.stderr.write(f"Error: {str(e)}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()