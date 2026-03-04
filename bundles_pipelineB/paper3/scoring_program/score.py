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
        required_columns = ['id', 'mu_16', 'mu_84', 'classification']
        for df, name in zip([predictions, reference], ['predictions', 'reference']):
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing columns in {name} file: {missing_columns}")

        # Merge data
        merged = pd.merge(reference, predictions, on=['id', 'mu_16', 'mu_84'], suffixes=['_true', '_pred'])

        # Extract true and predicted values
        y_true = merged['classification_true'].values
        y_pred = merged['classification_pred'].values

        # Compute metrics
        scores = compute_metrics(y_true, y_pred, task_type="classification")

        # Write scores to JSON
        scores_dir.mkdir(parents=True, exist_ok=True)
        with scores_path.open('w') as f:
            json.dump(scores, f)

    except Exception as e:
        # Handle errors and exit
        sys.stderr.write(f"Error: {str(e)}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()