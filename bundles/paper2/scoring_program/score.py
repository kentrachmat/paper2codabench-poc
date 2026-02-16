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

        # Load predictions and reference
        predictions_path = output_dir / 'predictions.csv'
        reference_path = reference_dir / 'reference.csv'

        if not predictions_path.exists():
            raise FileNotFoundError(f"Predictions file not found at {predictions_path}")
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference file not found at {reference_path}")

        predictions = pd.read_csv(predictions_path)
        reference = pd.read_csv(reference_path)

        # Validate columns
        required_columns = ['id', 'mu16', 'mu84']
        for df, name in [(predictions, "Predictions"), (reference, "Reference")]:
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"{name} file must contain all required columns: {required_columns}")

        # Merge on ['id', 'mu16']
        merged = pd.merge(reference, predictions, on=['id', 'mu16'], suffixes=['_true', '_pred'])

        # Extract true and predicted values
        y_true = merged['mu84_true'].values
        y_pred = merged['mu84_pred'].values

        # Compute metrics
        scores = compute_metrics(y_true, y_pred, task_type="other")

        # Write scores to scores.json
        scores_dir.mkdir(exist_ok=True, parents=True)
        scores_path = scores_dir / 'scores.json'
        with open(scores_path, 'w') as f:
            json.dump(scores, f)

    except Exception as e:
        # Handle exceptions and print error message
        sys.stderr.write(f"Error: {str(e)}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()