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

        predictions_file = output_dir / 'predictions.csv'
        reference_file = reference_dir / 'reference.csv'
        scores_file = scores_dir / 'scores.json'

        # Load predictions and reference
        if not predictions_file.exists():
            raise FileNotFoundError(f"Predictions file not found at {predictions_file}")
        if not reference_file.exists():
            raise FileNotFoundError(f"Reference file not found at {reference_file}")

        predictions = pd.read_csv(predictions_file)
        reference = pd.read_csv(reference_file)

        # Validate required columns
        required_columns = ['id', 'protein', 'pred']
        for df, name in [(predictions, "Predictions"), (reference, "Reference")]:
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"{name} file must contain the following columns: {required_columns}")

        # Merge on ['id', 'protein']
        merged = pd.merge(reference, predictions, on=['id', 'protein'], suffixes=['_true', '_pred'])

        # Extract prediction columns as numpy arrays
        y_true = merged['pred_true'].values
        y_pred = merged['pred_pred'].values

        # Compute metrics
        scores = compute_metrics(y_true, y_pred, task_type="classification")

        # Write scores to scores.json
        scores_dir.mkdir(parents=True, exist_ok=True)
        with open(scores_file, 'w') as f:
            json.dump(scores, f)

    except Exception as e:
        # Handle exceptions and print error message
        sys.stderr.write(str(e) + '\n')
        sys.exit(1)

if __name__ == "__main__":
    main()