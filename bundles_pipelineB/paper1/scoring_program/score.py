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

        # Check if files exist
        if not predictions_file.exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
        if not reference_file.exists():
            raise FileNotFoundError(f"Reference file not found: {reference_file}")

        # Load predictions and reference
        predictions = pd.read_csv(predictions_file)
        reference = pd.read_csv(reference_file)

        # Validate columns
        required_columns = ['id', 'ux', 'uy', 'p', 'νt', 'ps', 'CL', 'CD', 'ρD', 'ρL']
        for df, name in [(predictions, "Predictions"), (reference, "Reference")]:
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"{name} file is missing required columns. Expected columns: {required_columns}")

        # Merge on the specified keys
        merge_keys = ['id', 'ux', 'uy', 'p', 'νt', 'ps', 'CL', 'CD', 'ρD']
        merged = pd.merge(reference, predictions, on=merge_keys, suffixes=['_true', '_pred'])

        # Extract true and predicted values for the target column
        y_true = merged['ρL_true'].values
        y_pred = merged['ρL_pred'].values

        # Compute metrics
        scores = compute_metrics(y_true, y_pred, task_type="other")

        # Write scores to JSON
        scores_dir.mkdir(exist_ok=True, parents=True)
        with open(scores_file, 'w') as f:
            json.dump(scores, f)

    except Exception as e:
        # Handle exceptions and print error message
        sys.stderr.write(str(e) + "\n")
        sys.exit(1)

if __name__ == "__main__":
    main()