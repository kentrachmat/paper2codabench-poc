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
            raise FileNotFoundError(f"Predictions file not found at {predictions_file}")
        if not reference_file.exists():
            raise FileNotFoundError(f"Reference file not found at {reference_file}")

        # Load predictions and reference
        predictions = pd.read_csv(predictions_file)
        reference = pd.read_csv(reference_file)

        # Validate columns
        required_columns = ['u_x', 'u_y', 'p_s', 'nu_t', 'C_L', 'C_D']
        for df, name in [(predictions, "predictions"), (reference, "reference")]:
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Missing required columns in {name} file. Expected columns: {required_columns}")

        # Merge on the specified keys
        merge_keys = ['u_x', 'u_y', 'p_s', 'nu_t', 'C_L']
        merged = pd.merge(reference, predictions, on=merge_keys, suffixes=['_true', '_pred'])

        # Extract true and predicted values
        y_true = merged['C_D_true'].values
        y_pred = merged['C_D_pred'].values

        # Compute metrics
        scores = compute_metrics(y_true, y_pred, task_type="other")

        # Write scores to JSON
        scores_dir.mkdir(parents=True, exist_ok=True)
        with open(scores_file, 'w') as f:
            json.dump(scores, f)

    except Exception as e:
        # Handle exceptions and print error messages
        sys.stderr.write(str(e) + '\n')
        sys.exit(1)

if __name__ == "__main__":
    main()