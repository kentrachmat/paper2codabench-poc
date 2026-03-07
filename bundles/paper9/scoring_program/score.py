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

        # Validate columns
        required_columns = ['mmlu_answer', 'mmlu_var_continuation', 'sciq_label']
        for col in required_columns:
            if col not in predictions.columns:
                raise ValueError(f"Missing column '{col}' in predictions file.")
            if col not in reference.columns:
                raise ValueError(f"Missing column '{col}' in reference file.")

        # Merge on the required keys
        merged = pd.merge(reference, predictions, on=['mmlu_answer', 'mmlu_var_continuation'], suffixes=['_true', '_pred'])

        # Extract true and predicted labels
        y_true = merged['sciq_label_true'].values
        y_pred = merged['sciq_label_pred'].values

        # Compute metrics
        scores = compute_metrics(y_true, y_pred, task_type="other")

        # Write scores to JSON
        scores_dir.mkdir(exist_ok=True, parents=True)
        with open(scores_file, 'w') as f:
            json.dump(scores, f)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()