import json
import sys
import pandas as pd
from pathlib import Path
from metrics import compute_metrics

def main():
    # Set up paths
    output_dir = Path('/app/output') if Path('/app/output').exists() else Path('output')
    reference_dir = Path('/app/reference_data') if Path('/app/reference_data').exists() else Path('reference_data')
    scores_dir = Path('/app/scores') if Path('/app/scores').exists() else Path('scores')

    # File paths
    predictions_file = output_dir / 'predictions.csv'
    reference_file = reference_dir / 'reference.csv'
    scores_file = scores_dir / 'scores.json'

    # Validate file existence
    if not predictions_file.exists():
        sys.stderr.write(f"Error: Predictions file not found at {predictions_file}\n")
        sys.exit(1)
    if not reference_file.exists():
        sys.stderr.write(f"Error: Reference file not found at {reference_file}\n")
        sys.exit(1)

    # Load data
    try:
        predictions = pd.read_csv(predictions_file)
        reference = pd.read_csv(reference_file)
    except Exception as e:
        sys.stderr.write(f"Error loading files: {e}\n")
        sys.exit(1)

    # Validate columns
    required_columns = ['id', 'pred']
    for df, name in [(predictions, "Predictions"), (reference, "Reference")]:
        if not all(col in df.columns for col in required_columns):
            sys.stderr.write(f"Error: {name} file must contain columns {required_columns}\n")
            sys.exit(1)

    # Merge data
    try:
        merged = pd.merge(reference, predictions, on=['id'], suffixes=['_true', '_pred'])
    except Exception as e:
        sys.stderr.write(f"Error merging files: {e}\n")
        sys.exit(1)

    # Extract true and predicted values
    try:
        y_true = merged['pred_true'].values
        y_pred = merged['pred_pred'].values
    except KeyError as e:
        sys.stderr.write(f"Error extracting prediction columns: {e}\n")
        sys.exit(1)

    # Compute metrics
    try:
        scores = compute_metrics(y_true, y_pred, task_type="other")
    except Exception as e:
        sys.stderr.write(f"Error computing metrics: {e}\n")
        sys.exit(1)

    # Write scores to JSON
    try:
        scores_dir.mkdir(parents=True, exist_ok=True)
        with scores_file.open('w') as f:
            json.dump(scores, f)
    except Exception as e:
        sys.stderr.write(f"Error writing scores file: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()