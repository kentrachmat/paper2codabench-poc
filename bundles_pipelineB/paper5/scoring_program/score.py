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

    predictions_file = output_dir / 'predictions.csv'
    reference_file = reference_dir / 'reference.csv'
    scores_file = scores_dir / 'scores.json'

    # Check if files exist
    if not predictions_file.exists():
        sys.stderr.write(f"Error: Predictions file not found at {predictions_file}\n")
        sys.exit(1)
    if not reference_file.exists():
        sys.stderr.write(f"Error: Reference file not found at {reference_file}\n")
        sys.exit(1)

    # Load predictions and reference
    try:
        predictions = pd.read_csv(predictions_file)
        reference = pd.read_csv(reference_file)
    except Exception as e:
        sys.stderr.write(f"Error reading CSV files: {e}\n")
        sys.exit(1)

    # Validate columns
    required_columns = ['id', 'pred']
    for df, name in [(predictions, "predictions"), (reference, "reference")]:
        if not all(col in df.columns for col in required_columns):
            sys.stderr.write(f"Error: {name} file must contain columns: {required_columns}\n")
            sys.exit(1)

    # Merge on 'id'
    try:
        merged = pd.merge(reference, predictions, on=['id'], suffixes=['_true', '_pred'])
    except Exception as e:
        sys.stderr.write(f"Error merging files: {e}\n")
        sys.exit(1)

    # Extract prediction columns
    try:
        y_true = merged['pred_true'].values
        y_pred = merged['pred_pred'].values
    except KeyError as e:
        sys.stderr.write(f"Error: Missing required columns after merge: {e}\n")
        sys.exit(1)

    # Compute metrics
    try:
        scores = compute_metrics(y_true, y_pred, task_type="generation")
    except Exception as e:
        sys.stderr.write(f"Error computing metrics: {e}\n")
        sys.exit(1)

    # Write scores to JSON
    try:
        scores_dir.mkdir(exist_ok=True, parents=True)
        with open(scores_file, 'w') as f:
            json.dump(scores, f)
    except Exception as e:
        sys.stderr.write(f"Error writing scores.json: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()