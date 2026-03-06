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

    # Validate file existence
    if not predictions_file.exists():
        sys.exit(f"Error: Predictions file not found at {predictions_file}")
    if not reference_file.exists():
        sys.exit(f"Error: Reference file not found at {reference_file}")

    # Load predictions and reference
    predictions = pd.read_csv(predictions_file)
    reference = pd.read_csv(reference_file)

    # Validate required columns
    required_columns = ['id', 'pred']
    for file_name, df in [('Predictions', predictions), ('Reference', reference)]:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            sys.exit(f"Error: {file_name} file is missing required columns: {missing_columns}")

    # Merge on 'id'
    try:
        merged = pd.merge(reference, predictions, on=['id'], suffixes=['_true', '_pred'])
    except KeyError as e:
        sys.exit(f"Error during merge: {e}")

    # Extract prediction columns
    try:
        y_true = merged['pred_true'].values
        y_pred = merged['pred_pred'].values
    except KeyError as e:
        sys.exit(f"Error extracting prediction columns: {e}")

    # Compute metrics
    try:
        scores = compute_metrics(y_true, y_pred, task_type="classification")
    except Exception as e:
        sys.exit(f"Error computing metrics: {e}")

    # Write scores to JSON
    try:
        scores_dir.mkdir(parents=True, exist_ok=True)
        with scores_file.open('w') as f:
            json.dump(scores, f)
    except Exception as e:
        sys.exit(f"Error writing scores file: {e}")

if __name__ == "__main__":
    main()