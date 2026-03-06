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

    # Load predictions and reference
    predictions_path = output_dir / 'predictions.csv'
    reference_path = reference_dir / 'reference.csv'

    if not predictions_path.exists():
        sys.exit(f"Error: Predictions file not found at {predictions_path}")
    if not reference_path.exists():
        sys.exit(f"Error: Reference file not found at {reference_path}")

    predictions = pd.read_csv(predictions_path)
    reference = pd.read_csv(reference_path)

    # Validate columns
    required_columns = ['id', 'pred']
    for df, name in [(predictions, "Predictions"), (reference, "Reference")]:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            sys.exit(f"Error: {name} file is missing required columns: {missing_columns}")

    # Merge on 'id'
    merged = pd.merge(reference, predictions, on=['id'], suffixes=['_true', '_pred'])

    # Extract prediction columns as numpy arrays
    y_true = merged['pred_true'].values
    y_pred = merged['pred_pred'].values

    # Compute metrics
    scores = compute_metrics(y_true, y_pred, task_type="other")

    # Write scores to scores.json
    scores_path = scores_dir / 'scores.json'
    scores_dir.mkdir(parents=True, exist_ok=True)
    with scores_path.open('w') as f:
        json.dump(scores, f)

if __name__ == "__main__":
    main()