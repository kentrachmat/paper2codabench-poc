import json
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
        raise FileNotFoundError(f"Predictions file not found at {predictions_file}")
    if not reference_file.exists():
        raise FileNotFoundError(f"Reference file not found at {reference_file}")

    # Load predictions and reference
    predictions = pd.read_csv(predictions_file)
    reference = pd.read_csv(reference_file)

    # Validate required columns
    required_columns = ['id', 'action_description']
    for col in required_columns:
        if col not in predictions.columns:
            raise ValueError(f"Missing required column '{col}' in predictions file.")
        if col not in reference.columns:
            raise ValueError(f"Missing required column '{col}' in reference file.")

    # Merge on 'id'
    merged = pd.merge(reference, predictions, on=['id'], suffixes=['_true', '_pred'])

    # Extract true and predicted values
    y_true = merged['action_description_true'].values
    y_pred = merged['action_description_pred'].values

    # Compute metrics
    scores = compute_metrics(y_true, y_pred, task_type="other")

    # Write scores to JSON
    scores_dir.mkdir(exist_ok=True, parents=True)
    with open(scores_file, 'w') as f:
        json.dump(scores, f)

if __name__ == "__main__":
    main()