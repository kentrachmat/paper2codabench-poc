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

        predictions_path = output_dir / 'predictions.csv'
        reference_path = reference_dir / 'reference.csv'
        scores_path = scores_dir / 'scores.json'

        # Load predictions and reference
        if not predictions_path.exists():
            raise FileNotFoundError(f"Predictions file not found at {predictions_path}")
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference file not found at {reference_path}")

        predictions = pd.read_csv(predictions_path)
        reference = pd.read_csv(reference_path)

        # Validate columns
        required_columns = ['image_id', 'attacked_image']
        for df, name in [(predictions, "Predictions"), (reference, "Reference")]:
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"{name} file must contain the following columns: {required_columns}")

        # Merge on 'image_id'
        merged = pd.merge(reference, predictions, on=['image_id'], suffixes=['_true', '_pred'])

        # Extract true and predicted values
        y_true = merged['attacked_image_true'].values
        y_pred = merged['attacked_image_pred'].values

        # Compute metrics
        scores = compute_metrics(y_true, y_pred, task_type="generation")

        # Write scores to scores.json
        scores_dir.mkdir(parents=True, exist_ok=True)
        with open(scores_path, 'w') as f:
            json.dump(scores, f)

    except Exception as e:
        # Handle exceptions and print error messages
        sys.stderr.write(str(e) + '\n')
        sys.exit(1)

if __name__ == "__main__":
    main()