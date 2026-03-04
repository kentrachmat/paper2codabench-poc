import pandas as pd
import numpy as np
from pathlib import Path

def predict(input_dir, output_dir):
    """Generate predictions for the competition."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read input data (columns: ['image_id'])
    input_df = pd.read_csv(input_dir / 'input.csv')

    # Build predictions dataframe with all required columns
    predictions = input_df.copy()
    predictions['attacked_image'] = "dummy_prediction"  # Simple baseline

    # Save predictions with exact columns: ['image_id', 'attacked_image']
    predictions[['image_id', 'attacked_image']].to_csv(output_dir / 'predictions.csv', index=False)

if __name__ == "__main__":
    predict("input_data", "output")