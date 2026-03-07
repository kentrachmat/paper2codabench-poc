import pandas as pd
import numpy as np
from pathlib import Path

def predict(input_dir, output_dir):
    """Generate predictions for the competition."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read input data (columns: ['id', 'pred_response_time', 'pred_internalization', 'pred_externalization', 'pred_attention'])
    input_df = pd.read_csv(input_dir / 'input.csv')

    # Build predictions dataframe with all required columns
    predictions = input_df.copy()
    predictions['pred_p_factor'] = np.random.uniform(0, 1, size=len(input_df))  # Simple random baseline

    # Save predictions with exact columns: ['id', 'pred_response_time', 'pred_internalization', 'pred_externalization', 'pred_attention', 'pred_p_factor']
    predictions[['id', 'pred_response_time', 'pred_internalization', 'pred_externalization', 'pred_attention', 'pred_p_factor']].to_csv(output_dir / 'predictions.csv', index=False)

if __name__ == "__main__":
    predict("input_data", "output")