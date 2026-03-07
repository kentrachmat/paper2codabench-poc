import pandas as pd
import numpy as np
from pathlib import Path

def predict(input_dir, output_dir):
    """Generate predictions for the competition."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read input data (columns: ['response_time', 'internalization', 'externalization', 'attention'])
    input_df = pd.read_csv(input_dir / 'input.csv')

    # Build predictions dataframe with all required columns
    predictions = input_df.copy()
    predictions['p_factor'] = np.random.rand(len(input_df))  # Random baseline for 'p_factor'

    # Save predictions with exact columns: ['response_time', 'internalization', 'externalization', 'attention', 'p_factor']
    predictions[['response_time', 'internalization', 'externalization', 'attention', 'p_factor']].to_csv(output_dir / 'predictions.csv', index=False)

if __name__ == "__main__":
    predict("input_data", "output")