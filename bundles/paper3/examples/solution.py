import pandas as pd
import numpy as np
from pathlib import Path

def predict(input_dir, output_dir):
    """Generate predictions for the competition."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read input data (columns: ['mu_16'])
    input_df = pd.read_csv(input_dir / 'input.csv')

    # Build predictions dataframe with all required columns
    predictions = input_df.copy()
    predictions['mu_84'] = predictions['mu_16'] + np.random.uniform(0.1, 0.5, size=len(predictions))  # Simple baseline

    # Save predictions with exact columns: ['mu_16', 'mu_84']
    predictions[['mu_16', 'mu_84']].to_csv(output_dir / 'predictions.csv', index=False)

if __name__ == "__main__":
    predict("input_data", "output")