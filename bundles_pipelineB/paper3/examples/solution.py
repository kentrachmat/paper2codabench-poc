import pandas as pd
import numpy as np
from pathlib import Path

def predict(input_dir, output_dir):
    """Generate predictions for the competition."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read input data (columns: ['id', 'mu_16', 'mu_84'])
    input_df = pd.read_csv(input_dir / 'input.csv')

    # Build predictions dataframe with all required columns
    predictions = input_df.copy()
    
    # Simple baseline: classify as Signal (1) if mu_16 > mu_84, otherwise Background (0)
    predictions['classification'] = (predictions['mu_16'] > predictions['mu_84']).astype(int)

    # Save predictions with exact columns: ['id', 'mu_16', 'mu_84', 'classification']
    predictions[['id', 'mu_16', 'mu_84', 'classification']].to_csv(output_dir / 'predictions.csv', index=False)

if __name__ == "__main__":
    predict("input_data", "output")