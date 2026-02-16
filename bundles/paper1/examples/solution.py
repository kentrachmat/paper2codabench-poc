import pandas as pd
import numpy as np
from pathlib import Path

def predict(input_dir, output_dir):
    """Generate predictions for the competition."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read input data (columns: ['id', 'protein_name'])
    input_df = pd.read_csv(input_dir / 'input.csv')

    # Build predictions dataframe with all required columns
    predictions = input_df.copy()
    # Generate random predictions for demonstration (0, 1, or 2 for three targets)
    predictions['pred'] = np.random.choice([0, 1, 2], size=len(predictions))

    # Save predictions with exact columns: ['id', 'protein_name', 'pred']
    predictions[['id', 'protein_name', 'pred']].to_csv(output_dir / 'predictions.csv', index=False)

if __name__ == "__main__":
    predict("input_data", "output")