import pandas as pd
import numpy as np
from pathlib import Path

def predict(input_dir, output_dir):
    """Generate predictions for the Airfoil Design Simulation Challenge."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read input data (columns: ['id', 'ux', 'uy', 'ps', 'nu_t', 'cd'])
    input_df = pd.read_csv(input_dir / 'input.csv')

    # Build predictions dataframe with all required columns
    predictions = input_df.copy()
    predictions['cl'] = np.random.uniform(0, 1, size=len(input_df))  # Simple random baseline

    # Save predictions with exact columns: ['id', 'ux', 'uy', 'ps', 'nu_t', 'cd', 'cl']
    predictions[['id', 'ux', 'uy', 'ps', 'nu_t', 'cd', 'cl']].to_csv(output_dir / 'predictions.csv', index=False)

if __name__ == "__main__":
    predict("input_data", "output")