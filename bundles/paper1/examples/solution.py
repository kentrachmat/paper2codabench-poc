import pandas as pd
import numpy as np
from pathlib import Path

def predict(input_dir, output_dir):
    """Generate predictions for the competition."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read input data (columns: ['u_x', 'u_y', 'p_s', 'nu_t', 'C_L'])
    input_df = pd.read_csv(input_dir / 'input.csv')

    # Build predictions dataframe with all required columns
    predictions = input_df.copy()
    predictions['C_D'] = np.random.uniform(0, 1, size=len(input_df))  # Simple random baseline

    # Save predictions with exact columns: ['u_x', 'u_y', 'p_s', 'nu_t', 'C_L', 'C_D']
    predictions[['u_x', 'u_y', 'p_s', 'nu_t', 'C_L', 'C_D']].to_csv(output_dir / 'predictions.csv', index=False)

if __name__ == "__main__":
    predict("input_data", "output")