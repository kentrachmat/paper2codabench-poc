import pandas as pd
import numpy as np
from pathlib import Path

def predict(input_dir, output_dir):
    """Generate predictions for the competition."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read input data (columns: ['id', 'pred', 'skin_tone', 'gender'])
    input_df = pd.read_csv(input_dir / 'input.csv')

    # Build predictions dataframe with all required columns
    predictions = input_df.copy()
    predictions['age'] = np.random.randint(18, 60, size=len(predictions))  # Simple random baseline for 'age'

    # Save predictions with exact columns: ['id', 'pred', 'skin_tone', 'gender', 'age']
    predictions[['id', 'pred', 'skin_tone', 'gender', 'age']].to_csv(output_dir / 'predictions.csv', index=False)

if __name__ == "__main__":
    predict("input_data", "output")