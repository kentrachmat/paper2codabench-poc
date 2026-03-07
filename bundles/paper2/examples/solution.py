import pandas as pd
import numpy as np
from pathlib import Path

def predict(input_dir, output_dir):
    """Generate predictions for the competition."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read input data (columns: ['extracted_information', 'attack_success_rate', 'defense_effectiveness'])
    input_df = pd.read_csv(input_dir / 'input.csv')

    # Build predictions dataframe with all required columns
    predictions = input_df.copy()
    predictions['model_utility'] = np.random.uniform(0, 1, len(input_df))  # Simple random baseline

    # Save predictions with exact columns: ['extracted_information', 'attack_success_rate', 'defense_effectiveness', 'model_utility']
    predictions[['extracted_information', 'attack_success_rate', 'defense_effectiveness', 'model_utility']].to_csv(output_dir / 'predictions.csv', index=False)

if __name__ == "__main__":
    predict("input_data", "output")