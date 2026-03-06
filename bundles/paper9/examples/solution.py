import pandas as pd
import numpy as np
from pathlib import Path

def predict(input_dir, output_dir):
    """Generate predictions for the competition."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read input data (columns: ['mmlu_answer', 'mmlu_var_continuation'])
    input_df = pd.read_csv(input_dir / 'input.csv')

    # Build predictions dataframe with all required columns
    predictions = input_df.copy()
    predictions['sciq_label'] = np.random.choice([0, 1], size=len(predictions))  # Random baseline

    # Save predictions with exact columns: ['mmlu_answer', 'mmlu_var_continuation', 'sciq_label']
    predictions[['mmlu_answer', 'mmlu_var_continuation', 'sciq_label']].to_csv(output_dir / 'predictions.csv', index=False)

if __name__ == "__main__":
    predict("input_data", "output")