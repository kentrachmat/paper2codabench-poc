import numpy as np
from typing import Dict
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
    """
    Compute evaluation metrics for the competition.

    Parameters:
    - y_true (np.ndarray): Ground truth values (1D or 2D array).
    - y_pred (np.ndarray): Predicted values (1D or 2D array).
    - task_type (str): Task type, should match the competition specification.

    Returns:
    - Dict[str, float]: Dictionary with metric names as keys and float scores as values.
    """
    # Reshape 1D arrays to 2D for consistency
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Validation checks
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch: y_true and y_pred must have the same shape.")
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Empty input: y_true and y_pred must not be empty.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("NaN values detected: y_true and y_pred must not contain NaN values.")

    # Metric computations
    try:
        # Primary Metric: LM-based reward model score (using Pearson correlation as a proxy)
        lm_reward_score = pearsonr(y_true.flatten(), y_pred.flatten())[0]

        # Secondary Metric: Social Welfare (mean squared error as a proxy for welfare alignment)
        social_welfare = mean_squared_error(y_true, y_pred)

        # Tertiary Metric: Qualitative Analysis (placeholder, set to 0.0 as it's qualitative)
        qualitative_analysis = 0.0

    except Exception as e:
        raise RuntimeError(f"Error during metric computation: {e}")

    # Return metrics dictionary with primary metric first
    return {
        "LM-based reward model score": lm_reward_score,
        "Social Welfare": social_welfare,
        "Qualitative Analysis": qualitative_analysis
    }