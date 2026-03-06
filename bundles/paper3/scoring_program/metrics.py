import numpy as np
from typing import Dict
from scipy.stats import norm

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
    """
    Compute evaluation metrics for the competition.

    Metrics:
    - Final Quantile Score (Primary Metric)
    - Average Interval Width (w)
    - Coverage (c)

    Args:
        y_true (np.ndarray): Ground truth values (1D or 2D array).
        y_pred (np.ndarray): Predicted confidence intervals (2D array with columns: mu_16, mu_84).
        task_type (str): Task type (should be 'other' for this competition).

    Returns:
        Dict[str, float]: Dictionary containing metric names and their computed values.
    """
    # Reshape 1D arrays to 2D
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Validate input shapes
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same number of samples.")
    if y_pred.shape[1] != 2:
        raise ValueError("y_pred must have exactly two columns: mu_16 and mu_84.")

    # Handle edge cases
    if y_true.size == 0 or y_pred.size == 0:
        return {"Final Quantile Score": float('-inf'), "Average Interval Width": float('nan'), "Coverage": float('nan')}
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        return {"Final Quantile Score": float('-inf'), "Average Interval Width": float('nan'), "Coverage": float('nan')}

    # Extract confidence interval bounds
    mu_16 = y_pred[:, 0]
    mu_84 = y_pred[:, 1]

    # Compute Average Interval Width (w)
    interval_widths = mu_84 - mu_16
    average_interval_width = np.mean(interval_widths)

    # Compute Coverage (c)
    coverage = np.mean((y_true >= mu_16) & (y_true <= mu_84))

    # Compute Final Quantile Score
    epsilon = 1e-2
    penalty_function = lambda c: np.exp(-100 * (c - 0.6827)**2) if c < 0.6827 else np.exp(-10 * (c - 0.6827)**2)
    final_quantile_score = -np.log((average_interval_width + epsilon) * penalty_function(coverage))

    # Return metrics with primary metric first
    return {
        "Final Quantile Score": final_quantile_score,
        "Average Interval Width": average_interval_width,
        "Coverage": coverage
    }