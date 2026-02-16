import numpy as np
from typing import Dict
from scipy.stats import norm

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
    """
    Compute evaluation metrics for the competition.

    Metrics:
        - coverage_score: Combines precision (interval width) and coverage (proportion of intervals containing µtruth).
        - precision: Measures the average width of the prediction intervals.
        - coverage: Proportion of intervals containing the true value.

    Args:
        y_true (np.ndarray): Ground truth values (µtruth). Shape (n_samples, 1).
        y_pred (np.ndarray): Predicted intervals (mu16, mu84). Shape (n_samples, 2).
        task_type (str): Task type, should be 'other' for this competition.

    Returns:
        Dict[str, float]: Dictionary containing the computed metrics.
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
        raise ValueError("y_pred must have exactly two columns (mu16, mu84).")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("y_true and y_pred must not contain NaN values.")

    # Extract lower and upper bounds of the prediction intervals
    mu16 = y_pred[:, 0]
    mu84 = y_pred[:, 1]

    # Validate interval bounds
    if np.any(mu16 > mu84):
        raise ValueError("Lower bounds (mu16) must not exceed upper bounds (mu84).")

    # Compute coverage: proportion of intervals containing the true value
    coverage = np.mean((y_true[:, 0] >= mu16) & (y_true[:, 0] <= mu84))

    # Compute precision: average width of the prediction intervals
    precision = np.mean(mu84 - mu16)

    # Compute coverage_score: penalizes under-coverage more than over-coverage
    coverage_score = coverage - 0.5 * np.maximum(0, 1 - coverage) - 0.1 * np.maximum(0, coverage - 1)

    # Return metrics with primary metric first
    return {
        "coverage_score": coverage_score,
        "precision": precision,
        "coverage": coverage
    }