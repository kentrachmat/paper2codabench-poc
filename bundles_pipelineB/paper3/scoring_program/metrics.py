import numpy as np
from typing import Dict
from scipy.stats import norm

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
    """
    Compute evaluation metrics for the Codabench competition.

    Parameters:
    - y_true (np.ndarray): Ground truth values, expected shape (n_samples, 2) where columns are [mu16, mu84].
    - y_pred (np.ndarray): Predicted values, expected shape (n_samples, 2) where columns are [mu16, mu84].
    - task_type (str): Task type, expected to be 'other'.

    Returns:
    - Dict[str, float]: Dictionary containing computed metrics.
    """
    # Reshape 1D arrays to 2D
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Validate input shapes
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch: y_true and y_pred must have the same shape.")
    if y_true.shape[1] != 2 or y_pred.shape[1] != 2:
        raise ValueError("Invalid shape: y_true and y_pred must have exactly 2 columns (mu16, mu84).")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Input contains NaN values.")

    # Extract lower and upper bounds
    mu16_true, mu84_true = y_true[:, 0], y_true[:, 1]
    mu16_pred, mu84_pred = y_pred[:, 0], y_pred[:, 1]

    # Compute metrics
    n_samples = len(y_true)
    epsilon = 1e-2  # Regularization term
    target_coverage = 0.6827  # Target coverage for 68.27% CI

    # Average Interval Width (w)
    interval_widths = mu84_pred - mu16_pred
    average_interval_width = np.mean(interval_widths)

    # Coverage (c)
    coverage = np.mean((mu16_pred <= mu16_true) & (mu84_pred >= mu84_true))

    # Binomial statistical error (σ68)
    sigma68 = np.sqrt(target_coverage * (1 - target_coverage) / n_samples)

    # Penalty Function (f(c))
    def penalty_function(c, target, sigma):
        if c >= target:
            return 1
        else:
            return np.exp(-((target - c) / sigma)**2)

    penalty = penalty_function(coverage, target_coverage, sigma68)

    # Quantile Score
    quantile_score = -np.log((average_interval_width + epsilon) * penalty)

    # Return metrics
    return {
        "Quantile Score": quantile_score,  # Primary metric
        "Average Interval Width (w)": average_interval_width,
        "Coverage (c)": coverage,
        "Penalty Function (f(c))": penalty
    }