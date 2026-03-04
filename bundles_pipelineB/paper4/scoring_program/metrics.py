import numpy as np
from typing import Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
    """
    Compute evaluation metrics for the competition.

    Parameters:
    - y_true (np.ndarray): Ground truth values, can be 1D or 2D.
    - y_pred (np.ndarray): Predicted values, can be 1D or 2D.
    - task_type (str): Task type, e.g., 'other'. Currently not used but included for extensibility.

    Returns:
    - Dict[str, float]: Dictionary containing metric names as keys and their corresponding scores as values.
      The primary metric is listed first.
    """
    # Reshape 1D arrays to 2D for consistency
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Validate input shapes
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch: y_true and y_pred must have the same shape.")
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Empty input: y_true and y_pred must not be empty.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("NaN values detected: y_true and y_pred must not contain NaN values.")

    # Compute metrics
    try:
        # Primary metric: Mean Squared Error (lower is better, but higher is better for this task)
        mse = mean_squared_error(y_true, y_pred)
        # Secondary metric: Mean Absolute Error
        mae = mean_absolute_error(y_true, y_pred)
        # Additional metric: Spearman Correlation (measuring rank correlation)
        spearman_corr = np.mean([
            spearmanr(y_true[:, i], y_pred[:, i]).correlation
            for i in range(y_true.shape[1])
        ])

        # Social welfare scores (example: average of predictions)
        social_welfare = np.mean(y_pred)

    except Exception as e:
        raise RuntimeError(f"Error during metric computation: {e}")

    # Return metrics dictionary with primary metric first
    return {
        "mean_squared_error": mse,
        "mean_absolute_error": mae,
        "spearman_correlation": spearman_corr,
        "social_welfare": social_welfare
    }