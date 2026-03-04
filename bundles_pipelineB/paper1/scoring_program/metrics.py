import numpy as np
from typing import Dict
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

def mean_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the mean relative error."""
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_error = np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, np.nan))
    return np.nanmean(relative_error)

def compute_spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the Spearman correlation."""
    correlations = []
    for i in range(y_true.shape[1]):
        if np.all(y_true[:, i] == y_true[0, i]) or np.all(y_pred[:, i] == y_pred[0, i]):
            correlations.append(0.0)  # Handle constant arrays
        else:
            corr, _ = spearmanr(y_true[:, i], y_pred[:, i], nan_policy='omit')
            correlations.append(corr)
    return np.nanmean(correlations)

def compute_speedup(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the speedup metric (normalized to a max score of 1)."""
    # Placeholder: Replace with actual speedup computation logic
    return min(1.0, np.random.random())  # Random value for demonstration

def compute_global_score(ml_score: float, physics_score: float, ood_score: float) -> float:
    """Compute the global score as a weighted combination of metrics."""
    return 0.4 * ml_score + 0.3 * physics_score + 0.3 * ood_score

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
    """
    Compute evaluation metrics for the competition.

    Parameters:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.
        task_type (str): Task type (not used in this implementation).

    Returns:
        Dict[str, float]: Dictionary of computed metrics.
    """
    # Reshape 1D arrays to 2D
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Validate input shapes
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch: y_true and y_pred must have the same shape.")
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Empty arrays are not allowed for y_true or y_pred.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("NaN values detected in y_true or y_pred.")

    # Compute individual metrics
    mse = mean_squared_error(y_true, y_pred)
    spearman_corr = compute_spearman_correlation(y_true, y_pred)
    mre = mean_relative_error(y_true, y_pred)
    speedup = compute_speedup(y_true, y_pred)

    # Compute global score
    global_score = compute_global_score(ml_score=1 - mse, physics_score=spearman_corr, ood_score=1 - mre)

    # Return metrics dictionary with primary metric first
    return {
        "global_score": global_score,
        "mean_squared_error": mse,
        "spearman_correlation": spearman_corr,
        "mean_relative_error": mre,
        "speedup": speedup
    }