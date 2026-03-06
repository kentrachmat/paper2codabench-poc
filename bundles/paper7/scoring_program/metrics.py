import numpy as np
from typing import Dict
from sklearn.metrics import mean_squared_error

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
    """
    Compute evaluation metrics for the competition.

    Parameters:
    - y_true (np.ndarray): Ground truth values. Can be 1D or 2D.
    - y_pred (np.ndarray): Predicted values. Can be 1D or 2D.
    - task_type (str): Task type (not used in this implementation, included for extensibility).

    Returns:
    - Dict[str, float]: Dictionary containing the computed metrics.
    """
    # Reshape 1D arrays to 2D
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Validate input shapes
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true shape {y_true.shape} and y_pred shape {y_pred.shape} must be the same.")
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Input arrays must not be empty.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Input arrays must not contain NaN values.")

    # Compute metrics
    metrics = {}
    try:
        # Normalized Root Mean Square Error (nRMSE)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values'))
        std_dev = np.std(y_true, axis=0, ddof=1)
        nrmse = rmse / std_dev
        nrmse_mean = np.mean(nrmse)  # Average across all targets
        metrics["nRMSE"] = nrmse_mean
    except ZeroDivisionError:
        raise ValueError("Standard deviation of true values is zero, cannot compute nRMSE.")

    # Return metrics with primary metric first
    return metrics