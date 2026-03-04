import numpy as np
from typing import Dict
from sklearn.metrics import mean_squared_error

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
    """
    Compute evaluation metrics for the Codabench competition.

    Parameters:
    - y_true (np.ndarray): Ground truth values (1D or 2D array).
    - y_pred (np.ndarray): Predicted values (1D or 2D array).
    - task_type (str): Task type (not used in this implementation but required by Codabench).

    Returns:
    - Dict[str, float]: Dictionary containing the computed metrics.
      Primary metric (nRMSE) is listed first.
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
        raise ValueError("Empty arrays: y_true and y_pred must not be empty.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("NaN values detected: y_true and y_pred must not contain NaN values.")

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    std_true = np.std(y_true)
    if std_true == 0:
        raise ValueError("Standard deviation of y_true is zero, cannot compute normalized RMSE.")
    nrmse = rmse / std_true
    std_pred = np.std(y_pred)

    # Return metrics dictionary
    return {
        "nRMSE": nrmse,  # Primary metric
        "RMSE": rmse,
        "std": std_pred
    }