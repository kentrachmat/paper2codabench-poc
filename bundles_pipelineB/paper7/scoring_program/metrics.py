import numpy as np
from typing import Dict
from sklearn.metrics import mean_squared_error

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
    """
    Compute evaluation metrics for the competition.

    Parameters:
    - y_true (np.ndarray): Ground truth values, shape (n_samples, n_targets).
    - y_pred (np.ndarray): Predicted values, shape (n_samples, n_targets).
    - task_type (str): Task type, not used in this competition but included for compatibility.

    Returns:
    - Dict[str, float]: Dictionary containing the computed metrics.
    """
    # Reshape 1D arrays to 2D
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Validate inputs
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch: y_true and y_pred must have the same shape.")
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Empty input: y_true and y_pred must not be empty.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("NaN values detected: y_true and y_pred must not contain NaN values.")

    # Compute nRMSE for each target
    nrmse_scores = []
    for i in range(y_true.shape[1]):
        true_values = y_true[:, i]
        pred_values = y_pred[:, i]
        rmse = np.sqrt(mean_squared_error(true_values, pred_values))
        std_dev = np.std(true_values)
        if std_dev == 0:
            raise ValueError(f"Standard deviation of true values is zero for target {i}. Cannot compute nRMSE.")
        nrmse = rmse / std_dev
        nrmse_scores.append(nrmse)

    # Weighted average of nRMSE scores for overall ranking
    if len(nrmse_scores) != 2:
        raise ValueError("Expected exactly 2 targets for weighted averaging, but got {len(nrmse_scores)}.")
    overall_nrmse = 0.3 * nrmse_scores[0] + 0.7 * nrmse_scores[1]

    # Return metrics
    return {
        "nRMSE": overall_nrmse,  # Primary metric
        "nRMSE_S1": nrmse_scores[0],
        "nRMSE_S2": nrmse_scores[1],
    }