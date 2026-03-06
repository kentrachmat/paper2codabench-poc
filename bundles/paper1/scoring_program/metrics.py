import numpy as np
from typing import Dict
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
    """
    Compute evaluation metrics for the competition.

    Parameters:
    - y_true (np.ndarray): Ground truth values.
    - y_pred (np.ndarray): Predicted values.
    - task_type (str): Type of task (e.g., 'other').

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
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Empty input: y_true and y_pred must not be empty.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("NaN values detected: y_true and y_pred must not contain NaNs.")

    # Metric computations
    mse = mean_squared_error(y_true, y_pred)
    speedup = np.log10(np.maximum(1e-10, np.mean(y_true) / np.mean(y_pred)))  # Avoid division by zero
    spearman_rho_D, _ = spearmanr(y_true[:, 0], y_pred[:, 0])  # Spearman correlation for C_D
    spearman_rho_L, _ = spearmanr(y_true[:, 1], y_pred[:, 1])  # Spearman correlation for C_L
    mean_relative_error_C_D = np.mean(np.abs((y_true[:, 0] - y_pred[:, 0]) / np.maximum(1e-10, y_true[:, 0])))
    mean_relative_error_C_L = np.mean(np.abs((y_true[:, 1] - y_pred[:, 1]) / np.maximum(1e-10, y_true[:, 1])))

    # Global Score calculation
    ml_score = 1 - mse  # Example: Higher is better
    physics_compliance = (spearman_rho_D + spearman_rho_L) / 2
    ood_generalization = 1 - (mean_relative_error_C_D + mean_relative_error_C_L) / 2
    global_score = 0.4 * ml_score + 0.3 * physics_compliance + 0.3 * ood_generalization

    # Return metrics dictionary
    return {
        "Global Score": global_score,
        "Mean Squared Error (MSE)": mse,
        "Speedup (logarithmic scale)": speedup,
        "Spearman correlation (rho_D)": spearman_rho_D,
        "Spearman correlation (rho_L)": spearman_rho_L,
        "Mean Relative Error (C_D)": mean_relative_error_C_D,
        "Mean Relative Error (C_L)": mean_relative_error_C_L
    }