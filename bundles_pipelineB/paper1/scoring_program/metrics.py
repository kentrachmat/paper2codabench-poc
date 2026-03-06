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
    - task_type (str): Task type (not used in this implementation).

    Returns:
    - Dict[str, float]: Dictionary of computed metrics.
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
        raise ValueError("NaN values detected: y_true and y_pred must not contain NaNs.")

    # Compute Mean Squared Error (MSE)
    mse = mean_squared_error(y_true, y_pred)

    # Compute Speedup (log10 normalization)
    # Assuming speedup is the ratio of baseline time to model time, provided in y_pred[:, -1]
    # Here, we use a placeholder calculation for demonstration purposes
    baseline_time = 1.0  # Placeholder for baseline time
    model_time = 0.1     # Placeholder for model time
    speedup = np.log10(baseline_time / model_time)

    # Compute Spearman correlation for drag coefficient (cd) and lift coefficient (cl)
    rho_d, _ = spearmanr(y_true[:, -2], y_pred[:, -2])  # Assuming cd is the second-to-last column
    rho_l, _ = spearmanr(y_true[:, -1], y_pred[:, -1])  # Assuming cl is the last column

    # Compute Mean Relative Error
    relative_error = np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))
    mean_relative_error = np.mean(relative_error)

    # Compute Global Score (weighted combination of metrics)
    # Placeholder weights for demonstration purposes
    w_mse = 0.4
    w_speedup = 0.2
    w_rho_d = 0.2
    w_rho_l = 0.2
    global_score = (
        w_mse * (1 / (1 + mse)) +  # Inverse MSE (higher is better)
        w_speedup * speedup +
        w_rho_d * rho_d +
        w_rho_l * rho_l
    )

    # Return metrics dictionary with primary metric first
    return {
        "Global Score": global_score,
        "Mean Squared Error (MSE)": mse,
        "Speedup": speedup,
        "Spearman correlation (ρ_D)": rho_d,
        "Spearman correlation (ρ_L)": rho_l,
        "Mean Relative Error": mean_relative_error
    }