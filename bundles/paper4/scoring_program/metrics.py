import numpy as np
from typing import Dict
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
    """
    Compute evaluation metrics for the competition.

    Parameters:
    - y_true (np.ndarray): Ground truth values.
    - y_pred (np.ndarray): Predicted values.
    - task_type (str): Type of task (e.g., 'other').

    Returns:
    - Dict[str, float]: Dictionary containing metric names as keys and their computed float scores as values.
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
        # Primary Metric: Quantitative scores based on cooperative intelligence (mean squared error as a proxy)
        mse = mean_squared_error(y_true, y_pred)

        # Additional Metrics
        # Social welfare functions (R^2 score as a proxy for collective performance)
        r2 = r2_score(y_true, y_pred)

        # Qualitative transcript analysis for cooperative skills (Pearson correlation as a proxy for alignment)
        pearson_corr, _ = pearsonr(y_true.flatten(), y_pred.flatten())

        # Individual returns (mean absolute error as a proxy for individual performance)
        mae = np.mean(np.abs(y_true - y_pred))

    except Exception as e:
        raise RuntimeError(f"Error during metric computation: {e}")

    # Return metrics dictionary with primary metric first
    return {
        "Quantitative_Cooperative_Intelligence": mse,
        "Social_Welfare_Functions": r2,
        "Qualitative_Cooperative_Skills": pearson_corr,
        "Individual_Returns": mae
    }