import numpy as np
from typing import Dict
from sklearn.metrics import average_precision_score

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
    """
    Compute evaluation metrics for a classification task.

    Parameters:
    - y_true (np.ndarray): Ground truth binary labels, shape (n_samples, n_targets).
    - y_pred (np.ndarray): Predicted scores or probabilities, shape (n_samples, n_targets).
    - task_type (str): Task type, must be 'classification'.

    Returns:
    - Dict[str, float]: Dictionary of computed metrics with the primary metric first.
    """
    # Reshape 1D arrays to 2D
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Validate inputs
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch: y_true and y_pred must have the same shape.")
    if task_type != "classification":
        raise ValueError("Invalid task type. Only 'classification' is supported.")
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Empty arrays are not allowed.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("NaN values detected in inputs.")

    # Initialize metrics dictionary
    metrics = {}

    # Compute Mean Average Precision (Primary Metric)
    mean_avg_precision = 0.0
    for i in range(y_true.shape[1]):  # Compute per target
        mean_avg_precision += average_precision_score(y_true[:, i], y_pred[:, i])
    mean_avg_precision /= y_true.shape[1]
    metrics["Mean Average Precision"] = mean_avg_precision

    # Compute Top-k Precision (k=100)
    top_k_100_precision = compute_top_k_precision(y_true, y_pred, k=100)
    metrics["Top-100 Precision"] = top_k_100_precision

    # Compute Top-k Precision (k=1000)
    top_k_1000_precision = compute_top_k_precision(y_true, y_pred, k=1000)
    metrics["Top-1000 Precision"] = top_k_1000_precision

    return metrics


def compute_top_k_precision(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """
    Compute Top-k Precision for a classification task.

    Parameters:
    - y_true (np.ndarray): Ground truth binary labels, shape (n_samples, n_targets).
    - y_pred (np.ndarray): Predicted scores or probabilities, shape (n_samples, n_targets).
    - k (int): Number of top predictions to consider.

    Returns:
    - float: Top-k Precision score.
    """
    total_precision = 0.0

    for i in range(y_true.shape[1]):  # Compute per target
        # Sort predictions and corresponding true labels by predicted score
        sorted_indices = np.argsort(-y_pred[:, i])  # Descending order
        top_k_indices = sorted_indices[:k]
        top_k_true = y_true[top_k_indices, i]

        # Compute precision at k
        precision_at_k = np.sum(top_k_true) / k
        total_precision += precision_at_k

    # Average across all targets
    return total_precision / y_true.shape[1]