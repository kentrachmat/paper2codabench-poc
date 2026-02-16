import numpy as np
from typing import Dict
from sklearn.metrics import average_precision_score, roc_auc_score

def mean_average_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Average Precision (MAP) across all targets.
    """
    average_precisions = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i]) == 0:  # Avoid division by zero for targets with no positives
            continue
        average_precisions.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    return np.mean(average_precisions) if average_precisions else 0.0

def top_k_precision(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """
    Compute Top-k Precision for a given k.
    """
    top_k_precisions = []
    for i in range(y_true.shape[1]):
        sorted_indices = np.argsort(-y_pred[:, i])[:k]
        top_k_true = y_true[sorted_indices, i]
        top_k_precisions.append(np.sum(top_k_true) / k)
    return np.mean(top_k_precisions)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
    """
    Compute evaluation metrics for classification tasks.

    Args:
        y_true (np.ndarray): Ground truth binary labels (0 or 1), shape (n_samples, n_targets).
        y_pred (np.ndarray): Predicted probabilities, shape (n_samples, n_targets).
        task_type (str): Task type, must be 'classification'.

    Returns:
        Dict[str, float]: Dictionary containing metric names and their computed values.
    """
    # Reshape 1D arrays to 2D
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Validate inputs
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch: y_true and y_pred must have the same shape.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Input contains NaN values.")
    if task_type != "classification":
        raise ValueError("Invalid task type. Only 'classification' is supported.")

    # Compute metrics
    metrics = {}
    metrics["mean_average_precision"] = mean_average_precision(y_true, y_pred)
    metrics["top_100_precision"] = top_k_precision(y_true, y_pred, k=100)
    metrics["top_1000_precision"] = top_k_precision(y_true, y_pred, k=1000)
    metrics["auc_roc"] = roc_auc_score(y_true, y_pred, average="macro", multi_class="ovr")

    return metrics