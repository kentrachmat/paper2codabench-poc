import numpy as np
from typing import Dict
from sklearn.metrics import average_precision_score

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
    """
    Compute evaluation metrics for classification tasks.

    Parameters:
    - y_true (np.ndarray): Ground truth binary labels (shape: [n_samples, n_targets]).
    - y_pred (np.ndarray): Predicted scores or probabilities (shape: [n_samples, n_targets]).
    - task_type (str): Task type, expected to be 'classification'.

    Returns:
    - Dict[str, float]: Dictionary containing the computed metrics.
    """
    # Reshape 1D arrays to 2D for single target tasks
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Validate inputs
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch: y_true and y_pred must have the same shape.")
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Empty input arrays are not allowed.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("NaN values detected in input arrays.")

    # Compute Mean Average Precision (MAP)
    map_scores = []
    for i in range(y_true.shape[1]):  # Iterate over targets
        if np.unique(y_true[:, i]).size > 1:  # Avoid undefined MAP for constant labels
            map_scores.append(average_precision_score(y_true[:, i], y_pred[:, i]))
        else:
            map_scores.append(0.0)  # Assign 0 if MAP is undefined
    mean_average_precision = np.mean(map_scores)

    # Compute Top-k Precision for k=100 and k=1000
    def top_k_precision(y_true_col, y_pred_col, k):
        """Compute Top-k Precision for a single target."""
        sorted_indices = np.argsort(-y_pred_col)  # Sort in descending order
        top_k_indices = sorted_indices[:k]
        return np.sum(y_true_col[top_k_indices]) / k

    top_k_precision_100 = []
    top_k_precision_1000 = []
    for i in range(y_true.shape[1]):  # Iterate over targets
        top_k_precision_100.append(top_k_precision(y_true[:, i], y_pred[:, i], k=100))
        top_k_precision_1000.append(top_k_precision(y_true[:, i], y_pred[:, i], k=1000))
    mean_top_k_precision_100 = np.mean(top_k_precision_100)
    mean_top_k_precision_1000 = np.mean(top_k_precision_1000)

    # Return metrics dictionary with primary metric first
    return {
        "Mean Average Precision (MAP)": mean_average_precision,
        "Top-100 Precision": mean_top_k_precision_100,
        "Top-1000 Precision": mean_top_k_precision_1000
    }