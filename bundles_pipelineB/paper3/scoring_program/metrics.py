import numpy as np
from typing import Dict
from sklearn.metrics import accuracy_score
from scipy.stats import norm

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
    """
    Compute evaluation metrics for classification tasks.

    Parameters:
    - y_true (np.ndarray): Ground truth labels or values.
    - y_pred (np.ndarray): Predicted labels or interval bounds.
    - task_type (str): Type of task, expected to be "classification".

    Returns:
    - Dict[str, float]: Dictionary containing metric names and their computed values.
    """
    # Reshape 1D arrays to 2D
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Validate input shapes
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same number of samples.")
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("y_true and y_pred cannot be empty.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("y_true and y_pred cannot contain NaN values.")

    # Extract components from y_pred
    mu_16 = y_pred[:, 0]  # Lower bound of confidence interval
    mu_84 = y_pred[:, 1]  # Upper bound of confidence interval
    predicted_class = y_pred[:, 2]  # Predicted classification labels

    # Compute metrics
    # Average Interval Width (w)
    interval_width = mu_84 - mu_16
    average_interval_width = np.mean(interval_width)

    # Coverage (c)
    coverage = np.mean((y_true[:, 0] >= mu_16) & (y_true[:, 0] <= mu_84))

    # Final Quantile Score
    epsilon = 1e-2
    def coverage_penalty(c):
        target_coverage = 0.6827
        if c < target_coverage:
            return np.exp(target_coverage - c)
        else:
            return np.exp(c - target_coverage)
    final_quantile_score = -np.log((average_interval_width + epsilon) * coverage_penalty(coverage))

    # Classification Accuracy
    classification_accuracy = accuracy_score(y_true[:, 1], predicted_class)

    # Return metrics dictionary
    return {
        "Final Quantile Score": final_quantile_score,
        "Average Interval Width (w)": average_interval_width,
        "Coverage (c)": coverage,
        "Classification Accuracy": classification_accuracy
    }