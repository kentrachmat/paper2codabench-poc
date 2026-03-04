import numpy as np
from typing import Dict
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_ind

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
    """
    Compute evaluation metrics for classification tasks.

    Parameters:
    - y_true (np.ndarray): Ground truth labels (1D or 2D array).
    - y_pred (np.ndarray): Predicted labels (1D or 2D array).
    - task_type (str): Type of task, e.g., 'classification'.

    Returns:
    - Dict[str, float]: Dictionary of computed metrics.
    """
    # Reshape 1D arrays to 2D for consistency
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Validate input shapes
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("Mismatch in number of samples between y_true and y_pred.")
    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError("Mismatch in number of targets between y_true and y_pred.")
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Empty arrays provided for y_true or y_pred.")

    # Handle NaN values
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("NaN values detected in y_true or y_pred.")

    # Metrics computation
    metrics = {}

    # Primary Metric: Attack Accuracy
    attack_accuracy = accuracy_score(y_true, y_pred)
    metrics["Attack Accuracy"] = attack_accuracy

    # Efficiency: Measure computational overhead (example: fewer tokens is better)
    # Assuming efficiency is inversely proportional to the number of tokens in predictions
    efficiency = 1 / np.mean([len(str(pred)) for pred in y_pred.flatten()])
    metrics["Efficiency"] = efficiency

    # Model Effectiveness: Ensure minimal degradation in utility
    # Example: Compare utility scores (mock implementation)
    utility_scores = np.random.rand(y_true.shape[0])  # Mock utility scores
    effectiveness = np.mean(utility_scores)  # Higher utility is better
    metrics["Model Effectiveness"] = effectiveness

    # Error Bars: Compute standard deviation of Attack Accuracy
    error_bars = np.std([accuracy_score(y_true, np.random.choice(y_pred.flatten(), len(y_pred.flatten()))) for _ in range(100)])
    metrics["Error Bars"] = error_bars

    # Significance Testing: Compare Attack Accuracy with random baseline
    random_baseline = np.random.choice(y_true.flatten(), len(y_true.flatten()))
    _, p_value = ttest_ind(y_true.flatten(), random_baseline, equal_var=False)
    metrics["Significance Testing"] = p_value

    # Ensure primary metric is first in the dictionary
    metrics = {k: metrics[k] for k in sorted(metrics.keys(), key=lambda x: x != "Attack Accuracy")}

    return metrics