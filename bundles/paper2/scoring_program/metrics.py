import numpy as np
from typing import Dict
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_ind

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
    """
    Compute evaluation metrics for the competition.

    Parameters:
    - y_true (np.ndarray): Ground truth labels or values.
    - y_pred (np.ndarray): Predicted labels or values.
    - task_type (str): Type of task (e.g., 'other').

    Returns:
    - Dict[str, float]: Dictionary containing metric names and their computed values.
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
        raise ValueError("Empty arrays: y_true and y_pred must contain data.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("NaN values detected: y_true and y_pred must not contain NaN values.")

    # Metric computations
    attack_accuracy = accuracy_score(y_true, y_pred)  # Example metric for attack success rate
    defense_effectiveness = 1.0 - attack_accuracy  # Defense effectiveness is inverse of attack success
    model_utility = np.mean(y_pred)  # Placeholder for model utility (adjust based on task specifics)

    # Statistical significance (example: t-test between y_true and y_pred)
    try:
        _, p_value = ttest_ind(y_true.flatten(), y_pred.flatten(), equal_var=False)
        statistical_significance = p_value
    except Exception as e:
        statistical_significance = np.nan  # Handle edge cases where t-test fails

    # Efficiency metric (example: number of tokens/queries)
    efficiency = np.sum(y_pred)  # Placeholder for efficiency computation

    # Return metrics dictionary with primary metric first
    metrics = {
        "attack_success_rate": attack_accuracy,  # Primary metric
        "defense_effectiveness": defense_effectiveness,
        "model_utility": model_utility,
        "statistical_significance": statistical_significance,
        "efficiency": efficiency
    }

    return metrics