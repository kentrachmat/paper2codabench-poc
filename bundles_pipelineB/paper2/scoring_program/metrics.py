import numpy as np
from typing import Dict
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_ind, f_oneway

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
    """
    Compute evaluation metrics for the Codabench competition.

    Parameters:
    - y_true (np.ndarray): Ground truth labels or values.
    - y_pred (np.ndarray): Predicted labels or values.
    - task_type (str): The task type, either 'red_team' or 'blue_team'.

    Returns:
    - Dict[str, float]: Dictionary of computed metrics with the primary metric first.
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
        raise ValueError("NaN values detected: y_true and y_pred must not contain NaN values.")

    # Initialize metrics dictionary
    metrics = {}

    # Compute metrics based on task type
    if task_type == "red_team":
        # Primary Metric: Attack Accuracy
        attack_accuracy = accuracy_score(y_true, y_pred)
        metrics["Attack Accuracy"] = attack_accuracy

        # Efficiency: Number of tokens/queries required (lower is better)
        efficiency = np.mean([len(str(pred)) for pred in y_pred.flatten()])
        metrics["Efficiency"] = efficiency

    elif task_type == "blue_team":
        # Primary Metric: Defense Effectiveness
        defense_effectiveness = accuracy_score(y_true, y_pred)
        metrics["Defense Effectiveness"] = defense_effectiveness

        # Model Effectiveness: Same as Defense Effectiveness for simplicity
        model_effectiveness = defense_effectiveness
        metrics["Model Effectiveness"] = model_effectiveness

        # Efficiency: Computational overhead (lower is better)
        efficiency = np.mean([len(str(pred)) for pred in y_pred.flatten()])
        metrics["Efficiency"] = efficiency

    else:
        raise ValueError("Invalid task_type. Must be 'red_team' or 'blue_team'.")

    # Statistical Significance: Perform t-test and ANOVA
    try:
        t_stat, t_p_value = ttest_ind(y_true.flatten(), y_pred.flatten(), equal_var=False)
        metrics["Statistical Significance (t-test p-value)"] = t_p_value

        f_stat, f_p_value = f_oneway(y_true.flatten(), y_pred.flatten())
        metrics["Statistical Significance (ANOVA p-value)"] = f_p_value
    except Exception as e:
        metrics["Statistical Significance (t-test p-value)"] = np.nan
        metrics["Statistical Significance (ANOVA p-value)"] = np.nan

    # Ensure primary metric is first in the dictionary
    if task_type == "red_team":
        metrics = {k: metrics[k] for k in ["Attack Accuracy", "Efficiency", "Statistical Significance (t-test p-value)", "Statistical Significance (ANOVA p-value)"]}
    elif task_type == "blue_team":
        metrics = {k: metrics[k] for k in ["Defense Effectiveness", "Model Effectiveness", "Efficiency", "Statistical Significance (t-test p-value)", "Statistical Significance (ANOVA p-value)"]}

    return metrics