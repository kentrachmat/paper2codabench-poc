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
    - task_type (str): Task type, e.g., 'other'.

    Returns:
    - Dict[str, float]: Dictionary containing calculated metrics.
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
        raise ValueError("Invalid input: y_true and y_pred must not contain NaN values.")

    # Metric 1: Signal Quality (SQ)
    # Sub-metric 1.1: Monotonicity Score (Spearman correlation)
    monotonicity_scores = []
    for i in range(y_true.shape[1]):
        corr, _ = spearmanr(y_true[:, i], y_pred[:, i])
        monotonicity_scores.append(corr if not np.isnan(corr) else 0.0)
    monotonicity_score = np.mean(monotonicity_scores)

    # Sub-metric 1.2: Autocorrelation Strength (1 - normalized MSE)
    autocorrelation_scores = []
    for i in range(y_true.shape[1]):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        variance = np.var(y_true[:, i])
        autocorrelation_scores.append(1 - mse / variance if variance > 0 else 0.0)
    autocorrelation_strength = np.mean(autocorrelation_scores)

    signal_quality = (monotonicity_score + autocorrelation_strength) / 2

    # Metric 2: Ranking Consistency (RC)
    ranking_consistency_scores = []
    for i in range(y_true.shape[1]):
        corr, _ = spearmanr(np.argsort(y_true[:, i]), np.argsort(y_pred[:, i]))
        ranking_consistency_scores.append(corr if not np.isnan(corr) else 0.0)
    ranking_consistency = np.mean(ranking_consistency_scores)

    # Metric 3: Compliance Score (CS)
    compliance_scores = []
    for i in range(y_true.shape[1]):
        compliance_scores.append(1 - mean_squared_error(y_true[:, i], y_pred[:, i]))
    compliance_score = np.mean(compliance_scores)

    # Metric 4: Total Score
    total_score = 0.5 * signal_quality + 0.1 * ranking_consistency + 0.4 * compliance_score

    # Return metrics
    return {
        "Total Score": total_score,
        "Signal Quality": signal_quality,
        "Ranking Consistency": ranking_consistency,
        "Compliance Score": compliance_score
    }