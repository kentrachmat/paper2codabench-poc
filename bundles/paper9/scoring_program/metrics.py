import numpy as np
from typing import Dict
from scipy.stats import kendalltau
from sklearn.metrics import mean_squared_error

def compute_monotonicity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the monotonicity score, which evaluates whether predictions
    maintain the monotonic relationship present in the ground truth.
    """
    try:
        diffs_true = np.diff(y_true, axis=0)
        diffs_pred = np.diff(y_pred, axis=0)
        monotonicity = np.mean(np.sign(diffs_true) == np.sign(diffs_pred))
        return monotonicity
    except Exception:
        return 0.0

def compute_autocorrelation_strength(y_pred: np.ndarray) -> float:
    """
    Compute the autocorrelation strength of the predictions.
    """
    try:
        autocorr = np.corrcoef(y_pred[:-1].flatten(), y_pred[1:].flatten())[0, 1]
        return autocorr if not np.isnan(autocorr) else 0.0
    except Exception:
        return 0.0

def compute_signal_quality(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Signal Quality (SQ) metric as a combination of monotonicity
    score and autocorrelation strength.
    """
    monotonicity = compute_monotonicity_score(y_true, y_pred)
    autocorrelation = compute_autocorrelation_strength(y_pred)
    return (monotonicity + autocorrelation) / 2

def compute_ranking_consistency(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Ranking Consistency (RC) metric using Kendall's Tau coefficient.
    """
    try:
        tau, _ = kendalltau(y_true.flatten(), y_pred.flatten())
        return tau if not np.isnan(tau) else 0.0
    except Exception:
        return 0.0

def compute_compliance_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Compliance Score (CS), which assesses alignment with
    scientific knowledge domains. For simplicity, this example uses
    mean squared error as a placeholder.
    """
    try:
        mse = mean_squared_error(y_true, y_pred)
        compliance_score = 1 / (1 + mse)  # Transform MSE to a bounded score
        return compliance_score
    except Exception:
        return 0.0

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
    """
    Compute evaluation metrics for the competition.

    Parameters:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.
        task_type (str): Task type (not used in this implementation).

    Returns:
        Dict[str, float]: Dictionary containing all metrics, with the primary metric first.
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

    # Compute individual metrics
    score_sq = compute_signal_quality(y_true, y_pred)
    score_rc = compute_ranking_consistency(y_true, y_pred)
    score_cs = compute_compliance_score(y_true, y_pred)

    # Compute total score
    total_score = 0.4 * score_sq + 0.2 * score_rc + 0.4 * score_cs

    # Return metrics dictionary
    return {
        "Total Score": total_score,
        "Signal Quality (SQ)": score_sq,
        "Ranking Consistency (RC)": score_rc,
        "Compliance Score (CS)": score_cs
    }