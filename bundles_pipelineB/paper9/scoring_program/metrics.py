import numpy as np
from typing import Dict
from scipy.stats import kendalltau
from sklearn.metrics import accuracy_score

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
    """
    Compute evaluation metrics for the competition.

    Parameters:
    - y_true (np.ndarray): Ground truth labels or values.
    - y_pred (np.ndarray): Predicted labels or values.
    - task_type (str): Type of task (e.g., 'classification', 'regression', etc.).

    Returns:
    - Dict[str, float]: Dictionary containing metric names as keys and their computed values as floats.
    """
    # Reshape 1D arrays to 2D
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Validation checks
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch: y_true and y_pred must have the same shape.")
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Empty input: y_true and y_pred must not be empty.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("NaN values detected: y_true and y_pred must not contain NaN values.")

    # Metric computations
    def signal_quality(y_true, y_pred):
        """Compute Signal Quality (ScoreSQ) as a combination of Monotonicity Score and Autocorrelation Strength."""
        monotonicity_score = np.mean([
            kendalltau(y_true[:, i], y_pred[:, i])[0] for i in range(y_true.shape[1])
        ])
        autocorrelation_strength = np.mean([
            np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1] for i in range(y_true.shape[1])
        ])
        return 0.5 * monotonicity_score + 0.5 * autocorrelation_strength

    def ranking_consistency(y_true, y_pred):
        """Compute Ranking Consistency (ScoreRC) using Kendall's Tau."""
        return np.mean([
            kendalltau(y_true[:, i], y_pred[:, i])[0] for i in range(y_true.shape[1])
        ])

    def compliance_score(y_true, y_pred):
        """Compute Compliance Score (ScoreCS) as a domain-specific alignment metric."""
        # Placeholder for domain-specific logic; here we use a simple correlation
        return np.mean([
            np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1] for i in range(y_true.shape[1])
        ])

    def classification_accuracy(y_true, y_pred):
        """Compute Classification Accuracy for classification tasks."""
        if task_type != 'classification':
            return 0.0
        return accuracy_score(y_true, y_pred)

    # Calculate metrics
    score_sq = signal_quality(y_true, y_pred)
    score_rc = ranking_consistency(y_true, y_pred)
    score_cs = compliance_score(y_true, y_pred)
    score_acc = classification_accuracy(y_true, y_pred)

    # Weighted total score (example weights: 0.4, 0.3, 0.2, 0.1)
    total_score = 0.4 * score_sq + 0.3 * score_rc + 0.2 * score_cs + 0.1 * score_acc

    # Return metrics dictionary with primary metric first
    return {
        "Signal Quality (ScoreSQ)": score_sq,
        "Ranking Consistency (ScoreRC)": score_rc,
        "Compliance Score (ScoreCS)": score_cs,
        "Classification Accuracy": score_acc,
        "Total Score": total_score
    }