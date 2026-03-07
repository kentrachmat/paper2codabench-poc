import numpy as np
from typing import Dict
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
    """
    Compute evaluation metrics for classification tasks, including utility and fairness metrics.

    Parameters:
    - y_true (np.ndarray): Ground truth labels (binary classification: 0 or 1).
    - y_pred (np.ndarray): Predicted probabilities or scores for the positive class.
    - task_type (str): Task type, e.g., 'classification'.

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
        raise ValueError("y_true and y_pred must have the same number of samples.")
    if y_true.shape[1] != 1 or y_pred.shape[1] != 1:
        raise ValueError("y_true and y_pred must be single-target arrays.")

    # Flatten arrays for metric computation
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    # Handle edge cases
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("y_true and y_pred must not contain NaN values.")

    # Utility metrics
    metrics = {}
    try:
        auc = roc_auc_score(y_true, y_pred)
        metrics['AUC'] = auc
    except ValueError:
        metrics['AUC'] = np.nan

    try:
        acc = accuracy_score(y_true, (y_pred >= 0.5).astype(int))
        metrics['ACC'] = acc
    except ValueError:
        metrics['ACC'] = np.nan

    try:
        ap = average_precision_score(y_true, y_pred)
        metrics['AP'] = ap
    except ValueError:
        metrics['AP'] = np.nan

    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        metrics['EER'] = eer
    except ValueError:
        metrics['EER'] = np.nan

    # Fairness metrics (placeholders for actual subgroup-based computations)
    # These metrics require subgroup information, which is not provided in y_true/y_pred.
    # Replace these placeholders with actual fairness metric computations as needed.
    metrics['FPR'] = np.nan  # False Positive Rate
    metrics['TPR'] = np.nan  # True Positive Rate
    metrics['FEO'] = np.nan  # Fairness Equal Opportunity
    metrics['FMEO'] = np.nan  # Fairness Mean Equal Opportunity
    metrics['FDP'] = np.nan  # Fairness Demographic Parity
    metrics['FOAE'] = np.nan  # Fairness Overall Accuracy Equality
    metrics['FIND'] = np.nan  # Fairness Independence

    # Ensure primary metric is first in the dictionary
    metrics = {k: metrics[k] for k in sorted(metrics.keys(), key=lambda x: (x != 'AUC', x))}

    return metrics