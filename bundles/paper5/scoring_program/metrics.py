import numpy as np
from typing import Dict
from scipy.spatial.distance import euclidean
from sklearn.metrics import roc_curve
from scipy.stats import wasserstein_distance

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
    """
    Compute evaluation metrics for the generation task.

    Parameters:
        y_true (np.ndarray): Ground truth values (2D array for Q and A metrics).
        y_pred (np.ndarray): Predicted values (2D array for Q and A metrics).
        task_type (str): Task type, should be 'generation'.

    Returns:
        Dict[str, float]: Dictionary containing the computed metrics.
    """
    # Ensure inputs are 2D arrays
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Validate shapes
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch: y_true and y_pred must have the same shape.")
    if y_true.shape[1] != 2:
        raise ValueError("Input arrays must have exactly 2 columns for Q and A metrics.")

    # Extract Q and A components
    Q_true, A_true = y_true[:, 0], y_true[:, 1]
    Q_pred, A_pred = y_pred[:, 0], y_pred[:, 1]

    # Compute individual metrics
    # 1. Final Score (Euclidean distance in Q-A space)
    final_scores = [euclidean((Q_t, A_t), (Q_p, A_p)) for Q_t, A_t, Q_p, A_p in zip(Q_true, A_true, Q_pred, A_pred)]
    final_score = np.mean(final_scores)

    # 2. Watermark Removal Performance (A) - 1 - TPR@0.1%FPR
    fpr, tpr, _ = roc_curve(A_true, A_pred)
    idx = np.where(fpr <= 0.001)[0]
    if len(idx) > 0:
        tpr_at_fpr = tpr[idx[-1]]
    else:
        tpr_at_fpr = 0.0
    watermark_removal_performance = 1 - tpr_at_fpr

    # 3. Image Quality Degradation (Q) - weighted aggregation of 8 IQMs
    # Assuming Q_pred already represents the weighted aggregation of 8 IQMs
    image_quality_degradation = np.mean(Q_pred)

    # Return metrics as a dictionary
    return {
        "Final Score": final_score,  # Primary metric
        "Watermark Removal Performance (A)": watermark_removal_performance,
        "Image Quality Degradation (Q)": image_quality_degradation
    }