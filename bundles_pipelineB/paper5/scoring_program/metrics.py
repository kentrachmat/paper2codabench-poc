import numpy as np
from typing import Dict
from sklearn.metrics import roc_curve
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import euclidean
from scipy.stats import entropy

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
    """
    Compute evaluation metrics for the generation task.

    Parameters:
    - y_true (np.ndarray): Ground truth values.
    - y_pred (np.ndarray): Predicted values.
    - task_type (str): Task type, should be 'generation'.

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

    # Initialize metrics
    metrics = {}

    # Compute TPR@0.1%FPR
    fpr, tpr, _ = roc_curve(y_true.ravel(), y_pred.ravel())
    tpr_at_fpr = tpr[np.searchsorted(fpr, 0.001, side='right') - 1] if np.any(fpr <= 0.001) else 0.0
    metrics["TPR@0.1%FPR"] = tpr_at_fpr

    # Compute PSNR
    try:
        psnr_value = psnr(y_true, y_pred, data_range=y_true.max() - y_true.min())
    except ValueError:
        psnr_value = 0.0
    metrics["PSNR"] = psnr_value

    # Compute SSIM
    try:
        ssim_value = ssim(y_true, y_pred, data_range=y_true.max() - y_true.min(), multichannel=True)
    except ValueError:
        ssim_value = 0.0
    metrics["SSIM"] = ssim_value

    # Compute FID (placeholder, requires specific implementation)
    fid_value = compute_fid_placeholder(y_true, y_pred)
    metrics["FID"] = fid_value

    # Compute CLIP-FID (placeholder, requires specific implementation)
    clip_fid_value = compute_clip_fid_placeholder(y_true, y_pred)
    metrics["CLIP-FID"] = clip_fid_value

    # Compute LPIPS (placeholder, requires specific implementation)
    lpips_value = compute_lpips_placeholder(y_true, y_pred)
    metrics["LPIPS"] = lpips_value

    # Compute NMI (Normalized Mutual Information)
    nmi_value = compute_nmi_placeholder(y_true, y_pred)
    metrics["NMI"] = nmi_value

    # Compute Delta Aesthetics Score (placeholder, requires specific implementation)
    delta_aesthetics_value = compute_delta_aesthetics_placeholder(y_true, y_pred)
    metrics["Delta Aesthetics Score"] = delta_aesthetics_value

    # Compute Delta Artifacts Score (placeholder, requires specific implementation)
    delta_artifacts_value = compute_delta_artifacts_placeholder(y_true, y_pred)
    metrics["Delta Artifacts Score"] = delta_artifacts_value

    # Compute final score
    A = 1 - metrics["TPR@0.1%FPR"]
    quality_metrics = [
        metrics["PSNR"],
        metrics["SSIM"],
        metrics["FID"],
        metrics["CLIP-FID"],
        metrics["LPIPS"],
        metrics["NMI"],
        metrics["Delta Aesthetics Score"],
        metrics["Delta Artifacts Score"]
    ]
    normalized_quality_metrics = [
        np.clip((metric - 0.1) / (0.9 - 0.1), 0.1, 0.9) for metric in quality_metrics
    ]
    Q = np.mean(normalized_quality_metrics)
    final_score = euclidean([Q, A], [0, 0])
    metrics["final score"] = final_score

    # Ensure primary metric is first
    metrics = {"final score": metrics.pop("final score"), **metrics}

    return metrics

def compute_fid_placeholder(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Placeholder for FID computation."""
    return 0.0

def compute_clip_fid_placeholder(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Placeholder for CLIP-FID computation."""
    return 0.0

def compute_lpips_placeholder(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Placeholder for LPIPS computation."""
    return 0.0

def compute_nmi_placeholder(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Placeholder for NMI computation."""
    return 0.0

def compute_delta_aesthetics_placeholder(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Placeholder for Delta Aesthetics Score computation."""
    return 0.0

def compute_delta_artifacts_placeholder(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Placeholder for Delta Artifacts Score computation."""
    return 0.0