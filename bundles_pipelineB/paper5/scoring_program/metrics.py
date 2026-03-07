import numpy as np
from typing import Dict
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy
from scipy.spatial.distance import euclidean
from scipy.linalg import sqrtm
import torch
import clip
from PIL import Image

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
    """
    Compute evaluation metrics for the generation task.

    Parameters:
    - y_true (np.ndarray): Ground truth images or metrics.
    - y_pred (np.ndarray): Predicted images or metrics.
    - task_type (str): Task type, e.g., 'generation'.

    Returns:
    - Dict[str, float]: Dictionary of computed metrics.
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
        raise ValueError("Empty input arrays are not allowed.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Input arrays contain NaN values.")

    # Metrics computation
    metrics = {}

    # Image Quality Degradation (Q) - normalized MSE
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    Q = np.clip(1 - mse, 0.1, 0.9)  # Normalize to [0.1, 0.9]
    metrics['Image Quality Degradation (Q)'] = Q

    # Watermark Removal Rate (A) - example metric (can be replaced with actual implementation)
    A = np.clip(np.mean(np.abs(y_true - y_pred)), 0.1, 0.9)  # Normalize to [0.1, 0.9]
    metrics['Watermark Removal Rate (A)'] = A

    # Primary Metric: Euclidean distance sqrt(Q^2 + A^2)
    primary_metric = np.sqrt(Q**2 + A**2)
    metrics['Euclidean Distance'] = primary_metric

    # Peak Signal-to-Noise Ratio (PSNR)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    metrics['Peak Signal-to-Noise Ratio (PSNR)'] = psnr

    # Structural Similarity Index (SSIM)
    ssim_value = ssim(y_true.squeeze(), y_pred.squeeze(), multichannel=True)
    metrics['Structural Similarity Index (SSIM)'] = ssim_value

    # Normalized Mutual Information (NMI)
    hist_2d, _, _ = np.histogram2d(y_true.flatten(), y_pred.flatten(), bins=20)
    nmi = entropy(hist_2d.flatten()) / (entropy(y_true.flatten()) + entropy(y_pred.flatten()))
    metrics['Normalized Mutual Information (NMI)'] = nmi

    # Frechet Inception Distance (FID) - placeholder, requires pre-trained model
    def calculate_fid(mu1, sigma1, mu2, sigma2):
        diff = mu1 - mu2
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

    fid = calculate_fid(np.mean(y_true, axis=0), np.cov(y_true, rowvar=False),
                        np.mean(y_pred, axis=0), np.cov(y_pred, rowvar=False))
    metrics['Frechet Inception Distance (FID)'] = fid

    # CLIP Image Fidelity (CLIP-FID) - placeholder, requires CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    def clip_fid(image1, image2):
        image1 = preprocess(Image.fromarray(image1)).unsqueeze(0).to(device)
        image2 = preprocess(Image.fromarray(image2)).unsqueeze(0).to(device)
        with torch.no_grad():
            features1 = model.encode_image(image1)
            features2 = model.encode_image(image2)
        return euclidean(features1.cpu().numpy(), features2.cpu().numpy())

    clip_fid_value = clip_fid(y_true[0], y_pred[0])  # Example for first image
    metrics['CLIP Image Fidelity (CLIP-FID)'] = clip_fid_value

    # Learned Perceptual Image Patch Similarity (LPIPS) - placeholder
    lpips_value = np.mean(np.abs(y_true - y_pred))  # Simplified placeholder
    metrics['Learned Perceptual Image Patch Similarity (LPIPS)'] = lpips_value

    # Delta Aesthetics Score (∆Aesthetics) - placeholder
    delta_aesthetics = np.mean(y_pred) - np.mean(y_true)  # Simplified placeholder
    metrics['Delta Aesthetics Score (∆Aesthetics)'] = delta_aesthetics

    # Delta Artifacts Score (∆Artifacts) - placeholder
    delta_artifacts = np.std(y_pred) - np.std(y_true)  # Simplified placeholder
    metrics['Delta Artifacts Score (∆Artifacts)'] = delta_artifacts

    # Return metrics with primary metric first
    return metrics