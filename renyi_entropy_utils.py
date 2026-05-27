import numpy as np
import cv2


def compute_2d_histogram(image, kernel_size=3):
    """
    Compute the joint 2D histogram of (pixel intensity, local neighborhood mean).
    Returns a normalized 256x256 probability matrix.
    """
    local_mean = cv2.boxFilter(image.astype(np.float32), -1, (kernel_size, kernel_size))
    local_mean = np.clip(np.round(local_mean), 0, 255).astype(np.int32)
    flat_intensity = image.ravel().astype(np.int32)
    flat_mean = local_mean.ravel()

    hist_2d = np.zeros((256, 256), dtype=np.float64)
    np.add.at(hist_2d, (flat_intensity, flat_mean), 1)

    total = hist_2d.sum()
    if total > 0:
        hist_2d /= total
    return hist_2d


def renyi_entropy_region(p_region, q=2):
    """
    Compute Rényi entropy of order q for a 2D probability sub-region.
    The region probabilities are normalised internally before computing entropy.
    """
    p_flat = p_region.ravel()
    p_flat = p_flat[p_flat > 1e-12]
    if len(p_flat) == 0:
        return 0.0
    p_norm = p_flat / p_flat.sum()
    if q == 1:
        # Limit case → Shannon entropy
        return float(-np.sum(p_norm * np.log(p_norm)))
    return float((1.0 / (1.0 - q)) * np.log(np.sum(p_norm ** q)))


def calculate_renyi_entropy_2d(image, thresholds, q=2, kernel_size=3):
    """
    2D Rényi entropy objective function for multilevel thresholding.

    Partitions the 2D joint histogram into (n+1) x (n+1) sub-regions
    defined by the threshold values and sums the Rényi entropy across all regions.

    Parameters
    ----------
    image       : np.ndarray (H x W, uint8 grayscale)
    thresholds  : list of int threshold values in [1, 255]
    q           : float, Rényi order (default 2)
    kernel_size : int, neighbourhood size for local mean (default 3)

    Returns
    -------
    float : total 2D Rényi entropy (higher = better segmentation)
    """
    thresholds = sorted(thresholds)
    boundaries = [0] + thresholds + [256]

    hist_2d = compute_2d_histogram(image, kernel_size)

    total_entropy = 0.0
    for i in range(len(boundaries) - 1):
        for j in range(len(boundaries) - 1):
            region = hist_2d[boundaries[i]:boundaries[i + 1],
                             boundaries[j]:boundaries[j + 1]]
            total_entropy += renyi_entropy_region(region, q)

    return total_entropy
