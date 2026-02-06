"""Calibration helper functions for LAF Z-stack benchmarking."""

from __future__ import annotations

import cv2
import numpy as np

from data_types import ZStackFrame


def generate_nonuniform_z_positions(
    z_range_um: float,
    fine_range_um: float = 15.0,
    fine_step_um: float = 1.0,
    coarse_step_um: float = 14.0,
) -> np.ndarray:
    """Generate non-uniform Z positions: fine steps near center, coarse elsewhere.

    Args:
        z_range_um: Total range (±z_range_um/2 from center)
        fine_range_um: Range around center with fine steps (±fine_range_um)
        fine_step_um: Step size in the fine region
        coarse_step_um: Step size in the coarse region

    Returns:
        Sorted array of Z positions in µm, centered at 0
    """
    positions = []

    # Fine region: -fine_range to +fine_range with fine steps
    fine_positions = np.arange(-fine_range_um, fine_range_um + fine_step_um / 2, fine_step_um)
    positions.extend(fine_positions)

    # Coarse region: from fine_range to z_range/2
    half_range = z_range_um / 2
    if half_range > fine_range_um:
        # Positive coarse region
        coarse_start = fine_range_um + coarse_step_um
        coarse_pos = np.arange(coarse_start, half_range + coarse_step_um / 2, coarse_step_um)
        positions.extend(coarse_pos)

        # Negative coarse region
        coarse_neg = np.arange(-coarse_start, -half_range - coarse_step_um / 2, -coarse_step_um)
        positions.extend(coarse_neg)

    # Remove duplicates and sort
    positions = np.unique(np.array(positions))
    return np.sort(positions)


def compute_focus_measure(image: np.ndarray, method: str = "laplacian_var") -> float:
    """Compute a focus measure for an image (higher = more in focus).

    Args:
        image: Grayscale image
        method: Focus measure method
            - "laplacian_var": Variance of Laplacian (default, good general purpose)
            - "gradient": Sum of gradient magnitudes
            - "tenengrad": Tenengrad (gradient-based)
            - "peak_intensity": Max intensity (sharp dots = brighter peaks)
            - "contrast": Std/mean ratio (sharp = high contrast)
            - "neg_entropy": Negative entropy (sharp = concentrated = low entropy)
            - "percentile_ratio": 99th/50th percentile ratio
            - "local_variance": Mean of local variance (texture measure)

    Returns:
        Focus measure score (higher = better focus)
    """
    if image is None:
        return 0.0

    img = image.astype(np.float64)

    if method == "laplacian_var":
        # Variance of Laplacian - classic focus measure
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        return float(laplacian.var())

    elif method == "gradient":
        # Sum of gradient magnitudes
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.sqrt(gx**2 + gy**2).sum())

    elif method == "tenengrad":
        # Tenengrad - sum of squared gradients above threshold
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        grad_sq = gx**2 + gy**2
        return float(grad_sq.sum())

    elif method == "peak_intensity":
        # Maximum intensity - sharp dots have higher peaks
        return float(img.max())

    elif method == "contrast":
        # Coefficient of variation (std/mean) - sharp dots = high contrast
        mean = img.mean()
        if mean < 1e-6:
            return 0.0
        return float(img.std() / mean)

    elif method == "neg_entropy":
        # Negative entropy - sharp/concentrated = lower entropy = higher score
        # Normalize to [0, 1] range for histogram
        img_norm = img / (img.max() + 1e-6)
        hist, _ = np.histogram(img_norm.flatten(), bins=256, range=(0, 1))
        hist = hist / hist.sum()  # Normalize to probabilities
        hist = hist[hist > 0]  # Remove zeros for log
        entropy = -np.sum(hist * np.log2(hist))
        # Return negative so higher = more focused
        return float(-entropy)

    elif method == "percentile_ratio":
        # Ratio of high to mid percentile - sharp dots have high peaks vs background
        p99 = np.percentile(img, 99)
        p50 = np.percentile(img, 50)
        if p50 < 1e-6:
            return 0.0
        return float(p99 / p50)

    elif method == "local_variance":
        # Mean local variance using a sliding window
        # High local variance = sharp edges/features
        kernel_size = 15
        mean_filter = cv2.blur(img, (kernel_size, kernel_size))
        sqr_mean_filter = cv2.blur(img**2, (kernel_size, kernel_size))
        local_var = sqr_mean_filter - mean_filter**2
        return float(local_var.mean())

    else:
        raise ValueError(f"Unknown focus method: {method}")


def find_best_focus_index(
    frames: list[ZStackFrame],
    use_bf: bool = True,
    method: str = "laplacian_var",
) -> tuple[int, np.ndarray]:
    """Find the frame with best focus in a z-stack.

    Args:
        frames: List of ZStackFrame objects
        use_bf: If True, use BF images; if False, use LAF images
        method: Focus measure method

    Returns:
        Tuple of (best_index, focus_scores_array)
    """
    scores = []
    for frame in frames:
        img = frame.bf_image if use_bf else frame.laf_image
        score = compute_focus_measure(img, method) if img is not None else 0.0
        scores.append(score)

    scores_arr = np.array(scores)
    best_idx = int(np.argmax(scores_arr))
    return best_idx, scores_arr


def adjust_z_positions_to_focus(
    frames: list[ZStackFrame],
    focus_z_um: float,
) -> list[ZStackFrame]:
    """Adjust Z positions in frames so that focus_z_um becomes 0.

    Args:
        frames: List of ZStackFrame objects
        focus_z_um: The Z position that should become the new zero

    Returns:
        New list of frames with adjusted Z positions
    """
    adjusted = []
    for frame in frames:
        adjusted.append(ZStackFrame(
            z_position_um=frame.z_position_um - focus_z_um,
            bf_image=frame.bf_image,
            laf_image=frame.laf_image,
        ))
    return adjusted
