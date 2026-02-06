"""Peak-based Z estimation trackers (centroid, gaussian fit, etc.)."""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import linregress

from data_types import CalibrationData, TrackerResult
from trackers.base import BaseTracker


class Peak1DTracker(BaseTracker):
    """Track Z by measuring LEFT peak position in 1D profile (sum projection)."""

    name = "Peak 1D"

    def __init__(self, calibration: CalibrationData, roi_size: int = 256, verbose: bool = False):
        super().__init__(calibration, roi_size, verbose)
        self.peak_positions = self._measure_peak_positions()
        self._fit_spline()

    def _find_left_peak_centroid(self, img: np.ndarray) -> float:
        """Find centroid of left peak using sum projection with smoothing."""
        roi = self._extract_roi(img)
        profile = roi.astype(np.float64).sum(axis=0)
        profile_smooth = gaussian_filter1d(profile, sigma=2)

        # Find peaks to identify the main dots
        height_thresh = profile_smooth.min() + (profile_smooth.max() - profile_smooth.min()) * 0.3
        peaks, props = find_peaks(profile_smooth, distance=25, height=height_thresh)

        if len(peaks) >= 2:
            # Get top 2 peaks by height
            heights = props['peak_heights']
            top_indices = np.argsort(heights)[-2:]
            top_peaks = sorted(peaks[top_indices])
            # Use leftmost peak position
            return float(top_peaks[0])
        elif len(peaks) == 1:
            return float(peaks[0])
        else:
            # Fallback: find max in left half
            mid = len(profile_smooth) // 2
            return float(np.argmax(profile_smooth[:mid]))

    def _measure_peak_positions(self) -> np.ndarray:
        """Measure left peak position for each calibration image."""
        positions = []
        for img in self.calibration.reference_images:
            pos = self._find_left_peak_centroid(img)
            positions.append(pos)
        return np.array(positions)

    def _fit_spline(self) -> None:
        """Fit linear regression mapping peak position to Z."""
        result = linregress(self.peak_positions, self.calibration.z_positions)
        self.slope = result.slope
        self.intercept = result.intercept

    def estimate_z(self, image: np.ndarray) -> TrackerResult:
        peak_pos = self._find_left_peak_centroid(image)

        # Linear model
        z_estimate = float(self.slope * peak_pos + self.intercept)

        # Confidence based on signal strength
        roi = self._extract_roi(image)
        profile = roi.astype(np.float64).sum(axis=0)
        signal = profile.max() - profile.min()
        noise = np.std(profile[:20])  # Estimate noise from edge
        confidence = float(np.clip(signal / (noise + 1e-10) / 50, 0, 1))

        return TrackerResult(z_estimate, confidence, 0)


class GaussFit1DTracker(BaseTracker):
    """Track Z by fitting Gaussian to LEFT peak in 1D profile."""

    name = "Gauss 1D"

    def __init__(self, calibration: CalibrationData, roi_size: int = 256, verbose: bool = False):
        super().__init__(calibration, roi_size, verbose)
        self.peak_positions = self._measure_peak_positions()
        self._fit_spline()

    def _fit_gaussian_to_left_peak(self, img: np.ndarray) -> float:
        """Fit Gaussian to LEFT peak and return center position."""
        from scipy.optimize import curve_fit

        def gaussian(x, amp, mu, sigma, offset):
            return amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + offset

        roi = self._extract_roi(img)
        profile = roi.astype(np.float64).sum(axis=0)
        profile_smooth = gaussian_filter1d(profile, sigma=2)

        # Find peaks to identify the left dot
        height_thresh = profile_smooth.min() + (profile_smooth.max() - profile_smooth.min()) * 0.3
        peaks, props = find_peaks(profile_smooth, distance=25, height=height_thresh)

        if len(peaks) >= 2:
            heights = props['peak_heights']
            top_indices = np.argsort(heights)[-2:]
            top_peaks = sorted(peaks[top_indices])
            peak_pos = top_peaks[0]  # Leftmost of top 2
        elif len(peaks) == 1:
            peak_pos = peaks[0]
        else:
            mid = len(profile_smooth) // 2
            peak_pos = np.argmax(profile_smooth[:mid])

        # Extract region around peak for Gaussian fit
        window = 30
        start = max(0, peak_pos - window)
        end = min(len(profile), peak_pos + window)
        region = profile[start:end]
        x = np.arange(len(region))

        try:
            amp0 = region.max() - region.min()
            mu0 = float(np.argmax(region))
            sigma0 = 8.0
            offset0 = region.min()

            popt, _ = curve_fit(
                gaussian,
                x,
                region,
                p0=[amp0, mu0, sigma0, offset0],
                bounds=([0, 0, 1, 0], [np.inf, len(region), len(region) / 2, np.inf]),
                maxfev=1000,
            )
            return start + popt[1]  # Convert to full profile coordinates
        except Exception:
            return float(peak_pos)

    def _measure_peak_positions(self) -> np.ndarray:
        """Measure Gaussian-fitted left peak position for each calibration image."""
        positions = []
        for img in self.calibration.reference_images:
            pos = self._fit_gaussian_to_left_peak(img)
            positions.append(pos)
        return np.array(positions)

    def _fit_spline(self) -> None:
        """Fit linear regression (more robust than spline for potentially non-monotonic data)."""
        # Use linear regression instead of spline - more robust
        result = linregress(self.peak_positions, self.calibration.z_positions)
        self.slope = result.slope
        self.intercept = result.intercept
        self.peak_min = self.peak_positions.min()
        self.peak_max = self.peak_positions.max()

    def estimate_z(self, image: np.ndarray) -> TrackerResult:
        peak_pos = self._fit_gaussian_to_left_peak(image)

        # Linear model
        z_estimate = float(self.slope * peak_pos + self.intercept)

        roi = self._extract_roi(image)
        profile = roi.astype(np.float64).sum(axis=0)
        signal = profile.max() - profile.min()
        noise = np.std(profile[:20])
        confidence = float(np.clip(signal / (noise + 1e-10) / 50, 0, 1))

        return TrackerResult(z_estimate, confidence, 0)


class CentroidTracker(BaseTracker):
    """Track Z by measuring centroid of the LEFT dot (more stable, like SQUID)."""

    name = "Centroid"

    def __init__(self, calibration: CalibrationData, roi_size: int = 256, verbose: bool = False):
        super().__init__(calibration, roi_size, verbose)
        self.centroid_positions = self._measure_centroids()
        self._fit_spline()

    def _find_left_dot_centroid(self, img: np.ndarray) -> float:
        """Find centroid of the LEFT dot using max projection with smoothing."""
        roi = self._extract_roi(img)
        # Use max projection (like SQUID) and smooth
        profile = roi.astype(np.float64).max(axis=0)
        profile_smooth = gaussian_filter1d(profile, sigma=2)

        # Find peaks to identify the main dots
        height_thresh = profile_smooth.min() + (profile_smooth.max() - profile_smooth.min()) * 0.3
        peaks, props = find_peaks(profile_smooth, distance=25, height=height_thresh)

        if len(peaks) >= 2:
            # Get top 2 peaks by height
            heights = props['peak_heights']
            top_indices = np.argsort(heights)[-2:]
            top_peaks = sorted(peaks[top_indices])
            # Use leftmost peak position
            return float(top_peaks[0])
        elif len(peaks) == 1:
            return float(peaks[0])
        else:
            # Fallback: find max in left half
            mid = len(profile_smooth) // 2
            return float(np.argmax(profile_smooth[:mid]))

    def _measure_centroids(self) -> np.ndarray:
        """Measure left dot centroid for each calibration image."""
        positions = []
        for img in self.calibration.reference_images:
            pos = self._find_left_dot_centroid(img)
            positions.append(pos)
        return np.array(positions)

    def _fit_spline(self) -> None:
        """Fit linear regression mapping centroid position to Z."""
        result = linregress(self.centroid_positions, self.calibration.z_positions)
        self.slope = result.slope
        self.intercept = result.intercept

        # Diagnostic: show µm/pixel ratio
        pixel_range = self.centroid_positions.max() - self.centroid_positions.min()
        z_range = self.calibration.z_positions.max() - self.calibration.z_positions.min()
        self.um_per_pixel = abs(self.slope)
        print(f"\n    [Centroid] Z sensitivity: {self.um_per_pixel:.2f} µm/pixel")
        print(f"    [Centroid] Pixel range: {pixel_range:.1f} px over {z_range:.0f} µm Z range")

    def estimate_z(self, image: np.ndarray) -> TrackerResult:
        centroid = self._find_left_dot_centroid(image)

        # Linear model
        z_estimate = float(self.slope * centroid + self.intercept)

        roi = self._extract_roi(image)
        signal = roi.max() - roi.min()
        confidence = float(np.clip(signal / 1000, 0, 1))

        return TrackerResult(z_estimate, confidence, 0)


class Centroid2DTracker(BaseTracker):
    """
    2D lookup tracker using centroid position + total brightness.

    Maps each calibration image to 2D coordinates (centroid_x, brightness),
    then interpolates Z based on proximity in this 2D space.
    """

    name = "Centroid 2D"

    def __init__(self, calibration: CalibrationData, roi_size: int = 256, verbose: bool = False):
        super().__init__(calibration, roi_size, verbose)
        self.cal_points, self.cal_z = self._build_calibration_map()

    def _measure_features(self, img: np.ndarray) -> tuple[float, float]:
        """Extract centroid position and total brightness from image."""
        roi = self._extract_roi(img)
        # Total brightness (normalized by image size for consistency)
        brightness = float(roi.astype(np.float64).sum() / roi.size)

        # Centroid using intensity-weighted position along x-axis
        profile = roi.astype(np.float64).sum(axis=0)
        profile_smooth = gaussian_filter1d(profile, sigma=2)

        # Intensity-weighted centroid
        x_coords = np.arange(len(profile_smooth))
        total = profile_smooth.sum()
        if total > 0:
            centroid_x = float((x_coords * profile_smooth).sum() / total)
        else:
            centroid_x = len(profile_smooth) / 2

        return centroid_x, brightness

    def _build_calibration_map(self) -> tuple[np.ndarray, np.ndarray]:
        """Build 2D calibration points (centroid, brightness) -> Z mapping."""
        points = []
        z_values = []
        for img, z in zip(self.calibration.reference_images, self.calibration.z_positions):
            centroid_x, brightness = self._measure_features(img)
            points.append([centroid_x, brightness])
            z_values.append(z)
        return np.array(points), np.array(z_values)

    def estimate_z(self, image: np.ndarray) -> TrackerResult:
        """Estimate Z by finding closest calibration points in 2D feature space."""
        centroid_x, brightness = self._measure_features(image)
        query = np.array([centroid_x, brightness])

        # Normalize features for distance calculation
        # (centroid range ~hundreds of pixels, brightness range varies)
        cal_std = self.cal_points.std(axis=0)
        cal_std[cal_std < 1e-10] = 1  # Avoid division by zero
        cal_mean = self.cal_points.mean(axis=0)

        cal_norm = (self.cal_points - cal_mean) / cal_std
        query_norm = (query - cal_mean) / cal_std

        # Find distances to all calibration points
        distances = np.linalg.norm(cal_norm - query_norm, axis=1)

        # Weighted average of k nearest neighbors
        k = min(3, len(distances))
        nearest_idx = np.argsort(distances)[:k]
        nearest_dist = distances[nearest_idx]

        # Inverse distance weighting (with small epsilon to avoid div by zero)
        weights = 1.0 / (nearest_dist + 1e-6)
        weights /= weights.sum()

        z_estimate = float((weights * self.cal_z[nearest_idx]).sum())

        # Confidence based on how close the nearest match is
        min_dist = distances.min()
        confidence = float(np.clip(1.0 / (1.0 + min_dist), 0, 1))

        best_idx = int(nearest_idx[0])
        return TrackerResult(z_estimate, confidence, best_idx)

    def get_model_size_bytes(self) -> int:
        """Size = calibration points (N×2×8) + z_values (N×8)."""
        n = len(self.cal_z)
        return n * 2 * 8 + n * 8  # 2D points + Z values
