"""SQUID-style Z estimation trackers."""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress

from data_types import CalibrationData, TrackerResult
from trackers.base import BaseTracker


class SquidTracker(BaseTracker):
    """
    SQUID-style tracker: uses max projection + find_peaks to track a selected peak.

    This matches the actual SQUID microscope hardware implementation:
    - Max projection along axis 0
    - find_peaks with distance=300, height=10 (same as seafront/hardware/squid.py)
    - Take top 2 peaks by height, sort by x position
    - use_right_dot selects whether to track the left or right peak
    """

    name = "★SQUIDL"  # Left peak (default in seafront)
    sigma = 0  # Images already have 2D blur from server (cv2.GaussianBlur in squid.py:1782)
    use_right_dot = False  # Match seafront: False = left peak, True = right peak

    def __init__(self, calibration: CalibrationData, roi_size: int = 256, verbose: bool = False):
        super().__init__(calibration, roi_size, verbose)
        self.peak_positions = self._measure_peak_positions()
        self._fit_linear_regression()

    def _find_peak(self, img: np.ndarray) -> float:
        """Find the selected peak (left or right) using SQUID's exact method.

        Matches seafront/hardware/squid.py _get_peak_coords exactly:
        1. Apply 2D Gaussian blur (3x3, sigma=1.0)
        2. Max projection along axis 0
        3. find_peaks with distance=300, height=10
        4. Take top 2 peaks by height
        5. Sort by x position
        6. Return rightmost or leftmost based on use_right_dot
        """
        roi = self._extract_roi(img)

        # Validate image size - must match real SQUID LAF camera (3088 x 2064)
        if roi.shape[1] != 3088:
            raise ValueError(
                f"LAF image width must be 3088px (got {roi.shape[1]}px). "
                f"Mock images must use real camera resolution."
            )

        # Apply 2D Gaussian blur before max projection (matching seafront/hardware/squid.py _get_peak_coords)
        roi_blurred = cv2.GaussianBlur(roi, (3, 3), sigmaX=1.0, borderType=cv2.BORDER_DEFAULT)

        # Use MAX projection along axis 0 (SQUID uses this)
        profile = roi_blurred.astype(np.float64).max(axis=0)

        # Optional additional 1D smoothing (sigma=0 means no extra smoothing, matching seafront)
        if self.sigma > 0:
            from scipy.ndimage import gaussian_filter1d
            profile = gaussian_filter1d(profile, sigma=self.sigma)

        # SQUID uses distance=300, height=10 (exact values from seafront/hardware/squid.py:652)
        peak_locations, _ = find_peaks(profile, distance=300, height=10)

        if len(peak_locations) == 0:
            # Fallback: find max in appropriate half
            mid = len(profile) // 2
            if self.use_right_dot:
                return float(mid + np.argmax(profile[mid:]))
            else:
                return float(np.argmax(profile[:mid]))

        # SQUID approach: order by height, pick top 2, then sort by x
        heights = profile[peak_locations]
        tallest_indices = np.argsort(heights)[-2:]  # Top 2 by height
        top_peaks = sorted(peak_locations[tallest_indices])  # Sort by x coordinate

        rightmost_x = float(top_peaks[-1])
        leftmost_x = float(top_peaks[0]) if len(top_peaks) >= 2 else rightmost_x

        return rightmost_x if self.use_right_dot else leftmost_x

    def _measure_peak_positions(self) -> np.ndarray:
        """Measure peak position for each calibration image."""
        positions = []
        for img in self.calibration.reference_images:
            pos = self._find_peak(img)
            positions.append(pos)
        return np.array(positions)

    def _fit_linear_regression(self) -> None:
        """Fit linear regression with intercept adjusted so error at z=0 is zero.

        We fit: dot_x = slope * z + intercept
        Then invert: z = (dot_x - x_reference) / slope

        The adjustment: instead of using the regression intercept, we interpolate
        the actual peak position at z=0 and use that as x_reference. This ensures
        that at z=0, z_estimate = 0 exactly (zero error at focus).
        """
        from scipy.interpolate import interp1d

        # Fit regression to get slope (sensitivity)
        result = linregress(self.calibration.z_positions, self.peak_positions)
        self.um_per_px = result.slope  # pixels per um

        # Interpolate peak position at z=0 for zero error at focus
        z_positions = self.calibration.z_positions
        peak_positions = self.peak_positions

        # Sort by z for interpolation
        sort_idx = np.argsort(z_positions)
        z_sorted = z_positions[sort_idx]
        peak_sorted = peak_positions[sort_idx]

        # Check if z=0 is within calibration range
        if z_sorted[0] <= 0 <= z_sorted[-1]:
            # Interpolate peak position at z=0
            f = interp1d(z_sorted, peak_sorted, kind='linear')
            self.x_reference = float(f(0.0))
        else:
            # z=0 is outside calibration range, fall back to regression intercept
            self.x_reference = result.intercept

        # Debug output for both SQUIDL and SQUIDR
        if self.name in ("★SQUIDL", "★SQUIDR", "SQUID+σ2"):
            print(f"\n    [DEBUG {self.name}] Calibration regression:")
            print(f"      Z range: {self.calibration.z_positions.min():.1f} to {self.calibration.z_positions.max():.1f} um")
            print(f"      Peak range: {self.peak_positions.min():.1f} to {self.peak_positions.max():.1f} px")
            print(f"      Sensitivity: {abs(self.um_per_px):.3f} px/um ({abs(1/self.um_per_px):.3f} um/px)")
            print(f"      um_per_px (slope): {self.um_per_px:.6f} px/um")
            print(f"      x_reference (adjusted): {self.x_reference:.1f} px (regression: {result.intercept:.1f} px)")
            print(f"      R²: {result.rvalue**2:.6f}")

    def estimate_z(self, image: np.ndarray) -> TrackerResult:
        peak_x = self._find_peak(image)

        # Seafront formula: z = (dot_x - x_reference) / um_per_px
        z_estimate = float((peak_x - self.x_reference) / self.um_per_px) if abs(self.um_per_px) > 1e-10 else 0.0

        roi = self._extract_roi(image)
        signal = roi.max() - roi.min()
        confidence = float(np.clip(signal / 1000, 0, 1))

        return TrackerResult(z_estimate, confidence, 0)


class SquidTrackerRight(SquidTracker):
    """SQUID tracker using the RIGHT peak (use_right_dot=True)."""
    name = "★SQUIDR"
    use_right_dot = True


# =============================================================================
# SQUID LEFT variants with different Gaussian smoothing sigma values
# =============================================================================

class SquidTrackerL_Sigma0(SquidTracker):
    """SQUID LEFT tracker with no additional Gaussian smoothing."""
    name = "SQUIDL+σ0"
    sigma = 0


class SquidTrackerL_Sigma2(SquidTracker):
    """SQUID LEFT tracker with sigma=2 Gaussian smoothing."""
    name = "SQUIDL+σ2"
    sigma = 2


class SquidTrackerL_Sigma4(SquidTracker):
    """SQUID LEFT tracker with sigma=4 Gaussian smoothing."""
    name = "SQUIDL+σ4"
    sigma = 4


class SquidTrackerL_Sigma10(SquidTracker):
    """SQUID LEFT tracker with sigma=10 Gaussian smoothing."""
    name = "SQUIDL+σ10"
    sigma = 10


class SquidTrackerL_Sigma20(SquidTracker):
    """SQUID LEFT tracker with sigma=20 Gaussian smoothing."""
    name = "SQUIDL+σ20"
    sigma = 20


class SquidTrackerL_Sigma50(SquidTracker):
    """SQUID LEFT tracker with sigma=50 Gaussian smoothing."""
    name = "SQUIDL+σ50"
    sigma = 50


# =============================================================================
# SQUID RIGHT variants with different Gaussian smoothing sigma values
# =============================================================================

class SquidTrackerR_Sigma0(SquidTrackerRight):
    """SQUID RIGHT tracker with no additional Gaussian smoothing."""
    name = "SQUIDR+σ0"
    sigma = 0


class SquidTrackerR_Sigma2(SquidTrackerRight):
    """SQUID RIGHT tracker with sigma=2 Gaussian smoothing."""
    name = "SQUIDR+σ2"
    sigma = 2


class SquidTrackerR_Sigma4(SquidTrackerRight):
    """SQUID RIGHT tracker with sigma=4 Gaussian smoothing."""
    name = "SQUIDR+σ4"
    sigma = 4


class SquidTrackerR_Sigma10(SquidTrackerRight):
    """SQUID RIGHT tracker with sigma=10 Gaussian smoothing."""
    name = "SQUIDR+σ10"
    sigma = 10


class SquidTrackerR_Sigma20(SquidTrackerRight):
    """SQUID RIGHT tracker with sigma=20 Gaussian smoothing."""
    name = "SQUIDR+σ20"
    sigma = 20


class SquidTrackerR_Sigma50(SquidTrackerRight):
    """SQUID RIGHT tracker with sigma=50 Gaussian smoothing."""
    name = "SQUIDR+σ50"
    sigma = 50
