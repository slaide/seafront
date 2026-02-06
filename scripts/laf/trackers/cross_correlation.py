"""Cross-correlation based Z estimation trackers."""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from data_types import CalibrationData, TrackerResult
from trackers.base import BaseTracker


class CrossCorrelationTracker(BaseTracker):
    """2D cross-correlation with reference templates."""

    name = "CC 2D"

    def __init__(self, calibration: CalibrationData, roi_size: int = 256, verbose: bool = False):
        super().__init__(calibration, roi_size, verbose)
        self.templates = self._prepare_templates()
        self.templates_flat = np.array([t.ravel() for t in self.templates])

    def _prepare_templates(self) -> list[np.ndarray]:
        """Prepare normalized templates from calibration images."""
        templates = []
        for img in self.calibration.reference_images:
            roi = self._extract_roi(img)
            roi_norm = roi.astype(np.float64)
            roi_norm = (roi_norm - roi_norm.mean()) / (roi_norm.std() + 1e-10)
            templates.append(roi_norm)
        return templates

    def estimate_z(self, image: np.ndarray) -> TrackerResult:
        roi = self._extract_roi(image)
        roi_norm = roi.astype(np.float64)
        roi_norm = (roi_norm - roi_norm.mean()) / (roi_norm.std() + 1e-10)
        roi_flat = roi_norm.ravel()

        correlations = self.templates_flat @ roi_flat / roi_flat.size
        best_idx = int(np.argmax(correlations))

        z_positions = self.calibration.z_positions
        z_estimate = self._interpolate_z(correlations, best_idx, z_positions)

        peak = correlations[best_idx]
        others = np.delete(correlations, best_idx)
        distinctiveness = (peak - others.max()) / (peak + 1e-10)
        confidence = float(np.clip(peak * (1 + distinctiveness), 0, 1))

        return TrackerResult(z_estimate, confidence, best_idx)

    def _interpolate_z(
        self, correlations: np.ndarray, best_idx: int, z_positions: np.ndarray
    ) -> float:
        """Parabolic interpolation for sub-step precision."""
        if best_idx > 0 and best_idx < len(correlations) - 1:
            c0, c1, c2 = correlations[best_idx - 1 : best_idx + 2]
            denom = 2 * (c0 - 2 * c1 + c2)
            if abs(denom) > 1e-10:
                offset = np.clip((c0 - c2) / denom, -0.5, 0.5)
            else:
                offset = 0
            dz = z_positions[1] - z_positions[0] if len(z_positions) > 1 else 1
            return float(z_positions[best_idx] + offset * dz)
        return float(z_positions[best_idx])

    def get_model_size_bytes(self) -> int:
        """Size = templates (N×H×W×8) + z_positions (N×8)."""
        n = len(self.templates)
        template_size = self.templates_flat.nbytes  # float64 templates
        z_size = n * 8  # float64 z positions
        return template_size + z_size


class CrossCorrelationSplineTracker(CrossCorrelationTracker):
    """2D cross-correlation with cubic spline interpolation for sub-step precision.

    Uses cubic spline fit through all correlation values to find the precise
    maximum, giving better accuracy than 3-point parabolic interpolation.
    """

    name = "CC 2D Spline"

    def _interpolate_z(
        self, correlations: np.ndarray, best_idx: int, z_positions: np.ndarray
    ) -> float:
        """Cubic spline interpolation through all correlation values."""
        from scipy.interpolate import CubicSpline
        from scipy.optimize import minimize_scalar

        # Fit cubic spline through all correlation values
        cs = CubicSpline(z_positions, correlations)

        # Find maximum near the best discrete index
        dz = abs(z_positions[1] - z_positions[0]) if len(z_positions) > 1 else 1
        z_best = z_positions[best_idx]
        # Search within ±1 step of the best discrete match
        z_min = max(z_positions[0], z_best - dz)
        z_max = min(z_positions[-1], z_best + dz)

        # Minimize negative of spline to find maximum
        result = minimize_scalar(lambda z: -cs(z), bounds=(z_min, z_max), method="bounded")
        return float(result.x)


class CrossCorrelationJPEGTracker(BaseTracker):
    """2D cross-correlation with JPEG-compressed reference templates.

    Templates are JPEG compressed during calibration to reduce storage size.
    During inference, templates are decompressed and used normally.
    """

    name = "CC 2D+JPEG"

    def __init__(self, calibration: CalibrationData, roi_size: int = 256, verbose: bool = False, jpeg_quality: int = 80):
        self.jpeg_quality = jpeg_quality
        super().__init__(calibration, roi_size, verbose)
        self.jpeg_templates = self._compress_templates()
        self.templates = self._decompress_templates()
        self.templates_flat = np.array([t.ravel() for t in self.templates])

    def _compress_templates(self) -> list[bytes]:
        """JPEG compress each calibration image ROI."""
        compressed = []
        for img in self.calibration.reference_images:
            roi = self._extract_roi(img)
            # Normalize to 8-bit for JPEG
            if roi.dtype == np.uint16:
                roi_8bit = (roi / 256).astype(np.uint8)
            else:
                roi_8bit = roi.astype(np.uint8)
            # JPEG encode
            _, encoded = cv2.imencode('.jpg', roi_8bit, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            compressed.append(encoded.tobytes())
        return compressed

    def _decompress_templates(self) -> list[np.ndarray]:
        """Decompress JPEG templates and normalize for correlation."""
        templates = []
        for jpeg_bytes in self.jpeg_templates:
            # Decode JPEG
            img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            # Normalize
            roi_norm = img.astype(np.float64)
            roi_norm = (roi_norm - roi_norm.mean()) / (roi_norm.std() + 1e-10)
            templates.append(roi_norm)
        return templates

    def estimate_z(self, image: np.ndarray) -> TrackerResult:
        roi = self._extract_roi(image)
        roi_norm = roi.astype(np.float64)
        roi_norm = (roi_norm - roi_norm.mean()) / (roi_norm.std() + 1e-10)
        roi_flat = roi_norm.ravel()

        correlations = self.templates_flat @ roi_flat / roi_flat.size
        best_idx = int(np.argmax(correlations))

        z_positions = self.calibration.z_positions
        z_estimate = self._interpolate_z(correlations, best_idx, z_positions)

        peak = correlations[best_idx]
        others = np.delete(correlations, best_idx)
        distinctiveness = (peak - others.max()) / (peak + 1e-10)
        confidence = float(np.clip(peak * (1 + distinctiveness), 0, 1))

        return TrackerResult(z_estimate, confidence, best_idx)

    def _interpolate_z(
        self, correlations: np.ndarray, best_idx: int, z_positions: np.ndarray
    ) -> float:
        """Parabolic interpolation for sub-step precision."""
        if best_idx > 0 and best_idx < len(correlations) - 1:
            c0, c1, c2 = correlations[best_idx - 1 : best_idx + 2]
            denom = 2 * (c0 - 2 * c1 + c2)
            if abs(denom) > 1e-10:
                offset = np.clip((c0 - c2) / denom, -0.5, 0.5)
            else:
                offset = 0
            dz = z_positions[1] - z_positions[0] if len(z_positions) > 1 else 1
            return float(z_positions[best_idx] + offset * dz)
        return float(z_positions[best_idx])

    def get_model_size_bytes(self) -> int:
        """Size = sum of JPEG bytes + z_positions (N×8)."""
        jpeg_size = sum(len(b) for b in self.jpeg_templates)
        z_size = len(self.jpeg_templates) * 8  # float64 z positions
        return jpeg_size + z_size


class CrossCorrelation1DTracker(BaseTracker):
    """1D cross-correlation using projected profiles (sum along Y)."""

    name = "CC 1D"

    def __init__(self, calibration: CalibrationData, roi_size: int = 256, verbose: bool = False):
        super().__init__(calibration, roi_size, verbose)
        self.templates_1d = self._prepare_templates()
        self.templates_stack = np.array(self.templates_1d)
        # PNG-compress raw profiles for compact storage
        self.compressed_profiles = self._compress_profiles()

    def _prepare_templates(self) -> list[np.ndarray]:
        """Prepare normalized 1D templates."""
        templates = []
        for img in self.calibration.reference_images:
            roi = self._extract_roi(img)
            profile = roi.astype(np.float64).sum(axis=0)
            profile = (profile - profile.mean()) / (profile.std() + 1e-10)
            templates.append(profile)
        return templates

    def _compress_profiles(self) -> bytes:
        """PNG-compress stacked 1D profiles for compact storage."""
        # Stack profiles into 2D array (N x W)
        raw_profiles = []
        for img in self.calibration.reference_images:
            roi = self._extract_roi(img)
            profile = roi.astype(np.float64).sum(axis=0)
            raw_profiles.append(profile)
        stacked = np.array(raw_profiles)

        # Normalize to 16-bit for lossless PNG compression
        stacked_min = stacked.min()
        stacked_max = stacked.max()
        if stacked_max > stacked_min:
            normalized = ((stacked - stacked_min) / (stacked_max - stacked_min) * 65535).astype(np.uint16)
        else:
            normalized = np.zeros_like(stacked, dtype=np.uint16)

        # PNG encode
        _, encoded = cv2.imencode('.png', normalized)
        return encoded.tobytes()

    def estimate_z(self, image: np.ndarray) -> TrackerResult:
        roi = self._extract_roi(image)
        profile = roi.astype(np.float64).sum(axis=0)
        profile = (profile - profile.mean()) / (profile.std() + 1e-10)

        correlations = self.templates_stack @ profile / profile.size
        best_idx = int(np.argmax(correlations))

        z_positions = self.calibration.z_positions
        z_estimate = self._interpolate_z(correlations, best_idx, z_positions)

        peak = correlations[best_idx]
        others = np.delete(correlations, best_idx)
        distinctiveness = (peak - others.max()) / (peak + 1e-10)
        confidence = float(np.clip(peak * (1 + distinctiveness), 0, 1))

        return TrackerResult(z_estimate, confidence, best_idx)

    def _interpolate_z(
        self, correlations: np.ndarray, best_idx: int, z_positions: np.ndarray
    ) -> float:
        if best_idx > 0 and best_idx < len(correlations) - 1:
            c0, c1, c2 = correlations[best_idx - 1 : best_idx + 2]
            denom = 2 * (c0 - 2 * c1 + c2)
            if abs(denom) > 1e-10:
                offset = np.clip((c0 - c2) / denom, -0.5, 0.5)
            else:
                offset = 0
            dz = z_positions[1] - z_positions[0] if len(z_positions) > 1 else 1
            return float(z_positions[best_idx] + offset * dz)
        return float(z_positions[best_idx])

    def get_model_size_bytes(self) -> int:
        """Size = PNG-compressed profiles + z_positions (N×8)."""
        png_size = len(self.compressed_profiles)
        z_size = len(self.templates_1d) * 8
        return png_size + z_size
