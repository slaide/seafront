"""Base tracker class and simple baseline trackers."""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from data_types import CalibrationData, TrackerResult


class BaseTracker:
    """Base class for Z estimation trackers."""

    name: str = "BaseTracker"

    def __init__(self, calibration: CalibrationData, roi_size: int = 256, verbose: bool = False):
        self.calibration = calibration
        self.roi_size = roi_size  # Kept for compatibility but not used for cropping
        self.verbose = verbose

    def _extract_roi(self, img: np.ndarray) -> np.ndarray:
        """Return full image (no ROI extraction - using whole image for better tracking)."""
        return img

    def estimate_z(self, image: np.ndarray) -> TrackerResult:
        """Estimate Z position from image. Override in subclasses."""
        raise NotImplementedError

    def get_model_size_bytes(self) -> int:
        """Return size in bytes of data needed to store/reload this model."""
        # Default: just slope + intercept (16 bytes for 2 float64)
        return 16


class ZeroPredictor(BaseTracker):
    """Baseline model that always predicts z=0. Worst-case reference."""

    name = "Zero"

    def __init__(self, calibration: CalibrationData, roi_size: int = 256, verbose: bool = False):
        super().__init__(calibration, roi_size, verbose)

    def estimate_z(self, image: np.ndarray) -> TrackerResult:
        return TrackerResult(z_estimate=0.0, confidence=0.0, best_idx=0)

    def get_model_size_bytes(self) -> int:
        return 0
