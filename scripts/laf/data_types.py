"""Data structures for LAF Z-stack benchmarking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ZStackFrame:
    """Single frame in the z-stack with BF and LAF images."""

    z_position_um: float
    bf_image: np.ndarray | None
    laf_image: np.ndarray | None


@dataclass
class CalibrationData:
    """Stores calibration information for the autofocus system."""

    z_positions: np.ndarray
    reference_images: list[np.ndarray]
    separation_distance: float = 60.0

    @classmethod
    def from_frames(
        cls, frames: list[ZStackFrame], separation_distance: float = 60.0
    ) -> "CalibrationData":
        """Create calibration data from captured frames."""
        z_positions = np.array([f.z_position_um for f in frames if f.laf_image is not None])
        images = [f.laf_image for f in frames if f.laf_image is not None]
        return cls(
            z_positions=z_positions,
            reference_images=images,
            separation_distance=separation_distance,
        )

    @classmethod
    def from_multiple_stacks(
        cls, stacks: list[list[ZStackFrame]], separation_distance: float = 60.0
    ) -> "CalibrationData":
        """Create calibration data by averaging images from multiple stacks.

        Each stack should have the same Z positions. Images at matching Z levels
        are averaged to create more robust templates.
        """
        if not stacks:
            raise ValueError("No stacks provided")

        # Get Z positions from first stack
        first_stack = [f for f in stacks[0] if f.laf_image is not None]
        z_positions = np.array([f.z_position_um for f in first_stack])

        # Collect images at each Z level from all stacks
        averaged_images = []
        for i, z in enumerate(z_positions):
            images_at_z = []
            for stack in stacks:
                valid_frames = [f for f in stack if f.laf_image is not None]
                if i < len(valid_frames):
                    images_at_z.append(valid_frames[i].laf_image.astype(np.float64))

            if images_at_z:
                # Average images at this Z level
                avg_img = np.mean(images_at_z, axis=0).astype(images_at_z[0].dtype)
                averaged_images.append(avg_img)

        return cls(
            z_positions=z_positions[:len(averaged_images)],
            reference_images=averaged_images,
            separation_distance=separation_distance,
        )

    @classmethod
    def from_multiple_stacks_concatenated(
        cls, stacks: list[list[ZStackFrame]], separation_distance: float = 60.0
    ) -> "CalibrationData":
        """Create calibration data by concatenating all images from multiple stacks.

        Used for CNN training where we want more training examples, not averaged templates.
        Each stack should have the same Z positions. All images are included with their
        Z positions repeated for each stack.
        """
        if not stacks:
            raise ValueError("No stacks provided")

        all_z_positions = []
        all_images = []

        for stack in stacks:
            for frame in stack:
                if frame.laf_image is not None:
                    all_z_positions.append(frame.z_position_um)
                    all_images.append(frame.laf_image)

        return cls(
            z_positions=np.array(all_z_positions),
            reference_images=all_images,
            separation_distance=separation_distance,
        )


@dataclass
class TrackerResult:
    """Result from a Z estimation tracker."""

    z_estimate: float
    confidence: float
    best_idx: int


@dataclass
class BenchmarkResult:
    """Results from benchmarking a single tracker."""

    name: str
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    max_error: float
    narrow_mae: float  # MAE for |z| <= 15um (typical autofocus range)
    super_narrow_mae: float  # MAE for |z| <= 5um (refinement range)
    errors: np.ndarray
    estimates: np.ndarray
    actual: np.ndarray
    tracker: object | None = None  # Store tracker for additional metrics
    model_size_bytes: int = 0  # Size of model data needed to reload
    calibration_time_ms: float = 0.0  # Time to train/calibrate
    inference_time_ms: float = 0.0  # Time for single estimate


@dataclass
class SiteResult:
    """Results for a single site in multi-site benchmark."""

    site_idx: int
    x_mm: float
    y_mm: float
    z_mm: float
    is_calibration_site: bool
    mae: float
    narrow_mae: float
    super_narrow_mae: float  # MAE for |z| <= 5um
    rmse: float
    max_error: float
    errors: np.ndarray
    estimates: np.ndarray
    actual: np.ndarray


@dataclass
class MultiSiteResult:
    """Results from multi-site benchmarking of a single tracker."""

    name: str
    model_size_bytes: int
    calibration_time_ms: float
    inference_time_ms: float  # Average across all sites
    site_results: list[SiteResult]

    # Aggregate metrics
    @property
    def avg_mae(self) -> float:
        return float(np.mean([s.mae for s in self.site_results]))

    @property
    def avg_narrow_mae(self) -> float:
        vals = [s.narrow_mae for s in self.site_results if not np.isnan(s.narrow_mae)]
        return float(np.mean(vals)) if vals else float('nan')

    @property
    def avg_super_narrow_mae(self) -> float:
        vals = [s.super_narrow_mae for s in self.site_results if not np.isnan(s.super_narrow_mae)]
        return float(np.mean(vals)) if vals else float('nan')

    @property
    def avg_rmse(self) -> float:
        return float(np.mean([s.rmse for s in self.site_results]))

    @property
    def avg_max_error(self) -> float:
        return float(np.mean([s.max_error for s in self.site_results]))

    @property
    def cal_site_mae(self) -> float:
        cal = [s for s in self.site_results if s.is_calibration_site]
        return cal[0].mae if cal else float('nan')

    @property
    def cal_site_narrow_mae(self) -> float:
        cal = [s for s in self.site_results if s.is_calibration_site]
        return cal[0].narrow_mae if cal else float('nan')

    @property
    def other_sites_mae(self) -> float:
        others = [s.mae for s in self.site_results if not s.is_calibration_site]
        return float(np.mean(others)) if others else float('nan')

    @property
    def other_sites_narrow_mae(self) -> float:
        others = [s.narrow_mae for s in self.site_results if not s.is_calibration_site and not np.isnan(s.narrow_mae)]
        return float(np.mean(others)) if others else float('nan')


@dataclass
class BenchmarkParams:
    """Parameters used for the benchmark run."""
    cal_range_um: float
    cal_steps: int
    test_range_um: float
    test_steps: int
    laf_exposure_ms: float
    laf_gain: float
    separation_distance: float
    n_positions: int
    cal_all_sites: bool
    autofocus_cal: bool = False
    fine_range_um: float = 15.0
    fine_step_um: float = 1.0
    coarse_step_um: float = 14.0
