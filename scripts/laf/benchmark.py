"""Benchmarking functions for LAF Z-stack trackers."""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

from data_types import BenchmarkResult, CalibrationData, ZStackFrame
from trackers.base import BaseTracker, ZeroPredictor
from trackers.cnn import CNNTracker
from trackers.cross_correlation import (
    CrossCorrelation1DTracker,
    CrossCorrelationJPEGTracker,
    CrossCorrelationSplineTracker,
    CrossCorrelationTracker,
)
from trackers.peak_based import (
    Centroid2DTracker,
    CentroidTracker,
    GaussFit1DTracker,
    Peak1DTracker,
)
from trackers.squid import (
    SquidTracker,
    SquidTrackerRight,
    SquidTrackerL_Sigma0,
    SquidTrackerL_Sigma2,
    SquidTrackerL_Sigma4,
    SquidTrackerL_Sigma10,
    SquidTrackerL_Sigma20,
    SquidTrackerL_Sigma50,
    SquidTrackerR_Sigma0,
    SquidTrackerR_Sigma2,
    SquidTrackerR_Sigma4,
    SquidTrackerR_Sigma10,
    SquidTrackerR_Sigma20,
    SquidTrackerR_Sigma50,
)


def get_tracker_classes() -> list[type[BaseTracker]]:
    """Get list of all tracker classes to benchmark."""
    tracker_classes: list[type[BaseTracker]] = [
        ZeroPredictor,
        # SQUID base (no extra smoothing)
        SquidTracker,  # SQUID left peak (default in seafront)
        SquidTrackerRight,  # SQUID right peak
        # SQUID LEFT with various sigma
        SquidTrackerL_Sigma0,
        SquidTrackerL_Sigma2,
        SquidTrackerL_Sigma4,
        SquidTrackerL_Sigma10,
        SquidTrackerL_Sigma20,
        SquidTrackerL_Sigma50,
        # SQUID RIGHT with various sigma
        SquidTrackerR_Sigma0,
        SquidTrackerR_Sigma2,
        SquidTrackerR_Sigma4,
        SquidTrackerR_Sigma10,
        SquidTrackerR_Sigma20,
        SquidTrackerR_Sigma50,
        # Cross-correlation methods
        CrossCorrelationTracker,
        CrossCorrelationSplineTracker,  # CC 2D with cubic spline interpolation
        CrossCorrelationJPEGTracker,
        CrossCorrelation1DTracker,
        # Peak-based methods
        Peak1DTracker,
        GaussFit1DTracker,
        CentroidTracker,
        Centroid2DTracker,
    ]

    # Add CNN tracker if PyTorch is available
    try:
        import torch  # noqa: F401
        tracker_classes.append(CNNTracker)
    except ImportError:
        pass

    return tracker_classes


def benchmark_tracker(
    tracker_class: type[BaseTracker],
    cal_frames: list[ZStackFrame] | list[list[ZStackFrame]],
    test_frames: list[ZStackFrame],
    separation_distance: float = 60.0,
) -> BenchmarkResult:
    """Benchmark a tracker: train on cal_frames, test on test_frames.

    cal_frames can be either:
    - A single list of ZStackFrames (single-position calibration)
    - A list of lists of ZStackFrames (multi-position calibration, images averaged)
    """
    # Create calibration data from calibration frames
    if cal_frames and isinstance(cal_frames[0], list):
        # Multi-position: list of stacks
        calibration = CalibrationData.from_multiple_stacks(cal_frames, separation_distance)  # type: ignore
    else:
        # Single-position: single stack
        calibration = CalibrationData.from_frames(cal_frames, separation_distance)  # type: ignore

    # Time calibration/training
    t0 = time.perf_counter()
    tracker = tracker_class(calibration, verbose=False)
    calibration_time_ms = (time.perf_counter() - t0) * 1000

    # Test on test frames
    valid_test = [f for f in test_frames if f.laf_image is not None]

    errors = []
    estimates = []
    actual = []
    inference_times = []

    for frame in valid_test:
        t0 = time.perf_counter()
        result = tracker.estimate_z(frame.laf_image)  # type: ignore
        inference_times.append((time.perf_counter() - t0) * 1000)
        error = result.z_estimate - frame.z_position_um
        errors.append(error)
        estimates.append(result.z_estimate)
        actual.append(frame.z_position_um)

    # Average inference time
    inference_time_ms = sum(inference_times) / len(inference_times) if inference_times else 0

    errors_arr = np.array(errors)
    estimates_arr = np.array(estimates)
    actual_arr = np.array(actual)

    mae = float(np.mean(np.abs(errors_arr)))
    rmse = float(np.sqrt(np.mean(errors_arr**2)))
    max_error = float(np.max(np.abs(errors_arr)))

    # Narrow-range MAE (|z| <= 15um) - typical autofocus correction range
    narrow_mask = np.abs(actual_arr) <= 15.0
    if np.any(narrow_mask):
        narrow_mae = float(np.mean(np.abs(errors_arr[narrow_mask])))
    else:
        narrow_mae = float('nan')

    # Super narrow-range MAE (|z| <= 5um) - refinement range
    super_narrow_mask = np.abs(actual_arr) <= 5.0
    if np.any(super_narrow_mask):
        super_narrow_mae = float(np.mean(np.abs(errors_arr[super_narrow_mask])))
    else:
        super_narrow_mae = float('nan')

    model_size = tracker.get_model_size_bytes()

    return BenchmarkResult(
        name=tracker_class.name,
        mae=mae,
        rmse=rmse,
        max_error=max_error,
        narrow_mae=narrow_mae,
        super_narrow_mae=super_narrow_mae,
        errors=errors_arr,
        estimates=estimates_arr,
        actual=actual_arr,
        tracker=tracker,
        model_size_bytes=model_size,
        calibration_time_ms=calibration_time_ms,
        inference_time_ms=inference_time_ms,
    )


def run_all_benchmarks(
    cal_frames: list[ZStackFrame] | list[list[ZStackFrame]],
    test_frames: list[ZStackFrame],
    separation_distance: float = 60.0,
    verbose: bool = False,
) -> list[BenchmarkResult]:
    """Run benchmarks for all tracker types.

    cal_frames can be either:
    - A single list of ZStackFrames (single-position calibration)
    - A list of lists of ZStackFrames (multi-position calibration, images averaged)
    """
    tracker_classes = get_tracker_classes()

    # Check if CNN is available
    has_cnn = any(tc.name == "CNN" for tc in tracker_classes)
    if not has_cnn:
        print("  [Note: CNN tracker skipped - PyTorch not installed (uv pip install torch)]")

    # Determine if multi-position calibration
    is_multipos = cal_frames and isinstance(cal_frames[0], list)
    if is_multipos:
        num_positions = len(cal_frames)
        first_stack = cal_frames[0]  # type: ignore
        print(f"  Multi-position calibration: {num_positions} positions, averaging templates")
    else:
        first_stack = cal_frames  # type: ignore

    if verbose:
        # Show calibration data summary
        print("\n  Calibration summary:")
        for frame in [first_stack[0], first_stack[len(first_stack)//2], first_stack[-1]]:
            if frame.laf_image is not None:
                img = frame.laf_image
                print(f"    Z={frame.z_position_um:+.0f}um: image range [{img.min()}-{img.max()}]")

    results = []
    for tracker_class in tracker_classes:
        print(f"  Benchmarking {tracker_class.name}...")
        try:
            result = benchmark_tracker(tracker_class, cal_frames, test_frames, separation_distance)
            results.append(result)

            # Format model size
            size = result.model_size_bytes
            if size >= 1024 * 1024:
                size_str = f"{size / 1024 / 1024:.1f}MB"
            elif size >= 1024:
                size_str = f"{size / 1024:.1f}KB"
            else:
                size_str = f"{size}B"

            # Format times
            cal_time = result.calibration_time_ms
            inf_time = result.inference_time_ms
            if cal_time >= 1000:
                cal_str = f"{cal_time/1000:.2f}s"
            else:
                cal_str = f"{cal_time:.1f}ms"
            inf_str = f"{inf_time:.2f}ms"

            # Show overfitting info for CNN
            if result.name == "CNN" and result.tracker is not None:
                train_mae = getattr(result.tracker, 'train_mae', 0.0)
                model = getattr(result.tracker, 'model', None)
                if model is not None:
                    overfit_ratio = result.mae / max(train_mae, 0.001)
                    print(f"    MAE: {result.mae:.2f} um (train: {train_mae:.2f}, overfit {overfit_ratio:.0f}x) | {size_str} | cal: {cal_str}, inf: {inf_str}")
                else:
                    print(f"    MAE: {result.mae:.2f} um (model not trained)")
            else:
                print(f"    MAE: {result.mae:.2f} um, RMSE: {result.rmse:.2f} um, Max: {result.max_error:.2f} um | {size_str} | cal: {cal_str}, inf: {inf_str}")
        except Exception as e:
            print(f"    Failed: {e}")

    # Print summary table
    if results:
        print("\n" + "=" * 95)
        print("SUMMARY")
        print("=" * 95)
        print(f"{'Tracker':<12} {'MAE (um)':>10} {'|z|≤15':>10} {'RMSE':>10} {'Max Err':>10} {'Size':>10} {'Cal Time':>10} {'Inf Time':>10}")
        print("-" * 95)

        # Sort by MAE
        sorted_results = sorted(results, key=lambda r: r.mae)
        for r in sorted_results:
            # Format size
            if r.model_size_bytes >= 1024 * 1024:
                size_str = f"{r.model_size_bytes / 1024 / 1024:.1f}MB"
            elif r.model_size_bytes >= 1024:
                size_str = f"{r.model_size_bytes / 1024:.1f}KB"
            else:
                size_str = f"{r.model_size_bytes}B"

            # Format cal time
            if r.calibration_time_ms >= 1000:
                cal_str = f"{r.calibration_time_ms/1000:.2f}s"
            else:
                cal_str = f"{r.calibration_time_ms:.1f}ms"

            # Format inf time
            inf_str = f"{r.inference_time_ms:.2f}ms"

            # Format narrow MAE (may be NaN if no data in range)
            narrow_str = f"{r.narrow_mae:.2f}" if not np.isnan(r.narrow_mae) else "N/A"

            print(f"{r.name:<12} {r.mae:>10.2f} {narrow_str:>10} {r.rmse:>10.2f} {r.max_error:>10.2f} {size_str:>10} {cal_str:>10} {inf_str:>10}")

        print("=" * 95)

    return results


def display_benchmark_results(
    results: list[BenchmarkResult],
    cal_frames: list[ZStackFrame],
    test_frames: list[ZStackFrame],
) -> None:
    """Display benchmark results as a table and plot."""
    if not results:
        print("No benchmark results to display!")
        return

    # Print table
    print("\n" + "=" * 75)
    print("BENCHMARK RESULTS")
    print("=" * 75)
    cal_z = [f.z_position_um for f in cal_frames if f.laf_image is not None]
    test_z = [f.z_position_um for f in test_frames if f.laf_image is not None]
    print(f"Calibration: {len(cal_z)} images, Z range [{min(cal_z):.1f}, {max(cal_z):.1f}] um")
    print(f"Test:        {len(test_z)} images, Z range [{min(test_z):.1f}, {max(test_z):.1f}] um")
    print("-" * 75)
    print(f"{'Tracker':<20} {'MAE (um)':<12} {'|z|≤15 (um)':<12} {'RMSE (um)':<12} {'Max Error (um)':<15}")
    print("-" * 75)
    for r in sorted(results, key=lambda x: x.mae):
        narrow_str = f"{r.narrow_mae:.2f}" if not np.isnan(r.narrow_mae) else "N/A"
        print(f"{r.name:<20} {r.mae:<12.2f} {narrow_str:<12} {r.rmse:<12.2f} {r.max_error:<15.2f}")
    print("=" * 75)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Error comparison bar chart
    ax1 = axes[0]
    names = [r.name for r in results]
    maes = [r.mae for r in results]
    rmses = [r.rmse for r in results]

    x = np.arange(len(names))
    width = 0.35

    ax1.bar(x - width / 2, maes, width, label="MAE", color="steelblue")
    ax1.bar(x + width / 2, rmses, width, label="RMSE", color="coral")
    ax1.set_ylabel("Error (um)")
    ax1.set_title("Z Estimation Error by Tracker")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Right: Actual vs Estimated LINE GRAPHS for ALL trackers
    ax2 = axes[1]

    # Color map for different trackers
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']

    z_min, z_max = float('inf'), float('-inf')

    for i, r in enumerate(sorted(results, key=lambda x: x.mae)):
        # Sort by actual Z for proper line plotting
        sort_idx = np.argsort(r.actual)
        actual_sorted = r.actual[sort_idx]
        estimates_sorted = r.estimates[sort_idx]

        ax2.plot(
            actual_sorted, estimates_sorted,
            color=colors[i],
            marker=markers[i % len(markers)],
            markersize=5,
            linewidth=1.5,
            alpha=0.8,
            label=f"{r.name} (MAE={r.mae:.1f})"
        )
        z_min = min(z_min, r.actual.min(), r.estimates.min())
        z_max = max(z_max, r.actual.max(), r.estimates.max())

    # Perfect line
    margin = (z_max - z_min) * 0.05
    ax2.plot([z_min - margin, z_max + margin], [z_min - margin, z_max + margin],
             'k--', alpha=0.5, label="Perfect", linewidth=1)

    ax2.set_xlabel("Actual Z (um)")
    ax2.set_ylabel("Estimated Z (um)")
    ax2.set_title("Estimated vs Actual Z (all methods)")
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_aspect("equal")
    ax2.set_xlim(z_min - margin, z_max + margin)
    ax2.set_ylim(z_min - margin, z_max + margin)

    plt.tight_layout()
    plt.show()


def display_zstack(frames: list[ZStackFrame], title: str = "Z-Stack") -> None:
    """Display the captured z-stack as a grid."""
    if not frames:
        print("No frames to display!")
        return

    num_frames = len(frames)
    n_cols = min(4, num_frames)
    n_rows = (num_frames + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(4 * n_cols, 6 * n_rows))

    if n_rows * 2 == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows * 2 == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, frame in enumerate(frames):
        row = (i // n_cols) * 2
        col = i % n_cols

        ax_bf = axes[row, col]
        if frame.bf_image is not None:
            ax_bf.imshow(frame.bf_image, cmap="gray")
            ax_bf.set_title(f"BF @ {frame.z_position_um:.1f} um", fontsize=10)
        else:
            ax_bf.text(0.5, 0.5, "No image", ha="center", va="center", transform=ax_bf.transAxes)
            ax_bf.set_title(f"BF @ {frame.z_position_um:.1f} um (missing)", fontsize=10)
        ax_bf.axis("off")

        ax_laf = axes[row + 1, col]
        if frame.laf_image is not None:
            ax_laf.imshow(frame.laf_image, cmap="gray")
            ax_laf.set_title(f"LAF @ {frame.z_position_um:.1f} um", fontsize=10)
        else:
            ax_laf.text(0.5, 0.5, "No image", ha="center", va="center", transform=ax_laf.transAxes)
            ax_laf.set_title(f"LAF @ {frame.z_position_um:.1f} um (missing)", fontsize=10)
        ax_laf.axis("off")

    for i in range(num_frames, n_rows * n_cols):
        row = (i // n_cols) * 2
        col = i % n_cols
        axes[row, col].axis("off")
        axes[row + 1, col].axis("off")

    plt.suptitle(f"{title}: BF (top) and LAF (bottom) pairs", fontsize=14)
    plt.tight_layout()
    plt.show()
