"""Multi-site benchmarking functions for LAF Z-stack trackers."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from acquisition import (
    capture_calibration_with_autofocus,
    capture_zstack,
)
from benchmark import get_tracker_classes
from client import SeafrontClient
from data_types import (
    BenchmarkParams,
    CalibrationData,
    MultiSiteResult,
    SiteResult,
    ZStackFrame,
)
from trackers.cnn import CNNTracker


def run_multisite_benchmark(
    client: SeafrontClient,
    base_url: str,
    positions: list[tuple[float, float, float]],
    cal_range_um: float,
    cal_steps: int,
    test_range_um: float,
    test_steps: int,
    bf_channel: dict,
    bf_channel_config: dict,
    laf_exposure_ms: float,
    laf_gain: float,
    separation_distance: float = 60.0,
    cal_all_sites: bool = False,
    save_dir: Path | None = None,
    capture_bf: bool = False,
    autofocus_cal: bool = False,
    fine_range_um: float = 15.0,
    fine_step_um: float = 1.0,
    coarse_step_um: float = 14.0,
) -> list[MultiSiteResult]:
    """Run multi-site benchmark: calibrate at first position (or all), test at all positions.

    Args:
        positions: List of (x_mm, y_mm, z_mm) tuples. z_mm is the focus plane at each position.
        cal_all_sites: If True, calibrate at ALL positions (averaged for CC, concatenated for CNN).
                       If False, only calibrate at first position.
        save_dir: If provided, save LAF images to this directory.
        capture_bf: If True, also capture brightfield images (default: False).
        autofocus_cal: If True, use software autofocus to find true focus plane.
        fine_range_um: Range around focus with fine steps (for autofocus_cal).
        fine_step_um: Step size in fine region (for autofocus_cal).
        coarse_step_um: Step size in coarse region (for autofocus_cal).
    """
    if len(positions) < 1:
        raise ValueError("Need at least 1 position")

    tracker_classes = get_tracker_classes()
    results: list[MultiSiteResult] = []

    # Step 1: Calibration
    if cal_all_sites:
        print(f"\n{'='*60}")
        print(f"CALIBRATION at ALL {len(positions)} POSITIONS")
        if autofocus_cal:
            print("  (with software autofocus)")
        print(f"{'='*60}")

        cal_stacks: list[list[ZStackFrame]] = []
        for site_idx, (x_mm, y_mm, z_mm) in enumerate(positions):
            print(f"\n  Calibration site {site_idx}: X={x_mm:.3f}, Y={y_mm:.3f}, Z={z_mm:.4f} mm")
            client.move_to(x_mm, y_mm, z_mm)

            if autofocus_cal:
                stack = capture_calibration_with_autofocus(
                    client, base_url, bf_channel, bf_channel_config,
                    z_mm, cal_range_um, fine_range_um, fine_step_um, coarse_step_um,
                    laf_exposure_ms, laf_gain, f"CAL-{site_idx}",
                    save_dir=save_dir,
                )
            else:
                stack = capture_zstack(
                    client, base_url, bf_channel, bf_channel_config,
                    z_mm, cal_range_um, cal_steps,
                    laf_exposure_ms, laf_gain, f"CAL-{site_idx}",
                    save_dir=save_dir,
                    capture_bf=capture_bf,
                )
            cal_stacks.append(stack)

        # Create two calibration datasets:
        # - Averaged for CC methods (more robust templates)
        # - Concatenated for CNN (more training data)
        calibration_averaged = CalibrationData.from_multiple_stacks(cal_stacks, separation_distance)
        calibration_concatenated = CalibrationData.from_multiple_stacks_concatenated(cal_stacks, separation_distance)
        print(f"\n  Calibration: {len(cal_stacks)} sites, {len(calibration_averaged.z_positions)} Z levels")
        print(f"  CNN training data: {len(calibration_concatenated.z_positions)} total images")
    else:
        # Single-site calibration
        cal_x, cal_y, cal_z = positions[0]
        print(f"\n{'='*60}")
        print(f"CALIBRATION at position 0: X={cal_x:.3f}, Y={cal_y:.3f}, Z={cal_z:.4f} mm")
        if autofocus_cal:
            print("  (with software autofocus)")
        print(f"{'='*60}")

        client.move_to(cal_x, cal_y, cal_z)

        if autofocus_cal:
            cal_frames = capture_calibration_with_autofocus(
                client, base_url, bf_channel, bf_channel_config,
                cal_z, cal_range_um, fine_range_um, fine_step_um, coarse_step_um,
                laf_exposure_ms, laf_gain, "CAL",
                save_dir=save_dir,
            )
        else:
            cal_frames = capture_zstack(
                client, base_url, bf_channel, bf_channel_config,
                cal_z, cal_range_um, cal_steps,
                laf_exposure_ms, laf_gain, "CAL",
                save_dir=save_dir,
                capture_bf=capture_bf,
            )

        calibration_averaged = CalibrationData.from_frames(cal_frames, separation_distance)
        calibration_concatenated = calibration_averaged  # Same for single site

    # Step 2: Train all trackers and record calibration time
    print(f"\n{'='*60}")
    print("TRAINING TRACKERS")
    print(f"{'='*60}")

    from trackers.base import BaseTracker

    trained_trackers: list[tuple[type[BaseTracker], BaseTracker, float, int]] = []
    for tracker_class in tracker_classes:
        print(f"  Training {tracker_class.name}...", end=" ", flush=True)
        try:
            # CNN uses concatenated data (all images), others use averaged data
            if tracker_class.name == "CNN":
                calibration = calibration_concatenated
            else:
                calibration = calibration_averaged

            t0 = time.perf_counter()
            tracker = tracker_class(calibration, verbose=False)
            cal_time_ms = (time.perf_counter() - t0) * 1000
            model_size = tracker.get_model_size_bytes()
            trained_trackers.append((tracker_class, tracker, cal_time_ms, model_size))

            # Show extra info for CNN (overfitting detection)
            if tracker_class.name == "CNN" and isinstance(tracker, CNNTracker):
                overfit_ratio = tracker.val_mae / tracker.train_mae if tracker.train_mae > 0 else 0
                overfit_str = "OVERFIT" if overfit_ratio > 2 else "ok"
                print(f"OK ({cal_time_ms:.1f}ms) - train: {tracker.train_mae:.1f}um, val: {tracker.val_mae:.1f}um ({overfit_ratio:.1f}x {overfit_str})")
            else:
                print(f"OK ({cal_time_ms:.1f}ms)")
        except Exception as e:
            print(f"FAILED: {e}")

    # Step 3: Test at each position
    print(f"\n{'='*60}")
    print(f"TESTING AT {len(positions)} POSITIONS")
    print(f"{'='*60}")

    # Initialize results structure
    for tracker_class, tracker, cal_time_ms, model_size in trained_trackers:
        results.append(MultiSiteResult(
            name=tracker_class.name,
            model_size_bytes=model_size,
            calibration_time_ms=cal_time_ms,
            inference_time_ms=0.0,
            site_results=[],
        ))

    total_inference_times: list[list[float]] = [[] for _ in trained_trackers]

    for site_idx, (x_mm, y_mm, z_mm) in enumerate(positions):
        is_cal_site = cal_all_sites or (site_idx == 0)
        cal_label = "(calibration site)" if is_cal_site else ""
        print(f"\n  Site {site_idx}: X={x_mm:.3f}, Y={y_mm:.3f}, Z={z_mm:.4f} mm {cal_label}")

        client.move_to(x_mm, y_mm, z_mm)

        test_frames = capture_zstack(
            client, base_url, bf_channel, bf_channel_config,
            z_mm, test_range_um, test_steps,
            laf_exposure_ms, laf_gain, f"TEST-{site_idx}",
            save_dir=save_dir,
            capture_bf=capture_bf,
        )

        valid_test = [f for f in test_frames if f.laf_image is not None]

        # Test each tracker at this site
        for tracker_idx, (tracker_class, tracker, _, _) in enumerate(trained_trackers):
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

            errors_arr = np.array(errors)
            estimates_arr = np.array(estimates)
            actual_arr = np.array(actual)

            mae = float(np.mean(np.abs(errors_arr)))
            rmse = float(np.sqrt(np.mean(errors_arr**2)))
            max_error = float(np.max(np.abs(errors_arr)))

            # Narrow-range MAE
            narrow_mask = np.abs(actual_arr) <= 15.0
            narrow_mae = float(np.mean(np.abs(errors_arr[narrow_mask]))) if np.any(narrow_mask) else float('nan')

            # Super narrow-range MAE (|z| <= 5um)
            super_narrow_mask = np.abs(actual_arr) <= 5.0
            super_narrow_mae = float(np.mean(np.abs(errors_arr[super_narrow_mask]))) if np.any(super_narrow_mask) else float('nan')

            total_inference_times[tracker_idx].extend(inference_times)

            results[tracker_idx].site_results.append(SiteResult(
                site_idx=site_idx,
                x_mm=x_mm,
                y_mm=y_mm,
                z_mm=z_mm,
                is_calibration_site=is_cal_site,
                mae=mae,
                narrow_mae=narrow_mae,
                super_narrow_mae=super_narrow_mae,
                rmse=rmse,
                max_error=max_error,
                errors=errors_arr,
                estimates=estimates_arr,
                actual=actual_arr,
            ))

    # Update average inference times
    for i, times in enumerate(total_inference_times):
        if times:
            results[i] = MultiSiteResult(
                name=results[i].name,
                model_size_bytes=results[i].model_size_bytes,
                calibration_time_ms=results[i].calibration_time_ms,
                inference_time_ms=float(np.mean(times)),
                site_results=results[i].site_results,
            )

    return results


def print_multisite_summary(
    results: list[MultiSiteResult],
    positions: list[tuple[float, float, float]],
    params: BenchmarkParams | None = None,
) -> None:
    """Print summary tables for multi-site benchmark results."""

    def fmt_size(size: int) -> str:
        if size >= 1024 * 1024:
            return f"{size / 1024 / 1024:.1f}MB"
        elif size >= 1024:
            return f"{size / 1024:.1f}KB"
        return f"{size}B"

    def fmt_time(ms: float) -> str:
        if ms >= 1000:
            return f"{ms/1000:.2f}s"
        return f"{ms:.1f}ms"

    def fmt_mae(v: float) -> str:
        return f"{v:.2f}" if not np.isnan(v) else "N/A"

    # Parameters summary
    if params:
        print("\n" + "=" * 60)
        print("BENCHMARK PARAMETERS")
        print("=" * 60)
        cal_step_size = params.cal_range_um / (params.cal_steps - 1) if params.cal_steps > 1 else 0
        test_step_size = params.test_range_um / (params.test_steps - 1) if params.test_steps > 1 else 0
        print(f"  Positions:    {params.n_positions} {'(cal all)' if params.cal_all_sites else '(cal first only)'}")
        if params.autofocus_cal:
            print(f"  Calibration:  non-uniform steps + software autofocus")
            print(f"    Fine:       ±{params.fine_range_um:.0f}um @ {params.fine_step_um:.1f}um step")
            print(f"    Coarse:     ±{params.cal_range_um/2:.0f}um @ {params.coarse_step_um:.1f}um step")
        else:
            print(f"  Calibration:  {params.cal_steps} steps, ±{params.cal_range_um/2:.0f}um range, {cal_step_size:.1f}um step")
        print(f"  Test:         {params.test_steps} steps, ±{params.test_range_um/2:.0f}um range, {test_step_size:.1f}um step")
        print(f"  LAF:          {params.laf_exposure_ms}ms exposure, {params.laf_gain} gain")
        print(f"  Separation:   {params.separation_distance}px (expected dot spacing)")
        print("=" * 60)

    # Table 1: Model sizes and calibration time
    print("\n" + "=" * 60)
    print("TABLE 1: MODEL SIZE AND CALIBRATION TIME")
    print("=" * 60)
    print(f"{'Tracker':<14} {'Model Size':>12} {'Cal Time':>12} {'Inf Time':>12}")
    print("-" * 60)

    for r in sorted(results, key=lambda x: x.avg_mae):
        print(f"{r.name:<14} {fmt_size(r.model_size_bytes):>12} {fmt_time(r.calibration_time_ms):>12} {fmt_time(r.inference_time_ms):>12}")
    print("=" * 60)

    # Table 2: Average results across all positions
    # Short descriptions for each method
    method_desc: dict[str, str] = {
        "Zero": "always predicts 0",
        "CC 1D": "1D cross-corr of profiles",
        "CC 2D": "2D cross-corr full img",
        "CC 2D Spline": "2D CC + spline interp",
        "CC 2D+JPEG": "2D CC w/ JPEG compression",
        "CNN": "CNN on 64x64 resized",
        "Peak 1D": "track peak in 1D profile",
        "Gauss 1D": "gaussian fit to 1D profile",
        "Centroid": "intensity centroid tracking",
        "Centroid 2D": "centroid + brightness 2D lookup",
        "★SQUIDL": "SQUID left peak (default)",
        "★SQUIDR": "SQUID right peak (default)",
        "SQUIDL+σ0": "SQUID left, no extra smoothing",
        "SQUIDL+σ2": "SQUID left + gaussian σ=2",
        "SQUIDL+σ4": "SQUID left + gaussian σ=4",
        "SQUIDL+σ10": "SQUID left + gaussian σ=10",
        "SQUIDL+σ20": "SQUID left + gaussian σ=20",
        "SQUIDL+σ50": "SQUID left + gaussian σ=50",
        "SQUIDR+σ0": "SQUID right, no extra smoothing",
        "SQUIDR+σ2": "SQUID right + gaussian σ=2",
        "SQUIDR+σ4": "SQUID right + gaussian σ=4",
        "SQUIDR+σ10": "SQUID right + gaussian σ=10",
        "SQUIDR+σ20": "SQUID right + gaussian σ=20",
        "SQUIDR+σ50": "SQUID right + gaussian σ=50",
    }

    print("\n" + "=" * 120)
    print(f"TABLE 2: AVERAGE RESULTS ACROSS {len(positions)} POSITIONS")
    print("=" * 120)
    print(f"{'Tracker':<14} {'Avg MAE':>8} {'|z|≤15':>8} {'|z|≤5':>8} {'MaxErr':>8}  {'Method':<30}")
    print("-" * 120)

    for r in sorted(results, key=lambda x: x.avg_mae):
        desc = method_desc.get(r.name, "")
        print(f"{r.name:<14} {r.avg_mae:>8.2f} {fmt_mae(r.avg_narrow_mae):>8} {fmt_mae(r.avg_super_narrow_mae):>8} {r.avg_max_error:>8.1f}  {desc:<30}")
    print("=" * 120)

    # Check if all sites are calibration sites
    all_sites_calibrated = results and all(s.is_calibration_site for s in results[0].site_results)

    # Table 3: Calibration site vs other sites (only if not all sites calibrated)
    if not all_sites_calibrated:
        print("\n" + "=" * 100)
        print("TABLE 3: CALIBRATION SITE vs OTHER SITES (GENERALIZATION)")
        print("=" * 100)
        print(f"{'Tracker':<14} {'Cal MAE':>10} {'Cal |z|≤15':>12} {'Other MAE':>12} {'Other |z|≤15':>14} {'Δ MAE':>10} {'Δ |z|≤15':>10}")
        print("-" * 100)

        for r in sorted(results, key=lambda x: x.avg_mae):
            cal_mae = r.cal_site_mae
            cal_narrow = r.cal_site_narrow_mae
            other_mae = r.other_sites_mae
            other_narrow = r.other_sites_narrow_mae

            delta_mae = other_mae - cal_mae if not (np.isnan(cal_mae) or np.isnan(other_mae)) else float('nan')
            delta_narrow = other_narrow - cal_narrow if not (np.isnan(cal_narrow) or np.isnan(other_narrow)) else float('nan')

            delta_mae_str = f"{delta_mae:+.2f}" if not np.isnan(delta_mae) else "N/A"
            delta_narrow_str = f"{delta_narrow:+.2f}" if not np.isnan(delta_narrow) else "N/A"

            print(f"{r.name:<14} {fmt_mae(cal_mae):>10} {fmt_mae(cal_narrow):>12} {fmt_mae(other_mae):>12} {fmt_mae(other_narrow):>14} {delta_mae_str:>10} {delta_narrow_str:>10}")
        print("=" * 100)
    else:
        print("\n" + "=" * 60)
        print("TABLE 3: SKIPPED (all sites used for calibration)")
        print("=" * 60)

    # Per-site breakdown for top methods
    print("\n" + "=" * 100)
    print("TABLE 4: PER-SITE BREAKDOWN (top 5 methods by avg MAE)")
    print("=" * 100)

    top_methods = sorted(results, key=lambda x: x.avg_mae)[:5]
    header = f"{'Site':>6} {'X':>8} {'Y':>8} {'Z':>8} {'Cal?':>5}"
    for r in top_methods:
        header += f" {r.name[:8]:>10}"
    print(header)
    print("-" * 100)

    for site_idx, (x, y, z) in enumerate(positions):
        # Check actual site result for calibration status
        is_cal = "Yes" if results[0].site_results[site_idx].is_calibration_site else ""
        row = f"{site_idx:>6} {x:>8.3f} {y:>8.3f} {z:>8.4f} {is_cal:>5}"
        for r in top_methods:
            site_result = r.site_results[site_idx]
            row += f" {site_result.mae:>10.2f}"
        print(row)
    print("=" * 100)
