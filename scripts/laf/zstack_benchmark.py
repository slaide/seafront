#!/usr/bin/env python3
"""
Z-Stack Benchmark Script with LAF Z-Estimation

Captures calibration and test z-stacks of BF and LAF image pairs,
then benchmarks multiple Z estimation algorithms.

Usage:
    uv run python scripts/laf/zstack_benchmark.py --microscope "microscope" \\
        --cal-range 100 --cal-steps 21 \\
        --test-range 80 --test-steps 17
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
from pathlib import Path

# Add this directory to path for local imports
_dir = Path(__file__).parent
sys.path.insert(0, str(_dir))
sys.path.insert(0, str(_dir / "trackers"))

import numpy as np

from acquisition import find_bf_channel, get_filter_handles, run_acquisition
from benchmark import display_benchmark_results, display_zstack, run_all_benchmarks
from client import SeafrontClient
from data_types import BenchmarkParams
from multisite import print_multisite_summary, run_multisite_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="Capture z-stack and benchmark LAF Z-estimation algorithms"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:5000",
        help="Seafront server URL (default: http://localhost:5000)",
    )
    parser.add_argument(
        "--microscope",
        required=True,
        help="Microscope name (e.g., 'squid-royal-shell')",
    )
    # Calibration parameters
    parser.add_argument(
        "--cal-range",
        type=float,
        default=100.0,
        help="Calibration z-stack range in um (default: 100)",
    )
    parser.add_argument(
        "--cal-steps",
        type=int,
        default=21,
        help="Number of calibration z positions (default: 21)",
    )
    # Test parameters
    parser.add_argument(
        "--test-range",
        type=float,
        default=80.0,
        help="Test z-stack range in um (default: 80)",
    )
    parser.add_argument(
        "--test-steps",
        type=int,
        default=17,
        help="Number of test z positions (default: 17)",
    )
    # Imaging parameters
    parser.add_argument(
        "--bf-exposure",
        type=float,
        default=10.0,
        help="Brightfield exposure time in ms (default: 10)",
    )
    parser.add_argument(
        "--bf-illum",
        type=float,
        default=20.0,
        help="Brightfield illumination percentage (default: 20)",
    )
    parser.add_argument(
        "--laf-exposure",
        type=float,
        default=5.0,
        help="LAF exposure time in ms (default: 5)",
    )
    parser.add_argument(
        "--laf-gain",
        type=float,
        default=10.0,
        help="LAF analog gain (default: 10)",
    )
    parser.add_argument(
        "--separation",
        type=float,
        default=60.0,
        help="Expected dot separation in pixels (default: 60)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Skip displaying plots",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show debug information about peak detection",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible results (sets np.random.seed before capture)",
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Skip running the Z estimation benchmarks",
    )
    parser.add_argument(
        "--show-images",
        action="store_true",
        help="Show the captured image grids",
    )
    # Multi-position calibration (legacy, for template averaging)
    parser.add_argument(
        "--cal-positions",
        type=int,
        default=1,
        help="Number of XY positions to capture calibration at (default: 1). "
             "If >1, templates are averaged across positions for robustness.",
    )
    parser.add_argument(
        "--cal-offset",
        type=float,
        default=0.5,
        help="XY offset between calibration positions in mm (default: 0.5)",
    )
    # Multi-site benchmarking (calibrate at one site, test at all sites)
    parser.add_argument(
        "--positions",
        type=str,
        default=None,
        help="Path to JSON file with positions: [[x, y, z], ...] in mm. "
             "First position is calibration site. Enables multi-site benchmark mode.",
    )
    parser.add_argument(
        "--cal-all-sites",
        action="store_true",
        help="Calibrate at ALL positions (not just first). "
             "CC methods use averaged templates, CNN trains on all images.",
    )
    parser.add_argument(
        "--save-images",
        type=str,
        nargs="?",
        const="scripts/laf/laf_images",
        default=None,
        help="Save LAF images to directory (default: scripts/laf/laf_images if flag given without path)",
    )
    parser.add_argument(
        "--with-bf",
        action="store_true",
        help="Also capture brightfield images (disabled by default)",
    )
    # Software autofocus calibration
    parser.add_argument(
        "--autofocus-cal",
        action="store_true",
        help="Use software autofocus during calibration to find true focus plane. "
             "Enables non-uniform step sizes (fine near focus, coarse elsewhere).",
    )
    parser.add_argument(
        "--fine-range",
        type=float,
        default=15.0,
        help="Range around focus with fine steps in µm (default: 15)",
    )
    parser.add_argument(
        "--fine-step",
        type=float,
        default=1.0,
        help="Step size in fine region in µm (default: 1)",
    )
    parser.add_argument(
        "--coarse-step",
        type=float,
        default=14.0,
        help="Step size in coarse region in µm (default: 14)",
    )

    args = parser.parse_args()

    if args.cal_steps < 3:
        parser.error("--cal-steps must be at least 3")
    if args.test_steps < 1:
        parser.error("--test-steps must be at least 1")

    try:
        # Set random seed for reproducibility if specified
        if args.seed is not None:
            np.random.seed(args.seed)
            print(f"Random seed set to {args.seed} for reproducible results")

        # Set up image save directory if requested
        save_dir: Path | None = None
        if args.save_images:
            save_dir = Path(args.save_images)
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving LAF images to: {save_dir}")

        # Check for multi-site mode
        if args.positions:
            # Load positions from JSON file
            with open(args.positions) as f:
                positions_raw = json.load(f)
            positions: list[tuple[float, float, float]] = [tuple(p) for p in positions_raw]  # type: ignore
            print(f"Multi-site benchmark mode: {len(positions)} positions loaded from {args.positions}")

            # Initialize client and get hardware info
            client = SeafrontClient(args.url, args.microscope)
            print(f"Connecting to {args.url} for microscope '{args.microscope}'...")

            capabilities = client.get_hardware_capabilities()
            bf_channel = find_bf_channel(capabilities)
            if bf_channel is None:
                raise RuntimeError("No brightfield channel found")

            machine_defaults = client.get_machine_defaults()
            _, no_filter = get_filter_handles(machine_defaults)

            bf_channel_config = {
                "name": bf_channel["name"],
                "handle": bf_channel["handle"],
                "illum_perc": args.bf_illum,
                "exposure_time_ms": args.bf_exposure,
                "analog_gain": 0.0,
                "z_offset_um": 0.0,
                "num_z_planes": 1,
                "delta_z_um": 1.0,
                "filter_handle": no_filter,
                "enabled": True,
            }

            # Run multi-site benchmark
            results = run_multisite_benchmark(
                client=client,
                base_url=args.url,
                positions=positions,
                cal_range_um=args.cal_range,
                cal_steps=args.cal_steps,
                test_range_um=args.test_range,
                test_steps=args.test_steps,
                bf_channel=bf_channel,
                bf_channel_config=bf_channel_config,
                laf_exposure_ms=args.laf_exposure,
                laf_gain=args.laf_gain,
                separation_distance=args.separation,
                cal_all_sites=args.cal_all_sites,
                save_dir=save_dir,
                capture_bf=args.with_bf,
                autofocus_cal=args.autofocus_cal,
                fine_range_um=args.fine_range,
                fine_step_um=args.fine_step,
                coarse_step_um=args.coarse_step,
            )

            # Print summary tables
            params = BenchmarkParams(
                cal_range_um=args.cal_range,
                cal_steps=args.cal_steps,
                test_range_um=args.test_range,
                test_steps=args.test_steps,
                laf_exposure_ms=args.laf_exposure,
                laf_gain=args.laf_gain,
                separation_distance=args.separation,
                n_positions=len(positions),
                cal_all_sites=args.cal_all_sites,
                autofocus_cal=args.autofocus_cal,
                fine_range_um=args.fine_range,
                fine_step_um=args.fine_step,
                coarse_step_um=args.coarse_step,
            )
            print_multisite_summary(results, positions, params)

            # Return to first position
            client.move_to(*positions[0])

        else:
            # Standard single-site mode
            # Acquire calibration and test z-stacks
            cal_frames, test_frames = run_acquisition(
                base_url=args.url,
                microscope_name=args.microscope,
                cal_range_um=args.cal_range,
                cal_steps=args.cal_steps,
                test_range_um=args.test_range,
                test_steps=args.test_steps,
                bf_exposure_ms=args.bf_exposure,
                bf_illum_perc=args.bf_illum,
                laf_exposure_ms=args.laf_exposure,
                laf_gain=args.laf_gain,
                cal_positions=args.cal_positions,
                cal_offset_mm=args.cal_offset,
            )

            # Run benchmarks
            if not args.no_benchmark:
                print("\n" + "=" * 60)
                print("RUNNING Z-ESTIMATION BENCHMARKS")
                print("=" * 60)
                results = run_all_benchmarks(cal_frames, test_frames, args.separation, verbose=args.verbose)

                if results and not args.no_display:
                    # For display, use first calibration stack if multi-position
                    cal_for_display = cal_frames[0] if args.cal_positions > 1 else cal_frames  # type: ignore
                    display_benchmark_results(results, cal_for_display, test_frames)  # type: ignore

            # Show images if requested
            if args.show_images and not args.no_display:
                # For display, use first calibration stack if multi-position
                cal_for_display = cal_frames[0] if args.cal_positions > 1 else cal_frames  # type: ignore
                display_zstack(cal_for_display, "Calibration")  # type: ignore
                display_zstack(test_frames, "Test")

    except urllib.error.URLError as e:
        print(f"Error connecting to server: {e}")
        print("Make sure the seafront server is running.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        raise

    return 0


if __name__ == "__main__":
    exit(main())
