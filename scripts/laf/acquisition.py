"""Image acquisition functions for LAF Z-stack benchmarking."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import cv2
import numpy as np

from calibration import (
    adjust_z_positions_to_focus,
    find_best_focus_index,
    generate_nonuniform_z_positions,
)
from client import SeafrontClient
from data_types import ZStackFrame


async def fetch_image_via_websocket(
    base_url: str, microscope_name: str, channel_handle: str
) -> np.ndarray | None:
    """Fetch an acquired image via websocket."""
    try:
        import websockets
    except ImportError:
        print("websockets package not installed. Install with: uv add websockets")
        return None

    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/ws/get_info/acquired_image"

    try:
        async with websockets.connect(ws_url, max_size=50 * 1024 * 1024) as ws:
            await ws.send(json.dumps({"target_device": microscope_name}))
            await ws.send(channel_handle)

            metadata_str = await ws.recv()
            metadata = json.loads(metadata_str)

            if not metadata:
                print(f"  No image data available for channel '{channel_handle}'")
                return None

            width = metadata["width"]
            height = metadata["height"]
            bit_depth = metadata["bit_depth"]

            await ws.send("1")
            img_bytes = await ws.recv()
            assert isinstance(img_bytes, bytes), "Expected binary image data"

            if bit_depth == 8:
                img = np.frombuffer(img_bytes, dtype=np.uint8).reshape((height, width))
            elif bit_depth == 16:
                img = np.frombuffer(img_bytes, dtype=np.uint16).reshape((height, width))
            else:
                img = np.frombuffer(img_bytes, dtype=np.uint8).reshape((height, width))

            return img

    except Exception as e:
        print(f"  Error fetching image for channel '{channel_handle}': {e}")
        return None


def find_bf_channel(capabilities: dict) -> dict | None:
    """Find a brightfield channel from hardware capabilities."""
    channels = capabilities.get("main_camera_imaging_channels", [])
    for ch in channels:
        handle = ch.get("handle", "")
        if handle.startswith("bfled"):
            return ch
    return None


def get_filter_handles(machine_defaults: list[dict]) -> tuple[str | None, str | None]:
    """Get filter handles from machine defaults.

    Returns (first_filter, no_filter) tuple.
    """
    for item in machine_defaults:
        if item.get("handle") == "filter.wheel.configuration":
            filters = item.get("value", [])
            if not filters:
                return None, None

            first_filter = filters[0].get("handle")
            no_filter = None

            # Look for "no filter" position (em_none or similar)
            for f in filters:
                handle = f.get("handle", "")
                if "none" in handle.lower() or "empty" in handle.lower():
                    no_filter = handle
                    break

            return first_filter, no_filter
    return None, None


def capture_zstack_nonuniform(
    client: SeafrontClient,
    base_url: str,
    bf_channel: dict,
    bf_channel_config: dict,
    ref_z_mm: float,
    z_positions_um: np.ndarray,
    laf_exposure_ms: float,
    laf_gain: float,
    label: str,
    save_dir: Path | None = None,
    capture_bf: bool = True,
) -> list[ZStackFrame]:
    """Capture a z-stack with non-uniform (pre-specified) Z positions.

    Args:
        z_positions_um: Array of Z offsets in µm relative to ref_z_mm
        capture_bf: If True, capture BF images (needed for software autofocus)
    """
    print(f"\n{label} Z-stack (non-uniform):")
    print(f"  {len(z_positions_um)} positions, range [{z_positions_um.min():.1f}, {z_positions_um.max():.1f}] µm")

    frames: list[ZStackFrame] = []

    for i, z_offset_um in enumerate(z_positions_um):
        target_z_mm = ref_z_mm + z_offset_um / 1000.0
        client.move_to_z(target_z_mm)

        current_z_mm = client.get_z_position_mm()
        z_relative_um = (current_z_mm - ref_z_mm) * 1000

        print(f"  [{label}] Position {i+1}/{len(z_positions_um)}: Z offset = {z_relative_um:.2f} um")

        # Snap BF image (for software autofocus)
        bf_image = None
        if capture_bf:
            client.snap_bf_channel(bf_channel_config)
            bf_image = asyncio.run(
                fetch_image_via_websocket(base_url, client.microscope_name, bf_channel["handle"])
            )

        # Snap LAF image
        client.snap_laf(exposure_time_ms=laf_exposure_ms, analog_gain=laf_gain)
        laf_image = asyncio.run(
            fetch_image_via_websocket(base_url, client.microscope_name, "laser_autofocus")
        )

        # Save LAF image if requested
        if save_dir is not None and laf_image is not None:
            filename = f"laf-{label}-{i:03d}-z{z_relative_um:+.1f}um.jpeg"
            filepath = save_dir / filename
            cv2.imwrite(str(filepath), laf_image, [cv2.IMWRITE_JPEG_QUALITY, 100])

        frames.append(ZStackFrame(
            z_position_um=z_relative_um,
            bf_image=bf_image,
            laf_image=laf_image,
        ))

    print(f"  {label} complete: {len(frames)} frames acquired.")
    return frames


def capture_calibration_with_autofocus(
    client: SeafrontClient,
    base_url: str,
    bf_channel: dict,
    bf_channel_config: dict,
    ref_z_mm: float,
    z_range_um: float,
    fine_range_um: float,
    fine_step_um: float,
    coarse_step_um: float,
    laf_exposure_ms: float,
    laf_gain: float,
    label: str,
    save_dir: Path | None = None,
) -> list[ZStackFrame]:
    """Capture calibration z-stack with non-uniform steps and software autofocus.

    1. Generates non-uniform Z positions (fine near center, coarse at edges)
    2. Captures BF + LAF images at each position
    3. Runs software autofocus to find true focus plane
    4. Adjusts all Z positions so focus = 0

    Returns:
        List of ZStackFrame with Z positions adjusted so focus = 0
    """
    # Generate non-uniform positions
    z_positions = generate_nonuniform_z_positions(
        z_range_um=z_range_um,
        fine_range_um=fine_range_um,
        fine_step_um=fine_step_um,
        coarse_step_um=coarse_step_um,
    )

    print(f"\n  Non-uniform calibration: {len(z_positions)} steps")
    print(f"    Fine region: ±{fine_range_um}µm with {fine_step_um}µm steps")
    print(f"    Coarse region: ±{z_range_um/2}µm with {coarse_step_um}µm steps")

    # Capture stack (LAF images used for autofocus, BF kept for debugging)
    frames = capture_zstack_nonuniform(
        client=client,
        base_url=base_url,
        bf_channel=bf_channel,
        bf_channel_config=bf_channel_config,
        ref_z_mm=ref_z_mm,
        z_positions_um=z_positions,
        laf_exposure_ms=laf_exposure_ms,
        laf_gain=laf_gain,
        label=label,
        save_dir=save_dir,
        capture_bf=True,
    )

    # Try all focus methods and compare results
    focus_methods = [
        "laplacian_var",
        "gradient",
        "tenengrad",
        "peak_intensity",
        "contrast",
        "neg_entropy",
        "percentile_ratio",
        "local_variance",
    ]

    print("\n  Focus method comparison (LAF images):")
    print(f"  {'Method':<20} {'Best Z (µm)':>12} {'Frame':>6}")
    print("  " + "-" * 42)

    method_results: dict[str, float] = {}
    for method in focus_methods:
        best_idx, _ = find_best_focus_index(frames, use_bf=False, method=method)
        focus_z = frames[best_idx].z_position_um
        method_results[method] = focus_z
        print(f"  {method:<20} {focus_z:>12.2f} {best_idx:>6}")

    # Use the method that finds focus closest to 0 (our expected focus)
    # This is a heuristic - the "true" focus should be near the user-specified Z
    best_method = min(method_results, key=lambda m: abs(method_results[m]))
    focus_z_um = method_results[best_method]

    print("  " + "-" * 42)
    print(f"  Selected: {best_method} (Z = {focus_z_um:.2f} µm, closest to 0)")

    # Adjust Z positions so focus = 0
    adjusted_frames = adjust_z_positions_to_focus(frames, focus_z_um)

    return adjusted_frames


def capture_zstack(
    client: SeafrontClient,
    base_url: str,
    bf_channel: dict,
    bf_channel_config: dict,
    ref_z_mm: float,
    z_range_um: float,
    num_steps: int,
    laf_exposure_ms: float,
    laf_gain: float,
    label: str,
    save_dir: Path | None = None,
    capture_bf: bool = False,
) -> list[ZStackFrame]:
    """Capture a z-stack centered on the reference position.

    Args:
        save_dir: If provided, save LAF images to this directory as JPEG files.
        capture_bf: If True, also capture brightfield images (default: False).
    """
    step_size_um = z_range_um / (num_steps - 1) if num_steps > 1 else 0
    step_size_mm = step_size_um / 1000.0
    half_range_mm = (z_range_um / 2) / 1000.0

    print(f"\n{label} Z-stack:")
    print(f"  Range: {z_range_um} um, Steps: {num_steps}, Step size: {step_size_um:.2f} um")

    # Move to start position
    start_z_mm = ref_z_mm - half_range_mm
    client.move_to_z(start_z_mm)

    frames: list[ZStackFrame] = []

    for i in range(num_steps):
        current_z_mm = client.get_z_position_mm()
        z_relative_um = (current_z_mm - ref_z_mm) * 1000

        print(f"  [{label}] Position {i+1}/{num_steps}: Z offset = {z_relative_um:.2f} um")

        # Snap BF image (optional)
        bf_image = None
        if capture_bf:
            client.snap_bf_channel(bf_channel_config)
            bf_image = asyncio.run(
                fetch_image_via_websocket(base_url, client.microscope_name, bf_channel["handle"])
            )

        # Snap LAF image
        client.snap_laf(exposure_time_ms=laf_exposure_ms, analog_gain=laf_gain)
        laf_image = asyncio.run(
            fetch_image_via_websocket(base_url, client.microscope_name, "laser_autofocus")
        )

        # Save LAF image if save_dir is provided
        if save_dir is not None and laf_image is not None:
            # Format: laf-{label}-{step}-z{offset}um.jpeg
            # Step number ensures correct sorting order
            filename = f"laf-{label}-{i:03d}-z{z_relative_um:+.1f}um.jpeg"
            filepath = save_dir / filename
            cv2.imwrite(str(filepath), laf_image, [cv2.IMWRITE_JPEG_QUALITY, 100])

        frames.append(
            ZStackFrame(
                z_position_um=z_relative_um,
                bf_image=bf_image,
                laf_image=laf_image,
            )
        )

        if i < num_steps - 1:
            client.move_by_z(step_size_mm)

    print(f"  {label} complete: {len(frames)} frames acquired.")
    return frames


def run_acquisition(
    base_url: str,
    microscope_name: str,
    cal_range_um: float,
    cal_steps: int,
    test_range_um: float,
    test_steps: int,
    bf_exposure_ms: float = 10.0,
    bf_illum_perc: float = 20.0,
    laf_exposure_ms: float = 5.0,
    laf_gain: float = 10.0,
    cal_positions: int = 1,
    cal_offset_mm: float = 0.5,
) -> tuple[list[ZStackFrame] | list[list[ZStackFrame]], list[ZStackFrame]]:
    """Run calibration and test z-stack acquisitions.

    Args:
        cal_positions: Number of XY positions to capture calibration stacks at.
            If >1, returns list of stacks for multi-position averaging.
        cal_offset_mm: XY offset between calibration positions (mm).
    """
    client = SeafrontClient(base_url, microscope_name)

    print(f"Connecting to {base_url} for microscope '{microscope_name}'...")

    # Try to stay at current position if valid, otherwise move to safe position D4
    current_x, current_y, current_z = client.get_position()
    print(f"Current position: X={current_x:.2f}, Y={current_y:.2f}, Z={current_z:.2f} mm")

    if client.try_move_to(current_x, current_y, current_z):
        print("  Current position is valid, staying here.")
        start_x, start_y, start_z = current_x, current_y, current_z
    else:
        # Move to safe position: center of well D4 on 384-well plate
        # 384-well plate: A1 at (12.13, 8.99) mm, 4.5mm spacing
        # D4 = row 3 (0-indexed), col 3 (0-indexed)
        start_x = 12.13 + 3 * 4.5  # 25.63 mm
        start_y = 8.99 + 3 * 4.5   # 22.49 mm
        start_z = 1.0  # mm
        print(f"  Current position is forbidden, moving to D4: X={start_x:.2f}, Y={start_y:.2f}, Z={start_z:.2f} mm")
        client.move_to(start_x, start_y, start_z)

    capabilities = client.get_hardware_capabilities()
    bf_channel = find_bf_channel(capabilities)
    if bf_channel is None:
        raise RuntimeError("No brightfield channel found in hardware capabilities")

    print(f"Using BF channel: {bf_channel['name']} (handle: {bf_channel['handle']})")

    # Get filter handles from machine defaults
    machine_defaults = client.get_machine_defaults()
    first_filter, no_filter = get_filter_handles(machine_defaults)
    print(f"Filters: first={first_filter}, no_filter={no_filter}")

    # Use no_filter for BF channels (or first_filter as fallback)
    bf_filter = no_filter or first_filter

    bf_channel_config = {
        "name": bf_channel["name"],
        "handle": bf_channel["handle"],
        "illum_perc": bf_illum_perc,
        "exposure_time_ms": bf_exposure_ms,
        "analog_gain": 0.0,
        "z_offset_um": 0.0,
        "num_z_planes": 1,
        "delta_z_um": 1.0,
        "filter_handle": bf_filter,
        "enabled": True,
    }

    config_list = client.get_config_list()
    if not config_list:
        raise RuntimeError("No acquisition configs found. Run generate_default_protocol.py first.")

    config_response = client.get_config(config_list[0]["filename"])
    acq_config = config_response["file"]

    # Check if LAF is calibrated from the server's machine config
    laf_is_calibrated = False
    for item in machine_defaults:
        if item.get("handle") == "laser.autofocus.calibration.is_calibrated":
            laf_is_calibrated = item.get("boolvalue", False)
            break

    # Step 1: Get reference Z position
    if laf_is_calibrated:
        print("\nStep 1: LAF is calibrated, approaching target offset 0 um (focus plane)...")
        result = client.approach_target_offset(0.0, acq_config)
        print(
            f"  Approach result: {result['num_compensating_moves']} compensating moves, "
            f"uncompensated offset: {result['uncompensated_offset_mm']*1000:.2f} um"
        )
    else:
        print("\nStep 1: LAF is not calibrated, using current Z as reference...")

    ref_x_mm, ref_y_mm, ref_z_mm = client.get_position()
    print(f"  Reference position: X={ref_x_mm:.4f}, Y={ref_y_mm:.4f}, Z={ref_z_mm:.4f} mm")

    # Step 2: Capture calibration z-stack(s)
    if cal_positions == 1:
        print("\nStep 2: Capturing CALIBRATION z-stack...")
        cal_frames: list[ZStackFrame] | list[list[ZStackFrame]] = capture_zstack(
            client, base_url, bf_channel, bf_channel_config,
            ref_z_mm, cal_range_um, cal_steps,
            laf_exposure_ms, laf_gain, "CAL"
        )
    else:
        print(f"\nStep 2: Capturing CALIBRATION z-stacks at {cal_positions} positions...")
        # Generate positions in a small grid around reference
        # Positions arranged in a cross pattern: center, +X, -X, +Y, -Y, etc.
        offsets = [(0, 0)]  # Start with center
        for i in range(1, cal_positions):
            if i % 4 == 1:
                offsets.append((cal_offset_mm * ((i + 3) // 4), 0))  # +X
            elif i % 4 == 2:
                offsets.append((-cal_offset_mm * ((i + 2) // 4), 0))  # -X
            elif i % 4 == 3:
                offsets.append((0, cal_offset_mm * ((i + 1) // 4)))  # +Y
            else:
                offsets.append((0, -cal_offset_mm * (i // 4)))  # -Y

        cal_stacks: list[list[ZStackFrame]] = []
        for pos_idx, (dx, dy) in enumerate(offsets[:cal_positions]):
            pos_x = ref_x_mm + dx
            pos_y = ref_y_mm + dy
            print(f"\n  Position {pos_idx + 1}/{cal_positions}: X={pos_x:.3f}, Y={pos_y:.3f} mm")
            client.move_to(pos_x, pos_y, ref_z_mm)

            stack = capture_zstack(
                client, base_url, bf_channel, bf_channel_config,
                ref_z_mm, cal_range_um, cal_steps,
                laf_exposure_ms, laf_gain, f"CAL-{pos_idx + 1}"
            )
            cal_stacks.append(stack)

        cal_frames = cal_stacks
        # Return to reference position
        client.move_to(ref_x_mm, ref_y_mm, ref_z_mm)

    # Step 3: Return to reference and capture test z-stack
    print("\nStep 3: Capturing TEST z-stack...")
    client.move_to_z(ref_z_mm)  # Return to reference
    test_frames = capture_zstack(
        client, base_url, bf_channel, bf_channel_config,
        ref_z_mm, test_range_um, test_steps,
        laf_exposure_ms, laf_gain, "TEST"
    )

    # Step 4: Return to starting position
    print(f"\nStep 4: Returning to starting position ({start_x:.4f}, {start_y:.4f}, {start_z:.4f} mm)...")
    client.move_to(start_x, start_y, start_z)

    print(f"\nAcquisition complete!")
    if cal_positions == 1:
        print(f"  Calibration: {len(cal_frames)} frames")  # type: ignore
    else:
        print(f"  Calibration: {cal_positions} positions x {len(cal_frames[0])} frames")  # type: ignore
    print(f"  Test: {len(test_frames)} frames")

    return cal_frames, test_frames
