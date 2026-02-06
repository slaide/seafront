#!/usr/bin/env python3
"""
Probe and snapshot utility for ToupCam cameras using Seafront's current camera API.

Usage:
  uv run python scripts/toupcam_probe.py --list
  uv run python scripts/toupcam_probe.py --usb-id <camera_id> --pixel-format mono8
"""

import argparse
from enum import IntEnum
from pathlib import Path

import numpy as np
from seaconfig import AcquisitionChannelConfig, ConfigItemOption

from seafront.config.core_config import register_core_config, register_laser_autofocus_config
from seafront.config.handles import CameraConfig, LaserAutofocusConfig
from seafront.config.registry import ConfigRegistry
from seafront.hardware.camera.toupcam_camera import ToupCamCamera


class HRESULT(IntEnum):
    S_OK = 0x00000000
    S_FALSE = 0x00000001
    E_ACCESSDENIED = 0x80070005
    E_INVALIDARG = 0x80070057
    E_NOTIMPL = 0x80004001
    E_POINTER = 0x80004003
    E_UNEXPECTED = 0x8000FFFF
    E_WRONG_THREAD = 0x8001010E
    E_GEN_FAILURE = 0x8007001F
    E_BUSY = 0x800700AA
    E_PENDING = 0x8000000A
    E_TIMEOUT = 0x8001011F
    E_FAIL = 0x80004005


HRESULT_DESCRIPTIONS: dict[HRESULT, str] = {
    HRESULT.S_OK: "Success.",
    HRESULT.S_FALSE: "No-op success (requested value already set).",
    HRESULT.E_ACCESSDENIED: "Permission denied (udev/rules/root permissions issue).",
    HRESULT.E_INVALIDARG: "Invalid argument provided to Toupcam SDK.",
    HRESULT.E_NOTIMPL: "Feature not supported by this camera model.",
    HRESULT.E_POINTER: "Invalid/null pointer passed to Toupcam SDK.",
    HRESULT.E_UNEXPECTED: "Unexpected failure (often invalid state or unsupported runtime option change).",
    HRESULT.E_WRONG_THREAD: "Toupcam API called from wrong thread context.",
    HRESULT.E_GEN_FAILURE: "Device not functioning (USB link/cable/port/hardware issue).",
    HRESULT.E_BUSY: "Camera/resource busy (already opened/used elsewhere).",
    HRESULT.E_PENDING: "Requested data not available yet.",
    HRESULT.E_TIMEOUT: "Operation timed out.",
    HRESULT.E_FAIL: "Unspecified Toupcam SDK failure.",
}


def _decode_hresult(exc: Exception) -> tuple[HRESULT | None, int | None]:
    if not exc.args:
        return None, None

    raw = exc.args[0]
    if not isinstance(raw, int):
        return None, None

    as_uint32 = raw & 0xFFFFFFFF
    try:
        return HRESULT(as_uint32), as_uint32
    except ValueError:
        return None, as_uint32


def _print_exception_with_hresult(exc: Exception) -> None:
    hresult, code = _decode_hresult(exc)
    if code is None:
        print(f"Error: {type(exc).__name__}: {exc}")
        return

    code_hex = f"0x{code:08X}"
    if hresult is None:
        print(f"Error: {type(exc).__name__}: unknown Toupcam HRESULT {code_hex} ({exc})")
        return

    description = HRESULT_DESCRIPTIONS.get(hresult, "No description available.")
    print(f"Error: {type(exc).__name__}: {hresult.name} ({code_hex}) - {description}")
    if str(exc):
        print(f"Details: {exc}")


def _configure_pixel_format(
    device_type: str,
    pixel_format: str,
) -> None:
    """
    Prepare config registry so ToupCamCamera.snap() can resolve pixel format by device type.
    """
    ConfigRegistry.reset()
    ConfigRegistry.init({})
    register_core_config(default_image_dir="/tmp", default_channels=[], default_forbidden_areas=[])

    pixel_format_options = [
        ConfigItemOption(name="8 Bit", handle="mono8"),
        ConfigItemOption(name="10 Bit", handle="mono10"),
        ConfigItemOption(name="12 Bit", handle="mono12"),
        ConfigItemOption(name="14 Bit", handle="mono14"),
        ConfigItemOption(name="16 Bit", handle="mono16"),
    ]

    if device_type == "main":
        ConfigRegistry.set_value(CameraConfig.MAIN_PIXEL_FORMAT.value, pixel_format)
        ConfigRegistry.get(CameraConfig.MAIN_PIXEL_FORMAT.value).options = pixel_format_options
    else:
        register_laser_autofocus_config()
        ConfigRegistry.set_value(LaserAutofocusConfig.CAMERA_PIXEL_FORMAT.value, pixel_format)
        ConfigRegistry.get(LaserAutofocusConfig.CAMERA_PIXEL_FORMAT.value).options = pixel_format_options


def _print_camera_list(cameras: list[ToupCamCamera]) -> None:
    if not cameras:
        print("No ToupCam cameras found.")
        return

    print("Detected ToupCam cameras:")
    for cam in cameras:
        info = cam.device_info
        print(f"- id={info.id} name={info.displayname}")
        for i, res in enumerate(info.model.res):
            print(f"  res[{i}]={res.width}x{res.height}")


def main() -> int:
    parser = argparse.ArgumentParser(description="ToupCam probe utility")
    parser.add_argument("--list", action="store_true", help="List available ToupCam cameras and exit")
    parser.add_argument("--usb-id", type=str, default=None, help="Camera ID from --list output")
    parser.add_argument(
        "--device-type",
        type=str,
        default="main",
        choices=["main", "autofocus"],
        help="Device role used to resolve pixel-format config",
    )
    parser.add_argument(
        "--pixel-format",
        type=str,
        default="mono8",
        choices=["mono8", "mono10", "mono12", "mono14", "mono16"],
        help="Requested pixel format",
    )
    parser.add_argument("--exposure-ms", type=float, default=10.0, help="Exposure time in ms")
    parser.add_argument("--gain-db", type=float, default=0.0, help="Analog gain in dB")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output .npy path for captured image",
    )
    args = parser.parse_args()

    selected: ToupCamCamera | None = None
    opened = False
    try:
        cameras = ToupCamCamera.get_all()
        if args.list:
            _print_camera_list(cameras)
            return 0

        if not cameras:
            print("No ToupCam cameras found.")
            return 1

        if args.usb_id is None:
            selected = cameras[0]
        else:
            for cam in cameras:
                if cam.sn == args.usb_id:
                    selected = cam
                    break

        if selected is None:
            print(f"Camera '{args.usb_id}' not found.")
            _print_camera_list(cameras)
            return 1

        _configure_pixel_format(device_type=args.device_type, pixel_format=args.pixel_format)

        selected.open(device_type=args.device_type)  # type: ignore[arg-type]
        opened = True

        print(f"Opened: id={selected.sn} model={selected.model_name} vendor={selected.vendor_name}")
        print(f"Sensor size after open: {selected.width}x{selected.height}")
        print(f"Supported pixel formats: {selected.get_supported_pixel_formats()}")
        print(f"Exposure limits (ms): {selected.get_exposure_time_limits().to_dict()}")
        print(f"Analog gain limits (dB): {selected.get_analog_gain_limits().to_dict()}")

        channel = AcquisitionChannelConfig(
            name="Toupcam Test",
            handle="toupcam_test",
            illum_perc=0.0,
            exposure_time_ms=args.exposure_ms,
            analog_gain=args.gain_db,
            z_offset_um=0.0,
            num_z_planes=1,
            delta_z_um=1.0,
            enabled=True,
            filter_handle=None,
        )
        img = selected.snap(channel)

        print(
            "Captured image:",
            f"shape={img.shape}",
            f"dtype={img.dtype}",
            f"min={int(np.min(img))}",
            f"max={int(np.max(img))}",
            f"mean={float(np.mean(img)):.2f}",
        )

        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.output, img)
            print(f"Saved: {args.output}")
        return 0
    except Exception as exc:
        _print_exception_with_hresult(exc)
        return 1
    finally:
        if selected is not None and opened:
            selected.close()
            print("Camera closed.")


if __name__ == "__main__":
    raise SystemExit(main())
