#!/usr/bin/env python3
"""Compare image histograms before/after Toupcam temperature control.

Workflow:
1. Capture N fluorescence and brightfield snapshots (baseline).
2. Apply Toupcam machine-config temperature target (absolute or current+delta).
3. Wait until camera reaches target temperature.
4. Capture N fluorescence and brightfield snapshots again.
5. Save histogram comparison plot.
"""

from __future__ import annotations

import argparse
import asyncio
import http.client
import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TOUP_TEMP_CURRENT_HANDLE = "camera.toupcam.temperature.current_c"
TOUP_TEMP_TEC_ENABLED_HANDLE = "camera.toupcam.tec_enabled"
TOUP_TEMP_TARGET_MODE_HANDLE = "camera.toupcam.temperature.target_mode"
TOUP_TEMP_TARGET_C_HANDLE = "camera.toupcam.temperature.target_c"
TOUP_TEMP_DELTA_C_HANDLE = "camera.toupcam.temperature.delta_from_current_c"


def _request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout_s: float = 30.0,
) -> Any:
    req = urllib.request.Request(url=url, method=method.upper())
    req.add_header("Content-Type", "application/json")
    data = json.dumps(payload).encode("utf-8") if payload is not None else None

    try:
        with urllib.request.urlopen(req, data=data, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed for {url}: {exc}") from exc


def _post_api(
    base_url: str,
    microscope_name: str,
    endpoint: str,
    payload: dict[str, Any] | None = None,
) -> Any:
    body = dict(payload or {})
    body["target_device"] = microscope_name
    return _request_json("POST", f"{base_url}{endpoint}", payload=body)


def _post_api_with_retry(
    base_url: str,
    microscope_name: str,
    endpoint: str,
    payload: dict[str, Any] | None = None,
    retries: int = 4,
    delay_s: float = 1.5,
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return _post_api(base_url, microscope_name, endpoint, payload=payload)
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, socket.timeout) as exc:
            last_exc = exc
            if attempt == retries:
                break
            print(
                f"  transient API failure on {endpoint} "
                f"(attempt {attempt}/{retries}): {exc}. Retrying in {delay_s:.1f}s"
            )
            try:
                _post_api(base_url, microscope_name, "/api/action/establish_hardware_connection")
            except Exception:
                pass
            time.sleep(delay_s)
    assert last_exc is not None
    raise last_exc


def _extract_config_item_value(item: dict[str, Any]) -> Any:
    for key in ("value", "objectvalue", "strvalue", "boolvalue", "intvalue", "floatvalue"):
        if key in item and item[key] is not None:
            return item[key]
    return None


def _find_config_item(items: list[dict[str, Any]], handle: str) -> dict[str, Any] | None:
    for item in items:
        if item.get("handle") == handle:
            return item
    return None


async def _fetch_image_via_websocket(
    base_url: str, microscope_name: str, channel_handle: str
) -> np.ndarray:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError(
            "websockets package is required for image retrieval. Install with: uv add websockets"
        ) from exc

    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/ws/get_info/acquired_image"

    async with websockets.connect(ws_url, max_size=100 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"target_device": microscope_name}))
        await ws.send(channel_handle)

        metadata_raw = await ws.recv()
        metadata = json.loads(metadata_raw)
        if not metadata:
            raise RuntimeError(f"no image metadata returned for channel '{channel_handle}'")

        width = int(metadata["width"])
        height = int(metadata["height"])
        bit_depth = int(metadata["bit_depth"])

        await ws.send("1")
        img_bytes = await ws.recv()
        if not isinstance(img_bytes, (bytes, bytearray)):
            raise RuntimeError("expected binary image payload")

        if bit_depth <= 8:
            dtype = np.uint8
        else:
            dtype = np.uint16

        img = np.frombuffer(img_bytes, dtype=dtype).reshape((height, width))
        return img.copy()


def _build_channel_payload(
    channel: dict[str, Any],
    exposure_ms: float,
    gain_db: float,
    illum_perc: float,
) -> dict[str, Any]:
    payload = dict(channel)
    payload["enabled"] = True
    payload["exposure_time_ms"] = exposure_ms
    payload["analog_gain"] = gain_db
    payload["illum_perc"] = illum_perc
    payload["num_z_planes"] = 1
    payload["delta_z_um"] = 1.0
    payload["z_offset_um"] = 0.0
    return payload


def _pick_channel(
    channels: list[dict[str, Any]],
    requested_handle: str,
    name_hint: str,
) -> dict[str, Any] | None:
    by_handle = {str(ch.get("handle")): ch for ch in channels}
    picked = by_handle.get(requested_handle)
    if picked is not None:
        return picked

    hint = name_hint.lower()
    for ch in channels:
        name = str(ch.get("name", "")).lower()
        handle = str(ch.get("handle", "")).lower()
        if hint in name or hint in handle:
            return ch
    return None


def _resolve_filter_handles(
    machine_defaults: list[dict[str, Any]],
    fluo_filter_handle: str | None,
    bf_filter_handle: str | None,
) -> tuple[str | None, str | None]:
    available_item = _find_config_item(machine_defaults, "filter.wheel.available")
    available_value = _extract_config_item_value(available_item or {})
    wheel_available = str(available_value).lower() in ("yes", "true", "1")
    if not wheel_available:
        return None, None

    config_item = _find_config_item(machine_defaults, "filter.wheel.configuration")
    filters_value = _extract_config_item_value(config_item or {})
    filters = filters_value if isinstance(filters_value, list) else []
    if not filters:
        raise RuntimeError("filter wheel is marked available, but no filter configuration exists")

    all_handles = [str(f.get("handle", "")) for f in filters]
    if fluo_filter_handle is not None and fluo_filter_handle not in all_handles:
        raise RuntimeError(
            f"requested fluorescence filter '{fluo_filter_handle}' not in configured filters: {all_handles}"
        )
    if bf_filter_handle is not None and bf_filter_handle not in all_handles:
        raise RuntimeError(
            f"requested brightfield filter '{bf_filter_handle}' not in configured filters: {all_handles}"
        )

    selected_fluo = fluo_filter_handle
    selected_bf = bf_filter_handle

    if selected_fluo is None and all_handles:
        selected_fluo = all_handles[0]

    if selected_bf is None:
        for handle in all_handles:
            low = handle.lower()
            if "none" in low or "empty" in low:
                selected_bf = handle
                break
        if selected_bf is None and all_handles:
            selected_bf = all_handles[0]

    return selected_fluo, selected_bf


def _capture_replicates(
    base_url: str,
    microscope_name: str,
    channel_payload: dict[str, Any],
    repeats: int,
    saturation_guard: bool = False,
    saturation_threshold: float = 0.995,
) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    handle = str(channel_payload["handle"])
    for i in range(repeats):
        _post_api_with_retry(
            base_url,
            microscope_name,
            "/api/action/snap_channel",
            payload={"channel": channel_payload},
            retries=5,
            delay_s=2.0,
        )
        img = asyncio.run(_fetch_image_via_websocket(base_url, microscope_name, handle))
        img_min = float(img.min())
        img_max = float(img.max())
        img_mean = float(img.mean())
        sat_value = 255 if img.dtype == np.uint8 else 65535
        sat_fraction = float((img == sat_value).mean())
        print(
            f"  captured {handle} replicate {i + 1}/{repeats} "
            f"(min={img_min:.1f} max={img_max:.1f} mean={img_mean:.2f} sat={sat_fraction*100:.2f}%)"
        )

        if saturation_guard and sat_fraction >= saturation_threshold:
            raise RuntimeError(
                f"{handle} capture appears saturated/invalid: "
                f"{sat_fraction*100:.2f}% pixels at {sat_value}"
            )

        images.append(img)
    return images


def _compute_mean_hist(images: list[np.ndarray], bins: int) -> tuple[np.ndarray, np.ndarray]:
    if not images:
        raise ValueError("images list must not be empty")

    first = images[0]
    max_value = 255 if first.dtype == np.uint8 else 65535
    hist_stack = []
    edges = None
    for img in images:
        hist, edges = np.histogram(img.ravel(), bins=bins, range=(0, max_value + 1), density=True)
        hist_stack.append(hist)

    assert edges is not None
    mean_hist = np.mean(np.stack(hist_stack, axis=0), axis=0)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, mean_hist


def _get_toupcam_temp_c(base_url: str, microscope_name: str) -> float:
    # Trigger a state poll so server-side temperature read happens every time.
    _post_api(base_url, microscope_name, "/api/get_info/current_state")
    machine_defaults = _post_api(base_url, microscope_name, "/api/get_features/machine_defaults")
    if not isinstance(machine_defaults, list):
        raise RuntimeError("machine defaults response is not a list")

    temp_item = _find_config_item(machine_defaults, TOUP_TEMP_CURRENT_HANDLE)
    if temp_item is None:
        raise RuntimeError(
            f"temperature handle '{TOUP_TEMP_CURRENT_HANDLE}' not found in machine defaults"
        )
    temp_value = _extract_config_item_value(temp_item)
    if temp_value is None:
        raise RuntimeError("current temperature value is missing")
    return float(temp_value)


def _flush_temperature_config(
    base_url: str,
    microscope_name: str,
    target_mode: str,
    target_c: float,
    delta_c: float,
    tec_enabled: bool,
) -> None:
    payload = [
        {
            "handle": TOUP_TEMP_TEC_ENABLED_HANDLE,
            "name": "toupcam TEC enabled",
            "value_kind": "option",
            "value": "yes" if tec_enabled else "no",
            "options": [
                {"name": "No", "handle": "no"},
                {"name": "Yes", "handle": "yes"},
            ],
            "frozen": False,
        },
        {
            "handle": TOUP_TEMP_TARGET_MODE_HANDLE,
            "name": "toupcam temperature target mode",
            "value_kind": "option",
            "value": target_mode,
            "options": [
                {"name": "Absolute Target", "handle": "absolute"},
                {"name": "Current + Delta", "handle": "relative_to_current"},
            ],
            "frozen": False,
        },
        {
            "handle": TOUP_TEMP_TARGET_C_HANDLE,
            "name": "toupcam temperature absolute target [C]",
            "value_kind": "float",
            "value": float(target_c),
            "frozen": False,
        },
        {
            "handle": TOUP_TEMP_DELTA_C_HANDLE,
            "name": "toupcam temperature delta from current [C]",
            "value_kind": "float",
            "value": float(delta_c),
            "frozen": False,
        },
    ]
    _post_api(
        base_url,
        microscope_name,
        "/api/action/machine_config_flush",
        payload={"machine_config": payload},
    )


def _wait_for_target_temperature(
    base_url: str,
    microscope_name: str,
    target_c: float,
    tolerance_c: float,
    timeout_s: float,
    poll_interval_s: float,
    plateau_delta_c: float | None = None,
    plateau_window_s: float | None = None,
    stop_when_within_abs_tolerance: bool = False,
) -> float:
    start = time.time()
    plateau_reference_time = start
    plateau_reference_temp = _get_toupcam_temp_c(base_url, microscope_name)
    while True:
        current = _get_toupcam_temp_c(base_url, microscope_name)
        elapsed = time.time() - start
        print(f"  cooling status: current={current:.1f}C target={target_c:.1f}C elapsed={elapsed:.0f}s")
        if stop_when_within_abs_tolerance:
            if abs(current - target_c) <= tolerance_c:
                return current
        elif current <= target_c + tolerance_c:
            return current

        if plateau_delta_c is not None and plateau_window_s is not None:
            plateau_elapsed = time.time() - plateau_reference_time
            if plateau_elapsed >= plateau_window_s:
                delta = abs(current - plateau_reference_temp)
                if delta < plateau_delta_c:
                    print(
                        "  cooling plateau reached: "
                        f"change={delta:.2f}C in {plateau_elapsed:.0f}s "
                        f"(threshold {plateau_delta_c:.2f}C). Accepting current temperature."
                    )
                    return current
                plateau_reference_time = time.time()
                plateau_reference_temp = current

        if elapsed > timeout_s:
            raise TimeoutError(
                f"timed out waiting for target temperature: current={current:.2f}C target={target_c:.2f}C"
            )
        time.sleep(poll_interval_s)


def _wait_for_min_temperature(
    base_url: str,
    microscope_name: str,
    min_temp_c: float,
    timeout_s: float,
    poll_interval_s: float,
) -> float:
    start = time.time()
    while True:
        current = _get_toupcam_temp_c(base_url, microscope_name)
        elapsed = time.time() - start
        print(
            f"  warmup status: current={current:.1f}C min_required={min_temp_c:.1f}C elapsed={elapsed:.0f}s"
        )
        if current >= min_temp_c:
            return current
        if elapsed > timeout_s:
            raise TimeoutError(
                f"timed out waiting for warm baseline: current={current:.2f}C min_required={min_temp_c:.2f}C"
            )
        time.sleep(poll_interval_s)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Toupcam histograms with/without cooling.")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--microscope", required=True)
    parser.add_argument("--fluo-handle", default="fluo640")
    parser.add_argument("--bf-handle", default="bfledfull")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fluo-exposure-ms", type=float, default=50.0)
    parser.add_argument("--bf-exposure-ms", type=float, default=5.0)
    parser.add_argument("--fluo-gain-db", type=float, default=14.0)
    parser.add_argument("--bf-gain-db", type=float, default=10.0)
    parser.add_argument("--fluo-illum-perc", type=float, default=100.0)
    parser.add_argument("--bf-illum-perc", type=float, default=20.0)
    parser.add_argument(
        "--fluo-filter-handle",
        default=None,
        help="Filter handle for fluorescence channel when filter wheel is enabled (default: auto-first filter)",
    )
    parser.add_argument(
        "--bf-filter-handle",
        default=None,
        help="Filter handle for brightfield channel when filter wheel is enabled (default: auto none/empty or first filter)",
    )
    parser.add_argument("--temperature-tolerance-c", type=float, default=1.0)
    parser.add_argument("--cooling-timeout-s", type=float, default=900.0)
    parser.add_argument("--cooling-poll-s", type=float, default=5.0)
    parser.add_argument(
        "--cool-target-c",
        type=float,
        default=-20.0,
        help="Cooling target temperature in C (absolute)",
    )
    parser.add_argument(
        "--ambient-temp-c",
        type=float,
        default=20.0,
        help="Minimum warm baseline temperature in C before first image set",
    )
    parser.add_argument("--ambient-tolerance-c", type=float, default=0.5)
    parser.add_argument("--ambient-timeout-s", type=float, default=1800.0)
    parser.add_argument("--cooling-plateau-delta-c", type=float, default=0.1)
    parser.add_argument("--cooling-plateau-window-s", type=float, default=60.0)
    parser.add_argument("--start-x-mm", type=float, default=52.61)
    parser.add_argument("--start-y-mm", type=float, default=37.08)
    parser.add_argument("--start-z-um", type=float, default=3385.9)
    parser.add_argument(
        "--skip-start-move",
        action="store_true",
        help="Do not move to the configured start position at script startup.",
    )
    parser.add_argument("--hist-bins-8bit", type=int, default=256)
    parser.add_argument("--hist-bins-16bit", type=int, default=1024)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("toupcam_temperature_hist_compare.png"),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    base_url = args.base_url.rstrip("/")

    _post_api(base_url, args.microscope, "/api/action/establish_hardware_connection")
    if args.skip_start_move:
        print("Skipping startup move_to; using current stage position.")
    else:
        start_z_mm = args.start_z_um / 1000.0
        print(
            "Moving to start position: "
            f"X={args.start_x_mm:.2f}mm Y={args.start_y_mm:.2f}mm Z={args.start_z_um:.1f}um"
        )
        _post_api(
            base_url,
            args.microscope,
            "/api/action/move_to",
            payload={"x_mm": args.start_x_mm, "y_mm": args.start_y_mm, "z_mm": start_z_mm},
        )

    capabilities = _post_api(base_url, args.microscope, "/api/get_features/hardware_capabilities")
    channels = capabilities.get("main_camera_imaging_channels", [])
    if not isinstance(channels, list):
        raise RuntimeError("hardware capabilities missing main_camera_imaging_channels")

    fluo_channel = _pick_channel(channels, args.fluo_handle, "640")
    bf_channel = _pick_channel(channels, args.bf_handle, "bf")
    if fluo_channel is None:
        raise RuntimeError(f"fluorescence channel '{args.fluo_handle}' not found")
    if bf_channel is None:
        raise RuntimeError(f"brightfield channel '{args.bf_handle}' not found")

    machine_defaults = _post_api(base_url, args.microscope, "/api/get_features/machine_defaults")
    if not isinstance(machine_defaults, list):
        raise RuntimeError("machine defaults response is not a list")
    fluo_filter_handle, bf_filter_handle = _resolve_filter_handles(
        machine_defaults=machine_defaults,
        fluo_filter_handle=args.fluo_filter_handle,
        bf_filter_handle=args.bf_filter_handle,
    )
    if fluo_filter_handle is not None:
        print(f"Using fluorescence filter: {fluo_filter_handle}")
    if bf_filter_handle is not None:
        print(f"Using brightfield filter: {bf_filter_handle}")

    fluo_payload = _build_channel_payload(
        channel=fluo_channel,
        exposure_ms=args.fluo_exposure_ms,
        gain_db=args.fluo_gain_db,
        illum_perc=args.fluo_illum_perc,
    )
    bf_payload = _build_channel_payload(
        channel=bf_channel,
        exposure_ms=args.bf_exposure_ms,
        gain_db=args.bf_gain_db,
        illum_perc=args.bf_illum_perc,
    )
    fluo_payload["filter_handle"] = fluo_filter_handle
    bf_payload["filter_handle"] = bf_filter_handle

    current_temp_c = _get_toupcam_temp_c(base_url, args.microscope)
    ambient_temp_c = float(args.ambient_temp_c)
    print(
        "Setting baseline with TEC off and waiting for minimum warm temperature: "
        f"current={current_temp_c:.1f}C min_required={ambient_temp_c:.1f}C"
    )
    _flush_temperature_config(
        base_url,
        args.microscope,
        target_mode="absolute",
        target_c=ambient_temp_c,
        delta_c=-20.0,
        tec_enabled=False,
    )
    baseline_reached_c = _wait_for_min_temperature(
        base_url=base_url,
        microscope_name=args.microscope,
        min_temp_c=ambient_temp_c,
        timeout_s=args.ambient_timeout_s,
        poll_interval_s=args.cooling_poll_s,
    )
    print(f"Baseline ready at {baseline_reached_c:.1f}C")

    print("Capturing baseline images")
    fluo_baseline = _capture_replicates(
        base_url, args.microscope, fluo_payload, repeats=args.repeats, saturation_guard=True
    )
    bf_baseline = _capture_replicates(base_url, args.microscope, bf_payload, repeats=args.repeats)

    expected_target_c = float(args.cool_target_c)
    print(f"Applying cooling config: TEC=on mode=absolute target={expected_target_c:.1f}C")
    _flush_temperature_config(
        base_url,
        args.microscope,
        target_mode="absolute",
        target_c=expected_target_c,
        delta_c=-20.0,
        tec_enabled=True,
    )
    reached_c = _wait_for_target_temperature(
        base_url=base_url,
        microscope_name=args.microscope,
        target_c=expected_target_c,
        tolerance_c=args.temperature_tolerance_c,
        timeout_s=args.cooling_timeout_s,
        poll_interval_s=args.cooling_poll_s,
        plateau_delta_c=args.cooling_plateau_delta_c,
        plateau_window_s=args.cooling_plateau_window_s,
    )
    print(f"Reached cooling target: {reached_c:.1f}C")
    print(f"Locking TEC target to reached temperature: {reached_c:.1f}C")
    _flush_temperature_config(
        base_url,
        args.microscope,
        target_mode="absolute",
        target_c=reached_c,
        delta_c=expected_target_c - reached_c,
        tec_enabled=True,
    )
    time.sleep(1.0)

    print("Capturing cooled images")
    fluo_cooled = _capture_replicates(
        base_url, args.microscope, fluo_payload, repeats=args.repeats, saturation_guard=True
    )
    bf_cooled = _capture_replicates(base_url, args.microscope, bf_payload, repeats=args.repeats)

    fluo_bins = args.hist_bins_8bit if fluo_baseline[0].dtype == np.uint8 else args.hist_bins_16bit
    bf_bins = args.hist_bins_8bit if bf_baseline[0].dtype == np.uint8 else args.hist_bins_16bit

    fluo_x0, fluo_h0 = _compute_mean_hist(fluo_baseline, bins=fluo_bins)
    fluo_x1, fluo_h1 = _compute_mean_hist(fluo_cooled, bins=fluo_bins)
    bf_x0, bf_h0 = _compute_mean_hist(bf_baseline, bins=bf_bins)
    bf_x1, bf_h1 = _compute_mean_hist(bf_cooled, bins=bf_bins)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), tight_layout=True)

    axes[0].plot(fluo_x0, fluo_h0, label="Baseline", linewidth=1.6)
    axes[0].plot(fluo_x1, fluo_h1, label="Temp Control Active", linewidth=1.6)
    axes[0].set_title(f"Fluorescence ({args.fluo_handle})")
    axes[0].set_xlabel("Pixel Intensity")
    axes[0].set_ylabel("Density")
    axes[0].set_yscale("log")
    axes[0].set_ylim(bottom=1e-6)
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(bf_x0, bf_h0, label="Baseline", linewidth=1.6)
    axes[1].plot(bf_x1, bf_h1, label="Temp Control Active", linewidth=1.6)
    axes[1].set_title(f"Brightfield ({args.bf_handle})")
    axes[1].set_xlabel("Pixel Intensity")
    axes[1].set_ylabel("Density")
    axes[1].set_yscale("log")
    axes[1].set_ylim(bottom=1e-6)
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    fig.suptitle(
        "Toupcam Histogram Comparison\n"
        f"repeats={args.repeats}, cooling=absolute, expected_target={expected_target_c:.1f}C",
        fontsize=11,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Wrote plot: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
