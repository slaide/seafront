#!/usr/bin/env python3
"""
Run an imaging protocol from the outside, one API request at a time.

This performs the same steps as the server's built-in acquisition
(seafront/server/protocol.py: ProtocolGenerator.generate) but issues each of them as a
plain HTTP request from a client, including taking the device lock first. The output
directory layout, file names and OME-TIFF metadata match what the server writes. The point
is to demonstrate that an external client has exactly the capabilities of the built-in
protocol; nothing here touches hardware directly, and the only seafront code reused is pure
data handling (position/forbidden-area math, OME metadata construction).

Per-command mapping (built-in command -> API route):

  EstablishHardwareConnection         POST /api/action/establish_hardware_connection
  MC_getLastPosition                  POST /api/get_info/current_state  (see note below)
  MoveTo                              POST /api/action/move_to
  AutofocusApproachTargetDisplacement POST /api/action/laser_autofocus_move_to_target_offset
  ChannelSnapshot                     POST /api/action/snap_channel
  (image pixels)                      WS   /ws/get_info/acquired_image
  (lock)                              POST /api/lock/acquire, /api/lock/release

Usage:
  uv run scripts/acquire_via_api.py --protocol default.json
  uv run scripts/acquire_via_api.py --protocol-file my_protocol.json --owner "pipeline 3" --output-dir ./out
  uv run scripts/acquire_via_api.py --list-protocols

Exit codes: 0 done, 1 error, 2 microscope locked by someone else, 130 interrupted.

Known difference to the server's output: the built-in protocol records the raw microcontroller
position in OME metadata (MC_getLastPosition), the API only reports the calibrated stage position
(current_state), i.e. the frame the protocol itself is specified in. Positions in the two OME files
therefore differ by the calibration offsets; file names, pixels and everything else match.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import getpass
import json
import math
import re
import socket
import sys
import time
import typing as tp
import urllib.error
import urllib.request
from pathlib import Path

import json5
import numpy as np
import seaconfig as sc
import tifffile
from websockets.sync.client import connect as ws_connect

from seafront.config.basics import ChannelConfig
from seafront.server import commands as cmd
from seafront.server.protocol import (
    _format_well_name_with_padding,
    build_ome_instrument,
    build_ome_metadata,
)

# --- constants mirrored from ProtocolGenerator.generate -------------------------------------
Z_STACK_COUNTER_BACKLASH_MM = 40e-3
PIXEL_SIZE_UM = 900 / 3000
DISPLACEMENT_THRESHOLD_MM = 0.5e-3
UNCOMPENSATED_Z_FAILURE_THRESHOLD_MM = 10e-3
AUTOFOCUS_NUM_ATTEMPTS_MAX = 3

REQUEST_TIMEOUT_S = 600.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


class ApiError(Exception):
    def __init__(self, status: int, detail: tp.Any):
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


class SeafrontClient:
    """Thin HTTP/WS client. Every request carries target_device and, once held, lock_token."""

    def __init__(self, server: str, target_device: str = "", lock_token: str | None = None):
        self.server = server.rstrip("/")
        self.target_device = target_device
        self.lock_token = lock_token
        self._image_ws: tp.Any = None

    def post(self, path: str, body: dict[str, tp.Any] | None = None) -> tp.Any:
        payload: dict[str, tp.Any] = dict(body or {})
        payload["target_device"] = self.target_device
        if self.lock_token:
            payload["lock_token"] = self.lock_token
        req = urllib.request.Request(
            self.server + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            raw = e.read().decode(errors="replace")
            try:
                detail = json.loads(raw).get("detail", raw)
            except ValueError:
                detail = raw
            raise ApiError(e.code, detail) from None

    # -- the built-in protocol's primitives, as requests ---------------------------------------

    def current_position(self) -> cmd.Position:
        state = self.post("/api/get_info/current_state")
        return cmd.Position(**state["adapter_state"]["stage_position"])

    def move_to(self, x_mm: float | None = None, y_mm: float | None = None, z_mm: float | None = None) -> None:
        body = {k: v for k, v in (("x_mm", x_mm), ("y_mm", y_mm), ("z_mm", z_mm)) if v is not None}
        self.post("/api/action/move_to", body)

    def machine_defaults(self) -> dict[str, dict[str, tp.Any]]:
        return {item["handle"]: item for item in self.post("/api/get_features/machine_defaults")}

    def fetch_image(self, channel_handle: str) -> np.ndarray:
        """Full-resolution pixels of the last image taken in this channel (same data the server stores)."""
        for attempt in range(2):
            try:
                if self._image_ws is None:
                    ws_url = re.sub(r"^http", "ws", self.server) + "/ws/get_info/acquired_image"
                    self._image_ws = ws_connect(ws_url, max_size=None, open_timeout=30)
                    self._image_ws.send(json.dumps({"target_device": self.target_device}))
                ws = self._image_ws
                ws.send(channel_handle)
                meta = json.loads(ws.recv(timeout=REQUEST_TIMEOUT_S))
                if not meta:
                    raise RuntimeError(f"server has no image for channel '{channel_handle}'")
                ws.send("1")  # downsample factor 1 = full resolution
                data = ws.recv(timeout=REQUEST_TIMEOUT_S)
                assert isinstance(data, bytes)
                dtype = {8: np.uint8, 16: np.uint16}.get(int(meta["bit_depth"]))
                if dtype is None:
                    raise RuntimeError(f"unsupported bit depth {meta['bit_depth']}")
                return np.frombuffer(data, dtype=dtype).reshape(int(meta["height"]), int(meta["width"])).copy()
            except (OSError, ConnectionError) as e:
                self._image_ws = None
                if attempt == 1:
                    raise RuntimeError(f"image websocket failed: {e}") from e
        raise AssertionError("unreachable")

    def close(self) -> None:
        if self._image_ws is not None:
            try:
                self._image_ws.close()
            finally:
                self._image_ws = None


# --- protocol geometry (mirrors ProtocolGenerator) --------------------------------------------

Well = sc.PlateWellConfig
Site = sc.AcquisitionWellSiteConfigurationSiteSelectionItem


class Plane(tp.NamedTuple):
    plane_index: int
    z_mm: float
    channel: sc.AcquisitionChannelConfig


def greedy_well_order(wells: list[Well]) -> list[Well]:
    wells = sorted(wells, key=lambda w: (w.row, w.col))
    if len(wells) <= 1:
        return wells
    current = wells.pop(0)
    ordered = [current]
    while wells:
        nearest = min(wells, key=lambda w: math.hypot(w.col - current.col, w.row - current.row))
        wells.remove(nearest)
        ordered.append(nearest)
        current = nearest
    return ordered


def optimized_site_order(
    sites: list[Site], position_of: tp.Callable[[Site], tuple[float, float]], last_xy: tuple[float, float]
) -> list[Site]:
    sites = list(sites)
    if len(sites) <= 1:
        return sites
    start = min(sites, key=lambda s: math.dist(position_of(s), last_xy))
    sites.remove(start)
    ordered = [start]
    current = start
    while sites:
        nearest = min(sites, key=lambda s: math.hypot(s.col - current.col, s.row - current.row))
        sites.remove(nearest)
        ordered.append(nearest)
        current = nearest
    return ordered


def order_channels(channels: list[sc.AcquisitionChannelConfig], imaging_order: str) -> list[sc.AcquisitionChannelConfig]:
    if imaging_order == "wavelength_order":
        def key(ch: sc.AcquisitionChannelConfig) -> tuple[int, str]:
            if getattr(ch, "is_brightfield", False):
                return (0, ch.name)
            return (getattr(ch, "emission_wavelength_nm", 500), ch.name)
        return sorted(channels, key=key, reverse=True)
    return channels  # protocol_order, z_order (z sort happens on planes)


def z_planes(channels: list[sc.AcquisitionChannelConfig], reference_z_mm: float, imaging_order: str) -> list[Plane]:
    planes: list[Plane] = []
    for ch in order_channels(channels, imaging_order):
        dz = ch.delta_z_um * 1e-3
        base_z = reference_z_mm + ch.z_offset_um * 1e-3 - ((ch.num_z_planes - 1) / 2) * dz
        for i in range(ch.num_z_planes):
            planes.append(Plane(i, base_z + i * dz, ch))
    if imaging_order == "z_order":
        planes.sort(key=lambda p: p.z_mm)
    return planes


# --- the run ---------------------------------------------------------------------------------


class Summary(tp.NamedTuple):
    output_dir: Path
    num_images: int
    bytes_written: int
    seconds: float


def _store_image(
    entry: cmd.ImageStoreEntry,
    plane_index: int,
    compression: str,
    protocol: sc.AcquisitionConfig,
    hardware_channels: list[ChannelConfig],
    ome_template: tp.Any,
    autofocus_succeeded: bool,
) -> int:
    """Same on-disk result as seafront.server.protocol.store_image."""
    assert entry.info.storage_path is not None
    path = Path(entry.info.storage_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ome_xml = build_ome_metadata(
        ome_template=ome_template,
        image_entry=entry,
        channel_config=entry.info.channel,
        position=entry.info.position,
        plane_index=plane_index,
        pixel_size_um=PIXEL_SIZE_UM,
        project_name=protocol.project_name,
        plate_name=protocol.plate_name,
        cell_line=protocol.cell_line,
        hardware_channels=hardware_channels,
        autofocus_succeeded=autofocus_succeeded,
    )
    tifffile.imwrite(
        path,
        entry._img,
        compression=compression,
        compressionargs={},
        maxworkers=1,
        description=ome_xml,
        photometric="minisblack",
    )
    return path.stat().st_size


def run_protocol(client: SeafrontClient, protocol: sc.AcquisitionConfig, out_root: Path, compression: str) -> Summary:
    md = client.machine_defaults()

    def cfg(handle: str, default: tp.Any = None) -> tp.Any:
        item = md.get(handle)
        return default if item is None else item["value"]

    def cfg_bool(handle: str, default: bool) -> bool:
        v = cfg(handle)
        return default if v is None else v == "yes"

    def cfg_int(handle: str, default: int) -> int:
        v = cfg(handle)
        return default if v is None else int(v)

    plate = protocol.wellplate_type
    grid = protocol.grid
    wells = [w for w in protocol.plate_wells if w.selected]
    sites = [s for s in grid.mask if s.selected]
    channels = [c for c in protocol.channels if c.enabled]
    if not wells or not sites or not channels:
        raise RuntimeError(f"nothing to acquire: {len(wells)} wells, {len(sites)} sites, {len(channels)} channels")

    # the site grid is centred in the well
    site_topleft_x_mm = plate.Well_size_x_mm / 2 - ((grid.num_x - 1) * grid.delta_x_mm) / 2
    site_topleft_y_mm = plate.Well_size_y_mm / 2 - ((grid.num_y - 1) * grid.delta_y_mm) / 2

    def site_position(well: Well, site: Site) -> tuple[float, float]:
        return (
            plate.get_well_offset_x(well.well_name) + site_topleft_x_mm + site.col * grid.delta_x_mm,
            plate.get_well_offset_y(well.well_name) + site_topleft_y_mm + site.row * grid.delta_y_mm,
        )

    # same refusal as /api/acquisition/start, before anything moves
    if protocol.autofocus_enabled and not cfg_bool("laser.autofocus.calibration.is_calibrated", False):
        raise RuntimeError("laser autofocus is enabled, but not calibrated")

    num_planes_per_site = sum(c.num_z_planes for c in channels)
    num_images_total = len(wells) * len(sites) * num_planes_per_site * grid.num_t
    log(f"protocol: {len(wells)} wells x {len(sites)} sites x {num_planes_per_site} channel/z planes x {grid.num_t} timepoints = {num_images_total} images")

    # output directory, laid out like the server's (base/project/plate_timestamp[/tN]/file)
    stamp = sc.datetime2str(dt.datetime.now(dt.UTC))
    project_out = out_root / protocol.project_name / f"{protocol.plate_name}_{stamp}"
    project_out.mkdir(parents=True)
    (project_out / "config.json").write_text(protocol.model_dump_json())

    # -- EstablishHardwareConnection
    client.post("/api/action/establish_hardware_connection")

    # -- pre-flight: every planned position against the server's forbidden areas
    areas_raw = cfg("protocol.forbidden_areas") or []
    forbidden = cmd.ForbiddenAreaList.model_validate({"areas": areas_raw}) if areas_raw else None
    problems: list[str] = []
    for well in wells:
        for site in sites:
            x, y = site_position(well, site)
            bad, why = cmd.positionIsForbidden(x, y, forbidden_areas=forbidden)
            if bad:
                problems.append(f"well {well.well_name} site ({site.col},{site.row}) at ({x:.1f},{y:.1f}) mm: {why}")
    if problems:
        raise RuntimeError("positions in forbidden areas:\n  " + "\n  ".join(problems))
    log("pre-flight: no planned position is in a forbidden area")

    # -- OME template, from the same camera identity the server uses
    camera = client.post("/api/get_features/hardware_capabilities")["main_camera"]
    hardware_channels = [ChannelConfig(**ch) for ch in (cfg("imaging.channels") or [])]
    mag = re.match(r"(\d+)x", str(cfg("camera.main.objective", "")))
    ome_template = build_ome_instrument(
        microscope_name=client.target_device,
        camera_vendor=camera["vendor"],
        camera_model=camera["model"],
        camera_sn=camera["serial_number"],
        hardware_channels=hardware_channels,
        objective_magnification=int(mag.group(1)) if mag else None,
    )
    pixel_format = str(cfg("camera.main.pixel_format", "mono8"))
    imaging_order = str(cfg("imaging.order", "protocol_order"))

    # file naming options (handles as registered in seafront/config/handles.py)
    use_channel_name = cfg_bool("image.filename.use_channel_name", True)
    xy_start = cfg_int("image.filename.xy_index_start", 0)
    z_start = cfg_int("image.filename.z_index_start", 0)
    site_start = cfg_int("image.filename.site_index_start", 1)
    zero_pad = cfg_bool("image.filename.zero_pad_column", True)

    # -- z reference: current z, or the laser autofocus reference when autofocus is on
    reference_z_mm = client.current_position().z_pos_mm
    refz = cfg("laser.autofocus.calibration.ref_z_mm")
    if protocol.autofocus_enabled and refz is not None:
        reference_z_mm = float(refz)
        client.move_to(z_mm=reference_z_mm)

    ordered_wells = greedy_well_order(wells)
    current_xy = (0.0, 0.0)
    protocol_dict = protocol.model_dump()

    start = time.time()
    num_images = 0
    bytes_written = 0
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    pending: list[concurrent.futures.Future[int]] = []

    def drain(block: bool) -> None:
        nonlocal bytes_written, pending
        done = [f for f in pending if f.done()] if not block else list(pending)
        for f in done:
            bytes_written += f.result()  # re-raises storage errors
        pending = [f for f in pending if not f.done()]

    try:
        for timepoint in range(1, grid.num_t + 1):
            t_start = time.time()
            if grid.num_t > 1:
                log(f"timepoint {timepoint}/{grid.num_t}")
            for well_index, well in enumerate(ordered_wells):
                ordered_sites = optimized_site_order(sites, lambda s, w=well: site_position(w, s), current_xy)
                for site_index, site in enumerate(ordered_sites):
                    x, y = site_position(well, site)
                    client.move_to(x_mm=x, y_mm=y)  # MoveTo(x, y)
                    current_xy = (x, y)

                    autofocus_succeeded = False
                    if protocol.autofocus_enabled:
                        for _attempt in range(AUTOFOCUS_NUM_ATTEMPTS_MAX):
                            res = client.post(
                                "/api/action/laser_autofocus_move_to_target_offset",
                                {"target_offset_um": 0, "config_file": protocol_dict, "pre_approach_refz": False},
                            )
                            if math.fabs(res["uncompensated_offset_mm"]) > UNCOMPENSATED_Z_FAILURE_THRESHOLD_MM:
                                client.move_to(z_mm=reference_z_mm)
                                break
                            if res["num_compensating_moves"] == 0 or res["reached_threshold"]:
                                autofocus_succeeded = True
                                break
                        reference_z_mm = client.current_position().z_pos_mm
                    else:
                        client.move_to(z_mm=reference_z_mm)

                    planes = z_planes(channels, reference_z_mm, imaging_order)
                    if not planes:
                        continue
                    first_z = planes[0].z_mm
                    client.move_to(z_mm=first_z - Z_STACK_COUNTER_BACKLASH_MM)  # backlash compensation
                    client.move_to(z_mm=first_z)

                    for plane in planes:
                        if math.fabs(plane.z_mm - client.current_position().z_pos_mm) > DISPLACEMENT_THRESHOLD_MM:
                            client.move_to(z_mm=plane.z_mm)
                        position = client.current_position()

                        # ChannelSnapshot, then the pixels
                        client.post("/api/action/snap_channel", {"channel": plane.channel.model_dump(), "machine_config": []})
                        img = client.fetch_image(plane.channel.handle)

                        channel_identifier = plane.channel.name.replace(" ", "_") if use_channel_name else plane.channel.handle
                        well_name = _format_well_name_with_padding(well.well_name) if zero_pad else well.well_name
                        filename = (
                            f"{well_name}_s{site_index + site_start}_x{site.col + xy_start}_y{site.row + xy_start}"
                            f"_z{plane.plane_index + z_start}_{channel_identifier}.tiff"
                        )
                        storage_path = project_out / (f"t{timepoint - 1}" if grid.num_t > 1 else "") / filename

                        entry = cmd.ImageStoreEntry(
                            pixel_format=pixel_format,
                            info=cmd.ImageStoreInfo(
                                channel=plane.channel,
                                width_px=img.shape[1],
                                height_px=img.shape[0],
                                timestamp=time.time(),
                                position=cmd.SitePosition(
                                    well_name=well.well_name,
                                    site_x=site.col, site_y=site.row, site_z=plane.plane_index,
                                    x_offset_mm=0, y_offset_mm=0, z_offset_mm=0,
                                    position=position,
                                ),
                                storage_path=str(storage_path),
                            ),
                        )
                        entry._img = img
                        pending.append(pool.submit(
                            _store_image, entry, plane.plane_index, compression, protocol,
                            hardware_channels, ome_template, autofocus_succeeded,
                        ))
                        drain(block=False)
                        num_images += 1
                        log(
                            f"[{num_images}/{num_images_total}] well {well.well_name} ({well_index + 1}/{len(ordered_wells)}) "
                            f"site {site_index + 1}/{len(ordered_sites)} z{plane.plane_index} '{plane.channel.name}' -> {storage_path.name}"
                        )

            if grid.num_t > timepoint:
                elapsed = time.time() - t_start
                delta_t_s = grid.delta_t.h * 3600 + grid.delta_t.m * 60 + grid.delta_t.s
                remaining = delta_t_s - elapsed
                if remaining < 0:
                    log(f"warning: timepoint took {elapsed:.1f}s, longer than the {delta_t_s}s interval")
                while remaining > 0:
                    time.sleep(min(remaining, 0.5))
                    remaining -= 0.5
        drain(block=True)
    finally:
        pool.shutdown(wait=True)

    return Summary(project_out, num_images, bytes_written, time.time() - start)


# --- CLI -------------------------------------------------------------------------------------


def load_protocol(client: SeafrontClient, name: str | None, file: Path | None) -> sc.AcquisitionConfig:
    if file is not None:
        with file.open() as f:
            return sc.AcquisitionConfig.model_validate(json5.load(f))
    assert name is not None
    resp = client.post("/api/acquisition/config_fetch", {"config_file": name})
    return sc.AcquisitionConfig.model_validate(resp["file"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="http://127.0.0.1:5000", help="seafront server URL")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--protocol", metavar="NAME", help="protocol stored on the server (see --list-protocols)")
    src.add_argument("--protocol-file", type=Path, metavar="PATH", help="local protocol JSON file")
    src.add_argument("--list-protocols", action="store_true", help="list protocols stored on the server and exit")
    parser.add_argument("--output-dir", type=Path, default=Path("acquisitions"), help="where to write images (default: ./acquisitions)")
    parser.add_argument("--owner", default=f"acquire_via_api ({getpass.getuser()}@{socket.gethostname()})", help="lock owner label shown to other clients")
    parser.add_argument("--ttl-s", type=float, default=600.0, help="lock lease length; renewed by every request")
    parser.add_argument("--lock-token", help="reuse an existing lease instead of taking a new one (it is not released afterwards)")
    parser.add_argument("--compression", choices=["LZW", "zlib"], default="LZW")
    args = parser.parse_args()

    client = SeafrontClient(args.server, lock_token=args.lock_token)
    try:
        state = client.post("/api/get_info/current_state")  # empty target_device is the discovery bypass
    except (urllib.error.URLError, ApiError) as e:
        log(f"cannot reach server at {args.server}: {e}")
        return 1
    client.target_device = state["microscope_name"]
    log(f"connected to '{client.target_device}' at {args.server}")

    if args.list_protocols:
        for entry in client.post("/api/acquisition/config_list")["configs"]:
            print(entry["filename"], "-", entry.get("comment") or "")
        return 0
    if not args.protocol and not args.protocol_file:
        parser.error("one of --protocol, --protocol-file, --list-protocols is required")

    # -- lock: fail fast if someone else has the microscope
    we_hold_lock = False
    if args.lock_token is None:
        try:
            lease = client.post("/api/lock/acquire", {"owner": args.owner, "ttl_s": args.ttl_s})
        except ApiError as e:
            if e.status == 423:
                d = e.detail if isinstance(e.detail, dict) else {}
                log(f"microscope is locked by '{d.get('owner')}' until {d.get('expires_at')}; not starting")
                return 2
            raise
        client.lock_token = lease["token"]
        we_hold_lock = True
        log(f"holding device lock as '{lease['owner']}' (lock {lease['lock_id']})")

    try:
        protocol = load_protocol(client, args.protocol, args.protocol_file)
        summary = run_protocol(client, protocol, args.output_dir, args.compression)
        log(f"done: {summary.num_images} images, {summary.bytes_written / 1e6:.1f} MB in {summary.seconds:.1f}s -> {summary.output_dir}")
        return 0
    except KeyboardInterrupt:
        log("interrupted; cancelling the command in flight")
        try:
            client.post("/api/action/cancel_current")
        except Exception as e:
            log(f"cancel failed: {e}")
        return 130
    except ApiError as e:
        log(f"server refused: {e}")
        return 1
    except Exception as e:
        log(f"error: {e}")
        return 1
    finally:
        client.close()
        if we_hold_lock:
            try:
                client.post("/api/lock/release")
                log("released device lock")
            except Exception as e:
                log(f"could not release lock: {e}")


if __name__ == "__main__":
    sys.exit(main())
