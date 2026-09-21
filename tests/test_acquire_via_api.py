"""
scripts/acquire_via_api.py runs the built-in imaging protocol from the outside, one API request
at a time. These tests are the proof of the claim behind it: against the mock server the script
produces the same files as /api/acquisition/start, refuses the same things for the same reasons
before anything moves, respects the device lock, and has working interrupt and error paths that
leave the lock the way they found it.

The server under test runs against a scratch copy of the seafront home (SEAFRONT_HOME), so
calibration writes and acquisitions never touch the real config or image directory.
"""

import contextlib
import json
import re
import signal
import subprocess
import sys
import time
import typing as tp
from pathlib import Path

import pytest
import seaconfig as sc
import tifffile
from ome_types import from_xml

from tests.mock_server import mock_server
from tests.test_api_operations import make_request

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "acquire_via_api.py"
TOTAL_TIMEOUT_S = 300

# the fixture picks a free port and rebinds these before any helper below runs
PORT = 0
SERVER = ""


# --------------------------------------------------------------------------------------------
# scratch server


@pytest.fixture(scope="module")
def scratch_server(tmp_path_factory):
    """mock server on a scratch seafront home, images redirected into the scratch dir"""
    global PORT, SERVER
    with mock_server(tmp_path_factory) as srv:
        PORT, SERVER = srv.port, srv.url
        yield srv.home


# --------------------------------------------------------------------------------------------
# helpers


def _post(path: str, data: dict | None = None, token: str | None = None) -> tuple[int, dict | None]:
    return make_request(f"{SERVER}{path}", data=data or {}, token=token)


def _ok(path: str, data: dict | None = None, token: str | None = None) -> dict:
    status, body = _post(path, data, token=token)
    assert status == 200, (path, status, body)
    assert body is not None
    return body


@contextlib.contextmanager
def _leased(owner: str = "test_acquire_via_api") -> tp.Iterator[str]:
    """
    hold the device lease for the block, the way the script does for its own run

    Hardware routes refuse requests without it, so anything here that drives the microscope
    directly (rather than through the script) has to take one. It is given back on the way
    out, which is also what lets the script take its own lease in the next line.
    """
    lease = _ok("/api/lock/acquire", {"owner": owner})
    try:
        yield lease["token"]
    finally:
        _post("/api/lock/release", {"lock_token": lease["token"]})


def _protocol(
    tmp_path: Path,
    wells: list[tuple[int, int]] | None = None,
    sites: int = 1,
    channels: int = 2,
    z_planes: int = 1,
    num_t: int = 1,
    delta_t_s: float = 1.0,
    autofocus: bool = False,
    name: str = "proto",
) -> tuple[Path, dict]:
    """the server's default protocol, reshaped; wells default to the plate centre"""
    proto = _ok("/api/acquisition/config_fetch", {"config_file": "default.json"})["file"]
    plate = proto["wellplate_type"]
    if wells is None:
        wells = [(plate["Num_wells_y"] // 2, plate["Num_wells_x"] // 2)]
    proto["plate_wells"] = [{"row": r, "col": c, "selected": True} for r, c in wells]
    grid = proto["grid"]
    grid["num_x"] = grid["num_y"] = sites
    grid["mask"] = [{"row": r, "col": c, "selected": True} for r in range(sites) for c in range(sites)]
    grid["num_t"] = num_t
    grid["delta_t"] = {"h": 0, "m": 0, "s": delta_t_s}
    for i, ch in enumerate(proto["channels"]):
        ch["enabled"] = i < channels
        ch["num_z_planes"] = z_planes
        ch["delta_z_um"] = 1.0
        ch["z_offset_um"] = 0.5 * i  # give channels distinct z references
    proto["autofocus_enabled"] = autofocus
    proto["project_name"] = "api-parity"
    proto["plate_name"] = name
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(proto))
    return path, proto


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--server", SERVER, *args],
        capture_output=True, text=True, timeout=TOTAL_TIMEOUT_S,
    )


def _start_script(*args: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, str(SCRIPT), "--server", SERVER, *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def _tiffs(root: Path) -> dict[str, Path]:
    return {str(f.relative_to(next(root.glob("*/*")))) : f for f in root.rglob("*.tiff")} if root.exists() and list(root.glob("*/*")) else {}


def _wait_files(root: Path, at_least: int, timeout_s: float = 30) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if len(list(root.rglob("*.tiff"))) >= at_least:
            return
        time.sleep(0.05)
    raise AssertionError(f"fewer than {at_least} images appeared under {root}")


def _move_to(position: tuple[float, float, float]) -> None:
    with _leased("parity test positioning") as token:
        _ok(
            "/api/action/move_to",
            {"x_mm": position[0], "y_mm": position[1], "z_mm": position[2]},
            token=token,
        )


def _builtin_run(
    proto: dict,
    out_dir: Path,
    *,
    start_at: tuple[float, float, float] | None = None,
) -> tuple[str, dict]:
    """run the protocol through /api/acquisition/start with images redirected to out_dir"""
    out_dir.mkdir(parents=True, exist_ok=True)
    proto = dict(proto)
    proto["machine_config"] = [{"name": "", "handle": "storage.base_image_output_dir", "value_kind": "text", "value": str(out_dir)}]
    if start_at is not None:
        _move_to(start_at)
    with _leased("builtin run") as token:
        resp = _ok("/api/acquisition/start", {"config_file": proto}, token=token)
        acq_id = resp["acquisition_id"]
        deadline = time.time() + TOTAL_TIMEOUT_S
        while time.time() < deadline:
            st = _ok("/api/acquisition/status", {"acquisition_id": acq_id})
            if st["acquisition_status"] not in ("running", "scheduled"):
                return acq_id, st
            time.sleep(0.05)
    raise AssertionError("built-in acquisition did not finish")


def _wait_for_acquisition(acq_id: str) -> dict:
    deadline = time.time() + TOTAL_TIMEOUT_S
    while time.time() < deadline:
        st = _ok("/api/acquisition/status", {"acquisition_id": acq_id})
        if st["acquisition_status"] not in ("running", "scheduled"):
            return st
        time.sleep(0.05)
    raise AssertionError("built-in acquisition did not finish")


def _norm_ome(xml: str) -> str:
    """drop what legitimately differs between two runs of the same protocol"""
    xml = re.sub(r"\d{4}-\d{2}-\d{2}T[0-9:.+\-]*", "TS", xml)
    xml = re.sub(r'(?<=ID=")[^"]*', "ID", xml)
    xml = re.sub(r'(?<=Name=")[^"]*\.tiff', "FILE", xml)
    xml = re.sub(r'/[^"<> ]*\.tiff', "PATH", xml)
    return xml


def _assert_same_output(client_root: Path, server_root: Path, expected_images: int) -> None:
    client, server = _tiffs(client_root), _tiffs(server_root)
    assert len(client) == expected_images, sorted(client)
    assert set(client) == set(server), (sorted(client), sorted(server))
    for rel, cf in client.items():
        with tifffile.TiffFile(cf) as a, tifffile.TiffFile(server[rel]) as b:
            assert a.asarray().shape == b.asarray().shape, rel
            assert a.asarray().dtype == b.asarray().dtype, rel
            assert a.pages[0].tags["Compression"].value == b.pages[0].tags["Compression"].value, rel
            assert _norm_ome(a.pages[0].description) == _norm_ome(b.pages[0].description), rel
    assert list(client_root.rglob("config.json")) and list(server_root.rglob("config.json"))


def _stage() -> tuple[float, float, float]:
    p = _ok("/api/get_info/current_state")["adapter_state"]["stage_position"]
    return (p["x_pos_mm"], p["y_pos_mm"], p["z_pos_mm"])


def _set_calibrated(value: bool) -> None:
    with _leased("calibration flag") as token:
        _ok("/api/action/machine_config_flush", {
            "privileged": True,
            "machine_config": [{"name": "", "handle": "laser.autofocus.calibration.is_calibrated", "value_kind": "option", "value": "yes" if value else "no"}],
        }, token=token)


# --------------------------------------------------------------------------------------------
# same output


@pytest.mark.usefixtures("scratch_server")
class TestSameOutput:
    START_POSITION = (60.0, 40.0, 2.0)

    @pytest.fixture(autouse=True)
    def start_external_run_at_known_position(self):
        _move_to(self.START_POSITION)

    def test_single_site(self, tmp_path):
        proto_path, proto = _protocol(tmp_path, channels=2)
        res = _run_script("--protocol-file", str(proto_path), "--output-dir", str(tmp_path / "client"))
        assert res.returncode == 0, res.stderr
        assert _ok("/api/lock/status")["held"] is False
        _, st = _builtin_run(proto, tmp_path / "server", start_at=self.START_POSITION)
        assert st["acquisition_status"] == "completed", st
        _assert_same_output(tmp_path / "client", tmp_path / "server", expected_images=2)

    def test_many_wells_sites_and_z_stacks(self, tmp_path):
        # three wells that are not in reading order (greedy traversal), 2x2 sites, 3-plane stacks
        # on two channels with different z offsets
        proto_path, proto = _protocol(tmp_path, wells=[(8, 12), (9, 14), (8, 14)], sites=2, channels=2, z_planes=3, name="grid")
        res = _run_script("--protocol-file", str(proto_path), "--output-dir", str(tmp_path / "client"))
        assert res.returncode == 0, res.stderr
        _, st = _builtin_run(proto, tmp_path / "server", start_at=self.START_POSITION)
        assert st["acquisition_status"] == "completed", st
        _assert_same_output(tmp_path / "client", tmp_path / "server", expected_images=3 * 4 * 2 * 3)

    def test_timepoints(self, tmp_path):
        proto_path, proto = _protocol(tmp_path, channels=1, num_t=2, delta_t_s=1.0, name="tseries")
        res = _run_script("--protocol-file", str(proto_path), "--output-dir", str(tmp_path / "client"))
        assert res.returncode == 0, res.stderr
        _, st = _builtin_run(proto, tmp_path / "server", start_at=self.START_POSITION)
        assert st["acquisition_status"] == "completed", st
        _assert_same_output(tmp_path / "client", tmp_path / "server", expected_images=2)
        assert {p.name for p in (tmp_path / "client").glob("*/*/t*")} == {"t0", "t1"}

    def test_with_laser_autofocus(self, tmp_path):
        # calibrate through the API first (the server writes the privileged calibration itself)
        with _leased("calibrate for autofocus test") as token:
            _ok("/api/action/move_to", {"x_mm": 60.0, "y_mm": 40.0, "z_mm": 2.0}, token=token)
            _ok("/api/action/laser_autofocus_calibrate", token=token)
        defaults = {i["handle"]: i["value"] for i in _ok("/api/get_features/machine_defaults")}
        assert defaults["laser.autofocus.calibration.is_calibrated"] == "yes"
        proto_path, proto = _protocol(tmp_path, wells=[(8, 12), (8, 13)], channels=2, autofocus=True, name="af")
        res = _run_script("--protocol-file", str(proto_path), "--output-dir", str(tmp_path / "client"))
        assert res.returncode == 0, res.stderr
        _, st = _builtin_run(proto, tmp_path / "server", start_at=self.START_POSITION)
        assert st["acquisition_status"] == "completed", st
        _assert_same_output(tmp_path / "client", tmp_path / "server", expected_images=4)

    @pytest.mark.parametrize("pad_low", ["no", "yes"])
    def test_twelve_bit_metadata_respects_pad_low(self, tmp_path, pad_low):
        """
        The API runner reads yes/no options from machine_defaults just like the server.

        Eight-bit images cannot expose a bad pad_low interpretation because their camera and
        container widths are both eight bits.  Twelve-bit samples ride in uint16: unpadded
        data must declare SignificantBits=12, while padded data fills and declares all 16.
        """
        defaults = {i["handle"]: i for i in _ok("/api/get_features/machine_defaults")}
        originals = [defaults["camera.main.pixel_format"], defaults["image.file.pad_low"]]
        expected_bits = 16 if pad_low == "yes" else 12
        try:
            with _leased("12-bit parity setup") as token:
                _ok(
                    "/api/action/machine_config_flush",
                    {
                        "privileged": True,
                        "machine_config": [
                            dict(defaults["camera.main.pixel_format"], value="mono12"),
                            dict(defaults["image.file.pad_low"], value=pad_low),
                        ],
                    },
                    token=token,
                )

            proto_path, proto = _protocol(tmp_path, channels=1, name=f"mono12-{pad_low}")
            client_root = tmp_path / "client"
            server_root = tmp_path / "server"
            res = _run_script(
                "--protocol-file", str(proto_path), "--output-dir", str(client_root)
            )
            assert res.returncode == 0, res.stderr
            _, st = _builtin_run(proto, server_root, start_at=self.START_POSITION)
            assert st["acquisition_status"] == "completed", st
            _assert_same_output(client_root, server_root, expected_images=1)

            for root in (client_root, server_root):
                image = next(root.rglob("*.tiff"))
                with tifffile.TiffFile(image) as tif:
                    page = tp.cast(tp.Any, tif.pages[0])
                    assert f'SignificantBits="{expected_bits}"' in page.description
        finally:
            with _leased("12-bit parity cleanup") as token:
                _ok(
                    "/api/action/machine_config_flush",
                    {"privileged": True, "machine_config": originals},
                    token=token,
                )

    def test_nondefault_filename_settings_match(self, tmp_path):
        """Exercise every configurable component of the per-plane filename."""
        handles = (
            "image.filename.use_channel_name",
            "image.filename.xy_index_start",
            "image.filename.z_index_start",
            "image.filename.site_index_start",
            "image.filename.zero_pad_column",
        )
        defaults = {i["handle"]: i for i in _ok("/api/get_features/machine_defaults")}
        originals = [defaults[handle] for handle in handles]
        changed = [
            dict(defaults["image.filename.use_channel_name"], value="no"),
            dict(defaults["image.filename.xy_index_start"], value=7),
            dict(defaults["image.filename.z_index_start"], value=3),
            dict(defaults["image.filename.site_index_start"], value=9),
            dict(defaults["image.filename.zero_pad_column"], value="no"),
        ]
        try:
            with _leased("filename parity setup") as token:
                _ok(
                    "/api/action/machine_config_flush",
                    {"machine_config": changed},
                    token=token,
                )

            proto_path, proto = _protocol(
                tmp_path, wells=[(8, 12)], channels=1, z_planes=2, name="filenames"
            )
            client_root = tmp_path / "client"
            server_root = tmp_path / "server"
            res = _run_script(
                "--protocol-file", str(proto_path), "--output-dir", str(client_root)
            )
            assert res.returncode == 0, res.stderr
            _, st = _builtin_run(proto, server_root, start_at=self.START_POSITION)
            assert st["acquisition_status"] == "completed", st
            _assert_same_output(client_root, server_root, expected_images=2)

            names = {path.name for path in client_root.rglob("*.tiff")}
            assert names == {
                "I13_s9_x7_y7_z3_fluo405.ome.tiff",
                "I13_s9_x7_y7_z4_fluo405.ome.tiff",
            }
        finally:
            with _leased("filename parity cleanup") as token:
                _ok(
                    "/api/action/machine_config_flush",
                    {"machine_config": originals},
                    token=token,
                )


@pytest.fixture(scope="class")
def calibrated_coordinate_runs(scratch_server, tmp_path_factory):
    """One external and one built-in run with offsets large enough to expose frame mixing."""
    tmp_path = tmp_path_factory.mktemp("calibrated-coordinate-parity")
    defaults = {i["handle"]: i for i in _ok("/api/get_features/machine_defaults")}
    handles = (
        "calibration.offset.x_mm",
        "calibration.offset.y_mm",
        "calibration.offset.z_mm",
    )
    originals = [defaults[handle] for handle in handles]
    offsets = (2.5, -1.25, 0.2)

    def move_to_start() -> None:
        with _leased("coordinate test positioning") as token:
            _ok(
                "/api/action/move_to",
                {"x_mm": 60.0, "y_mm": 40.0, "z_mm": 5.0},
                token=token,
            )

    def ome_positions(root: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        image = next(root.rglob("*.tiff"))
        with tifffile.TiffFile(image) as tif:
            xml = tp.cast(tp.Any, tif.pages[0]).description
        ome = from_xml(xml)
        image_meta = ome.images[0]
        plane = image_meta.pixels.planes[0]
        label = image_meta.stage_label
        assert label is not None
        return (
            (plane.position_x, plane.position_y, plane.position_z),
            (label.x, label.y, label.z),
        )

    try:
        with _leased("coordinate test calibration") as token:
            _ok(
                "/api/action/machine_config_flush",
                {
                    "privileged": True,
                    "machine_config": [
                        dict(defaults[handle], value=value)
                        for handle, value in zip(handles, offsets, strict=True)
                    ],
                },
                token=token,
            )

        proto_path, proto = _protocol(
            tmp_path, wells=[(8, 12)], channels=1, z_planes=1, name="coordinates"
        )
        parsed = sc.AcquisitionConfig.model_validate(proto)
        well = parsed.plate_wells[0]
        plate = parsed.wellplate_type
        expected_xy = (
            plate.get_well_offset_x(well.well_name) + plate.Well_size_x_mm / 2,
            plate.get_well_offset_y(well.well_name) + plate.Well_size_y_mm / 2,
        )

        move_to_start()
        external_root = tmp_path / "external"
        result = _run_script(
            "--protocol-file", str(proto_path), "--output-dir", str(external_root)
        )
        assert result.returncode == 0, result.stderr
        external_stage = _stage()

        move_to_start()
        builtin_root = tmp_path / "builtin"
        _, status = _builtin_run(proto, builtin_root)
        assert status["acquisition_status"] == "completed", status
        builtin_stage = _stage()

        yield {
            "expected": (*expected_xy, 5.0),
            "external_stage": external_stage,
            "builtin_stage": builtin_stage,
            "external_ome": ome_positions(external_root),
            "builtin_ome": ome_positions(builtin_root),
        }
    finally:
        with _leased("coordinate test cleanup") as token:
            _ok(
                "/api/action/machine_config_flush",
                {"privileged": True, "machine_config": originals},
                token=token,
            )


@pytest.mark.usefixtures("scratch_server")
class TestCalibratedCoordinateParity:
    """Physical motion and recorded positions must use the public calibrated frame."""

    def test_external_xy_motion_reaches_the_planned_site(self, calibrated_coordinate_runs):
        expected = calibrated_coordinate_runs["expected"]
        assert calibrated_coordinate_runs["external_stage"][:2] == pytest.approx(expected[:2])

    def test_builtin_xy_motion_reaches_the_planned_site(self, calibrated_coordinate_runs):
        expected = calibrated_coordinate_runs["expected"]
        assert calibrated_coordinate_runs["builtin_stage"][:2] == pytest.approx(expected[:2])

    def test_external_z_reference_round_trips_without_motion(self, calibrated_coordinate_runs):
        assert calibrated_coordinate_runs["external_stage"][2] == pytest.approx(5.0)

    def test_builtin_z_reference_round_trips_without_motion(self, calibrated_coordinate_runs):
        assert calibrated_coordinate_runs["builtin_stage"][2] == pytest.approx(5.0)

    def test_external_ome_plane_uses_calibrated_coordinates(self, calibrated_coordinate_runs):
        plane, _label = calibrated_coordinate_runs["external_ome"]
        assert plane == pytest.approx(calibrated_coordinate_runs["expected"])

    def test_builtin_ome_xy_uses_calibrated_coordinates(self, calibrated_coordinate_runs):
        plane, _label = calibrated_coordinate_runs["builtin_ome"]
        assert plane[:2] == pytest.approx(calibrated_coordinate_runs["expected"][:2])

    def test_builtin_ome_z_uses_calibrated_coordinates(self, calibrated_coordinate_runs):
        plane, _label = calibrated_coordinate_runs["builtin_ome"]
        assert plane[2] == pytest.approx(calibrated_coordinate_runs["expected"][2])

    @pytest.mark.parametrize("runner", ["external", "builtin"])
    def test_stage_label_and_plane_use_the_same_frame(self, calibrated_coordinate_runs, runner):
        plane, label = calibrated_coordinate_runs[f"{runner}_ome"]
        assert label == pytest.approx(plane)


# --------------------------------------------------------------------------------------------
# same refusals, before anything moves


@pytest.mark.usefixtures("scratch_server")
class TestSameRefusals:
    def test_autofocus_enabled_but_not_calibrated(self, tmp_path):
        _set_calibrated(False)
        try:
            proto_path, proto = _protocol(tmp_path, autofocus=True, name="uncal")
            before = _stage()
            with _leased() as token:
                status, body = _post("/api/acquisition/start", {"config_file": proto}, token=token)
            assert status == 500 and body is not None
            assert "not calibrated" in body["detail"]
            res = _run_script("--protocol-file", str(proto_path), "--output-dir", str(tmp_path / "client"))
            assert res.returncode == 1, res.stderr
            assert "not calibrated" in res.stderr
            assert _stage() == before, "neither may move the stage before refusing"
            assert _ok("/api/lock/status")["held"] is False
        finally:
            _set_calibrated(True)

    def test_forbidden_position(self, tmp_path):
        # A1 sits in a forbidden corner of the mock's plate holder
        proto_path, proto = _protocol(tmp_path, wells=[(0, 0)], name="forbidden")
        before = _stage()
        # built-in: refused at start, before anything is scheduled
        with _leased() as token:
            status, body = _post("/api/acquisition/start", {"config_file": proto}, token=token)
        assert status == 500 and body is not None, (status, body)
        err = str(body["detail"])
        assert "forbidden" in err.lower() and "A1" in err, err
        # script: refuses with the same information
        res = _run_script("--protocol-file", str(proto_path), "--output-dir", str(tmp_path / "client"))
        assert res.returncode == 1, res.stderr
        assert "forbidden" in res.stderr.lower() and "A1" in res.stderr
        assert not list((tmp_path / "client").rglob("*.tiff"))
        assert _stage() == before
        assert _ok("/api/lock/status")["held"] is False

    def test_lock_held_by_someone_else(self, tmp_path):
        proto_path, proto = _protocol(tmp_path, name="locked")
        lease = _ok("/api/lock/acquire", {"owner": "someone at the scope"})
        try:
            status, body = _post("/api/acquisition/start", {"config_file": proto})
            assert status == 423 and body is not None and body["detail"]["owner"] == "someone at the scope"
            res = _run_script("--protocol-file", str(proto_path), "--output-dir", str(tmp_path / "client"))
            assert res.returncode == 2, res.stderr
            assert "locked by 'someone at the scope'" in res.stderr
            assert not (tmp_path / "client").exists()
        finally:
            _ok("/api/lock/release", {"lock_token": lease["token"]})

    def test_unknown_protocol_name(self):
        status, _ = _post("/api/acquisition/config_fetch", {"config_file": "does-not-exist.json"})
        assert status >= 400
        res = _run_script("--protocol", "does-not-exist.json")
        assert res.returncode == 1, res.stderr
        assert "does-not-exist.json" in res.stderr or "server refused" in res.stderr
        assert _ok("/api/lock/status")["held"] is False, "the lock must not leak on a failed load"


# --------------------------------------------------------------------------------------------
# interrupts and error paths


@pytest.mark.usefixtures("scratch_server")
class TestInterruptsAndErrors:
    LONG: tp.ClassVar[dict[str, tp.Any]] = {"wells": [(8, 12), (8, 13), (9, 12), (9, 13), (10, 12), (10, 13)], "sites": 2, "channels": 2}  # 48 images

    def test_script_sigint_cancels_and_releases_the_lock(self, tmp_path):
        proto_path, _ = _protocol(tmp_path, name="long", **self.LONG)
        proc = _start_script("--protocol-file", str(proto_path), "--output-dir", str(tmp_path / "client"))
        try:
            _wait_files(tmp_path / "client", at_least=2)
            assert _ok("/api/lock/status")["held"] is True
            proc.send_signal(signal.SIGINT)
            _, err = proc.communicate(timeout=60)
        finally:
            if proc.poll() is None:
                proc.kill()
        assert proc.returncode == 130, err
        assert "interrupted" in err and "released device lock" in err
        assert _ok("/api/lock/status")["held"] is False
        assert 2 <= len(list((tmp_path / "client").rglob("*.tiff"))) < 48
        # the server is idle again, so the next client can work at once
        deadline = time.time() + 10
        while _ok("/api/get_info/current_state")["is_busy"] and time.time() < deadline:
            time.sleep(0.05)
        assert _ok("/api/get_info/current_state")["is_busy"] is False

    def test_builtin_cancel_stops_the_run(self, tmp_path):
        _, proto = _protocol(tmp_path, name="long-builtin", **self.LONG)
        out = tmp_path / "server"
        out.mkdir()
        proto["machine_config"] = [{"name": "", "handle": "storage.base_image_output_dir", "value_kind": "text", "value": str(out)}]
        with _leased("builtin cancel run") as token:
            acq_id = _ok("/api/acquisition/start", {"config_file": proto}, token=token)["acquisition_id"]
            _wait_files(out, at_least=2)
            _ok("/api/acquisition/cancel", {"acquisition_id": acq_id})
        deadline = time.time() + 30
        while time.time() < deadline:
            st = _ok("/api/acquisition/status", {"acquisition_id": acq_id})["acquisition_status"]
            if st not in ("running", "scheduled"):
                break
            time.sleep(0.05)
        assert st == "cancelled", st
        assert 2 <= len(list(out.rglob("*.tiff"))) < 48

    def test_script_stops_when_its_lock_is_taken_away(self, tmp_path):
        proto_path, _ = _protocol(tmp_path, name="long-intruder", **self.LONG)
        proc = _start_script("--protocol-file", str(proto_path), "--output-dir", str(tmp_path / "client"))
        intruder = None
        try:
            _wait_files(tmp_path / "client", at_least=2)
            _ok("/api/lock/break", {"reason": "test: taking over"})
            intruder = _ok("/api/lock/acquire", {"owner": "intruder"})
            _, err = proc.communicate(timeout=60)
            assert proc.returncode == 1, err
            assert "locked by 'intruder'" in err and "server refused" in err
            # the script must not have broken or released the intruder's lease on its way out
            st = _ok("/api/lock/status", {"lock_token": intruder["token"]})
            assert st["held"] is True and st["owner"] == "intruder" and st["token_matches"] is True
            assert "could not release lock" in err
        finally:
            if proc.poll() is None:
                proc.kill()
            if intruder is not None:
                _ok("/api/lock/release", {"lock_token": intruder["token"]})

    def test_script_runs_under_a_borrowed_lease_and_leaves_it(self, tmp_path):
        proto_path, _ = _protocol(tmp_path, name="borrowed")
        lease = _ok("/api/lock/acquire", {"owner": "pipeline"})
        try:
            res = _run_script("--protocol-file", str(proto_path), "--output-dir", str(tmp_path / "client"), "--lock-token", lease["token"])
            assert res.returncode == 0, res.stderr
            assert "released device lock" not in res.stderr
            st = _ok("/api/lock/status", {"lock_token": lease["token"]})
            assert st["held"] is True and st["token_matches"] is True
        finally:
            _ok("/api/lock/release", {"lock_token": lease["token"]})


@pytest.mark.usefixtures("scratch_server")
class TestS3Backend:
    """both the script and the built-in acquisition can store to S3 (MinIO here); same keys"""

    def _keys(self, minio_s3, prefix: str) -> dict[str, int]:
        listed = minio_s3["client"].list_objects_v2(Bucket=minio_s3["bucket"], Prefix=prefix + "/")
        return {o["Key"].removeprefix(prefix + "/").split("/", 2)[-1]: o["Size"] for o in listed.get("Contents", [])}

    def test_script_and_builtin_store_to_s3(self, minio_s3, tmp_path):
        proto_path, proto = _protocol(tmp_path, channels=2, name="s3")
        run_id = f"api-{int(time.time())}"

        # script: direct upload, spill dir stays empty when the link keeps up
        res = _run_script(
            "--protocol-file", str(proto_path), "--output-dir", str(tmp_path / "spill"), "--storage", "s3",
            "--s3-bucket", minio_s3["bucket"], "--s3-prefix", f"{run_id}/script", "--s3-endpoint-url", minio_s3["endpoint_url"],
        )
        assert res.returncode == 0, res.stderr
        assert "s3://" in res.stderr and "spilled" not in res.stderr
        assert not list((tmp_path / "spill").rglob("*.tiff"))
        script_keys = self._keys(minio_s3, f"{run_id}/script")
        assert len([k for k in script_keys if k.endswith(".tiff")]) == 2 and any(k.endswith("config.json") for k in script_keys)

        # built-in: switch the server's backend through privileged machine config, then back
        defaults = {i["handle"]: i for i in _ok("/api/get_features/machine_defaults")}
        originals = [defaults[h] for h in ("storage.backend", "storage.s3.bucket", "storage.s3.prefix", "storage.s3.endpoint_url")]
        (tmp_path / "server-spill").mkdir()
        proto["machine_config"] = [dict(defaults["storage.base_image_output_dir"], value=str(tmp_path / "server-spill"))]
        try:
            with _leased("builtin s3 run") as token:
                _ok("/api/action/machine_config_flush", {"privileged": True, "machine_config": [
                    dict(defaults["storage.backend"], value="s3"),
                    dict(defaults["storage.s3.bucket"], value=minio_s3["bucket"]),
                    dict(defaults["storage.s3.prefix"], value=f"{run_id}/builtin"),
                    dict(defaults["storage.s3.endpoint_url"], value=minio_s3["endpoint_url"]),
                ]}, token=token)
                resp = _ok("/api/acquisition/start", {"config_file": proto}, token=token)
                final = _wait_for_acquisition(resp["acquisition_id"])
            assert final["acquisition_status"] == "completed", final
            assert final["acquisition_progress"]["last_image"]["storage_path"].startswith(f"s3://{minio_s3['bucket']}/{run_id}/builtin/")
        finally:
            with _leased("s3 cleanup") as token:
                _ok("/api/action/machine_config_flush", {"privileged": True, "machine_config": originals}, token=token)
                _ok("/api/action/machine_config_flush", {"machine_config": [defaults["storage.base_image_output_dir"]]}, token=token)
        builtin_keys = self._keys(minio_s3, f"{run_id}/builtin")
        assert set(builtin_keys) == set(script_keys), (sorted(builtin_keys), sorted(script_keys))
        assert not list((tmp_path / "server-spill").rglob("*.tiff")), "server did not need to spill"
