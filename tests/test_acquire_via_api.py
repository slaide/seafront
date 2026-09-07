"""
scripts/acquire_via_api.py runs the built-in imaging protocol from the outside, one API request
at a time. These tests are the proof of the claim behind it: against the mock server the script
produces the same files as /api/acquisition/start, refuses the same things for the same reasons
before anything moves, respects the device lock, and has working interrupt and error paths that
leave the lock the way they found it.

The server under test runs against a scratch copy of the seafront home (SEAFRONT_HOME), so
calibration writes and acquisitions never touch the real config or image directory.
"""

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import typing as tp
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import tifffile

from tests.test_api_operations import make_request

MOCK = "mocroscope"
PORT = 5000
SERVER = f"http://127.0.0.1:{PORT}"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "acquire_via_api.py"
TOTAL_TIMEOUT_S = 300


# --------------------------------------------------------------------------------------------
# scratch server


def _port_free() -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", PORT)) != 0


@pytest.fixture(scope="module")
def scratch_server(tmp_path_factory):
    """mock server on a copy of the real seafront home, images redirected into the scratch dir"""
    real_home = Path.home() / "seafront"
    if not (real_home / "config.json").exists():
        pytest.skip("no ~/seafront/config.json to copy")
    home = tmp_path_factory.mktemp("seafront-home")
    cfg = json.loads(re.sub(r"//.*", "", (real_home / "config.json").read_text()))  # tolerate json5 comments
    images = home / "images"
    images.mkdir()
    mock = next((m for m in cfg["microscopes"] if m.get("system.microscope_name") == MOCK), None)
    if mock is None:
        pytest.skip(f"no '{MOCK}' in ~/seafront/config.json")
    mock["storage.base_image_output_dir"] = str(images)
    (home / "config.json").write_text(json.dumps(cfg, indent=2))
    if (real_home / "acquisition_configs").exists():
        shutil.copytree(real_home / "acquisition_configs", home / "acquisition_configs")

    assert _port_free(), f"port {PORT} is busy; another seafront is running"
    env = dict(os.environ, SEAFRONT_HOME=str(home), MOCK_NO_DELAYS="1")
    # server output goes to a file: an unread PIPE fills up after a few dozen images and then
    # blocks the server's event loop on every log write
    server_log = (home / "server.log").open("w")
    proc = subprocess.Popen(
        ["uv", "run", "python", "-m", "seafront", "--microscope", MOCK],
        stdout=server_log, stderr=subprocess.STDOUT, text=True, env=env,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{SERVER}/", timeout=1) as r:
                if r.status == 200:
                    break
        except (urllib.error.URLError, ConnectionRefusedError):
            time.sleep(0.3)
    else:
        proc.terminate()
        raise RuntimeError("scratch server did not start")
    yield home
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    server_log.close()


# --------------------------------------------------------------------------------------------
# helpers


def _post(path: str, data: dict | None = None) -> tuple[int, dict | None]:
    return make_request(f"{SERVER}{path}", data=data or {})


def _ok(path: str, data: dict | None = None) -> dict:
    status, body = _post(path, data)
    assert status == 200, (path, status, body)
    assert body is not None
    return body


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


def _builtin_run(proto: dict, out_dir: Path) -> tuple[str, dict]:
    """run the protocol through /api/acquisition/start with images redirected to out_dir"""
    out_dir.mkdir(parents=True, exist_ok=True)
    proto = dict(proto)
    proto["machine_config"] = [{"name": "", "handle": "storage.base_image_output_dir", "value_kind": "text", "value": str(out_dir)}]
    resp = _ok("/api/acquisition/start", {"config_file": proto})
    acq_id = resp["acquisition_id"]
    deadline = time.time() + TOTAL_TIMEOUT_S
    while time.time() < deadline:
        st = _ok("/api/acquisition/status", {"acquisition_id": acq_id})
        if st["acquisition_status"] not in ("running", "scheduled"):
            return acq_id, st
        time.sleep(0.05)
    raise AssertionError("built-in acquisition did not finish")


def _norm_ome(xml: str) -> str:
    """drop what legitimately differs between two runs of the same protocol"""
    xml = re.sub(r"\d{4}-\d{2}-\d{2}T[0-9:.+\-]*", "TS", xml)
    xml = re.sub(r'(?<=ID=")[^"]*', "ID", xml)
    xml = re.sub(r'(?<=Name=")[^"]*\.tiff', "FILE", xml)
    xml = re.sub(r'/[^"<> ]*\.tiff', "PATH", xml)
    # documented difference: the server records raw microcontroller coordinates, the API
    # exposes calibrated ones, so positions differ by the calibration offsets
    xml = re.sub(r'\b(Position[XYZ]|[XYZ])="[^"]*"', r'\1="POS"', xml)
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
    _ok("/api/action/machine_config_flush", {
        "privileged": True,
        "machine_config": [{"name": "", "handle": "laser.autofocus.calibration.is_calibrated", "value_kind": "option", "value": "yes" if value else "no"}],
    })


# --------------------------------------------------------------------------------------------
# same output


@pytest.mark.usefixtures("scratch_server")
class TestSameOutput:
    def test_single_site(self, tmp_path):
        proto_path, proto = _protocol(tmp_path, channels=2)
        res = _run_script("--protocol-file", str(proto_path), "--output-dir", str(tmp_path / "client"))
        assert res.returncode == 0, res.stderr
        assert _ok("/api/lock/status")["held"] is False
        _, st = _builtin_run(proto, tmp_path / "server")
        assert st["acquisition_status"] == "completed", st
        _assert_same_output(tmp_path / "client", tmp_path / "server", expected_images=2)

    def test_many_wells_sites_and_z_stacks(self, tmp_path):
        # three wells that are not in reading order (greedy traversal), 2x2 sites, 3-plane stacks
        # on two channels with different z offsets
        proto_path, proto = _protocol(tmp_path, wells=[(8, 12), (9, 14), (8, 14)], sites=2, channels=2, z_planes=3, name="grid")
        res = _run_script("--protocol-file", str(proto_path), "--output-dir", str(tmp_path / "client"))
        assert res.returncode == 0, res.stderr
        _, st = _builtin_run(proto, tmp_path / "server")
        assert st["acquisition_status"] == "completed", st
        _assert_same_output(tmp_path / "client", tmp_path / "server", expected_images=3 * 4 * 2 * 3)

    def test_timepoints(self, tmp_path):
        proto_path, proto = _protocol(tmp_path, channels=1, num_t=2, delta_t_s=1.0, name="tseries")
        res = _run_script("--protocol-file", str(proto_path), "--output-dir", str(tmp_path / "client"))
        assert res.returncode == 0, res.stderr
        _, st = _builtin_run(proto, tmp_path / "server")
        assert st["acquisition_status"] == "completed", st
        _assert_same_output(tmp_path / "client", tmp_path / "server", expected_images=2)
        assert {p.name for p in (tmp_path / "client").glob("*/*/t*")} == {"t0", "t1"}

    def test_with_laser_autofocus(self, tmp_path):
        # calibrate through the API first (the server writes the privileged calibration itself)
        _ok("/api/action/move_to", {"x_mm": 60.0, "y_mm": 40.0, "z_mm": 2.0})
        _ok("/api/action/laser_autofocus_calibrate")
        defaults = {i["handle"]: i["value"] for i in _ok("/api/get_features/machine_defaults")}
        assert defaults["laser.autofocus.calibration.is_calibrated"] == "yes"
        proto_path, proto = _protocol(tmp_path, wells=[(8, 12), (8, 13)], channels=2, autofocus=True, name="af")
        res = _run_script("--protocol-file", str(proto_path), "--output-dir", str(tmp_path / "client"))
        assert res.returncode == 0, res.stderr
        _, st = _builtin_run(proto, tmp_path / "server")
        assert st["acquisition_status"] == "completed", st
        _assert_same_output(tmp_path / "client", tmp_path / "server", expected_images=4)


# --------------------------------------------------------------------------------------------
# same refusals, before anything moves


@pytest.mark.usefixtures("scratch_server")
class TestSameRefusals:
    def test_autofocus_enabled_but_not_calibrated(self, tmp_path):
        _set_calibrated(False)
        try:
            proto_path, proto = _protocol(tmp_path, autofocus=True, name="uncal")
            before = _stage()
            status, body = _post("/api/acquisition/start", {"config_file": proto})
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
        status, body = _post("/api/acquisition/start", {"config_file": proto})
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
        acq_id = _ok("/api/acquisition/start", {"config_file": proto})["acquisition_id"]
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
