"""
scripts/acquire_via_api.py runs the built-in imaging protocol from the outside, one API request
at a time. These tests are the proof of the claim behind it: against the mock server the script
produces the same files as /api/acquisition/start (names, layout, pixel shape, compression, OME
metadata up to the documented position-frame difference), it respects the device lock, and it
leaves the lock the way it found it.
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
import tifffile

from tests.test_api_operations import make_request, running_server  # noqa: F401  (fixture re-export)

SERVER = "http://127.0.0.1:5000"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "acquire_via_api.py"


def _post(path: str, data: dict | None = None) -> tuple[int, dict | None]:
    return make_request(f"{SERVER}{path}", data=data or {})


def _small_protocol(tmp_path: Path) -> tuple[Path, dict]:
    """the server's default protocol cut down to one centre well, one site, two channels, one plane"""
    status, resp = _post("/api/acquisition/config_fetch", {"config_file": "default.json"})
    assert status == 200 and resp is not None
    proto = resp["file"]
    plate = proto["wellplate_type"]
    proto["plate_wells"] = [{"row": plate["Num_wells_y"] // 2, "col": plate["Num_wells_x"] // 2, "selected": True}]
    for site in proto["grid"]["mask"]:
        site["selected"] = site["row"] == 0 and site["col"] == 0
    for i, channel in enumerate(proto["channels"]):
        channel["enabled"] = i < 2
        channel["num_z_planes"] = 1
    proto["project_name"] = "api-parity"
    proto["plate_name"] = "plate"
    path = tmp_path / "small.json"
    path.write_text(json.dumps(proto))
    return path, proto


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--server", SERVER, *args],
        capture_output=True, text=True, timeout=300,
    )


def _norm_ome(xml: str) -> str:
    """drop what legitimately differs between two runs of the same protocol"""
    xml = re.sub(r"\d{4}-\d{2}-\d{2}T[0-9:.+\-]*", "TS", xml)  # timestamps
    xml = re.sub(r'(?<=ID=")[^"]*', "ID", xml)  # generated ids
    xml = re.sub(r'(?<=Name=")[^"]*\.tiff', "FILE", xml)  # file names carry the run directory
    xml = re.sub(r'/[^"<> ]*\.tiff', "PATH", xml)
    # documented difference: the server records raw microcontroller coordinates, the API exposes
    # calibrated ones, so positions differ by the calibration offsets
    xml = re.sub(r'\b(Position[XYZ]|[XYZ])="[^"]*"', r'\1="POS"', xml)
    return xml


def _wait_for_acquisition(acquisition_id: str) -> dict:
    for _ in range(600):
        status, st = _post("/api/acquisition/status", {"acquisition_id": acquisition_id})
        assert status == 200 and st is not None
        if st["acquisition_status"] not in ("running", "scheduled"):
            return st
        time.sleep(0.1)
    raise AssertionError("built-in acquisition did not finish")


@pytest.mark.usefixtures("running_server")
class TestAcquireViaApi:
    def test_script_produces_what_the_builtin_acquisition_produces(self, tmp_path):
        proto_path, proto = _small_protocol(tmp_path)

        # client-driven run: takes the lock, images, releases the lock
        res = _run_script("--protocol-file", str(proto_path), "--output-dir", str(tmp_path / "client"), "--owner", "parity test")
        assert res.returncode == 0, res.stderr
        client_files = {f.name: f for f in (tmp_path / "client").rglob("*.tiff")}
        assert len(client_files) == 2, sorted(client_files)
        assert list((tmp_path / "client").rglob("config.json")), "protocol copy next to the images"
        _status, lock = _post("/api/lock/status")
        assert lock is not None and lock["held"] is False

        # the same protocol through the server's own acquisition, output redirected via the
        # (non-privileged) storage handle so nothing lands in the real image directory
        _status, defaults = _post("/api/get_features/machine_defaults")
        assert defaults is not None
        base_dir_item = next(i for i in defaults if i["handle"] == "storage.base_image_output_dir")
        (tmp_path / "server").mkdir()  # the server requires the base directory to exist
        proto["machine_config"] = [dict(base_dir_item, value=str(tmp_path / "server"))]
        try:
            status, resp = _post("/api/acquisition/start", {"config_file": proto})
            assert status == 200, (status, resp)
            assert resp is not None
            final = _wait_for_acquisition(resp["acquisition_id"])
            assert final["acquisition_status"] == "completed", final
        finally:
            _post("/api/action/machine_config_flush", {"machine_config": [base_dir_item]})

        server_files = {f.name: f for f in (tmp_path / "server").rglob("*.tiff")}
        assert set(server_files) == set(client_files)
        for name, client_file in client_files.items():
            with tifffile.TiffFile(client_file) as a, tifffile.TiffFile(server_files[name]) as b:
                assert a.asarray().shape == b.asarray().shape
                assert a.asarray().dtype == b.asarray().dtype
                assert a.pages[0].tags["Compression"].value == b.pages[0].tags["Compression"].value
                assert _norm_ome(a.pages[0].description) == _norm_ome(b.pages[0].description), name

    def test_script_refuses_when_someone_else_holds_the_lock(self, tmp_path):
        proto_path, _ = _small_protocol(tmp_path)
        status, lease = _post("/api/lock/acquire", {"owner": "someone at the scope"})
        assert status == 200 and lease is not None
        try:
            res = _run_script("--protocol-file", str(proto_path), "--output-dir", str(tmp_path / "out"))
            assert res.returncode == 2, res.stderr
            assert "locked by 'someone at the scope'" in res.stderr
            assert not (tmp_path / "out").exists()
        finally:
            _post("/api/lock/release", {"lock_token": lease["token"]})

    def test_script_can_run_under_a_lease_it_did_not_take(self, tmp_path):
        # a pipeline that already holds the lock hands its token over; the script must not release it
        proto_path, _ = _small_protocol(tmp_path)
        status, lease = _post("/api/lock/acquire", {"owner": "pipeline"})
        assert status == 200 and lease is not None
        try:
            res = _run_script("--protocol-file", str(proto_path), "--output-dir", str(tmp_path / "out"), "--lock-token", lease["token"])
            assert res.returncode == 0, res.stderr
            _status, lock = _post("/api/lock/status", {"lock_token": lease["token"]})
            assert lock is not None and lock["held"] is True and lock["token_matches"] is True
        finally:
            _post("/api/lock/release", {"lock_token": lease["token"]})
