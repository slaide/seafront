"""
Concurrency safety-net tests for the reject-on-busy hardware model.

Phase 1 of the lock refactor (see REFACTOR_PLAN.md). These pin the CURRENT behavior
the refactor must preserve, so later phases can delete the lock machinery without
changing what the server does:

- The hardware has one owner. A command that cannot run the instant it is requested
  is rejected with 409 (busy), never queued to run later.
- Status reads are never rejected and never block behind a running operation.

All tests run against the mock backend only, with realistic delays enabled so a move
takes long enough to observe the busy window. No real hardware, no test hooks in
production code.
"""

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

MOCK_MICROSCOPE_NAME = "mocroscope"


def make_request(
    url: str, method: str = "POST", data: dict | None = None, timeout: float = 15
) -> tuple[int, dict | None]:
    """POST helper mirroring tests/test_api_operations.py, with a caller-set timeout.

    target_device is auto-injected. Returns (status_code, parsed_json_or_None).
    """
    req = urllib.request.Request(url)
    req.method = method

    if method == "POST":
        if data is None:
            data = {}
        if "target_device" not in data:
            data["target_device"] = MOCK_MICROSCOPE_NAME

    if data is not None:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(data).encode("utf-8")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode()
            return response.status, json.loads(body) if body else None
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        return e.code, json.loads(body) if body else None


@pytest.fixture(scope="class")
def running_server_with_delays(request):
    """Start the mock server WITH realistic delays (unlike test_api_operations, which
    sets MOCK_NO_DELAYS=1). Delays make a move take real wall-clock time, which is what
    lets these tests observe a command in flight."""
    env = os.environ.copy()
    env["MOCK_NO_DELAYS"] = "0"  # delays ON, so a move takes observable wall-clock time

    proc = subprocess.Popen(
        ["uv", "run", "python", "-m", "seafront", "--microscope", MOCK_MICROSCOPE_NAME],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    server_ready = False
    start_time = time.time()
    while time.time() - start_time < 15:
        try:
            with urllib.request.urlopen("http://127.0.0.1:5000/", timeout=1) as response:
                if response.status == 200:
                    server_ready = True
                    break
        except (urllib.error.URLError, ConnectionRefusedError):
            time.sleep(0.5)

    if not server_ready:
        proc.terminate()
        proc.wait()
        pytest.fail("Server did not start within timeout")

    request.cls.server_url = "http://127.0.0.1:5000"

    yield proc

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# start/end coordinates for a move long enough (~70mm) to leave a clear busy window,
# both well inside the 384-well plate travel range.
_START = {"x_mm": 30.0, "y_mm": 40.0, "z_mm": 1.0}
_FAR = {"x_mm": 100.0, "y_mm": 40.0, "z_mm": 1.0}


@pytest.mark.usefixtures("running_server_with_delays")
class TestRejectOnBusy:
    """The core rule: busy hardware rejects, it does not queue; status stays live."""

    def _url(self, path: str) -> str:
        return f"{self.server_url}{path}"

    def _current_state(self) -> tuple[int, dict | None]:
        return make_request(self._url("/api/get_info/current_state"), timeout=5)

    def _wait_until_busy(self, timeout_s: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            status, data = self._current_state()
            if status == 200 and data and data.get("is_busy"):
                return True
            time.sleep(0.02)
        return False

    def _start_long_move_in_background(self) -> tuple[threading.Thread, dict]:
        """Move to a known start (this first command also triggers homing), then kick
        off a long move on a background thread. Returns the thread and a result dict
        that the thread fills in when the long move returns."""
        status, data = make_request(self._url("/api/action/move_to"), data=_START, timeout=30)
        assert status == 200, f"setup move failed: {status} {data}"

        result: dict = {}

        def long_move():
            result["status"], result["data"] = make_request(
                self._url("/api/action/move_to"), data=_FAR, timeout=30
            )

        thread = threading.Thread(target=long_move)
        thread.start()
        return thread, result

    def test_busy_command_is_rejected_not_queued(self):
        """While a move is in flight, a second command returns 409 immediately instead
        of queueing to run after the move. If it queued, we would see 200."""
        thread, long_result = self._start_long_move_in_background()
        try:
            assert self._wait_until_busy(), "hardware never reported busy during the long move"

            status, data = make_request(
                self._url("/api/action/move_by"),
                data={"axis": "x", "distance_mm": 1.0},
                timeout=5,
            )
            assert status == 409, f"expected 409 (rejected as busy), got {status}: {data}"
        finally:
            thread.join()

        assert long_result["status"] == 200, f"the in-flight move should still succeed: {long_result}"

    def test_status_is_never_rejected_while_busy(self):
        """A status read during a move returns 200 with is_busy=True and a reason set.
        is_busy=True proves the read was served *during* the move (not blocked until it
        finished), and 200 (not 409) proves status is exempt from the busy rejection."""
        thread, _ = self._start_long_move_in_background()
        try:
            assert self._wait_until_busy(), "hardware never reported busy during the long move"

            status, data = self._current_state()
            assert status == 200, f"status was rejected while busy: {status} {data}"
            assert data is not None
            assert data["is_busy"] is True, f"expected is_busy True, got {data.get('is_busy')}"
            assert len(data["busy_reasons"]) >= 1, (
                f"expected a busy reason naming the running command, got {data.get('busy_reasons')}"
            )
        finally:
            thread.join()


# --- config helpers for the streaming and acquisition tests -----------------------
# A valid AcquisitionChannelConfig / AcquisitionConfig is awkward to hand-write, so we
# generate the mock's default protocol (same server config the running mock uses) and
# derive payloads from it. Generated once, deep-copied per use so tests can mutate.

_default_protocol_cache: dict | None = None


def _default_protocol() -> dict:
    global _default_protocol_cache
    if _default_protocol_cache is None:
        from scripts.generate_default_protocol import generate_default_protocol

        path = generate_default_protocol(microscope_name=MOCK_MICROSCOPE_NAME)
        _default_protocol_cache = json.loads(Path(path).read_text())
    return json.loads(json.dumps(_default_protocol_cache))  # deep copy


def _one_channel() -> dict:
    """A single channel config, usable as the `channel` payload for stream/snap."""
    return _default_protocol()["channels"][0]


def _cancellable_config() -> dict:
    """A config that runs long enough (with delays on) to be cancelled mid-flight:
    one channel enabled, a small z-stack at the single default site."""
    cfg = _default_protocol()
    # start_acquisition rejects empty project/plate names
    cfg["project_name"] = "test_concurrency_project"
    cfg["plate_name"] = "test_concurrency_plate"
    # the default corner well (A1) is a forbidden area on the mock; use a central well
    cfg["plate_wells"] = [{"row": 7, "col": 11, "selected": True}]
    cfg["channels"][0]["enabled"] = True
    cfg["channels"][0]["num_z_planes"] = 15
    cfg["channels"][0]["exposure_time_ms"] = 100.0
    cfg["channels"][0]["delta_z_um"] = 0.5
    return cfg


@pytest.mark.usefixtures("running_server_with_delays")
class TestStreamingRejectsAndLifecycle:
    """Streaming is exclusive (snap during it is rejected) and its stop path is clean
    enough to begin again — the stop-path deadlock is the risk this guards."""

    def _url(self, path: str) -> str:
        return f"{self.server_url}{path}"

    def test_snap_during_streaming_is_rejected(self):
        channel = _one_channel()
        status, data = make_request(
            self._url("/api/action/stream_channel_begin"), data={"channel": channel}
        )
        assert status == 200, f"stream begin failed: {status} {data}"
        try:
            status, data = make_request(
                self._url("/api/action/snap_channel"), data={"channel": channel}
            )
            # rejected by the streaming gate (400) — a different path from the 409 busy lock
            assert status == 400, f"expected 400 (rejected during streaming), got {status}: {data}"
        finally:
            make_request(self._url("/api/action/stream_channel_end"), data={"channel": channel})

    def test_stream_begin_end_begin_again(self):
        channel = _one_channel()

        s1, d1 = make_request(self._url("/api/action/stream_channel_begin"), data={"channel": channel})
        assert s1 == 200, f"first begin failed: {s1} {d1}"

        s2, d2 = make_request(self._url("/api/action/stream_channel_end"), data={"channel": channel})
        assert s2 == 200, f"end failed: {s2} {d2}"

        # begin again after a clean end must work — a stop that left bad state would fail here
        s3, d3 = make_request(self._url("/api/action/stream_channel_begin"), data={"channel": channel})
        assert s3 == 200, f"second begin failed (stop left bad state?): {s3} {d3}"

        s4, _ = make_request(self._url("/api/action/stream_channel_end"), data={"channel": channel})
        assert s4 == 200, f"final end failed: {s4}"


@pytest.mark.usefixtures("running_server_with_delays")
class TestAcquisitionCancel:
    """Cancel applies to a full acquisition run: once cancelled it reaches CANCELLED,
    not COMPLETED."""

    def _url(self, path: str) -> str:
        return f"{self.server_url}{path}"

    def _acq_status(self, acq_id: str) -> str:
        status, data = make_request(
            self._url("/api/acquisition/status"), data={"acquisition_id": acq_id}, timeout=10
        )
        assert status == 200, f"status query failed: {status} {data}"
        assert data is not None
        return data["acquisition_status"]

    def test_acquisition_cancel_reaches_cancelled(self):
        config = _cancellable_config()
        status, data = make_request(
            self._url("/api/acquisition/start"), data={"config_file": config}, timeout=30
        )
        assert status == 200, f"start failed: {status} {data}"
        assert data is not None
        acq_id = data["acquisition_id"]

        # wait until it is actually active, then cancel
        active_deadline = time.monotonic() + 10
        became_active = False
        while time.monotonic() < active_deadline:
            if self._acq_status(acq_id) in {"running", "scheduled"}:
                became_active = True
                break
            time.sleep(0.05)
        assert became_active, "acquisition never became active"

        status, data = make_request(
            self._url("/api/acquisition/cancel"), data={"acquisition_id": acq_id}, timeout=10
        )
        assert status == 200, f"cancel failed: {status} {data}"

        terminal = {"cancelled", "completed", "crashed"}
        final = None
        terminal_deadline = time.monotonic() + 15
        while time.monotonic() < terminal_deadline:
            final = self._acq_status(acq_id)
            if final in terminal:
                break
            time.sleep(0.1)
        assert final == "cancelled", f"expected cancelled, got {final}"
