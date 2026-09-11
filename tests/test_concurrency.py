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

import http.client
import json
import os
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

MOCK_MICROSCOPE_NAME = "mocroscope"


def make_request(
    url: str,
    method: str = "POST",
    data: dict | None = None,
    timeout: float = 15,
    token: str | None = None,
) -> tuple[int, dict | None]:
    """POST helper mirroring tests/test_api_operations.py, with a caller-set timeout.

    target_device is auto-injected; `token` carries the device lease that hardware routes
    require (see the hardware_lease fixture). Returns (status_code, parsed_json_or_None).
    """
    req = urllib.request.Request(url)
    req.method = method

    if method == "POST":
        # a copy: callers pass shared payload constants, and a token injected into one of
        # those would travel to every later caller that reuses it
        data = dict(data or {})
        if "target_device" not in data:
            data["target_device"] = MOCK_MICROSCOPE_NAME
        if token is not None and "lock_token" not in data:
            data["lock_token"] = token

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


def _wait_for_free_port(timeout_s: float = 20) -> None:
    """Wait for port 5000 to be free.

    The previous class's server is terminated at teardown but takes a moment to let go of
    the port. Starting before then gives a server that cannot bind and dies, while the
    readiness probe below happily talks to the dying one instead.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", 5000)) != 0:
                return
        time.sleep(0.2)
    pytest.fail("port 5000 is still in use; another seafront is running")


@pytest.fixture(scope="class")
def running_server_with_delays(request, tmp_path_factory):
    """Start the mock server WITH realistic delays (unlike test_api_operations, which
    sets MOCK_NO_DELAYS=1). Delays make a move take real wall-clock time, which is what
    lets these tests observe a command in flight."""
    env = os.environ.copy()
    env["MOCK_NO_DELAYS"] = "0"  # delays ON, so a move takes observable wall-clock time

    _wait_for_free_port()
    # Output goes to a file, never a PIPE: these tests fire thousands of requests, and an
    # unread pipe fills up after a few hundred access log lines and then blocks the server's
    # event loop on every further log write - which looks exactly like the server hanging.
    log_path = tmp_path_factory.mktemp("server-log") / "server.log"
    log = log_path.open("w")
    proc = subprocess.Popen(
        ["uv", "run", "python", "-m", "seafront", "--microscope", MOCK_MICROSCOPE_NAME],
        stdout=log,
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
        log.close()
        pytest.fail(f"Server did not start within timeout; log at {log_path}")

    request.cls.server_url = "http://127.0.0.1:5000"

    yield proc

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log.close()


@pytest.fixture(scope="class")
def hardware_lease(request, running_server_with_delays):
    """Hold the device lease for the class: hardware routes refuse requests without it."""
    url = request.cls.server_url
    status, body = make_request(f"{url}/api/lock/acquire", data={"owner": "test_concurrency"})
    assert status == 200, f"could not take the device lease: {status} {body}"
    assert body is not None
    request.cls.lease = body["token"]
    yield body["token"]
    make_request(f"{url}/api/lock/break", data={"reason": "test class teardown"})
    request.cls.lease = None


# start/end coordinates for a move long enough (~70mm) to leave a clear busy window,
# both well inside the 384-well plate travel range.
_START = {"x_mm": 30.0, "y_mm": 40.0, "z_mm": 1.0}
_FAR = {"x_mm": 100.0, "y_mm": 40.0, "z_mm": 1.0}


@pytest.mark.usefixtures("running_server_with_delays", "hardware_lease")
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
        status, data = make_request(self._url("/api/action/move_to"), data=_START, timeout=30, token=self.lease)
        assert status == 200, f"setup move failed: {status} {data}"

        result: dict = {}

        def long_move():
            result["status"], result["data"] = make_request(
                self._url("/api/action/move_to"), data=_FAR, timeout=30,
                token=self.lease,
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
                token=self.lease,
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


# a move short enough to keep the status samples below to a few thousand, long enough that
# the hardware is visibly held for a while afterwards (~0.5s at the mock's 40mm/s)
_NEAR = {"x_mm": 50.0, "y_mm": 40.0, "z_mm": 1.0}

@pytest.mark.usefixtures("running_server_with_delays", "hardware_lease")
class TestOneRequestIsOneClaim:
    """
    A request that drives the hardware claims it once, for the whole request.

    A request that claims the worker, hands it back, and claims again has a hole in the
    middle. Another command can run inside it, a client watching status sees the microscope
    blink idle while a move is still going, and the request that got there first can come
    back 409 because a later one took the slot it had just let go of.

    The hole is only visible from outside, and only to a client that looks often enough: it
    is well under a millisecond wide, so these tests watch over one kept-alive connection
    (a fresh connection per sample is ten times too slow to see it).
    """

    ROUNDS = 3

    def _url(self, path: str) -> str:
        return f"{self.server_url}{path}"

    def _settle_at(self, position: dict) -> None:
        status, data = make_request(self._url("/api/action/move_to"), data=position, timeout=30, token=self.lease)
        assert status == 200, f"setup move failed: {status} {data}"

    def _running_command(self) -> list[str]:
        status, data = make_request(self._url("/api/get_info/current_state"), timeout=5)
        assert status == 200 and data is not None, f"status read failed: {status} {data}"
        return list(data["busy_reasons"]) if data["is_busy"] else []

    def _watch_one_move(self) -> list[tuple[bool, tuple[str, ...]]]:
        """
        Run one move and sample status flat out from before it is sent until after it returns.

        The samples therefore cover the request's whole life on the hardware, including the
        moment it takes the hardware - which is where a second claim shows up.
        """
        self._settle_at(_START)
        samples: list[tuple[bool, tuple[str, ...]]] = []
        stop = threading.Event()

        def poll():
            conn = http.client.HTTPConnection("127.0.0.1", 5000, timeout=10)
            body = json.dumps({"target_device": MOCK_MICROSCOPE_NAME})
            headers = {"Content-Type": "application/json"}
            try:
                while not stop.is_set():
                    conn.request("POST", "/api/get_info/current_state", body, headers)
                    state = json.loads(conn.getresponse().read().decode())
                    samples.append((state["is_busy"], tuple(state["busy_reasons"])))
            finally:
                conn.close()

        poller = threading.Thread(target=poll)
        poller.start()
        try:
            time.sleep(0.05)  # let the poller get going before the move is sent
            status, data = make_request(
                self._url("/api/action/move_to"), data=_NEAR, timeout=30, token=self.lease
            )
            assert status == 200, f"the move was rejected: {status} {data}"
        finally:
            stop.set()
            poller.join()

        busy = [i for i, (is_busy, _) in enumerate(samples) if is_busy]
        assert len(busy) >= 50, f"never saw the move run ({len(samples)} samples)"
        return samples

    @staticmethod
    def _claims(samples: list[tuple[bool, tuple[str, ...]]]) -> int:
        """how many times the hardware went from free to taken across these samples"""
        return sum(
            1
            for i, (is_busy, _) in enumerate(samples)
            if is_busy and (i == 0 or not samples[i - 1][0])
        )

    def test_a_request_claims_the_hardware_once(self):
        """
        One move request, one claim.

        A second claim inside the same request means the hardware was handed back and taken
        again halfway through, and whoever was watching or waiting in between found a
        microscope that was free when it was not.
        """
        for round_no in range(self.ROUNDS):
            samples = self._watch_one_move()
            claims = self._claims(samples)
            assert claims == 1, (
                f"round {round_no}: the request took the hardware {claims} times, not once "
                f"(seen: {sorted({r for _, rs in samples for r in rs})})"
            )

    def test_busy_never_blinks_idle_inside_a_request(self):
        """
        Status is live, so a client watching it sees the hole too: the microscope must not
        read as free in the middle of a move that is still going.
        """
        for round_no in range(self.ROUNDS):
            samples = self._watch_one_move()
            busy_at = [i for i, (is_busy, _) in enumerate(samples) if is_busy]
            window = samples[busy_at[0] : busy_at[-1] + 1]
            idle = [i for i, (is_busy, _) in enumerate(window) if not is_busy]
            assert not idle, (
                f"round {round_no}: is_busy read false {len(idle)} time(s) in the middle of a "
                f"running move ({len(window)} samples between the first and last busy read)"
            )

    def test_busy_reason_names_the_command_the_client_asked_for(self):
        """
        The whole time a move request holds the hardware, the reason says MoveTo.

        Another command's name in there is the second claim wearing a label: the request ran
        as several claims, and the gaps between them are the hole. (The generic reason is the
        same claim before the worker thread has picked it up, so it is not a second claim.)
        """
        seen: set[str] = set()
        for _ in range(self.ROUNDS):
            seen.update(r for _, reasons in self._watch_one_move() for r in reasons)
        named = {r for r in seen if r != "hardware command in progress"}
        assert named == {"hardware busy: MoveTo"}, f"one request, several claims: {sorted(named)}"

    def test_no_other_command_runs_while_one_is_running(self):
        """
        While a move is running, every other command is refused with 409.

        The probe window ends well before the move does, and the move is checked to be still
        running afterwards, so a 200 here cannot be the harmless one a probe gets once the
        move has finished and the worker is free again.
        """
        self._settle_at(_START)
        result: dict = {}

        def run():
            result["status"], result["data"] = make_request(
                self._url("/api/action/move_to"), data=_FAR, timeout=30, token=self.lease
            )

        thread = threading.Thread(target=run)
        thread.start()
        attempts, accepted = 0, []
        try:
            deadline = time.monotonic() + 10
            while self._running_command() != ["hardware busy: MoveTo"]:
                assert time.monotonic() < deadline, f"the move never ran: {result}"
            deadline = time.monotonic() + 1.0  # the move still has ~0.7s to go after this
            while time.monotonic() < deadline:
                status, data = make_request(
                    self._url("/api/action/turn_off_all_illumination"), timeout=10, token=self.lease
                )
                attempts += 1
                if status != 409:
                    accepted.append((status, data))
            still_moving = self._running_command() == ["hardware busy: MoveTo"]
        finally:
            thread.join()

        assert still_moving, "the move ended before the probe window did; nothing was proven"
        assert attempts >= 5, f"too few probes to prove anything ({attempts})"
        assert not accepted, f"{len(accepted)} of {attempts} commands ran inside the move: {accepted[:3]}"

    def test_a_request_that_has_the_hardware_is_not_rejected(self):
        """
        Once a request holds the hardware, hammering the server must not make it fail.

        The probes start only after the server reports the move's claim, so the move was
        there first; a 409 for it means it lost hardware it already held to one of them.
        """
        for round_no in range(self.ROUNDS):
            self._settle_at(_START)
            move: dict = {}

            def run(slot: dict = move):
                slot["status"], slot["data"] = make_request(
                    self._url("/api/action/move_to"), data=_FAR, timeout=30, token=self.lease
                )

            thread = threading.Thread(target=run)
            thread.start()
            try:
                deadline = time.monotonic() + 10
                while not self._running_command():
                    assert time.monotonic() < deadline, f"the move never ran: {move}"
                while thread.is_alive():
                    make_request(
                        self._url("/api/action/turn_off_all_illumination"), timeout=10, token=self.lease
                    )
            finally:
                thread.join()
            assert move["status"] == 200, f"round {round_no}: the accepted move was rejected: {move}"


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


@pytest.mark.usefixtures("running_server_with_delays", "hardware_lease")
class TestStreamingRejectsAndLifecycle:
    """Streaming is exclusive (snap during it is rejected) and its stop path is clean
    enough to begin again — the stop-path deadlock is the risk this guards."""

    def _url(self, path: str) -> str:
        return f"{self.server_url}{path}"

    def test_snap_during_streaming_is_rejected(self):
        channel = _one_channel()
        status, data = make_request(
            self._url("/api/action/stream_channel_begin"), data={"channel": channel},
            token=self.lease,
        )
        assert status == 200, f"stream begin failed: {status} {data}"
        try:
            status, data = make_request(
                self._url("/api/action/snap_channel"), data={"channel": channel},
                token=self.lease,
            )
            # rejected by the streaming gate (400) — a different path from the 409 busy lock
            assert status == 400, f"expected 400 (rejected during streaming), got {status}: {data}"
        finally:
            make_request(self._url("/api/action/stream_channel_end"), data={"channel": channel}, token=self.lease)

    def test_stream_begin_end_begin_again(self):
        channel = _one_channel()

        s1, d1 = make_request(self._url("/api/action/stream_channel_begin"), data={"channel": channel}, token=self.lease)
        assert s1 == 200, f"first begin failed: {s1} {d1}"

        s2, d2 = make_request(self._url("/api/action/stream_channel_end"), data={"channel": channel}, token=self.lease)
        assert s2 == 200, f"end failed: {s2} {d2}"

        # begin again after a clean end must work — a stop that left bad state would fail here
        s3, d3 = make_request(self._url("/api/action/stream_channel_begin"), data={"channel": channel}, token=self.lease)
        assert s3 == 200, f"second begin failed (stop left bad state?): {s3} {d3}"

        s4, _ = make_request(self._url("/api/action/stream_channel_end"), data={"channel": channel}, token=self.lease)
        assert s4 == 200, f"final end failed: {s4}"


@pytest.mark.usefixtures("running_server_with_delays", "hardware_lease")
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
            self._url("/api/acquisition/start"), data={"config_file": config}, timeout=30,
            token=self.lease,
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
