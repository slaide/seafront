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
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect as ws_connect

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

# a millimetre apart: for tests that care how many commands run, not how long each one takes
_HOP_A = {"x_mm": 60.0, "y_mm": 40.0, "z_mm": 1.0}
_HOP_B = {"x_mm": 61.0, "y_mm": 40.0, "z_mm": 1.0}


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


@pytest.mark.usefixtures("running_server_with_delays", "hardware_lease")
class TestUnderLoad:
    """
    Repetition and contention, because a race that shows up once in fifty runs shows up every
    run if you do it fifty times. Each test states an outcome that must hold for *every*
    attempt, so there is no threshold to argue about afterwards.
    """

    def _url(self, path: str) -> str:
        return f"{self.server_url}{path}"

    def test_every_answer_is_either_done_or_busy(self):
        """
        Six clients, one lease, all pushing at once: every single response is 200 or 409.

        Anything else - a 423 because the lease got lost, a 500 from two commands meeting -
        means contention produced something other than "your turn" or "not now".
        """
        answers: list[tuple[int, dict | None]] = []
        answers_lock = threading.Lock()
        barrier = threading.Barrier(6)

        def client(n: int):
            barrier.wait()
            mine = []
            for i in range(12):
                target = {"x_mm": 60.0 + n * 0.1, "y_mm": 40.0 + i * 0.1, "z_mm": 1.0}
                mine.append(make_request(self._url("/api/action/move_to"), data=target, timeout=30, token=self.lease))
            with answers_lock:
                answers.extend(mine)

        threads = [threading.Thread(target=client, args=(n,)) for n in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        unexpected = [(s, d) for s, d in answers if s not in (200, 409)]
        assert not unexpected, f"{len(unexpected)} of {len(answers)} answers were neither done nor busy: {unexpected[:3]}"
        assert any(s == 200 for s, _ in answers), "nothing ran at all"
        assert any(s == 409 for s, _ in answers), "no contention happened; the test proved nothing"

        # the hardware is free afterwards, and still ours
        status, data = make_request(self._url("/api/get_info/current_state"), timeout=5)
        assert status == 200 and data is not None and data["is_busy"] is False
        status, _ = make_request(self._url("/api/action/turn_off_all_illumination"), timeout=30, token=self.lease)
        assert status == 200, "the hardware never came back after the storm"

    def test_rapid_cycles_never_leak_the_hardware(self):
        """
        Forty commands back to back, each waited for: all forty must run.

        A claim that is not released on some path shows up here as the first 409 that should
        never have happened, rather than as a server that mysteriously stops responding a
        week later.
        """
        for i in range(40):
            target = _HOP_A if i % 2 else _HOP_B
            status, data = make_request(self._url("/api/action/move_to"), data=target, timeout=30, token=self.lease)
            assert status == 200, f"command {i} was refused: {status} {data}"

    def test_cancelling_at_any_moment_leaves_the_hardware_usable(self):
        """
        Cancel fired continuously while commands come and go.

        Cancel is the one hatch that reaches into a running command from outside, so a cancel
        landing on the boundary between two commands is the likeliest way to wedge the worker
        or to kill a command nobody asked to stop. Every answer must still be a clean one,
        and the hardware must still work afterwards.
        """
        stop = threading.Event()

        def cancel_storm():
            while not stop.is_set():
                status, _ = make_request(self._url("/api/action/cancel_current"), timeout=10)
                assert status == 200, f"the cancel hatch itself was refused: {status}"

        storm = threading.Thread(target=cancel_storm)
        storm.start()
        answers: list[tuple[int, dict | None]] = []
        try:
            for i in range(15):
                target = _START if i % 2 else _NEAR
                answers.append(make_request(self._url("/api/action/move_to"), data=target, timeout=30, token=self.lease))
        finally:
            stop.set()
            storm.join()

        for status, data in answers:
            if status == 200:
                continue
            assert status == 500 and data is not None, f"unexpected answer under cancel: {status} {data}"
            assert "cancel" in str(data.get("detail", "")).lower(), f"500 that was not a cancellation: {data}"

        # not wedged: the worker is free and takes work again
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status, data = make_request(self._url("/api/get_info/current_state"), timeout=5)
            assert status == 200 and data is not None
            if not data["is_busy"]:
                break
        else:
            pytest.fail("the worker never went idle after the cancel storm")
        status, data = make_request(self._url("/api/action/move_to"), data=_START, timeout=30, token=self.lease)
        assert status == 200, f"the hardware was left unusable: {status} {data}"


@pytest.mark.usefixtures("running_server_with_delays", "hardware_lease")
class TestStatusNeverContradictsItself:
    """
    Every status read is one snapshot, and a snapshot must not disagree with itself.

    Two fields describing the same thing are two fields that can be updated a moment apart,
    and a client polling in that moment reads a microscope that is streaming nothing, or busy
    with no reason, or locked by nobody. These sample status flat out across the transitions
    where that gap would open.
    """

    CYCLES = 12

    def _url(self, path: str) -> str:
        return f"{self.server_url}{path}"

    @staticmethod
    def _contradictions(state: dict) -> list[str]:
        """everything in one status snapshot that cannot be true at the same time"""
        wrong = []
        if state["is_busy"] != bool(state["busy_reasons"]):
            wrong.append(f"is_busy={state['is_busy']} but busy_reasons={state['busy_reasons']}")
        if state["is_streaming"] != (state["streaming_channel_handle"] is not None):
            wrong.append(
                f"is_streaming={state['is_streaming']} but "
                f"streaming_channel_handle={state['streaming_channel_handle']!r}"
            )
        busy_with_run = [r for r in state["busy_reasons"] if r.startswith("hardware busy: acquisition:")]
        if busy_with_run:
            running = busy_with_run[0].split("acquisition:", 1)[1]
            if state["current_acquisition_id"] != running:
                wrong.append(
                    f"busy with acquisition {running} but "
                    f"current_acquisition_id={state['current_acquisition_id']!r}"
                )
        elif state["current_acquisition_id"] is not None and not state["is_busy"]:
            wrong.append(
                f"current_acquisition_id={state['current_acquisition_id']!r} with an idle microscope"
            )
        lock = state["lock"]
        if lock["held"] != (lock["owner"] is not None):
            wrong.append(f"lock held={lock['held']} but owner={lock['owner']!r}")
        if lock["held"] != (lock["expires_at"] is not None):
            wrong.append(f"lock held={lock['held']} but expires_at={lock['expires_at']!r}")
        return wrong

    def _poll_while(self, work) -> tuple[int, list[str]]:
        """run `work` while sampling status over a kept-alive connection"""
        samples = 0
        wrong: list[str] = []
        stop = threading.Event()

        def poll():
            nonlocal samples
            conn = http.client.HTTPConnection("127.0.0.1", 5000, timeout=10)
            body = json.dumps({"target_device": MOCK_MICROSCOPE_NAME})
            headers = {"Content-Type": "application/json"}
            try:
                while not stop.is_set():
                    conn.request("POST", "/api/get_info/current_state", body, headers)
                    state = json.loads(conn.getresponse().read().decode())
                    samples += 1
                    wrong.extend(f"{c} (sample {samples})" for c in self._contradictions(state))
            finally:
                conn.close()

        poller = threading.Thread(target=poll)
        poller.start()
        try:
            work()
        finally:
            stop.set()
            poller.join()
        return samples, wrong

    def test_streaming_transitions_never_show_half_a_stream(self):
        """
        Start and stop streaming over and over while watching.

        `is_streaming` and the channel handle come from one locked read of the adapter, so a
        client polling through a transition can never see a stream with no channel, or a
        channel that is not streaming. They used to be set a moment apart - the stream stopped
        by the command on the worker, the bookkeeping by the route callback that followed.
        """
        channel = _one_channel()

        def cycles():
            for i in range(self.CYCLES):
                s1, d1 = make_request(
                    self._url("/api/action/stream_channel_begin"),
                    data={"channel": channel}, timeout=30, token=self.lease,
                )
                assert s1 == 200, f"cycle {i}: begin failed: {s1} {d1}"
                s2, d2 = make_request(
                    self._url("/api/action/stream_channel_end"),
                    data={}, timeout=30, token=self.lease,
                )
                assert s2 == 200, f"cycle {i}: end failed: {s2} {d2}"

        samples, wrong = self._poll_while(cycles)
        assert samples >= 100, f"too few status samples to prove anything ({samples})"
        assert not wrong, f"{len(wrong)} of {samples} status reads contradicted themselves: {wrong[:5]}"

    def test_command_transitions_never_show_half_a_command(self):
        """the same watch across ordinary command starts and finishes"""

        def commands():
            for i in range(self.CYCLES * 3):
                target = _HOP_A if i % 2 else _HOP_B
                status, data = make_request(
                    self._url("/api/action/move_to"), data=target, timeout=30, token=self.lease
                )
                assert status == 200, f"cycle {i}: move failed: {status} {data}"

        samples, wrong = self._poll_while(commands)
        assert samples >= 100, f"too few status samples to prove anything ({samples})"
        assert not wrong, f"{len(wrong)} of {samples} status reads contradicted themselves: {wrong[:5]}"

    def test_acquisition_transitions_never_show_half_a_run(self):
        """
        Start a run, let it get going, cancel it, and watch the whole thing.

        Both ends of a run are windows: the job is on the worker before its bookkeeping says
        so, and its bookkeeping stops saying so while the worker still holds it. A client
        polling in either window sees a microscope busy with an acquisition it cannot name.
        """
        config = _cancellable_config()

        def run():
            status, data = make_request(
                self._url("/api/acquisition/start"), data={"config_file": config}, timeout=30, token=self.lease
            )
            assert status == 200 and data is not None, f"start failed: {status} {data}"
            acq_id = data["acquisition_id"]
            deadline = time.monotonic() + 30
            while True:
                status, state = make_request(
                    self._url("/api/acquisition/status"), data={"acquisition_id": acq_id}, timeout=10
                )
                assert status == 200 and state is not None
                if state["acquisition_progress"]["current_num_images"] >= 2:
                    break
                assert time.monotonic() < deadline, "the run never produced images"
            make_request(
                self._url("/api/acquisition/cancel"), data={"acquisition_id": acq_id}, timeout=10
            )
            while time.monotonic() < deadline:
                status, state = make_request(
                    self._url("/api/acquisition/status"), data={"acquisition_id": acq_id}, timeout=10
                )
                if state is not None and state["acquisition_status"] not in ("running", "scheduled"):
                    return
            pytest.fail("the run never ended")

        samples, wrong = self._poll_while(run)
        assert samples >= 100, f"too few status samples to prove anything ({samples})"
        assert not wrong, f"{len(wrong)} of {samples} status reads contradicted themselves: {wrong[:5]}"

    def test_lease_transitions_never_show_half_a_lease(self):
        """and across lease churn, which the status broadcast also carries"""

        def churn():
            for i in range(self.CYCLES * 20):
                status, body = make_request(
                    self._url("/api/lock/acquire"), data={"owner": f"churn-{i}"}, timeout=10
                )
                assert status == 200, f"acquire failed: {status} {body}"
                assert body is not None
                status, _ = make_request(
                    self._url("/api/lock/release"), data={"lock_token": body["token"]}, timeout=10
                )
                assert status == 200

        # this test owns the lease for its own churn, so hand the class lease back first
        make_request(self._url("/api/lock/break"), data={"reason": "lease churn test"})
        try:
            samples, wrong = self._poll_while(churn)
        finally:
            status, body = make_request(
                self._url("/api/lock/acquire"), data={"owner": "test_concurrency"}, timeout=10
            )
            assert status == 200 and body is not None
            type(self).lease = body["token"]
        assert samples >= 100, f"too few status samples to prove anything ({samples})"
        assert not wrong, f"{len(wrong)} of {samples} status reads contradicted themselves: {wrong[:5]}"


@pytest.mark.usefixtures("running_server_with_delays", "hardware_lease")
class TestStatusWebsocket:
    """
    The status socket is how the UI watches the microscope, and it shares the state read with
    the HTTP route. Several clients watching at once, or one watching while the hardware is
    busy, must not interfere with each other or with the request that is running.
    """

    def _url(self, path: str) -> str:
        return f"{self.server_url}{path}"

    @staticmethod
    def _state(ws) -> dict:
        """the socket answers every message with one state snapshot"""
        ws.send(json.dumps({"anything": "at all"}))
        return json.loads(ws.recv(timeout=10))

    def test_the_socket_answers_every_poke_with_a_state(self):
        with ws_connect("ws://127.0.0.1:5000/ws/get_info/current_state") as ws:
            ws.send(json.dumps({"target_device": MOCK_MICROSCOPE_NAME}))
            for _ in range(20):
                state = self._state(ws)
                assert state["microscope_name"] == MOCK_MICROSCOPE_NAME
                assert "adapter_state" in state and "is_busy" in state

    def test_a_socket_client_sees_the_hardware_go_busy_and_idle_again(self):
        """the socket is a live view, not a snapshot taken when it was opened"""
        with ws_connect("ws://127.0.0.1:5000/ws/get_info/current_state") as ws:
            ws.send(json.dumps({"target_device": MOCK_MICROSCOPE_NAME}))
            assert self._state(ws)["is_busy"] is False

            result: dict = {}

            def move():
                result["status"], _ = make_request(
                    self._url("/api/action/move_to"), data=_FAR, timeout=30, token=self.lease
                )

            make_request(self._url("/api/action/move_to"), data=_START, timeout=30, token=self.lease)
            thread = threading.Thread(target=move)
            thread.start()
            try:
                seen_busy = False
                while thread.is_alive():
                    state = self._state(ws)
                    if state["is_busy"]:
                        seen_busy = True
                        assert state["busy_reasons"], "busy with no reason over the socket"
            finally:
                thread.join()
            assert result["status"] == 200
            assert seen_busy, "the socket never showed the move running"
            assert self._state(ws)["is_busy"] is False

    def test_many_sockets_watch_at_once_without_disturbing_each_other(self):
        """eight watchers plus HTTP status: everyone gets served, nobody gets an error"""
        errors: list[str] = []
        stop = threading.Event()
        ready = threading.Barrier(9)

        def watcher():
            try:
                with ws_connect("ws://127.0.0.1:5000/ws/get_info/current_state") as ws:
                    ws.send(json.dumps({"target_device": MOCK_MICROSCOPE_NAME}))
                    ready.wait(timeout=20)
                    while not stop.is_set():
                        state = self._state(ws)
                        if "error" in state:
                            errors.append(f"socket error: {state}")
                        if state["is_busy"] != bool(state["busy_reasons"]):
                            errors.append(f"contradictory state over the socket: {state}")
            except Exception as e:  # any failure at all is the finding here
                errors.append(f"{type(e).__name__}: {e}")

        watchers = [threading.Thread(target=watcher) for _ in range(8)]
        for t in watchers:
            t.start()
        try:
            ready.wait(timeout=20)
            for i in range(10):
                target = _HOP_A if i % 2 else _HOP_B
                status, data = make_request(self._url("/api/action/move_to"), data=target, timeout=30, token=self.lease)
                assert status == 200, f"a command failed while eight sockets watched: {status} {data}"
                status, _ = make_request(self._url("/api/get_info/current_state"), timeout=5)
                assert status == 200, "an HTTP status read failed while eight sockets watched"
        finally:
            stop.set()
            for t in watchers:
                t.join()
        assert not errors, errors[:5]

    def test_a_socket_for_the_wrong_microscope_is_closed(self):
        with ws_connect("ws://127.0.0.1:5000/ws/get_info/current_state") as ws:
            ws.send(json.dumps({"target_device": "some-other-microscope"}))
            reply = json.loads(ws.recv(timeout=10))
            assert reply["error"] == "target_device_mismatch"
            with pytest.raises(ConnectionClosed) as closed:
                ws.recv(timeout=5)
            assert closed.value.rcvd is not None and closed.value.rcvd.code == 4421


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
            make_request(self._url("/api/action/stream_channel_end"), data={}, token=self.lease)

    def test_stream_begin_end_begin_again(self):
        channel = _one_channel()

        s1, d1 = make_request(self._url("/api/action/stream_channel_begin"), data={"channel": channel}, token=self.lease)
        assert s1 == 200, f"first begin failed: {s1} {d1}"

        s2, d2 = make_request(self._url("/api/action/stream_channel_end"), data={}, token=self.lease)
        assert s2 == 200, f"end failed: {s2} {d2}"

        # begin again after a clean end must work — a stop that left bad state would fail here
        s3, d3 = make_request(self._url("/api/action/stream_channel_begin"), data={"channel": channel}, token=self.lease)
        assert s3 == 200, f"second begin failed (stop left bad state?): {s3} {d3}"

        s4, _ = make_request(self._url("/api/action/stream_channel_end"), data={}, token=self.lease)
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


@pytest.mark.usefixtures("running_server_with_delays", "hardware_lease")
class TestAcquisitionLifecycle:
    """
    An acquisition holds the hardware for minutes, so it is the longest window in which
    anything else can go wrong: a second run started on top of it, progress read while images
    are being written, state left behind by a cancelled run. Each test states what must be
    true for the whole window, not just at the end.
    """

    def _url(self, path: str) -> str:
        return f"{self.server_url}{path}"

    def _start(self, config: dict) -> tuple[int, dict | None]:
        return make_request(
            self._url("/api/acquisition/start"), data={"config_file": config}, timeout=30, token=self.lease
        )

    def _status(self, acq_id: str) -> dict:
        status, data = make_request(
            self._url("/api/acquisition/status"), data={"acquisition_id": acq_id}, timeout=10
        )
        assert status == 200, f"status query failed: {status} {data}"
        assert data is not None
        return data

    def _wait_for_end(self, acq_id: str, timeout_s: float = 120) -> str:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            state = self._status(acq_id)["acquisition_status"]
            if state not in ("running", "scheduled"):
                return state
        pytest.fail(f"acquisition {acq_id} never finished")

    def _cancel(self, acq_id: str) -> None:
        status, data = make_request(
            self._url("/api/acquisition/cancel"), data={"acquisition_id": acq_id}, timeout=10
        )
        assert status == 200, f"cancel failed: {status} {data}"

    def test_a_second_acquisition_cannot_start_on_top_of_one(self):
        """
        The first run owns the hardware until it is done.

        A second start must be refused while it runs - and refusing it must not disturb the
        run that is already going.
        """
        status, data = self._start(_cancellable_config())
        assert status == 200 and data is not None, f"first start failed: {status} {data}"
        acq_id = data["acquisition_id"]
        try:
            refusals = []
            for _ in range(5):
                refusals.append(self._start(_cancellable_config()))
            assert all(s == 409 for s, _ in refusals), f"a second run was accepted: {refusals}"
            assert self._status(acq_id)["acquisition_status"] in ("running", "scheduled")
        finally:
            self._cancel(acq_id)
        assert self._wait_for_end(acq_id) == "cancelled"

    def test_progress_only_moves_forward_while_images_are_written(self):
        """
        Poll progress flat out for the length of a run.

        Status is read from the same server-side state the acquisition is writing to, so this
        is where a read of half-written bookkeeping would show up: a count that goes
        backwards, a total that changes under it, or a status read that simply fails.
        """
        status, data = self._start(_cancellable_config())
        assert status == 200 and data is not None, f"start failed: {status} {data}"
        acq_id = data["acquisition_id"]
        try:
            seen = 0
            total: int | None = None
            polls = 0
            while True:
                state = self._status(acq_id)
                progress = state["acquisition_progress"]
                done = progress["current_num_images"]
                assert done >= seen, f"progress went backwards: {seen} then {done}"
                seen = done
                if total is None:
                    total = state["acquisition_meta_information"]["total_num_images"]
                assert state["acquisition_meta_information"]["total_num_images"] == total, (
                    "the image total changed mid-run"
                )
                polls += 1
                # the whole-state read is served from the same bookkeeping; it must also hold
                # up, and name the run inside one snapshot: a microscope that reports itself
                # busy with an acquisition has to be able to say which one
                status, current = make_request(self._url("/api/get_info/current_state"), timeout=5)
                assert status == 200, f"a status read failed during the acquisition: {status} {current}"
                assert current is not None
                busy_with = [r for r in current["busy_reasons"] if r.startswith("hardware busy: acquisition:")]
                if busy_with:
                    running_id = busy_with[0].split("acquisition:", 1)[1]
                    assert current["current_acquisition_id"] == running_id, (
                        f"busy with {running_id} but current_acquisition_id="
                        f"{current['current_acquisition_id']!r}"
                    )
                if state["acquisition_status"] not in ("running", "scheduled"):
                    break
            assert polls >= 10, f"the run ended too fast to prove anything ({polls} polls)"
            assert seen > 0, "no images were reported at all"
        finally:
            if self._status(acq_id)["acquisition_status"] in ("running", "scheduled"):
                self._cancel(acq_id)
                self._wait_for_end(acq_id)

    def test_a_cancelled_run_leaves_nothing_behind(self):
        """
        Cancel, then start again: the second run must behave like the first.

        A run that leaves its future, its map entry or the worker claim behind is the reason
        the *next* acquisition fails for no visible reason.
        """
        status, data = self._start(_cancellable_config())
        assert status == 200 and data is not None
        first = data["acquisition_id"]
        # let it actually get going before pulling the rug
        deadline = time.monotonic() + 15
        while self._status(first)["acquisition_progress"]["current_num_images"] < 1:
            assert time.monotonic() < deadline, "the first run never produced an image"
        self._cancel(first)
        assert self._wait_for_end(first) == "cancelled"

        # the hardware comes back: a cancelled run keeps the worker while it tears down, so a
        # client waits for the microscope to report itself free before starting the next one
        deadline = time.monotonic() + 30
        while True:
            status, data = make_request(self._url("/api/get_info/current_state"), timeout=5)
            assert status == 200 and data is not None
            if not data["is_busy"] and data["current_acquisition_id"] is None:
                break
            assert time.monotonic() < deadline, f"the cancelled run never let go: {data['busy_reasons']}"

        config = _cancellable_config()
        config["channels"][0]["num_z_planes"] = 2  # short enough to run to the end
        status, data = self._start(config)
        assert status == 200 and data is not None, f"the run after a cancel was refused: {status} {data}"
        second = data["acquisition_id"]
        assert second != first
        assert self._wait_for_end(second) == "completed"

    def test_reads_and_cancel_stay_open_while_an_acquisition_runs(self):
        """
        The run owns the hardware, not the server: everyone can still look, and anyone can
        still stop it.
        """
        status, data = self._start(_cancellable_config())
        assert status == 200 and data is not None
        acq_id = data["acquisition_id"]
        try:
            for path, body in (
                ("/api/get_info/current_state", {}),
                ("/api/get_features/hardware_capabilities", {}),
                ("/api/acquisition/config_list", {}),
                ("/api/lock/status", {}),
            ):
                status, response = make_request(self._url(path), data=body, timeout=10)
                assert status == 200, f"{path} was refused during an acquisition: {status} {response}"

            # hardware commands are not: the run holds the worker
            status, _ = make_request(self._url("/api/action/move_to"), data=_START, timeout=10, token=self.lease)
            assert status in (400, 409), f"a command ran during an acquisition: {status}"
        finally:
            # cancel needs no lease: it is the safety hatch
            status, data = make_request(
                self._url("/api/acquisition/cancel"), data={"acquisition_id": acq_id}, timeout=10
            )
            assert status == 200, f"cancel was refused: {status} {data}"
        assert self._wait_for_end(acq_id) == "cancelled"

    def test_asking_about_an_acquisition_that_does_not_exist_is_an_error_not_a_crash(self):
        status, data = make_request(
            self._url("/api/acquisition/status"), data={"acquisition_id": "no-such-run"}, timeout=10
        )
        assert status >= 400 and data is not None
        assert "acquisition_id" in str(data.get("detail", "")).lower(), data
        status, data = make_request(
            self._url("/api/acquisition/cancel"), data={"acquisition_id": "no-such-run"}, timeout=10
        )
        assert status >= 400 and data is not None
        assert "acquisition_id" in str(data.get("detail", "")).lower(), data
