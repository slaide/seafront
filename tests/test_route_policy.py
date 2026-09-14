"""
The published policy is the enforced policy - all three parts of it, not just the lease.

`Policy` says three things about a route: does it need the device lease, may it run during an
acquisition, may it run during a live stream. All three are published per route in
/openapi.json (`x-requires-device-lease`, `x-allowed-while-acquiring`,
`x-allowed-while-streaming`), and all three are read back here and held against what the
running server actually does.

tests/test_lock_enforcement.py does this for the lease. This module does it for the other two,
which had no published form and no sweep at all: a route could be documented as safe during an
acquisition and be refused, or be flagged unsafe and run anyway, and nothing would notice.

The sweep asserts on the *refusal*, not on the status code, because a route that gets past the
gate may then fail for its own reasons (409 from the busy worker, 500 from a mock that dislikes
its arguments). What the policy promises is precisely whether the state gate speaks, so that is
what is checked: the exact 400 message `admit` produces, present or absent.
"""

import json
import os
import re
import shutil
import socket
import subprocess
import time
import typing as tp
import urllib.error
import urllib.request
from pathlib import Path

import pytest

MOCK = "mocroscope"
PORT = 5013  # 5000, 5011 and 5012 belong to the other server fixtures
SERVER = f"http://127.0.0.1:{PORT}"

ACQUIRING_REFUSAL = "cannot run this command while acquisition is running"
STREAMING_REFUSAL = "cannot run this command while live streaming is active"

# The intended policy, written out by hand.
#
# The sweeps below prove the server does what it publishes. They cannot prove that what it
# publishes is what anyone meant: relax a route's Policy and the published flag and the
# enforcement move together, so a consistency check stays green. These two sets are the
# other half - loosening one of them is a deliberate edit here, not a diff nobody notices.
#
# Everything absent from a set may run in that state.

NOT_WHILE_STREAMING = {
    # want the camera to themselves
    "/api/action/snap_channel",
    "/api/action/snap_selected_channels",
    "/api/action/snap_reflection_autofocus",
    "/api/action/laser_autofocus_calibrate",
    "/api/action/laser_autofocus_measure_displacement",
    "/api/action/laser_autofocus_move_to_target_offset",
    "/api/action/laser_autofocus_warm_up_laser",
    "/api/acquisition/start",
    # one stream at a time
    "/api/action/stream_channel_begin",
    # nothing to cancel: a stream is not an acquisition, and stopping one is stream_channel_end
    "/api/acquisition/cancel",
}

NOT_WHILE_ACQUIRING = {
    # move the stage somewhere a running protocol does not expect it to be
    "/api/action/enter_loading_position",
    "/api/action/leave_loading_position",
    # redefines where every well is, mid-plate
    "/api/action/calibrate_stage_xy_here",
    # takes the camera while the run needs it
    "/api/action/stream_channel_begin",
}

# Routes the sweep does not drive, and why. Everything else is fair game in every state.
SKIP = {
    # would end the very state the sweep is trying to hold the server in
    "/api/action/stream_channel_end",
    "/api/acquisition/cancel",
    "/api/action/cancel_current",
    # takes the lease away from the sweep itself
    "/api/lock/break",
    "/api/lock/release",
    "/api/lock/acquire",
    # writes a protocol file into the server's home; nothing to do with hardware state
    "/api/acquisition/config_store",
}


def _port_free() -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", PORT)) != 0


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """mock server on a scratch copy of the seafront home, on its own port"""
    real_home = Path.home() / "seafront"
    if not (real_home / "config.json").exists():
        pytest.skip("no ~/seafront/config.json to copy")
    home = tmp_path_factory.mktemp("seafront-home")
    cfg = json.loads(re.sub(r"//.*", "", (real_home / "config.json").read_text()))
    mock = next((m for m in cfg["microscopes"] if m.get("system.microscope_name") == MOCK), None)
    if mock is None:
        pytest.skip(f"no '{MOCK}' in ~/seafront/config.json")
    images = home / "images"
    images.mkdir()
    mock["storage.base_image_output_dir"] = str(images)
    (home / "config.json").write_text(json.dumps(cfg, indent=2))

    assert _port_free(), f"port {PORT} is busy; another seafront is running"
    # delays ON, unlike the other server fixtures: MOCK_NO_DELAYS skips exposure time, and
    # TestWhileAcquiring needs a run that lasts longer than the sweep it is holding open.
    # Costs little here - during a run every route is refused at the gate or by the busy
    # worker, so the sweep never waits for hardware.
    env = dict(os.environ, SEAFRONT_HOME=str(home))
    env.pop("MOCK_NO_DELAYS", None)
    # never a PIPE nobody reads: a full pipe buffer blocks the server's event loop
    log = (home / "server.log").open("w")
    proc = subprocess.Popen(
        ["uv", "run", "python", "-m", "seafront", "--microscope", MOCK, "--port", str(PORT)],
        stdout=log, stderr=subprocess.STDOUT, text=True, env=env,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{SERVER}/api/health", timeout=1) as r:
                if r.status == 200:
                    break
        except (urllib.error.URLError, ConnectionRefusedError):
            time.sleep(0.3)
    else:
        proc.terminate()
        raise RuntimeError("test server did not start")
    yield proc
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()
    shutil.rmtree(home, ignore_errors=True)


def post(path: str, data: dict | None = None, token: str | None = None) -> tuple[int, tp.Any]:
    body = dict(data or {})
    body["target_device"] = MOCK
    if token is not None:
        body["lock_token"] = token
    req = urllib.request.Request(f"{SERVER}{path}", method="POST")
    req.add_header("Content-Type", "application/json")
    req.data = json.dumps(body).encode()
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            raw = response.read().decode()
            return response.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raw = e.read().decode() if e.fp else ""
        return e.code, json.loads(raw) if raw else None


def refusal_text(status: int, body: tp.Any) -> str:
    """the flat {status, message} a 400 carries; empty for anything else"""
    if status != 400 or not isinstance(body, dict):
        return ""
    return str(body.get("message", ""))


def lease(owner: str = "policy sweep") -> str:
    post("/api/lock/break", {"reason": "policy sweep setup"})
    status, body = post("/api/lock/acquire", {"owner": owner})
    assert status == 200, (status, body)
    return body["token"]


@pytest.fixture(scope="module")
def schema(server) -> dict:
    with urllib.request.urlopen(f"{SERVER}/openapi.json", timeout=10) as r:
        return json.loads(r.read().decode())


@pytest.fixture(scope="module")
def policies(schema) -> dict[str, dict]:
    """every POST route's published policy, keyed by path"""
    out = {}
    for path, ops in schema["paths"].items():
        op = ops.get("post")
        if op is None:
            continue
        out[path] = {
            "lease": op.get("x-requires-device-lease"),
            "acquiring": op.get("x-allowed-while-acquiring"),
            "streaming": op.get("x-allowed-while-streaming"),
        }
    return out


@pytest.fixture(scope="module")
def bodies(server) -> dict[str, dict]:
    """a schema-valid body for every POST route the sweep drives"""
    status, listing = post("/api/acquisition/config_list")
    assert status == 200, (status, listing)
    name = listing["configs"][0]["filename"]
    status, fetched = post("/api/acquisition/config_fetch", {"config_file": name})
    assert status == 200, (status, fetched)
    protocol = fetched["file"]
    channel = protocol["channels"][0]
    plate = protocol["wellplate_type"]

    return {
        "/api/acquisition/start": {"config_file": protocol},
        "/api/acquisition/estimate": {"config_file": protocol},
        "/api/acquisition/status": {"acquisition_id": "does-not-exist"},
        "/api/acquisition/config_list": {},
        "/api/acquisition/config_fetch": {"config_file": name},
        "/api/action/calibrate_stage_xy_here": {},
        "/api/action/enter_loading_position": {},
        "/api/action/establish_hardware_connection": {},
        "/api/action/laser_autofocus_calibrate": {},
        "/api/action/laser_autofocus_measure_displacement": {"config_file": protocol},
        "/api/action/laser_autofocus_move_to_target_offset": {
            "config_file": protocol,
            "target_offset_um": 0.0,
        },
        "/api/action/laser_autofocus_warm_up_laser": {},
        "/api/action/leave_loading_position": {},
        "/api/action/machine_config_flush": {"machine_config": []},
        "/api/action/move_by": {"axis": "x", "distance_mm": 0.0},
        "/api/action/move_to": {"x_mm": 60.0, "y_mm": 40.0},
        "/api/action/move_to_well": {"plate_type": plate, "well_name": "B02"},
        "/api/action/snap_channel": {"channel": channel},
        "/api/action/snap_reflection_autofocus": {},
        "/api/action/snap_selected_channels": {"config_file": protocol},
        "/api/action/stream_channel_begin": {"channel": channel},
        "/api/action/turn_off_all_illumination": {},
        "/api/get_features/hardware_capabilities": {},
        "/api/get_features/machine_defaults": {},
        "/api/get_info/current_state": {},
        "/api/lock/status": {},
    }


@pytest.fixture(scope="module")
def long_protocol(bodies) -> dict:
    """
    a protocol that takes ~15s on the mock, so the sweep runs well inside it

    One well (B02 - A01 is inside a forbidden area), one channel, and the length comes from
    z-planes at a long exposure rather than from many sites, so nothing moves much.
    """
    protocol = json.loads(json.dumps(bodies["/api/acquisition/start"]["config_file"]))
    for channel in protocol["channels"]:
        channel["enabled"] = False
    protocol["channels"][0].update(
        enabled=True, exposure_time_ms=500.0, num_z_planes=30, delta_z_um=1.0
    )
    protocol["plate_wells"] = [{"row": 1, "col": 1, "selected": True}]
    protocol["autofocus_enabled"] = False
    protocol["project_name"] = "policysweep"
    protocol["plate_name"] = "p1"
    return protocol


@pytest.fixture(scope="module")
def driven(policies, bodies) -> list[str]:
    """the routes the sweep drives, in a stable order"""
    return sorted(set(policies) - SKIP)


@pytest.fixture
def idle(server):
    """start and end with nothing held, nothing streaming, nothing running"""
    _settle()
    yield
    _settle()


def _settle(timeout_s: float = 30.0) -> None:
    """
    Put the microscope back to doing nothing, and wait until it really is.

    Cancelling is cooperative: the run stops at its next checkpoint, so a test that cancels
    and returns leaves the worker busy for a moment afterwards. Without the wait the next
    test's first command comes back 409 and the failure lands nowhere near its cause.
    """
    post("/api/lock/break", {"reason": "policy sweep"})
    token = lease()

    state = post("/api/get_info/current_state")[1]
    if state["is_streaming"]:
        post("/api/action/stream_channel_end", {}, token=token)
    if state["current_acquisition_id"] is not None:
        post("/api/acquisition/cancel", {"acquisition_id": state["current_acquisition_id"]})

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        state = post("/api/get_info/current_state")[1]
        if not state["is_busy"] and not state["is_streaming"]:
            break
        time.sleep(0.2)
    else:
        raise RuntimeError(f"microscope would not go idle within {timeout_s}s: {state}")

    post("/api/lock/break", {"reason": "policy sweep"})


class TestEveryRouteDeclaresAllThreeParts:
    def test_the_policy_is_published_per_route(self, policies):
        for path, policy in policies.items():
            assert policy["lease"] is not None, f"{path} publishes no x-requires-device-lease"
            assert policy["acquiring"] is not None, f"{path} publishes no x-allowed-while-acquiring"
            assert policy["streaming"] is not None, f"{path} publishes no x-allowed-while-streaming"

    def test_the_sweep_has_a_body_for_every_route_it_drives(self, driven, bodies):
        assert set(driven) == set(bodies), (
            f"missing bodies: {set(driven) - set(bodies)}; stale bodies: {set(bodies) - set(driven)}"
        )

    def test_the_published_policy_is_the_intended_policy(self, policies):
        """
        what the server says, against what this repo means

        A mismatch here is either a route whose Policy was changed without anyone deciding
        to, or a deliberate change that has not been written down yet. Both want a human.
        """
        streaming = {path for path, p in policies.items() if p["streaming"] is False}
        acquiring = {path for path, p in policies.items() if p["acquiring"] is False}
        assert streaming == NOT_WHILE_STREAMING, (
            f"newly refused during a stream: {sorted(streaming - NOT_WHILE_STREAMING)}; "
            f"newly allowed during a stream: {sorted(NOT_WHILE_STREAMING - streaming)}"
        )
        assert acquiring == NOT_WHILE_ACQUIRING, (
            f"newly refused during a run: {sorted(acquiring - NOT_WHILE_ACQUIRING)}; "
            f"newly allowed during a run: {sorted(NOT_WHILE_ACQUIRING - acquiring)}"
        )

    def test_a_read_is_never_declared_state_restricted(self, policies):
        """
        a route that does not touch the hardware has no reason to be refused for what the
        hardware is doing - and a client watching somebody else's run depends on that
        """
        for path, policy in policies.items():
            if policy["lease"] is False and path.startswith(("/api/get_", "/api/lock/")):
                assert policy["acquiring"] is True, f"{path} is a read but refuses during a run"
                assert policy["streaming"] is True, f"{path} is a read but refuses during a stream"


class TestWhileIdle:
    def test_no_route_is_refused_for_state(self, driven, bodies, idle):
        """nothing is running, so no route may claim otherwise"""
        token = lease()
        wrong = []
        for path in driven:
            text = refusal_text(*post(path, bodies[path], token=token))
            if ACQUIRING_REFUSAL in text or STREAMING_REFUSAL in text:
                wrong.append(f"{path}: {text}")
            # a begin would leave a stream running under the rest of the sweep
            if path == "/api/action/stream_channel_begin":
                post("/api/action/stream_channel_end", {}, token=token)
        assert not wrong, wrong


class TestWhileStreaming:
    def test_each_route_is_refused_exactly_as_it_says_it_will_be(self, driven, bodies, policies, idle):
        token = lease()
        status, response = post(
            "/api/action/stream_channel_begin", bodies["/api/action/stream_channel_begin"], token=token
        )
        assert status == 200, (status, response)
        assert post("/api/get_info/current_state")[1]["is_streaming"] is True

        wrong = []
        for path in driven:
            if path == "/api/action/stream_channel_begin":
                continue  # would be refused by the adapter, not by the policy gate
            refused = STREAMING_REFUSAL in refusal_text(*post(path, bodies[path], token=token))
            allowed = policies[path]["streaming"]
            if refused == allowed:
                wrong.append(
                    f"{path}: published x-allowed-while-streaming={allowed}, "
                    f"but the server {'refused' if refused else 'allowed'} it"
                )
            # some allowed routes disturb the stream; put it back before judging the next one
            if not allowed:
                continue
            if not post("/api/get_info/current_state")[1]["is_streaming"]:
                post("/api/action/stream_channel_begin", bodies["/api/action/stream_channel_begin"], token=token)
        assert not wrong, wrong


class TestWhileAcquiring:
    def test_each_route_is_refused_exactly_as_it_says_it_will_be(
        self, driven, bodies, policies, long_protocol, idle
    ):
        token = lease()
        status, started = post("/api/acquisition/start", {"config_file": long_protocol}, token=token)
        assert status == 200, f"could not start an acquisition to sweep against: {status} {started}"

        try:
            wrong = []
            swept = 0
            for path in driven:
                if not post("/api/get_info/current_state")[1]["is_busy"]:
                    # Not a skip. A sweep that quietly stopped early would report green
                    # while having checked nothing, which is the failure mode this whole
                    # module exists to prevent.
                    pytest.fail(
                        f"the acquisition ended after {swept} of {len(driven)} routes; "
                        "lengthen long_protocol so the run outlasts the sweep"
                    )
                refused = ACQUIRING_REFUSAL in refusal_text(*post(path, bodies[path], token=token))
                allowed = policies[path]["acquiring"]
                if refused == allowed:
                    wrong.append(
                        f"{path}: published x-allowed-while-acquiring={allowed}, "
                        f"but the server {'refused' if refused else 'allowed'} it"
                    )
                swept += 1
            assert not wrong, wrong
            assert swept == len(driven)
        finally:
            post("/api/acquisition/cancel", {"acquisition_id": started["acquisition_id"]})


class TestUnknownFieldsAreRefused:
    """
    A field this server does not know is a 422, not something to quietly drop.

    Two of this repo's own tests used to send `framerate_hz` to stream_channel_begin, which
    has never had such a field; the server accepted the request and ignored it, so the tests
    passed and the framerate did nothing. A typo in a client has exactly that shape.
    """

    def test_an_unknown_field_is_refused(self, idle):
        token = lease()
        status, body = post(
            "/api/action/move_to", {"x_mm": 60.0, "y_mm": 40.0, "framerate_hz": 5.0}, token=token
        )
        assert status == 422, (status, body)
        assert any("framerate_hz" in str(err.get("loc", "")) for err in body["detail"]), body

    def test_the_same_body_without_it_is_accepted(self, idle):
        token = lease()
        assert post("/api/action/move_to", {"x_mm": 60.0, "y_mm": 40.0}, token=token)[0] == 200

    def test_the_envelope_fields_are_not_extra(self, idle):
        """target_device and lock_token are the envelope, not stowaways on the command"""
        token = lease()
        assert post("/api/action/move_to", {"x_mm": 60.0, "y_mm": 40.0}, token=token)[0] == 200

    def test_an_unknown_field_is_refused_before_the_lease_is_consulted(self, idle):
        """
        a malformed body is 422 whatever the lease says

        Same ordering the lease tests rely on in reverse: body validation comes first, so a
        422 can never be a disguised 423 and a 423 can never be a disguised typo.
        """
        post("/api/lock/break", {"reason": "unknown field test"})
        status, body = post("/api/action/move_to", {"x_mm": 60.0, "nonsense": 1})
        assert status == 422, (status, body)
