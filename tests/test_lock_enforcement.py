"""
The device lease is mandatory for hardware: these tests are the proof, from the outside.

tests/test_device_lock.py covers the lease object. This module covers the thing that actually
protects the microscope - that every route which drives hardware refuses a request that does
not carry the lease, whether that is because somebody else holds it or because nobody does,
and that reads and cancel stay open to everyone.

The route list is not written down here twice. The server publishes
`x-requires-device-lease` per route in /openapi.json and per websocket channel in
/asyncapi.json, and the sweeps below run over those, so a route added later is covered the day
it is added (and a route added without a sample request body fails TestSchemaContract until
someone writes one).
"""

import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import typing as tp
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect as ws_connect

from seafront.server.lock import LockReason

MOCK = "mocroscope"
PORT = 5011  # not 5000: the other server fixtures own that one
SERVER = f"http://127.0.0.1:{PORT}"
WS_SERVER = f"ws://127.0.0.1:{PORT}"

# routes that do NOT need the lease, and why. Everything else the server exposes must be
# gated; TestSchemaContract holds this list against the running server.
UNGATED_ROUTES = {
    # reads: never touch the hardware, so anyone may watch a run they do not own
    "/api/get_info/current_state",
    "/api/get_features/hardware_capabilities",
    "/api/get_features/machine_defaults",
    "/api/acquisition/status",
    "/api/acquisition/estimate",
    "/api/acquisition/config_list",
    "/api/acquisition/config_fetch",
    # writes a protocol file, not the microscope: preparing a protocol while somebody else
    # images is legitimate
    "/api/acquisition/config_store",
    # safety hatch: stopping the hardware must never depend on holding a lease
    "/api/acquisition/cancel",
    "/api/action/cancel_current",
    # the lease routes themselves
    "/api/lock/acquire",
    "/api/lock/release",
    "/api/lock/status",
    "/api/lock/break",
}

# hardware routes that are safe (and quick) to actually run on the mock, used to prove the
# gate lets the holder through rather than refusing everybody
SAFE_TO_RUN = [
    "/api/action/establish_hardware_connection",
    "/api/action/turn_off_all_illumination",
    "/api/action/machine_config_flush",
    "/api/action/move_to",
]


# --------------------------------------------------------------------------------------------
# server


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
    env = dict(os.environ, SEAFRONT_HOME=str(home), MOCK_NO_DELAYS="1")
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


# --------------------------------------------------------------------------------------------
# request helpers


def post(
    path: str,
    data: dict | None = None,
    token: str | None = None,
    device: str = MOCK,
) -> tuple[int, tp.Any]:
    """POST to the test server; returns (status, parsed body or None)"""
    body = dict(data or {})
    body["target_device"] = device
    if token is not None:
        body["lock_token"] = token
    req = urllib.request.Request(f"{SERVER}{path}", method="POST")
    req.add_header("Content-Type", "application/json")
    req.data = json.dumps(body).encode()
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read().decode()
            return response.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raw = e.read().decode() if e.fp else ""
        return e.code, json.loads(raw) if raw else None


def acquire(owner: str, ttl_s: float | None = None, token: str | None = None) -> tuple[int, tp.Any]:
    body: dict = {"owner": owner}
    if ttl_s is not None:
        body["ttl_s"] = ttl_s
    return post("/api/lock/acquire", body, token=token)


def lease(owner: str = "enforcement tests", ttl_s: float | None = None) -> str:
    status, body = acquire(owner, ttl_s=ttl_s)
    assert status == 200, (status, body)
    return body["token"]


def break_lease() -> None:
    status, _ = post("/api/lock/break", {"reason": "test cleanup"})
    assert status == 200


@pytest.fixture
def unlocked(server):
    """each test starts and ends with nobody holding the microscope"""
    break_lease()
    yield
    break_lease()


@pytest.fixture(scope="module")
def schema(server) -> dict:
    with urllib.request.urlopen(f"{SERVER}/openapi.json", timeout=10) as r:
        return json.loads(r.read().decode())


@pytest.fixture(scope="module")
def async_schema(server) -> dict:
    """the websocket contract; sockets are not describable in OpenAPI, so they live here"""
    with urllib.request.urlopen(f"{SERVER}/asyncapi.json", timeout=10) as r:
        return json.loads(r.read().decode())


@pytest.fixture(scope="module")
def gated_channels(async_schema) -> set[str]:
    return {
        channel["address"]
        for channel in async_schema["channels"].values()
        if channel.get("x-requires-device-lease")
    }


@pytest.fixture(scope="module")
def gated_paths(schema) -> set[str]:
    return {
        path
        for path, ops in schema["paths"].items()
        for op in ops.values()
        if op.get("x-requires-device-lease")
    }


@pytest.fixture(scope="module")
def api_paths(schema) -> set[str]:
    """every route that goes through route_wrapper (i.e. every POST under /api)"""
    return {path for path, ops in schema["paths"].items() if "post" in ops}


@pytest.fixture(scope="module")
def bodies(server) -> dict[str, dict]:
    """
    a schema-valid request body for every gated route

    Valid matters: an invalid body is rejected (422) before the lease is ever looked at, which
    would make every refusal test below pass for the wrong reason.
    TestSchemaContract::test_bodies_reach_the_gate is what rules that out.
    """
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
        "/api/action/stream_channel_begin": {"channel": channel, "framerate_hz": 1.0},
        "/api/action/stream_channel_end": {"channel": channel},
        "/api/action/turn_off_all_illumination": {},
    }


# --------------------------------------------------------------------------------------------


class TestSchemaContract:
    """what the server says about itself, held against what it must be"""

    def test_ungated_routes_are_the_documented_exceptions(self, api_paths, gated_paths):
        assert api_paths - gated_paths == UNGATED_ROUTES

    def test_every_other_route_is_gated(self, api_paths, gated_paths):
        assert gated_paths & api_paths == api_paths - UNGATED_ROUTES
        assert len(gated_paths & api_paths) >= 15  # the hardware surface, not a handful

    def test_no_gated_route_is_a_get(self, schema, gated_paths):
        """a GET carries no body, so it could never carry lock_token"""
        for path in gated_paths:
            methods = set(schema["paths"][path])
            assert "get" not in methods, path

    def test_the_hardware_websocket_is_gated(self, gated_channels):
        assert "/ws/action/snap_selected_channels_progressive" in gated_channels

    def test_the_gated_channel_documents_its_close_code(self, async_schema, gated_channels):
        """4423 is the websocket mirror of the 423 a POST would have got"""
        for channel in async_schema["channels"].values():
            if channel["address"] in gated_channels:
                assert "4423" in channel["x-close-codes"], channel["address"]

    def test_gated_routes_document_the_423(self, schema, gated_paths):
        for path in gated_paths:
            for op in schema["paths"][path].values():
                assert "423" in op["responses"], path

    def test_bodies_cover_every_gated_route(self, bodies, gated_paths, api_paths):
        assert set(bodies) == gated_paths & api_paths

    def test_bodies_reach_the_gate(self, bodies, unlocked):
        """
        every sample body is schema-valid, so a 423 below is the lease talking and not a
        validation error standing in for it

        The probe sends a target_device for a different microscope: that check sits between
        body validation and the lease check, so a valid body answers 421 without the request
        ever reaching the hardware.
        """
        for path, body in bodies.items():
            status, detail = post(path, body, device="some-other-microscope")
            assert status == 421, (path, status, detail)


class TestWithoutALease:
    """nobody holds the microscope: hardware is closed, everything else is open"""

    def test_every_gated_route_refuses(self, bodies, unlocked):
        for path, body in bodies.items():
            status, response = post(path, body)
            assert status == 423, (path, status, response)
            detail = response["detail"]
            assert detail["reason"] == LockReason.NOT_HELD, (path, detail)
            assert detail["held"] is False, (path, detail)
            assert "/api/lock/acquire" in detail["message"], (path, detail)

    def test_a_made_up_token_does_not_help(self, bodies, unlocked):
        for path, body in bodies.items():
            status, response = post(path, body, token="not-a-real-token")
            assert status == 423, (path, status, response)
            assert response["detail"]["reason"] == LockReason.NOT_HELD, path

    def test_reads_are_open(self, unlocked):
        for path, body in (
            ("/api/get_info/current_state", {}),
            ("/api/get_features/hardware_capabilities", {}),
            ("/api/get_features/machine_defaults", {}),
            ("/api/acquisition/config_list", {}),
            ("/api/lock/status", {}),
        ):
            status, response = post(path, body)
            assert status == 200, (path, status, response)

    def test_cancel_is_open(self, unlocked):
        """cancel protects the hardware, so it is never gated - even with nothing to cancel"""
        status, response = post("/api/action/cancel_current")
        assert status != 423, (status, response)
        status, response = post("/api/acquisition/cancel", {"acquisition_id": "nonexistent"})
        assert status != 423, (status, response)


class TestWithAForeignLease:
    """somebody else holds the microscope"""

    @pytest.fixture
    def other_client(self, unlocked) -> str:
        return lease(owner="alice")

    def test_every_gated_route_refuses_everyone_else(self, bodies, other_client):
        for path, body in bodies.items():
            for token in (None, "wrong-token"):
                status, response = post(path, body, token=token)
                assert status == 423, (path, token, status, response)
                detail = response["detail"]
                assert detail["reason"] == LockReason.HELD_BY_OTHER, (path, detail)
                assert detail["held"] is True and detail["owner"] == "alice", (path, detail)
                assert detail["expires_at"] is not None, (path, detail)

    def test_reads_stay_open(self, other_client):
        status, response = post("/api/get_info/current_state")
        assert status == 200, (status, response)
        assert response["lock"]["held"] is True
        assert response["lock"]["owner"] == "alice"

    def test_cancel_stays_open(self, other_client):
        status, response = post("/api/action/cancel_current")
        assert status != 423, (status, response)

    def test_acquire_and_release_are_refused(self, other_client):
        status, response = acquire("bob")
        assert status == 423 and response["detail"]["owner"] == "alice"
        status, response = post("/api/lock/release", token="bob's token")
        assert status == 423 and response["detail"]["reason"] == LockReason.HELD_BY_OTHER
        status, response = post("/api/lock/release")
        assert status == 423, (status, response)

    def test_status_tells_a_stranger_it_is_not_theirs(self, other_client):
        status, response = post("/api/lock/status")
        assert status == 200
        assert response["held"] is True and response["token_matches"] is False
        status, response = post("/api/lock/status", token=other_client)
        assert response["token_matches"] is True

    def test_break_hands_the_microscope_to_the_next_client(self, bodies, other_client):
        break_lease()
        # broken means free, not open: the hardware is gated again until someone acquires
        status, response = post("/api/action/move_to", bodies["/api/action/move_to"])
        assert status == 423 and response["detail"]["reason"] == LockReason.NOT_HELD
        # and alice's token is dead
        status, response = post(
            "/api/action/move_to", bodies["/api/action/move_to"], token=other_client
        )
        assert status == 423 and response["detail"]["reason"] == LockReason.NOT_HELD
        bob = lease(owner="bob")
        status, response = post("/api/action/move_to", bodies["/api/action/move_to"], token=bob)
        assert status == 200, (status, response)


class TestWithOwnLease:
    """the holder can drive the hardware, and stops being able to once it is given back"""

    @pytest.fixture
    def token(self, unlocked) -> str:
        return lease(owner="me")

    def test_holder_gets_through(self, bodies, token):
        for path in SAFE_TO_RUN:
            status, response = post(path, bodies[path], token=token)
            assert status == 200, (path, status, response)

    def test_gate_closes_again_after_release(self, bodies, token):
        status, response = post("/api/action/move_to", bodies["/api/action/move_to"], token=token)
        assert status == 200, (status, response)
        status, _ = post("/api/lock/release", token=token)
        assert status == 200
        status, response = post("/api/action/move_to", bodies["/api/action/move_to"], token=token)
        assert status == 423 and response["detail"]["reason"] == LockReason.NOT_HELD

    def test_state_broadcast_carries_the_lease_without_the_token(self, token):
        status, response = post("/api/get_info/current_state")
        assert status == 200
        lock = response["lock"]
        assert lock["held"] is True and lock["owner"] == "me"
        assert lock["expires_at"] is not None
        assert token not in json.dumps(response)

    def test_reacquiring_with_own_token_renews_instead_of_refusing(self, token):
        status, renewed = acquire("me, renamed", token=token)
        assert status == 200, (status, renewed)
        assert renewed["token"] == token
        assert renewed["owner"] == "me, renamed"


class TestLeaseLifetimeOverHttp:
    """the TTL, seen through the API rather than through the object"""

    def test_lease_lapses_and_leaves_the_hardware_gated(self, bodies, unlocked):
        token = lease(owner="alice", ttl_s=5)  # 5s is DeviceLock.MIN_TTL_S
        status, _ = post("/api/action/move_to", bodies["/api/action/move_to"], token=token)
        assert status == 200
        time.sleep(6)
        status, response = post("/api/lock/status")
        assert status == 200 and response["held"] is False
        # the lapse frees the lease for the next client; it does not unlock the hardware
        status, response = post("/api/action/move_to", bodies["/api/action/move_to"], token=token)
        assert status == 423 and response["detail"]["reason"] == LockReason.NOT_HELD
        assert acquire("bob")[0] == 200

    def test_use_keeps_the_lease_alive(self, bodies, unlocked):
        token = lease(owner="alice", ttl_s=5)
        for _ in range(3):
            time.sleep(2)
            status, _ = post("/api/lock/status", token=token)  # a read renews it too
            assert status == 200
        # 6s after acquire, and still alice's
        status, response = post("/api/lock/status")
        assert response["held"] is True and response["owner"] == "alice"
        status, response = post("/api/action/move_to", bodies["/api/action/move_to"], token=token)
        assert status == 200, (status, response)


# --------------------------------------------------------------------------------------------
# websocket


def ws_handshake(token: str | None) -> tuple[dict | None, int | None]:
    """
    open the progressive-snap socket, send its init message, return (first reply, close code)

    The reply is None when the server sent nothing before closing. The close code is None while
    the socket is still open.
    """
    init: dict = {"target_device": MOCK}
    if token is not None:
        init["lock_token"] = token
    with ws_connect(f"{WS_SERVER}/ws/action/snap_selected_channels_progressive") as ws:
        ws.send(json.dumps(init))
        try:
            reply = json.loads(ws.recv(timeout=10))
        except ConnectionClosed as e:
            return None, e.rcvd.code if e.rcvd else None
        try:
            ws.recv(timeout=2)
        except ConnectionClosed as e:
            return reply, e.rcvd.code if e.rcvd else None
        except TimeoutError:
            return reply, None
    return reply, None


class TestHardwareWebsocket:
    """the progressive snap socket drives the hardware, so it needs the lease too"""

    def test_refused_without_a_lease(self, unlocked):
        reply, code = ws_handshake(None)
        assert reply is not None and reply["error"] == "device_locked"
        assert reply["reason"] == LockReason.NOT_HELD
        assert code == 4423

    def test_refused_for_a_foreign_lease(self, unlocked):
        lease(owner="alice")
        reply, code = ws_handshake("wrong-token")
        assert reply is not None and reply["error"] == "device_locked"
        assert reply["reason"] == LockReason.HELD_BY_OTHER and reply["owner"] == "alice"
        assert code == 4423

    def test_holder_gets_past_the_gate(self, unlocked):
        """the holder is let through; the socket then waits for its config, so nothing runs"""
        token = lease(owner="me")
        init = {"target_device": MOCK, "lock_token": token}
        with ws_connect(f"{WS_SERVER}/ws/action/snap_selected_channels_progressive") as ws:
            ws.send(json.dumps(init))
            try:
                reply = json.loads(ws.recv(timeout=2))
            except TimeoutError:
                reply = None  # nothing to say: it is waiting for the config, i.e. past the gate
            except ConnectionClosed as e:
                pytest.fail(f"holder was closed out with {e.rcvd.code if e.rcvd else '?'}")
            assert reply is None or reply.get("error") != "device_locked", reply


class TestLeaseUnderConcurrentClients:
    """
    The lease decides who owns the microscope, so two clients believing they own it is the
    worst outcome the whole mechanism can have. These hammer the decision points - several
    clients acquiring at once, breaking and taking over at once, a released token still in
    use - and check the one thing that must always hold: at most one holder, and only that
    holder's token works.
    """

    THREADS = 8

    def test_simultaneous_acquires_produce_exactly_one_holder(self, unlocked, bodies):
        """eight clients, one instant, one lease"""
        results: list[tuple[int, tp.Any]] = []
        barrier = threading.Barrier(self.THREADS)

        def contend(n: int):
            barrier.wait()
            results.append(acquire(f"client-{n}"))

        threads = [threading.Thread(target=contend, args=(n,)) for n in range(self.THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        won = [body for status, body in results if status == 200]
        lost = [(status, body) for status, body in results if status != 200]
        assert len(won) == 1, f"{len(won)} clients were given the lease at once: {won}"
        assert all(status == 423 for status, _ in lost), lost
        assert all(body["detail"]["reason"] == LockReason.HELD_BY_OTHER for _, body in lost)

        # the winner drives the hardware; every loser is refused, naming the winner
        token = won[0]["token"]
        move = bodies["/api/action/move_to"]
        assert post("/api/action/move_to", move, token=token)[0] == 200
        for _, body in lost:
            assert body["detail"]["owner"] == won[0]["owner"]
        status, response = post("/api/action/move_to", move)
        assert status == 423 and response["detail"]["owner"] == won[0]["owner"]

    def test_break_and_acquire_storm_leaves_one_consistent_holder(self, unlocked, bodies):
        """
        Four clients taking the microscope from each other as fast as they can.

        Breaking is open to anyone, so this is the messiest legal sequence there is. However
        it interleaves, the lease must never report itself half-held, and when the dust
        settles exactly one client may drive the hardware.
        """
        rounds = 25
        owners: list[str] = []
        contradictions: list[dict] = []

        def churn(n: int):
            for _ in range(rounds):
                post("/api/lock/break", {"reason": f"client-{n} taking over"})
                status, body = acquire(f"client-{n}")
                if status == 200:
                    owners.append(body["owner"])
                state = post("/api/lock/status")[1]
                if state["held"] != (state["owner"] is not None) or state["held"] != (
                    state["lock_id"] is not None
                ):
                    contradictions.append(state)

        threads = [threading.Thread(target=churn, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not contradictions, f"the lease reported itself half-held: {contradictions[:3]}"
        assert owners, "nobody ever got the lease"

        # settle it: whoever holds it now is the only one who can drive
        state = post("/api/lock/status")[1]
        assert state["held"] is True and state["owner"] is not None
        assert post("/api/action/move_to", bodies["/api/action/move_to"])[0] == 423

        break_lease()
        final = lease(owner="the last one standing")
        assert post("/api/action/move_to", bodies["/api/action/move_to"], token=final)[0] == 200

    def test_a_released_token_is_dead_even_when_reused_at_once(self, unlocked, bodies):
        """
        A token that has been given back must not work again, including from requests that
        were already on their way out when it was released.
        """
        token = lease(owner="alice")
        assert post("/api/lock/release", token=token)[0] == 200

        refusals: list[tuple[int, tp.Any]] = []
        barrier = threading.Barrier(self.THREADS)
        move = bodies["/api/action/move_to"]

        def reuse():
            barrier.wait()
            refusals.append(post("/api/action/move_to", move, token=token))

        threads = [threading.Thread(target=reuse) for _ in range(self.THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(status == 423 for status, _ in refusals), refusals
        assert all(body["detail"]["reason"] == LockReason.NOT_HELD for _, body in refusals)

    def test_the_holder_keeps_the_lease_while_everyone_else_pushes(self, unlocked, bodies):
        """
        Being hammered by other clients must not cost the holder its lease.

        Every refused request goes through the same check as the holder's, so a check that
        renewed - or dropped - the lease for the wrong client would show up here.
        """
        token = lease(owner="alice", ttl_s=5)
        stop = threading.Event()

        def outsiders():
            while not stop.is_set():
                post("/api/action/move_to", bodies["/api/action/move_to"])
                acquire("gatecrasher")
                post("/api/lock/status")

        pushers = [threading.Thread(target=outsiders) for _ in range(3)]
        for t in pushers:
            t.start()
        try:
            deadline = time.time() + 7  # longer than the 5s TTL: alice's own use must renew it
            while time.time() < deadline:
                status, response = post("/api/action/move_to", bodies["/api/action/move_to"], token=token)
                assert status == 200, f"the holder lost its own lease: {status} {response}"
        finally:
            stop.set()
            for t in pushers:
                t.join()

        state = post("/api/lock/status", token=token)[1]
        assert state["held"] is True and state["owner"] == "alice" and state["token_matches"] is True
