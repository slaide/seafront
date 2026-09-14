"""
The socket gate: what `socketspec.socket()` does before a handler runs.

tests/test_lock_enforcement.py checks the sockets this server actually serves. These check the
mechanism underneath, with sockets invented here, because the property worth pinning is not
"the progressive snap socket happens to refuse" - it is that *declaring* `requires_lease`
is what refuses, so a socket added later cannot advertise 4423 in asyncapi.json and then let
everyone through. The handlers below record whether they ran; a handler that ran is the gate
having failed, whatever the socket sent back.
"""

import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from seafront.server.api import socketspec
from seafront.server.api.socketspec import Frame, socket
from seafront.server.lock import DeviceLock

DEVICE = "test-scope"


class StubCore:
    """what the gate reads off a Core: the device's name, and the lease"""

    def __init__(self, in_use: bool = False) -> None:
        self.microscope_name = DEVICE
        self.lock = DeviceLock(device_in_use=lambda: in_use)


@pytest.fixture
def sockets():
    """
    an app with one gated socket and one open one, plus the record of what ran

    `SOCKETS` is module-global (asyncapi.py reads it), so the declarations these tests make
    are taken back out again afterwards.
    """
    before = list(socketspec.SOCKETS)
    app = FastAPI()
    app.state.core = StubCore()
    ran: list[str] = []

    frames = {"receive": [Frame("x", "anything", schema={})], "send": [Frame("y", "anything", schema={})]}

    @socket(app.router, "/ws/test/gated", requires_lease=True, **frames)
    async def gated(ws: WebSocket, core, opening):
        """a socket that drives the hardware"""
        ran.append("gated")
        await ws.send_json({"ran": True, "saw_token": opening.lock_token is not None})

    @socket(app.router, "/ws/test/open", **frames)
    async def open_socket(ws: WebSocket, core, opening):
        """a socket that only reads"""
        ran.append("open")
        await ws.send_json({"ran": True})

    yield TestClient(app), app.state.core, ran

    socketspec.SOCKETS[:] = before


# TestClient's websocket receive blocks on a queue with no timeout of its own, so a server
# that neither answers nor closes hangs the test run forever rather than failing it. Every
# receive below goes through this, which is the difference between "the gate broke" arriving
# as a red test and arriving as a suite that never finishes.
RECEIVE_TIMEOUT_S = 10.0


def _receive(ws, executor: ThreadPoolExecutor):
    """one JSON frame, or the close that came instead; TimeoutError if neither arrives"""
    return executor.submit(ws.receive_json).result(timeout=RECEIVE_TIMEOUT_S)


def handshake(client, path: str, body: dict) -> tuple[dict | None, int | None]:
    """open `path`, send `body`, return (first reply, close code)"""
    with ThreadPoolExecutor(max_workers=1) as executor, client.websocket_connect(path) as ws:
        ws.send_text(json.dumps(body))
        try:
            return _receive(ws, executor), None
        except WebSocketDisconnect as closed:
            return None, closed.code
        except FutureTimeout:
            pytest.fail(f"{path}: the server neither replied nor closed within {RECEIVE_TIMEOUT_S}s")


def reply_then_close(client, path: str, body: dict | str) -> tuple[dict, int | None]:
    """
    the gate sends its refusal *then* closes; this collects both

    `body` is the handshake to send, or a raw string for the malformed case.
    """
    with ThreadPoolExecutor(max_workers=1) as executor, client.websocket_connect(path) as ws:
        ws.send_text(body if isinstance(body, str) else json.dumps(body))
        try:
            first = _receive(ws, executor)
        except FutureTimeout:
            pytest.fail(f"{path}: no reply within {RECEIVE_TIMEOUT_S}s")
        try:
            _receive(ws, executor)
            return first, None
        except WebSocketDisconnect as closed:
            return first, closed.code
        except FutureTimeout:
            pytest.fail(
                f"{path}: replied {first!r} but never closed within {RECEIVE_TIMEOUT_S}s - "
                "a gate that refuses must also close the socket"
            )


class TestTheGateRunsBeforeTheHandler:
    def test_a_gated_socket_refuses_without_a_lease_and_the_handler_never_runs(self, sockets):
        client, _core, ran = sockets
        reply, code = reply_then_close(client, "/ws/test/gated", {"target_device": DEVICE})
        assert code == 4423
        assert reply["error"] == "device_locked" and reply["reason"] == "not_held"
        assert ran == [], "the handler ran on a socket that had no lease"

    def test_a_gated_socket_refuses_a_foreign_lease(self, sockets):
        client, core, ran = sockets
        core.lock.acquire("alice")
        reply, code = reply_then_close(
            client, "/ws/test/gated", {"target_device": DEVICE, "lock_token": "not-alice's"}
        )
        assert code == 4423
        assert reply["reason"] == "held_by_other" and reply["owner"] == "alice"
        assert ran == []

    def test_the_holder_reaches_the_handler(self, sockets):
        client, core, ran = sockets
        lease = core.lock.acquire("alice")
        reply, _ = handshake(
            client, "/ws/test/gated", {"target_device": DEVICE, "lock_token": lease.token}
        )
        assert reply == {"ran": True, "saw_token": True}
        assert ran == ["gated"], "the holder was not let through"

    def test_an_open_socket_needs_no_lease(self, sockets):
        client, _core, ran = sockets
        reply, _ = handshake(client, "/ws/test/open", {"target_device": DEVICE})
        assert reply == {"ran": True}
        assert ran == ["open"]

    def test_an_open_socket_is_not_refused_by_somebody_else_s_lease(self, sockets):
        """reads stay open: watching a run you do not own is the point of them"""
        client, core, ran = sockets
        core.lock.acquire("alice")
        reply, _ = handshake(client, "/ws/test/open", {"target_device": DEVICE})
        assert reply == {"ran": True}
        assert ran == ["open"]


class TestTheGateChecksRoutingAndShape:
    def test_a_misdirected_socket_is_closed_before_the_handler(self, sockets):
        client, _core, ran = sockets
        reply, code = reply_then_close(client, "/ws/test/open", {"target_device": "another-scope"})
        assert code == 4421
        assert reply["error"] == "target_device_mismatch"
        assert ran == []

    def test_an_empty_target_device_is_the_documented_bypass(self, sockets):
        """a client that does not know the device name yet may still connect"""
        client, _core, ran = sockets
        reply, _ = handshake(client, "/ws/test/open", {})
        assert reply == {"ran": True}
        assert ran == ["open"]

    def test_a_first_message_that_is_not_a_handshake_is_closed(self, sockets):
        client, _core, ran = sockets
        first, code = reply_then_close(client, "/ws/test/open", "not an object")
        assert code == 4400
        assert "handshake" in first["message"]
        assert ran == []

    def test_routing_is_checked_before_the_lease(self, sockets):
        """
        a socket aimed at the wrong microscope is 4421, not 4423

        Same order as the HTTP gate. A client that has its device name wrong should be told
        that, rather than being sent off to acquire a lease on a server it did not mean.
        """
        client, _core, ran = sockets
        _reply, code = reply_then_close(client, "/ws/test/gated", {"target_device": "another-scope"})
        assert code == 4421
        assert ran == []


class TestTheDeclarationIsWhatEnforces:
    def test_the_gated_socket_is_published_as_gated(self, sockets):
        declared = {spec.path: spec.requires_lease for spec in socketspec.SOCKETS}
        assert declared["/ws/test/gated"] is True
        assert declared["/ws/test/open"] is False

    def test_declaring_a_lease_is_the_only_thing_the_two_sockets_differ_by(self, sockets):
        """
        The handlers above are the same code. One refuses an unleased client and one does not,
        and the single difference between them is `requires_lease=True` on the declaration -
        which is the property that makes the published contract trustworthy.
        """
        client, _core, ran = sockets
        _reply, code = reply_then_close(client, "/ws/test/gated", {"target_device": DEVICE})
        assert code == 4423 and ran == []

        reply, _ = handshake(client, "/ws/test/open", {"target_device": DEVICE})
        assert reply == {"ran": True} and ran == ["open"]
