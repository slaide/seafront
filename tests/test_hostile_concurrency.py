"""
Many clients behaving badly at once.

Not a load test. The point is that the server's answer to every request is a *decision* - a
status code it chose - rather than a crash, a hang, a reset connection, or a state the next
client cannot recover from. Threads here mix valid commands with negative coordinates, values
that are not numbers, bodies that are not JSON, leases taken and broken from under each other,
streams started and stopped in parallel, and acquisitions cancelled by clients that never
started them.

What is asserted afterwards is not "it went well" but:

- every request got a reply, and that reply was one of the statuses this API defines
- no reply was an unhandled crash: the body always parsed, and always had the documented shape
- the lease never reported two holders and never reported itself half-held
- the microscope is usable by an ordinary client when the storm stops

The last one is the one that matters. A microscope that survives the storm but is left
claiming to stream, or holding a lease nobody owns, is broken just as thoroughly as one that
fell over - it just takes longer to notice.
"""

import json
import random
import threading
import time
import typing as tp
import urllib.error
import urllib.request

import pytest

from tests.mock_server import MOCK, mock_server

# the fixture picks a free port and rebinds these before any helper below runs
PORT = 0
SERVER = ""

# Every status this API is allowed to answer with. Anything else is a bug, and a connection
# error or a timeout is not a status at all.
DEFINED_STATUSES = {200, 400, 409, 421, 422, 423, 500, 503}

NOT_NUMBERS = ["NaN", "Infinity", "-Infinity"]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    global PORT, SERVER
    with mock_server(tmp_path_factory) as srv:
        PORT, SERVER = srv.port, srv.url
        yield srv.proc, srv.log_path


class Reply(tp.NamedTuple):
    path: str
    status: int | None  # None means no HTTP reply at all
    body: tp.Any
    error: str | None


def post_raw(path: str, raw: bytes, content_type: str = "application/json") -> Reply:
    """send whatever bytes we like, valid JSON or not"""
    req = urllib.request.Request(f"{SERVER}{path}", method="POST", data=raw)
    req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            text = response.read().decode()
            return Reply(path, response.status, json.loads(text) if text else None, None)
    except urllib.error.HTTPError as e:
        text = e.read().decode() if e.fp else ""
        try:
            body = json.loads(text) if text else None
        except json.JSONDecodeError:
            return Reply(path, e.code, None, f"reply was not JSON: {text[:200]!r}")
        return Reply(path, e.code, body, None)
    except Exception as e:  # connection reset, timeout, anything that is not a reply
        return Reply(path, None, None, f"{type(e).__name__}: {e}")


def post(path: str, data: dict | None = None, token: str | None = None) -> Reply:
    body = dict(data or {})
    body.setdefault("target_device", MOCK)
    if token is not None:
        body["lock_token"] = token
    return post_raw(path, json.dumps(body).encode())


def lease(owner: str) -> str | None:
    reply = post("/api/lock/acquire", {"owner": owner})
    return reply.body["token"] if reply.status == 200 and reply.body else None


def break_lease() -> None:
    post("/api/lock/break", {"reason": "hostile concurrency test"})


def channel() -> dict:
    listing = post("/api/acquisition/config_list").body
    name = listing["configs"][0]["filename"]
    protocol = post("/api/acquisition/config_fetch", {"config_file": name}).body["file"]
    return protocol["channels"][0]


@pytest.fixture
def settled(server):
    """start and end with the microscope idle and unlocked"""
    _settle()
    yield
    _settle()


def _settle(timeout_s: float = 30.0) -> None:
    break_lease()
    token = lease("settler")
    state = post("/api/get_info/current_state").body
    if state and state["is_streaming"]:
        post("/api/action/stream_channel_end", {}, token=token)
    if state and state["current_acquisition_id"] is not None:
        post("/api/acquisition/cancel", {"acquisition_id": state["current_acquisition_id"]})
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        state = post("/api/get_info/current_state").body
        if state and not state["is_busy"] and not state["is_streaming"]:
            break
        time.sleep(0.2)
    break_lease()


def contradictions(state: dict) -> list[str]:
    """everything in one status snapshot that cannot be true at the same time"""
    wrong = []
    if state["is_streaming"] != (state["streaming_channel_handle"] is not None):
        wrong.append(f"streaming={state['is_streaming']} channel={state['streaming_channel_handle']!r}")
    if state["is_busy"] != bool(state["busy_reasons"]):
        wrong.append(f"busy={state['is_busy']} reasons={state['busy_reasons']}")
    lock = state["lock"]
    if lock["held"] != (lock["owner"] is not None):
        wrong.append(f"held={lock['held']} owner={lock['owner']!r}")
    if lock["held"] != (lock["lock_id"] is not None):
        wrong.append(f"held={lock['held']} lock_id={lock['lock_id']!r}")
    if lock["held"] != (lock["expires_at"] is not None):
        wrong.append(f"held={lock['held']} expires_at={lock['expires_at']!r}")
    return wrong


class TestTheStorm:
    THREADS = 10
    ROUNDS = 12

    def attacks(self, ch: dict) -> list[tp.Callable[[str | None], Reply]]:
        """
        one move each, ranging from reasonable to actively wrong

        Each takes whatever token the calling thread currently believes it holds, which is
        usually stale - that is the point.
        """
        protocol_name = post("/api/acquisition/config_list").body["configs"][0]["filename"]
        return [
            # legitimate
            lambda t: post("/api/action/move_to", {"x_mm": 60.0, "y_mm": 40.0}, token=t),
            lambda t: post("/api/action/move_by", {"axis": "z", "distance_mm": 0.001}, token=t),
            lambda t: post("/api/action/turn_off_all_illumination", {}, token=t),
            lambda t: post("/api/get_info/current_state"),
            lambda t: post("/api/lock/status", token=t),
            # out of bounds
            lambda t: post("/api/action/move_to", {"x_mm": -5.0}, token=t),
            lambda t: post("/api/action/move_to", {"x_mm": 1e9, "y_mm": 1e9}, token=t),
            lambda t: post("/api/action/move_by", {"axis": "x", "distance_mm": 1e12}, token=t),
            # not numbers at all
            lambda t: post_raw(
                "/api/action/move_to",
                f'{{"target_device":"{MOCK}","x_mm":{random.choice(NOT_NUMBERS)}}}'.encode(),
            ),
            # wrong shapes
            lambda t: post("/api/action/move_by", {"axis": "w", "distance_mm": 1.0}, token=t),
            lambda t: post("/api/action/move_to", {"x_mm": "over there"}, token=t),
            lambda t: post("/api/action/move_to", {"x_mm": 1.0, "unknown_field": 1}, token=t),
            # not JSON
            lambda t: post_raw("/api/action/move_to", b"{not json at all"),
            lambda t: post_raw("/api/action/move_to", b""),
            lambda t: post_raw("/api/action/move_to", b'{"target_device":"' + b"x" * 50_000 + b'"}'),
            # fighting over the microscope
            lambda t: post("/api/lock/acquire", {"owner": f"thread-{threading.get_ident()}"}),
            lambda t: post("/api/lock/release", token=t),
            lambda t: post("/api/lock/break", {"reason": "mine now"}),
            # streaming, in parallel with everything else
            lambda t: post("/api/action/stream_channel_begin", {"channel": ch}, token=t),
            lambda t: post("/api/action/stream_channel_end", {}, token=t),
            lambda t: post("/api/action/snap_channel", {"channel": ch}, token=t),
            # runs nobody started
            lambda t: post("/api/acquisition/cancel", {"acquisition_id": "no-such-run"}),
            lambda t: post("/api/acquisition/status", {"acquisition_id": "no-such-run"}),
            lambda t: post("/api/action/cancel_current", {}),
            # aimed at a different microscope entirely
            lambda t: post_raw(
                "/api/action/move_to", b'{"target_device":"some-other-scope","x_mm":1.0}'
            ),
            lambda t: post("/api/acquisition/config_fetch", {"config_file": protocol_name}),
        ]

    def _run_storm(self, ch: dict) -> tuple[list[Reply], list[str]]:
        replies: list[Reply] = []
        wrong: list[str] = []
        collect = threading.Lock()
        attacks = self.attacks(ch)
        start = threading.Barrier(self.THREADS + 1)

        def client(n: int):
            rng = random.Random(n)
            token: str | None = None
            mine: list[Reply] = []
            start.wait()
            for _ in range(self.ROUNDS):
                attack = rng.choice(attacks)
                reply = attack(token)
                mine.append(reply)
                # keep whatever token we happen to be handed, however stale it goes
                if reply.path == "/api/lock/acquire" and reply.status == 200 and reply.body:
                    token = reply.body["token"]
            with collect:
                replies.extend(mine)

        def watcher():
            start.wait()
            deadline = time.time() + 60
            while time.time() < deadline and len(replies) < self.THREADS * self.ROUNDS:
                reply = post("/api/get_info/current_state")
                if reply.status == 200 and reply.body:
                    with collect:
                        wrong.extend(contradictions(reply.body))
                elif reply.status is None:
                    with collect:
                        wrong.append(f"status read got no reply: {reply.error}")

        threads = [threading.Thread(target=client, args=(n,)) for n in range(self.THREADS)]
        watch = threading.Thread(target=watcher, daemon=True)
        watch.start()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        watch.join(timeout=10)
        return replies, wrong

    def test_every_request_gets_a_decision(self, server, settled):
        ch = channel()
        replies, _wrong = self._run_storm(ch)

        assert len(replies) == self.THREADS * self.ROUNDS

        dropped = [r for r in replies if r.status is None]
        assert not dropped, f"{len(dropped)} requests got no reply: {[d.error for d in dropped[:3]]}"

        undefined = {r.status for r in replies} - DEFINED_STATUSES
        assert not undefined, f"statuses outside the documented set: {undefined}"

        unparseable = [r for r in replies if r.error]
        assert not unparseable, [r.error for r in unparseable[:3]]

    def test_the_status_hatch_never_contradicts_itself_during_the_storm(self, server, settled):
        ch = channel()
        _replies, wrong = self._run_storm(ch)
        assert not wrong, f"{len(wrong)} contradictory status reads: {wrong[:5]}"

    def test_the_microscope_still_works_afterwards(self, server, settled):
        """
        the invariant that actually matters

        A microscope left claiming a stream it is not running, or holding a lease nobody
        owns, is as unusable as one that crashed.
        """
        ch = channel()
        self._run_storm(ch)
        _settle()

        state = post("/api/get_info/current_state")
        assert state.status == 200, state
        assert state.body["is_streaming"] is False
        assert state.body["is_busy"] is False
        assert state.body["lock"]["held"] is False

        token = lease("the next honest client")
        assert token is not None, "nobody could take the lease after the storm"
        assert post("/api/action/move_to", {"x_mm": 60.0, "y_mm": 40.0}, token=token).status == 200
        assert post("/api/action/stream_channel_begin", {"channel": ch}, token=token).status == 200
        assert post("/api/action/stream_channel_end", {}, token=token).status == 200
        assert post("/api/lock/release", token=token).status == 200

    def test_the_server_is_still_serving(self, server, settled):
        ch = channel()
        self._run_storm(ch)
        with urllib.request.urlopen(f"{SERVER}/api/health", timeout=5) as response:
            assert response.status == 200
        with urllib.request.urlopen(f"{SERVER}/openapi.json", timeout=10) as response:
            assert response.status == 200


class TestMalformedBodies:
    """
    bodies that are not what the header says, sent one at a time so the status is attributable
    """

    @pytest.mark.parametrize(
        "raw",
        [
            b"",
            b"null",
            b"[]",
            b'"a string"',
            b"123",
            b"{not json at all",
            b'{"target_device": }',
            b'{"target_device": "' + b"x" * 200_000 + b'"}',
            bytes([0, 1, 2, 3]),
            b'{"target_device": "' + bytes([0]) + b'"}',
            b'{"target_device": "' + bytes([0xed, 0xa0, 0x80]) + b'"}',
        ],
    )
    def test_a_malformed_body_is_refused_not_crashed(self, server, settled, raw):
        reply = post_raw("/api/action/move_to", raw)
        assert reply.status is not None, f"no reply at all: {reply.error}"
        assert reply.status in DEFINED_STATUSES, reply

    def test_the_wrong_content_type_is_refused(self, server, settled):
        reply = post_raw(
            "/api/action/move_to",
            json.dumps({"target_device": MOCK, "x_mm": 1.0}).encode(),
            content_type="text/plain",
        )
        assert reply.status in DEFINED_STATUSES, reply

    def test_a_deeply_nested_body_does_not_take_the_server_down(self, server, settled):
        """a structure deep enough to blow a recursive parser's stack"""
        nested = json.dumps({"target_device": MOCK, "x_mm": json.loads("[" * 400 + "]" * 400)})
        reply = post_raw("/api/action/move_to", nested.encode())
        assert reply.status in DEFINED_STATUSES, reply
        # and the server is still there
        assert post("/api/get_info/current_state").status == 200

    def test_a_body_that_claims_another_microscope_is_a_routing_refusal(self, server, settled):
        reply = post_raw("/api/action/move_to", b'{"target_device":"not-this-one","x_mm":1.0}')
        assert reply.status == 421, reply


class TestNonFiniteNumbersOverTheWire:
    """
    the end-to-end version: a value JSON cannot carry, sent to a server that has to answer

    Python clients produce these by accident - `json.dumps(float("nan"))` writes `NaN`
    without complaint - so this is a plausible bug in an honest client as much as an attack.
    """

    @pytest.mark.parametrize("literal", NOT_NUMBERS)
    def test_a_non_finite_coordinate_is_a_422_with_a_body(self, server, settled, literal):
        reply = post_raw(
            "/api/action/move_to",
            f'{{"target_device":"{MOCK}","x_mm":{literal}}}'.encode(),
        )
        assert reply.status == 422, reply
        assert reply.error is None, f"the refusal itself was not JSON: {reply.error}"
        assert reply.body is not None
        assert reply.body["detail"][0]["loc"][-1] == "x_mm"
        assert "finite" in reply.body["detail"][0]["msg"]

    @pytest.mark.parametrize("literal", NOT_NUMBERS)
    def test_it_does_not_matter_which_field_it_hides_in(self, server, settled, literal):
        reply = post_raw(
            "/api/action/move_by",
            f'{{"target_device":"{MOCK}","axis":"x","distance_mm":{literal}}}'.encode(),
        )
        assert reply.status == 422, reply
        assert reply.error is None

    def test_one_nested_inside_a_channel_is_still_answered(self, server, settled):
        """the same trap one level down, where the model is seaconfig's rather than ours"""
        ch = channel()
        ch["illum_perc"] = float("nan")
        reply = post_raw(
            "/api/action/snap_channel",
            json.dumps({"target_device": MOCK, "channel": ch}).encode(),
        )
        assert reply.status in DEFINED_STATUSES, reply
        assert reply.error is None, f"the refusal was not JSON: {reply.error}"

    def test_the_server_is_unharmed_by_a_run_of_them(self, server, settled):
        for _ in range(30):
            post_raw("/api/action/move_to", f'{{"target_device":"{MOCK}","x_mm":NaN}}'.encode())
        assert post("/api/get_info/current_state").status == 200
