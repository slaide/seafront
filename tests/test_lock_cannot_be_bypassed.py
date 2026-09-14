"""
Ways the device lease could be got around, closed off one at a time.

The existing sweeps check that the server enforces the policy it *publishes*. That leaves the
question the policy cannot answer on its own: is the published policy true of what the handler
does? A route declared `READ` that quietly submits work to the worker is gated by nothing,
passes every sweep, and is invisible in the schema - it says it is a read and it moves the
stage.

Three separate things are checked here, because the lease can be got around in three places:

- **the server**, by a handler doing more than its policy admits to, or by a route registered
  outside the policy-carrying router altogether
- **the browser**, by a call that does not go through the client that attaches the token -
  one raw `fetch`, one new file, one websocket opened without a handshake
- **the offline mock client**, which has its own idea of which endpoints need a lease and can
  drift from the server's

All of it is derived: the route table comes from the app, the gated set from the published
schema, the hardware-driving Core methods from Core's own source. Nothing here is a list of
endpoints someone has to remember to update.
"""

import ast
import inspect
import re
import textwrap
from pathlib import Path
from typing import ClassVar

import pytest

from seafront.server.app import make_app
from seafront.server.core import Core

WEB_STATIC = Path(__file__).resolve().parent.parent / "seafront" / "web-static"


def strip_comments(source: str) -> str:
    """
    blank out // and /* */ comments, keeping line numbers intact

    A JSDoc line naming an endpoint is documentation, not a call, and flagging it would make
    the sweep something people route around rather than read.
    """
    source = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group()), source, flags=re.S)
    source = re.sub(r"(?m)//[^\n]*", lambda m: " " * len(m.group()), source)
    return re.sub(r"(?s)<!--.*?-->", lambda m: re.sub(r"[^\n]", " ", m.group()), source)

# The one ungated route that is allowed to reach the hardware, and why. Adding to this is a
# deliberate act: everything in it is a route any client can use to drive the microscope
# without holding it.
UNGATED_BUT_REACHES_HARDWARE = {
    # "are the devices there?", and connect if not. The connect that moves the stage is the
    # server's own at startup, so by the time a client can call this it is a presence check.
    "/api/action/establish_hardware_connection",
}

# Endpoints the browser is allowed to call without the token-attaching client.
FRONTEND_RAW_FETCH_ALLOWED = {
    "/api/health",  # liveness probe, used to decide whether the server is there at all
}


@pytest.fixture(scope="module")
def app():
    return make_app(None)


@pytest.fixture(scope="module")
def routes(app):
    """every POST route, with its published policy and its unwrapped handler"""
    found = []
    for route in app.routes:
        if "POST" not in getattr(route, "methods", set()):
            continue
        extra = getattr(route, "openapi_extra", None) or {}
        found.append(
            {
                "path": route.path,
                "gated": extra.get("x-requires-device-lease"),
                "handler": inspect.unwrap(route.endpoint),
            }
        )
    assert found, "no POST routes found; the introspection is broken, not the server"
    return found


def is_ours(handler) -> bool:
    """whether this callable was written in this codebase rather than by fastapi/starlette"""
    module = getattr(handler, "__module__", "") or ""
    return module.startswith("seafront")


@pytest.fixture(scope="module")
def hardware_methods() -> set[str]:
    """
    the Core methods that put work on the worker, read off Core itself

    Derived rather than listed: a new way to reach the hardware is covered the day it is
    written, without anybody remembering to add it here.
    """
    tree = ast.parse(inspect.getsource(Core))
    class_def = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    reaching = set()
    for node in class_def.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        source = ast.unparse(node)
        if "self.worker.submit" in source or "self.microscope.execute" in source:
            reaching.add(node.name)
    assert "run" in reaching, "Core.run no longer submits to the worker; this check is stale"
    return reaching


def core_calls(handler) -> set[str]:
    """
    the names of `core.<something>(...)` calls made directly in a handler

    Empty for anything whose source cannot be read or parsed - a closure defined inside
    another function, or a callable that came from a library rather than from this codebase.
    Those are excluded from the sweeps by `is_ours` rather than silently passing here.
    """
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(handler)))
    except (OSError, TypeError, SyntaxError, IndentationError):
        return set()
    called = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id == "core":
                called.add(func.attr)
    return called


class TestNoUngatedRouteReachesTheHardware:
    """
    the gap the published policy cannot see

    `admit` enforces whatever the route declared. It has no opinion about whether the
    declaration is honest, so a handler marked READ that calls `core.run` drives the stage
    for anybody who asks.
    """

    def test_every_ungated_handler_is_checked(self, routes):
        ungated = [route for route in routes if route["gated"] is False]
        assert len(ungated) >= 10, f"only {len(ungated)} ungated routes; introspection looks wrong"

    def test_no_ungated_handler_submits_work_to_the_worker(self, routes, hardware_methods):
        offenders = []
        for route in routes:
            if route["gated"] is not False:
                continue
            reaching = core_calls(route["handler"]) & hardware_methods
            if reaching and route["path"] not in UNGATED_BUT_REACHES_HARDWARE:
                offenders.append(f"{route['path']} calls core.{'/'.join(sorted(reaching))}")
        assert not offenders, (
            "these routes need no device lease but drive the hardware:\n  " + "\n  ".join(offenders)
        )

    def test_the_documented_exception_is_still_the_only_one(self, routes, hardware_methods):
        """
        and that it still is one

        If `establish_hardware_connection` stops reaching the hardware, the exception is
        stale and should go - a permanently-empty exception list is how a real one sneaks
        back in unnoticed.
        """
        actually_reaching = {
            route["path"]
            for route in routes
            if route["gated"] is False and core_calls(route["handler"]) & hardware_methods
        }
        assert actually_reaching == UNGATED_BUT_REACHES_HARDWARE

    def test_gated_routes_are_the_ones_that_drive_the_hardware(self, routes, hardware_methods):
        """the other direction: a route that drives the hardware and is not gated is the bug"""
        for route in routes:
            if core_calls(route["handler"]) & hardware_methods:
                assert route["gated"] is True or route["path"] in UNGATED_BUT_REACHES_HARDWARE, (
                    f"{route['path']} drives the hardware but is not gated"
                )


class TestNoRouteEscapesThePolicyGate:
    def test_every_post_route_carries_a_policy(self, routes):
        """
        a POST registered on a plain APIRouter has no gate at all

        `info.py` keeps one for the GET endpoints. A POST added to it would be served
        without `admit` ever running, and would publish no policy - which is what this
        notices.
        """
        without = [route["path"] for route in routes if route["gated"] is None]
        assert not without, f"POST routes with no policy: {without}"

    def test_a_new_route_is_gated_unless_someone_opts_out(self):
        """
        the default matters more than any individual route

        `Api.post` defaults to requiring the lease, so the failure mode for a forgotten
        decision is a route that is too strict rather than one that is open.
        """
        from seafront.server.api.routing import Api, Policy

        assert Policy().requires_lease is True
        signature = inspect.signature(Api.post)
        assert signature.parameters["policy"].default.requires_lease is True

    def test_no_get_route_reaches_the_hardware(self, app, hardware_methods):
        """
        a GET carries no body, so it could never carry a token

        Which means a GET that drives the hardware is not gateable at all, however it is
        declared.
        """
        offenders = []
        checked = 0
        for route in app.routes:
            if "GET" not in getattr(route, "methods", set()):
                continue
            handler = inspect.unwrap(route.endpoint)
            if not is_ours(handler):
                continue  # fastapi's own docs and schema endpoints
            checked += 1
            reaching = core_calls(handler) & hardware_methods
            if reaching:
                offenders.append(f"{route.path} calls core.{'/'.join(sorted(reaching))}")
        assert checked, "no GET handlers of ours were found; the sweep checked nothing"
        assert not offenders, offenders


class TestTheBrowserCannotSlipPast:
    """
    every hardware call in the web ui goes through the client that attaches the token

    `APIClient.post` is the only place `lock_token` is added. A call made any other way -
    a raw `fetch`, a new module that builds its own request - reaches the server without one
    and is refused, which is safe but broken; or worse, reaches a route that is not gated.
    """

    # the two clients are implementations of the API rather than callers of it: one attaches
    # the token, the other answers offline. Both name every endpoint by design, and both have
    # their own tests below.
    CLIENTS: ClassVar[set[str]] = {"src/api-client.js", "src/mock-api-client.js"}

    @pytest.fixture(scope="class")
    def frontend_sources(self) -> dict[str, str]:
        """every file that *calls* the api, with comments stripped"""
        sources = {}
        for path in sorted(WEB_STATIC.rglob("*.js")):
            if "vendor" in path.parts or "types" in path.parts:
                continue
            name = str(path.relative_to(WEB_STATIC))
            if name in self.CLIENTS:
                continue
            sources[name] = strip_comments(path.read_text())
        sources["index.html"] = strip_comments((WEB_STATIC / "index.html").read_text())
        assert len(sources) > 3, "the frontend sources were not found"
        return sources

    @pytest.fixture(scope="class")
    def gated_paths(self) -> set[str]:
        schema = make_app(None).openapi()
        return {
            path
            for path, operations in schema["paths"].items()
            for operation in operations.values()
            if operation.get("x-requires-device-lease")
        }

    def test_every_gated_endpoint_is_only_ever_posted_through_the_client(
        self, frontend_sources, gated_paths
    ):
        """
        the string appears, and the call around it is api.post

        Checked as text rather than by running the page: the property is about every call
        site existing in the source, and a test that exercises the ui would only cover the
        paths it happened to click.
        """
        offenders = []
        seen = 0
        for filename, source in frontend_sources.items():
            for path in gated_paths:
                for match in re.finditer(re.escape(f"'{path}'"), source):
                    seen += 1
                    line_start = source.rfind("\n", 0, match.start()) + 1
                    line = source[line_start : source.find("\n", match.end())]
                    if "api.post(" not in line:
                        offenders.append(f"{filename}: {line.strip()[:100]}")
        assert seen, "no gated endpoint was mentioned anywhere; the scan found nothing to check"
        assert not offenders, (
            "gated endpoints reached other than through api.post:\n  " + "\n  ".join(offenders)
        )

    def test_the_only_raw_fetch_is_the_liveness_probe(self, frontend_sources):
        """
        one `fetch(` outside the client is all it takes

        `APIClient.post` is where the token is attached; anything calling fetch directly is
        building a request the lease knows nothing about.
        """
        offenders = []
        for filename, source in frontend_sources.items():
            for match in re.finditer(r"\bfetch\s*\(", source):
                line_start = source.rfind("\n", 0, match.start()) + 1
                line_end = source.find("\n", match.end())
                window = source[line_start : line_end + 200]
                if any(allowed in window for allowed in FRONTEND_RAW_FETCH_ALLOWED):
                    continue
                offenders.append(f"{filename}: {source[line_start:line_end].strip()[:100]}")
        assert not offenders, "raw fetch calls outside the api client:\n  " + "\n  ".join(offenders)

    def test_the_client_attaches_the_token_to_every_post(self):
        """the mechanism itself, so a refactor cannot quietly drop it"""
        source = (WEB_STATIC / "src" / "api-client.js").read_text()
        post_body = source[source.index("    async post("): source.index("    async get(")]
        assert "this.getLockToken()" in post_body
        assert "lock_token" in post_body

    def test_the_client_only_retries_after_taking_a_lease_nobody_held(self):
        """
        `not_held` is retried, `held_by_other` is not

        Retrying the second would be the browser taking a microscope somebody else is using,
        which is precisely the sneaking-around worth worrying about.
        """
        source = (WEB_STATIC / "src" / "api-client.js").read_text()
        retry = source[source.index("if (response.status === 423"): source.index("return this.handleResponse")]
        assert "'not_held'" in retry
        assert "held_by_other" not in retry
        assert "leaseRetry: false" in retry, "the replay must not be able to retry again"


class TestTheOfflineMockClientAgrees:
    """
    the mock client refuses the same endpoints the server does

    It exists so the ui works with no server, and it keeps its own list of which endpoints
    need a lease. A list that drifts teaches whoever is developing against it the wrong
    rules, and the mistake only shows up on a microscope.
    """

    @pytest.fixture(scope="class")
    def mock_gated(self) -> set[str]:
        source = (WEB_STATIC / "src" / "mock-api-client.js").read_text()
        block = source[source.index("MOCK_LEASE_REQUIRED = new Set([") : source.index("]);", source.index("MOCK_LEASE_REQUIRED"))]
        return set(re.findall(r"'(/api/[^']+)'", block))

    @pytest.fixture(scope="class")
    def server_gated(self) -> set[str]:
        schema = make_app(None).openapi()
        return {
            path
            for path, operations in schema["paths"].items()
            for operation in operations.values()
            if operation.get("x-requires-device-lease")
        }

    def test_the_two_lists_are_the_same(self, mock_gated, server_gated):
        assert mock_gated == server_gated, (
            f"only the mock gates: {sorted(mock_gated - server_gated)}; "
            f"only the server gates: {sorted(server_gated - mock_gated)}"
        )

    def test_the_mock_actually_refuses_without_a_token(self):
        source = (WEB_STATIC / "src" / "mock-api-client.js").read_text()
        assert "MOCK_LEASE_REQUIRED.has(" in source, "the list is declared but never consulted"
