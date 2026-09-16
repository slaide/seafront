"""
A real seafront, over HTTP, for the tests that need one.

Two things used to make this fragile.

The scratch home was copied from `~/seafront/config.json`, and the fixtures called
`pytest.skip` when there wasn't one. On any machine but the author's - every fresh clone,
every collaborator, every reviewer - ninety tests across four files reported success without
running a single line. `examples/mock_config.json` is in the repository and says the same
thing, so the home is built from that instead and there is nothing left to skip on.

The port was a module constant, hand-allocated per file (5000, 5011, 5013, 5014) with comments
explaining which file owned which. Two test runs at once, or one beside a real seafront, and
the `assert _port_free()` guarding it fired. The kernel hands out a free port perfectly well.
"""

import contextlib
import json
import os
import socket
import subprocess
import time
import typing as tp
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MOCK_CONFIG = REPO_ROOT / "examples" / "mock_config.json"


def _mock_microscope_name() -> str:
    "the name the example config gives its mock, so the two cannot drift apart"
    cfg = json.loads(MOCK_CONFIG.read_text())
    for m in cfg["microscopes"]:
        if m.get("system.microscope_type") == "mock":
            return m["system.microscope_name"]
    raise AssertionError(f"{MOCK_CONFIG} defines no microscope of type 'mock'")


MOCK = _mock_microscope_name()


def free_port() -> int:
    """
    A port the kernel says is free right now.

    Bind to 0, read the assignment back, close. Something else could in principle take it in
    the gap before the server binds, but nothing else here binds ports, and the fixed
    constants this replaces collided every single time two runs overlapped.
    """
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class MockServer(tp.NamedTuple):
    home: Path
    port: int
    url: str
    proc: "subprocess.Popen[str]"
    log_path: Path


@contextlib.contextmanager
def mock_server(tmp_path_factory, *, no_delays: bool = True) -> tp.Iterator[MockServer]:
    """
    Boot `python -m seafront` against a scratch SEAFRONT_HOME built from the example config.

    `no_delays=False` leaves MOCK_NO_DELAYS unset, so the mock spends real time on exposures -
    which a test wanting an acquisition that outlives the requests it makes during one needs.
    """
    home = Path(tmp_path_factory.mktemp("seafront-home"))
    images = home / "images"
    images.mkdir()

    cfg = json.loads(MOCK_CONFIG.read_text())
    port = free_port()
    cfg["port"] = port
    for m in cfg["microscopes"]:
        if m.get("system.microscope_name") == MOCK:
            m["storage.base_image_output_dir"] = str(images)
    (home / "config.json").write_text(json.dumps(cfg, indent=2))
    # no acquisition_configs is seeded: the server generates default.json on first start, and
    # that generated protocol is the one every test here fetches and reshapes.

    env = dict(os.environ, SEAFRONT_HOME=str(home))
    if no_delays:
        env["MOCK_NO_DELAYS"] = "1"
    else:
        env.pop("MOCK_NO_DELAYS", None)

    url = f"http://127.0.0.1:{port}"
    # never a PIPE nobody reads: a full pipe buffer blocks the server's event loop on every
    # log write, which after a few dozen images is every few milliseconds
    log_path = home / "server.log"
    log = log_path.open("w")
    proc = subprocess.Popen(
        ["uv", "run", "python", "-m", "seafront", "--microscope", MOCK, "--port", str(port)],
        stdout=log, stderr=subprocess.STDOUT, text=True, env=env,
    )
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"test server exited with {proc.returncode}\n{_tail(log_path)}"
                )
            try:
                with urllib.request.urlopen(f"{url}/api/health", timeout=1) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, ConnectionRefusedError):
                time.sleep(0.3)
        else:
            raise RuntimeError(f"test server did not start\n{_tail(log_path)}")
        yield MockServer(home=home, port=port, url=url, proc=proc, log_path=log_path)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


def _tail(log_path: Path, limit: int = 3000) -> str:
    try:
        return log_path.read_text()[-limit:]
    except OSError:
        return "<no server log>"
