"""
What the log says about each command the worker runs.

- every accepted command gets a number, and a start line and an end line carrying it
- the end line says how it ended: finished, cancelled, refused (with the reason), or failed
  (with the error), and how long it took
- a command turned away because the worker is busy is logged too, naming what holds it
"""

import re
import threading

import pytest
from fastapi import HTTPException

from seafront.hardware.microscope import OperationCancelledError
from seafront.logger import logger
from seafront.server import commands as cmd
from seafront.server.worker import MicroscopeWorker


class _Scope:
    "runs each command by calling `answer`; blocks until `release` is set if `gate` is"

    def __init__(self, answer=lambda: "done", gate: bool = False) -> None:
        self.answer = answer
        self.started = threading.Event()
        self.release = threading.Event()
        if not gate:
            self.release.set()

    def set_current_cancel(self, cancel):
        pass

    async def execute(self, command):
        if isinstance(command, cmd.EstablishHardwareConnection):
            return None
        self.started.set()
        assert self.release.wait(timeout=10)
        return self.answer()


@pytest.fixture
def log_lines():
    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(message.record["message"]), level="INFO")
    yield lines
    logger.remove(sink)


def run_one(scope, command=None):
    worker = MicroscopeWorker(scope)  # type: ignore[arg-type]
    worker.start()
    try:
        submitted = worker.submit(command or cmd.MoveTo(x_mm=1.0))
        assert submitted is not None
        try:
            return submitted[0].result(timeout=10)
        except BaseException as e:
            return e
    finally:
        worker.stop()


def command_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if line.startswith("command ")]


def test_a_command_that_finishes(log_lines):
    run_one(_Scope())
    start, end = command_lines(log_lines)
    assert start == "command #1 MoveTo - started"
    assert re.fullmatch(r"command #1 MoveTo - finished in \d+\.\d{3} s", end)


def test_a_cancelled_command(log_lines):
    def cancelled():
        raise OperationCancelledError()

    run_one(_Scope(cancelled))
    assert re.fullmatch(r"command #1 MoveTo - cancelled after \d+\.\d{3} s", command_lines(log_lines)[-1])


def test_a_command_refusing_its_input_says_why(log_lines):
    def refuse():
        raise HTTPException(status_code=400, detail="x_mm is outside the stage range")

    run_one(_Scope(refuse))
    end = command_lines(log_lines)[-1]
    assert re.fullmatch(
        r"command #1 MoveTo - refused after \d+\.\d{3} s: x_mm is outside the stage range", end
    )


def test_a_failed_command_names_the_error(log_lines):
    def fail():
        raise RuntimeError("camera gone")

    run_one(_Scope(fail))
    end = command_lines(log_lines)[-1]
    assert re.fullmatch(r"command #1 MoveTo - failed after \d+\.\d{3} s: RuntimeError: camera gone", end)


def test_commands_are_numbered_in_order(log_lines):
    scope = _Scope()
    worker = MicroscopeWorker(scope)  # type: ignore[arg-type]
    worker.start()
    try:
        for x in (1.0, 2.0):
            submitted = worker.submit(cmd.MoveTo(x_mm=x))
            assert submitted is not None
            submitted[0].result(timeout=10)
    finally:
        worker.stop()
    starts = [line for line in command_lines(log_lines) if line.endswith("started")]
    assert starts == ["command #1 MoveTo - started", "command #2 MoveTo - started"]


def test_a_busy_refusal_names_what_holds_the_worker(log_lines):
    scope = _Scope(gate=True)
    worker = MicroscopeWorker(scope)  # type: ignore[arg-type]
    worker.start()
    try:
        first = worker.submit(cmd.MoveTo(x_mm=1.0))
        assert first is not None
        assert scope.started.wait(timeout=10)
        assert worker.submit(cmd.ChannelStreamEnd.model_construct()) is None
        scope.release.set()
        first[0].result(timeout=10)
    finally:
        worker.stop()
    assert "command ChannelStreamEnd - refused: busy with #1 MoveTo" in log_lines


def test_a_refusal_while_shutting_down_is_logged(log_lines):
    worker = MicroscopeWorker(_Scope())  # type: ignore[arg-type]
    worker.start()
    worker.stop()
    worker._shutdown.set()
    assert worker.submit(cmd.MoveTo(x_mm=1.0)) is None
    assert "command MoveTo - refused: the microscope worker is shutting down" in log_lines
