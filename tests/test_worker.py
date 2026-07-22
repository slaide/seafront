"""Tests for the single-owner MicroscopeWorker (Phase 3).

Covers the reject-on-busy contract and end-to-end command execution through the
worker against a real MockMicroscope.
"""

import asyncio
import os
import threading
import time

import pytest

from seafront.hardware.microscope import OperationCancelledError
from seafront.server import commands as cmd
from seafront.server.worker import MicroscopeWorker


def test_worker_runs_command_to_completion(mock_scope):
    """A submitted command runs on the worker thread and its result resolves."""
    worker = MicroscopeWorker(mock_scope)
    worker.start()
    try:
        submitted = worker.submit(cmd.MoveTo(x_mm=10.0, y_mm=10.0, z_mm=1.0))
        assert submitted is not None, "first submit must be accepted"
        future, _cancel = submitted

        result = future.result(timeout=10)
        assert result is not None

        # the move actually happened on the worker thread
        state = asyncio.run(mock_scope.get_current_state())
        assert state.stage_position.x_pos_mm == pytest.approx(10.0, abs=0.1)
        assert state.stage_position.y_pos_mm == pytest.approx(10.0, abs=0.1)
    finally:
        worker.stop()


class _GatedScope:
    """Minimal scope stand-in whose execute() blocks until released.

    Lets the reject-on-busy test hold the worker busy deterministically, with no
    reliance on wall-clock timing.
    """

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def set_current_cancel(self, cancel):  # noqa: ANN001, ANN201 - test stub
        pass

    async def execute(self, command):  # noqa: ANN001, ANN201 - test stub
        self.started.set()
        # blocks the worker thread until the test releases it
        assert self.release.wait(timeout=10), "release was never set"
        return "done"


def test_worker_rejects_second_command_while_busy():
    """While a command is in flight, a second submit is rejected (not queued)."""
    scope = _GatedScope()
    worker = MicroscopeWorker(scope)  # type: ignore[arg-type]
    worker.start()
    try:
        first = worker.submit(cmd.MoveTo(x_mm=1.0))
        assert first is not None, "first submit must be accepted"
        future_a, _ = first

        # wait until the worker is actually executing the first command
        assert scope.started.wait(timeout=10)
        assert worker.is_busy

        # reject-on-busy: second submit returns None instead of queuing
        assert worker.submit(cmd.MoveTo(x_mm=2.0)) is None

        # let the first finish, and confirm the worker frees up again
        scope.release.set()
        assert future_a.result(timeout=10) == "done"

        second = worker.submit(cmd.MoveTo(x_mm=3.0))
        assert second is not None, "worker must accept again once idle"
        scope.release.set()  # already set; the stub returns immediately
        assert second[0].result(timeout=10) == "done"
    finally:
        worker.stop()


def test_worker_cancel_interrupts_long_move(mock_scope):
    """Setting the in-flight cancel event aborts a long move and frees the worker."""
    # make the move take real wall-clock time so it can be cancelled mid-flight
    # (_realistic_delays_enabled reads this env var live; the fixture clears it on teardown)
    os.environ["MOCK_NO_DELAYS"] = "0"

    worker = MicroscopeWorker(mock_scope)
    worker.start()
    try:
        submitted = worker.submit(cmd.MoveTo(x_mm=50.0, y_mm=50.0, z_mm=1.0))
        assert submitted is not None
        future, cancel = submitted

        # request cancellation almost immediately; a move that far takes ~1s, so the
        # per-step checkpoint aborts it well before completion
        time.sleep(0.05)
        cancel.set()

        with pytest.raises(OperationCancelledError):
            future.result(timeout=5)

        # the worker is free again and accepts new work
        assert not worker.is_busy
        assert worker.submit(cmd.MoveTo(x_mm=1.0)) is not None
    finally:
        worker.stop()


def test_worker_reports_exception_through_future():
    """An exception in execute is delivered via the future, and the worker recovers."""

    class _BoomScope:
        def set_current_cancel(self, cancel):  # noqa: ANN001, ANN201 - test stub
            pass

        async def execute(self, command):  # noqa: ANN001, ANN201 - test stub
            raise RuntimeError("boom")

    worker = MicroscopeWorker(_BoomScope())  # type: ignore[arg-type]
    worker.start()
    try:
        submitted = worker.submit(cmd.MoveTo(x_mm=1.0))
        assert submitted is not None
        future, _ = submitted
        with pytest.raises(RuntimeError, match="boom"):
            future.result(timeout=10)

        # worker is not wedged: it accepts a new command after the failure
        assert not worker.is_busy
        assert worker.submit(cmd.MoveTo(x_mm=2.0)) is not None
    finally:
        worker.stop()
