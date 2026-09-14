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
        # the in-flight label is exposed so a busy rejection can name what's running
        assert worker.current_label() == "MoveTo"

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


class _CountingScope:
    """Scope stand-in that records what it was asked to run and can be held mid-command."""

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancels: list[threading.Event | None] = []

    def set_current_cancel(self, cancel):
        self.cancels.append(cancel)

    async def execute(self, command):
        self.commands.append(type(command).__qualname__)
        self.started.set()
        assert self.release.wait(timeout=10), "release was never set"
        return "done"


class TestWorkerUnderConcurrency:
    """
    The single-slot handoff, hammered.

    These are the races that an occasional red test in a bigger suite would be blamed on: two
    submits landing together, a cancel arriving as a command ends, a resubmit the instant a
    result comes back. All of it is synchronised with events and barriers rather than sleeps,
    so a failure here is a real defect and not a slow machine.
    """

    def test_simultaneous_submits_accept_exactly_one(self):
        """Eight threads submitting at the same instant: one runs, seven are told busy."""
        scope = _CountingScope()
        worker = MicroscopeWorker(scope)  # type: ignore[arg-type]
        worker.start()
        try:
            accepted = []
            barrier = threading.Barrier(8)

            def contend(n: int):
                barrier.wait()
                submitted = worker.submit(cmd.MoveTo(x_mm=float(n)))
                if submitted is not None:
                    accepted.append(submitted)

            threads = [threading.Thread(target=contend, args=(n,)) for n in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(accepted) == 1, f"{len(accepted)} submits were accepted, not 1"
            assert worker.is_busy
            scope.release.set()
            assert accepted[0][0].result(timeout=10) == "done"
        finally:
            scope.release.set()
            worker.stop()

    def test_a_cancel_cannot_reach_the_next_command(self):
        """
        A cancel aimed at the running command must not carry over to the one after it.

        Cancel is a hatch: it reads whatever is in flight and sets its event. If the event
        were reused, a cancel arriving just as a command ends would kill the next one - the
        kind of failure that shows up once in fifty runs and gets called flaky.
        """
        scope = _CountingScope()
        worker = MicroscopeWorker(scope)  # type: ignore[arg-type]
        worker.start()
        try:
            first = worker.submit(cmd.MoveTo(x_mm=1.0))
            assert first is not None
            assert scope.started.wait(timeout=10)

            # cancel the in-flight command the way /api/action/cancel_current does
            in_flight = worker.current_cancel()
            assert in_flight is not None
            in_flight.set()

            scope.release.set()
            assert first[0].result(timeout=10) == "done"

            # idle: nothing in flight to cancel, and nothing half-cleared
            assert worker.current_cancel() is None
            assert worker.current_label() is None
            assert not worker.is_busy

            scope.started.clear()
            second = worker.submit(cmd.MoveTo(x_mm=2.0))
            assert second is not None
            assert scope.started.wait(timeout=10)
            assert not second[1].is_set(), "the next command started out already cancelled"
            assert second[1] is not in_flight, "the cancel event was reused across commands"
            assert second[0].result(timeout=10) == "done"
        finally:
            scope.release.set()
            worker.stop()

    def test_back_to_back_submits_are_always_accepted(self, mock_scope):
        """
        Resubmitting the instant a result arrives must never be refused.

        The slot is released before the caller's future resolves precisely so this holds; if
        that order ever flips, this fails intermittently instead of silently costing every
        client a retry.
        """
        worker = MicroscopeWorker(mock_scope)
        worker.start()
        try:
            for i in range(25):
                submitted = worker.submit(cmd.MoveTo(x_mm=10.0 + i * 0.1, y_mm=10.0, z_mm=1.0))
                assert submitted is not None, f"submit {i} was refused right after the previous result"
                submitted[0].result(timeout=10)
        finally:
            worker.stop()

    def test_nothing_is_in_flight_between_commands(self, mock_scope):
        """label, cancel and busy agree with each other, before and after every command"""
        worker = MicroscopeWorker(mock_scope)
        worker.start()
        try:
            for i in range(10):
                assert not worker.is_busy
                assert worker.current_label() is None
                assert worker.current_cancel() is None
                submitted = worker.submit(cmd.MoveTo(x_mm=10.0 + i * 0.1, y_mm=10.0, z_mm=1.0))
                assert submitted is not None
                submitted[0].result(timeout=10)
        finally:
            worker.stop()


class TestWorkerTiming:
    """
    When the slot is taken and when it is given back.

    Two orderings in `_run` are load-bearing and invisible from outside unless something
    checks them. Both are written down as comments in the worker; comments do not fail.
    """

    def test_the_slot_is_taken_before_the_worker_thread_has_seen_the_command(self):
        """
        `is_busy` covers the handoff window

        `submit` claims the slot and *then* posts the item; the worker thread picks it up
        whenever it gets round to it. If `is_busy` were derived from the thread having
        started the command, there would be a window where a submitted command reports the
        microscope idle - and a status read landing there tells a client the hardware is
        free while a move is on its way to it.
        """
        scope = _CountingScope()
        worker = MicroscopeWorker(scope)  # type: ignore[arg-type]
        worker.start()
        try:
            submitted = worker.submit(cmd.MoveTo(x_mm=1.0))
            assert submitted is not None
            # no wait: this is the instant after submit returns
            assert worker.is_busy, "the slot was not claimed by the time submit() returned"
            scope.release.set()
            submitted[0].result(timeout=10)
        finally:
            scope.release.set()
            worker.stop()

    def test_a_second_submit_in_that_window_is_already_refused(self):
        """the same window, from the other side: reject-on-busy covers it too"""
        scope = _CountingScope()
        worker = MicroscopeWorker(scope)  # type: ignore[arg-type]
        worker.start()
        try:
            first = worker.submit(cmd.MoveTo(x_mm=1.0))
            assert first is not None
            assert worker.submit(cmd.MoveTo(x_mm=2.0)) is None
            scope.release.set()
            first[0].result(timeout=10)
        finally:
            scope.release.set()
            worker.stop()

    def test_the_slot_is_free_by_the_time_the_result_arrives(self):
        """
        the ordering that makes await-then-resubmit work

        `_run` releases the slot *before* completing the future, so a caller that awaits a
        result and immediately submits the next command finds the worker free. Completing
        the future first would make every back-to-back pair of commands a coin flip between
        working and coming back 409.
        """
        scope = _CountingScope()
        worker = MicroscopeWorker(scope)  # type: ignore[arg-type]
        worker.start()
        try:
            for n in range(25):
                scope.release.clear()
                scope.started.clear()
                submitted = worker.submit(cmd.MoveTo(x_mm=float(n)))
                assert submitted is not None, f"round {n}: the slot was still held"
                scope.release.set()
                submitted[0].result(timeout=10)
                # immediately, with no settling time at all
                assert not worker.is_busy, f"round {n}: slot still held when the result arrived"
        finally:
            scope.release.set()
            worker.stop()

    def test_the_job_handle_is_gone_by_the_time_the_result_arrives(self):
        """status must not still be naming a command whose result the caller already has"""
        scope = _CountingScope()
        worker = MicroscopeWorker(scope)  # type: ignore[arg-type]
        worker.start()
        try:
            submitted = worker.submit(cmd.MoveTo(x_mm=1.0))
            assert submitted is not None
            scope.release.set()
            submitted[0].result(timeout=10)
            assert worker.current_job() is None
            assert worker.current_label() is None
        finally:
            scope.release.set()
            worker.stop()

    def test_a_failing_command_frees_the_slot_too(self):
        """the exception path releases in the same order as the success path"""

        class _Exploding:
            def set_current_cancel(self, cancel):
                pass

            async def execute(self, command):
                raise RuntimeError("boom")

        worker = MicroscopeWorker(_Exploding())  # type: ignore[arg-type]
        worker.start()
        try:
            submitted = worker.submit(cmd.MoveTo(x_mm=1.0))
            assert submitted is not None
            with pytest.raises(RuntimeError, match="boom"):
                submitted[0].result(timeout=10)
            assert not worker.is_busy, "a failed command left the slot held"
            assert worker.current_job() is None
            # and the worker still takes work afterwards
            again = worker.submit(cmd.MoveTo(x_mm=2.0))
            assert again is not None
            with pytest.raises(RuntimeError):
                again[0].result(timeout=10)
        finally:
            worker.stop()

    def test_a_cancelled_command_gives_the_slot_back_promptly(self, mock_scope):
        """
        cancel latency, bounded

        A long move polls its cancel event at checkpoints. What matters for the client is
        that the worker is usable again shortly after the cancel, not eventually.
        """
        worker = MicroscopeWorker(mock_scope)
        worker.start()
        try:
            os.environ.pop("MOCK_NO_DELAYS", None)
            submitted = worker.submit(cmd.MoveTo(x_mm=100.0, y_mm=70.0))
            assert submitted is not None
            time.sleep(0.05)

            running = worker.current_job()
            assert running is not None
            started = time.monotonic()
            running.cancel.set()

            with pytest.raises((OperationCancelledError, Exception)):
                submitted[0].result(timeout=10)
            freed = time.monotonic() - started

            assert not worker.is_busy
            assert freed < 5.0, f"the worker stayed busy {freed:.1f}s after the cancel"
        finally:
            os.environ["MOCK_NO_DELAYS"] = "1"
            worker.stop()

    def test_stop_waits_for_the_command_in_flight(self):
        """`stop()` drains rather than abandoning: the command finishes before the join"""
        scope = _CountingScope()
        worker = MicroscopeWorker(scope)  # type: ignore[arg-type]
        worker.start()
        submitted = worker.submit(cmd.MoveTo(x_mm=1.0))
        assert submitted is not None
        assert scope.started.wait(timeout=10)

        stopped = threading.Event()

        def stop_it():
            worker.stop()
            stopped.set()

        stopper = threading.Thread(target=stop_it, daemon=True)
        stopper.start()
        try:
            assert not stopped.wait(timeout=0.3), "stop() abandoned a command in flight"
            scope.release.set()
            assert stopped.wait(timeout=10), "stop() never returned after the command finished"
            assert submitted[0].result(timeout=10) == "done"
        finally:
            scope.release.set()
            stopper.join(timeout=10)
