"""Single-owner microscope worker thread (reject-on-busy).

One dedicated OS thread owns the microscope and runs exactly one command at a
time. Submission is **reject-on-busy, never queue**: `submit` refuses (returns
`None`) when a command is already in flight, rather than buffering it. Status
reads and cancellation are handled outside this worker and must never go through
`submit`.

This corrects the `queue.Queue` + always-enqueue worker sketched in
REFACTOR_PLAN.md: there is no command queue, only a single in-flight slot whose
availability *is* the reject-on-busy gate.

Phase 3 status: this module exists and is started by `Core`, but no HTTP route
submits to it yet. `_run_command` still bridges to the (async) `Microscope.execute`
via `asyncio.run`; Phase 4 replaces that with a pure-sync execute path.
"""

import asyncio
import concurrent.futures
import queue
import threading
import typing as tp

from seafront.hardware.microscope import Microscope
from seafront.logger import logger
from seafront.server import commands as cmd


class WorkItem(tp.NamedTuple):
    command: cmd.BaseCommand[tp.Any]
    reply: concurrent.futures.Future[tp.Any]
    cancel: threading.Event


class MicroscopeWorker:
    """Owns a microscope on a dedicated thread and runs one command at a time.

    reject-on-busy: `submit` claims a single in-flight slot without blocking; if
    the slot is already taken it returns `None` (the caller turns that into a 409).
    The slot is released only after the in-flight command finishes, so no command
    ever waits behind another.
    """

    # sentinel pushed by stop() to make the worker loop exit
    _SHUTDOWN: tp.Final = None

    def __init__(self, scope: Microscope) -> None:
        self._scope = scope
        # Single-slot handoff. The `_inflight` lock guarantees at most one real
        # WorkItem is ever in flight, so this only ever holds one item (or the
        # shutdown sentinel). maxsize=1 documents that invariant.
        self._handoff: queue.Queue[WorkItem | None] = queue.Queue(maxsize=1)
        # The in-flight token. acquire(blocking=False) succeeding == the slot was
        # free == this submit is accepted. Released by the worker once the command
        # completes. This lock IS the reject-on-busy gate.
        self._inflight = threading.Lock()
        # Cancel event of the command currently running, or None when idle. Read by
        # the cancel hatch; only ever mutated by the worker thread.
        self._current_cancel: threading.Event | None = None
        self._thread = threading.Thread(
            target=self._run, name="microscope-worker", daemon=True
        )
        self._started = False
        self._shutdown = threading.Event()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start the worker thread. Idempotent."""
        if self._started:
            return
        self._started = True
        self._thread.start()
        logger.debug("microscope-worker - started")

    def stop(self) -> None:
        """Stop accepting work and join the worker thread. Idempotent.

        Blocks until any in-flight command drains, then joins.
        """
        if not self._started:
            return
        self._shutdown.set()
        self._handoff.put(self._SHUTDOWN)
        self._thread.join()
        self._started = False
        logger.debug("microscope-worker - stopped")

    # -- submission (reject-on-busy) --------------------------------------

    def submit(
        self, command: cmd.BaseCommand[tp.Any]
    ) -> tuple[concurrent.futures.Future[tp.Any], threading.Event] | None:
        """Hand a command to the worker.

        Returns `(future, cancel)` on acceptance, or `None` if a command is
        already in flight (reject-on-busy) or the worker is shutting down. The
        caller resolves `future` for the result and may `cancel.set()` to request
        cooperative cancellation.
        """
        if self._shutdown.is_set():
            return None
        # reject-on-busy: claim the single slot without waiting.
        if not self._inflight.acquire(blocking=False):
            return None
        reply: concurrent.futures.Future[tp.Any] = concurrent.futures.Future()
        cancel = threading.Event()
        self._handoff.put(WorkItem(command, reply, cancel))
        return reply, cancel

    def current_cancel(self) -> threading.Event | None:
        """The cancel event of the in-flight command, or None when idle.

        Used by the cancel hatch. Never blocks, never rejected.
        """
        return self._current_cancel

    @property
    def is_busy(self) -> bool:
        """True while a command is in flight (from acceptance until completion).

        Backed by the in-flight token, so it also covers the brief handoff window
        before the worker thread picks the command up. Read by the status hatch;
        never blocks.
        """
        return self._inflight.locked()

    # -- worker thread -----------------------------------------------------

    def _run(self) -> None:
        while True:
            item = self._handoff.get()
            if item is None:  # shutdown sentinel (see _SHUTDOWN / stop())
                return

            self._current_cancel = item.cancel
            # Publish the cancel event onto the scope so long operations can poll it
            # via raise_if_cancelled().
            self._scope.set_current_cancel(item.cancel)
            try:
                result = self._run_command(item.command)
            except BaseException as e:
                # Any failure is relayed to the caller's future. Release the slot
                # BEFORE completing the future so a caller that awaits the result
                # and immediately resubmits finds the worker free.
                self._current_cancel = None
                self._scope.set_current_cancel(None)
                self._inflight.release()
                item.reply.set_exception(e)
            else:
                self._current_cancel = None
                self._scope.set_current_cancel(None)
                self._inflight.release()
                item.reply.set_result(result)

    def _run_command(self, command: cmd.BaseCommand[tp.Any]) -> tp.Any:
        # Transitional bridge: Microscope.execute is still async at the HTTP
        # boundary. Everything under it is already sync (Phase 2), so this just
        # drives the coroutine to completion on the worker thread. Phase 4
        # introduces a pure-sync execute path and deletes this asyncio.run.
        return asyncio.run(self._scope.execute(command))
