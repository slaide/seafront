"""Single-owner microscope worker thread (reject-on-busy).

One dedicated OS thread owns the microscope and runs exactly one command at a
time. Submission is **reject-on-busy, never queue**: `submit` refuses (returns
`None`) when a command is already in flight, rather than buffering it. Status
reads and cancellation are handled outside this worker and must never go through
`submit`.

This corrects the `queue.Queue` + always-enqueue worker sketched in
REFACTOR_PLAN.md: there is no command queue, only a single in-flight slot whose
availability *is* the reject-on-busy gate.

Non-streaming commands and the multi-channel snap orchestrations (snap-all,
progressive) route through here via `submit` / `submit_job`. `_run_coro` still
bridges the (async) `Microscope.execute` and those orchestrations to the worker
thread via `asyncio.run`; a later phase replaces that with a pure-sync execute path.
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
    make_coro: tp.Callable[[], tp.Coroutine[tp.Any, tp.Any, tp.Any]]
    label: str
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
        # Label of the command currently running, or None when idle. Surfaced in the
        # 409 busy reason so callers see WHAT the hardware is doing.
        self._current_label: str | None = None
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
        return self.submit_job(
            lambda: self._scope.execute(command), type(command).__qualname__
        )

    def submit_job(
        self,
        make_coro: tp.Callable[[], tp.Coroutine[tp.Any, tp.Any, tp.Any]],
        label: str = "job",
    ) -> tuple[concurrent.futures.Future[tp.Any], threading.Event] | None:
        """Run an arbitrary async orchestration on the worker thread.

        Same reject-on-busy contract as `submit`, for multi-command work that must
        hold the single owner for its whole duration (e.g. snap-all-channels,
        progressive snap). `make_coro` is called on the worker thread and its
        coroutine driven to completion there; poll `cancel` via
        `scope.raise_if_cancelled()` at checkpoints inside it.
        """
        if self._shutdown.is_set():
            return None
        # reject-on-busy: claim the single slot without waiting.
        if not self._inflight.acquire(blocking=False):
            return None
        reply: concurrent.futures.Future[tp.Any] = concurrent.futures.Future()
        cancel = threading.Event()
        self._handoff.put(WorkItem(make_coro, label, reply, cancel))
        return reply, cancel

    def current_cancel(self) -> threading.Event | None:
        """The cancel event of the in-flight command, or None when idle.

        Used by the cancel hatch. Never blocks, never rejected.
        """
        return self._current_cancel

    def current_label(self) -> str | None:
        """Label of the in-flight command (e.g. "acquisition:<id>", "MoveTo"), or None
        when idle. Surfaced in the busy (409) reason for debuggability."""
        return self._current_label

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
            self._current_label = item.label
            # Publish the cancel event onto the scope so long operations can poll it
            # via raise_if_cancelled().
            self._scope.set_current_cancel(item.cancel)
            try:
                result = self._run_coro(item.make_coro, item.label)
            except BaseException as e:
                # Any failure is relayed to the caller's future. Release the slot
                # BEFORE completing the future so a caller that awaits the result
                # and immediately resubmits finds the worker free.
                self._current_cancel = None
                self._current_label = None
                self._scope.set_current_cancel(None)
                self._inflight.release()
                item.reply.set_exception(e)
            else:
                self._current_cancel = None
                self._current_label = None
                self._scope.set_current_cancel(None)
                self._inflight.release()
                item.reply.set_result(result)

    def _run_coro(self, make_coro: tp.Callable[[], tp.Coroutine[tp.Any, tp.Any, tp.Any]], label: str) -> tp.Any:
        # Transitional bridge: Microscope.execute (and the orchestrations that call
        # it) are still async. Everything under them is sync (Phase 2), so this just
        # drives the coroutine to completion on the worker thread. A later phase
        # introduces a pure-sync execute path and deletes this asyncio.run.
        logger.debug(f"microscope-worker - running {label}")
        return asyncio.run(make_coro())
