"""
Noticing a hardware call that does not come back.

A camera SDK call that hangs inside the vendor library cannot be interrupted from python: the
thread is in C, and nothing short of ending the process gets it back. What can be done is to
notice, and to stop it doing harm while it hangs:

- every guarded call has a deadline, well past anything it legitimately takes; a call still
  running at its deadline is a stall
- a stall marks its device unresponsive on the health board, logs the stuck thread's stack
  (which is what says where in the SDK it is stuck), and runs the call's stall action - for a
  snap, turning the illumination off, which would otherwise stay on the sample for as long as
  the SDK takes to come back, which may be never
- while it lasts, it is listed by `stalls()`, which is what a busy rejection names and what
  shutdown checks before waiting on hardware that cannot answer
- if the call does come back, the device is marked alive again, and the call's result is
  not handed out if the stall action ran: an image exposed while the light was switched off
  under it is not the image that was asked for, so the call raises `StalledCallError` instead
"""

import sys
import threading
import time
import traceback
import typing as tp
from contextlib import contextmanager

from pydantic import BaseModel

from seafront.hardware.health import Device, HealthBoard, describe
from seafront.logger import logger


class Stall(BaseModel):
    "a guarded call that is past its deadline and still running"

    device: Device
    operation: str
    started_at: float
    "unix time"
    deadline_s: float
    thread: str
    "name of the thread that is stuck in it"

    def describe(self, now: float | None = None) -> str:
        running = (time.time() if now is None else now) - self.started_at
        return (
            f"{self.device} {self.operation} has not returned after {running:.0f} s "
            f"(deadline {self.deadline_s:.0f} s)"
        )


class StalledCallError(Exception):
    """
    A guarded call returned after its deadline, and its stall action ran in the meantime, so
    what it returned is not what was asked for.
    """


class _Guard:
    def __init__(
        self,
        device: Device,
        operation: str,
        deadline_s: float,
        on_stall: tp.Callable[[], str] | None,
    ):
        self.device = device
        self.operation = operation
        self.deadline_s = deadline_s
        self.on_stall = on_stall
        self.started_at = time.time()
        self.due = time.monotonic() + deadline_s
        self.thread = threading.current_thread()
        self.stalled = False
        "set by the watchdog thread, once, when the deadline passes"
        self.stall_action: str | None = None
        "what the stall action did, or how it failed; None if there was none"
        self.handled = threading.Event()
        """
        set once the watchdog has finished with a stall. A call that returns while its stall
        action is still running waits for it: otherwise it could hand out its result before
        the action it is about to be invalidated by had said it ran.
        """

    def as_stall(self) -> Stall:
        return Stall(
            device=self.device,
            operation=self.operation,
            started_at=self.started_at,
            deadline_s=self.deadline_s,
            thread=self.thread.name,
        )


class StallWatchdog:
    """
    One thread watching every guarded call an adapter makes. Started on first use, and a
    daemon: a watchdog is no reason to keep a process alive that is otherwise exiting.
    """

    def __init__(self, board: HealthBoard):
        self._board = board
        self._lock = threading.Condition()
        self._active: set[_Guard] = set()
        self._thread: threading.Thread | None = None

    @contextmanager
    def guard(
        self,
        device: Device,
        operation: str,
        deadline_s: float,
        on_stall: tp.Callable[[], str] | None = None,
    ) -> tp.Iterator[None]:
        """
        watch the enclosed call. `on_stall` runs on the watchdog thread at the deadline, while
        the call is still stuck, and says what it did ("the illumination was turned off"); it
        must not need anything the stuck call holds.
        """
        guard = _Guard(device, operation, deadline_s, on_stall)
        with self._lock:
            self._active.add(guard)
            self._ensure_running()
            self._lock.notify()
        try:
            yield
        finally:
            with self._lock:
                self._active.discard(guard)
                stalled = guard.stalled
            if stalled:
                guard.handled.wait()
                self._returned_late(guard)
        # only reached when the call itself succeeded: a call that raised has already said
        # it failed, and that error is the more specific one
        if stalled and guard.stall_action is not None:
            raise StalledCallError(
                f"{device} {operation} returned after "
                f"{time.time() - guard.started_at:.0f} s, past its {deadline_s:.0f} s "
                f"deadline, and {guard.stall_action} while it was stuck"
            )

    def stalls(self) -> list[Stall]:
        with self._lock:
            return [guard.as_stall() for guard in self._active if guard.stalled]

    # -- the watchdog thread ----------------------------------------------------------------

    def _ensure_running(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._watch, name="stall-watchdog", daemon=True)
            self._thread.start()

    def _watch(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                overdue = [g for g in self._active if not g.stalled and g.due <= now]
                for guard in overdue:
                    guard.stalled = True
                if not overdue:
                    pending = [g.due for g in self._active if not g.stalled]
                    self._lock.wait(timeout=None if not pending else max(0.0, min(pending) - now))
                    continue
            for guard in overdue:
                self._stalled(guard)

    def _stalled(self, guard: _Guard) -> None:
        try:
            stall = guard.as_stall()
            frame = sys._current_frames().get(guard.thread.ident or -1)
            where = "".join(traceback.format_stack(frame)) if frame is not None else "(thread gone)"
            logger.error(f"stall - {stall.describe()}; {guard.thread.name} is stuck at:\n{where}")

            if guard.on_stall is not None:
                try:
                    guard.stall_action = guard.on_stall()
                except BaseException as e:
                    # BaseException: the microcontroller's timeout is one, and this thread is
                    # the only one that would ever see it
                    guard.stall_action = f"the stall action failed ({describe(e)})"
                    logger.opt(exception=e).error(f"stall - {stall.describe()}: stall action failed")
                message = f"{stall.describe()}; {guard.stall_action}"
            else:
                message = stall.describe()

            self._board.unresponsive(guard.device, guard.operation, message)
        finally:
            guard.handled.set()

    def _returned_late(self, guard: _Guard) -> None:
        logger.warning(
            f"stall - {guard.device} {guard.operation} returned after "
            f"{time.time() - guard.started_at:.0f} s (deadline {guard.deadline_s:.0f} s)"
        )
        self._board.heard(guard.device)
