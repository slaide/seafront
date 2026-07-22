"""Runtime introspection helpers: "where is everybody currently executing?"

Seafront runs every HTTP request in its own OS thread (``asyncio.to_thread(asyncio.run, ...)``
in ``__main__.py``), and hardware commands block synchronously on ``threading.RLock``
acquisitions and blocking driver I/O. That means a wedged request shows up as a real
thread whose Python stack ends exactly at the line it is stuck on -- so
``sys._current_frames()`` is enough to see where everything is, no matter how deeply
nested the async-inside-async-inside-thread structure is.

Three ways to get a dump, in order of robustness when the process is wedged:

1. ``kill -USR1 <pid>``  -> faulthandler C-level dump of all thread stacks to stderr.
   Works even when the GIL is held by a stuck C call (e.g. a camera SDK call). Wired
   up in ``__main__.py`` via ``faulthandler.register``.

2. ``kill -USR2 <pid>``  -> rich Python dump (thread stacks + microscope lock reasons +
   asyncio tasks) written to a timestamped file under ``~/seafront/logs/``. Needs the
   GIL to be obtainable by the main thread, but adds the lock-reason stack and asyncio
   task states that faulthandler cannot see. Registered via ``register_signal_dumper``.

3. ``GET /api/debug/stacks``  -> same rich dump returned as text/plain. Because each
   request runs in its own thread, this responds even while other request threads are
   blocked on locks. It cannot help if a thread is stuck holding the GIL in a C call
   (use option 1 for that).
"""

import asyncio
import io
import sys
import threading
import traceback
import typing as tp
from datetime import datetime
from pathlib import Path

if tp.TYPE_CHECKING:
    from seafront.hardware.microscope import Microscope


def format_stack_dump(
    microscope: "Microscope | None" = None,
    extra_loops: "list[asyncio.AbstractEventLoop] | None" = None,
) -> str:
    """Render a human-readable snapshot of every thread's current Python stack.

    Args:
        microscope: if given, its lock-reason stack and stream state are included --
            this is what tells you which command is holding the hardware lock (the same
            information returned in a 409 "microscope busy" response body).
        extra_loops: event loops whose asyncio tasks should also be dumped. Threads are
            always covered by the frame dump; pass the persistent acquisition worker loop
            here to also surface tasks that are suspended at an ``await`` (which do not
            appear in the thread frames because they are not currently running).
    """
    buf = io.StringIO()
    now = datetime.now().astimezone().isoformat()
    buf.write(f"=== seafront stack dump @ {now} ===\n")

    if microscope is not None:
        try:
            buf.write("\n--- microscope ---\n")
            buf.write(f"stream_callback set: {microscope.stream_callback is not None}\n")
        except Exception as e:
            buf.write(f"  (failed to read microscope state: {e!r})\n")

    frames = sys._current_frames()
    threads_by_id = {t.ident: t for t in threading.enumerate()}
    buf.write(f"\n--- thread stacks ({len(frames)} threads) ---\n")
    for tid, frame in sorted(frames.items()):
        th = threads_by_id.get(tid)
        name = th.name if th is not None else "<unknown>"
        daemon = th.daemon if th is not None else "?"
        buf.write(f"\n# Thread {name!r} (id={tid}, daemon={daemon})\n")
        buf.write("".join(traceback.format_stack(frame)))

    for loop in extra_loops or []:
        try:
            tasks = asyncio.all_tasks(loop)
        except RuntimeError:
            # loop not running
            continue
        buf.write(f"\n--- asyncio tasks on loop {loop!r} ({len(tasks)} tasks) ---\n")
        for task in tasks:
            buf.write(f"\n# Task {task!r}\n")
            try:
                task.print_stack(file=buf)
            except Exception as e:
                buf.write(f"  (failed to read task stack: {e!r})\n")

    buf.write("\n=== end stack dump ===\n")
    return buf.getvalue()


def register_signal_dumper(
    microscope: "Microscope",
    dump_dir: Path,
    extra_loops: "list[asyncio.AbstractEventLoop] | None" = None,
    signum: int = -1,
) -> None:
    """Install a SIGUSR2 handler that writes a rich stack dump to a timestamped file.

    Complements the faulthandler SIGUSR1 dump (which is more robust but cannot see lock
    reasons or asyncio task state). Safe to call once at startup.
    """
    import signal

    if signum < 0:
        signum = signal.SIGUSR2

    def handler(_signum: int, _frame: object) -> None:
        # Keep imports/logging out of the raw-signal-unsafe path as much as possible;
        # writing to a fresh file is the important, side-effect-free bit.
        try:
            text = format_stack_dump(microscope, extra_loops=extra_loops)
        except Exception as e:
            text = f"stack dump failed: {e!r}\n"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = dump_dir / f"stackdump-{stamp}.txt"
        try:
            path.write_text(text)
            sys.stderr.write(f"[debug_introspect] wrote stack dump to {path}\n")
            sys.stderr.flush()
        except Exception as e:
            # last resort: dump to stderr directly
            sys.stderr.write(f"[debug_introspect] failed to write {path}: {e!r}\n{text}")
            sys.stderr.flush()

    signal.signal(signum, handler)
