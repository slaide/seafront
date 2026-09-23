"""
The acquisitions the server knows about, and for how long.

An acquisition's status is kept while it runs and for a while after, so a client that
started one can read how it ended. Not forever: each status carries the whole protocol it
ran, and a server that stays up for months runs thousands. The most recent finished ones are
kept, up to `keep_finished`; the one running is always kept.

Only acquisitions the worker accepted are here. A start refused while being checked (bad
names, a forbidden site, a busy microscope) never ran, so it has no status to report, and a
record of it would only ever say "scheduled".
"""

import threading
import time

from seafront.server.commands import AcquisitionStatus


class AcquisitionRegistry:
    def __init__(self, keep_finished: int = 32):
        self.keep_finished = keep_finished
        self._lock = threading.Lock()
        "written from the worker thread (finish) and the event loop (add, discard, get)"
        self._statuses: dict[str, AcquisitionStatus] = {}

    def add(self, status: AcquisitionStatus) -> None:
        "an acquisition about to be handed to the worker"
        with self._lock:
            self._statuses[status.acquisition_id] = status

    def discard(self, acquisition_id: str) -> None:
        "an acquisition the worker refused, so it never ran"
        with self._lock:
            self._statuses.pop(acquisition_id, None)

    def finish(self, acquisition_id: str) -> None:
        "an acquisition that has stopped, however it stopped; forgets the oldest finished ones"
        with self._lock:
            self._statuses[acquisition_id].finished_at = time.time()
            finished = sorted(
                (s.finished_at, s.acquisition_id)
                for s in self._statuses.values()
                if s.finished_at is not None
            )
            for _, oldest in finished[: max(0, len(finished) - self.keep_finished)]:
                del self._statuses[oldest]

    def get(self, acquisition_id: str) -> AcquisitionStatus | None:
        with self._lock:
            return self._statuses.get(acquisition_id)

    def describe_unknown(self, acquisition_id: str) -> str:
        "what to tell a client asking about an acquisition that is not here"
        return (
            f"invalid acquisition_id {acquisition_id}: not found among the running "
            f"acquisition and the last {self.keep_finished} finished ones (none are kept from "
            f"before the server was last started)"
        )

