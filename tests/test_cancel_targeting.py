"""
Cancel stops the thing you asked it to stop.

`/api/acquisition/cancel` names a run. It is also ungated - anyone may call it, because
stopping the hardware must never wait on a lease - which makes it the one route a client can
aim at the microscope without owning it. It had better hit what it was aimed at.

It used to set whatever cancel event the worker happened to be holding, having decided
"is this run still going?" from `AcquisitionStatus.thread_is_running` refreshed against
`Core.acquisition_future` - the *latest* run's future, not this one's. Cancelling a finished
run while a different one was going therefore killed the different one.

The worker already knows, exactly: it labels an acquisition job `acquisition:<id>`. These
tests pin that the label is what decides, and that the label and the cancel event it belongs
to are read as one value rather than two that can come from different jobs.
"""

import asyncio
import threading
import time

import pytest
from fastapi import HTTPException

from seafront.server.commands import AcquisitionStatus
from seafront.server.core import ACQUISITION_LABEL_PREFIX
from seafront.server.worker import MicroscopeWorker


class Job:
    """a worker job that runs until released, so a test can hold the slot deliberately"""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.started = threading.Event()

    async def __call__(self) -> None:
        self.started.set()
        while not self.release.is_set():
            await asyncio.sleep(0.01)


class Harness:
    """
    hands out worker jobs and takes them all back again

    Every job here holds the single worker slot until something lets it go, so a test that
    fails partway through would otherwise leave the worker occupied forever - and `stop()`
    waits for the in-flight job to drain, which turns a failed assertion into a hung suite.
    Releasing happens in fixture teardown, where a failed assertion cannot skip it.
    """

    def __init__(self, worker: MicroscopeWorker) -> None:
        self.worker = worker
        self._jobs: list[Job] = []

    def occupy(self, label: str) -> tuple[Job, threading.Event]:
        job = Job()
        claimed = self.worker.submit_job(job, label)
        assert claimed is not None, f"could not claim the worker for {label}"
        self._jobs.append(job)
        assert job.started.wait(timeout=5), f"{label} never started"
        return job, claimed[1]

    def release_all(self, timeout_s: float = 10.0) -> None:
        for job in self._jobs:
            job.release.set()
        deadline = time.time() + timeout_s
        while self.worker.is_busy and time.time() < deadline:
            time.sleep(0.01)


@pytest.fixture
def harness(mock_scope):
    """a worker of its own, emptied and stopped however the test leaves it"""
    worker = MicroscopeWorker(mock_scope)
    worker.start()
    h = Harness(worker)
    yield h
    h.release_all()
    worker.stop()


@pytest.fixture
def core_jobs(mock_core):
    """the Core's own worker, with the same guarantee"""
    h = Harness(mock_core.worker)
    yield h
    h.release_all()


def register(core, acquisition_id: str, running: bool) -> None:
    """put an acquisition in the map the way start_acquisition would"""
    core.acquisitions.add(AcquisitionStatus(acquisition_id=acquisition_id))
    if not running:
        core.acquisitions.finish(acquisition_id)


class TestTheWorkerPublishesOneJob:
    """
    label and cancel event come from one read

    They were two fields, set in two statements and cleared in two more, so a reader could
    take the label of the job that just finished and the cancel event of the one that just
    started. Anything that decides *whether* to cancel from one and *what* to cancel from
    the other needs them to be the same job.
    """

    def test_idle_worker_has_no_job(self, harness):
        assert harness.worker.current_job() is None
        assert harness.worker.current_label() is None
        assert harness.worker.current_cancel() is None

    def test_a_running_job_is_one_value(self, harness):
        _job, cancel = harness.occupy("acquisition:abc")
        running = harness.worker.current_job()
        assert running is not None
        assert running.label == "acquisition:abc"
        assert running.cancel is cancel
        # the convenience accessors read that same value
        assert harness.worker.current_label() == running.label
        assert harness.worker.current_cancel() is running.cancel

    def test_the_pair_is_never_half_set_while_jobs_churn(self, harness):
        """
        hammer the accessor while jobs start and finish

        A label with no cancel event, or the other way round, is the shape of the bug: it
        means the two came from different instants.
        """
        stop = threading.Event()
        contradictions: list[str] = []

        def watch():
            while not stop.is_set():
                running = harness.worker.current_job()
                if running is not None and (running.label is None or running.cancel is None):
                    contradictions.append(repr(running))

        watcher = threading.Thread(target=watch)
        watcher.start()
        try:
            for n in range(60):
                job, _cancel = harness.occupy(f"job-{n}")
                job.release.set()
                while harness.worker.is_busy:
                    time.sleep(0.001)
        finally:
            stop.set()
            watcher.join()

        assert not contradictions, contradictions[:3]


class TestCancelHitsTheRunItNames:
    def test_an_unknown_id_is_refused(self, mock_core):
        with pytest.raises(HTTPException) as caught:
            asyncio.run(mock_core.cancel_acquisition("never-existed"))
        assert "not found" in str(caught.value.detail)

    def test_cancelling_a_run_that_is_not_on_the_hardware_is_refused(self, mock_core):
        """the bookkeeping says it runs; the worker says otherwise, and the worker is right"""
        register(mock_core, "stale", running=True)
        with pytest.raises(HTTPException) as caught:
            asyncio.run(mock_core.cancel_acquisition("stale"))
        assert "not running" in str(caught.value.detail)

    def test_cancelling_the_running_run_sets_its_cancel_event(self, mock_core, core_jobs):
        register(mock_core, "live", running=True)
        _job, cancel = core_jobs.occupy(f"{ACQUISITION_LABEL_PREFIX}live")
        asyncio.run(mock_core.cancel_acquisition("live"))
        assert cancel.is_set()

    def test_cancelling_a_finished_run_does_not_stop_the_one_that_is_going(self, mock_core, core_jobs):
        """
        the bug, exactly

        Run A has finished but its bookkeeping still says otherwise; run B is on the
        hardware. Cancelling A must refuse, and must leave B alone.
        """
        register(mock_core, "A", running=True)
        register(mock_core, "B", running=True)
        _job, b_cancel = core_jobs.occupy(f"{ACQUISITION_LABEL_PREFIX}B")

        with pytest.raises(HTTPException) as caught:
            asyncio.run(mock_core.cancel_acquisition("A"))
        assert "not running" in str(caught.value.detail)
        assert not b_cancel.is_set(), "cancelling A stopped B"

    def test_cancelling_a_run_does_not_stop_an_ordinary_command(self, mock_core, core_jobs):
        """a move is not an acquisition, whatever the acquisition map believes"""
        register(mock_core, "A", running=True)
        _job, move_cancel = core_jobs.occupy("MoveTo")

        with pytest.raises(HTTPException):
            asyncio.run(mock_core.cancel_acquisition("A"))
        assert not move_cancel.is_set(), "cancelling an acquisition stopped a move"


class TestTheCancelHatchIsDeliberatelyIndiscriminate:
    """
    `cancel_current` is the other one, and it is meant to stop whatever is running.

    It exists to get the hardware to stop, so it names nothing and refuses nothing. Keeping
    the two apart is the point: one is aimed, the other is a big red button.
    """

    def test_it_stops_whatever_is_in_flight(self, mock_core, core_jobs):
        _job, cancel = core_jobs.occupy("MoveTo")
        asyncio.run(mock_core.cancel_current_command())
        assert cancel.is_set()

    def test_it_is_a_noop_when_nothing_runs(self, mock_core):
        assert mock_core.worker.current_job() is None
        asyncio.run(mock_core.cancel_current_command())  # must not raise
