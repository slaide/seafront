"""
How long the server remembers an acquisition.

Each status carries the whole protocol it ran, so keeping every one for as long as the server
is up is a leak on a machine that runs for months. The running one is always kept, and so are
the most recent finished ones; a start that was refused never ran and is not kept at all.
"""

import asyncio
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).parent))

from test_acquisition_edges import protocol

from seafront.server.acquisitions import AcquisitionRegistry
from seafront.server.commands import AcquisitionStatus, AcquisitionStatusStage


def status(acquisition_id: str) -> AcquisitionStatus:
    return AcquisitionStatus(acquisition_id=acquisition_id)


def test_only_the_most_recent_finished_acquisitions_are_kept():
    registry = AcquisitionRegistry(keep_finished=2)
    for name in ["a", "b", "c"]:
        registry.add(status(name))
        registry.finish(name)

    assert registry.get("a") is None
    assert registry.get("b") is not None
    assert registry.get("c") is not None


def test_the_running_acquisition_is_never_forgotten():
    registry = AcquisitionRegistry(keep_finished=1)
    registry.add(status("running"))
    for name in ["a", "b", "c"]:
        registry.add(status(name))
        registry.finish(name)

    running = registry.get("running")
    assert running is not None and running.finished_at is None


def test_a_refused_start_is_not_kept():
    registry = AcquisitionRegistry()
    registry.add(status("refused"))
    registry.discard("refused")

    assert registry.get("refused") is None


def test_an_unknown_acquisition_says_why_it_may_be_gone():
    registry = AcquisitionRegistry(keep_finished=7)
    message = registry.describe_unknown("x")
    assert "x" in message and "last 7 finished" in message


@pytest.mark.parametrize(
    "kwargs", [{"wells": ()}, {"project": "has space"}, {"plate": "a/b"}]
)
def test_a_start_refused_in_checking_leaves_nothing_behind(mock_core, kwargs):
    """
    refused before it reaches the worker. These used to be registered first, and then
    report "scheduled" for as long as the server ran.
    """
    with pytest.raises(HTTPException):
        asyncio.run(mock_core.start_acquisition(config_file=protocol(**kwargs)))

    assert mock_core.acquisitions._statuses == {}


def test_an_unexpected_failure_ends_the_run_crashed_with_its_reason(mock_core, monkeypatch):
    """
    it used to be marked crashed with whatever message the last progress update left, so the
    status said "crashed" and nothing about why
    """

    async def broken(self, command):
        raise RuntimeError("the specific thing that broke")

    monkeypatch.setattr(type(mock_core.microscope), "execute", broken)
    started = asyncio.run(mock_core.start_acquisition(config_file=protocol()))
    assert mock_core.acquisition_future is not None
    mock_core.acquisition_future.result(timeout=30)

    record = mock_core.acquisitions.get(started.acquisition_id)
    assert record is not None and record.finished_at is not None
    last = asyncio.run(mock_core.get_acquisition_status(started.acquisition_id))
    assert last.acquisition_status == AcquisitionStatusStage.CRASHED
    assert "the specific thing that broke" in last.message
    assert "the specific thing that broke" in (mock_core._last_acquisition_error or "")


def test_asking_about_a_forgotten_acquisition_says_why(mock_core):
    with pytest.raises(HTTPException) as caught:
        asyncio.run(mock_core.get_acquisition_status("long-gone"))
    assert "last 32 finished" in str(caught.value.detail)
