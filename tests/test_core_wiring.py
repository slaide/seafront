"""
What the Core wires together, checked without a server in the way.

The HTTP tests prove the contract a client sees. These prove the joins underneath it: that the
frame sink is installed at construction rather than around each stream, that a streamed frame
reaches the image store carrying the channel it was taken in, and that status answers from the
adapter rather than from bookkeeping the Core keeps on the side.

They run against a real `Core` over a mock microscope in this process, so they cost
milliseconds instead of a subprocess boot, and they fail at the seam rather than three layers
away at an HTTP status code.
"""

import asyncio
import threading
import time

import pytest
import seaconfig as sc
from fastapi import HTTPException

from seafront.server import commands as cmd


def channel(handle: str = "bfledfull") -> sc.AcquisitionChannelConfig:
    return sc.AcquisitionChannelConfig(
        name=f"channel {handle}",
        handle=handle,
        illum_perc=20.0,
        exposure_time_ms=5.0,
        analog_gain=0.0,
        z_offset_um=0.0,
        num_z_planes=1,
        delta_z_um=1.0,
        filter_handle=None,
        enabled=True,
    )


def run(core, command):
    return asyncio.run(core.run(command))


def state(core):
    return asyncio.run(core.get_current_state())


@pytest.fixture
def connected(mock_core):
    """a Core whose hardware is open, left tidy afterwards"""
    run(mock_core, cmd.EstablishHardwareConnection())
    yield mock_core
    if mock_core.microscope.is_streaming:
        run(mock_core, cmd.ChannelStreamEnd())


def wait_for(predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


class TestTheFrameSinkIsWiredAtConstruction:
    def test_a_fresh_core_already_has_a_sink(self, mock_core):
        """
        no route installs it

        The old shape had the route hand the Core's handler to the adapter *after* the
        begin command had already started the camera, so the frames in between went
        nowhere. Nothing installs a sink per stream any more, so there is nothing to be
        late.
        """
        assert mock_core.microscope._frame_sink is not None
        assert mock_core.microscope.is_streaming is False

    def test_streamed_frames_land_in_the_image_store(self, connected):
        assert connected.latest_images == {}
        run(connected, cmd.ChannelStreamBegin(channel=channel()))

        assert wait_for(lambda: "bfledfull" in connected.latest_images), (
            "no streamed frame reached the image store within 5s"
        )
        stored = connected.latest_images["bfledfull"]
        assert stored.info.channel.handle == "bfledfull"
        assert stored._img is not None

    def test_a_frame_is_filed_under_the_channel_it_was_taken_in(self, connected):
        """
        the channel travels with the frame

        It used to be read from a field the Core kept beside the adapter, which meant a
        frame in flight across a stream change could be filed under the wrong channel.
        """
        run(connected, cmd.ChannelStreamBegin(channel=channel("f488")))
        assert wait_for(lambda: "f488" in connected.latest_images)
        run(connected, cmd.ChannelStreamEnd())

        assert "bfledfull" not in connected.latest_images
        assert connected.latest_images["f488"].info.channel.handle == "f488"


class TestStatusAnswersFromTheAdapter:
    def test_idle_status_reports_no_stream(self, connected):
        reported = state(connected)
        assert reported.is_streaming is False
        assert reported.streaming_channel_handle is None

    def test_streaming_status_reports_the_channel(self, connected):
        run(connected, cmd.ChannelStreamBegin(channel=channel("f488")))
        reported = state(connected)
        assert reported.is_streaming is True
        assert reported.streaming_channel_handle == "f488"

    def test_status_and_adapter_never_disagree_across_a_cycle(self, connected):
        """
        the pair comes from one locked read, so there is no order to get wrong

        Both halves used to be set a moment apart - the adapter's during the command, the
        Core's in the route callback after it - which is what let a status read catch a
        stream with no channel.
        """
        for _ in range(25):
            run(connected, cmd.ChannelStreamBegin(channel=channel()))
            reported = state(connected)
            assert reported.is_streaming == (reported.streaming_channel_handle is not None)
            assert reported.is_streaming is connected.microscope.is_streaming

            run(connected, cmd.ChannelStreamEnd())
            reported = state(connected)
            assert reported.is_streaming == (reported.streaming_channel_handle is not None)
            assert reported.is_streaming is connected.microscope.is_streaming

    def test_status_carries_the_lease(self, connected):
        assert state(connected).lock.held is False
        connected.lock.acquire("alice")
        reported = state(connected)
        assert reported.lock.held is True and reported.lock.owner == "alice"


class TestTheLeaseKnowsWhatTheDeviceIsDoing:
    """
    Core is what tells the lease "in use", and it has to mean both kinds of use.

    The lease object's own rules are covered in tests/test_device_lock.py against a fake
    predicate. What is checked here is the predicate Core actually supplies - a stream leaves
    the worker idle, so a lease watching only the worker would lapse under a running camera.
    """

    def test_an_idle_microscope_is_not_in_use(self, connected):
        assert connected.worker.is_busy is False
        assert connected.microscope.is_streaming is False
        assert connected.lock._device_in_use() is False

    def test_a_running_stream_counts_as_in_use(self, connected):
        run(connected, cmd.ChannelStreamBegin(channel=channel()))
        assert connected.worker.is_busy is False, "a stream is meant to give the worker back"
        assert connected.lock._device_in_use() is True

    def test_and_stops_counting_once_the_stream_does(self, connected):
        run(connected, cmd.ChannelStreamBegin(channel=channel()))
        run(connected, cmd.ChannelStreamEnd())
        assert connected.lock._device_in_use() is False


class TestConnectingIsSerialisedLikeEverythingElse:
    """
    Ungating the connect route did not take it off the worker.

    It no longer needs the device lease, but it is still one command at a time: it cannot
    interleave with a move, and it cannot run twice at once.
    """

    def test_connecting_is_idempotent(self, connected):
        assert connected.microscope.is_connected is True
        run(connected, cmd.EstablishHardwareConnection())
        assert connected.microscope.is_connected is True

    def test_it_goes_through_the_worker(self, connected):
        """a second command while one is in flight is refused, connect included"""
        release = threading.Event()

        async def occupy() -> None:
            while not release.is_set():
                await asyncio.sleep(0.02)

        claimed = connected.worker.submit_job(occupy, "occupier")
        assert claimed is not None, "could not take the worker slot to test against"
        try:
            assert connected.worker.is_busy is True
            with pytest.raises(HTTPException) as caught:
                run(connected, cmd.EstablishHardwareConnection())
            assert "busy" in str(caught.value).lower()
        finally:
            release.set()
            claimed[0].result(timeout=10)
