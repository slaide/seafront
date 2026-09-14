"""
Who owns "the camera is streaming".

One answer, in the adapter, set and cleared inside the ChannelStreamBegin/End commands. The
old shape had it in four places at once - a callback the Core installed after the command but
the adapter cleared during it, a channel the route tracked on the side, the camera's own
state, and a second copy inside the mock - which is how a status read could see a stream with
no channel, how the begin side dropped every frame between the camera starting and the
callback being installed, and how ending a stream needed the client to remember which channel
it had started.

These tests pin the replacement: `stream_state()` is the truth, the frame sink is installed
once and gated by that state, and ending stops whatever is streaming without being told what
that is. tests/test_lock_enforcement.py covers the device lease over the same ground.
"""

import asyncio
import threading
import time

import pytest
import seaconfig as sc
from fastapi import HTTPException
from pydantic import ValidationError

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


def begin(scope, ch: sc.AcquisitionChannelConfig | None = None):
    return asyncio.run(scope.execute(cmd.ChannelStreamBegin(channel=ch or channel())))


def end(scope):
    return asyncio.run(scope.execute(cmd.ChannelStreamEnd()))


@pytest.fixture
def scope(mock_scope):
    """a connected mock microscope; `mock_scope` hands one back with its camera still shut"""
    mock_scope.open_connections()
    yield mock_scope
    mock_scope.close()


@pytest.fixture
def streaming(scope):
    """a scope that is streaming, stopped again however the test leaves it"""
    begin(scope)
    yield scope
    if scope.is_streaming:
        end(scope)


class TestTheAdapterIsTheSourceOfTruth:
    def test_nothing_streams_to_begin_with(self, scope):
        assert scope.stream_state() is None
        assert scope.is_streaming is False

    def test_begin_records_the_channel(self, scope):
        begin(scope, channel("f488"))
        try:
            state = scope.stream_state()
            assert state is not None
            assert state.channel.handle == "f488"
            assert scope.is_streaming is True
        finally:
            end(scope)

    def test_end_clears_it(self, streaming):
        end(streaming)
        assert streaming.stream_state() is None
        assert streaming.is_streaming is False

    def test_the_state_is_a_snapshot_that_cannot_be_edited(self, streaming):
        state = streaming.stream_state()
        assert state is not None
        with pytest.raises(ValidationError):
            state.channel = channel("f488")  # type: ignore[misc]

    def test_close_stops_the_stream(self, streaming):
        """a disconnect must not leave the adapter claiming to stream"""
        streaming.close()
        assert streaming.stream_state() is None
        assert streaming.is_streaming is False


class TestEndingDoesNotNeedToBeTold:
    """the adapter lit the channel, so the adapter knows which one to turn off"""

    def test_end_takes_no_channel(self, streaming):
        """ChannelStreamEnd carries nothing: a client cannot name the wrong stream"""
        assert "channel" not in cmd.ChannelStreamEnd.model_fields

    def test_end_stops_whatever_is_streaming(self, scope):
        begin(scope, channel("f488"))
        end(scope)
        assert scope.is_streaming is False

    def test_ending_when_nothing_streams_is_refused(self, scope):
        with pytest.raises(HTTPException) as caught:
            end(scope)
        assert "not currently streaming" in str(caught.value).lower()

    def test_beginning_twice_is_refused(self, streaming):
        with pytest.raises(HTTPException) as caught:
            begin(streaming)
        assert "already" in str(caught.value).lower()
        assert streaming.is_streaming is True, "the refused second begin must not disturb the first"


class TestTheFrameSink:
    """
    installed once, at construction, and gated by the stream state

    The old shape installed the sink *after* the camera had started, so every frame in
    between was dropped on the floor. Registering it once means the only question at frame
    time is whether a stream is running.
    """

    def collect(self, scope) -> list:
        frames: list = []
        gate = threading.Lock()

        def sink(img, ch):
            with gate:
                frames.append((img, ch))

        scope.set_frame_sink(sink)
        return frames

    def test_frames_reach_the_sink_with_the_channel_they_belong_to(self, scope):
        frames = self.collect(scope)
        begin(scope, channel("f488"))
        try:
            deadline = time.time() + 5
            while not frames and time.time() < deadline:
                time.sleep(0.05)
            assert frames, "no frame reached the sink within 5s of the stream starting"
            _img, ch = frames[0]
            assert ch.handle == "f488"
        finally:
            end(scope)

    def test_no_frames_reach_the_sink_once_the_stream_ends(self, scope):
        frames = self.collect(scope)
        begin(scope)
        deadline = time.time() + 5
        while not frames and time.time() < deadline:
            time.sleep(0.05)
        assert frames, "no frame reached the sink at all, so this proves nothing"
        end(scope)

        time.sleep(0.2)  # let anything already in flight land
        settled = len(frames)
        time.sleep(0.5)
        assert len(frames) == settled, "frames kept arriving after the stream ended"


class TestAFailedBeginLeavesNothingBehind:
    def test_an_unknown_channel_does_not_mark_the_adapter_streaming(self, scope):
        with pytest.raises(HTTPException):
            begin(scope, channel("no-such-channel"))
        assert scope.is_streaming is False
        assert scope.stream_state() is None

    def test_and_a_real_stream_still_starts_afterwards(self, scope):
        with pytest.raises(HTTPException):
            begin(scope, channel("no-such-channel"))
        begin(scope)
        try:
            assert scope.is_streaming is True
        finally:
            end(scope)
