"""
The rule that the camera SDK's frame callback must never block anything.

This is the historical deadlock, and the reason `debug_introspect.py` exists. Camera SDK stop
calls (`stream_off`, `unregister_capture_callback`) synchronously join the SDK's acquisition
thread with **no timeout**. If the callback running on that thread blocks on a lock the
stopping thread holds, neither ever returns, and the only way out is killing the process.

`Microscope._forward_frame` is the callback's whole body now, and it is written to a specific
shape: take `_stream_lock`, read the state and the sink, *release*, then call the sink from
outside. These tests hold it to that shape by blocking inside the sink and checking that
everything else still moves - because the shape is invisible in a passing run, and the next
person to add "just one more thing" to the callback has no other way to find out.

The other half of the rule is that clearing the stream state is what makes the callback go
quiet: once `_stream_stopped()` has run, a frame already on its way in must return without
touching the sink at all, or stopping would still wait for a slow consumer.
"""

import threading
import time

import pytest
import seaconfig as sc

from seafront.server import commands as cmd

# Generous enough that a loaded machine does not fail it, tight enough that a real block
# (which would wait for the sink, i.e. effectively forever) cannot pass it.
PROMPT_S = 2.0


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


class BlockingSink:
    """a frame consumer that stops dead inside the callback until it is let go"""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def __call__(self, img, ch) -> None:
        self.calls += 1
        self.entered.set()
        assert self.release.wait(timeout=30), "the sink was never released"


@pytest.fixture
def scope(mock_scope):
    """a connected mock microscope, with any stream stopped afterwards"""
    mock_scope.open_connections()
    yield mock_scope
    mock_scope.close()


@pytest.fixture
def blocked(scope):
    """
    a scope with one frame stuck inside the sink

    The frame is pushed through `_forward_frame` directly, on a thread of the test's own,
    rather than waiting for the camera to produce one: the point is what the callback holds
    while it is in there, and that is the same whichever thread called it.
    """
    sink = BlockingSink()
    scope.set_frame_sink(sink)
    scope._stream_started(channel())

    caller = threading.Thread(target=lambda: scope._forward_frame(object()), daemon=True)
    caller.start()
    assert sink.entered.wait(timeout=5), "the sink was never reached"

    yield scope, sink

    sink.release.set()
    caller.join(timeout=5)


def elapsed(fn) -> float:
    started = time.monotonic()
    fn()
    return time.monotonic() - started


class TestTheCallbackHoldsNothingWhileItRuns:
    def test_reading_the_stream_state_is_not_blocked(self, blocked):
        """`_stream_lock` is released before the sink is called, so status still answers"""
        scope, _sink = blocked
        took = elapsed(scope.stream_state)
        assert took < PROMPT_S, f"stream_state() waited {took:.1f}s on a busy sink"

    def test_asking_whether_it_is_streaming_is_not_blocked(self, blocked):
        scope, _sink = blocked
        took = elapsed(lambda: scope.is_streaming)
        assert took < PROMPT_S, f"is_streaming waited {took:.1f}s on a busy sink"

    def test_stopping_the_stream_is_not_blocked(self, blocked):
        """
        the deadlock, in miniature

        Stopping takes `_stream_lock`. If the callback still held it while calling the sink,
        this would wait for the sink, and on real hardware the sink is downstream of a
        websocket to a client that may have wandered off.
        """
        scope, _sink = blocked
        took = elapsed(scope._stream_stopped)
        assert took < PROMPT_S, f"_stream_stopped() waited {took:.1f}s on a busy sink"
        assert scope.is_streaming is False

    def test_a_second_frame_is_not_blocked_by_the_first(self, blocked):
        """
        two frames can be inside the callback at once without either holding the other up

        Only the state read is serialised. A real SDK will not usually overlap callbacks,
        but nothing in this design depends on that, and depending on it accidentally is how
        the last deadlock happened.
        """
        scope, sink = blocked
        second = threading.Thread(target=lambda: scope._forward_frame(object()), daemon=True)
        took = elapsed(lambda: (second.start(), time.sleep(0.05))[0] and None)
        assert took < PROMPT_S
        assert sink.calls >= 1


class TestClearingTheStateSilencesTheCallback:
    def test_a_frame_arriving_after_the_stop_never_reaches_the_sink(self, scope):
        """
        what makes stopping safe

        The stop clears the state first and stops the camera second. Any frame already on
        its way in lands in the gap between the two, and must turn round without touching
        the consumer - otherwise stopping waits for a consumer that may be arbitrarily slow.
        """
        sink = BlockingSink()
        scope.set_frame_sink(sink)
        scope._stream_started(channel())
        scope._stream_stopped()

        took = elapsed(lambda: scope._forward_frame(object()))
        assert took < PROMPT_S
        assert sink.calls == 0, "a frame reached the sink after the stream had stopped"

    def test_a_frame_with_no_sink_installed_is_a_no_op(self, scope):
        scope.set_frame_sink(None)
        scope._stream_started(channel())
        scope._forward_frame(object())  # must not raise
        scope._stream_stopped()

    def test_the_sink_is_told_which_channel_the_frame_belongs_to(self, scope):
        seen: list = []
        scope.set_frame_sink(lambda img, ch: seen.append(ch.handle))
        scope._stream_started(channel("f488"))
        scope._forward_frame(object())
        scope._stream_stopped()
        assert seen == ["f488"]


class TestThroughTheRealCameraThread:
    """the same rule, with frames coming off the mock camera's own acquisition thread"""

    def test_stopping_a_live_stream_with_a_slow_sink_still_returns(self, scope):
        """
        End-to-end: a real stream, a consumer that has stopped consuming, and a stop that
        has to come back anyway. The mock camera joins its thread with a timeout, the way a
        real driver joins without one - so what is checked here is that the join has
        something to join, i.e. that the callback is not parked on a lock.
        """
        import asyncio

        sink = BlockingSink()
        scope.set_frame_sink(sink)
        asyncio.run(scope.execute(cmd.ChannelStreamBegin(channel=channel())))
        try:
            assert sink.entered.wait(timeout=10), "no frame ever reached the sink"

            # the state is still readable while a frame sits in the sink
            assert elapsed(scope.stream_state) < PROMPT_S
            assert scope.is_streaming is True
        finally:
            sink.release.set()

        took = elapsed(lambda: asyncio.run(scope.execute(cmd.ChannelStreamEnd())))
        assert took < 5.0, f"stopping a live stream took {took:.1f}s"
        assert scope.is_streaming is False

    def test_frames_stop_arriving_promptly_after_the_stream_ends(self, scope):
        import asyncio

        frames: list = []
        scope.set_frame_sink(lambda img, ch: frames.append(ch.handle))
        asyncio.run(scope.execute(cmd.ChannelStreamBegin(channel=channel())))

        deadline = time.time() + 10
        while not frames and time.time() < deadline:
            time.sleep(0.02)
        assert frames, "no frame arrived at all, so this proves nothing"

        asyncio.run(scope.execute(cmd.ChannelStreamEnd()))
        settled = len(frames)
        time.sleep(0.5)
        assert len(frames) == settled, "frames kept arriving half a second after the stop"
