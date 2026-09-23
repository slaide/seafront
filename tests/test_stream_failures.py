"""
A stream that ends before it is stopped.

Frames arrive on the camera SDK's thread, so a failure there (a frame that cannot be read,
the next frame that cannot be triggered, the frame consumer raising) has nobody to raise to.
What the camera owes instead:

- the failure is logged when it happens
- stop_streaming, after it has stopped, raises it; a second stop does not raise it again
- a stream that is already running is not started again: that is an error, not a warning
"""

import asyncio
import threading

import numpy as np
import pytest
import seaconfig as sc

from seafront.hardware.camera.camera import StreamEndedEarlyError
from seafront.hardware.camera.mock_camera import MockCamera
from seafront.server import commands as cmd


def channel() -> sc.AcquisitionChannelConfig:
    return sc.AcquisitionChannelConfig(
        name="c", handle="bfledfull", illum_perc=20.0, exposure_time_ms=1.0, analog_gain=0.0,
        z_offset_um=0.0, num_z_planes=1, delta_z_um=1.0, filter_handle=None, enabled=True,
    )


@pytest.fixture
def camera(mock_microscope_env):
    camera = MockCamera()
    camera.open("main")
    yield camera
    camera._streaming_active = False
    camera._streaming_stop_event.set()


def failing_consumer():
    failed = threading.Event()

    def consume(_img: np.ndarray) -> None:
        failed.set()
        raise RuntimeError("frame store is full")

    return consume, failed


def test_starting_a_running_stream_again_is_refused(camera):
    camera.start_streaming(channel(), callback=lambda _img: None)
    with pytest.raises(RuntimeError, match="already streaming"):
        camera.start_streaming(channel(), callback=lambda _img: None)
    camera.stop_streaming()


def test_a_stream_that_ended_early_is_reported_by_stop(camera):
    consume, failed = failing_consumer()
    camera.start_streaming(channel(), callback=consume)
    assert failed.wait(timeout=5)

    with pytest.raises(StreamEndedEarlyError, match="frame store is full"):
        camera.stop_streaming()
    camera.stop_streaming()


def test_a_new_stream_starts_clean(camera):
    consume, failed = failing_consumer()
    camera.start_streaming(channel(), callback=consume)
    assert failed.wait(timeout=5)
    with pytest.raises(StreamEndedEarlyError):
        camera.stop_streaming()

    camera.start_streaming(channel(), callback=lambda _img: None)
    camera.stop_streaming()


def test_the_adapter_reports_it_and_still_turns_the_light_off(squid_adapter):
    adapter, device = squid_adapter
    from seafront.hardware.microcontroller.teensy_microcontroller import CommandName

    failed = threading.Event()

    def sink(_img, _channel):
        failed.set()
        raise RuntimeError("frame store is full")

    adapter.set_frame_sink(sink)
    asyncio.run(adapter.execute(cmd.ChannelStreamBegin(channel=channel())))
    assert failed.wait(timeout=5)

    device.commands_received.clear()
    with pytest.raises(Exception, match="frame store is full"):
        asyncio.run(adapter.execute(cmd.ChannelStreamEnd()))
    assert any(p[1] == CommandName.TURN_OFF_ILLUMINATION.value for p in device.commands_received)
    assert not adapter.is_streaming


# --- the drivers, over stand-in SDK handles ---------------------------------------------------


def _galaxy(monkeypatch):
    from seafront.hardware.camera.galaxy_camera import GalaxyCamera

    camera = GalaxyCamera({"vendor_name": "test", "model_name": "test", "sn": "GX1"})
    camera.handle = object()
    camera.device_type = "main"
    camera._streaming_active = False
    camera._incomplete_stream_frames = 0
    installed: list = []
    monkeypatch.setattr(camera, "_set_pixel_format", lambda *a: None)
    monkeypatch.setattr(camera, "_set_exposure_time", lambda *a: None)
    monkeypatch.setattr(camera, "_set_analog_gain", lambda *a: None)
    monkeypatch.setattr(camera, "_clear_acquisition_state", lambda: None)
    monkeypatch.setattr(
        camera, "_set_acquisition_mode", lambda mode, with_cb=None: installed.append(with_cb)
    )
    return camera, installed


class _RawImage:
    def __init__(self, status):
        self.status = status

    def get_status(self):
        return self.status

    def get_numpy_array(self):
        return np.zeros((2, 2), dtype=np.uint8)


def test_galaxy_refuses_a_second_start(monkeypatch):
    camera, _installed = _galaxy(monkeypatch)
    camera.start_streaming(channel(), callback=lambda _img: None)
    with pytest.raises(RuntimeError, match="already streaming"):
        camera.start_streaming(channel(), callback=lambda _img: None)


def test_galaxy_reports_a_consumer_failure_at_stop_and_drops_later_frames(monkeypatch):
    from gxipy import gxiapi

    camera, installed = _galaxy(monkeypatch)
    calls: list[int] = []

    def consume(_img):
        calls.append(1)
        raise RuntimeError("frame store is full")

    camera.start_streaming(channel(), callback=consume)
    frame_arrived = installed[0]
    frame_arrived(_RawImage(gxiapi.GxFrameStatusList.SUCCESS))
    frame_arrived(_RawImage(gxiapi.GxFrameStatusList.SUCCESS))
    assert calls == [1]

    with pytest.raises(StreamEndedEarlyError, match="frame store is full"):
        camera.stop_streaming()


def test_galaxy_skips_incomplete_frames_without_ending_the_stream(monkeypatch):
    from gxipy import gxiapi

    camera, installed = _galaxy(monkeypatch)
    received: list[np.ndarray] = []
    camera.start_streaming(channel(), callback=received.append)
    installed[0](_RawImage(gxiapi.GxFrameStatusList.INCOMPLETE))
    installed[0](_RawImage(gxiapi.GxFrameStatusList.SUCCESS))
    assert len(received) == 1
    camera.stop_streaming()


def test_galaxy_a_failed_start_leaves_it_not_streaming(monkeypatch):
    camera, _installed = _galaxy(monkeypatch)

    def refuse(*_args):
        raise RuntimeError("exposure refused")

    monkeypatch.setattr(camera, "_set_exposure_time", refuse)
    with pytest.raises(RuntimeError, match="exposure refused"):
        camera.start_streaming(channel(), callback=lambda _img: None)
    assert camera._streaming_active is False


def _toupcam(monkeypatch, trigger_fails: bool):
    import types

    from seafront.hardware.camera.toupcam_camera import ToupCamCamera

    camera = ToupCamCamera(types.SimpleNamespace(displayname="test", id="TP1"))  # type: ignore[arg-type]
    camera.device_type = "main"

    def trigger(n):
        if trigger_fails and n == 1:
            raise RuntimeError("trigger refused")

    camera.handle = types.SimpleNamespace(  # type: ignore[assignment]
        get_ExpoTime=lambda: 1000, get_RealExpoTime=lambda: 1000, get_ExpoAGain=lambda: 100,
        Flush=lambda: None, Trigger=trigger,
    )
    monkeypatch.setattr(camera, "_set_pixel_format", lambda *a: (1, np.uint8, 8))
    monkeypatch.setattr(camera, "_set_exposure_time", lambda *a: None)
    monkeypatch.setattr(camera, "_set_analog_gain", lambda *a: None)
    monkeypatch.setattr(camera, "_api_trigger", lambda *a: None)
    return camera


def test_toupcam_refuses_a_second_start(monkeypatch):
    camera = _toupcam(monkeypatch, trigger_fails=False)
    camera.start_streaming(channel(), callback=lambda _img: None)
    with pytest.raises(RuntimeError, match="already streaming"):
        camera.start_streaming(channel(), callback=lambda _img: None)


def test_toupcam_reports_a_failed_next_trigger_at_stop(monkeypatch):
    camera = _toupcam(monkeypatch, trigger_fails=True)
    camera.start_streaming(channel(), callback=lambda _img: None)
    assert camera.ctx.callback is not None
    assert camera.ctx.callback(np.zeros((2, 2), dtype=np.uint8)) is True, "the stream went on"

    with pytest.raises(StreamEndedEarlyError, match="trigger refused"):
        camera.stop_streaming()


def test_toupcam_a_failed_start_leaves_it_not_streaming(monkeypatch):
    camera = _toupcam(monkeypatch, trigger_fails=False)

    def refuse(*_args):
        raise RuntimeError("exposure refused")

    monkeypatch.setattr(camera, "_set_exposure_time", refuse)
    with pytest.raises(RuntimeError, match="exposure refused"):
        camera.start_streaming(channel(), callback=lambda _img: None)
    assert camera._streaming_active is False
