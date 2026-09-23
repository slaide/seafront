"""
Closing hardware, and losing it.

What a correct teardown owes its caller:

- every device is attempted, whatever happened to the ones before it
- every device is released, even the ones that failed to close - the driver lets go of the
  handle first, so a failed close cannot leave it holding one that blocks the next open
- and whatever failed is *raised*, after all of that, naming the device and the step - not
  swallowed, and not merely logged

Losing a device mid-command is the same teardown with one more obligation: the error that
comes out says which device went, and the health board records it as lost.
"""

import asyncio
import sys
import types
from pathlib import Path

import pytest
import seaconfig as sc

sys.path.insert(0, str(Path(__file__).parent))

from seafront.hardware.camera.camera import CameraOfflineError
from seafront.hardware.camera.mock_camera import MockCamera
from seafront.hardware.health import (
    Condition,
    Device,
    DeviceCloseError,
    HardwareCloseError,
)
from seafront.hardware.microcontroller.teensy_microcontroller import CommandName
from seafront.hardware.microscope import DisconnectError
from seafront.server import commands as cmd


class FailingClose(MockCamera):
    "a camera whose close fails, after releasing, the way the drivers' closes do"

    def __init__(self, label: str):
        super().__init__()
        self.sn = label
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        raise DeviceCloseError(f"camera {self.sn}", [("close_device", RuntimeError("usb stall"))])


def run(adapter, command):
    return asyncio.run(adapter.execute(command))


# --- the adapter's close -------------------------------------------------------------------


def test_close_attempts_every_device_and_raises_what_failed(squid_adapter):
    adapter, device = squid_adapter
    main = FailingClose("main-sn")
    adapter.main_camera.t = main

    with pytest.raises(HardwareCloseError) as caught:
        adapter.close()

    # the main camera failing did not stop the others being closed
    assert main.close_calls == 1
    assert adapter.microcontroller.handle is None
    assert set(caught.value.failures) == {Device.MAIN_CAMERA}
    assert "usb stall" in str(caught.value)
    assert adapter.is_connected is False


def test_close_records_each_device_on_the_health_board(squid_adapter):
    adapter, _ = squid_adapter
    adapter.main_camera.t = FailingClose("main-sn")

    with pytest.raises(HardwareCloseError):
        adapter.close()

    health = {entry.device: entry for entry in adapter.health()}
    assert health[Device.MAIN_CAMERA].condition == Condition.NOT_CONNECTED
    assert health[Device.MAIN_CAMERA].last_fault is not None
    assert health[Device.MAIN_CAMERA].last_fault.operation == "close"
    assert "usb stall" in health[Device.MAIN_CAMERA].last_fault.message
    assert health[Device.FOCUS_CAMERA].last_fault is None
    assert health[Device.MICROCONTROLLER].last_fault is None


def test_close_does_not_skip_devices_when_the_adapter_never_finished_connecting(squid_adapter):
    """
    A partial open leaves some devices open while the adapter as a whole is not connected.
    Closing must still release those, or the next start finds them "in use".
    """
    adapter, _ = squid_adapter
    adapter.is_connected = False
    main = FailingClose("main-sn")
    adapter.main_camera.t = main

    with pytest.raises(HardwareCloseError):
        adapter.close()
    assert main.close_calls == 1
    assert adapter.microcontroller.handle is None


def test_close_twice_is_fine(squid_adapter):
    adapter, _ = squid_adapter
    adapter.close()
    adapter.close()


# --- losing a device -----------------------------------------------------------------------


def test_microcontroller_io_error_drops_the_hardware_and_names_the_device(squid_adapter, monkeypatch):
    adapter, _ = squid_adapter

    def broken(*args, **kwargs):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(type(adapter.microcontroller), "send_cmd", broken)

    with pytest.raises(DisconnectError) as caught:
        run(adapter, cmd.IlluminationEndAll())

    assert caught.value.device == Device.MICROCONTROLLER
    assert adapter.is_connected is False
    health = {entry.device: entry for entry in adapter.health()}
    assert health[Device.MICROCONTROLLER].condition == Condition.DISCONNECTED
    assert health[Device.MICROCONTROLLER].last_fault is not None
    assert "Input/output error" in health[Device.MICROCONTROLLER].last_fault.message
    assert health[Device.MICROCONTROLLER].last_fault.operation == "IlluminationEndAll"


def test_a_failed_close_while_dropping_is_attached_to_the_disconnect(squid_adapter, monkeypatch):
    adapter, _ = squid_adapter
    adapter.main_camera.t = FailingClose("main-sn")

    def broken(*args, **kwargs):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(type(adapter.microcontroller), "send_cmd", broken)

    with pytest.raises(DisconnectError) as caught:
        run(adapter, cmd.IlluminationEndAll())

    notes = "\n".join(caught.value.__notes__)
    assert "Input/output error" in notes
    assert "usb stall" in notes


def test_camera_offline_drops_the_hardware_and_names_that_camera(squid_adapter, monkeypatch):
    adapter, _ = squid_adapter
    focus = adapter.focus_camera.value

    def offline(*args, **kwargs):
        raise CameraOfflineError(focus)

    monkeypatch.setattr(focus, "get_exposure_time_limits", offline)
    monkeypatch.setattr(adapter, "_dispatch", lambda qmc, command: focus.get_exposure_time_limits())

    with pytest.raises(DisconnectError) as caught:
        run(adapter, cmd.IlluminationEndAll())

    assert caught.value.device == Device.FOCUS_CAMERA
    assert adapter.is_connected is False
    health = {entry.device: entry for entry in adapter.health()}
    assert health[Device.FOCUS_CAMERA].condition == Condition.DISCONNECTED
    assert health[Device.MAIN_CAMERA].condition == Condition.NOT_CONNECTED


# --- stopping a stream ---------------------------------------------------------------------


def _start_stream(adapter):
    channel = sc.AcquisitionChannelConfig(
        name="Brightfield LED matrix full",
        handle="bfledfull",
        illum_perc=10.0,
        exposure_time_ms=5.0,
        analog_gain=0.0,
        z_offset_um=0.0,
        num_z_planes=1,
        delta_z_um=1.0,
        filter_handle=None,
        enabled=True,
    )
    run(adapter, cmd.ChannelStreamBegin(channel=channel))


def _illumination_turned_off(device) -> bool:
    return any(packet[1] == CommandName.TURN_OFF_ILLUMINATION.value for packet in device.commands_received)


def test_stream_end_turns_the_light_off_even_when_the_camera_will_not_stop(squid_adapter, monkeypatch):
    adapter, device = squid_adapter
    _start_stream(adapter)
    device.commands_received.clear()

    def stuck():
        raise RuntimeError("camera refused to leave continuous mode")

    monkeypatch.setattr(adapter.main_camera.value, "stop_streaming", stuck)

    with pytest.raises(RuntimeError, match="continuous mode"):
        run(adapter, cmd.ChannelStreamEnd())

    assert _illumination_turned_off(device)


def test_stream_end_turns_all_light_off_when_the_streamed_channel_is_gone(squid_adapter):
    """
    The channel list changed under a running stream. Which source it lit is unknown, so all
    of them go off - rather than none.
    """
    adapter, device = squid_adapter
    _start_stream(adapter)
    device.commands_received.clear()
    adapter.channels = [ch for ch in adapter.channels if ch.handle != "bfledfull"]

    run(adapter, cmd.ChannelStreamEnd())

    assert _illumination_turned_off(device)


def test_illumination_end_all_tries_every_source_and_fails_if_one_failed(squid_adapter, monkeypatch):
    adapter, _ = squid_adapter
    attempted: list[int] = []

    def end(self, source=None):
        attempted.append(source)
        if len(attempted) == 1:
            raise ValueError("first source refused")

    monkeypatch.setattr(type(adapter.microcontroller), "illumination_end", end)

    with pytest.raises(ValueError, match="first source refused"):
        run(adapter, cmd.IlluminationEndAll())
    assert len(attempted) == len(adapter.channels)


# --- the drivers ---------------------------------------------------------------------------


class _Handle:
    def __init__(self, fail: set[str]):
        self.fail = fail
        self.calls: list[str] = []

    def _call(self, name: str):
        self.calls.append(name)
        if name in self.fail:
            raise RuntimeError(f"{name} failed")

    def stream_off(self):
        self._call("stream_off")

    def close_device(self):
        self._call("close_device")

    def Close(self):
        self._call("Close")


def _galaxy(handle: _Handle, streaming: bool):
    from seafront.hardware.camera.galaxy_camera import GalaxyCamera

    camera = GalaxyCamera({"vendor_name": "test", "model_name": "test", "sn": "GX1"})
    camera.handle = handle
    camera.is_streaming = streaming
    return camera


def test_galaxy_close_releases_the_handle_and_reports_every_failed_step():
    handle = _Handle(fail={"stream_off", "close_device"})
    camera = _galaxy(handle, streaming=True)

    with pytest.raises(DeviceCloseError) as caught:
        camera.close()

    assert camera.handle is None
    assert handle.calls == ["stream_off", "close_device"]
    assert [step for step, _ in caught.value.failures] == ["stream_off", "close_device"]


def test_galaxy_close_does_not_reopen_the_camera_it_is_closing(monkeypatch):
    """
    Stream-off used to go through the reconnecting wrapper, which answers a failure by closing
    and reopening - from inside close, that reopened the camera being closed.
    """
    handle = _Handle(fail={"stream_off"})
    camera = _galaxy(handle, streaming=True)
    opened: list[str] = []
    monkeypatch.setattr(camera, "open", lambda device_type: opened.append(device_type))

    with pytest.raises(DeviceCloseError):
        camera.close()
    assert opened == []
    assert handle.calls == ["stream_off", "close_device"]


def test_galaxy_close_does_not_stop_a_stream_that_is_not_running():
    handle = _Handle(fail=set())
    camera = _galaxy(handle, streaming=False)
    camera.close()
    assert handle.calls == ["close_device"]


def test_galaxy_close_of_a_closed_camera_is_a_no_op():
    camera = _galaxy(_Handle(fail=set()), streaming=False)
    camera.handle = None
    camera.close()


def test_toupcam_close_releases_the_handle_and_reports_the_failure():
    from seafront.hardware.camera.toupcam_camera import ToupCamCamera

    camera = ToupCamCamera(types.SimpleNamespace(displayname="test", id="TP1"))  # type: ignore[arg-type]
    handle = _Handle(fail={"Close"})
    camera.handle = handle  # type: ignore[assignment]

    with pytest.raises(DeviceCloseError, match="Close failed"):
        camera.close()
    assert camera.handle is None


def test_microcontroller_close_releases_the_port_even_when_closing_it_fails(squid_adapter, monkeypatch):
    adapter, _ = squid_adapter
    port = adapter.microcontroller.handle
    assert port is not None

    def broken():
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(port, "close", broken)

    with pytest.raises(DeviceCloseError, match="Input/output error"):
        adapter.microcontroller.close()
    assert adapter.microcontroller.handle is None


# --- shutdown ------------------------------------------------------------------------------


def test_core_close_reports_a_failed_hardware_close_and_still_finishes(mock_core, monkeypatch):
    from seafront.config.basics import GlobalConfigHandler

    def failing_close(self):
        raise HardwareCloseError({Device.MAIN_CAMERA: RuntimeError("usb stall")})

    stored: list[bool] = []
    monkeypatch.setattr(type(mock_core.microscope), "close", failing_close)
    monkeypatch.setattr(GlobalConfigHandler, "store", lambda: stored.append(True))

    assert mock_core.close() is False
    assert stored == [True]


def test_core_close_reports_a_clean_shutdown(mock_core):
    assert mock_core.close() is True


# --- the http boundary ---------------------------------------------------------------------


def test_an_unexpected_error_reaches_the_client_with_its_message():
    from fastapi.testclient import TestClient

    from seafront.server.app import make_app

    app = make_app()

    @app.get("/test-boom")
    async def boom():
        error = RuntimeError("the specific thing that broke")
        error.add_note("while doing the other thing")
        raise error

    response = TestClient(app, raise_server_exceptions=False).get("/test-boom")
    assert response.status_code == 500
    assert response.json() == {
        "detail": "RuntimeError: the specific thing that broke; while doing the other thing"
    }
