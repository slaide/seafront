"""
What a status read tells the operator about the machine.

- every device's condition and last fault, and any stalled call
- which lights are on; a light counts as on until it has confirmed turning off
- the configured objective, and the main camera's identity and pixel format
- a disconnected stage still gets an answer: the last position it reported, with the
  microcontroller shown as not connected, instead of the whole status read failing
"""

import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_stall_watchdog import channel

from seafront.config.handles import CameraConfig
from seafront.config.registry import ConfigRegistry
from seafront.hardware.health import Device
from seafront.hardware.microscope import DisconnectError
from seafront.hardware.watchdog import Stall
from seafront.server import commands as cmd


def state(core) -> cmd.CoreCurrentState:
    return asyncio.run(core.get_current_state())


# --- through the core ----------------------------------------------------------------------


def test_every_device_is_listed(mock_core):
    devices = {h.device for h in state(mock_core).hardware}
    assert devices == set(mock_core.microscope.hardware_devices())


def test_the_objective_and_camera_are_listed(mock_core):
    ConfigRegistry.set_value(CameraConfig.MAIN_OBJECTIVE, "20xolympus")
    current = state(mock_core)
    assert current.objective == "20xolympus"
    assert current.main_camera.identity.vendor == "Mock"
    assert current.main_camera.identity.serial_number == "MOCK-001"
    assert current.main_camera.pixel_format == ConfigRegistry.get(CameraConfig.MAIN_PIXEL_FORMAT).strvalue


def test_nothing_is_lit_at_rest(mock_core):
    current = state(mock_core)
    assert current.illumination.lit == []
    assert current.illumination.af_laser_on is False


def test_a_stall_is_listed(mock_core, monkeypatch):
    stall = Stall(
        device=Device.MAIN_CAMERA, operation="snap", started_at=time.time() - 40,
        deadline_s=30.0, thread="microscope-worker",
    )
    monkeypatch.setattr(type(mock_core.microscope), "stalls", lambda self: [stall])
    assert state(mock_core).stalls == [stall]


def test_a_disconnected_stage_still_gets_an_answer(mock_core, monkeypatch):
    before = state(mock_core).adapter_state
    assert before is not None

    async def gone(self):
        raise DisconnectError("usb unplugged")

    monkeypatch.setattr(type(mock_core.microscope), "get_current_state", gone)
    current = state(mock_core)
    assert current.adapter_state == before


def test_a_stage_never_read_is_none_rather_than_made_up(mock_core, monkeypatch):
    async def gone(self):
        raise DisconnectError("usb unplugged")

    monkeypatch.setattr(type(mock_core.microscope), "get_current_state", gone)
    mock_core.microscope.last_state = None
    assert state(mock_core).adapter_state is None


# --- the squid adapter's lights --------------------------------------------------------------


def lit_handles(adapter) -> list[str]:
    return [lit.handle for lit in adapter.illumination_state().lit]


def test_a_source_turned_on_is_listed_by_its_channel(squid_adapter):
    adapter, _device = squid_adapter
    adapter.microcontroller.illumination_begin(1, 42.0)
    [lit] = adapter.illumination_state().lit
    assert lit.handle == "f488" and lit.intensity_percent == 42.0


def test_a_source_turned_off_is_not(squid_adapter):
    adapter, _device = squid_adapter
    adapter.microcontroller.illumination_begin(1, 42.0)
    adapter.microcontroller.illumination_end(1)
    assert lit_handles(adapter) == []


def test_turning_everything_off_clears_every_source(squid_adapter):
    adapter, _device = squid_adapter
    adapter.microcontroller.illumination_begin(0, 10.0)
    adapter.microcontroller.illumination_begin(1, 20.0)
    adapter.microcontroller.illumination_end(None)
    assert lit_handles(adapter) == []


def test_a_failed_off_leaves_the_source_counted_as_on(squid_adapter, monkeypatch):
    "the light may still be on, and saying it is off is the side that cannot be acted on"
    adapter, _device = squid_adapter
    controller = adapter.microcontroller
    controller.illumination_begin(1, 42.0)

    def refuse(self, command):
        raise OSError("serial write failed")

    monkeypatch.setattr(type(controller), "send_cmd", refuse)
    with pytest.raises(OSError):
        controller.illumination_end(1)
    assert lit_handles(adapter) == ["f488"]


def test_a_failed_on_counts_as_on(squid_adapter, monkeypatch):
    adapter, _device = squid_adapter
    controller = adapter.microcontroller

    def refuse(self, command):
        raise OSError("serial write failed")

    monkeypatch.setattr(type(controller), "send_cmd", refuse)
    with pytest.raises(OSError):
        controller.illumination_begin(1, 42.0)
    assert lit_handles(adapter) == ["f488"]


def test_a_source_with_no_channel_is_still_listed(squid_adapter):
    adapter, _device = squid_adapter
    adapter.microcontroller.illumination_begin(11, 5.0)
    [lit] = adapter.illumination_state().lit
    assert lit.handle == "" and "no channel is configured" in lit.name


def test_the_autofocus_laser(squid_adapter):
    adapter, _device = squid_adapter
    adapter.microcontroller.af_laser_on()
    assert adapter.illumination_state().af_laser_on is True
    adapter.microcontroller.af_laser_off()
    assert adapter.illumination_state().af_laser_on is False


def test_a_snapshot_leaves_nothing_lit(squid_adapter):
    adapter, _device = squid_adapter
    asyncio.run(adapter.execute(cmd.ChannelSnapshot(channel=channel())))
    assert lit_handles(adapter) == []


def test_a_stream_is_lit_while_it_runs(squid_adapter):
    adapter, _device = squid_adapter
    asyncio.run(adapter.execute(cmd.ChannelStreamBegin(channel=channel())))
    try:
        assert lit_handles(adapter) == ["bfledfull"]
    finally:
        asyncio.run(adapter.execute(cmd.ChannelStreamEnd(channel=channel())))
    assert lit_handles(adapter) == []



# --- /api/health ---------------------------------------------------------------------------


def health(core):
    return core.health_report(time_since_last_request_s=None, now=time.time())


def test_a_working_microscope_is_ok(mock_core):
    mock_core.microscope.open_connections()
    report = health(mock_core)
    assert report.status == "ok", report.reasons
    assert report.reasons == []


def test_an_unresponsive_device_is_degraded_and_says_which(mock_core):
    mock_core.microscope.open_connections()
    mock_core.microscope._health.unresponsive(Device.MAIN_CAMERA, "snap", "no frame in 5 s")
    report = health(mock_core)
    assert report.status == "degraded"
    assert report.reasons == ["main_camera is not answering: no frame in 5 s"]


def test_a_stall_is_down(mock_core, monkeypatch):
    stall = Stall(
        device=Device.MAIN_CAMERA, operation="snap", started_at=time.time() - 40,
        deadline_s=30.0, thread="microscope-worker",
    )
    monkeypatch.setattr(type(mock_core.microscope), "stalls", lambda self: [stall])
    report = health(mock_core)
    assert report.status == "down"
    assert any("main_camera snap has not returned" in r for r in report.reasons)


def test_a_device_that_never_connected_is_down(mock_core):
    report = health(mock_core)
    assert report.status == "down"
    assert "main_camera is not connected" in report.reasons


def test_the_last_acquisition_error_is_reported(mock_core):
    mock_core._last_acquisition_error = "image storage failed: disk full"
    assert health(mock_core).last_acquisition_error == "image storage failed: disk full"
