"""
A filter wheel that failed to home.

Homing the microscope also homes the filter wheel. When the wheel's part fails, the stage is
still homed, but the wheel is at a position nobody knows, so:

- every filter move is refused, saying why the wheel cannot be used
- the refusal happens before the light goes on
- homing again retries the wheel, and a wheel that homes is usable again
"""

import asyncio

import pytest
import seaconfig as sc
from fastapi import HTTPException

from seafront.config.basics import FilterConfig
from seafront.config.handles import FilterWheelConfig
from seafront.config.registry import ConfigRegistry
from seafront.hardware.microcontroller.teensy_microcontroller import CommandName
from seafront.server import commands as cmd


def run(adapter, command):
    return asyncio.run(adapter.execute(command))


def channel() -> sc.AcquisitionChannelConfig:
    return sc.AcquisitionChannelConfig(
        name="c", handle="f488", illum_perc=20.0, exposure_time_ms=1.0, analog_gain=0.0,
        z_offset_um=0.0, num_z_planes=1, delta_z_um=1.0, filter_handle="gfp", enabled=True,
    )


@pytest.fixture
def with_wheel(squid_adapter):
    adapter, device = squid_adapter
    ConfigRegistry.set_value(FilterWheelConfig.AVAILABLE, "yes")
    adapter.filters = [FilterConfig(name="GFP", handle="gfp", slot=2)]
    return adapter, device


def break_wheel_homing(adapter, monkeypatch):
    def fail(self):
        raise RuntimeError("wheel limit switch never triggered")

    monkeypatch.setattr(type(adapter.microcontroller), "filter_wheel_home", fail)


def light_went_on(device) -> bool:
    return any(p[1] == CommandName.TURN_ON_ILLUMINATION.value for p in device.commands_received)


def test_a_wheel_that_failed_to_home_refuses_filter_moves_with_the_reason(with_wheel, monkeypatch):
    adapter, device = with_wheel
    break_wheel_homing(adapter, monkeypatch)
    asyncio.run(adapter.home())

    device.commands_received.clear()
    with pytest.raises(HTTPException) as refused:
        run(adapter, cmd.ChannelSnapshot(channel=channel()))
    assert "wheel limit switch never triggered" in str(refused.value.detail)
    assert not light_went_on(device)


def test_a_stream_is_refused_the_same_way(with_wheel, monkeypatch):
    adapter, device = with_wheel
    break_wheel_homing(adapter, monkeypatch)
    asyncio.run(adapter.home())

    device.commands_received.clear()
    with pytest.raises(HTTPException, match="filter wheel"):
        run(adapter, cmd.ChannelStreamBegin(channel=channel()))
    assert not light_went_on(device)
    assert not adapter.is_streaming


def test_homing_again_makes_the_wheel_usable(with_wheel, monkeypatch):
    adapter, _device = with_wheel
    with monkeypatch.context() as broken:
        break_wheel_homing(adapter, broken)
        asyncio.run(adapter.home())
    asyncio.run(adapter.home())

    run(adapter, cmd.ChannelSnapshot(channel=channel()))
