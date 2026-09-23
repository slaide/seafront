"""
A hardware call that does not come back.

A camera SDK call stuck in the vendor library cannot be interrupted. What the layers owe the
operator instead (see hardware/watchdog.py):

- the stall is noticed at its deadline: the device is marked unresponsive, and the stall is
  listed, so a busy rejection can say the hardware is stuck rather than merely busy
- a snap stuck with the light on has the light turned off, while it is still stuck
- a call that comes back after its light was turned off does not hand out what it got
- shutdown does not wait forever on hardware that cannot answer
"""

import asyncio
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from seafront.hardware.health import Condition, Device, HealthBoard
from seafront.hardware.microcontroller.teensy_microcontroller import MCU_PINS, CommandName
from seafront.hardware.watchdog import StalledCallError, StallWatchdog
from seafront.server import commands as cmd


def wait_for(condition, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def board() -> HealthBoard:
    health = HealthBoard([Device.MAIN_CAMERA])
    health.connected(Device.MAIN_CAMERA)
    return health


def condition_of(health: HealthBoard, device=Device.MAIN_CAMERA) -> Condition:
    return health.get(device).condition


# --- the watchdog --------------------------------------------------------------------------


def test_a_call_within_its_deadline_is_left_alone():
    health = board()
    watchdog = StallWatchdog(health)

    with watchdog.guard(Device.MAIN_CAMERA, "snap", deadline_s=5.0):
        pass

    assert watchdog.stalls() == []
    assert condition_of(health) == Condition.OK


def test_a_call_past_its_deadline_is_a_stall_and_the_device_is_unresponsive():
    health = board()
    watchdog = StallWatchdog(health)
    release = threading.Event()

    def stuck():
        with watchdog.guard(Device.MAIN_CAMERA, "snap", deadline_s=0.05):
            release.wait(5)

    caller = threading.Thread(target=stuck, name="the-stuck-one")
    caller.start()
    try:
        assert wait_for(lambda: watchdog.stalls() != [])
        [stall] = watchdog.stalls()
        assert stall.device == Device.MAIN_CAMERA
        assert stall.operation == "snap"
        assert stall.thread == "the-stuck-one"
        assert wait_for(lambda: condition_of(health) == Condition.UNRESPONSIVE)
        fault = health.get(Device.MAIN_CAMERA).last_fault
        assert fault is not None and "has not returned" in fault.message
    finally:
        release.set()
        caller.join()

    # it came back: no longer a stall, and the device is alive again
    assert watchdog.stalls() == []
    assert condition_of(health) == Condition.OK


def test_a_late_return_without_a_stall_action_keeps_its_result():
    health = board()
    watchdog = StallWatchdog(health)

    with watchdog.guard(Device.MAIN_CAMERA, "limits query", deadline_s=0.05):
        assert wait_for(lambda: condition_of(health) == Condition.UNRESPONSIVE)
        result = "still good"

    assert result == "still good"


def test_the_stall_action_runs_while_stuck_and_the_late_result_is_refused():
    health = board()
    watchdog = StallWatchdog(health)
    acted = threading.Event()

    def light_off() -> str:
        acted.set()
        return "all illumination was turned off"

    with pytest.raises(StalledCallError, match="all illumination was turned off"):
        with watchdog.guard(Device.MAIN_CAMERA, "snap", deadline_s=0.05, on_stall=light_off):
            # the action runs while this is still stuck, not once it returns
            assert acted.wait(5)

    fault = health.get(Device.MAIN_CAMERA).last_fault
    assert fault is not None and "all illumination was turned off" in fault.message


def test_a_failing_stall_action_is_reported_and_still_refuses_the_result():
    health = board()
    watchdog = StallWatchdog(health)

    def broken() -> str:
        raise RuntimeError("microcontroller would not listen")

    with pytest.raises(StalledCallError, match="would not listen"):
        with watchdog.guard(Device.MAIN_CAMERA, "snap", deadline_s=0.05, on_stall=broken):
            assert wait_for(lambda: condition_of(health) == Condition.UNRESPONSIVE)

    fault = health.get(Device.MAIN_CAMERA).last_fault
    assert fault is not None and "stall action failed" in fault.message


def test_a_call_that_fails_late_raises_its_own_error():
    "the camera's error says more than 'it was late'"
    health = board()
    watchdog = StallWatchdog(health)

    with pytest.raises(RuntimeError, match="sensor fell over"):
        with watchdog.guard(Device.MAIN_CAMERA, "snap", deadline_s=0.05, on_stall=lambda: "x"):
            assert wait_for(lambda: condition_of(health) == Condition.UNRESPONSIVE)
            raise RuntimeError("sensor fell over")


# --- the adapter ---------------------------------------------------------------------------


def channel():
    import seaconfig as sc

    return sc.AcquisitionChannelConfig(
        name="c", handle="bfledfull", illum_perc=20.0, exposure_time_ms=1.0,
        analog_gain=0.0, z_offset_um=0.0, num_z_planes=1, delta_z_um=1.0,
        filter_handle=None, enabled=True,
    )


def light_turned_off(device) -> bool:
    return any(p[1] == CommandName.TURN_OFF_ILLUMINATION.value for p in device.commands_received)


def af_laser_turned_off(device) -> bool:
    return any(
        p[1] == CommandName.SET_PIN_LEVEL.value and p[2] == MCU_PINS.AF_LASER and p[3] == 0
        for p in device.commands_received
    )


def run_in_thread(adapter, command):
    outcome: list[object] = []

    def target():
        try:
            outcome.append(asyncio.run(adapter.execute(command)))
        except BaseException as e:
            outcome.append(e)

    thread = threading.Thread(target=target, name="microscope-worker")
    thread.start()
    return thread, outcome


def test_a_stuck_snap_has_its_light_turned_off_and_its_image_refused(squid_adapter, monkeypatch):
    from seafront.hardware import squid

    adapter, device = squid_adapter
    monkeypatch.setattr(squid, "SNAP_DEADLINE_MARGIN_S", 0.1)
    release = threading.Event()
    real_snap = adapter.main_camera.value.snap

    def stuck_snap(config):
        release.wait(5)
        return real_snap(config)

    adapter.main_camera.value.snap = stuck_snap
    thread, outcome = run_in_thread(adapter, cmd.ChannelSnapshot(channel=channel()))
    try:
        assert wait_for(lambda: adapter.stalls() != [])
        # still stuck, and the light is already off
        assert wait_for(lambda: light_turned_off(device))
        health = {h.device: h for h in adapter.health()}[Device.MAIN_CAMERA]
        assert health.condition == Condition.UNRESPONSIVE
    finally:
        release.set()
        thread.join()

    [error] = outcome
    assert isinstance(error, StalledCallError)
    assert "illumination was turned off" in str(error)
    # the device came back, so it is no longer unresponsive; the stall stays on record
    health = {h.device: h for h in adapter.health()}[Device.MAIN_CAMERA]
    assert health.condition == Condition.OK
    assert health.last_fault is not None and health.last_fault.operation == "snap"


def test_a_stuck_focus_snap_has_the_autofocus_laser_turned_off(squid_adapter, monkeypatch):
    from seafront.hardware import squid

    adapter, device = squid_adapter
    monkeypatch.setattr(squid, "SNAP_DEADLINE_MARGIN_S", 0.1)
    release = threading.Event()
    real_snap = adapter.focus_camera.value.snap

    def stuck_snap(config):
        release.wait(5)
        return real_snap(config)

    adapter.focus_camera.value.snap = stuck_snap
    # the laser is left on by the command itself: only the stall action can turn it off
    command = cmd.AutofocusSnap(
        exposure_time_ms=1.0, analog_gain=0.0, turn_laser_on=True, turn_laser_off=False
    )
    thread, outcome = run_in_thread(adapter, command)
    try:
        assert wait_for(lambda: af_laser_turned_off(device))
    finally:
        release.set()
        thread.join()

    [error] = outcome
    assert isinstance(error, StalledCallError)


# --- the server ----------------------------------------------------------------------------


def _fake_stall():
    from seafront.hardware.watchdog import Stall

    return Stall(
        device=Device.MAIN_CAMERA,
        operation="snap",
        started_at=time.time() - 40,
        deadline_s=30.0,
        thread="microscope-worker",
    )


def test_a_busy_rejection_names_the_stall(mock_core, monkeypatch):
    stall = _fake_stall()
    monkeypatch.setattr(type(mock_core.microscope), "stalls", lambda self: [stall])

    reasons = mock_core._busy_reasons("ChannelSnapshot")

    assert reasons[0] == "hardware busy: ChannelSnapshot"
    assert any("main_camera snap has not returned after 40 s" in r for r in reasons), reasons


def test_shutdown_does_not_wait_on_stalled_hardware(mock_core, monkeypatch):
    stall = _fake_stall()
    monkeypatch.setattr(type(mock_core.microscope), "stalls", lambda self: [stall])
    closed: list[bool] = []
    monkeypatch.setattr(type(mock_core.microscope), "close", lambda self: closed.append(True))

    started = time.monotonic()
    assert mock_core.close() is False
    assert time.monotonic() - started < 5
    assert closed == [], "closing would have waited on the stuck call's device lock"


def test_shutdown_gives_up_on_a_close_that_hangs(mock_core, monkeypatch):
    from seafront.server.core import Core

    hang = threading.Event()
    monkeypatch.setattr(Core, "SHUTDOWN_CLOSE_DEADLINE_S", 0.2)
    monkeypatch.setattr(type(mock_core.microscope), "close", lambda self: hang.wait(10))
    try:
        started = time.monotonic()
        assert mock_core.close() is False
        assert time.monotonic() - started < 5
    finally:
        hang.set()
