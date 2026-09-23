"""
Knowing whether the microcontroller is still there.

The device streams a status packet about every 10 ms whether or not anything is happening, so
silence is information: a device that has sent nothing for a quarter of a second is not slow,
it is gone or wedged. What the layers owe each other:

- the driver remembers when it last heard anything, and a position read that meets silence
  raises instead of handing out the cached position as if it were current
- a read that comes back short raises, rather than parsing a partial packet
- the status read (which runs beside the worker) records the silence on the health board
  and does not drop the connection out from under whatever command is running
- the health board shows when the device last spoke, and a device that speaks again is
  marked healthy again - but only by a packet that arrived after it went quiet
"""

import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fake_mcu import FakeMCU, FakeSerial, make_microcontroller

from seafront.hardware.health import Condition, Device, HealthBoard


def health_of(adapter, device=Device.MICROCONTROLLER):
    return {entry.device: entry for entry in adapter.health()}[device]


# --- the driver ----------------------------------------------------------------------------


def test_reading_a_packet_records_when_the_device_was_heard():
    controller, _ = make_microcontroller()
    assert controller.last_heard_at() is None

    before = time.time()
    controller.get_last_position()

    heard = controller.last_heard_at()
    assert heard is not None
    assert before <= heard <= time.time()


def test_a_silent_device_fails_the_position_read_instead_of_returning_the_cache():
    controller, device = make_microcontroller()
    controller.get_last_position()
    device.queue()  # stop streaming: nothing more arrives

    started = time.monotonic()
    with pytest.raises(OSError, match="not responding"):
        controller.get_last_position()

    # bounded by the status read timeout, not by the 5 s a command would wait
    assert time.monotonic() - started < 1.0


def test_the_cached_position_is_available_without_talking_to_the_device():
    controller, device = make_microcontroller(FakeMCU(position=(1000, 0, 0)))
    measured = controller.get_last_position()
    device.queue()

    assert controller.last_position_cached() == measured


class ShortReads(FakeSerial):
    "a port that reports a packet waiting and then delivers half of it"

    def read(self, size: int) -> bytes:
        return super().read(size)[: size // 2]


def test_a_short_read_reopens_the_port_rather_than_reading_on_out_of_step(monkeypatch):
    from seafront.hardware.microcontroller import teensy_microcontroller

    controller, device = make_microcontroller()
    port = ShortReads(device)
    controller.handle = port
    reads: list[int] = []
    original_read = ShortReads.read

    def counting_read(self, size):
        reads.append(size)
        return original_read(self, size)

    monkeypatch.setattr(ShortReads, "read", counting_read)
    reopened: list[bool] = []

    def reopen(self):
        # the fresh port is a well-behaved one: the reopened stream starts on a boundary
        reopened.append(True)
        self.handle = FakeSerial(device)

    monkeypatch.setattr(type(controller), "_open_connection", reopen)
    monkeypatch.setattr(teensy_microcontroller.time, "sleep", lambda _s: None)

    controller.get_last_position()

    # one short read, then straight to reopening: no second read on the out-of-step stream
    assert len(reads) == 1
    assert reopened == [True]
    assert port.closed_count == 1


def test_a_short_read_that_keeps_happening_is_raised(monkeypatch):
    from seafront.hardware.microcontroller import teensy_microcontroller

    controller, device = make_microcontroller()
    controller.handle = ShortReads(device)

    def reopen(self):
        self.handle = ShortReads(device)

    monkeypatch.setattr(type(controller), "_open_connection", reopen)
    monkeypatch.setattr(teensy_microcontroller.time, "sleep", lambda _s: None)

    with pytest.raises(teensy_microcontroller.ShortReadError, match="bytes the port reported"):
        controller.get_last_position()


# --- the adapter ---------------------------------------------------------------------------


def test_silence_during_the_status_read_marks_unresponsive_without_dropping(squid_adapter):
    adapter, device = squid_adapter
    known = asyncio.run(adapter.get_current_state()).stage_position
    device.queue()

    state = asyncio.run(adapter.get_current_state())

    # the connection is still up: a command running on the worker keeps its hardware
    assert adapter.is_connected is True
    assert adapter.microcontroller.handle is not None
    # the position handed back is the last one the device reported
    assert state.stage_position == known
    health = health_of(adapter)
    assert health.condition == Condition.UNRESPONSIVE
    assert health.last_fault is not None
    assert health.last_fault.operation == "status read"
    assert "not responding" in health.last_fault.message


def test_the_device_speaking_again_lifts_unresponsive(squid_adapter):
    adapter, device = squid_adapter
    device.queue()
    asyncio.run(adapter.get_current_state())
    assert health_of(adapter).condition == Condition.UNRESPONSIVE

    # streaming again
    device._auto = True
    asyncio.run(adapter.get_current_state())

    health = health_of(adapter)
    assert health.condition == Condition.OK
    # the fault that happened is kept
    assert health.last_fault is not None and health.last_fault.operation == "status read"


def test_health_reports_when_the_microcontroller_last_spoke(squid_adapter):
    adapter, _ = squid_adapter
    asyncio.run(adapter.get_current_state())

    heard = health_of(adapter).last_heard_at
    assert heard is not None
    assert heard == adapter.microcontroller.last_heard_at()


# --- the board -----------------------------------------------------------------------------


def test_a_packet_from_before_the_silence_does_not_lift_unresponsive():
    board = HealthBoard([Device.MICROCONTROLLER])
    board.connected(Device.MICROCONTROLLER)
    earlier = time.time()
    board.unresponsive(Device.MICROCONTROLLER, "status read", "no packet")

    board.heard(Device.MICROCONTROLLER, at=earlier)
    assert board.get(Device.MICROCONTROLLER).condition == Condition.UNRESPONSIVE

    board.heard(Device.MICROCONTROLLER, at=time.time() + 1)
    assert board.get(Device.MICROCONTROLLER).condition == Condition.OK


def test_hearing_does_not_revive_a_device_that_was_dropped():
    board = HealthBoard([Device.MICROCONTROLLER])
    board.connected(Device.MICROCONTROLLER)
    board.lost(Device.MICROCONTROLLER, "MoveBy", OSError("gone"))

    board.heard(Device.MICROCONTROLLER, at=time.time() + 1)
    assert board.get(Device.MICROCONTROLLER).condition == Condition.DISCONNECTED
