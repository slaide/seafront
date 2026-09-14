"""
The wire protocol, against a fake Teensy.

Everything below `TeensyMicrocontroller` used to be untestable because it is a USB device;
`tests/fake_mcu.py` supplies one. What is checked here is the part that is correct and worth
keeping correct: how a command is laid out, how ids are allocated, how a position comes back,
and how the driver decides a command has finished.

The defects this layer is already known to have live next door in
`test_microcontroller_defects.py`, which fails on purpose.
"""

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fake_mcu import (
    COMMAND_LENGTH,
    PACKET_LENGTH,
    FakeMCU,
    make_microcontroller,
    status_packet,
)

from seafront.hardware.firmware_config import get_firmware_config
from seafront.hardware.microcontroller.teensy_microcontroller import (
    ILLUMINATION_CODE,
    Command,
    CommandName,
    HandleIsNone,
    MicrocontrollerStatusPackage,
    MicrocontrollerTimeout,
)

firmware = get_firmware_config()
SOURCE = ILLUMINATION_CODE.from_slot(0)

# the illumination lock is reentrant, so a test that wants it held has to hold it from
# another thread - taking it on this one would let send_cmd straight through
_holding = threading.Event()
_release = threading.Event()


def _hold_illumination(controller) -> None:
    with controller.illum:
        _holding.set()
        _release.wait(timeout=30)


@pytest.fixture
def mcu():
    """a driver talking to a well-behaved fake device"""
    controller, device = make_microcontroller(FakeMCU())
    return controller, device


class TestCommandLayout:
    """
    what goes out on the wire

    A command is 8 bytes: id, command name, four payload bytes, then the checksum in the last
    one. Pinned byte by byte because nothing else in the codebase describes this format, and
    a field that moves silently is a different command arriving at the firmware.
    """

    def test_a_command_is_the_declared_length(self):
        assert len(Command.move_to_mm("x", 1.0).bytes) == COMMAND_LENGTH
        assert COMMAND_LENGTH == 8

    def test_the_command_name_is_the_second_byte(self):
        assert Command.move_to_mm("x", 1.0)[1] == CommandName.MOVETO_X.value
        assert Command.move_to_mm("y", 1.0)[1] == CommandName.MOVETO_Y.value
        assert Command.move_to_mm("z", 1.0)[1] == CommandName.MOVETO_Z.value

    def test_an_unknown_axis_is_refused(self):
        with pytest.raises(RuntimeError, match="invalid axis"):
            Command.move_to_mm("w", 1.0)  # type: ignore[arg-type]

    def test_the_position_is_four_bytes_big_endian(self):
        command = Command.move_to_mm("x", 1.0)
        usteps = firmware.mm_to_ustep_x(1.0)
        assert list(command.bytes[2:6]) == [
            (usteps >> 24) & 0xFF, (usteps >> 16) & 0xFF, (usteps >> 8) & 0xFF, usteps & 0xFF
        ]

    def test_a_move_command_says_it_is_one(self):
        """the driver waits differently for moves; getting this wrong changes the timeout"""
        assert Command.move_to_mm("x", 1.0).is_move_cmd is True

    def test_the_id_is_the_first_byte(self):
        command = Command.move_to_mm("x", 1.0)
        command.set_id(42)
        assert command[0] == 42

    @pytest.mark.parametrize("mm", [0.0, 1.0, -1.0, 60.0, 120.5])
    def test_a_position_survives_encoding(self, mm):
        command = Command.move_to_mm("x", mm)
        from seafront.hardware.microcontroller.teensy_microcontroller import (
            intFromPayload,
            twos_complement_rev,
        )

        usteps = twos_complement_rev(intFromPayload(command.bytes, 2, 4), 4)
        assert usteps * firmware.mm_per_ustep_x == pytest.approx(mm, abs=1e-3)


class TestCommandIds:
    def test_the_first_id_is_zero(self, mcu):
        controller, _device = mcu
        assert controller._get_next_cmd_id() == 0

    def test_ids_increment(self, mcu):
        controller, _device = mcu
        assert [controller._get_next_cmd_id() for _ in range(5)] == [0, 1, 2, 3, 4]

    def test_they_wrap_at_256(self, mcu):
        """one byte on the wire, so 256 is all there is"""
        controller, _device = mcu
        ids = [controller._get_next_cmd_id() for _ in range(260)]
        assert ids[255] == 255
        assert ids[256] == 0
        assert set(ids) == set(range(256))

    def test_every_command_sent_carries_a_fresh_id(self, mcu):
        controller, device = mcu
        for _ in range(4):
            controller.send_cmd(Command.initialize())
        assert device.command_ids() == [0, 1, 2, 3]

    def test_the_checksum_is_recomputed_after_the_id_is_set(self, mcu):
        """
        the id is assigned inside send_cmd, so a checksum taken before it would be wrong

        The firmware answers CMD_CHECKSUM_ERROR to a packet whose checksum does not match,
        which the driver would then wait for forever.
        """
        controller, device = mcu
        controller.send_cmd(Command.initialize())
        sent = device.last_command()
        assert sent is not None
        expected = controller.crc_calculator.calculate_checksum(sent[:-1])
        assert sent[-1] == expected


class TestStatusPackets:
    """what comes back, and how it is read"""

    def test_a_packet_is_the_declared_length(self):
        assert len(status_packet()) == PACKET_LENGTH
        assert PACKET_LENGTH == 24

    def test_the_fields_land_where_they_are_expected(self):
        packet = MicrocontrollerStatusPackage(
            status_packet(last_cmd_id=17, exec_status=1, x_usteps=1000, y_usteps=-2000, buttons=3)
        )
        assert packet.last_cmd_id == 17
        assert packet.exec_status == 1
        assert packet.buttons_and_switches == 3

    @pytest.mark.parametrize("usteps", [0, 1, -1, 1_000_000, -1_000_000, 2**31 - 1, -(2**31)])
    def test_a_position_survives_the_round_trip(self, usteps):
        packet = MicrocontrollerStatusPackage(status_packet(x_usteps=usteps))
        assert packet.x_pos_usteps == usteps * firmware.STAGE_MOVEMENT_SIGN_X

    def test_a_negative_position_is_not_read_as_a_huge_positive_one(self):
        """the two's complement boundary, where a sign error hides"""
        packet = MicrocontrollerStatusPackage(status_packet(x_usteps=-1))
        assert packet.x_pos_usteps == -1 * firmware.STAGE_MOVEMENT_SIGN_X


class TestCompletingACommand:
    def test_a_command_the_device_acknowledges_completes(self, mcu):
        controller, device = mcu
        controller.send_cmd(Command.initialize())
        assert len(device.commands_received) == 1

    def test_the_driver_waits_for_the_id_it_sent(self, mcu):
        """
        a packet naming a different command does not complete this one

        Queued packets name command 200; the command being waited for is 7, so the wait runs
        out rather than returning on somebody else's acknowledgement.
        """
        controller, device = mcu
        command = Command.initialize()
        command.set_id(7)
        device.queue(*[status_packet(last_cmd_id=200) for _ in range(3)])
        with pytest.raises(MicrocontrollerTimeout):
            controller._wait_until_cmd_is_finished(command, timeout_s=0.3)

    def test_a_move_waits_for_the_stage_to_stop(self, mcu):
        """
        the firmware does not report "arrived", so the driver infers it

        While the reported position keeps changing the move is still going, whatever the
        command id says.
        """
        controller, device = mcu
        command = Command.move_to_mm("x", 10.0)
        command.set_id(7)
        # every packet acknowledges the command, but the position keeps advancing
        device.queue(*[status_packet(last_cmd_id=7, x_usteps=n * 100) for n in range(8)])
        with pytest.raises(MicrocontrollerTimeout):
            controller._wait_until_cmd_is_finished(command, timeout_s=0.3)

    def test_a_move_completes_once_the_position_settles(self, mcu):
        """
        packets arrive one at a time here rather than pre-queued

        A batch would not work: the reader skips all but the last packet of whatever is
        waiting (see test_microcontroller_defects), so two identical packets queued together
        are read as one, and "the position did not change" can never be observed.
        """
        controller, device = mcu
        device.moves_by = (0, 0, 0)  # the stage has stopped
        device.ack(7)
        command = Command.move_to_mm("x", 10.0)
        command.set_id(7)
        controller._wait_until_cmd_is_finished(command, timeout_s=2.0)

    def test_an_unfinished_exec_status_does_not_complete_a_command(self, mcu):
        controller, device = mcu
        command = Command.initialize()
        command.set_id(7)
        device.queue(*[status_packet(last_cmd_id=7, exec_status=1) for _ in range(3)])
        with pytest.raises(MicrocontrollerTimeout):
            controller._wait_until_cmd_is_finished(command, timeout_s=0.3)

    def test_the_timeout_says_what_it_was_waiting_for(self, mcu):
        """
        a bare timeout is undebuggable on a microscope

        The info carries the command it wanted and the one the device last reported, which
        is the difference between "the device is behind" and "the device is gone".
        """
        controller, device = mcu
        command = Command.move_to_mm("x", 1.0)
        command.set_id(7)
        device.queue(*[status_packet(last_cmd_id=200) for _ in range(2)])
        with pytest.raises(MicrocontrollerTimeout) as caught:
            controller._wait_until_cmd_is_finished(command, timeout_s=0.3)
        info = caught.value.info
        assert info is not None
        assert info.target_cmd_id == 7
        assert info.current_cmd_id == 200
        assert info.await_move is True


class TestPositionTracking:
    def test_the_last_position_follows_the_packets(self, mcu):
        controller, _device = mcu
        controller.send_cmd(Command.initialize())
        position = controller.get_last_position()
        assert position.x_pos_mm == pytest.approx(0.0, abs=1e-6)

    def test_a_moving_stage_is_reported_as_it_moves(self, mcu):
        controller, device = mcu
        device.moves_by = (1000, 0, 0)
        controller.send_cmd(Command.initialize())
        first = controller.get_last_position().x_pos_mm
        controller.send_cmd(Command.initialize())
        second = controller.get_last_position().x_pos_mm
        assert second > first


class TestIlluminationIsExclusive:
    """
    overlapping illumination is the one thing send_cmd guards directly

    Two light sources on at once is a sample exposed to something nobody asked for, so
    turning one on takes a lock that turning it off releases.
    """

    def test_turning_illumination_on_twice_is_refused(self, mcu):
        controller, _device = mcu
        held = threading.Thread(target=_hold_illumination, args=(controller,), daemon=True)
        held.start()
        _holding.wait(timeout=5)
        try:
            with pytest.raises(RuntimeError, match="illumination already on"):
                controller.send_cmd(Command.illumination_begin(SOURCE, 10.0))
        finally:
            _release.set()
            held.join(timeout=5)
            _holding.clear()
            _release.clear()

    def test_the_lock_is_released_after_the_command(self, mcu):
        controller, _device = mcu
        controller.send_cmd(Command.illumination_begin(SOURCE, 10.0))
        assert controller.illum.acquire(blocking=False), "the illumination lock was not released"
        controller.illum.release()

    def test_the_lock_is_released_even_when_the_command_fails(self, mcu):
        """
        a failed illumination command must not leave the lock held

        It is held for the duration of one command; leaking it means no light source can
        ever be turned on again without restarting the process.

        The write fails with an IOError, which `_usb_operation_with_reconnect` treats as a
        disconnect - it retries, drops the handle, and the caller finally sees HandleIsNone.
        That conversion is the real behaviour and is pinned here with it.
        """
        controller, device = mcu

        def explode(_data):
            raise OSError("cable fell out")

        device.receive_command = explode  # type: ignore[method-assign]
        with pytest.raises(HandleIsNone):
            controller.send_cmd(Command.illumination_begin(SOURCE, 10.0))
        assert controller.illum.acquire(blocking=False), "the illumination lock leaked on failure"
        controller.illum.release()
