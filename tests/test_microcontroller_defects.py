"""
The microcontroller defects from DEFECTS.md, as tests. These FAIL, on purpose.

DEFECTS.md has catalogued these since 2026-09-09 in prose. Prose does not fail, does not
appear in a test run, and cannot tell you the day somebody fixes one by accident. Each test
here describes what a correct driver does; each one is red until the driver does it.

Per CLAUDE.md: a failing test means the code is unfinished, not that the test needs managing.
None of these is xfailed, skipped, or softened to match what the code currently does - that
would launder a real signal into a fake green and bless the behaviour.

    C1  inbound status packets carry a CRC byte that is parsed and never checked
    C2  the reader discards every packet but the last one waiting, so the completion of a
        fast command is thrown away

A1 ("timeouts are swallowed") turned out to be half right and is now covered in
`test_microcontroller_silence.py` instead. The firmware in service genuinely does not
acknowledge the execution of every command, so a timeout is the routine case and raising on
it would turn working acquisitions into crashes. What was actually missing was the ability to
tell that case apart from a device that has stopped talking.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fake_mcu import FakeMCU, make_microcontroller, status_packet

from seafront.hardware.microcontroller.teensy_microcontroller import (
    Command,
    MicrocontrollerTimeout,
)


@pytest.fixture
def silent_device():
    """a device that is powered but has stopped answering: the 'slow versus dead' case"""
    return make_microcontroller(FakeMCU(acknowledge=False))


class TestC1InboundPacketsAreNotChecked:
    """
    DEFECTS.md C1 - outbound commands are checksummed, inbound packets are not

    `MicrocontrollerStatusPackage` reads the CRC byte into a field and nothing ever compares
    it. A corrupted packet is indistinguishable from a good one, so corrupt position data
    passes as truth - and a corrupted command id completes the wrong command.
    """

    def test_a_packet_with_a_wrong_checksum_does_not_complete_a_command(self):
        controller, device = make_microcontroller(FakeMCU())
        command = Command.initialize()
        command.set_id(7)
        device.queue(status_packet(last_cmd_id=7, crc_byte=0x00))
        with pytest.raises(MicrocontrollerTimeout):
            controller._wait_until_cmd_is_finished(command, timeout_s=0.5)

    def test_a_packet_with_a_wrong_checksum_does_not_move_the_reported_position(self):
        """
        the quieter half

        A corrupt position is not an error anybody sees; it is a stage that the server
        believes is somewhere it is not, which is the input to the next relative move.
        """
        controller, device = make_microcontroller(FakeMCU())
        before = controller.get_last_position().x_pos_mm
        command = Command.initialize()
        command.set_id(7)
        device.queue(status_packet(last_cmd_id=7, x_usteps=999_999, crc_byte=0x00))
        with pytest.raises(MicrocontrollerTimeout):
            controller._wait_until_cmd_is_finished(command, timeout_s=0.5)
        assert controller.get_last_position().x_pos_mm == before

    def test_a_correctly_checksummed_packet_is_still_accepted(self):
        """the check has to accept good packets, which is the half that is easy to get wrong"""
        controller, device = make_microcontroller(FakeMCU())
        device.ack(7)
        device.moves_by = (0, 0, 0)
        command = Command.initialize()
        command.set_id(7)
        controller._wait_until_cmd_is_finished(command, timeout_s=2.0)


class TestC2TheReaderThrowsPacketsAway:
    """
    DEFECTS.md C2 - the reader skips every packet but the last one waiting

    `_read_packets` computes how many whole packets are buffered and discards all but the
    final one, with its own TODO saying why. A command that completes quickly has its
    acknowledgement buffered behind later status packets and thrown away unread, so the
    driver waits out the full timeout for a command that finished immediately.
    """

    def test_a_completion_buffered_behind_later_packets_is_not_missed(self):
        controller, device = make_microcontroller(FakeMCU())
        command = Command.initialize()
        command.set_id(7)
        # the completion arrives first, then two further status packets land behind it
        device.queue(
            status_packet(last_cmd_id=7, exec_status=0),
            status_packet(last_cmd_id=200),
            status_packet(last_cmd_id=200),
        )
        controller._wait_until_cmd_is_finished(command, timeout_s=0.5)

    def test_no_packet_is_read_past_without_its_command_id_being_looked_at(self):
        """
        the property, rather than one arrangement of it

        Whatever the driver does for speed, a packet it never looks at is an
        acknowledgement it can never see.
        """
        controller, device = make_microcontroller(FakeMCU())
        command = Command.initialize()
        command.set_id(42)
        device.queue(
            *[status_packet(last_cmd_id=1) for _ in range(5)],
            status_packet(last_cmd_id=42, exec_status=0),
            *[status_packet(last_cmd_id=1) for _ in range(5)],
        )
        controller._wait_until_cmd_is_finished(command, timeout_s=0.5)
