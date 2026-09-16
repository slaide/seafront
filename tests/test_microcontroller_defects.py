"""
The microcontroller defects from DEFECTS.md, as tests. These FAIL, on purpose.

DEFECTS.md has catalogued these since 2026-09-09 in prose. Prose does not fail, does not
appear in a test run, and cannot tell you the day somebody fixes one by accident. Each test
here describes what a correct driver does; each one is red until the driver does it.

Per CLAUDE.md: a failing test means the code is unfinished, not that the test needs managing.
None of these is xfailed, skipped, or softened to match what the code currently does - that
would launder a real signal into a fake green and bless the behaviour.

    C2  the reader discards every packet but the last one waiting, so the completion of a
        fast command is thrown away

Two of the three catalogued defects turned out not to be what they looked like, both times
because of something only the person running the microscopes knew:

A1 ("timeouts are swallowed") was half right. The firmware in service genuinely does not
acknowledge the execution of every command, so a timeout is the routine case and raising on
it would turn working acquisitions into crashes. What was missing was the ability to tell
that apart from a device that has stopped talking; that is now
`test_microcontroller_silence.py`.

C1 ("inbound packets are not CRC-checked") is a deliberate non-concern: packet integrity has
never been a problem on this hardware, and the firmware in service predates the CRC and
version bytes, so validating that byte could reject every packet. What is left of it below is
a record of the decision rather than a demand for a check.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fake_mcu import FakeMCU, make_microcontroller, status_packet

from seafront.hardware.microcontroller.teensy_microcontroller import (
    Command,
    MicrocontrollerStatusPackage,
    MicrocontrollerTimeout,
)


@pytest.fixture
def silent_device():
    """a device that is powered but has stopped answering: the 'slow versus dead' case"""
    return make_microcontroller(FakeMCU(acknowledge=False))


class TestC1TheChecksumByteIsDeliberatelyNotChecked:
    """
    DEFECTS.md C1, resolved as a decision rather than a defect.

    Outbound commands are checksummed; inbound packets carry a CRC byte that is read into the
    struct and never compared. That is on purpose: packet integrity has not been a problem on
    this hardware, and the firmware in service predates the CRC and version bytes, so a check
    could reject every packet from the fleet.

    Recorded here rather than deleted because the byte is still parsed and still available.
    If corruption ever does turn up, the check is one comparison away, and these say where.
    """

    def test_the_byte_is_parsed_and_reachable(self):
        packet = MicrocontrollerStatusPackage(status_packet(last_cmd_id=7))
        assert isinstance(packet.crc, int)
        assert 0 <= packet.crc <= 255

    def test_the_fake_computes_a_real_checksum_over_the_body(self):
        """
        so the day somebody wants to check it, there is something to check against

        The same CRC-8/CCITT the driver uses for outbound commands.
        """
        import crc

        packet_bytes = status_packet(last_cmd_id=7, x_usteps=1234)
        calculator = crc.CrcCalculator(crc.Crc8.CCITT, table_based=True)
        assert packet_bytes[-1] == calculator.calculate_checksum(packet_bytes[:-1])

    def test_a_packet_whose_checksum_is_wrong_is_still_accepted(self):
        """
        the decision, stated

        Not an assertion that this is good - an assertion that this is what happens, so that
        changing it is a deliberate act with a test to update.
        """
        controller, device = make_microcontroller(FakeMCU())
        device.ack(7)
        device.moves_by = (0, 0, 0)
        device.corrupt_crc = True
        command = Command.initialize()
        command.set_id(7)
        controller._wait_until_cmd_is_finished(command, timeout_s=2.0)


class TestC2TheReaderThrowsPacketsAway:
    """
    DEFECTS.md C2, fixed: the reader reads and examines every whole packet waiting.

    It used to take the last packet of whatever was buffered and discard the rest, so a
    command that completed quickly had its acknowledgement thrown away unread and the driver
    waited out the whole timeout for a command that had already answered.
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


class TestHowDeepAnAcknowledgementCanBeBuried:
    """
    How much can pile up behind an acknowledgement before it is lost. Nothing, now.

    These used to measure the defect instead of the fix - one buffered packet ahead was
    enough to lose a completion, and the driver then spent the full timeout on a command that
    had already answered. Depth is still the right axis to test on, so they are the same
    arrangement with the answer a working reader gives.

    At the ~100 Hz this device streams status a packet arrives every ~10 ms against a 0.4 ms
    poll, so one or two waiting is the normal case and a long burst means the reader was away
    - one image copy, one burst of GIL contention, one slow log write. None of it may cost an
    acknowledgement.
    """

    def _completion_survives(self, packets_ahead: int) -> bool:
        controller, device = make_microcontroller(FakeMCU())
        command = Command.initialize()
        command.set_id(7)
        device.queue(
            status_packet(last_cmd_id=7, exec_status=0),
            *[status_packet(last_cmd_id=200) for _ in range(packets_ahead)],
        )
        try:
            controller._wait_until_cmd_is_finished(command, timeout_s=0.3)
            return True
        except MicrocontrollerTimeout:
            return False

    def test_a_completion_read_on_its_own_survives(self):
        assert self._completion_survives(0) is True

    def test_one_packet_behind_it_does_not_lose_it(self):
        """the old threshold, and it was this low: two waiting, and the answer was gone"""
        assert self._completion_survives(1) is True

    def test_nor_does_a_whole_burst_behind_it(self):
        assert self._completion_survives(5) is True

    def test_nor_a_burst_longer_than_any_real_stall(self):
        """50 packets is half a second of silence from the reader, far past anything real"""
        assert self._completion_survives(50) is True

    def test_the_driver_answers_at_once_instead_of_waiting_out_the_timeout(self):
        """
        what the fix buys

        The completion is already in the buffer, so the wait costs a poll, not the 0.3 s the
        old reader spent discarding it and then timing out.
        """
        controller, device = make_microcontroller(FakeMCU())
        command = Command.initialize()
        command.set_id(7)
        device.queue(
            status_packet(last_cmd_id=7, exec_status=0),
            status_packet(last_cmd_id=200),
        )
        started = time.monotonic()
        controller._wait_until_cmd_is_finished(command, timeout_s=0.3)
        assert time.monotonic() - started < 0.1
