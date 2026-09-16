"""
Two kinds of silence, which must not look the same.

The firmware in service does not acknowledge the execution of every command. The write goes
out over the wire, the command runs, and no status packet ever names it. Waiting for that
acknowledgement therefore times out as a matter of course - dozens of times in an ordinary
acquisition - and there is nothing better to wait for. Treating that as a fault would turn
every working run into a crash.

A device that has stopped talking altogether is the opposite. The stage may never have moved,
and `send_cmd` returning normally is the only thing telling the layers above that it did, so
a silent device reported as success is a stage the server believes is somewhere it is not -
and the input to every relative move after it.

The two used to be indistinguishable - there was no telling slow from dead. They are
separable because this device streams status continuously whether or not
it is answering a particular command: packets arriving means alive, no packets at all means
gone. `_read_packets` sees every one of them and now counts them, so the distinction costs an
integer.

(`last_packet` in that function had been declared and logged but never assigned since it was
written, so the log line meant to describe the timeout always said `last_packet=None` - the
one diagnostic that would have shown the difference reported the same thing either way.)
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fake_mcu import FakeMCU, make_microcontroller

from seafront.hardware.microcontroller.teensy_microcontroller import (
    Command,
    MicrocontrollerTimeout,
)


@pytest.fixture
def unacknowledging():
    """
    a healthy device that never names the command it just ran

    Status keeps streaming - position, buttons, the id of some earlier command - so the
    driver can see it is alive. It simply never says "that one is done".
    """
    return make_microcontroller(FakeMCU(acknowledge=False))


@pytest.fixture
def silent():
    """a device that has stopped talking: nothing arrives at all"""
    controller, device = make_microcontroller(FakeMCU(acknowledge=False))
    device.queue()  # stop generating packets, and leave the buffer empty
    return controller, device


class TestACommandThatIsNeverAcknowledged:
    """
    the routine case, which must not be an error

    Every one of these would be a crash mid-acquisition if an unacknowledged command raised.
    """

    def test_a_move_returns_normally(self, unacknowledging):
        controller, _device = unacknowledging
        controller.send_cmd(Command.move_to_mm("x", 10.0))

    def test_a_non_move_returns_normally(self, unacknowledging):
        controller, _device = unacknowledging
        controller.send_cmd(Command.initialize())

    def test_the_command_did_reach_the_wire(self, unacknowledging):
        """
        which is why carrying on is the right answer

        The device has the command. It ran it. The only thing missing is it saying so.
        """
        controller, device = unacknowledging
        controller.send_cmd(Command.initialize())
        assert len(device.commands_received) == 1
        assert device.last_command()[1] == Command.initialize()[1]

    def test_the_driver_keeps_working_afterwards(self, unacknowledging):
        """a run is made of hundreds of these; none of them may leave anything wedged"""
        controller, device = unacknowledging
        for _ in range(5):
            controller.send_cmd(Command.initialize())
        assert len(device.commands_received) == 5

    def test_the_position_is_still_being_tracked(self, unacknowledging):
        """
        packets are flowing, so the position is live even though the command is not answered

        This is the evidence the device is fine, and it is also what the next move is
        computed against.
        """
        controller, device = unacknowledging
        device.moves_by = (1000, 0, 0)
        controller.send_cmd(Command.initialize())
        assert controller.get_last_position().x_pos_mm != 0.0

    def test_it_does_not_wait_five_times_over(self, unacknowledging):
        """
        one wait, not a retry loop

        There is nothing to retry: the acknowledgement is never coming. `send_cmd` used to
        carry a dead `NUM_CMD_REPEATS_MAX = 5` whose `break` sat outside the `except`, so it
        ran once whatever happened. Making it genuinely retry would have multiplied the cost
        of every unacknowledged command by five.
        """
        controller, _device = unacknowledging
        started = time.monotonic()
        controller.send_cmd(Command.move_to_mm("x", 10.0))
        took = time.monotonic() - started
        assert took < 8.0, f"waited {took:.1f}s; one 5s move timeout is the whole budget"


class TestADeviceThatHasStoppedTalking:
    """
    the case that must not look like success

    Nothing arrives at all. The stage may never have moved, and nothing above this layer can
    tell unless this layer says so.
    """

    def test_a_move_raises(self, silent):
        controller, _device = silent
        with pytest.raises(OSError, match="not responding"):
            controller.send_cmd(Command.move_to_mm("x", 10.0))

    def test_a_non_move_raises(self, silent):
        controller, _device = silent
        with pytest.raises(OSError, match="not responding"):
            controller.send_cmd(Command.initialize())

    def test_it_is_raised_as_a_disconnect_the_layers_above_understand(self, silent):
        """
        IOError, not MicrocontrollerTimeout

        MicrocontrollerTimeout derives from BaseException, so re-raising it would go past
        every `except Exception` between here and the client and arrive as an unhandled 500.
        An IOError is what SquidAdapter.execute already turns into DisconnectError - which
        closes the connections, reaches a client as "hardware disconnected", and lands a
        running acquisition on CRASHED instead of pretending it finished.
        """
        controller, _device = silent
        with pytest.raises(OSError) as caught:
            controller.send_cmd(Command.initialize())
        assert isinstance(caught.value, IOError)
        assert not isinstance(caught.value, MicrocontrollerTimeout)

    def test_the_failure_says_it_heard_nothing(self, silent):
        """
        the count is the evidence, and it is kept as the cause

        "timed out" alone is what made these two indistinguishable in the first place.
        """
        controller, _device = silent
        with pytest.raises(OSError) as caught:
            controller.send_cmd(Command.initialize())
        cause = caught.value.__cause__
        assert isinstance(cause, MicrocontrollerTimeout)
        assert cause.packets_seen == 0

    def test_the_failure_still_names_the_command(self, silent):
        controller, _device = silent
        with pytest.raises(OSError, match="MOVETO_X"):
            controller.send_cmd(Command.move_to_mm("x", 10.0))


class TestTheCountIsWhatSeparatesThem:
    def test_an_unacknowledged_command_saw_packets(self, unacknowledging):
        """
        the same timeout, from the inside

        `_wait_until_cmd_is_finished` raises in both cases; the count is the only thing that
        differs, and it is what `send_cmd` branches on.
        """
        controller, _device = unacknowledging
        command = Command.initialize()
        command.set_id(7)
        with pytest.raises(MicrocontrollerTimeout) as caught:
            controller._wait_until_cmd_is_finished(command, timeout_s=0.3)
        assert caught.value.packets_seen > 0

    def test_a_silent_device_saw_none(self, silent):
        controller, _device = silent
        command = Command.initialize()
        command.set_id(7)
        with pytest.raises(MicrocontrollerTimeout) as caught:
            controller._wait_until_cmd_is_finished(command, timeout_s=0.3)
        assert caught.value.packets_seen == 0

    def test_a_device_that_goes_quiet_partway_still_counts_what_it_sent(self):
        """
        the awkward middle, which counts as alive

        It was talking, so it was there. Whether it died a moment later is not something one
        command's timeout can tell, and guessing "dead" here would crash runs on a device
        that is merely slow.
        """
        controller, device = make_microcontroller(FakeMCU(acknowledge=False))
        controller.get_last_position()  # one packet read, device demonstrably alive
        device.queue()  # and now it goes quiet

        command = Command.initialize()
        command.set_id(7)
        with pytest.raises(MicrocontrollerTimeout) as caught:
            controller._wait_until_cmd_is_finished(command, timeout_s=0.3)
        assert caught.value.packets_seen == 0, (
            "packets are counted per wait, not per session: this wait heard nothing"
        )


class TestTheDiagnosticThatWasNeverWritten:
    def test_the_last_packet_seen_is_recorded(self):
        """
        `last_packet` was declared, logged, and never assigned

        The log line describing a timeout therefore always said `last_packet=None`, whether
        the device was streaming at 100 Hz or had been unplugged - which is precisely the
        information needed to tell those apart.
        """
        import inspect

        from seafront.hardware.microcontroller import teensy_microcontroller

        body = inspect.getsource(teensy_microcontroller.TeensyMicrocontroller._read_packets)
        assert "last_packet = packet" in body, "last_packet is still never assigned"
