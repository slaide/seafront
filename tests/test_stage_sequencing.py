"""
The order the axes move in, which is the difference between parking a plate and breaking one.

Reaching the loading position means getting the stage out from under the objective and past
the clamp. There is only one safe order to do it in, and it is not enforced by anything except
the sequence written in `_execute_LoadingPositionEnter`:

    home z          objective up and out of the way, before anything moves sideways
    move y to 30    clear the clamp in y
    move x to 30    then in x
    home y          and only now drive both to their switches
    home x

Z first is the one that matters most. Travelling in XY with the objective down drags it across
whatever is on the stage; the plate loses, and so does the objective. Leaving reverses it: XY
first, Z down last.

None of this was reachable from a test until now - `SquidAdapter.make()` wants camera SDKs
that only exist on a microscope. Constructed directly over the fake microcontroller, every
command the sequence sends is recorded, so the order is checkable rather than a comment.
"""

import asyncio
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).parent))

from seafront.hardware.firmware_config import get_firmware_config
from seafront.hardware.microcontroller.teensy_microcontroller import CommandName
from seafront.server import commands as cmd

firmware = get_firmware_config()

AXIS_NAME = {
    firmware.AXIS_X: "x",
    firmware.AXIS_Y: "y",
    firmware.AXIS_Z: "z",
    firmware.AXIS_W: "w",
}
MOVE_AXIS = {
    CommandName.MOVETO_X.value: "x",
    CommandName.MOVETO_Y.value: "y",
    CommandName.MOVETO_Z.value: "z",
}


def steps(device) -> list[str]:
    """
    the recorded commands as readable steps, e.g. ["home z", "move y", "move x", ...]

    Only what the sequencing is about: which axis, and whether it was homed or moved to a
    coordinate. Anything else is reported by name so an unexpected command is visible rather
    than silently skipped.
    """
    out = []
    for packet in device.commands_received:
        name = packet[1]
        if name == CommandName.HOME_OR_ZERO.value:
            out.append(f"home {AXIS_NAME.get(packet[2], packet[2])}")
        elif name in MOVE_AXIS:
            out.append(f"move {MOVE_AXIS[name]}")
        else:
            out.append(CommandName(name).name.lower())
    return out


def run(adapter, command):
    return asyncio.run(adapter.execute(command))


class TestReachingTheLoadingPosition:
    def test_the_whole_sequence_in_order(self, squid_adapter):
        adapter, device = squid_adapter
        run(adapter, cmd.LoadingPositionEnter())
        assert steps(device) == ["home z", "move y", "move x", "home y", "home x"]

    def test_z_is_cleared_before_anything_moves_sideways(self, squid_adapter):
        """
        the one that breaks things

        Travelling in XY with the objective still down drags it across the plate. Stated on
        its own as well as inside the full sequence, because this is the property - the rest
        of the order is about the clamp, this is about the objective.
        """
        adapter, device = squid_adapter
        run(adapter, cmd.LoadingPositionEnter())
        order = steps(device)
        assert order[0] == "home z"
        assert all(step != "home z" for step in order[1:]), "z was touched again mid-sequence"

    def test_the_clamp_is_cleared_in_y_before_x(self, squid_adapter):
        adapter, device = squid_adapter
        run(adapter, cmd.LoadingPositionEnter())
        order = steps(device)
        assert order.index("move y") < order.index("move x")

    def test_both_axes_are_cleared_before_either_is_homed(self, squid_adapter):
        """
        homing drives to a limit switch, so it must not start until the clamp is behind it
        """
        adapter, device = squid_adapter
        run(adapter, cmd.LoadingPositionEnter())
        order = steps(device)
        last_clear = max(order.index("move x"), order.index("move y"))
        first_home = min(order.index("home x"), order.index("home y"))
        assert last_clear < first_home

    def test_the_flag_is_only_set_once_the_stage_has_arrived(self, squid_adapter):
        adapter, _device = squid_adapter
        assert adapter.is_in_loading_position is False
        run(adapter, cmd.LoadingPositionEnter())
        assert adapter.is_in_loading_position is True

    def test_entering_twice_is_refused(self, squid_adapter):
        adapter, device = squid_adapter
        run(adapter, cmd.LoadingPositionEnter())
        before = len(device.commands_received)
        with pytest.raises(HTTPException, match="already in loading position"):
            run(adapter, cmd.LoadingPositionEnter())
        assert len(device.commands_received) == before, "the refused entry still moved the stage"


class TestLeavingTheLoadingPosition:
    def test_the_whole_sequence_in_order(self, squid_adapter):
        adapter, device = squid_adapter
        run(adapter, cmd.LoadingPositionEnter())
        device.commands_received.clear()
        run(adapter, cmd.LoadingPositionLeave())
        assert steps(device) == ["move x", "move y", "move z"]

    def test_z_comes_down_last(self, squid_adapter):
        """
        the mirror of entering

        The objective only descends once the stage is where it is meant to be; descending
        first and then travelling is the same collision in the other order.
        """
        adapter, device = squid_adapter
        run(adapter, cmd.LoadingPositionEnter())
        device.commands_received.clear()
        run(adapter, cmd.LoadingPositionLeave())
        order = steps(device)
        assert order[-1] == "move z"

    def test_the_flag_is_cleared(self, squid_adapter):
        adapter, _device = squid_adapter
        run(adapter, cmd.LoadingPositionEnter())
        run(adapter, cmd.LoadingPositionLeave())
        assert adapter.is_in_loading_position is False

    def test_leaving_when_not_there_is_refused(self, squid_adapter):
        adapter, device = squid_adapter
        with pytest.raises(HTTPException, match="not in loading position"):
            run(adapter, cmd.LoadingPositionLeave())
        assert device.commands_received == []

    def test_a_round_trip_returns_to_a_usable_state(self, squid_adapter):
        adapter, _device = squid_adapter
        run(adapter, cmd.LoadingPositionEnter())
        run(adapter, cmd.LoadingPositionLeave())
        run(adapter, cmd.MoveTo(x_mm=60.0, y_mm=40.0))  # must not be refused


class TestNothingElseMovesWhileParked:
    """
    the stage is out from under the objective and past the clamp

    Any command that assumes the normal working area would drive into something, so the
    guard is on the adapter rather than on the route - a script talking to the hardware
    layer gets it too.
    """

    @pytest.fixture
    def parked(self, squid_adapter):
        adapter, device = squid_adapter
        run(adapter, cmd.LoadingPositionEnter())
        device.commands_received.clear()
        return adapter, device

    def test_an_absolute_move_is_refused(self, parked):
        adapter, device = parked
        with pytest.raises(HTTPException, match="loading position"):
            run(adapter, cmd.MoveTo(x_mm=60.0, y_mm=40.0))
        assert device.commands_received == []

    def test_a_move_to_a_well_is_refused(self, parked):
        adapter, device = parked
        import seaconfig as sc

        with pytest.raises(HTTPException, match="loading position"):
            run(adapter, cmd.MoveToWell(plate_type=sc.Plates[0], well_name="B02"))
        assert device.commands_received == []

    def test_the_refusal_names_the_reason(self, parked):
        adapter, _device = parked
        with pytest.raises(HTTPException) as caught:
            run(adapter, cmd.MoveTo(z_mm=1.0))
        assert "loading position" in str(caught.value.detail)


class TestApproachingATargetPosition:
    """
    which axis goes first on an ordinary move

    The adapter picks the order by which intermediate corner sits closer to the middle of the
    plate, so the stage travels through the plate rather than around its edge. Either order
    is safe; what matters is that one is chosen deliberately and that both axes arrive.
    """

    def test_both_axes_are_moved(self, squid_adapter):
        adapter, device = squid_adapter
        run(adapter, cmd.MoveTo(x_mm=60.0, y_mm=40.0))
        assert set(steps(device)) >= {"move x", "move y"}

    def test_a_single_axis_move_touches_only_that_axis(self, squid_adapter):
        adapter, device = squid_adapter
        run(adapter, cmd.MoveTo(x_mm=60.0))
        assert steps(device) == ["move x"]

    def test_z_alone_is_a_z_move(self, squid_adapter):
        adapter, device = squid_adapter
        run(adapter, cmd.MoveTo(z_mm=1.0))
        assert steps(device) == ["move z"]

    def test_moving_nowhere_moves_nothing(self, squid_adapter):
        adapter, device = squid_adapter
        run(adapter, cmd.MoveTo())
        assert device.commands_received == []

    def test_the_approach_order_depends_on_where_the_target_is(self, squid_adapter):
        """
        not always x first

        Both orders must be reachable, or the choice is not being made and the comment
        describing it is wrong.
        """
        adapter, device = squid_adapter
        orders = set()
        for x, y in ((5.0, 70.0), (110.0, 5.0), (60.0, 40.0), (10.0, 10.0)):
            device.commands_received.clear()
            run(adapter, cmd.MoveTo(x_mm=x, y_mm=y))
            order = [step for step in steps(device) if step in ("move x", "move y")]
            orders.add(tuple(order))
        assert len(orders) > 1, f"the same order every time: {orders}"


class TestCoordinatesAreCalibrated:
    """
    the offset between what a client asks for and what the stage is told

    A request is in plate coordinates; the stage works in its own. The conversion is applied
    on the way down and undone on the way back, so a client that asks for 60 mm and then
    reads the position back must see 60 mm.
    """

    def test_a_requested_position_survives_the_round_trip(self, squid_adapter):
        adapter, _device = squid_adapter
        run(adapter, cmd.MoveTo(x_mm=60.0, y_mm=40.0))
        state = asyncio.run(adapter.get_current_state())
        assert state.stage_position.x_pos_mm == pytest.approx(60.0, abs=0.01)
        assert state.stage_position.y_pos_mm == pytest.approx(40.0, abs=0.01)

    def test_a_calibration_offset_moves_the_stage_somewhere_else(self, squid_adapter):
        """
        the offset is real, not a no-op

        With a 1 mm offset the stage is commanded to a different place for the same request -
        which is the whole point - and reading back still reports what was asked for.
        """
        from seafront.config.handles import CalibrationConfig
        from seafront.config.registry import ConfigRegistry

        adapter, device = squid_adapter
        run(adapter, cmd.MoveTo(x_mm=60.0))
        uncalibrated = device.commands_received[-1]

        device.commands_received.clear()
        ConfigRegistry.set_value(CalibrationConfig.OFFSET_X_MM, 1.0)
        run(adapter, cmd.MoveTo(x_mm=60.0))
        calibrated = device.commands_received[-1]

        assert calibrated[2:6] != uncalibrated[2:6], "the calibration offset changed nothing"
