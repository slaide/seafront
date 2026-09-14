"""
The filter wheel: a position the driver tracks and the firmware never reports.

Unlike the stage, there is no encoder and no status field for the filter wheel. The driver
moves it by a computed number of microsteps and remembers where it must therefore be, so
`filter_wheel_position` is a dead-reckoned belief rather than a measurement. Everything that
belief depends on is worth pinning:

- the arithmetic turning a position change into microsteps
- that a move which fails does not update the belief, since an unmoved wheel recorded as
  moved puts every later move off by that much, permanently
- the bounds, because a position outside them is a wheel driven into its stop

An acquisition selects a filter per channel, so a wheel that believes it is at slot 3 while
sitting at slot 2 images every fluorescence channel through the wrong filter, and nothing in
the resulting data says so.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fake_mcu import FakeMCU, make_microcontroller

from seafront.hardware.firmware_config import get_firmware_config
from seafront.hardware.microcontroller.teensy_microcontroller import CommandName, HandleIsNone

firmware = get_firmware_config()
MIN_SLOT = firmware.FILTERWHEEL_MIN_INDEX
MAX_SLOT = firmware.FILTERWHEEL_MAX_INDEX


@pytest.fixture
def wheel():
    controller, device = make_microcontroller(FakeMCU())
    return controller, device


class TestBounds:
    @pytest.mark.parametrize("position", [MIN_SLOT - 1, MAX_SLOT + 1, -1, 0, 999])
    def test_a_position_outside_the_range_is_refused(self, wheel, position):
        controller, device = wheel
        if MIN_SLOT <= position <= MAX_SLOT:
            pytest.skip(f"{position} is inside the configured range")
        with pytest.raises(ValueError, match="out of range"):
            controller.filter_wheel_set_position(position)
        assert device.commands_received == [], "a refused position still moved the wheel"

    @pytest.mark.parametrize("position", [MIN_SLOT, MAX_SLOT])
    def test_the_ends_of_the_range_are_reachable(self, wheel, position):
        """an off-by-one in the bounds makes a configured filter unusable"""
        controller, _device = wheel
        controller.filter_wheel_set_position(position)
        assert controller.filter_wheel_position == position

    def test_a_refused_position_does_not_change_what_the_driver_believes(self, wheel):
        controller, _device = wheel
        controller.filter_wheel_set_position(MIN_SLOT + 1)
        before = controller.filter_wheel_position
        with pytest.raises(ValueError):
            controller.filter_wheel_set_position(MAX_SLOT + 5)
        assert controller.filter_wheel_position == before


class TestMovement:
    def test_it_starts_at_the_minimum(self, wheel):
        controller, _device = wheel
        assert controller.filter_wheel_position == MIN_SLOT

    def test_moving_somewhere_else_sends_a_command(self, wheel):
        controller, device = wheel
        controller.filter_wheel_set_position(MIN_SLOT + 2)
        assert device.commands_received
        assert device.last_command()[1] == CommandName.MOVE_W.value

    def test_moving_where_it_already_is_sends_nothing(self, wheel):
        """
        no command, and no drift

        Every move is dead reckoning, so the cheapest way to stay accurate is not to move.
        """
        controller, device = wheel
        controller.filter_wheel_set_position(MIN_SLOT)
        assert device.commands_received == []

    def test_the_belief_follows_the_move(self, wheel):
        controller, _device = wheel
        for position in (MIN_SLOT + 1, MAX_SLOT, MIN_SLOT + 2, MIN_SLOT):
            controller.filter_wheel_set_position(position)
            assert controller.filter_wheel_position == position

    def test_a_move_backwards_is_the_negative_of_the_move_forwards(self, wheel):
        """
        the wheel has to come back to where it started

        A sign error here is a wheel that walks one direction over an acquisition, which
        looks like nothing at all until the filters stop matching the channels.
        """
        controller, device = wheel
        controller.filter_wheel_set_position(MIN_SLOT + 3)
        forward = device.last_command()
        controller.filter_wheel_set_position(MIN_SLOT)
        backward = device.last_command()

        from seafront.hardware.microcontroller.teensy_microcontroller import (
            intFromPayload,
            twos_complement_rev,
        )

        forward_usteps = twos_complement_rev(intFromPayload(forward, 2, 4), 4)
        backward_usteps = twos_complement_rev(intFromPayload(backward, 2, 4), 4)
        assert backward_usteps == -forward_usteps

    def test_the_step_size_is_one_slot_of_the_screw_pitch(self, wheel):
        """
        the arithmetic, written out

        One full turn of the W screw covers the whole wheel, so one slot is that pitch
        divided by the number of slots.
        """
        controller, device = wheel
        controller.filter_wheel_set_position(MIN_SLOT + 1)

        from seafront.hardware.microcontroller.teensy_microcontroller import (
            intFromPayload,
            twos_complement_rev,
        )

        sent = twos_complement_rev(intFromPayload(device.last_command(), 2, 4), 4)
        per_slot_mm = firmware.SCREW_PITCH_W_MM / (MAX_SLOT - MIN_SLOT + 1)
        assert sent == firmware.mm_to_ustep_w(per_slot_mm)

    def test_the_distance_scales_with_the_number_of_slots_crossed(self, wheel):
        controller, device = wheel
        from seafront.hardware.microcontroller.teensy_microcontroller import (
            intFromPayload,
            twos_complement_rev,
        )

        controller.filter_wheel_set_position(MIN_SLOT + 1)
        one = twos_complement_rev(intFromPayload(device.last_command(), 2, 4), 4)
        controller.filter_wheel_position = MIN_SLOT  # pretend it went back
        controller.filter_wheel_set_position(MIN_SLOT + 3)
        three = twos_complement_rev(intFromPayload(device.last_command(), 2, 4), 4)
        assert three == pytest.approx(3 * one, rel=0.01)


class TestAMoveThatFails:
    def test_a_failed_move_does_not_update_the_belief(self, wheel):
        """
        the one that compounds

        If the driver records a move that did not happen, every later move is computed from
        the wrong place and the error never washes out - the wheel is permanently offset,
        and no measurement exists to notice it.
        """
        controller, device = wheel
        controller.filter_wheel_set_position(MIN_SLOT + 1)
        before = controller.filter_wheel_position

        def explode(_data):
            raise OSError("cable fell out")

        device.receive_command = explode  # type: ignore[method-assign]
        # the write failure reaches the caller as HandleIsNone: the retry decorator treats an
        # IOError as a disconnect, retries, and drops the handle
        with pytest.raises(HandleIsNone):
            controller.filter_wheel_set_position(MIN_SLOT + 3)

        assert controller.filter_wheel_position == before, (
            "the driver recorded a move the wheel never made"
        )
