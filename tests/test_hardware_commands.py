"""
The commands that move metal and turn on light, byte by byte.

`tests/test_microcontroller_abstraction.py` checks that these methods exist. This checks what
they put on the wire, because the wire is where a mistake becomes an axis driven into a limit
switch or a laser at an intensity nobody asked for.

Homing is the one with the least margin: it deliberately drives an axis until it hits its
stop, so the direction byte is the difference between finding the switch and pushing against
the far end of the travel. It is derived from the configured movement sign rather than
written down per axis, which is exactly the kind of arithmetic that is right until somebody
changes a sign in the firmware profile.

The illumination commands have a quieter trap, pinned below: for an LED matrix source the
intensity argument is computed and then not used - brightness comes from the RGB bytes - so
asking for 20% on a matrix source without passing RGB gives full brightness.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fake_mcu import COMMAND_LENGTH, FakeMCU, make_microcontroller

from seafront.hardware.firmware_config import get_firmware_config
from seafront.hardware.microcontroller.teensy_microcontroller import (
    ILLUMINATION_CODE,
    Command,
    CommandName,
    HandleIsNone,
)

firmware = get_firmware_config()

LED_MATRIX = ILLUMINATION_CODE.from_slot(0)
LASER = next(code for code in ILLUMINATION_CODE if not code.is_led_matrix)


@pytest.fixture
def mcu():
    controller, device = make_microcontroller(FakeMCU())
    return controller, device


def sent(device) -> list[bytes]:
    return list(device.commands_received)


class TestHoming:
    """
    driving an axis into its limit switch, on purpose

    The direction byte is `(sign + 1) / 2`, so a sign of +1 means one direction and -1 the
    other. Getting it backwards drives the axis away from the switch, into the far end of
    the travel, and keeps going until something stops it.
    """

    @pytest.mark.parametrize(
        "axis,axis_byte",
        [("x", firmware.AXIS_X), ("y", firmware.AXIS_Y), ("z", firmware.AXIS_Z), ("w", firmware.AXIS_W)],
    )
    def test_the_axis_is_named_in_the_command(self, axis, axis_byte):
        command = Command.home(axis)
        assert command[1] == CommandName.HOME_OR_ZERO.value
        assert command[2] == axis_byte

    @pytest.mark.parametrize(
        "axis,sign",
        [
            ("x", firmware.STAGE_MOVEMENT_SIGN_X),
            ("y", firmware.STAGE_MOVEMENT_SIGN_Y),
            ("z", firmware.STAGE_MOVEMENT_SIGN_Z),
            ("w", firmware.STAGE_MOVEMENT_SIGN_W),
        ],
    )
    def test_the_direction_follows_the_configured_sign(self, axis, sign):
        assert Command.home(axis)[3] == int((sign + 1) / 2)

    def test_the_direction_byte_is_a_flag(self):
        """
        0 or 1, never anything else

        The arithmetic takes a sign of +1 or -1. Any other value in the firmware profile
        produces a direction byte the firmware has no meaning for.
        """
        for axis in ("x", "y", "z", "w"):
            assert Command.home(axis)[3] in (0, 1), axis

    def test_homing_is_a_move(self):
        """
        which is what gives it the longer timeout and the settle-detection

        Treated as an instant command it would be declared finished while the axis was still
        travelling.
        """
        assert Command.home("x").is_move_cmd is True

    def test_an_unknown_axis_is_refused(self):
        with pytest.raises(RuntimeError, match="invalid direction"):
            Command.home("q")  # type: ignore[arg-type]

    def test_homing_one_axis_sends_exactly_one_command(self, mcu):
        controller, device = mcu
        controller.home("z")
        assert len(sent(device)) == 1
        assert sent(device)[0][1] == CommandName.HOME_OR_ZERO.value


class TestSettingZero:
    """
    redefining where the origin is

    Every absolute move afterwards is measured from here, so a set_zero at the wrong place
    silently relocates the whole coordinate system.
    """

    @pytest.mark.parametrize(
        "axis,axis_byte",
        [("x", firmware.AXIS_X), ("y", firmware.AXIS_Y), ("z", firmware.AXIS_Z)],
    )
    def test_the_axis_is_named(self, axis, axis_byte):
        command = Command.set_zero(axis)
        assert command[1] == CommandName.HOME_OR_ZERO.value
        assert command[2] == axis_byte

    def test_it_is_distinguishable_from_homing(self):
        """
        same command name, different payload

        Both are HOME_OR_ZERO; only a later byte says which. Confusing them means homing
        when you meant to redefine the origin, or the reverse.
        """
        assert Command.set_zero("x").bytes != Command.home("x").bytes

    def test_an_unknown_axis_is_refused(self):
        with pytest.raises(RuntimeError):
            Command.set_zero("q")  # type: ignore[arg-type]


class TestIlluminationOnALaser:
    """intensity reaches the DAC as a 16-bit value, most significant byte first"""

    def test_two_commands_are_sent_set_then_turn_on(self):
        """
        the order matters

        Turning on before setting the intensity means a moment at whatever the previous
        intensity was, which on a laser is a moment of the wrong power on the sample.
        """
        commands = Command.illumination_begin(LASER, 50.0)
        assert len(commands) == 2
        assert commands[0][1] == CommandName.SET_ILLUMINATION.value
        assert commands[1][1] == CommandName.TURN_ON_ILLUMINATION.value

    def test_the_source_is_named(self):
        assert Command.illumination_begin(LASER, 50.0)[0][2] == LASER.value

    @pytest.mark.parametrize(
        "percent,raw",
        [(0.0, 0), (50.0, 0x7FFF), (100.0, 0xFFFF)],
    )
    def test_the_intensity_scales_across_the_full_range(self, percent, raw):
        command = Command.illumination_begin(LASER, percent)[0]
        assert (command[3] << 8) | command[4] == raw

    @pytest.mark.parametrize("percent", [100.1, 150.0, 1e6])
    def test_more_than_full_power_is_clamped_to_full_power(self, percent):
        command = Command.illumination_begin(LASER, percent)[0]
        assert (command[3] << 8) | command[4] == 0xFFFF

    @pytest.mark.parametrize("percent", [-0.1, -50.0, -1e6])
    def test_negative_power_is_clamped_to_off(self, percent):
        command = Command.illumination_begin(LASER, percent)[0]
        assert (command[3] << 8) | command[4] == 0

    def test_the_intensity_is_monotonic(self):
        raws = [
            (lambda c: (c[3] << 8) | c[4])(Command.illumination_begin(LASER, p)[0])
            for p in range(0, 101, 5)
        ]
        assert raws == sorted(raws)

    @pytest.mark.parametrize("percent", [float("nan"), float("inf")])
    def test_a_non_finite_intensity_cannot_be_encoded(self, percent):
        """
        it raises here rather than sending something

        The controller above refuses these first; this is the layer that would otherwise
        have to turn one into an integer.
        """
        with pytest.raises((ValueError, OverflowError)):
            Command.illumination_begin(LASER, percent)


class TestIlluminationOnAnLedMatrix:
    def test_brightness_comes_from_the_rgb_bytes(self):
        command = Command.illumination_begin(LED_MATRIX, 100.0, 0.5, 0.25, 0.0)[0]
        assert command[1] == CommandName.SET_ILLUMINATION_LED_MATRIX.value
        assert command[3] == 127
        assert command[4] == 63
        assert command[5] == 0

    @pytest.mark.parametrize("colour", [-1.0, 0.0, 1.0, 2.0])
    def test_colour_components_are_clamped_to_a_byte(self, colour):
        command = Command.illumination_begin(LED_MATRIX, 100.0, colour, colour, colour)[0]
        for channel in (3, 4, 5):
            assert 0 <= command[channel] <= 255

    def test_the_intensity_argument_is_ignored_for_a_matrix_source(self):
        """
        the trap, written down

        `intensity_percent` is computed and then unused on this path: brightness is the RGB
        bytes. So asking for 20% on a matrix source without passing RGB gives *full*
        brightness, silently. SquidAdapter knows this and converts the calibrated intensity
        into RGB before calling; anything else reaching this directly does not.
        """
        dim = Command.illumination_begin(LED_MATRIX, 20.0)[0]
        bright = Command.illumination_begin(LED_MATRIX, 100.0)[0]
        assert dim.bytes == bright.bytes
        assert dim[3] == dim[4] == dim[5] == 255, "defaulted to full white regardless of 20%"

    def test_passing_rgb_is_how_brightness_is_actually_set(self):
        dim = Command.illumination_begin(LED_MATRIX, 100.0, 0.2, 0.2, 0.2)[0]
        bright = Command.illumination_begin(LED_MATRIX, 100.0, 1.0, 1.0, 1.0)[0]
        assert dim[3] < bright[3]


class TestTurningIlluminationOff:
    def test_it_is_its_own_command(self):
        commands = Command.illumination_end()
        assert any(c[1] == CommandName.TURN_OFF_ILLUMINATION.value for c in commands)

    def test_the_driver_releases_the_lock_after_turning_off(self, mcu):
        """
        off has to be callable when the state is unknown

        Startup turns everything off before it knows what was on, so this path must not
        depend on having acquired the lock by turning something on first.
        """
        controller, _device = mcu
        controller.illumination_end()
        assert controller.illum.acquire(blocking=False)
        controller.illum.release()


class TestEveryCommandIsTheRightSize:
    """a packet of the wrong length desynchronises the firmware's parser for everything after"""

    def test_all_of_them(self):
        built = [
            Command.home("x"), Command.set_zero("x"), Command.reset(), Command.initialize(),
            Command.move_to_mm("x", 1.0),
            # these answer with a list: a relative move is split across as many packets as
            # the microstep count needs, and illumination is "set" followed by "turn on"
            Command.move_by_mm("x", 1.0),
            Command.illumination_begin(LASER, 50.0),
            Command.illumination_end(),
        ]
        for item in built:
            for command in item if isinstance(item, list) else [item]:
                assert len(command.bytes) == COMMAND_LENGTH, command[1]

    def test_a_relative_move_too_long_for_one_packet_becomes_several(self):
        """
        the split, which is why move_by is safe where move_to was not

        A relative move clamps to what fits in four signed bytes and emits another packet
        for the remainder, so a huge distance is many correct moves rather than one wrong
        one.
        """
        one_packet = Command.move_by_mm("x", 1.0)
        many_packets = Command.move_by_mm("x", 1e6)
        assert len(one_packet) == 1
        assert len(many_packets) > 1
        for command in many_packets:
            assert len(command.bytes) == COMMAND_LENGTH


class TestTheUsbRetryLayer:
    """
    what happens between a failed write and giving up

    USB on a long cable drops packets. The decorator retries on IOError and on a dropped
    handle, and on nothing else - a bug in the driver must not be retried five times and
    then reported as a disconnect.
    """

    def test_a_transient_failure_is_retried_and_succeeds(self, mcu):
        controller, device = mcu
        failures = {"left": 2}
        real = device.receive_command

        def flaky(data):
            if failures["left"]:
                failures["left"] -= 1
                raise OSError("transient")
            real(data)

        device.receive_command = flaky  # type: ignore[method-assign]
        controller.send_cmd(Command.initialize())
        assert failures["left"] == 0
        assert sent(device), "the command never got through"

    def test_a_persistent_failure_ends_as_a_disconnect(self, mcu):
        controller, device = mcu

        def always_fails(_data):
            raise OSError("cable is out")

        device.receive_command = always_fails  # type: ignore[method-assign]
        with pytest.raises(HandleIsNone):
            controller.send_cmd(Command.initialize())

    def test_an_unexpected_error_is_not_retried(self, mcu):
        """
        a bug is not a transient fault

        Retrying a TypeError five times and reporting it as a disconnect loses the actual
        error and blames the cable.
        """
        controller, device = mcu
        attempts = {"n": 0}

        def broken(_data):
            attempts["n"] += 1
            raise TypeError("this is a bug, not a cable")

        device.receive_command = broken  # type: ignore[method-assign]
        with pytest.raises(TypeError, match="not a cable"):
            controller.send_cmd(Command.initialize())
        assert attempts["n"] == 1, f"a non-IO error was retried {attempts['n']} times"


class TestExclusiveAccess:
    def test_the_lock_can_be_taken_and_given_back(self, mcu):
        controller, _device = mcu
        with controller.locked() as held:
            assert held is controller
        with controller.locked(blocking=False) as held:
            assert held is not None

    def test_it_is_reentrant_for_the_holder(self, mcu):
        """send_cmd takes it too, so a caller holding it must still be able to send"""
        controller, device = mcu
        with controller.locked():
            controller.send_cmd(Command.initialize())
        assert sent(device)
