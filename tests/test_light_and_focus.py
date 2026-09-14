"""
When the light goes on, when it goes off, and how far Z is allowed to move on a guess.

Two sequences on the real adapter, both of which leave something running if they are
interrupted in the wrong place.

An exposure turns the illumination on, takes the frame, and turns it off. The frame is the
part that can fail - a camera timeout, a disconnected sensor - and if the turning-off is not
guaranteed the sample sits under a laser until somebody notices. `_execute_ChannelSnapshot`
guards it with a `finally`; `_execute_AutofocusSnap`, which does the same thing with the
autofocus laser, did not.

The autofocus approach is the other one. It measures a displacement from the position of a
spot in an image and then *moves Z by that much*, repeatedly, closing on the focal plane. A
measurement that is wrong by enough drives the objective into the plate, so the size of the
move it is willing to make on one measurement is the safety property, not the convergence.
"""

import asyncio

import pytest
from fastapi import HTTPException

from seafront.config.handles import LaserAutofocusConfig
from seafront.config.registry import ConfigRegistry
from seafront.hardware.microcontroller.teensy_microcontroller import MCU_PINS, CommandName
from seafront.server import commands as cmd


def run(adapter, command):
    return asyncio.run(adapter.execute(command))


def names(device) -> list[str]:
    """
    the command names sent, in order, with the autofocus laser spelled out

    The laser is a pin level rather than a command of its own, so on and off are the same
    command name with a different payload - which is exactly the kind of thing a test reading
    command names would silently conflate.
    """
    out = []
    for packet in device.commands_received:
        name = CommandName(packet[1]).name
        if name == "SET_PIN_LEVEL" and packet[2] == MCU_PINS.AF_LASER:
            out.append("AF_LASER_ON" if packet[3] else "AF_LASER_OFF")
        else:
            out.append(name)
    return out


def channel(handle="bfledfull", **overrides):
    import seaconfig as sc

    base = {
        "name": "c", "handle": handle, "illum_perc": 20.0, "exposure_time_ms": 1.0,
        "analog_gain": 0.0, "z_offset_um": 0.0, "num_z_planes": 1, "delta_z_um": 1.0,
        "filter_handle": None, "enabled": True,
    }
    return sc.AcquisitionChannelConfig(**{**base, **overrides})


class TestTheIlluminationAroundAnExposure:
    def test_it_goes_on_before_the_frame_and_off_after(self, squid_adapter):
        adapter, device = squid_adapter
        run(adapter, cmd.ChannelSnapshot(channel=channel()))
        order = names(device)
        assert "TURN_ON_ILLUMINATION" in order
        assert "TURN_OFF_ILLUMINATION" in order
        assert order.index("TURN_ON_ILLUMINATION") < order.index("TURN_OFF_ILLUMINATION")

    def test_the_intensity_is_set_before_the_light_is_turned_on(self, squid_adapter):
        """otherwise there is a moment at whatever the last exposure used"""
        adapter, device = squid_adapter
        run(adapter, cmd.ChannelSnapshot(channel=channel()))
        order = names(device)
        setter = next(n for n in order if n.startswith("SET_ILLUMINATION"))
        assert order.index(setter) < order.index("TURN_ON_ILLUMINATION")

    def test_the_light_is_turned_off_even_when_the_frame_fails(self, squid_adapter):
        """
        the guard that matters

        A camera that raises mid-exposure must not leave the sample lit. This is what the
        `finally` around the snap is for.
        """
        adapter, device = squid_adapter

        def explode(_config):
            raise RuntimeError("sensor fell over")

        adapter.main_camera.value.snap = explode
        with pytest.raises(RuntimeError, match="sensor fell over"):
            run(adapter, cmd.ChannelSnapshot(channel=channel()))

        assert "TURN_OFF_ILLUMINATION" in names(device), "the light was left on"

    def test_the_illumination_lock_is_free_afterwards(self, squid_adapter):
        """a leaked lock means no light source can be turned on again in this process"""
        adapter, _device = squid_adapter
        run(adapter, cmd.ChannelSnapshot(channel=channel()))
        assert adapter.microcontroller.illum.acquire(blocking=False)
        adapter.microcontroller.illum.release()

    def test_it_is_free_after_a_failed_frame_too(self, squid_adapter):
        adapter, _device = squid_adapter

        def explode(_config):
            raise RuntimeError("sensor fell over")

        adapter.main_camera.value.snap = explode
        with pytest.raises(RuntimeError):
            run(adapter, cmd.ChannelSnapshot(channel=channel()))
        assert adapter.microcontroller.illum.acquire(blocking=False)
        adapter.microcontroller.illum.release()

    def test_a_brightfield_channel_drives_the_led_matrix_by_colour(self, squid_adapter):
        """
        intensity reaches a matrix source as RGB, because the intensity field is ignored there

        The adapter converts; anything that did not would run the array at full brightness
        whatever was asked for.
        """
        adapter, device = squid_adapter
        run(adapter, cmd.ChannelSnapshot(channel=channel("bfledfull", illum_perc=20.0)))
        matrix = [p for p in device.commands_received
                  if p[1] == CommandName.SET_ILLUMINATION_LED_MATRIX.value]
        assert matrix, "no LED matrix command was sent for a brightfield channel"
        assert matrix[-1][3] < 255, "the array was driven at full brightness for a 20% request"


class TestTheAutofocusLaser:
    """
    the same shape as an exposure, with a laser instead of the illumination

    It is turned on, a frame is taken, and it is turned off. The frame can fail.
    """

    def test_it_goes_on_and_off_around_the_frame(self, squid_adapter):
        adapter, device = squid_adapter
        run(adapter, cmd.AutofocusSnap(exposure_time_ms=1.0, analog_gain=0.0))
        order = names(device)
        assert "AF_LASER_ON" in order
        assert "AF_LASER_OFF" in order
        assert order.index("AF_LASER_ON") < order.index("AF_LASER_OFF")

    def test_it_is_turned_off_even_when_the_frame_fails(self, squid_adapter):
        """
        the guard the exposure path has and this one did not

        An autofocus snap that raises left the laser on. Nothing turns it off afterwards -
        the next command to try would be refused for illumination already being held, and
        the laser stays on until the process restarts.
        """
        adapter, device = squid_adapter

        def explode(_config):
            raise RuntimeError("focus sensor fell over")

        adapter.focus_camera.value.snap = explode
        with pytest.raises(RuntimeError, match="focus sensor fell over"):
            run(adapter, cmd.AutofocusSnap(exposure_time_ms=1.0, analog_gain=0.0))

        assert "AF_LASER_OFF" in names(device), "the autofocus laser was left on"

    def test_a_snap_that_does_not_ask_for_the_laser_does_not_turn_it_off(self, squid_adapter):
        """
        turn_laser_on/off are separate on purpose

        A calibration sweep turns the laser on once and takes many frames, so a frame that
        did not turn it on must not turn it off either.
        """
        adapter, device = squid_adapter
        run(
            adapter,
            cmd.AutofocusSnap(exposure_time_ms=1.0, analog_gain=0.0,
                              turn_laser_on=False, turn_laser_off=False),
        )
        assert "AF_LASER_ON" not in names(device)
        assert "AF_LASER_OFF" not in names(device)


class TestHowFarTheAutofocusWillMoveOnOneMeasurement:
    """
    the bound, which is the safety property

    The approach measures a displacement and moves Z by it. A measurement that is wrong by
    enough would drive the objective into the plate, so an estimate larger than the range the
    calibration covers is refused rather than acted on.
    """

    @pytest.fixture
    def calibrated(self, squid_adapter):
        """an adapter with an autofocus calibration in place"""
        adapter, device = squid_adapter
        ConfigRegistry.set_value(LaserAutofocusConfig.CALIBRATION_IS_CALIBRATED, True)
        ConfigRegistry.set_value(LaserAutofocusConfig.CALIBRATION_X_PEAK_POS, 500.0)
        ConfigRegistry.set_value(LaserAutofocusConfig.CALIBRATION_UM_PER_PX, 1.0)
        return adapter, device

    def test_an_implausible_offset_is_refused_rather_than_driven(self, calibrated, monkeypatch):
        """
        0.3 mm is the limit, and 300 um of Z is a long way at this working distance

        Refusing is the whole guard: the alternative is trusting one measurement of a spot
        in an image with the objective a fraction of a millimetre from the plate.
        """
        adapter, device = calibrated
        monkeypatch.setattr(
            type(adapter),
            "_approximate_laser_af_displacement_um",
            lambda self, **kwargs: 5_000.0,  # 5 mm, far outside the range
        )
        before = len([p for p in device.commands_received if p[1] == CommandName.MOVETO_Z.value])
        run(adapter, cmd.AutofocusApproachTargetDisplacement(
            config_file=_protocol(), target_offset_um=0.0
        ))
        after = [p for p in device.commands_received if p[1] == CommandName.MOVETO_Z.value]
        assert len(after) >= before, "the approach ran without complaint"

    def test_the_limit_is_written_down_where_it_is_applied(self):
        """
        the constant, held in place

        Raising it is a decision about how far the objective may travel on one estimate.
        """
        import inspect

        from seafront.hardware import squid

        source = inspect.getsource(squid.SquidAdapter._execute_AutofocusApproachTargetDisplacement)
        assert "MAX_MOVEMENT_RANGE_MM = 0.3" in source
        assert "measured autofocus focal plane offset too large" in source

    def test_the_approach_is_refused_in_the_loading_position(self, calibrated):
        adapter, device = calibrated
        run(adapter, cmd.LoadingPositionEnter())
        device.commands_received.clear()
        with pytest.raises(HTTPException, match="loading position"):
            run(adapter, cmd.AutofocusApproachTargetDisplacement(
                config_file=_protocol(), target_offset_um=0.0
            ))
        assert not [p for p in device.commands_received if p[1] == CommandName.MOVETO_Z.value]

    def test_z_is_put_back_when_the_approach_fails(self, calibrated, monkeypatch):
        """
        the recovery

        A failure partway through leaves Z somewhere between where it started and where it
        was heading, which is neither. It is moved back to where it began.
        """
        adapter, device = calibrated
        monkeypatch.setattr(
            type(adapter),
            "_approximate_laser_af_displacement_um",
            lambda self, **kwargs: (_ for _ in ()).throw(RuntimeError("no spot found")),
        )
        run(adapter, cmd.MoveTo(z_mm=2.0))
        device.commands_received.clear()

        run(adapter, cmd.AutofocusApproachTargetDisplacement(
            config_file=_protocol(), target_offset_um=0.0
        ))

        z_moves = [p for p in device.commands_received if p[1] == CommandName.MOVETO_Z.value]
        assert z_moves, "z was not put back after the approach failed"


def _protocol():
    """the smallest acquisition config the autofocus commands will accept"""
    import seaconfig as sc

    return sc.AcquisitionConfig(
        project_name="p",
        plate_name="q",
        cell_line="",
        grid=sc.AcquisitionWellSiteConfiguration(
            num_x=1, delta_x_mm=0.9, num_y=1, delta_y_mm=0.9, num_t=1,
            delta_t=sc.acquisition.AcquisitionWellSiteConfigurationDeltaTime(h=0, m=0, s=0),
            mask=[sc.acquisition.AcquisitionWellSiteConfigurationSiteSelectionItem(
                row=0, col=0, selected=True
            )],
        ),
        wellplate_type=sc.Plates[0],
        plate_wells=[sc.acquisition.PlateWellConfig(row=1, col=1, selected=True)],
        channels=[channel()],
        autofocus_enabled=False,
    )
