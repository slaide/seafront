"""
Turning a requested power into a number for the laser.

`IlluminationController` exists because DAC percent is not optical power: on a real source,
asking for 25% of full power might mean driving the DAC to 50%. A channel configured with
`use_power_calibration` is one where somebody measured the curve, and the whole point is that
the requested figure is *not* what reaches the hardware.

Which makes every way the lookup can quietly go wrong a way of putting an unknown amount of
light on a sample:

- `np.interp` requires its x values to increase. A calibration table that is not sorted by
  optical power interpolates between the wrong pair of points and returns a plausible number.
- normalising by the maximum power divides by zero when every measured power is zero, and
  `nan` is not a DAC value.
- a calibration that fails to load falls back to using the requested percentage as the DAC
  value directly - which is precisely the thing the calibration exists to avoid, applied
  silently to a channel whose configuration asked for the opposite.

None of these are hypothetical; each was reachable through `PowerCalibration.validate_data`
before these tests.
"""

import math

import pytest

from seafront.config.basics import ChannelConfig, PowerCalibration
from seafront.hardware.illumination import IlluminationController


def calibrated(dac, mw, handle="laser") -> ChannelConfig:
    return ChannelConfig(
        name="a laser",
        handle=handle,
        source_slot=11,
        use_power_calibration=True,
        power_calibration=PowerCalibration(dac_percent=dac, optical_power_mw=mw),
    )


def uncalibrated(handle="bfledfull") -> ChannelConfig:
    return ChannelConfig(name="an led", handle=handle, source_slot=0)


# a convex curve, the shape a real source has: half the DAC is well under half the power
CURVE_DAC = [0.0, 25.0, 50.0, 75.0, 100.0]
CURVE_MW = [0.0, 1.0, 4.0, 9.0, 16.0]


class TestACalibratedChannel:
    def test_the_requested_power_is_not_the_dac_value(self):
        """
        the entire purpose of the thing

        If these were equal the calibration would be doing nothing, and a test that passed
        either way would be worthless.
        """
        controller = IlluminationController([calibrated(CURVE_DAC, CURVE_MW)])
        assert controller.get_calibrated_intensity("laser", 25.0) == pytest.approx(50.0)

    @pytest.mark.parametrize(
        "requested,dac", [(0.0, 0.0), (6.25, 25.0), (25.0, 50.0), (56.25, 75.0), (100.0, 100.0)]
    )
    def test_the_measured_points_map_exactly(self, requested, dac):
        """each measured point is on the curve, so it must come back unchanged"""
        controller = IlluminationController([calibrated(CURVE_DAC, CURVE_MW)])
        assert controller.get_calibrated_intensity("laser", requested) == pytest.approx(dac)

    def test_between_points_it_interpolates(self):
        controller = IlluminationController([calibrated(CURVE_DAC, CURVE_MW)])
        between = controller.get_calibrated_intensity("laser", 15.625)  # halfway 6.25 -> 25
        assert 25.0 < between < 50.0

    def test_more_power_asked_for_is_never_less_power_delivered(self):
        """
        monotonic, or the slider does the wrong thing

        A curve that dips means turning the requested power up turns the laser down, which is
        both wrong and extremely confusing to diagnose from the other side of a microscope.
        """
        controller = IlluminationController([calibrated(CURVE_DAC, CURVE_MW)])
        values = [controller.get_calibrated_intensity("laser", p) for p in range(0, 101, 5)]
        assert values == sorted(values)

    @pytest.mark.parametrize("requested", [-10.0, 150.0])
    def test_a_request_outside_the_range_is_clamped_into_it(self, requested):
        controller = IlluminationController([calibrated(CURVE_DAC, CURVE_MW)])
        result = controller.get_calibrated_intensity("laser", requested)
        assert 0.0 <= result <= 100.0

    def test_the_maximum_power_is_reported(self):
        controller = IlluminationController([calibrated(CURVE_DAC, CURVE_MW)])
        assert controller.get_max_power("laser") == pytest.approx(16.0)


class TestAnUncalibratedChannel:
    """no calibration asked for, so the requested percentage is the DAC value"""

    def test_the_request_passes_through(self):
        controller = IlluminationController([uncalibrated()])
        assert controller.get_calibrated_intensity("bfledfull", 20.0) == pytest.approx(20.0)

    def test_an_unknown_channel_passes_through(self):
        controller = IlluminationController([uncalibrated()])
        assert controller.get_calibrated_intensity("never-heard-of-it", 20.0) == pytest.approx(20.0)

    def test_it_has_no_maximum_power(self):
        controller = IlluminationController([uncalibrated()])
        assert controller.get_max_power("bfledfull") is None


class TestCalibrationDataThatCannotBeInterpolated:
    """
    what `validate_data` must reject, because `np.interp` will not complain

    `np.interp` documents that its x values must be increasing and does not check. Given a
    table sorted some other way it returns a number, from the wrong pair of points.
    """

    def test_a_table_not_sorted_by_power_is_refused(self):
        with pytest.raises(ValueError, match="increas"):
            PowerCalibration(
                dac_percent=[0.0, 25.0, 50.0, 75.0, 100.0],
                optical_power_mw=[0.0, 9.0, 4.0, 1.0, 16.0],
            ).validate_data()

    def test_a_table_with_a_repeated_power_is_refused(self):
        """
        two DAC values for one power is not a function

        Interpolation picks one of them arbitrarily, and which one is an implementation
        detail of numpy.
        """
        with pytest.raises(ValueError, match="increas"):
            PowerCalibration(
                dac_percent=[0.0, 50.0, 100.0], optical_power_mw=[0.0, 4.0, 4.0]
            ).validate_data()

    def test_a_table_where_nothing_was_measured_is_refused(self):
        """
        every power zero divides by zero when normalising, and produces a nan DAC value

        `validate_data` accepted it: the powers are all non-negative.
        """
        with pytest.raises(ValueError, match="positive"):
            PowerCalibration(
                dac_percent=[0.0, 50.0, 100.0], optical_power_mw=[0.0, 0.0, 0.0]
            ).validate_data()

    def test_a_decreasing_table_is_refused(self):
        with pytest.raises(ValueError, match="increas"):
            PowerCalibration(
                dac_percent=[0.0, 50.0, 100.0], optical_power_mw=[16.0, 4.0, 0.0]
            ).validate_data()

    def test_an_ascending_table_is_accepted(self):
        """the check has to accept real calibrations, which is the half that matters"""
        PowerCalibration(dac_percent=CURVE_DAC, optical_power_mw=CURVE_MW).validate_data()

    def test_the_existing_checks_still_apply(self):
        with pytest.raises(ValueError, match="same length"):
            PowerCalibration(dac_percent=[0.0, 100.0], optical_power_mw=[5.0]).validate_data()
        with pytest.raises(ValueError, match="at least 2"):
            PowerCalibration(dac_percent=[50.0], optical_power_mw=[5.0]).validate_data()
        with pytest.raises(ValueError, match="between 0 and 100"):
            PowerCalibration(
                dac_percent=[0.0, 150.0], optical_power_mw=[0.0, 5.0]
            ).validate_data()


class TestNoDacValueIsEverNotANumber:
    def test_a_well_formed_curve_never_produces_nan(self):
        controller = IlluminationController([calibrated(CURVE_DAC, CURVE_MW)])
        for requested in range(0, 101):
            value = controller.get_calibrated_intensity("laser", float(requested))
            assert math.isfinite(value), f"{requested}% produced {value}"

    def test_a_non_finite_request_does_not_become_a_non_finite_dac_value(self):
        """
        the request is clamped before interpolation, but nan survives a clip

        Whatever comes in, what goes to the hardware has to be a number.
        """
        controller = IlluminationController([calibrated(CURVE_DAC, CURVE_MW)])
        for requested in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError, match="finite"):
                controller.get_calibrated_intensity("laser", requested)


class TestACalibrationThatFailedToLoad:
    """
    the silent fallback, which is the dangerous one

    A channel configured `use_power_calibration=True` is one where somebody decided the
    requested percentage must NOT be sent to the hardware directly. When the calibration
    fails to load, that is exactly what used to happen - an error in the log, and then the
    uncalibrated value, on a channel whose configuration says not to.
    """

    def test_a_channel_that_asked_for_calibration_and_has_none_refuses(self):
        broken = ChannelConfig(
            name="a laser",
            handle="laser",
            source_slot=11,
            use_power_calibration=True,
            power_calibration=PowerCalibration(
                dac_percent=[0.0, 100.0], optical_power_mw=[5.0]  # lengths disagree
            ),
        )
        controller = IlluminationController([broken])
        with pytest.raises(ValueError, match="calibration"):
            controller.get_calibrated_intensity("laser", 10.0)

    def test_it_does_not_quietly_use_the_requested_value(self):
        """the specific wrong answer: 10% requested arriving at the DAC as 10%"""
        broken = ChannelConfig(
            name="a laser", handle="laser", source_slot=11, use_power_calibration=True,
            power_calibration=PowerCalibration(dac_percent=[0.0, 100.0], optical_power_mw=[5.0]),
        )
        controller = IlluminationController([broken])
        with pytest.raises(ValueError):
            controller.get_calibrated_intensity("laser", 10.0)

    def test_a_channel_with_no_calibration_at_all_is_refused_too(self):
        """`use_power_calibration=True` with nothing attached is the same mistake"""
        broken = ChannelConfig(
            name="a laser", handle="laser", source_slot=11,
            use_power_calibration=True, power_calibration=None,
        )
        controller = IlluminationController([broken])
        with pytest.raises(ValueError, match="calibration"):
            controller.get_calibrated_intensity("laser", 10.0)

    def test_the_other_channels_still_work(self):
        """
        refused at use, not at construction

        A miscalibrated laser must not stop the microscope from starting or from using its
        other light sources - it must stop that one channel from firing.
        """
        broken = ChannelConfig(
            name="a laser", handle="laser", source_slot=11, use_power_calibration=True,
            power_calibration=PowerCalibration(dac_percent=[0.0, 100.0], optical_power_mw=[5.0]),
        )
        controller = IlluminationController([broken, uncalibrated(), calibrated(CURVE_DAC, CURVE_MW, "good")])
        assert controller.get_calibrated_intensity("bfledfull", 20.0) == pytest.approx(20.0)
        assert controller.get_calibrated_intensity("good", 25.0) == pytest.approx(50.0)
        with pytest.raises(ValueError):
            controller.get_calibrated_intensity("laser", 20.0)
