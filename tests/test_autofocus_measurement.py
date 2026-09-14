"""
Reading a laser spot's position off an image, which is what tells Z where to go.

The reflection autofocus works by bouncing a laser off the coverslip; how far the reflected
spot sits from where it sat at calibration is proportional to how far out of focus the sample
is. `_measure_laser_autofocus` turns an image into that spot position, and the number it
returns is multiplied by a calibration constant and handed straight to a Z move.

So the arithmetic here has an objective on the end of it, and two of its properties are worth
more than the rest:

- **the sign**. A spot to the right of the reference means one direction; getting it backwards
  drives the objective the wrong way, and the approach loop's "am I getting closer" check then
  reads the growing error as convergence in the other direction.
- **refusing when there is no signal**. A dim laser, a missing sample, a coverslip that is not
  there: the honest answer is "I cannot see the spot", and the dangerous one is a number
  derived from noise.

The function is pure - image in, measurement out - so none of this needs a camera. The images
below are drawn with spots at known positions, which is more control than a real autofocus
camera would ever give a test.
"""

import numpy as np
import pytest
from fastapi import HTTPException

from seafront.server import commands as cmd

# The detector's own thresholds, from find_peaks in _measure_laser_autofocus. Named here so a
# test that depends on one says which.
MIN_HEIGHT = 20  # a spot dimmer than this is not a spot
MIN_SEPARATION = 200  # two spots closer than this are one spot
MIN_WIDTH = 10  # a spot narrower than this is noise

WIDTH = 1400
HEIGHT = 120


def spots(*positions: int, brightness: int = 200, sigma: float = 12.0) -> np.ndarray:
    """
    an autofocus frame with a gaussian spot at each x given

    Drawn rather than captured, so the answer is known before the function is called. `sigma`
    is wide enough to clear the detector's minimum width; `brightness` is the peak value.
    """
    image = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    x = np.arange(WIDTH)
    for position in positions:
        profile = brightness * np.exp(-((x - position) ** 2) / (2 * sigma**2))
        image = np.maximum(image, np.tile(profile, (HEIGHT, 1)).astype(np.uint8))
    return image


def measure(adapter, image, use_right_dot=True, calibration=None):
    return adapter._measure_laser_autofocus(image, use_right_dot, calibration)


def calibration(x_reference: float, um_per_px: float = 1.0):
    return cmd.LaserAutofocusCalibrationData(
        um_per_px=um_per_px,
        x_reference=x_reference,
        calibration_position=cmd.Position.zero(),
    )


@pytest.fixture
def adapter(squid_adapter):
    scope, _device = squid_adapter
    return scope


class TestFindingTheSpot:
    def test_a_single_spot_is_found_where_it_was_drawn(self, adapter):
        result = measure(adapter, spots(700))
        assert result.tracked_dot_x == pytest.approx(700, abs=5)

    @pytest.mark.parametrize("position", [300, 700, 1100])
    def test_it_is_found_wherever_it_is(self, adapter, position):
        assert measure(adapter, spots(position)).tracked_dot_x == pytest.approx(position, abs=5)

    def test_with_one_spot_the_left_right_choice_does_not_matter(self, adapter):
        """there is nothing to choose between, so both settings must agree"""
        left = measure(adapter, spots(700), use_right_dot=False)
        right = measure(adapter, spots(700), use_right_dot=True)
        assert left.tracked_dot_x == right.tracked_dot_x

    def test_an_empty_frame_is_refused(self, adapter):
        """
        the answer that matters when the laser is off or the sample is missing

        A number derived from noise here becomes a Z move.
        """
        with pytest.raises(HTTPException, match="no signal"):
            measure(adapter, np.zeros((HEIGHT, WIDTH), dtype=np.uint8))

    def test_a_spot_too_dim_to_see_is_refused(self, adapter):
        with pytest.raises(HTTPException, match="no signal"):
            measure(adapter, spots(700, brightness=MIN_HEIGHT - 5))

    def test_a_spot_just_bright_enough_is_found(self, adapter):
        """the threshold has to admit a real but weak signal, or focus fails on dim samples"""
        result = measure(adapter, spots(700, brightness=MIN_HEIGHT * 3))
        assert result.tracked_dot_x == pytest.approx(700, abs=10)

    def test_a_spot_too_narrow_to_be_a_spot_is_refused(self, adapter):
        """a single hot pixel is not a reflection"""
        image = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        image[:, 700] = 255
        with pytest.raises(HTTPException, match="no signal"):
            measure(adapter, image)


class TestChoosingBetweenTwoSpots:
    """
    the reflection produces two dots, and which one is tracked is configuration

    They are a fixed distance apart, so the left one is the right one minus that distance -
    which is how the function computes it rather than reading the left peak directly.
    """

    def test_the_right_dot_is_the_rightmost(self, adapter):
        result = measure(adapter, spots(400, 900), use_right_dot=True)
        assert result.tracked_dot_x == pytest.approx(900, abs=5)

    def test_the_left_dot_is_the_rightmost_minus_their_separation(self, adapter):
        result = measure(adapter, spots(400, 900), use_right_dot=False)
        assert result.tracked_dot_x == pytest.approx(400, abs=10)

    def test_the_separation_is_reported(self, adapter):
        result = measure(adapter, spots(400, 900))
        assert result.interpeakdistances == pytest.approx([500], abs=10)

    def test_the_rightmost_is_reported_whichever_is_tracked(self, adapter):
        for use_right in (True, False):
            result = measure(adapter, spots(400, 900), use_right_dot=use_right)
            assert result.rightmost_x == pytest.approx(900, abs=5)

    def test_two_spots_closer_than_the_detector_separates_are_one_spot(self, adapter):
        """
        the failure mode when the dots overlap

        Below the detector's minimum separation they merge, and the measurement becomes a
        single-dot one - which is why `use_right_dot` stops mattering rather than silently
        tracking half of a blur.
        """
        result = measure(adapter, spots(700, 700 + MIN_SEPARATION // 2))
        assert result.interpeakdistances == []

    def test_only_the_two_brightest_are_considered(self, adapter):
        """
        a third reflection must not turn into a third inter-peak distance

        The function keeps the two tallest, so there is always exactly zero or one gap - the
        `ValueError` for more than one is unreachable by construction, and this is what keeps
        it that way.
        """
        image = np.maximum(spots(300, 900), spots(600, brightness=60))
        result = measure(adapter, image)
        assert len(result.interpeakdistances) <= 1


class TestTurningAPositionIntoADisplacement:
    """
    `(tracked_dot_x - x_reference) * um_per_px`, which then becomes a Z move

    The approach converts it with `(target - current) * 1e-3` and calls move_by on z, so a
    sign error here is the objective travelling away from focus while the loop believes it is
    closing in.
    """

    def test_at_the_reference_the_displacement_is_zero(self, adapter):
        result = measure(adapter, spots(700), calibration=calibration(x_reference=700.0))
        assert result.estimated_displacement_um == pytest.approx(0.0, abs=5)

    def test_a_spot_right_of_the_reference_is_a_positive_displacement(self, adapter):
        result = measure(adapter, spots(800), calibration=calibration(x_reference=700.0))
        assert result.estimated_displacement_um > 0

    def test_a_spot_left_of_the_reference_is_a_negative_displacement(self, adapter):
        result = measure(adapter, spots(600), calibration=calibration(x_reference=700.0))
        assert result.estimated_displacement_um < 0

    def test_the_magnitude_is_the_pixel_offset_times_the_scale(self, adapter):
        result = measure(adapter, spots(800), calibration=calibration(700.0, um_per_px=2.5))
        assert result.estimated_displacement_um == pytest.approx(100 * 2.5, abs=15)

    def test_doubling_the_scale_doubles_the_displacement(self, adapter):
        one = measure(adapter, spots(800), calibration=calibration(700.0, um_per_px=1.0))
        two = measure(adapter, spots(800), calibration=calibration(700.0, um_per_px=2.0))
        assert two.estimated_displacement_um == pytest.approx(
            2 * one.estimated_displacement_um, rel=1e-6
        )

    def test_a_negative_scale_flips_the_direction(self, adapter):
        """
        which is how a microscope whose spot moves the other way is configured

        Worth pinning: it means the sign of the calibration constant is load-bearing, and a
        calibration that produced the wrong sign would drive Z away from focus with no
        complaint from anything here.
        """
        result = measure(adapter, spots(800), calibration=calibration(700.0, um_per_px=-1.0))
        assert result.estimated_displacement_um < 0

    def test_without_a_calibration_no_displacement_is_claimed(self, adapter):
        """
        the spot is still located; the conversion to micrometres is what needs a calibration

        Returning a number here would be inventing a scale factor.
        """
        result = measure(adapter, spots(700), calibration=None)
        assert result.estimated_displacement_um is None
        assert result.tracked_dot_x == pytest.approx(700, abs=5)

    def test_the_tracked_dot_is_what_the_displacement_is_measured_from(self, adapter):
        """with two dots and the left one tracked, the displacement follows the left one"""
        result = measure(
            adapter, spots(400, 900), use_right_dot=False, calibration=calibration(400.0)
        )
        assert result.estimated_displacement_um == pytest.approx(0.0, abs=15)


class TestACalibrationThatCannotBeUsed:
    """
    `um_per_px = 1000 / px_per_mm`

    The slope comes from a regression of spot position against z. A slope of zero means the
    spot did not move as z changed - the laser was off, or nothing was reflecting - and the
    division is undefined.
    """

    def test_a_zero_scale_makes_every_displacement_zero(self, adapter):
        """
        which reads as "perfectly in focus" wherever the stage is

        The approach would then declare itself converged immediately, at whatever z it
        happened to start from.
        """
        result = measure(adapter, spots(900), calibration=calibration(700.0, um_per_px=0.0))
        assert result.estimated_displacement_um == 0.0

    def test_the_division_that_produces_it_is_guarded(self):
        """
        a regression slope of zero must be reported, not divided by

        Checked as source because reaching it needs a full calibration sweep against
        hardware; what matters is that the guard is there and says what went wrong.
        """
        import inspect

        from seafront.hardware import squid

        source = inspect.getsource(squid.SquidAdapter._execute_LaserAutofocusCalibrate)
        assert "1000 / px_per_mm" in source, "the conversion moved; this check is stale"
        assert "px_per_mm == 0" in source or "px_per_mm != 0" in source, (
            "a zero regression slope is divided by without being checked"
        )
