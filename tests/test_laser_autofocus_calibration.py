"""
The laser autofocus calibration, which is where the focus scale comes from.

`_measure_laser_autofocus` - finding the spots in one frame, and turning a spot position into
a displacement - is covered in `test_autofocus_measurement.py`. This covers the procedure that
produces the number those measurements are multiplied by: sweep z across a span, measure the
spot at each step, fit a line, and invert its slope into um/px.

At 340 lines it was the largest untested block in the repository, and it is not incidental
code: `um_per_px` scales every autofocus correction the microscope ever makes, so an error in
it is an error in the z of every image of every acquisition afterwards, in the same direction,
with nothing downstream able to see it.

Testing it needs a focus camera that answers with the spot pattern for wherever z currently
is, which `_simulate_laf_image` already draws - the mock microscope uses it. Wiring it to the
fake stage's z makes the whole procedure run and its output checkable against the simulation's
own geometry: the dots there move 1.7 px per um, so a correct calibration must land on
1000/1700 um/px.
"""

import itertools

import numpy as np
import pytest
from fastapi import HTTPException
import seaconfig as sc

from seafront.config.handles import LaserAutofocusConfig
from seafront.config.registry import ConfigRegistry
from seafront.hardware.firmware_config import get_firmware_config
from seafront.hardware.microcontroller.teensy_microcontroller import CommandName
from seafront.hardware.mock_microscope import _simulate_laf_image
from seafront.hardware.squid import linear_regression
from seafront.server import commands as cmd
from tests.fake_mcu import device_z_mm
from tests.test_autofocus_measurement import spots

firmware = get_firmware_config()

# _simulate_laf_image shifts the dots by this many pixels per micrometre of z
SIMULATED_PX_PER_UM = 1.7
EXPECTED_UM_PER_PX = 1.0 / SIMULATED_PX_PER_UM


class ZAwareFocusCamera:
    """
    a focus camera that draws the spot pattern for wherever the fake stage currently is

    The stand-in cameras in `squid_adapter` answer with the same generic frame whatever z the
    stage is at, so a calibration sweep over them measures a spot that never moves - which is
    exactly the failure the procedure is supposed to refuse. This one reads z off the fake
    microcontroller, so the sweep sees the spot travel and the fit has something real to fit.
    """

    def __init__(self, device, z_offset_um: float = 0.0):
        self.device = device
        self.z_offset_um = z_offset_um
        self.device_type = "autofocus"
        self.acquisition_ongoing = False
        self.snaps: list[float] = []

    def _z_um(self) -> float:
        "the fake's z, in micrometres, the way the status packet decoder reads it"
        return device_z_mm(self.device) * 1e3 + self.z_offset_um

    def snap(self, config: sc.AcquisitionChannelConfig) -> np.ndarray:
        z_um = self._z_um()
        self.snaps.append(z_um)
        # noise off: a calibration is a fit, and a test of the fit should not be a test of
        # how well it tolerates this particular RNG draw
        return _simulate_laf_image(z_um, noise_level=0.0)

    def close(self):
        pass


class DrawnSpotFocusCamera:
    """
    spots drawn at a position that moves with z, at a rate the test picks

    The simulator is the right thing for "does the scale come out at the value the optics
    imply". It is the wrong thing for the refusals, because it always draws something
    findable at every z. Here the frame is drawn, so a test can ask for a spot that does not
    move, or for there to be only one of them.
    """

    SEPARATION_PX = 700
    BASE_X = 400

    def __init__(self, device, px_per_um: float = SIMULATED_PX_PER_UM, num_spots: int = 2):
        self.device = device
        self.px_per_um = px_per_um
        self.num_spots = num_spots
        self.device_type = "autofocus"
        self.acquisition_ongoing = False
        self.snaps: list[float] = []

    def snap(self, config: sc.AcquisitionChannelConfig) -> np.ndarray:
        z_um = device_z_mm(self.device) * 1e3
        self.snaps.append(z_um)
        left = self.BASE_X + z_um * self.px_per_um
        xs = [left] if self.num_spots == 1 else [left, left + self.SEPARATION_PX]
        return spots(*[round(x) for x in xs])

    def close(self):
        pass


@pytest.fixture
def drawn_spots(squid_adapter):
    """a SquidAdapter whose focus camera draws its spots; returns (adapter, camera)"""
    from seafront.hardware.microscope import Locked

    adapter, device = squid_adapter
    camera = DrawnSpotFocusCamera(device)
    adapter.focus_camera = Locked(camera)  # type: ignore[assignment]
    ConfigRegistry.set_value(LaserAutofocusConfig.CALIBRATION_NUM_Z_STEPS, 5)
    ConfigRegistry.set_value(LaserAutofocusConfig.CALIBRATION_Z_SPAN_MM, 0.2)
    ConfigRegistry.set_value(LaserAutofocusConfig.USE_RIGHT_DOT, "no")
    return adapter, camera


@pytest.fixture
def calibrating(squid_adapter):
    """
    a SquidAdapter whose focus camera answers for the current z

    Returns (adapter, device, camera) and a smaller sweep than the default seven steps, to
    keep a test that renders 3088x2064 frames to a couple of seconds.
    """
    from seafront.hardware.microscope import Locked

    adapter, device = squid_adapter
    camera = ZAwareFocusCamera(device)
    adapter.focus_camera = Locked(camera)  # type: ignore[assignment]

    ConfigRegistry.set_value(LaserAutofocusConfig.CALIBRATION_NUM_Z_STEPS, 5)
    ConfigRegistry.set_value(LaserAutofocusConfig.CALIBRATION_Z_SPAN_MM, 0.2)
    ConfigRegistry.set_value(LaserAutofocusConfig.USE_RIGHT_DOT, "no")
    return adapter, device, camera


def _calibrate(adapter):
    return adapter._execute_LaserAutofocusCalibrate(
        adapter.microcontroller, cmd.LaserAutofocusCalibrate()
    )


def _z_moves_mm(device) -> list[float]:
    "every relative z move the calibration asked for, in mm, in order"
    moves = []
    for packet in device.commands_received:
        if packet[1] != CommandName.MOVE_Z.value:
            continue
        raw = int.from_bytes(packet[2:6], "big", signed=True)
        moves.append(raw * firmware.STAGE_POS_SIGN_Z * firmware.mm_per_ustep_z)
    return moves


class TestTheScaleItDerives:
    """
    um_per_px, the one number the whole procedure exists to produce

    The fit is dot_x = slope * z_mm + intercept, so the slope is px/mm and the scale is its
    reciprocal in um. Checked against the simulation's own geometry rather than against
    whatever the code happens to return.
    """

    def test_the_scale_matches_the_geometry_it_was_measured_from(self, calibrating):
        adapter, _device, _camera = calibrating
        result = _calibrate(adapter)
        assert result.calibration_data is not None
        assert result.calibration_data.um_per_px == pytest.approx(EXPECTED_UM_PER_PX, rel=0.05)

    def test_the_scale_is_positive(self, calibrating):
        """a sign error here inverts every autofocus correction into a correction away"""
        adapter, _device, _camera = calibrating
        assert _calibrate(adapter).calibration_data.um_per_px > 0

    def test_a_reference_spot_position_is_recorded(self, calibrating):
        """
        displacement is measured from it, so a calibration without one is unusable

        It is the tracked dot's x at the calibration position, in pixels, and every later
        measurement is the distance from it.
        """
        adapter, _device, _camera = calibrating
        data = _calibrate(adapter).calibration_data
        assert 0 < data.x_reference < 3088

    def test_the_scale_does_not_depend_on_where_the_sweep_started(self, calibrating):
        """
        the same optics give the same scale from any starting z

        A scale that moved with the start position would mean the fit was picking up the
        offset rather than the slope.
        """
        adapter, _device, camera = calibrating
        first = _calibrate(adapter).calibration_data.um_per_px
        camera.z_offset_um = 20.0
        second = _calibrate(adapter).calibration_data.um_per_px
        assert second == pytest.approx(first, rel=0.05)


class TestTheSweep:
    """
    where the stage goes, which matters because it goes there on real hardware

    The sweep descends half the span, steps up across it, and returns. Every move is relative
    and every one is paid for in time and in backlash, so the shape of it is worth pinning.
    """

    def test_it_snaps_once_per_configured_step(self, calibrating):
        adapter, _device, camera = calibrating
        _calibrate(adapter)
        # the configured 5 sweep frames, plus the reference measurement afterwards and the
        # offset measurements that zero the error at the calibration position
        assert len(camera.snaps) >= 5

    def test_the_sweep_covers_the_configured_span(self, calibrating):
        adapter, _device, camera = calibrating
        _calibrate(adapter)
        sweep = camera.snaps[:5]
        assert max(sweep) - min(sweep) == pytest.approx(200.0, abs=1.0)  # 0.2 mm in um

    def test_the_steps_are_even(self, calibrating):
        """z_span / (num_steps - 1), so five steps across 0.2 mm is 50 um apart"""
        adapter, _device, camera = calibrating
        _calibrate(adapter)
        sweep = sorted(camera.snaps[:5])
        gaps = [b - a for a, b in itertools.pairwise(sweep)]
        assert gaps == pytest.approx([50.0] * 4, abs=1.0)

    def test_the_sweep_is_centred_on_where_it_started(self, calibrating):
        """
        half the span below and half above

        The calibration is meant to characterise the optics around the current focus, not to
        wander off to one side of it.
        """
        adapter, _device, camera = calibrating
        _calibrate(adapter)
        sweep = camera.snaps[:5]
        assert min(sweep) == pytest.approx(-100.0, abs=1.0)
        assert max(sweep) == pytest.approx(+100.0, abs=1.0)

    def test_the_descent_overshoots_and_comes_back(self, calibrating):
        """
        backlash: the first approach arrives from a fixed direction

        Every other move in the sweep is upward, so the one downward move is taken 40 um too
        far and then reversed, which leaves the whole sweep approaching from below.
        """
        adapter, device, _camera = calibrating
        _calibrate(adapter)
        moves = _z_moves_mm(device)
        assert moves[0] == pytest.approx(-(0.1 + 40e-3), abs=1e-4)
        assert moves[1] == pytest.approx(+40e-3, abs=1e-4)

    def test_the_stage_is_left_where_it_was_found(self, calibrating):
        """
        a calibration that moved the focus would invalidate itself

        The reference measurement is taken at the calibration position, so the stage has to be
        back at the start by then.
        """
        adapter, device, _camera = calibrating
        start_z = device_z_mm(device)
        _calibrate(adapter)
        end_mm = device_z_mm(device) - start_z
        assert end_mm == pytest.approx(0.0, abs=1e-3)


class TestWhenItCannotProduceAScale:
    """
    the refusals, exercised with drawn spots rather than the simulator

    `_simulate_laf_image` always shows something detectable at every z, so the cases below -
    a spot that does not travel, a right dot that is not there - cannot be reached through
    it. Drawing the frame puts the spots exactly where the case needs them.
    """

    def test_a_spot_that_does_not_travel_is_refused(self, drawn_spots):
        """
        the laser off, or nothing reflecting: no slope, and so no scale to derive

        Dividing by that slope is what produces um/px, so this was a ZeroDivisionError
        partway through a calibration before it was caught. The spots here are visible and
        perfectly findable - they simply do not move with z, which is the whole point.
        """
        adapter, camera = drawn_spots
        camera.px_per_um = 0.0
        with pytest.raises(Exception) as excinfo:
            _calibrate(adapter)
        assert "did not move" in str(excinfo.value).lower()

    def test_a_frame_with_no_spot_at_all_is_refused(self, squid_adapter):
        """the stand-in camera draws no laser spot, and a fit over noise is worse than a stop"""
        adapter, _device = squid_adapter
        ConfigRegistry.set_value(LaserAutofocusConfig.CALIBRATION_NUM_Z_STEPS, 3)
        ConfigRegistry.set_value(LaserAutofocusConfig.CALIBRATION_Z_SPAN_MM, 0.2)
        with pytest.raises(Exception) as excinfo:
            _calibrate(adapter)
        assert "no signal" in str(excinfo.value).lower()

    def test_asking_for_a_right_dot_that_was_never_seen_is_refused(self, drawn_spots):
        """
        rather than quietly calibrating against the left dot and reporting success

        Only one dot is ever drawn here, so there is no right-dot regression to use. The
        scale that would have come back is the left dot's, which is a different measurement
        than the one that was asked for.
        """
        adapter, camera = drawn_spots
        camera.num_spots = 1
        ConfigRegistry.set_value(LaserAutofocusConfig.USE_RIGHT_DOT, "yes")
        with pytest.raises(Exception) as excinfo:
            _calibrate(adapter)
        assert "right dot" in str(excinfo.value).lower()

    def test_one_dot_is_enough_when_the_right_one_was_not_asked_for(self, drawn_spots):
        """the counterpart: the refusal above is about the request, not about the frame"""
        adapter, camera = drawn_spots
        camera.num_spots = 1
        result = _calibrate(adapter)
        assert result.calibration_data.um_per_px == pytest.approx(EXPECTED_UM_PER_PX, rel=0.05)

    def test_it_is_refused_in_the_loading_position(self, calibrating):
        """there is no sample under the objective there, and z is not where it thinks"""
        adapter, _device, _camera = calibrating
        adapter.is_in_loading_position = True
        with pytest.raises(HTTPException) as excinfo:
            _calibrate(adapter)
        assert "loading position" in str(excinfo.value)


class TestTheFitUnderneath:
    """
    linear_regression, which the calibration's whole result rests on

    Pure, and used for nothing else, so it is worth pinning directly rather than only through
    a sweep that could mask a sign or a reciprocal error.
    """

    def test_a_line_is_recovered_exactly(self):
        xs = [0.0, 1.0, 2.0, 3.0]
        slope, intercept = linear_regression(xs, [1.0 + 2.0 * x for x in xs])
        assert slope == pytest.approx(2.0)
        assert intercept == pytest.approx(1.0)

    def test_a_negative_slope_stays_negative(self):
        xs = [0.0, 1.0, 2.0, 3.0]
        slope, _ = linear_regression(xs, [10.0 - 3.0 * x for x in xs])
        assert slope == pytest.approx(-3.0)

    def test_noise_does_not_move_the_slope_much(self):
        rng = np.random.default_rng(0)
        xs = np.linspace(0, 10, 50)
        ys = 4.0 * xs + 1.0 + rng.normal(0, 0.1, xs.size)
        slope, _ = linear_regression(xs, ys)
        assert slope == pytest.approx(4.0, rel=0.02)

    def test_a_flat_line_has_no_slope(self):
        """the case the calibration has to notice, since it divides by this"""
        slope, _ = linear_regression([0.0, 1.0, 2.0, 3.0], [7.0, 7.0, 7.0, 7.0])
        assert slope == pytest.approx(0.0)
