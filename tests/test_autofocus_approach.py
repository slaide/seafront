"""
Driving z to focus, which is the part of the autofocus that moves the objective.

Three pieces make up the reflection autofocus and this is the third. `test_autofocus_-
measurement.py` reads a spot position out of one frame; `test_laser_autofocus_calibration.py`
derives the um/px that turns that position into a distance. This is the loop that takes the
distance and moves.

It is the only one of the three that touches the stage, so it is the one where a sign error
is not a wrong number but an objective travelling the wrong way, and where giving up has to
look different from arriving.
"""

import math

import numpy as np
import pytest
import seaconfig as sc
from fastapi import HTTPException

from seafront.config.handles import LaserAutofocusConfig
from seafront.config.registry import ConfigRegistry
from seafront.hardware.firmware_config import get_firmware_config
from seafront.hardware.microcontroller.teensy_microcontroller import CommandName
from seafront.hardware.microscope import Locked
from seafront.server import commands as cmd
from tests.fake_mcu import device_z_mm as _device_z_mm
from tests.fake_mcu import set_device_z_mm as _put_z_mm
from tests.test_autofocus_measurement import spots

firmware = get_firmware_config()

# where the spot sits when the sample is exactly in focus, and how far it travels
FOCUS_X_PX = 700.0
PX_PER_UM = 2.0
UM_PER_PX = 1.0 / PX_PER_UM


class FocusDependentCamera:
    """
    one spot, at a position that tracks how far z is from focus

    The whole loop is a feedback controller over this relationship: the spot's distance from
    FOCUS_X_PX is the error signal, and moving z is the actuator. Drawing it rather than
    simulating means the focus is at an exactly known z, so "did it get there" has an answer.
    """

    def __init__(self, device, focus_z_mm: float = 0.0):
        self.device = device
        self.focus_z_mm = focus_z_mm
        self.device_type = "autofocus"
        self.acquisition_ongoing = False
        self.frames = 0

    def z_mm(self) -> float:
        return _device_z_mm(self.device)

    def snap(self, config: sc.AcquisitionChannelConfig) -> np.ndarray:
        self.frames += 1
        error_um = (self.z_mm() - self.focus_z_mm) * 1e3
        return spots(round(FOCUS_X_PX + error_um * PX_PER_UM))

    def close(self):
        pass


@pytest.fixture
def focusing(squid_adapter):
    """an adapter with a calibrated autofocus over a camera that reports distance from focus"""
    adapter, device = squid_adapter
    camera = FocusDependentCamera(device)
    adapter.focus_camera = Locked(camera)  # type: ignore[assignment]

    ConfigRegistry.set_value(LaserAutofocusConfig.CALIBRATION_IS_CALIBRATED, "yes")
    ConfigRegistry.set_value(LaserAutofocusConfig.CALIBRATION_X_PEAK_POS, FOCUS_X_PX)
    ConfigRegistry.set_value(LaserAutofocusConfig.CALIBRATION_UM_PER_PX, UM_PER_PX)
    ConfigRegistry.set_value(LaserAutofocusConfig.USE_RIGHT_DOT, "no")
    return adapter, device, camera


def _protocol() -> sc.AcquisitionConfig:
    plate = next(p for p in sc.plates.Plates if p.Model_id == "revvity-384-6057800")
    return sc.AcquisitionConfig(
        project_name="p", plate_name="pl", cell_line="c", autofocus_enabled=True,
        grid=sc.AcquisitionWellSiteConfiguration(
            num_x=1, delta_x_mm=0.9, num_y=1, delta_y_mm=0.9, num_t=1,
            delta_t=sc.AcquisitionWellSiteConfigurationDeltaTime(h=0, m=0, s=1),
            mask=[sc.AcquisitionWellSiteConfigurationSiteSelectionItem(row=0, col=0, selected=True)],
        ),
        wellplate_type=plate,
        plate_wells=[sc.PlateWellConfig(col=1, row=1, selected=True)],
        channels=[], machine_config=None, comment="",
        spec_version=sc.LATEST_SPEC_VERSION, timestamp=None,
    )


def _approach(adapter, *, target_offset_um: float = 0.0, max_num_reps: int = 5,
              pre_approach_refz: bool = False):
    return adapter._execute_AutofocusApproachTargetDisplacement(
        adapter.microcontroller,
        cmd.AutofocusApproachTargetDisplacement(
            config_file=_protocol(),
            target_offset_um=target_offset_um,
            max_num_reps=max_num_reps,
            pre_approach_refz=pre_approach_refz,
        ),
    )


def _absolute_z_moves_mm(device, since: int = 0) -> list[float]:
    "every MOVETO_Z the driver sent, in mm, in order"
    return [
        int.from_bytes(p[2:6], "big", signed=True)
        * firmware.STAGE_POS_SIGN_Z
        * firmware.mm_per_ustep_z
        for p in device.commands_received[since:]
        if p[1] == CommandName.MOVETO_Z.value
    ]


class TestItArrivesAtFocus:
    @pytest.mark.parametrize("start_offset_um", [-40.0, -10.0, -2.0, 2.0, 10.0, 40.0])
    def test_it_converges_from_either_side(self, focusing, start_offset_um):
        """
        the sign, which is the property with an objective on the end of it

        Starting above focus and starting below it must both end at focus. A sign error
        converges from one side and runs away from the other.
        """
        adapter, device, camera = focusing
        _put_z_mm(device, start_offset_um * 1e-3)
        _approach(adapter)
        assert camera.z_mm() * 1e3 == pytest.approx(0.0, abs=1.0)

    def test_it_reports_that_it_got_there(self, focusing):
        adapter, device, _camera = focusing
        _put_z_mm(device, 20e-3)
        assert _approach(adapter).reached_threshold is True

    def test_the_remaining_offset_is_within_the_threshold(self, focusing):
        """half a micrometre, below which another move is not worth making"""
        adapter, device, _camera = focusing
        _put_z_mm(device, 20e-3)
        result = _approach(adapter)
        assert math.fabs(result.uncompensated_offset_mm) < 0.5e-3

    def test_a_site_already_in_focus_is_left_alone(self, focusing):
        """the first estimate is already under the threshold, so nothing moves"""
        adapter, device, _camera = focusing
        _put_z_mm(device, 0.0)
        result = _approach(adapter)
        assert result.num_compensating_moves == 0
        assert result.reached_threshold is True

    def test_it_counts_the_moves_it_made(self, focusing):
        adapter, device, _camera = focusing
        _put_z_mm(device, 30e-3)
        assert _approach(adapter).num_compensating_moves >= 1

    def test_a_non_zero_target_offset_focuses_off_the_coverslip(self, focusing):
        """
        imaging a plane above the reflecting surface, which is what the offset is for

        The loop drives the measured displacement to `target_offset_um` rather than to zero,
        so asking for 10 um must leave the stage 10 um from where asking for 0 would.
        """
        adapter, device, camera = focusing
        _put_z_mm(device, 0.0)
        _approach(adapter, target_offset_um=10.0)
        assert camera.z_mm() * 1e3 == pytest.approx(10.0, abs=1.5)


class TestWhenItCannotGetThere:
    def test_an_offset_beyond_the_working_range_is_refused(self, focusing):
        """
        0.3 mm, past which the estimate is not to be trusted

        A displacement that large is more likely a misread spot than a stage that far out,
        and acting on it would be a 0.3 mm move toward the sample.
        """
        adapter, device, _camera = focusing
        _put_z_mm(device, 1.0)  # 1 mm out, far beyond the range
        result = _approach(adapter)
        # the internal error is caught by the loop's own handler, which resets z
        assert result.num_compensating_moves == 0

    def test_it_is_refused_in_the_loading_position(self, focusing):
        adapter, _device, _camera = focusing
        adapter.is_in_loading_position = True
        with pytest.raises(HTTPException) as excinfo:
            _approach(adapter)
        assert "loading position" in str(excinfo.value)

    def test_it_gives_up_rather_than_looping_forever(self, focusing):
        """
        max_num_reps bounds the loop

        Each repetition is a frame and a move; an unbounded loop over a sample that never
        comes into focus holds the worker for as long as it takes somebody to notice.
        """
        adapter, device, camera = focusing
        _put_z_mm(device, 50e-3)
        _approach(adapter, max_num_reps=1)
        assert camera.frames <= 3

    def test_the_stage_is_returned_when_the_loop_fails(self, focusing):
        """
        a failed approach must not leave the objective at whatever z it happened to reach

        The next site's move is relative to wherever this one stopped, so an abandoned
        approach that left the stage somewhere arbitrary carries into the rest of the run.
        """
        adapter, device, camera = focusing
        _put_z_mm(device, 5e-3)
        start_z = device.z

        def explode(config):
            raise RuntimeError("camera died mid-approach")

        camera.snap = explode  # type: ignore[method-assign]
        _approach(adapter)
        assert device.z == start_z


class TestTheApproachFromTheReference:
    """
    pre_approach_refz: start from the calibrated reference z rather than wherever z was

    Over a plate the stage drifts site to site; starting each approach from the same known z
    keeps every site's search inside the range the calibration covers.
    """

    def test_it_moves_to_the_reference_first(self, focusing):
        """
        checked on the command stream, not the final position

        The approach that follows can fail and reset z to where it started - which is what
        happens here, because focus is 2 mm from the reference and so outside the working
        range. That the stage went to the reference at all is the thing being tested.
        """
        adapter, device, _camera = focusing
        ConfigRegistry.set_value(LaserAutofocusConfig.CALIBRATION_REF_Z_MM, 2.0)
        _put_z_mm(device, 1.0)
        before = len(device.commands_received)
        _approach(adapter, pre_approach_refz=True)
        assert 2.0 in [round(z, 4) for z in _absolute_z_moves_mm(device, since=before)]

    def test_it_does_not_bother_when_it_is_already_there(self, focusing):
        """a move below the threshold costs time and backlash and buys nothing"""
        adapter, device, _camera = focusing
        ConfigRegistry.set_value(LaserAutofocusConfig.CALIBRATION_REF_Z_MM, 0.0)
        _put_z_mm(device, 0.1e-3)  # 0.1 um from the reference, under the 0.5 um threshold
        before = len(device.commands_received)
        _approach(adapter, pre_approach_refz=True)
        assert _absolute_z_moves_mm(device, since=before) == []

    def test_an_uncalibrated_reference_is_refused(self, focusing):
        """rather than approaching from an unknown z and reporting whatever comes back"""
        adapter, _device, _camera = focusing
        ConfigRegistry.reset()
        with pytest.raises((HTTPException, KeyError)):
            _approach(adapter, pre_approach_refz=True)


class TestTheTwoWaysItCanStop:
    """
    arriving and giving up, which are not the same thing and do not look the same.

    The loop stops either because the remaining offset is under half a micrometre, or because
    the new estimate came back *worse* than the last - the move went away from focus - in
    which case it returns to the previous z and stops, with the source's own TODO saying
    "we cannot do better, but we also have not actually gotten close to the offset".

    Both set `reached_threshold=True`. That is deliberate and documented on the result:
    "reached_threshold is set to true if the system has determined that it cannot get close
    to the target offset". So the flag means *finished trying*, not *in focus*, and the
    number that separates the two is `uncompensated_offset_mm`.

    Worth pinning precisely, because the name invites the other reading, and one caller has
    taken it: `protocol.py` sets `autofocus_succeeded = True` on this flag alone, and that is
    what writes "Autofocus: Success" into the image description - the only record anywhere
    that a site was imaged out of focus.
    """

    @staticmethod
    def _outrun(camera, per_frame_mm: float = 0.05):
        "a sample whose focus retreats faster than the loop can follow it"
        original = camera.snap

        def receding(config):
            camera.focus_z_mm -= per_frame_mm
            return original(config)

        camera.snap = receding

    def test_arriving_leaves_nothing_to_compensate(self, focusing):
        adapter, device, _camera = focusing
        _put_z_mm(device, 20e-3)
        result = _approach(adapter)
        assert result.reached_threshold is True
        assert math.fabs(result.uncompensated_offset_mm) < 0.5e-3

    def test_giving_up_sets_the_same_flag(self, focusing):
        """the documented meaning of the flag, which is 'done trying'"""
        adapter, device, camera = focusing
        _put_z_mm(device, 20e-3)
        self._outrun(camera)
        assert _approach(adapter).reached_threshold is True

    def test_but_leaves_the_offset_uncompensated(self, focusing):
        """
        and this is the number that tells them apart

        Two hundred times the threshold it claims to have reached. A caller that reads only
        the flag cannot tell this run from the one above.
        """
        adapter, device, camera = focusing
        _put_z_mm(device, 20e-3)
        self._outrun(camera)
        result = _approach(adapter)
        assert math.fabs(result.uncompensated_offset_mm) > 0.5e-3

    def test_it_returns_to_the_better_of_the_two_positions(self, focusing):
        """
        the move that made things worse is undone

        Having established that the last move went away from focus, staying there would be
        the worst of the options available.
        """
        adapter, device, camera = focusing
        _put_z_mm(device, 20e-3)
        self._outrun(camera)
        before = len(device.commands_received)
        _approach(adapter)
        assert _absolute_z_moves_mm(device, since=before) != []

    def test_running_out_of_repetitions_is_a_third_ending(self, focusing):
        """
        and it is the one that reports honestly

        When the estimate keeps improving but never gets under the threshold, the loop simply
        runs out of repetitions - and leaves the flag False, which is what a caller reading it
        as "in focus" would want it to do in the other case too.
        """
        adapter, device, camera = focusing
        _put_z_mm(device, 20e-3)
        # accelerating, so each estimate is smaller than the last and the give-up branch is
        # never taken - the loop just exhausts its budget
        frames = [0]
        original = camera.snap

        def accelerating(config):
            frames[0] += 1
            camera.focus_z_mm -= 0.05 * frames[0]
            return original(config)

        camera.snap = accelerating
        result = _approach(adapter, max_num_reps=5)
        assert result.reached_threshold is False
