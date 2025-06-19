import asyncio
import math
import threading
import typing as tp
from contextlib import contextmanager
from functools import wraps

import cv2
import numpy as np
import scipy  # to find peaks in a signal
import scipy.ndimage  # for guassian blur
import seaconfig as sc
from gxipy import gxiapi
from matplotlib import pyplot as plt
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from scipy import stats  # for linear regression

from seafront.config.basics import GlobalConfigHandler
from seafront.hardware import microcontroller as mc
from seafront.hardware.adapter import AdapterState, CoreState
from seafront.hardware.camera import AcquisitionMode, Camera
from seafront.logger import logger
from seafront.server import commands as cmd
from seafront.server.commands import (
    BasicSuccessResponse,
    IlluminationEndAll,
    error_internal,
)

# utility functions


class DisconnectError(BaseException):
    """indicate that the hardware was disconnected"""

    def __init__(self):
        super().__init__()


def linear_regression(
    x: list[float] | np.ndarray, y: list[float] | np.ndarray
) -> tuple[float, float]:
    "returns (slope,intercept)"
    slope, intercept, _, _, _ = stats.linregress(x, y)
    return slope, intercept  # type:ignore


def _process_image(img: np.ndarray, camera: Camera) -> tuple[np.ndarray, int]:
    """
    center crop main camera image to target size

    also pad so that top bits are used. e.g. if imaging bit depth is 12, top 12 bits will be set.
    """

    g_config = GlobalConfigHandler.get_dict()

    cam_img_width = g_config["main_camera_image_width_px"]
    assert cam_img_width is not None
    target_width: int = cam_img_width.intvalue
    assert isinstance(target_width, int), f"{type(target_width) = }"
    cam_img_height = g_config["main_camera_image_height_px"]
    assert cam_img_height is not None
    target_height: int = cam_img_height.intvalue
    assert isinstance(target_height, int), f"{type(target_height) = }"

    current_height = img.shape[0]
    current_width = img.shape[1]

    assert target_height > 0, "target height must be positive"
    assert target_width > 0, "target width must be positive"

    assert target_height <= current_height, (
        f"target height {target_height} is larger than max {current_height}"
    )
    assert target_width <= current_width, (
        f"target width {target_width} is larger than max {current_width}"
    )

    x_offset = (current_width - target_width) // 2
    y_offset = (current_height - target_height) // 2

    # seemingly swap x and y because of numpy's row-major order
    ret = img[y_offset : y_offset + target_height, x_offset : x_offset + target_width]

    flip_img_horizontal = g_config["main_camera_image_flip_horizontal"]
    assert flip_img_horizontal is not None
    if flip_img_horizontal.boolvalue:
        ret = np.flip(ret, axis=1)

    flip_img_vertical = g_config["main_camera_image_flip_vertical"]
    assert flip_img_vertical is not None
    if flip_img_vertical.boolvalue:
        ret = np.flip(ret, axis=0)

    image_file_pad_low = g_config["image_file_pad_low"]
    assert image_file_pad_low is not None
    cambits = 0

    match camera.pixel_format:
        case gxiapi.GxPixelFormatEntry.MONO8:
            cambits = 8
        case gxiapi.GxPixelFormatEntry.MONO10:
            cambits = 10
        case gxiapi.GxPixelFormatEntry.MONO12:
            cambits = 12
        case gxiapi.GxPixelFormatEntry.MONO14:
            cambits = 14
        case gxiapi.GxPixelFormatEntry.MONO16:
            cambits = 16
        case _:
            raise ValueError(f"unsupported pixel format {camera.pixel_format}")

    if image_file_pad_low.boolvalue:
        # e.g. ret = ret << (16 - 12)
        bytes_per_pixel = ret.dtype.itemsize
        bits_per_pixel = bytes_per_pixel * 8
        if bits_per_pixel != cambits:
            ret = ret << (bits_per_pixel - cambits)

    return ret, cambits


class Locked[T]:
    def __init__(self, t: T):
        self.lock = threading.RLock()
        self.t = t

    @contextmanager
    def __call__(self, blocking: bool = True) -> tp.Iterator[T | None]:
        if self.lock.acquire(blocking=blocking):
            try:
                yield self.t
            finally:
                self.lock.release()
        else:
            yield None


def squid_exclusive(f):
    "wrapper to ensure exclusive access to some code sections"

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return f(self, *args, **kwargs)

    return wrapper


class SquidAdapter(BaseModel):
    """interface to squid microscope"""

    main_camera: Locked[Camera]
    focus_camera: Locked[Camera]
    microcontroller: mc.Microcontroller

    state: CoreState = CoreState.Idle
    is_connected: bool = False
    is_in_loading_position: bool = False

    stream_callback: tp.Callable[[np.ndarray | bool], bool] | None = Field(default=None)

    last_state: AdapterState | None = None

    _lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @contextmanager
    def lock(self, blocking: bool = True) -> tp.Iterator[tp.Self | None]:
        "lock all hardware devices"
        if self._lock.acquire(blocking=blocking):
            try:
                with (
                    self.main_camera(blocking=blocking),
                    self.focus_camera(blocking=blocking),
                    self.microcontroller.locked(blocking=blocking),
                ):
                    yield self
            finally:
                self._lock.release()
        else:
            yield None

    @staticmethod
    def make() -> "SquidAdapter":
        g_dict = GlobalConfigHandler.get_dict()

        microcontrollers = mc.Microcontroller.get_all()
        cams = Camera.get_all()

        abort_startup = False
        if len(microcontrollers) == 0:
            logger.critical("startup - no microcontrollers found.")
            abort_startup = True
        if len(cams) < 2:
            logger.critical(f"startup - found less than two cameras (found {len(cams)})")
            abort_startup = True

        if abort_startup:
            error_msg = "did not find microscope hardware"
            logger.critical(f"startup - {error_msg}")
            raise RuntimeError(error_msg)

        main_camera_model_name = g_dict["main_camera_model"].value
        _main_cameras = [c for c in cams if c.model_name == main_camera_model_name]
        if len(_main_cameras) == 0:
            error_msg = f"no camera with model name {main_camera_model_name} found"
            logger.critical(f"startup - {error_msg}")
            cmd.error_internal(detail=error_msg)

        main_camera = _main_cameras[0]

        focus_camera_model_name = g_dict["laser_autofocus_camera_model"].value
        _focus_cameras = [c for c in cams if c.model_name == focus_camera_model_name]
        if len(_focus_cameras) == 0:
            error_msg = f"no camera with model name {focus_camera_model_name} found"
            logger.critical(f"startup - {error_msg}")
            cmd.error_internal(detail=error_msg)

        focus_camera = _focus_cameras[0]

        microcontroller = microcontrollers[0]

        squid = SquidAdapter(
            main_camera=Locked(main_camera),
            focus_camera=Locked(focus_camera),
            microcontroller=microcontroller,
        )

        # do NOT connect yet

        return squid

    @squid_exclusive
    def open_connections(self):
        """open connections to devices"""
        if self.is_connected:
            return

        with (
            self.microcontroller.locked() as mc,
            self.main_camera() as main_camera,
            self.focus_camera() as focus_camera,
        ):
            if mc is None:
                error_internal("microcontroller is busy")
            if main_camera is None:
                error_internal("main camera is busy")
            if focus_camera is None:
                error_internal("focus camera is busy")

            # small round trip because short disconnects from the cameras do not notify the cameras of the disconnect
            # so an attempted reconnect will throw an error indicating an existing connection
            # which cannot be severed without a physical connection (which may disrupted on the disconnect)
            # hence we ensure proper disconnect before a reconnect (even though this could waste an on->off->on roundtrip)
            self.is_connected = True
            self.close()
            self.is_connected = False

            try:
                main_camera.open(device_type="main")
                logger.debug("startup - connected to main cam")
                focus_camera.open(device_type="autofocus")
                logger.debug("startup - connected to focus cam")
                mc.open()
                logger.debug("startup - connected to microcontroller")
            except gxiapi.OffLine as e:
                logger.critical("startup - camera offline")
                raise DisconnectError() from e
            except IOError as e:
                logger.critical("startup - microcontroller offline")
                raise DisconnectError() from e

            logger.info("startup - connection to hardware devices established")
            self.is_connected = True

    def close(self):
        """
        close connection to microcontroller and cameras

        may also be used to close connection to remaining devices if connection to one has failed
        """
        if not self.is_connected:
            return

        with (
            self.microcontroller.locked() as mc,
            self.main_camera() as main_camera,
            self.focus_camera() as focus_camera,
        ):
            if mc is None:
                error_internal("microcontroller is busy")
            if main_camera is None:
                error_internal("main camera is busy")
            if focus_camera is None:
                error_internal("focus camera is busy")

            self.is_connected = False

            logger.debug("closing microcontroller")
            try:
                mc.close()
            except Exception:
                pass

            logger.debug("closing main camera")
            try:
                main_camera.close()
            except Exception:
                pass
            logger.debug("closing focus camera")
            try:
                focus_camera.close()
            except Exception:
                pass
            logger.debug("closing microcontroller")

            logger.info("closed connection to hardware devices")

    async def home(self):
        """perform homing maneuver"""

        with self.microcontroller.locked() as qmc:
            if qmc is None:
                error_internal("microcontroller is busy")

            try:
                logger.info("starting stage calibration (by entering loading position)")

                # reset the MCU
                logger.debug("resetting mcu")
                awaitme = qmc.send_cmd(mc.Command.reset())
                logger.debug("?")
                await awaitme
                logger.debug("done")

                # reinitialize motor drivers and DAC
                logger.debug("initializing microcontroller")
                await qmc.send_cmd(mc.Command.initialize())
                logger.debug("done initializing microcontroller")
                logger.debug("configure_actuators")
                await qmc.send_cmd(mc.Command.configure_actuators())
                logger.debug("done configuring actuators")

                logger.info("ensuring illumination is off")
                # make sure all illumination is off
                for illum_src in [
                    mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_405NM,
                    mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_488NM,
                    mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_561NM,
                    mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_638NM,
                    mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_730NM,
                    mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_LED_ARRAY_FULL,  # this will turn off the led matrix
                ]:
                    await qmc.send_cmd(mc.Command.illumination_end(illum_src))

                logger.debug("calibrating xy stage")

                # when starting up the microscope, the initial position is considered (0,0,0)
                # even homing considers the limits, so before homing, we need to disable the limits
                await qmc.send_cmd(mc.Command.set_limit_mm("z", -10.0, "lower"))
                await qmc.send_cmd(mc.Command.set_limit_mm("z", 10.0, "upper"))

                # move objective out of the way
                await qmc.send_cmd(mc.Command.home("z"))
                await qmc.send_cmd(mc.Command.set_zero("z"))
                # set z limit to (or below) 6.7mm, because above that, the motor can get stuck
                await qmc.send_cmd(mc.Command.set_limit_mm("z", 0.0, "lower"))
                await qmc.send_cmd(mc.Command.set_limit_mm("z", 6.7, "upper"))
                # home x to set x reference
                await qmc.send_cmd(mc.Command.home("x"))
                await qmc.send_cmd(mc.Command.set_zero("x"))
                # clear clamp in x
                await qmc.send_cmd(mc.Command.move_by_mm("x", 30))
                # then move in position to properly apply clamp
                await qmc.send_cmd(mc.Command.home("y"))
                await qmc.send_cmd(mc.Command.set_zero("y"))
                # home x again to engage clamp
                await qmc.send_cmd(mc.Command.home("x"))

                # move to an arbitrary position to disengage the clamp
                await qmc.send_cmd(mc.Command.move_by_mm("x", 30))
                await qmc.send_cmd(mc.Command.move_by_mm("y", 30))

                # and move objective up, slightly
                await qmc.send_cmd(mc.Command.move_by_mm("z", 1))

                logger.info("done initializing microscope")

            except IOError as e:
                logger.critical("lost connection to microcontroller")
                self.close()
                raise DisconnectError() from e

    async def snap_selected_channels(
        self, config_file: sc.AcquisitionConfig
    ) -> cmd.BasicSuccessResponse:
        """
        take a snapshot of all selected channels

        these images will be stored into the local buffer for immediate retrieval, i.e. NOT stored to disk.

        if autofocus is calibrated, this will automatically run the autofocus and take channel z offsets into account
        """

        if self.is_in_loading_position:
            cmd.error_internal(detail="now allowed while in loading position")

        with self.microcontroller.locked() as qmc:
            if qmc is None:
                error_internal("microcontroller is busy")

            # get machine config
            if config_file.machine_config is not None:
                GlobalConfigHandler.override(config_file.machine_config)

            g_config = GlobalConfigHandler.get_dict()

            laf_is_calibrated = g_config.get("laser_autofocus_is_calibrated")

            # get channels from that, filter for selected/enabled channels
            channels = [c for c in config_file.channels if c.enabled]

            # then:
            # if autofocus is available, measure and approach 0 in a loop up to 5 times
            try:
                current_stage_position = await qmc.get_last_position()
            except IOError as e:
                self.close()
                raise DisconnectError() from e
            reference_z_mm = current_stage_position.z_pos_mm

            if laf_is_calibrated is not None and laf_is_calibrated.boolvalue:
                for _ in range(5):
                    displacement_measure_data = await self.execute(
                        cmd.AutofocusMeasureDisplacement(config_file=config_file)
                    )

                    current_displacement_um = displacement_measure_data.displacement_um
                    assert current_displacement_um is not None

                    if math.fabs(current_displacement_um) < 0.5:
                        break

                    _ = await self.execute(
                        cmd.MoveBy(axis="z", distance_mm=-1e-3 * current_displacement_um)
                    )

                # then store current z coordinate as reference z
                try:
                    current_stage_position = await qmc.get_last_position()
                except IOError as e:
                    self.close()
                    raise DisconnectError() from e
                reference_z_mm = current_stage_position.z_pos_mm

            # then go through list of channels, and approach each channel with offset relative to reference z
            for channel in channels:
                _ = await self.execute(
                    cmd.MoveTo(
                        x_mm=None,
                        y_mm=None,
                        z_mm=reference_z_mm + channel.z_offset_um * 1e-3,
                    )
                )

                _ = await self.execute(cmd.ChannelSnapshot(channel=channel))

            return cmd.BasicSuccessResponse()

    @property
    def calibrated_stage_position(self) -> tuple[float, float, float]:
        """
        return calibrated XY stage offset from GlobalConfigHandler in order (x_mm,y_mm)
        """

        off_x_mm = GlobalConfigHandler.get_dict()["calibration_offset_x_mm"].floatvalue
        off_y_mm = GlobalConfigHandler.get_dict()["calibration_offset_y_mm"].floatvalue
        off_z_mm = GlobalConfigHandler.get_dict()["calibration_offset_z_mm"].floatvalue

        return (off_x_mm, off_y_mm, off_z_mm)

    # real position = measured position + calibrated offset

    def _pos_x_measured_to_real(self, x_mm: float) -> float:
        """convert measured x position to real position"""
        return x_mm + self.calibrated_stage_position[0]

    def _pos_y_measured_to_real(self, y_mm: float) -> float:
        """convert measured y position to real position"""
        return y_mm + self.calibrated_stage_position[1]

    def _pos_z_measured_to_real(self, z_mm: float) -> float:
        """convert measured z position to real position"""
        return z_mm + self.calibrated_stage_position[2]

    def _pos_x_real_to_measured(self, x_mm: float) -> float:
        """convert real x position to measured position"""
        return x_mm - self.calibrated_stage_position[0]

    def _pos_y_real_to_measured(self, y_mm: float) -> float:
        """convert real y position to measured position"""
        return y_mm - self.calibrated_stage_position[1]

    def _pos_z_real_to_measured(self, z_mm: float) -> float:
        """convert real z position to measured position"""
        return z_mm - self.calibrated_stage_position[2]

    def _get_peak_coords(
        self, img: np.ndarray, use_glass_top: bool = False, TOP_N_PEAKS: int = 2
    ) -> tuple[float, list[float]]:
        """
        get peaks in laser autofocus signal

        used to derive actual location information. by itself not terribly useful.

        returns rightmost_peak_x, distances_between_peaks
        """

        # 8 bit signal -> max value 255
        I_1d: np.ndarray = img.max(
            axis=0
        )  # use max to avoid issues with noise (sum is another option, but prone to issues with noise)
        x = np.array(range(len(I_1d)))
        y = I_1d

        # locate peaks == locate dots
        peak_locations, _ = scipy.signal.find_peaks(I_1d, distance=300, height=10)

        if len(peak_locations.tolist()) == 0:
            error_internal("no signal found")

        # order by height
        tallestpeaks_x = sorted(peak_locations.tolist(), key=lambda x: float(I_1d[x]))
        # pick top N
        tallestpeaks_x = tallestpeaks_x[-TOP_N_PEAKS:]
        # then order n tallest peaks by x
        tallestpeaks_x = sorted(tallestpeaks_x)

        # Find the rightmost (largest x) peak
        rightmost_peak: float = max(tallestpeaks_x)

        # Compute distances between consecutive peaks
        distances_between_peaks: list[float] = [
            tallestpeaks_x[i + 1] - tallestpeaks_x[i] for i in range(len(tallestpeaks_x) - 1)
        ]

        # Output rightmost peak and distances between consecutive peaks
        return rightmost_peak, distances_between_peaks

    async def _approximate_laser_af_z_offset_mm(
        self,
        calib_params: cmd.LaserAutofocusCalibrationData,
        _leftmostxinsteadofestimatedz: bool = False,
    ) -> float:
        """
        approximate current z offset (distance from current imaging plane to focus plane)

        args:
            calib_params:
            _leftmostxinsteadofestimatedz: if True, return the coordinate of the leftmost dot in the laser autofocus signal instead of the estimated z value that is based on this coordinate
        """

        g_config = GlobalConfigHandler.get_dict()

        conf_af_exp_ms_item = g_config["laser_autofocus_exposure_time_ms"]
        assert conf_af_exp_ms_item is not None
        conf_af_exp_ms = conf_af_exp_ms_item.floatvalue

        conf_af_exp_ag_item = g_config["laser_autofocus_analog_gain"]
        assert conf_af_exp_ag_item is not None
        conf_af_exp_ag = conf_af_exp_ag_item.floatvalue

        # get params to describe current signal
        res = await self.execute(
            cmd.AutofocusSnap(exposure_time_ms=conf_af_exp_ms, analog_gain=conf_af_exp_ag)
        )
        new_params = self._get_peak_coords(res._img)
        rightmost_x, interpeakdistances = new_params

        if len(interpeakdistances) == 0:
            leftmost_x = rightmost_x
        elif len(interpeakdistances) == 1:
            leftmost_x = rightmost_x - interpeakdistances[0]
        else:
            raise ValueError(f"expected 0 or 1 peaks, got {len(interpeakdistances)}")

        if _leftmostxinsteadofestimatedz:
            return leftmost_x

        def find_x_for_y(y_measured, regression_params):
            "find x (input value, z position) for given y (measured value, dot location on sensor)"
            slope, intercept = regression_params
            return (y_measured - intercept) / slope

        regression_params = (calib_params.um_per_px, calib_params.x_reference)

        return find_x_for_y(leftmost_x, regression_params)

    async def _laser_af_calibrate_here(
        self,
        Z_MM_MOVEMENT_RANGE_MM: float = 0.3,
        Z_MM_BACKLASH_COUNTER: float = 40e-3,
        NUM_Z_STEPS_CALIBRATE: int = 7,
        DEBUG_LASER_AF_CALIBRATION=bool(0),
        DEBUG_LASER_AF_SHOW_REGRESSION_FIT=bool(0),
        DEBUG_LASER_AF_SHOW_EVAL_FIT=True,
    ) -> cmd.LaserAutofocusCalibrationResponse:
        """
        see cmd.LaserAutofocusCalibrate
        """

        if self.is_in_loading_position:
            cmd.error_internal(detail="now allowed while in loading position")

        with self.microcontroller.locked() as qmc:
            if qmc is None:
                error_internal("microcontroller is busy")

            g_config = GlobalConfigHandler.get_dict()

            conf_af_exp_ms_item = g_config["laser_autofocus_exposure_time_ms"]
            assert conf_af_exp_ms_item is not None
            conf_af_exp_ms = conf_af_exp_ms_item.floatvalue

            conf_af_exp_ag_item = g_config["laser_autofocus_analog_gain"]
            assert conf_af_exp_ag_item is not None
            conf_af_exp_ag = conf_af_exp_ag_item.floatvalue

            z_step_mm = Z_MM_MOVEMENT_RANGE_MM / (NUM_Z_STEPS_CALIBRATE - 1)
            half_z_mm = Z_MM_MOVEMENT_RANGE_MM / 2

            try:
                current_pos = await qmc.get_last_position()
                _start_z_mm: float = current_pos.z_pos_mm

                # move down by half z range
                if Z_MM_BACKLASH_COUNTER != 0:  # is not None:
                    await qmc.send_cmd(
                        mc.Command.move_by_mm("z", -(half_z_mm + Z_MM_BACKLASH_COUNTER))
                    )
                    await qmc.send_cmd(mc.Command.move_by_mm("z", Z_MM_BACKLASH_COUNTER))
                else:
                    await qmc.send_cmd(mc.Command.move_by_mm("z", -half_z_mm))
            except IOError as e:
                self.close()
                raise DisconnectError() from e

            # Display each peak's height and width
            class CalibrationData(BaseModel):
                z_mm: float
                p: tuple[float, list[float]]

            async def measure_dot_params():
                # measure pos
                res = await self.execute(
                    cmd.AutofocusSnap(exposure_time_ms=conf_af_exp_ms, analog_gain=conf_af_exp_ag)
                )

                params = self._get_peak_coords(res._img)
                return params

            peak_info: list[CalibrationData] = []

            for i in range(NUM_Z_STEPS_CALIBRATE):
                if i > 0:
                    # move up by half z range to get position at original position, but moved to from fixed direction to counter backlash
                    try:
                        await qmc.send_cmd(mc.Command.move_by_mm("z", z_step_mm))
                    except IOError as e:
                        self.close()
                        raise DisconnectError() from e

                params = await measure_dot_params()

                peak_info.append(CalibrationData(z_mm=-half_z_mm + i * z_step_mm, p=params))

            # move to original position
            try:
                await qmc.send_cmd(mc.Command.move_by_mm("z", -half_z_mm))
            except IOError as e:
                self.close()
                raise DisconnectError() from e

            class DomainInfo(BaseModel):
                lowest_x: float
                highest_x: float
                peak_xs: list[tuple[list[float], list[float]]]

            # in the two peak domain, both dots are visible
            two_peak_domain = DomainInfo(lowest_x=3000, highest_x=0, peak_xs=[])
            # in the one peak domain, only one dot is visible (the one with the lower x in the two dot domain)
            one_peak_domain = DomainInfo(lowest_x=3000, highest_x=0, peak_xs=[])

            distances = []
            distance_x = []
            for i in peak_info:
                rightmost_x, p_distances = i.p

                new_distances = [rightmost_x]
                for p_d in p_distances:
                    new_distances.append(new_distances[-1] - p_d)

                new_z = [i.z_mm] * len(new_distances)

                distances.extend(new_distances)
                distance_x.extend(new_z)

                if len(p_distances) == 0:
                    target_domain = one_peak_domain
                elif len(p_distances) == 1:
                    target_domain = two_peak_domain
                else:
                    raise ValueError(f"expected 0 or 1 peaks, got {len(p_distances)}")

                for x in new_distances:
                    target_domain.lowest_x = min(target_domain.lowest_x, x)
                    target_domain.highest_x = max(target_domain.highest_x, x)
                    target_domain.peak_xs.append((new_distances, new_z))

            if DEBUG_LASER_AF_CALIBRATION and DEBUG_LASER_AF_SHOW_REGRESSION_FIT:
                plt.figure(figsize=(8, 6))
                plt.scatter(distance_x, distances, color="blue", label="all peaks")

            # x is dot x
            # y is z coordinate
            left_dot_x: list[float] = []
            left_dot_y: list[float] = []
            right_dot_x: list[float] = []
            right_dot_y: list[float] = []

            for peak_y, peak_x in one_peak_domain.peak_xs:
                left_dot_x.append(peak_x[0])
                left_dot_y.append(peak_y[0])

            for peak_y, peak_x in two_peak_domain.peak_xs:
                right_dot_x.append(peak_x[0])
                right_dot_y.append(peak_y[0])
                left_dot_x.append(peak_x[1])
                left_dot_y.append(peak_y[1])

            left_dot_regression = linear_regression(left_dot_x, left_dot_y)
            right_dot_regression = 0, 0
            if len(right_dot_x) > 0:
                # there are at least two possible issues here, that we ignore:
                # 1) no values present (zero/low signal)
                # 2) all values are identical (caused by only measuring noise)
                try:
                    right_dot_regression = linear_regression(right_dot_x, right_dot_y)
                except Exception:
                    pass

            if DEBUG_LASER_AF_CALIBRATION and DEBUG_LASER_AF_SHOW_REGRESSION_FIT:
                # plot one peak domain
                domain_x = []
                domain_y = []
                for py, pz in one_peak_domain.peak_xs:
                    domain_x.extend(pz)
                    domain_y.extend(py)

                plt.scatter(
                    domain_x,
                    domain_y,
                    color="green",
                    marker="x",
                    label="one peak domain",
                )

                # plot two peak domain
                domain_x = []
                domain_y = []
                for py, pz in two_peak_domain.peak_xs:
                    domain_x.extend(pz)
                    domain_y.extend(py)

                plt.scatter(domain_x, domain_y, color="red", marker="x", label="two peak domain")

                # plot left dot regression
                slope, intercept = left_dot_regression
                plt.axline(
                    (0, intercept),
                    slope=slope,
                    color="purple",
                    label="left dot regression",
                )

                # plot right dot regression
                slope, intercept = right_dot_regression
                plt.axline(
                    (0, intercept),
                    slope=slope,
                    color="black",
                    label="right dot regression",
                )

                plt.xlabel("physical z coordinate")
                plt.ylabel("sensor x coordinate")
                plt.legend()
                plt.grid(True)
                plt.show()

            # -- eval performance, display with pyplot
            if DEBUG_LASER_AF_CALIBRATION and DEBUG_LASER_AF_SHOW_EVAL_FIT:
                current_pos = await qmc.get_last_position()
                start_z = current_pos.z_pos_mm

                half_z_mm = Z_MM_MOVEMENT_RANGE_MM / 2
                num_z_steps_eval = 51
                z_step_mm = Z_MM_MOVEMENT_RANGE_MM / (num_z_steps_eval - 1)

                if Z_MM_BACKLASH_COUNTER != 0:  # is not None:
                    await qmc.send_cmd(
                        mc.Command.move_by_mm("z", -(half_z_mm + Z_MM_BACKLASH_COUNTER))
                    )
                    await qmc.send_cmd(mc.Command.move_by_mm("z", Z_MM_BACKLASH_COUNTER))
                else:
                    await qmc.send_cmd(mc.Command.move_by_mm("z", -half_z_mm))

                approximated_z: list[tuple[float, float]] = []
                for i in range(num_z_steps_eval):
                    if i > 0:
                        # move up by half z range to get position at original position, but moved to from fixed direction to counter backlash
                        await qmc.send_cmd(mc.Command.move_by_mm("z", z_step_mm))

                    approx = await self._approximate_laser_af_z_offset_mm(
                        cmd.LaserAutofocusCalibrationData(
                            um_per_px=left_dot_regression[0],
                            x_reference=left_dot_regression[1],
                            calibration_position=cmd.Position.zero(),
                        )
                    )
                    current_pos = await qmc.get_last_position()
                    current_z_mm = current_pos.z_pos_mm
                    approximated_z.append((current_z_mm - start_z, approx))

                # move to original position
                await qmc.send_cmd(mc.Command.move_by_mm("z", -half_z_mm))

                plt.figure(figsize=(8, 6))
                plt.scatter(
                    [v[0] * 1e3 for v in approximated_z],
                    [v[0] * 1e3 for v in approximated_z],
                    color="green",
                    label="real/expected",
                    marker="o",
                    linestyle="-",
                )
                plt.scatter(
                    [v[0] * 1e3 for v in approximated_z],
                    [v[1] * 1e3 for v in approximated_z],
                    color="blue",
                    label="estimated",
                    marker="o",
                    linestyle="-",
                )
                plt.scatter(
                    [v[0] * 1e3 for v in approximated_z],
                    [(v[1] - v[0]) * 1e3 for v in approximated_z],
                    color="red",
                    label="error",
                    marker="o",
                    linestyle="-",
                )
                plt.xlabel("real z [um]")
                plt.ylabel("measured z [um]")
                plt.legend()
                plt.grid(True)
                plt.show()

            um_per_px, _x_reference = left_dot_regression
            x_reference = (await measure_dot_params())[0]

            try:
                calibration_position = await qmc.get_last_position()
            except IOError as e:
                self.close()
                raise DisconnectError() from e

            calibration_data = cmd.LaserAutofocusCalibrationData(
                # calculate the conversion factor, based on lowest and highest measured position
                um_per_px=um_per_px,
                # set reference position
                x_reference=x_reference,
                calibration_position=calibration_position.pos,
            )

            return cmd.LaserAutofocusCalibrationResponse(calibration_data=calibration_data)

    async def get_current_state(self) -> AdapterState:
        try:
            last_stage_position = await self.microcontroller.get_last_position()
        except IOError as e:
            self.close()
            raise DisconnectError() from e

        # supposed=real-calib
        x_pos_mm = self._pos_x_measured_to_real(last_stage_position.x_pos_mm)
        y_pos_mm = self._pos_y_measured_to_real(last_stage_position.y_pos_mm)
        z_pos_mm = self._pos_z_measured_to_real(last_stage_position.z_pos_mm)

        self.last_state = AdapterState(
            state=self.state,
            is_in_loading_position=self.is_in_loading_position,
            stage_position=cmd.Position(
                x_pos_mm=x_pos_mm,
                y_pos_mm=y_pos_mm,
                z_pos_mm=z_pos_mm,
            ),
        )

        return self.last_state

    async def execute[T](self, command: cmd.BaseCommand[T]) -> T:
        with self.microcontroller.locked() as qmc:
            if qmc is None:
                error_internal("microcontroller is busy")

            try:
                logger.debug(f"squid - executing {type(command).__qualname__}")

                if isinstance(command, cmd.EstablishHardwareConnection):
                    if not self.is_connected:
                        try:
                            # if no connection has yet been established, connecting will have the hardware in an undefined state
                            logger.debug("squid - connect - connecting to hardware")
                            self.open_connections()
                            # so after connecting:
                            # 1) turn off all illumination
                            logger.debug("squid - connect - turning off illumination")
                            await self.execute(IlluminationEndAll())
                            logger.debug("squid - connect - turned off illumination")
                            # 2) perform home maneuver to reset stage position to known values
                            logger.debug("squid - connect - calibrating stage position")

                            await self.home()

                            logger.info("squid - connect - calibrated stage")

                        except DisconnectError:
                            error_message = "hardware connection could not be established"
                            logger.critical(f"squid - connect - {error_message}")
                            error_internal(error_message)

                    logger.debug("squid - connected.")

                    result = BasicSuccessResponse()
                    return result  # type: ignore[no-any-return]

                elif isinstance(command, cmd.LoadingPositionEnter):
                    if self.is_in_loading_position:
                        cmd.error_internal(detail="already in loading position")

                    self.state = CoreState.Moving

                    # home z
                    await qmc.send_cmd(mc.Command.home("z"))

                    # clear clamp in y first
                    await qmc.send_cmd(mc.Command.move_to_mm("y", 30))
                    # then clear clamp in x
                    await qmc.send_cmd(mc.Command.move_to_mm("x", 30))

                    # then home y, x
                    await qmc.send_cmd(mc.Command.home("y"))
                    await qmc.send_cmd(mc.Command.home("x"))

                    self.is_in_loading_position = True

                    self.state = CoreState.LoadingPosition

                    logger.debug("squid - entered loading position")

                    result = cmd.BasicSuccessResponse()
                    return result  # type: ignore[no-any-return]

                elif isinstance(command, cmd.LoadingPositionLeave):
                    if not self.is_in_loading_position:
                        cmd.error_internal(detail="not in loading position")

                    self.state = CoreState.Moving

                    await qmc.send_cmd(mc.Command.move_to_mm("x", 30))
                    await qmc.send_cmd(mc.Command.move_to_mm("y", 30))
                    await qmc.send_cmd(mc.Command.move_to_mm("z", 1))

                    self.is_in_loading_position = False

                    self.state = CoreState.Idle

                    logger.debug("squid - left loading position")

                    result = cmd.BasicSuccessResponse()
                    return result  # type: ignore[no-any-return]

                elif isinstance(command, cmd.MoveBy):
                    if self.is_in_loading_position:
                        cmd.error_internal(detail="now allowed while in loading position")

                    self.state = CoreState.Moving

                    await qmc.send_cmd(mc.Command.move_by_mm(command.axis, command.distance_mm))

                    self.state = CoreState.Idle

                    logger.debug("squid - moved by")

                    result = cmd.MoveByResult(moved_by_mm=command.distance_mm, axis=command.axis)
                    return result  # type: ignore[no-any-return]

                elif isinstance(command, cmd.MoveTo):
                    if self.is_in_loading_position:
                        cmd.error_internal(detail="now allowed while in loading position")

                    if command.x_mm is not None and command.x_mm < 0:
                        cmd.error_internal(detail=f"x coordinate out of bounds {command.x_mm = }")
                    if command.y_mm is not None and command.y_mm < 0:
                        cmd.error_internal(detail=f"y coordinate out of bounds {command.y_mm = }")
                    if command.z_mm is not None and command.z_mm < 0:
                        cmd.error_internal(detail=f"z coordinate out of bounds {command.z_mm = }")

                    prev_state = self.state
                    self.state = CoreState.Moving

                    approach_x_before_y = True

                    if command.x_mm is not None and command.y_mm is not None:
                        current_stage_position = await qmc.get_last_position()

                        # plate center is (very) rougly at x=61mm, y=40mm
                        # we have: start position, target position, and two possible edges to move across

                        center = 61.0, 40.0
                        _start = (
                            current_stage_position.x_pos_mm,
                            current_stage_position.y_pos_mm,
                        )
                        _target = command.x_mm, command.y_mm

                        # if edge1 is closer to center, then approach_x_before_y=True, else approach_x_before_y=False
                        edge1 = command.x_mm, current_stage_position.y_pos_mm
                        edge2 = current_stage_position.x_pos_mm, command.y_mm

                        def dist(p1: tuple[float, float], p2: tuple[float, float]) -> float:
                            return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

                        approach_x_before_y = dist(edge1, center) < dist(edge2, center)

                        # we want to choose the edge that is closest to the center, because this avoid moving through the forbidden plate corners

                    if approach_x_before_y:
                        if command.x_mm is not None:
                            x_mm = self._pos_x_real_to_measured(command.x_mm)
                            if x_mm < 0:
                                cmd.error_internal(
                                    detail=f"calibrated x coordinate out of bounds {x_mm = }"
                                )
                            await qmc.send_cmd(mc.Command.move_to_mm("x", x_mm))

                        if command.y_mm is not None:
                            y_mm = self._pos_y_real_to_measured(command.y_mm)
                            if y_mm < 0:
                                cmd.error_internal(
                                    detail=f"calibrated y coordinate out of bounds {y_mm = }"
                                )
                            await qmc.send_cmd(mc.Command.move_to_mm("y", y_mm))
                    else:
                        if command.y_mm is not None:
                            y_mm = self._pos_y_real_to_measured(command.y_mm)
                            if y_mm < 0:
                                cmd.error_internal(
                                    detail=f"calibrated y coordinate out of bounds {y_mm = }"
                                )
                            await qmc.send_cmd(mc.Command.move_to_mm("y", y_mm))

                        if command.x_mm is not None:
                            x_mm = self._pos_x_real_to_measured(command.x_mm)
                            if x_mm < 0:
                                cmd.error_internal(
                                    detail=f"calibrated x coordinate out of bounds {x_mm = }"
                                )
                            await qmc.send_cmd(mc.Command.move_to_mm("x", x_mm))

                    if command.z_mm is not None:
                        z_mm = self._pos_z_real_to_measured(command.z_mm)
                        if z_mm < 0:
                            cmd.error_internal(
                                detail=f"calibrated z coordinate out of bounds {z_mm = }"
                            )
                        await qmc.send_cmd(mc.Command.move_to_mm("z", z_mm))

                    self.state = prev_state

                    logger.debug("squid - moved to")

                    result = cmd.BasicSuccessResponse()
                    return result  # type: ignore[no-any-return]

                elif isinstance(command, cmd.MoveToWell):
                    if self.is_in_loading_position:
                        cmd.error_internal(detail="now allowed while in loading position")

                    plate = command.plate_type

                    if cmd.wellIsForbidden(command.well_name, plate):
                        cmd.error_internal(detail="well is forbidden")

                    x_mm = plate.get_well_offset_x(command.well_name) + plate.Well_size_x_mm / 2
                    y_mm = plate.get_well_offset_y(command.well_name) + plate.Well_size_y_mm / 2

                    # move in 2d grid, no diagonals (slower, but safer)
                    res = await self.execute(cmd.MoveTo(x_mm=x_mm, y_mm=None))
                    res = await self.execute(cmd.MoveTo(x_mm=None, y_mm=y_mm))

                    logger.debug("squid - moved to well")

                    result = cmd.BasicSuccessResponse()
                    return result  # type: ignore[no-any-return]

                elif isinstance(command, cmd.AutofocusMeasureDisplacement):
                    if command.config_file.machine_config is not None:
                        GlobalConfigHandler.override(command.config_file.machine_config)

                    g_config = GlobalConfigHandler.get_dict()

                    conf_af_if_calibrated = g_config.get("laser_autofocus_is_calibrated")
                    conf_af_calib_x = g_config.get("laser_autofocus_calibration_x")
                    conf_af_calib_umpx = g_config.get("laser_autofocus_calibration_umpx")
                    if (
                        conf_af_if_calibrated is None
                        or conf_af_calib_x is None
                        or conf_af_calib_umpx is None
                        or not conf_af_if_calibrated.boolvalue
                    ):
                        cmd.error_internal(detail="laser autofocus not calibrated")

                    # get laser spot location
                    # sometimes one of the two expected dots cannot be found in _get_laser_spot_centroid because the plate is so far off the focus plane though, catch that case
                    try:
                        calib_params = cmd.LaserAutofocusCalibrationData(
                            um_per_px=conf_af_calib_umpx.floatvalue,
                            x_reference=conf_af_calib_x.floatvalue,
                            calibration_position=cmd.Position.zero(),
                        )
                        displacement_um = 0

                        num_images = 3 or command.override_num_images
                        for _ in range(num_images):
                            latest_esimated_z_offset_mm = (
                                await self._approximate_laser_af_z_offset_mm(calib_params)
                            )
                            displacement_um += latest_esimated_z_offset_mm * 1e3 / num_images

                    except Exception as e:
                        cmd.error_internal(
                            detail=f"failed to measure displacement (got no signal): {e!s}"
                        )

                    logger.debug("squid - used autofocus to measure displacement")

                    result = cmd.AutofocusMeasureDisplacementResult(displacement_um=displacement_um)
                    return result  # type: ignore[no-any-return]

                elif isinstance(command, cmd.AutofocusSnap):
                    if command.turn_laser_on:
                        await qmc.send_cmd(mc.Command.af_laser_illum_begin())
                        logger.debug("squid - autofocus - turned laser on")

                    channel_config = sc.AcquisitionChannelConfig(
                        name="Laser Autofocus",  # unused
                        handle="laser_autofocus",  # unused
                        illum_perc=100,  # unused
                        exposure_time_ms=command.exposure_time_ms,
                        analog_gain=command.analog_gain,
                        z_offset_um=0,  # unused
                        num_z_planes=0,  # unused
                        delta_z_um=0,  # unused
                    )

                    with self.focus_camera() as focus_camera:
                        if focus_camera is None:
                            error_internal("focus camera is busy")
                        img = focus_camera.acquire_with_config(channel_config)

                    logger.debug("squid - autofocus - acquired image")
                    if img is None:
                        self.state = CoreState.Idle
                        cmd.error_internal(detail="failed to acquire image")

                    if command.turn_laser_off:
                        await qmc.send_cmd(mc.Command.af_laser_illum_end())
                        logger.debug("squid - autofocus - turned laser off")

                    result = cmd.AutofocusSnapResult(
                        width_px=img.shape[1],
                        height_px=img.shape[0],
                    )

                    # blur laser autofocus image to get rid of some noise
                    # img = scipy.ndimage.gaussian_filter(img, sigma=1.0) # this takes 5 times as long as cv2
                    img = cv2.GaussianBlur(img, (3, 3), sigmaX=1.0, borderType=cv2.BORDER_DEFAULT)
                    logger.debug("squid - autofocus - applied blur to image")

                    result._img = img
                    result._channel = channel_config
                    return result  # type: ignore[no-any-return]

                elif isinstance(command, cmd.AutofocusLaserWarmup):
                    await qmc.send_cmd(mc.Command.af_laser_illum_begin())

                    # wait for the laser to warm up
                    await asyncio.sleep(command.warmup_time_s)

                    await qmc.send_cmd(mc.Command.af_laser_illum_end())

                    logger.debug("squid - warmed up autofocus laser")

                    result = cmd.BasicSuccessResponse()
                    return result  # type: ignore[no-any-return]

                elif isinstance(command, cmd.IlluminationEndAll):
                    # make sure all illumination is off
                    for illum_src in [
                        mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_405NM,
                        mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_488NM,
                        mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_561NM,
                        mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_638NM,
                        mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_730NM,
                        mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_LED_ARRAY_FULL,  # this will turn off the led matrix
                    ]:
                        await qmc.send_cmd(mc.Command.illumination_end(illum_src))
                        logger.debug(
                            f"squid - illumination end all - turned off illumination for {illum_src}"
                        )

                    logger.debug("squid - turned off all illumination")

                    ret = cmd.BasicSuccessResponse()
                    return ret  # type: ignore[no-any-return]

                elif isinstance(command, cmd.ChannelSnapshot):
                    try:
                        illum_code = mc.ILLUMINATION_CODE.from_handle(command.channel.handle)
                    except Exception:
                        cmd.error_internal(
                            detail=f"invalid channel handle: {command.channel.handle}"
                        )

                    GlobalConfigHandler.override(command.machine_config)

                    if self.stream_callback is not None:
                        cmd.error_internal(detail="already streaming")

                    self.state = CoreState.ChannelSnap

                    logger.debug("channel snap - before illum on")
                    await qmc.send_cmd(
                        mc.Command.illumination_begin(illum_code, command.channel.illum_perc)
                    )

                    with self.main_camera() as main_camera:
                        if main_camera is None:
                            error_internal("main camera is busy")

                        logger.debug("channel snap - before acq")
                        img = main_camera.acquire_with_config(command.channel)
                        logger.debug("channel snap - after acq")
                        await qmc.send_cmd(mc.Command.illumination_end(illum_code))
                        logger.debug("channel snap - after illum off")
                        if img is None:
                            logger.critical("failed to acquire image")
                            self.state = CoreState.Idle
                            cmd.error_internal(detail="failed to acquire image")

                        self.state = CoreState.Idle

                        img, cambits = _process_image(img, camera=main_camera)

                    logger.debug("squid - took channel snapshot")

                    result = cmd.ImageAcquiredResponse()
                    result._img = img
                    result._cambits = cambits
                    return result  # type: ignore[no-any-return]

                elif isinstance(command, cmd.ChannelSnapSelection):
                    channel_handles: list[str] = []
                    channel_images: dict[str, np.ndarray] = {}
                    for channel in command.config_file.channels:
                        if not channel.enabled:
                            continue

                        cmd_snap = cmd.ChannelSnapshot(
                            channel=channel,
                            machine_config=command.config_file.machine_config or [],
                        )
                        res = await self.execute(cmd_snap)

                        channel_images[channel.handle] = res._img
                        channel_handles.append(channel.handle)

                    logger.debug("squid - took snapshot in multiple channels")

                    result = cmd.ChannelSnapSelectionResult(channel_handles=channel_handles)
                    result._images = channel_images
                    return result  # type: ignore[no-any-return]

                elif isinstance(command, cmd.ChannelStreamBegin):
                    if self.stream_callback is not None:
                        cmd.error_internal(detail="already streaming")

                    try:
                        illum_code = mc.ILLUMINATION_CODE.from_handle(command.channel.handle)
                    except Exception:
                        cmd.error_internal(
                            detail=f"invalid channel handle: {command.channel.handle}"
                        )

                    GlobalConfigHandler.override(command.machine_config)

                    self.state = CoreState.ChannelStream

                    await qmc.send_cmd(
                        mc.Command.illumination_begin(illum_code, command.channel.illum_perc)
                    )

                    # returns true if should stop
                    forward_image_callback = {}

                    def forward_image(img: gxiapi.RawImage) -> bool:
                        if self.stream_callback is not None:
                            forward_image_callback["callback"] = self.stream_callback

                            match img.get_status():
                                case gxiapi.GxFrameStatusList.INCOMPLETE:
                                    cmd.error_internal(detail="incomplete frame")
                                case gxiapi.GxFrameStatusList.SUCCESS:
                                    pass

                            img_np = img.get_numpy_array()
                            assert img_np is not None
                            img_np = img_np.copy()

                            # camera must only be locked for specific image acquisition, not for the whole duration where an image may be acquired
                            # there is currently no way to solve this well.. (this current solution is quite brittle)
                            with self.main_camera() as main_camera:
                                if main_camera is None:
                                    error_internal("main camera is busy")
                                img_np, cambits = _process_image(img_np, camera=main_camera)

                            return forward_image_callback["callback"](img_np)
                        else:
                            return forward_image_callback["callback"](True)

                    with self.main_camera() as main_camera:
                        if main_camera is None:
                            error_internal("main camera is busy")

                        main_camera.acquire_with_config(
                            command.channel,
                            mode="until_stop",
                            callback=forward_image,
                            target_framerate_hz=command.framerate_hz,
                        )

                    logger.debug("squid - channel stream begin")

                    result = cmd.StreamingStartedResponse(channel=command.channel)
                    return result  # type: ignore[no-any-return]

                elif isinstance(command, cmd.ChannelStreamEnd):
                    with self.main_camera() as main_camera:
                        if main_camera is None:
                            error_internal("main camera is busy")

                        if (self.stream_callback is None) or (not main_camera.acquisition_ongoing):
                            cmd.error_internal(detail="not currently streaming")

                        try:
                            illum_code = mc.ILLUMINATION_CODE.from_handle(command.channel.handle)
                        except Exception:
                            cmd.error_internal(
                                detail=f"invalid channel handle: {command.channel.handle}"
                            )

                        GlobalConfigHandler.override(command.machine_config)

                        self.stream_callback = None
                        await qmc.send_cmd(mc.Command.illumination_end(illum_code))
                        # cancel ongoing acquisition
                        main_camera.acquisition_ongoing = False
                        main_camera._set_acquisition_mode(AcquisitionMode.ON_TRIGGER)

                    self.state = CoreState.Idle

                    logger.debug("squid - channel stream end")

                    result = cmd.BasicSuccessResponse()
                    return result  # type: ignore[no-any-return]

                elif isinstance(command, cmd.LaserAutofocusCalibrate):
                    result = await self._laser_af_calibrate_here(**command.dict())
                    logger.debug("squid - calibrated laser autofocus")
                    return result  # type: ignore[no-any-return]

                elif isinstance(command, cmd.AutofocusApproachTargetDisplacement):

                    async def _estimate_offset_mm():
                        res = await self.execute(
                            cmd.AutofocusMeasureDisplacement(config_file=command.config_file)
                        )

                        current_displacement_um = res.displacement_um
                        assert current_displacement_um is not None

                        return (command.target_offset_um - current_displacement_um) * 1e-3

                    if self.is_in_loading_position:
                        cmd.error_internal(detail="now allowed while in loading position")

                    if self.state != CoreState.Idle:
                        cmd.error_internal(detail="cannot move while in non-idle state")

                    g_config = GlobalConfigHandler.get_dict()

                    # get autofocus calibration data
                    conf_af_calib_x = g_config["laser_autofocus_calibration_x"].floatvalue
                    conf_af_calib_umpx = g_config["laser_autofocus_calibration_umpx"].floatvalue
                    # autofocus_calib=LaserAutofocusCalibrationData(um_per_px=conf_af_calib_umpx,x_reference=conf_af_calib_x,calibration_position=Position.zero())

                    # we are looking for a z coordinate where the measured dot_x is equal to this target_x.
                    # we can estimate the current z offset based on the currently measured dot_x.
                    # then we loop:
                    #   we move by the estimated offset to reduce the difference between target_x and dot_x.
                    #   then we check if we are at target_x.
                    #     if we have not reached it, we move further in that direction, based on another estimate.
                    #     if have overshot (moved past) it, we move back by some estimate.
                    #     terminate when dot_x is within a margin of target_x.

                    OFFSET_MOVEMENT_THRESHOLD_MM = 0.5e-3

                    current_state = await qmc.get_last_position()
                    current_z = current_state.z_pos_mm
                    initial_z = current_z

                    if command.pre_approach_refz:
                        gconfig_refzmm_item = g_config.get("laser_autofocus_calibration_refzmm")
                        if gconfig_refzmm_item is None:
                            cmd.error_internal(
                                detail="laser_autofocus_calibration_refzmm is not available when AutofocusApproachTargetDisplacement had pre_approach_refz set"
                            )

                        # move to reference z, only if it is far enough away to make a move worth it
                        if (
                            math.fabs(current_z - gconfig_refzmm_item.floatvalue)
                            > OFFSET_MOVEMENT_THRESHOLD_MM
                        ):
                            res = await self.execute(
                                cmd.MoveTo(
                                    x_mm=None,
                                    y_mm=None,
                                    z_mm=gconfig_refzmm_item.floatvalue,
                                )
                            )

                        logger.debug("autofocus - did pre approach refz")

                    old_state = self.state
                    self.state = CoreState.Moving

                    last_distance_estimate_mm = 0.0
                    num_compensating_moves = 0
                    reached_threshold = False
                    try:
                        last_distance_estimate_mm = await _estimate_offset_mm()
                        logger.debug("autofocus - estimated offset")
                        last_z_mm = (await qmc.get_last_position()).z_pos_mm
                        MAX_MOVEMENT_RANGE_MM = 0.3  # should be derived from the calibration data, but this value works fine in practice
                        if math.fabs(last_distance_estimate_mm) > MAX_MOVEMENT_RANGE_MM:
                            cmd.error_internal(
                                detail="measured autofocus focal plane offset too large"
                            )

                        for rep_i in range(command.max_num_reps):
                            if rep_i == 0:
                                distance_estimate_mm = last_distance_estimate_mm
                            else:
                                distance_estimate_mm = await _estimate_offset_mm()
                                logger.debug("autofocus - estimated offset")

                            # stop if the new estimate indicates a larger distance to the focal plane than the previous estimate
                            # (since this indicates that we have moved away from the focal plane, which should not happen)
                            if rep_i > 0 and math.fabs(last_distance_estimate_mm) < math.fabs(
                                distance_estimate_mm
                            ):
                                # move back to last z, since that seemed like the better position to be in
                                await qmc.send_cmd(mc.Command.move_to_mm("z", last_z_mm))
                                logger.debug("autofocus - reset z to known good position")
                                # TODO unsure if this is the best approach. we cannot do better, but we also have not actually gotten close to the offset
                                reached_threshold = True
                                break

                            last_distance_estimate_mm = distance_estimate_mm
                            last_z_mm = (await qmc.get_last_position()).z_pos_mm

                            # if movement distance is not worth compensating, stop
                            if math.fabs(distance_estimate_mm) < OFFSET_MOVEMENT_THRESHOLD_MM:
                                reached_threshold = True
                                break

                            await qmc.send_cmd(mc.Command.move_by_mm("z", distance_estimate_mm))
                            num_compensating_moves += 1
                            logger.debug("autofocus - refined z")

                    except Exception:
                        # if any interaction failed, attempt to reset z position to known somewhat-good position
                        await qmc.send_cmd(mc.Command.move_to_mm("z", initial_z))
                        logger.debug("autofocus - reset z position")
                    finally:
                        self.state = old_state

                    logger.debug("squid - used autofocus to approach target displacement")

                    res = cmd.AutofocusApproachTargetDisplacementResult(
                        num_compensating_moves=num_compensating_moves,
                        uncompensated_offset_mm=last_distance_estimate_mm,
                        reached_threshold=reached_threshold,
                    )
                    return res  # type: ignore[no-any-return]

                elif isinstance(command, cmd.MC_getLastPosition):
                    logger.debug("squid - fetching last stage position")
                    res = await qmc.get_last_position()
                    logger.debug("squid - fetched last stage position")
                    return res  # type: ignore[no-any-return]

                else:
                    cmd.error_internal(detail=f"Unsupported command type {type(command)}")

            except gxiapi.OffLine as e:
                logger.critical("squid - lost camera connection")
                raise DisconnectError() from e
            except IOError as e:
                logger.critical("squid - lost connection to microcontroller")
                raise DisconnectError() from e
