import math
import os
import time
import typing as tp

import cv2
import numpy as np
import scipy  # to find peaks in a signal

# import scipy.ndimage  # for guassian blur
import seaconfig as sc

from matplotlib import pyplot as plt
from fastapi import HTTPException
from pydantic import BaseModel
from scipy import stats  # for linear regression

from seafront.config.basics import (
    CameraDriver,
    ChannelConfig,
    ConfigItem,
    FilterConfig,
    GlobalConfigHandler,
)
from seafront.config.core_config import check_device_ids_configured
from seafront.config.registry import ConfigRegistry
from seafront.config.handles import (
    CalibrationConfig,
    CameraConfig,
    FilterWheelConfig,
    ImageConfig,
    ImagingConfig,
    LaserAutofocusConfig,
    MicrocontrollerConfig,
    SystemConfig,
)
from seafront.hardware.microcontroller import (
    Microcontroller,
    MicrocontrollerDriver,
    MicrocontrollerOpenRequest,
    microcontroller_open,
)
from seafront.hardware.microcontroller.teensy_microcontroller import (
    ILLUMINATION_CODE,
)
from seafront.hardware.adapter import AdapterState, Position
from seafront.hardware.camera import (
    Camera,
    CameraOfflineError,
    CameraOpenRequest,
    HardwareLimitValue,
    bits_for_pixel_format,
    camera_open,
)
from seafront.hardware.illumination import IlluminationController
from seafront.hardware.adapter import DeviceAlreadyInUseError
from seafront.hardware.health import Device, DeviceHealth, HardwareCloseError, describe
from seafront.hardware.microscope import DisconnectError, HardwareLimits, Locked, Microscope
from seafront.logger import logger
from seafront.server import commands as cmd
from seafront.server.commands import (
    BasicSuccessResponse,
    IlluminationEndAll,
    error_device_in_use,
    error_internal,
)


# Deadlines on the hardware calls that can hang (see hardware/watchdog.py): the camera SDKs,
# and opening and closing. Each is far past anything the call takes, including the drivers'
# own retries and reconnects, so reaching one means the call is stuck.
OPEN_DEADLINE_S = 60.0
CALL_DEADLINE_S = 30.0
"close, stream start/stop, and the queries (limits, pixel formats)"
SNAP_DEADLINE_MARGIN_S = 30.0
"a snap's deadline is its exposure time plus this"

# utility functions


def linear_regression(
    x: list[float] | np.ndarray, y: list[float] | np.ndarray
) -> tuple[float, float]:
    "returns (slope,intercept)"
    slope, intercept, _, _, _ = stats.linregress(x, y)
    return slope, intercept


def _process_image(img: np.ndarray, pixel_format: str) -> tuple[np.ndarray, int]:
    """
    center crop main camera image to target size

    also pad so that top bits are used. e.g. if imaging bit depth is 12, top 12 bits will be set.

    Takes ``pixel_format`` (e.g. "mono12") rather than the camera object so it can run
    without holding the camera lock -- this is called from the streaming capture callback,
    where holding the camera lock would deadlock against stream_off()/unregister_capture_callback().
    """

    cam_img_width = ConfigRegistry.get(CameraConfig.MAIN_IMAGE_WIDTH_PX)
    assert cam_img_width is not None
    target_width: int = cam_img_width.intvalue
    assert isinstance(target_width, int), f"{type(target_width) = }"
    cam_img_height = ConfigRegistry.get(CameraConfig.MAIN_IMAGE_HEIGHT_PX)
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

    flip_img_horizontal = ConfigRegistry.get(CameraConfig.MAIN_IMAGE_FLIP_HORIZONTAL)
    assert flip_img_horizontal is not None
    if flip_img_horizontal.boolvalue:
        ret = np.flip(ret, axis=1)

    flip_img_vertical = ConfigRegistry.get(CameraConfig.MAIN_IMAGE_FLIP_VERTICAL)
    assert flip_img_vertical is not None
    if flip_img_vertical.boolvalue:
        ret = np.flip(ret, axis=0)

    image_file_pad_low = ConfigRegistry.get(ImageConfig.FILE_PAD_LOW)
    assert image_file_pad_low is not None
    cambits = bits_for_pixel_format(pixel_format)

    if image_file_pad_low.boolvalue:
        # e.g. ret = ret << (16 - 12)
        bytes_per_pixel = ret.dtype.itemsize
        bits_per_pixel = bytes_per_pixel * 8
        if bits_per_pixel != cambits:
            ret = ret << (bits_per_pixel - cambits)

    return ret, cambits


def _connection_lost(error: BaseException) -> bool:
    "whether an error means the device is gone, rather than that one operation on it failed"
    return isinstance(error, (CameraOfflineError, OSError))


def _raise_collected(failures: list[tuple[str, BaseException]]) -> None:
    """
    Raise after a run of steps that were each attempted whatever happened to the others.

    One error comes out, so the execute boundary can act on it: the first that means a
    connection was lost if there is one (that decides whether the hardware gets dropped),
    otherwise the first. Every other failure rides along as a note.
    """
    if not failures:
        return
    ordered = sorted(failures, key=lambda failure: not _connection_lost(failure[1]))
    (_, primary), rest = ordered[0], ordered[1:]
    for step, error in rest:
        primary.add_note(f"also failed - {step}: {describe(error)}")
    raise primary


class LaserAutofocusMeasurement(BaseModel):
    """Result of measuring the laser autofocus signal from an image."""
    rightmost_x: float
    """x coordinate of the rightmost detected peak"""
    interpeakdistances: list[float]
    """distances between consecutive peaks (right to left, empty if only one peak, one element if two peaks)"""
    tracked_dot_x: float
    """x coordinate of the tracked dot (depends on use_right_dot setting)"""
    estimated_displacement_um: float | None
    """estimated displacement in micrometers from reference, or None if no calibration data provided"""

class PeakData(BaseModel):
    loc_x:float
    height:float

class SquidAdapter(Microscope):
    """interface to squid microscope"""

    # SQUID-specific hardware components
    main_camera: Locked[Camera]
    focus_camera: Locked[Camera]
    microcontroller: Microcontroller
    illumination_controller: IlluminationController

    @classmethod
    def make(cls) -> "SquidAdapter":
        # refuse placeholder / empty device ids before any driver is asked to find them
        check_device_ids_configured(
            microscope_name=ConfigRegistry.get(SystemConfig.MICROSCOPE_NAME).strvalue,
            config_file=str(GlobalConfigHandler.home_config()),
        )

        # Get device configuration (USB IDs and drivers)
        main_camera_id = ConfigRegistry.get(CameraConfig.MAIN_ID).strvalue
        main_camera_driver = ConfigRegistry.get(CameraConfig.MAIN_DRIVER).strvalue
        focus_camera_id = ConfigRegistry.get(LaserAutofocusConfig.CAMERA_ID).strvalue
        focus_camera_driver = ConfigRegistry.get(LaserAutofocusConfig.CAMERA_DRIVER).strvalue
        microcontroller_id = ConfigRegistry.get(MicrocontrollerConfig.ID).strvalue
        microcontroller_driver = ConfigRegistry.get(MicrocontrollerConfig.DRIVER).strvalue

        # Validate camera drivers
        valid_camera_drivers = tp.get_args(CameraDriver)  # Get valid values from the Literal type
        if main_camera_driver not in valid_camera_drivers:
            error_msg = f"invalid main camera driver '{main_camera_driver}'. Valid drivers: {valid_camera_drivers}"
            logger.critical(f"startup - {error_msg}")
            cmd.error_internal(detail=error_msg)

        if focus_camera_driver not in valid_camera_drivers:
            error_msg = f"invalid focus camera driver '{focus_camera_driver}'. Valid drivers: {valid_camera_drivers}"
            logger.critical(f"startup - {error_msg}")
            cmd.error_internal(detail=error_msg)

        # Validate microcontroller driver
        valid_mc_drivers = tp.get_args(MicrocontrollerDriver)
        if microcontroller_driver not in valid_mc_drivers:
            error_msg = f"invalid microcontroller driver '{microcontroller_driver}'. Valid drivers: {valid_mc_drivers}"
            logger.critical(f"startup - {error_msg}")
            cmd.error_internal(detail=error_msg)

        # Find devices by USB ID
        main_camera_request = CameraOpenRequest(driver=main_camera_driver, usb_id=main_camera_id)  # type: ignore
        focus_camera_request = CameraOpenRequest(driver=focus_camera_driver, usb_id=focus_camera_id)  # type: ignore
        mc_request = MicrocontrollerOpenRequest(driver=microcontroller_driver, usb_id=microcontroller_id)  # type: ignore

        main_camera = camera_open(main_camera_request)
        focus_camera = camera_open(focus_camera_request)
        microcontroller = microcontroller_open(mc_request)

        # Extract channel and filter configurations (native objects from config)
        channels_data = ConfigRegistry.get(ImagingConfig.CHANNELS).objectvalue
        channel_configs = [ChannelConfig(**ch) for ch in channels_data] #type: ignore

        # Only process filters if filter wheel is available
        filter_wheel_available = ConfigRegistry.get(FilterWheelConfig.AVAILABLE)
        if filter_wheel_available and filter_wheel_available.boolvalue:
            filters_data = ConfigRegistry.get(FilterWheelConfig.CONFIGURATION).objectvalue
            filter_configs = [FilterConfig(**f) for f in filters_data] #type: ignore
        else:
            filter_configs = []

        # Initialize illumination controller with channel configurations
        illumination_controller = IlluminationController(channel_configs)

        squid = cls(
            main_camera=Locked(main_camera),
            focus_camera=Locked(focus_camera),
            microcontroller=microcontroller,
            illumination_controller=illumination_controller,
            channels=channel_configs,
            filters=filter_configs,
        )

        # do NOT connect yet

        return squid

    def open_connections(self):
        """open connections to devices"""
        if self.is_connected:
            return

        with (
            self.main_camera.locked() as main_camera,
            self.focus_camera.locked() as focus_camera,
        ):
            if main_camera is None:
                error_internal("main camera is busy")
            if focus_camera is None:
                error_internal("focus camera is busy")

            # Close before opening, always. A short disconnect does not reach the camera, so it
            # still holds the old connection and refuses a new one until that is severed - and
            # after a partial open or a drop, some devices are still open from last time.
            try:
                self.close()
            except HardwareCloseError as e:
                # Expected after a drop: the devices being closed are the ones that went away.
                # They are released regardless, which is all opening needs, and each failure is
                # on the health board as that device's last fault.
                logger.info(f"startup - {e}")

            opening: list[tuple[Device, tp.Callable[[], None]]] = [
                (Device.MAIN_CAMERA, lambda: main_camera.open(device_type="main")),
                (Device.FOCUS_CAMERA, lambda: focus_camera.open(device_type="autofocus")),
                (Device.MICROCONTROLLER, self.microcontroller.open),
            ]
            for device, open_device in opening:
                try:
                    with self._watchdog.guard(device, "open", OPEN_DEADLINE_S):
                        open_device()
                except DeviceAlreadyInUseError as e:
                    # its own error, so the client can say "close the other seafront"
                    self._health.open_failed(device, e)
                    raise
                except Exception as e:
                    # Whatever opening raised, the device is not available, and that is what
                    # the caller needs to know - with which one it was and why, kept as cause.
                    # Devices opened before this one stay open; the next attempt closes them
                    # first, as does shutdown.
                    self._health.open_failed(device, e)
                    logger.critical(f"startup - could not open {device}: {describe(e)}")
                    raise DisconnectError(device) from e
                self._health.connected(device)
                logger.debug(f"startup - connected to {device}")

            # Offer only the pixel formats the cameras actually have. Not optional: without
            # this the config lists formats a camera may not support, and choosing one fails
            # much later, far from the cause.
            from seafront.config.basics import GlobalConfigHandler

            with self._watchdog.guard(Device.MAIN_CAMERA, "pixel format query", CALL_DEADLINE_S):
                main_formats = main_camera.get_supported_pixel_formats()
            with self._watchdog.guard(Device.FOCUS_CAMERA, "pixel format query", CALL_DEADLINE_S):
                focus_formats = focus_camera.get_supported_pixel_formats()
            logger.debug(f"startup - main camera supports formats: {main_formats}")
            logger.debug(f"startup - focus camera supports formats: {focus_formats}")
            GlobalConfigHandler.update_pixel_format_options(main_formats, focus_formats)
            logger.info("startup - updated config with actual camera pixel format capabilities")

            logger.info("startup - connection to hardware devices established")
            self.is_connected = True

    @classmethod
    def hardware_devices(cls) -> list[Device]:
        return [Device.MAIN_CAMERA, Device.FOCUS_CAMERA, Device.MICROCONTROLLER]

    def _camera_device(self, camera: Camera) -> Device:
        if camera is self.main_camera.value:
            return Device.MAIN_CAMERA
        if camera is self.focus_camera.value:
            return Device.FOCUS_CAMERA
        raise ValueError(f"camera {camera.sn} is not one of this microscope's")

    def _drop_connection(self, device: Device, operation: str, error: BaseException) -> DisconnectError:
        """
        `device` failed in a way that means its connection is gone. Close everything - a
        microscope missing one device cannot run anything, and the next command reconnects all
        of them - then record which device it was, and hand back the error to raise.

        Recorded after the close, which marks every device closed: the one that was lost must
        end up marked as such.
        """
        logger.critical(f"squid - lost {device} during {operation}: {describe(error)}")
        disconnect = DisconnectError(device)
        disconnect.add_note(f"{operation}: {describe(error)}")
        try:
            self.close()
        except HardwareCloseError as close_error:
            disconnect.add_note(str(close_error))
        self._health.lost(device, operation, error)
        return disconnect

    def close(self):
        """
        close connection to microcontroller and cameras (see Microscope.close for the contract)

        Does not look at is_connected: after a partial open, or a drop, some devices are open
        while the adapter as a whole is not, and those must be closed too.
        """
        # Outside the camera lock, and first: a disconnected adapter must not be left
        # claiming to stream, and clearing the state stops the SDK's capture callback from
        # forwarding into a camera that is being torn down.
        self._stream_stopped()

        with (
            self.main_camera.locked() as main_camera,
            self.focus_camera.locked() as focus_camera,
        ):
            if main_camera is None:
                error_internal("main camera is busy")
            if focus_camera is None:
                error_internal("focus camera is busy")

            self.is_connected = False

            closing: list[tuple[Device, tp.Callable[[], None]]] = [
                (Device.MICROCONTROLLER, self.microcontroller.close),
                (Device.MAIN_CAMERA, main_camera.close),
                (Device.FOCUS_CAMERA, focus_camera.close),
            ]
            failures: dict[Device, BaseException] = {}
            for device, close_device in closing:
                logger.debug(f"closing {device}")
                try:
                    with self._watchdog.guard(device, "close", CALL_DEADLINE_S):
                        close_device()
                except Exception as e:
                    failures[device] = e
                self._health.closed(device, failures.get(device))

            if failures:
                raise HardwareCloseError(failures)

            logger.info("closed connection to hardware devices")

    async def home(self):
        """perform homing maneuver (single owner: no lock needed)"""
        self._home_impl(self.microcontroller)

    def _home_impl(self, qmc: Microcontroller) -> None:
        """homing maneuver body; caller must already hold the microcontroller lock"""
        try:
            logger.info("starting stage calibration (by entering loading position)")

            # reset the MCU
            logger.debug("resetting mcu")
            qmc.reset()
            logger.debug("done")

            # reinitialize motor drivers and DAC
            logger.debug("initializing microcontroller")
            qmc.initialize()
            logger.debug("done initializing microcontroller")

            if True:
                # disable for testing (new firmware should have better defaults)
                logger.debug("configure_actuators")
                qmc.configure_actuators()
                logger.debug("done configuring actuators")

            logger.info("ensuring illumination is off")
            # make sure all illumination is off
            for illum_src in [
                # turn off all fluorescence LEDs
                ILLUMINATION_CODE.ILLUMINATION_SOURCE_FLUOSLOT11,
                ILLUMINATION_CODE.ILLUMINATION_SOURCE_FLUOSLOT12,
                ILLUMINATION_CODE.ILLUMINATION_SOURCE_FLUOSLOT13,
                ILLUMINATION_CODE.ILLUMINATION_SOURCE_FLUOSLOT14,
                ILLUMINATION_CODE.ILLUMINATION_SOURCE_FLUOSLOT15,
                # this will turn off the led matrix
                ILLUMINATION_CODE.ILLUMINATION_SOURCE_LED_ARRAY_FULL,
            ]:
                qmc.illumination_end(illum_src.value)

            logger.debug("calibrating xy stage")

            # when starting up the microscope, the initial position is considered (0,0,0)
            # even homing considers the limits, so before homing, we need to disable the limits
            qmc.set_limit_mm("z", -10.0, "lower")
            qmc.set_limit_mm("z", 10.0, "upper")

            # move objective out of the way
            qmc.home("z")
            qmc.set_zero("z")
            # set z limit to (or below) 6.7mm, because above that, the motor can get stuck
            qmc.set_limit_mm("z", 0.0, "lower")
            qmc.set_limit_mm("z", 6.7, "upper")
            # home x to set x reference
            qmc.home("x")
            qmc.set_zero("x")
            # clear clamp in x
            qmc.move_by_mm("x", 30)
            # then move in position to properly apply clamp
            qmc.home("y")
            qmc.set_zero("y")
            # home x again to engage clamp
            qmc.home("x")

            # move to an arbitrary position to disengage the clamp
            qmc.move_by_mm("x", 30)
            qmc.move_by_mm("y", 30)

            # and move objective up, slightly
            qmc.move_by_mm("z", 1)

            # Only initialize filter wheel if it's available
            filter_wheel_available = ConfigRegistry.get(FilterWheelConfig.AVAILABLE)
            if filter_wheel_available and filter_wheel_available.boolvalue:
                # Initialize filter wheel with homing sequence (matching Squid behavior)
                logger.info("initializing filter wheel...")
                try:
                    qmc.filter_wheel_init()
                    logger.info("configuring filter wheel actuator...")
                    qmc.filter_wheel_configure_actuator()
                    logger.info("performing filter wheel homing...")
                    qmc.filter_wheel_home()
                    logger.info("✓ Filter wheel initialized, configured, and homed")
                except Exception as e:
                    logger.warning(f"Filter wheel initialization/configuration/homing failed: {e}")
                    logger.info("Continuing with microscope initialization...")

            logger.info("done initializing microscope")

        except IOError as e:
            raise self._drop_connection(Device.MICROCONTROLLER, "home_impl", e) from e

    @property
    def calibrated_stage_position(self) -> tuple[float, float, float]:
        """
        return calibrated XY stage offset from GlobalConfigHandler in order (x_mm,y_mm)
        """

        g_config = GlobalConfigHandler.get_dict()

        def _get_offset(handle: CalibrationConfig, default: float = 0.0) -> float:
            item = g_config.get(handle.value)
            if item is None:
                logger.debug(
                    "calibration offset '%s' missing in config; falling back to %s",
                    handle.value,
                    default,
                )
                return default
            return item.floatvalue

        off_x_mm = _get_offset(CalibrationConfig.OFFSET_X_MM)
        off_y_mm = _get_offset(CalibrationConfig.OFFSET_Y_MM)
        off_z_mm = _get_offset(CalibrationConfig.OFFSET_Z_MM)

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

    def _measure_laser_autofocus(
        self,
        img: np.ndarray,
        use_right_dot: bool,
        calib_params: cmd.LaserAutofocusCalibrationData | None = None,
    ) -> LaserAutofocusMeasurement:
        """
        Measure the laser autofocus signal from an image.

        Args:
            img: the autofocus camera image
            use_right_dot: whether to track the right dot (True) or left dot (False)
            calib_params: calibration parameters for displacement estimation, or None to skip estimation

        Returns:
            measurement result including tracked dot position and optionally estimated displacement
        """
        # Apply blur to reduce noise before peak detection
        BLUR_SIZE=9 # 3
        img: np.ndarray = cv2.GaussianBlur(img, (BLUR_SIZE, BLUR_SIZE), 0)

        # 8 bit signal -> max value 255
        I_1d: np.ndarray = img.max(axis=0)

        # locate peaks == locate dots
        peak_locations, _ = scipy.signal.find_peaks(
            I_1d, 
            distance=200, 
            height=20,
            # from cephla:
            prominence=0.25,
            width=10, 
        )

        if len(peak_locations.tolist()) == 0:
            error_internal("no signal found")

        peak_data: list[PeakData]=[
            PeakData(
                loc_x=loc_x,
                height=float(I_1d[loc_x])
            )
            for loc_x
            in peak_locations.tolist()
        ]

        # order by height
        tallestpeaks: list[PeakData] = sorted(peak_data, key=lambda d: d.height)
        # pick two tallest peaks
        tallestpeaks = tallestpeaks[-2:]
        # then order by peaks x coordinate
        tallestpeaks = sorted(tallestpeaks, key=lambda d: d.loc_x)

        # Find the rightmost (largest x) peak
        rightmost_peak = max(tallestpeaks, key=lambda d: d.loc_x)

        # Compute distances between consecutive peaks
        interpeakdistances: list[float] = [
            tallestpeaks[i + 1].loc_x - tallestpeaks[i].loc_x
            for i in range(len(tallestpeaks) - 1)
        ]

        # Determine tracked dot x based on use_right_dot setting
        if len(interpeakdistances) == 0:
            # Only one peak detected, use_right_dot does not matter
            tracked_dot_x = rightmost_peak.loc_x
        elif len(interpeakdistances) == 1:
            if use_right_dot:
                tracked_dot_x = rightmost_peak.loc_x
            else:
                tracked_dot_x = rightmost_peak.loc_x - interpeakdistances[0]
        else:
            raise ValueError(f"expected 0 or 1 inter-peak distances, got {len(interpeakdistances)}")

        # Calculate displacement if calibration data provided
        estimated_displacement_um: float | None = None
        if calib_params is not None:
            estimated_displacement_um = (tracked_dot_x - calib_params.x_reference) * calib_params.um_per_px

        return LaserAutofocusMeasurement(
            rightmost_x=rightmost_peak.loc_x,
            interpeakdistances=interpeakdistances,
            tracked_dot_x=tracked_dot_x,
            estimated_displacement_um=estimated_displacement_um,
        )

    def _approximate_laser_af_displacement_um(
        self,
        qmc: Microcontroller,
        calib_params: cmd.LaserAutofocusCalibrationData | None = None,
        _dotxinsteadofestimatedz: bool = False,
    ) -> float:
        """
        Approximate current displacement from reference position in micrometers.

        Args:
            qmc: the already-locked microcontroller (caller holds the lock)
            calib_params: calibration parameters, or None if _dotxinsteadofestimatedz is True
            _dotxinsteadofestimatedz: if True, return the tracked dot x coordinate instead of estimated displacement

        Returns:
            estimated displacement in micrometers, or tracked dot x if _dotxinsteadofestimatedz is True
        """
        conf_af_exp_ms = ConfigRegistry.get(LaserAutofocusConfig.EXPOSURE_TIME_MS).floatvalue
        conf_af_exp_ag = ConfigRegistry.get(LaserAutofocusConfig.CAMERA_ANALOG_GAIN).floatvalue
        use_right_dot = ConfigRegistry.get(LaserAutofocusConfig.USE_RIGHT_DOT).boolvalue

        res = self._dispatch(
            qmc,
            cmd.AutofocusSnap(exposure_time_ms=conf_af_exp_ms, analog_gain=conf_af_exp_ag),
        )

        measurement = self._measure_laser_autofocus(res._img, use_right_dot, calib_params)

        if _dotxinsteadofestimatedz:
            return measurement.tracked_dot_x

        assert measurement.estimated_displacement_um is not None
        return measurement.estimated_displacement_um

    def _execute_LaserAutofocusCalibrate(
        self,
        qmc: Microcontroller,
        command:cmd.LaserAutofocusCalibrate,
    ) -> cmd.LaserAutofocusCalibrationResponse:
    
        # Hardcoded calibration parameters
        Z_MM_BACKLASH_COUNTER = 40e-3
        DEBUG_LASER_AF_CALIBRATION = os.getenv("DEBUG_LASER_AF_CALIBRATION", "").lower() == "true"
        DEBUG_LASER_AF_SHOW_REGRESSION_FIT = os.getenv("DEBUG_LASER_AF_SHOW_REGRESSION_FIT", "").lower() == "true"
        DEBUG_LASER_AF_SHOW_EVAL_FIT = os.getenv("DEBUG_LASER_AF_SHOW_EVAL_FIT", "true").lower() == "true"

        # Read number of z steps from machine config
        conf_num_steps_item = ConfigRegistry.get(LaserAutofocusConfig.CALIBRATION_NUM_Z_STEPS)
        assert conf_num_steps_item is not None
        num_z_steps_calibrate = conf_num_steps_item.intvalue

        if self.is_in_loading_position:
            cmd.error_internal(detail="now allowed while in loading position")

        # Read z_span from machine config
        conf_z_span_item = ConfigRegistry.get(LaserAutofocusConfig.CALIBRATION_Z_SPAN_MM)
        assert conf_z_span_item is not None
        z_mm_movement_range = conf_z_span_item.floatvalue
        logger.debug(f"laser autofocus calibration: z_span={z_mm_movement_range}mm, num_steps={num_z_steps_calibrate}")

        conf_af_exp_ms_item = ConfigRegistry.get(LaserAutofocusConfig.EXPOSURE_TIME_MS)
        assert conf_af_exp_ms_item is not None
        conf_af_exp_ms = conf_af_exp_ms_item.floatvalue

        conf_af_exp_ag_item = ConfigRegistry.get(LaserAutofocusConfig.CAMERA_ANALOG_GAIN)
        assert conf_af_exp_ag_item is not None
        conf_af_exp_ag = conf_af_exp_ag_item.floatvalue

        use_right_dot_item = ConfigRegistry.get(LaserAutofocusConfig.USE_RIGHT_DOT)
        assert use_right_dot_item is not None
        use_right_dot = use_right_dot_item.boolvalue

        z_step_mm = z_mm_movement_range / (num_z_steps_calibrate - 1)
        half_z_mm = z_mm_movement_range / 2

        try:
            current_pos = qmc.get_last_position()
            _start_z_mm: float = current_pos.z_pos_mm

            # move down by half z range
            if Z_MM_BACKLASH_COUNTER != 0:  # is not None:
                qmc.move_by_mm("z", -(half_z_mm + Z_MM_BACKLASH_COUNTER))
                qmc.move_by_mm("z", Z_MM_BACKLASH_COUNTER)
            else:
                qmc.move_by_mm("z", -half_z_mm)
        except IOError as e:
            raise self._drop_connection(Device.MICROCONTROLLER, "LaserAutofocusCalibrate", e) from e

        class CalibrationData(BaseModel):
            z_mm: float
            measurement: LaserAutofocusMeasurement

        def measure() -> LaserAutofocusMeasurement:
            res = self._dispatch(
                qmc,
                cmd.AutofocusSnap(exposure_time_ms=conf_af_exp_ms, analog_gain=conf_af_exp_ag),
            )
            return self._measure_laser_autofocus(res._img, use_right_dot, calib_params=None)

        peak_info: list[CalibrationData] = []

        for i in range(num_z_steps_calibrate):
            if i > 0:
                # move up by half z range to get position at original position, but moved to from fixed direction to counter backlash
                try:
                    qmc.move_by_mm("z", z_step_mm)
                except IOError as e:
                    raise self._drop_connection(Device.MICROCONTROLLER, "LaserAutofocusCalibrate", e) from e

            measurement = measure()

            peak_info.append(CalibrationData(z_mm=-half_z_mm + i * z_step_mm, measurement=measurement))

        # move to original position
        try:
            qmc.move_by_mm("z", -half_z_mm)
        except IOError as e:
            raise self._drop_connection(Device.MICROCONTROLLER, "LaserAutofocusCalibrate", e) from e

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
            m = i.measurement

            new_distances = [m.rightmost_x]
            for p_d in m.interpeakdistances:
                new_distances.append(new_distances[-1] - p_d)

            new_z = [i.z_mm] * len(new_distances)

            distances.extend(new_distances)
            distance_x.extend(new_z)

            if len(m.interpeakdistances) == 0:
                target_domain = one_peak_domain
            elif len(m.interpeakdistances) == 1:
                target_domain = two_peak_domain
            else:
                raise ValueError(f"expected 0 or 1 inter-peak distances, got {len(m.interpeakdistances)}")

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
            current_pos = qmc.get_last_position()
            start_z = current_pos.z_pos_mm

            half_z_mm = z_mm_movement_range / 2
            num_z_steps_eval = 51
            z_step_mm = z_mm_movement_range / (num_z_steps_eval - 1)

            if Z_MM_BACKLASH_COUNTER != 0:  # is not None:
                qmc.move_by_mm("z", -(half_z_mm + Z_MM_BACKLASH_COUNTER))
                qmc.move_by_mm("z", Z_MM_BACKLASH_COUNTER)
            else:
                qmc.move_by_mm("z", -half_z_mm)

            approximated_z: list[tuple[float, float]] = []
            for i in range(num_z_steps_eval):
                if i > 0:
                    # move up by half z range to get position at original position, but moved to from fixed direction to counter backlash
                    qmc.move_by_mm("z", z_step_mm)

                approx_um = self._approximate_laser_af_displacement_um(
                    qmc,
                    cmd.LaserAutofocusCalibrationData(
                        um_per_px=1000 / left_dot_regression[0],  # px/mm -> um/px
                        x_reference=left_dot_regression[1],
                        calibration_position=cmd.Position.zero(),
                    )
                )
                current_pos = qmc.get_last_position()
                current_z_mm = current_pos.z_pos_mm
                approximated_z.append((current_z_mm - start_z, approx_um))

            # move to original position
            qmc.move_by_mm("z", -half_z_mm)

            plt.figure(figsize=(8, 6))
            # v[0] is real z offset in mm, v[1] is measured displacement in um
            plt.scatter(
                [v[0] * 1e3 for v in approximated_z],  # mm -> um
                [v[0] * 1e3 for v in approximated_z],  # mm -> um
                color="green",
                label="real/expected",
                marker="o",
                linestyle="-",
            )
            plt.scatter(
                [v[0] * 1e3 for v in approximated_z],  # mm -> um
                [v[1] for v in approximated_z],  # already um
                color="blue",
                label="estimated",
                marker="o",
                linestyle="-",
            )
            plt.scatter(
                [v[0] * 1e3 for v in approximated_z],  # mm -> um
                [v[1] - v[0] * 1e3 for v in approximated_z],  # error in um
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

        # Select regression and reference dot based on use_right_dot config
        # Regression fits dot_x = slope * z_mm + intercept, so slope is px/mm
        # To get um_per_px: um_per_px = 1000 / slope_px_per_mm
        if use_right_dot:
            if right_dot_regression == (0, 0):
                cmd.error_internal(detail="use_right_dot is enabled but no right dot was detected during calibration")
            px_per_mm, _x_reference = right_dot_regression
        else:
            px_per_mm, _x_reference = left_dot_regression

        # A slope of zero means the spot did not move as z changed - the laser was off, or
        # nothing was reflecting - so there is no scale to compute and dividing produces a
        # ZeroDivisionError halfway through a calibration. Said plainly instead: the number
        # this would have produced is the one every later displacement is multiplied by.
        if px_per_mm == 0:
            cmd.error_internal(
                detail="laser autofocus calibration failed: the tracked dot did not move as z "
                "changed (regression slope is zero), so there is no um/px scale to derive. "
                "check that the autofocus laser is on and that a sample is in place."
            )

        um_per_px = 1000 / px_per_mm

        # Measure reference position (x_reference is the tracked dot x at calibration position)
        x_reference = (measure()).tracked_dot_x

        try:
            calibration_position = qmc.get_last_position()
        except IOError as e:
            raise self._drop_connection(Device.MICROCONTROLLER, "LaserAutofocusCalibrate", e) from e

        # Calculate offset correction to zero out error at calibration position
        # Measure displacement multiple times and average to reduce noise
        temp_calib = cmd.LaserAutofocusCalibrationData(
            um_per_px=um_per_px,
            x_reference=x_reference,
            calibration_position=cmd.Position.zero(),
        )
        num_offset_measurements = 3
        measured_displacements_um: list[float] = []
        for _ in range(num_offset_measurements):
            res = self._dispatch(qmc, cmd.AutofocusSnap(exposure_time_ms=conf_af_exp_ms, analog_gain=conf_af_exp_ag))
            measurement = self._measure_laser_autofocus(res._img, use_right_dot, temp_calib)
            assert measurement.estimated_displacement_um is not None
            measured_displacements_um.append(measurement.estimated_displacement_um)

        avg_displacement_um = sum(measured_displacements_um) / len(measured_displacements_um)
        # Offset correction: negate the measured displacement so error becomes zero
        offset_um = -avg_displacement_um

        logger.debug(f"laser autofocus offset correction: measured {avg_displacement_um:.3f} um at calibration position, setting offset to {offset_um:.3f} um")

        calibration_data = cmd.LaserAutofocusCalibrationData(
            # calculate the conversion factor, based on lowest and highest measured position
            um_per_px=um_per_px,
            # set reference position
            x_reference=x_reference,
            # offset to zero out error at calibration position
            offset_um=offset_um,
            calibration_position=calibration_position,
        )

        logger.debug(f"laser autofocus calibration result: um_per_px={um_per_px:.4f}, x_reference={x_reference:.1f}, offset_um={offset_um:.3f}")
        self.record_laser_autofocus_calibration(calibration_data)
        return cmd.LaserAutofocusCalibrationResponse(calibration_data=calibration_data)

    def health(self) -> list[DeviceHealth]:
        # the microcontroller keeps its own last-heard time as it reads; bring it over first,
        # so whoever asks sees when the device last spoke rather than when someone last looked
        heard_at = self.microcontroller.last_heard_at()
        if heard_at is not None:
            self._health.heard(Device.MICROCONTROLLER, at=heard_at)
        return super().health()

    async def get_current_state(self) -> AdapterState:
        try:
            current_position = self._get_current_position(self.microcontroller)
        except IOError as e:
            # The status read runs beside the worker, not on it, so dropping the connection
            # from here would pull the hardware out from under whatever command is running.
            # It records what it saw instead - the health board is what the operator reads -
            # and hands back the last position the device did report, which the board marks
            # as not current. The next command to talk to the device meets the same silence
            # and drops the connection from where that is safe to do.
            self._health.unresponsive(Device.MICROCONTROLLER, "status read", describe(e))
            current_position = self._position_to_real(self.microcontroller.last_position_cached())

        self.last_state = AdapterState(
            is_in_loading_position=self.is_in_loading_position,
            stage_position=current_position,
        )

        return self.last_state

    def illumination_state(self) -> cmd.IlluminationState:
        # the code sent is the slot number (ILLUMINATION_CODE.from_slot), without the check a
        # status read has no use for
        by_code = {channel.source_slot: channel for channel in self.channels}
        lit = []
        for code, intensity in sorted(self.microcontroller.illumination_lit().items()):
            channel = by_code.get(code)
            lit.append(
                cmd.LitIllumination(
                    handle=channel.handle if channel is not None else "",
                    name=channel.name if channel is not None
                    else f"illumination source {code} (no channel is configured for it)",
                    intensity_percent=intensity,
                )
            )
        return cmd.IlluminationState(lit=lit, af_laser_on=self.microcontroller.af_laser_is_on())

    def _get_current_position(self, qmc: Microcontroller) -> cmd.Position:
        """Stage position in the calibrated coordinate frame used by public commands."""
        return self._position_to_real(qmc.get_last_position())

    def _position_to_real(self, measured: Position) -> cmd.Position:
        """a measured stage position, in the calibrated frame"""
        return cmd.Position(
            x_pos_mm=self._pos_x_measured_to_real(measured.x_pos_mm),
            y_pos_mm=self._pos_y_measured_to_real(measured.y_pos_mm),
            z_pos_mm=self._pos_z_measured_to_real(measured.z_pos_mm),
        )

    def get_hardware_limits(self) -> HardwareLimits:
        """
        Get hardware-specific limits by querying camera and combining with SQUID mechanical limits.
        """
        # Get camera-specific limits (exposure time and analog gain)
        with self.main_camera.locked() as main_camera:
            if main_camera is None:
                raise RuntimeError("Main camera not available for hardware limits query")

            with self._watchdog.guard(Device.MAIN_CAMERA, "limits query", CALL_DEADLINE_S):
                exposure_limits = main_camera.get_exposure_time_limits()
                gain_limits = main_camera.get_analog_gain_limits()

        # Create SQUID-specific mechanical and power limits as HardwareLimitValue objects
        focus_offset_limits = HardwareLimitValue(min=-200, max=200, step=0.1)
        fluorescence_power_limits = HardwareLimitValue(min=5, max=100, step=0.1)
        brightfield_power_limits = HardwareLimitValue(min=3, max=100, step=0.1)
        generic_power_limits = HardwareLimitValue(min=5, max=100, step=0.1)  # Use fluorescence as default
        z_planes_limits = HardwareLimitValue(min=1, max=999, step=1)
        z_spacing_limits = HardwareLimitValue(min=0.1, max=1000, step=0.1)

        # Return properly typed HardwareLimits object
        return HardwareLimits(
            imaging_exposure_time_ms=exposure_limits,
            imaging_analog_gain_db=gain_limits,
            imaging_focus_offset_um=focus_offset_limits,
            imaging_illum_perc=generic_power_limits,
            imaging_illum_perc_fluorescence=fluorescence_power_limits,
            imaging_illum_perc_brightfield=brightfield_power_limits,
            imaging_number_z_planes=z_planes_limits,
            imaging_delta_z_um=z_spacing_limits,
        )

    def is_position_forbidden(self, x_mm: float, y_mm: float) -> tuple[bool, str]:
        """
        Check if a position is forbidden for movement.
        
        returns (<position is forbidden?>, <forbidden reason>)
        """
        return cmd.positionIsForbidden(x_mm, y_mm)

    def _execute_EstablishHardwareConnection(
        self,
        qmc: Microcontroller,
        command:cmd.EstablishHardwareConnection,
    ):
        if not self.is_connected:
            try:
                # if no connection has yet been established, connecting will have the hardware in an undefined state
                logger.debug("squid - connect - connecting to hardware")
                self.open_connections()
                # so after connecting:
                # 1) turn off all illumination
                logger.debug("squid - connect - turning off illumination")
                self._dispatch(qmc, IlluminationEndAll())
                logger.debug("squid - connect - turned off illumination")
                # 2) perform home maneuver to reset stage position to known values
                logger.debug("squid - connect - calibrating stage position")

                self._home_impl(qmc)

                logger.info("squid - connect - calibrated stage")

            except DeviceAlreadyInUseError as e:
                logger.critical(f"squid - connect - {e}")
                logger.critical("Please close any other instances of Seafront before starting a new one.")
                error_device_in_use(e.device_type, e.device_id)
            except DisconnectError as e:
                # say which device and why: "could not be established" alone sends the operator
                # to check every cable
                reason = describe(e.__cause__) if e.__cause__ is not None else "no further detail"
                error_message = f"hardware connection could not be established - {e}: {reason}"
                logger.critical(f"squid - connect - {error_message}")
                error_internal(error_message)

        logger.debug("squid - connected.")

        result = BasicSuccessResponse()
        return result  # type: ignore[no-any-return]

    def _execute_LoadingPositionEnter(
        self,
        qmc: Microcontroller,
        command:cmd.LoadingPositionEnter
    ):
        if self.is_in_loading_position:
            cmd.error_internal(detail="already in loading position")

        # home z
        qmc.home("z")

        # clear clamp in y first
        qmc.move_to_mm("y", 30)
        # then clear clamp in x
        qmc.move_to_mm("x", 30)

        # then home y, x
        qmc.home("y")
        qmc.home("x")

        self.is_in_loading_position = True

        logger.debug("squid - entered loading position")

        result = cmd.BasicSuccessResponse()
        return result  # type: ignore[no-any-return]

    def _execute_LoadingPositionLeave(
        self,
        qmc: Microcontroller,
        command:cmd.LoadingPositionLeave
    ):
        if not self.is_in_loading_position:
            cmd.error_internal(detail="not in loading position")

        qmc.move_to_mm("x", 30)
        qmc.move_to_mm("y", 30)
        qmc.move_to_mm("z", 1)

        self.is_in_loading_position = False

        logger.debug("squid - left loading position")

        result = cmd.BasicSuccessResponse()
        return result

    def _execute_MoveBy(
        self,
        qmc: Microcontroller,
        command:cmd.MoveBy
    ):
        self.raise_if_cancelled()
        if self.is_in_loading_position:
            cmd.error_internal(detail="now allowed while in loading position")

        # Calculate target position after relative movement
        try:
            current_stage_position = qmc.get_last_position()
        except IOError as e:
            raise self._drop_connection(Device.MICROCONTROLLER, "MoveBy", e) from e

        current_x = self._pos_x_measured_to_real(current_stage_position.x_pos_mm)
        current_y = self._pos_y_measured_to_real(current_stage_position.y_pos_mm)

        if command.axis == "x":
            target_x = current_x + command.distance_mm
            target_y = current_y
        elif command.axis == "y":
            target_x = current_x
            target_y = current_y + command.distance_mm
        else:  # z or other axis
            target_x = current_x
            target_y = current_y

        # Check forbidden areas for X,Y movement
        is_forbidden, error_message = self.is_position_forbidden(
            target_x, target_y
        )
        if is_forbidden:
            cmd.error_internal(detail=error_message)

        qmc.move_by_mm(command.axis, command.distance_mm)

        logger.debug("squid - moved by")

        result = cmd.MoveByResult(moved_by_mm=command.distance_mm, axis=command.axis)
        return result

    def _execute_MoveTo(
        self,
        qmc: Microcontroller,
        command:cmd.MoveTo
    ):
        self.raise_if_cancelled()
        if self.is_in_loading_position:
            cmd.error_internal(detail="now allowed while in loading position")

        if command.x_mm is not None and command.x_mm < 0:
            cmd.error_internal(detail=f"x coordinate out of bounds {command.x_mm = }")
        if command.y_mm is not None and command.y_mm < 0:
            cmd.error_internal(detail=f"y coordinate out of bounds {command.y_mm = }")
        if command.z_mm is not None and command.z_mm < 0:
            cmd.error_internal(detail=f"z coordinate out of bounds {command.z_mm = }")

        # Check forbidden areas if we have complete X,Y coordinates
        if command.x_mm is not None and command.y_mm is not None:
            is_forbidden, error_message = self.is_position_forbidden(
                command.x_mm, command.y_mm
            )
            if is_forbidden:
                cmd.error_internal(detail=error_message)

        approach_x_before_y = True

        if command.x_mm is not None and command.y_mm is not None:
            current_stage_position = qmc.get_last_position()

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
                qmc.move_to_mm("x", x_mm)

            if command.y_mm is not None:
                y_mm = self._pos_y_real_to_measured(command.y_mm)
                if y_mm < 0:
                    cmd.error_internal(
                        detail=f"calibrated y coordinate out of bounds {y_mm = }"
                    )
                qmc.move_to_mm("y", y_mm)
        else:
            if command.y_mm is not None:
                y_mm = self._pos_y_real_to_measured(command.y_mm)
                if y_mm < 0:
                    cmd.error_internal(
                        detail=f"calibrated y coordinate out of bounds {y_mm = }"
                    )
                qmc.move_to_mm("y", y_mm)

            if command.x_mm is not None:
                x_mm = self._pos_x_real_to_measured(command.x_mm)
                if x_mm < 0:
                    cmd.error_internal(
                        detail=f"calibrated x coordinate out of bounds {x_mm = }"
                    )
                qmc.move_to_mm("x", x_mm)

        if command.z_mm is not None:
            z_mm = self._pos_z_real_to_measured(command.z_mm)
            if z_mm < 0:
                cmd.error_internal(
                    detail=f"calibrated z coordinate out of bounds {z_mm = }"
                )
            qmc.move_to_mm("z", z_mm)

        logger.debug("squid - moved to")

        result = cmd.BasicSuccessResponse()
        return result

    def _execute_MoveToWell(
        self,
        qmc: Microcontroller,
        command:cmd.MoveToWell
    ):
        if self.is_in_loading_position:
            cmd.error_internal(detail="now allowed while in loading position")

        plate = command.plate_type

        x_mm = plate.get_well_offset_x(command.well_name) + plate.Well_size_x_mm / 2
        y_mm = plate.get_well_offset_y(command.well_name) + plate.Well_size_y_mm / 2

        # Check forbidden areas for the target well center position
        is_forbidden, error_message = self.is_position_forbidden(
            x_mm, y_mm
        )
        if is_forbidden:
            cmd.error_internal(detail=f"Well {command.well_name} center position is in forbidden area - {error_message}")

        # move in 2d grid, no diagonals (slower, but safer)
        res = self._dispatch(qmc, cmd.MoveTo(x_mm=x_mm, y_mm=None))
        res = self._dispatch(qmc, cmd.MoveTo(x_mm=None, y_mm=y_mm))
        _=res

        logger.debug("squid - moved to well")

        result = cmd.BasicSuccessResponse()
        return result

    def _execute_ChannelStreamEnd(
        self,
        command:cmd.ChannelStreamEnd
    ):
        # Clearing the state is what stops forwarding: any capture callback still in flight
        # on the camera SDK's acquisition thread sees no stream and returns immediately
        # without touching the sink. It also hands back what was streaming, which is how this
        # knows whose illumination to turn off - the caller never says.
        stopped = self._stream_stopped()
        if stopped is None:
            cmd.error_internal(detail="not currently streaming")

        logger.debug("squid - requested stream stop")

        # Stop the camera under the camera lock. This is deadlock-free now that the capture
        # callback no longer acquires the camera lock (see _execute_ChannelStreamBegin), so
        # stop_streaming()'s stream_off()/unregister_capture_callback() cannot block on it.
        # The lock still provides mutual exclusion against snap/start_streaming; blocking=True
        # so we reliably stop instead of silently leaving the camera streaming.
        #
        # Both halves are attempted whatever happens to the other: a camera that fails to stop
        # is no reason to leave the light on. Either failing fails the command.
        failures: list[tuple[str, BaseException]] = []
        try:
            with self.main_camera.locked() as cam:
                assert cam is not None, "blocking lock acquire returned no camera"

                logger.debug("squid - stopping camera streaming")
                with self._watchdog.guard(
                    Device.MAIN_CAMERA,
                    "stop streaming",
                    CALL_DEADLINE_S,
                    on_stall=self._illumination_off_during_stall,
                ):
                    cam.stop_streaming()
                logger.debug("squid - camera streaming stopped")
        except Exception as e:
            failures.append(("stop camera streaming", e))

        # Turn off illumination for the channel that was actually streaming. If that channel
        # is no longer configured (the config changed under a running stream), which source it
        # used is unknown, so all of them go off.
        channel_config = next((ch for ch in self.channels if ch.handle == stopped.channel.handle), None)
        try:
            if channel_config is None:
                logger.warning(
                    f"squid - streamed channel {stopped.channel.handle} is no longer configured, "
                    "turning off all illumination"
                )
                self._illumination_end_all(self.microcontroller)
            else:
                illum_code = ILLUMINATION_CODE.from_slot(channel_config.source_slot)
                self.microcontroller.illumination_end(illum_code.value)
                logger.debug(f"squid - illumination cleanup for channel {stopped.channel.handle}")
        except Exception as e:
            failures.append(("turn off illumination", e))

        _raise_collected(failures)

        logger.debug("squid - completed stream stop")

        result = cmd.BasicSuccessResponse()
        return result  # type: ignore[no-any-return]

    def _illumination_off_during_stall(self) -> str:
        """
        The stall action for main camera calls made with the light on. Runs on the watchdog
        thread while the worker is stuck in the camera, so the microcontroller is free.
        """
        self._illumination_end_all(self.microcontroller)
        return "all illumination was turned off"

    def _af_laser_off_during_stall(self) -> str:
        "the stall action for focus camera snaps, which are made with the autofocus laser on"
        self.microcontroller.af_laser_off()
        return "the autofocus laser was turned off"

    def _illumination_end_all(self, qmc: Microcontroller) -> None:
        """
        Turn off every configured illumination source. Each is attempted whatever happened to
        the others - one that fails is no reason to leave the rest on - and any failure fails
        the call.
        """
        failures: list[tuple[str, BaseException]] = []
        for channel_config in self.channels:
            try:
                illum_src = ILLUMINATION_CODE.from_slot(channel_config.source_slot)
                qmc.illumination_end(illum_src.value)
                logger.debug(
                    f"squid - illumination end all - turned off illumination for {channel_config.name} (slot {channel_config.source_slot})"
                )
            except Exception as e:
                failures.append((f"turn off {channel_config.name} (slot {channel_config.source_slot})", e))
        _raise_collected(failures)
        logger.debug("squid - turned off all illumination")

    def _execute_ChannelSnapshot(
        self,
        qmc: Microcontroller,
        command:cmd.ChannelSnapshot
    ):
        # Find channel config by handle and get illumination code from source slot
        channel_config = None
        for ch in self.channels:
            if ch.handle == command.channel.handle:
                channel_config = ch
                break

        if channel_config is None:
            cmd.error_internal(
                detail=f"Channel handle '{command.channel.handle}' not found in channel configuration"
            )

        try:
            illum_code = ILLUMINATION_CODE.from_slot(channel_config.source_slot)
        except Exception as e:
            cmd.error_internal(
                detail=f"Invalid illumination source slot {channel_config.source_slot} for channel {command.channel.handle}: {e}"
            )

        # Debug output for channel and filter selection
        logger.debug(f"channel snap - using channel '{channel_config.name}' (handle: {channel_config.handle}, illumination slot: {channel_config.source_slot})")

        GlobalConfigHandler.override(command.machine_config)

        if self.is_streaming:
            cmd.error_internal(detail="Cannot take snapshot while camera is streaming")

        # Validate channel configuration for acquisition
        self.validate_channel_for_acquisition(command.channel)

        # Handle filter wheel positioning if filter is specified
        if command.channel.filter_handle is not None:
            # Find filter config by handle
            filter_config = None
            for f in self.filters:
                if f.handle == command.channel.filter_handle:
                    filter_config = f
                    break

            if filter_config is None:
                cmd.error_internal(
                    detail=f"Filter handle '{command.channel.filter_handle}' not found in filter configuration"
                )

            try:
                logger.debug(f"channel snap - using filter '{filter_config.name}' (handle: {filter_config.handle}, wheel position: {filter_config.slot})")
                qmc.filter_wheel_set_position(filter_config.slot)
            except Exception as e:
                cmd.error_internal(
                    detail=f"Failed to set filter wheel to position {filter_config.slot}: {e}"
                )
        else:
            logger.debug("channel snap - no filter specified for this channel")

        logger.debug("channel snap - before illum on")
        # Get calibrated intensity for this channel
        calibrated_intensity = self.illumination_controller.get_calibrated_intensity(
            command.channel.handle, command.channel.illum_perc
        )

        # For LED matrix sources (brightfield), intensity is controlled via RGB values
        if illum_code.is_led_matrix:
            # Convert calibrated intensity to RGB brightness (0-1 range)
            rgb_brightness = calibrated_intensity / 100.0
            qmc.illumination_begin(
                illum_code.value,
                100.0,  # intensity_percent is ignored for LED matrix
                (rgb_brightness, rgb_brightness, rgb_brightness)
            )
        else:
            # For regular sources (lasers), use intensity directly
            qmc.illumination_begin(illum_code.value, calibrated_intensity)

        with self.main_camera.locked() as main_camera:
            if main_camera is None:
                error_internal("main camera is busy")

            logger.debug("channel snap - before acq")

            # even if acqusition fails, ensure light is turned off again!
            try:
                with self._watchdog.guard(
                    Device.MAIN_CAMERA,
                    "snap",
                    command.channel.exposure_time_ms / 1000 + SNAP_DEADLINE_MARGIN_S,
                    on_stall=self._illumination_off_during_stall,
                ):
                    img = main_camera.snap(command.channel)
            finally:
                logger.debug("channel snap - after acq")

                qmc.illumination_end(illum_code.value)
                logger.debug("channel snap - after illum off")

            pixel_format = main_camera.pixel_format
            img, cambits = _process_image(img, pixel_format=pixel_format)

        logger.debug("squid - took channel snapshot")

        result = cmd.ImageAcquiredResponse(
            channel=command.channel,
            width_px=img.shape[1],
            height_px=img.shape[0],
            timestamp=time.time(),
            position=self._get_current_position(qmc),
            pixel_format=pixel_format,
            camera_bit_depth=cambits,
            bit_depth=img.dtype.itemsize * 8,
        )
        from seafront.server.protocol import finalize_capture_ome_xml

        result.ome_xml = finalize_capture_ome_xml(
            result,
            command.ome_context,
            main_camera,
            self.channels,
        )
        result._img = img
        result._cambits = cambits
        return result

    def _execute_ChannelStreamBegin(
        self,
        qmc: Microcontroller,
        command:cmd.ChannelStreamBegin
    ):
        if self.is_streaming:
            cmd.error_internal(detail="Streaming already active - stop current stream before starting a new one")

        # Find channel config by handle and get illumination code from source slot
        channel_config = None
        for ch in self.channels:
            if ch.handle == command.channel.handle:
                channel_config = ch
                break

        if channel_config is None:
            cmd.error_internal(
                detail=f"Channel handle '{command.channel.handle}' not found in channel configuration"
            )

        try:
            illum_code = ILLUMINATION_CODE.from_slot(channel_config.source_slot)
        except Exception as e:
            cmd.error_internal(
                detail=f"Invalid illumination source slot {channel_config.source_slot} for channel {command.channel.handle}: {e}"
            )

        GlobalConfigHandler.override(command.machine_config)

        # Validate channel configuration for acquisition
        self.validate_channel_for_acquisition(command.channel)

        # Handle filter wheel positioning if filter is specified
        if command.channel.filter_handle is not None:
            # Find filter config by handle
            filter_config = None
            for f in self.filters:
                if f.handle == command.channel.filter_handle:
                    filter_config = f
                    break

            if filter_config is None:
                cmd.error_internal(
                    detail=f"Filter handle '{command.channel.filter_handle}' not found in filter configuration"
                )

            try:
                logger.debug(f"channel stream - setting filter wheel to position {filter_config.slot} ({filter_config.name})")
                qmc.filter_wheel_set_position(filter_config.slot)
            except Exception as e:
                cmd.error_internal(
                    detail=f"Failed to set filter wheel to position {filter_config.slot}: {e}"
                )

        # Get calibrated intensity for this channel
        calibrated_intensity = self.illumination_controller.get_calibrated_intensity(
            command.channel.handle, command.channel.illum_perc
        )

        # The light goes on before the camera starts, so anything that fails from here on
        # turns it off again before the failure is passed on.
        try:
            # For LED matrix sources (brightfield), intensity is controlled via RGB values
            if illum_code.is_led_matrix:
                # Convert calibrated intensity to RGB brightness (0-1 range)
                rgb_brightness = calibrated_intensity / 100.0
                qmc.illumination_begin(
                    illum_code.value,
                    100.0,  # intensity_percent is ignored for LED matrix
                    (rgb_brightness, rgb_brightness, rgb_brightness)
                )
            else:
                # For regular sources (lasers), use intensity directly
                qmc.illumination_begin(illum_code.value, calibrated_intensity)

            with self.main_camera.locked() as main_camera:
                if main_camera is None:
                    error_internal("main camera is busy")

                # Snapshot the pixel format now, while we hold the lock, so the streaming
                # callback (which runs on the camera SDK's acquisition thread) never has to
                # touch the camera or take the camera lock. Taking the camera lock inside the
                # callback deadlocks against stop_streaming(): stream_off()/unregister_capture_callback()
                # block until the in-flight callback returns, while the callback blocks on the lock.
                stream_pixel_format = main_camera.pixel_format

                def forward_image(img: np.ndarray) -> None:
                    # Runs on the SDK acquisition thread. _forward_frame checks the stream state
                    # under the narrow lock and calls the sink from outside it, so a slow
                    # consumer never holds up stream-end.
                    img_np, _cambits = _process_image(img.copy(), pixel_format=stream_pixel_format)
                    self._forward_frame(img_np)

                # Marked as streaming before the camera starts, so the first frame the SDK
                # produces already has somewhere to go. Rolled back if the start fails, so a
                # refused begin cannot leave the adapter claiming a stream that is not running.
                self._stream_started(command.channel)
                try:
                    with self._watchdog.guard(
                        Device.MAIN_CAMERA,
                        "start streaming",
                        CALL_DEADLINE_S,
                        on_stall=self._illumination_off_during_stall,
                    ):
                        main_camera.start_streaming(
                            command.channel,
                            callback=forward_image,
                        )
                except Exception:
                    self._stream_stopped()
                    raise
        except BaseException as e:
            try:
                qmc.illumination_end(illum_code.value)
            except Exception as off_error:
                e.add_note(f"turning the illumination off again also failed: {off_error!r}")
            raise

        logger.debug("squid - channel stream begin")

        result = cmd.StreamingStartedResponse(channel=command.channel)
        return result

    def _execute_AutofocusApproachTargetDisplacement(
        self,
        qmc: Microcontroller,
        command:cmd.AutofocusApproachTargetDisplacement
    ):
        # Extract values from command for use in nested function
        config_file = command.config_file
        target_offset_um = command.target_offset_um

        def _estimate_offset_mm():
            res = self._dispatch(
                qmc,
                cmd.AutofocusMeasureDisplacement(config_file=config_file),
            )

            current_displacement_um = res.displacement_um
            assert current_displacement_um is not None

            return (target_offset_um - current_displacement_um) * 1e-3

        if self.is_in_loading_position:
            cmd.error_internal(detail="now allowed while in loading position")

        # get autofocus calibration data
        conf_af_calib_x = ConfigRegistry.get(LaserAutofocusConfig.CALIBRATION_X_PEAK_POS).floatvalue
        conf_af_calib_umpx = ConfigRegistry.get(LaserAutofocusConfig.CALIBRATION_UM_PER_PX).floatvalue
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

        current_state = qmc.get_last_position()
        current_z = current_state.z_pos_mm
        initial_z = current_z

        if command.pre_approach_refz:
            refz_item = None
            try:
                refz_item = ConfigRegistry.get(LaserAutofocusConfig.CALIBRATION_REF_Z_MM)
            except KeyError:
                cmd.error_internal(
                    detail="laser.autofocus.calibration.ref_z_mm is not available when AutofocusApproachTargetDisplacement had pre_approach_refz set"
                )

            assert refz_item is not None

            # move to reference z, only if it is far enough away to make a move worth it
            if (
                math.fabs(current_z - refz_item.floatvalue)
                > OFFSET_MOVEMENT_THRESHOLD_MM
            ):
                res = self._dispatch(
                    qmc,
                    cmd.MoveTo(
                        x_mm=None,
                        y_mm=None,
                        z_mm=refz_item.floatvalue,
                    ),
                )

            logger.debug("autofocus - did pre approach refz")

        last_distance_estimate_mm = 0.0
        num_compensating_moves = 0
        reached_threshold = False
        # the loop only leaves early by setting this; running out of repetitions leaves it as is
        outcome = cmd.AutofocusOutcome.OUT_OF_MOVES
        failure: str | None = None
        try:
            last_distance_estimate_mm = _estimate_offset_mm()
            logger.debug("autofocus - estimated offset")
            last_z_mm = (qmc.get_last_position()).z_pos_mm
            MAX_MOVEMENT_RANGE_MM = 0.3  # should be derived from the calibration data, but this value works fine in practice
            if math.fabs(last_distance_estimate_mm) > MAX_MOVEMENT_RANGE_MM:
                raise ValueError(
                    f"measured autofocus focal plane offset too large: {last_distance_estimate_mm:.3f} mm "
                    f"is outside the {MAX_MOVEMENT_RANGE_MM} mm working range, so the estimate is not trusted"
                )

            for rep_i in range(command.max_num_reps):
                if rep_i == 0:
                    distance_estimate_mm = last_distance_estimate_mm
                else:
                    distance_estimate_mm = _estimate_offset_mm()
                    logger.debug("autofocus - estimated offset")

                # stop if the new estimate indicates a larger distance to the focal plane than the previous estimate
                # (since this indicates that we have moved away from the focal plane, which should not happen)
                if rep_i > 0 and math.fabs(last_distance_estimate_mm) < math.fabs(
                    distance_estimate_mm
                ):
                    # move back to last z, since that seemed like the better position to be in
                    qmc.move_to_mm("z", last_z_mm)
                    logger.debug("autofocus - reset z to known good position")
                    # TODO unsure if this is the best approach. we cannot do better, but we also have not actually gotten close to the offset
                    reached_threshold = True
                    outcome = cmd.AutofocusOutcome.GAVE_UP
                    break

                last_distance_estimate_mm = distance_estimate_mm
                last_z_mm = (qmc.get_position()).z_pos_mm

                # if movement distance is not worth compensating, stop
                if math.fabs(distance_estimate_mm) < OFFSET_MOVEMENT_THRESHOLD_MM:
                    reached_threshold = True
                    outcome = cmd.AutofocusOutcome.IN_FOCUS
                    break

                qmc.move_by_mm("z", distance_estimate_mm)
                num_compensating_moves += 1
                logger.debug("autofocus - refined z")

        except Exception as e:
            # a site the autofocus cannot measure (no spot, a spot out of range) is a result the
            # caller decides about, so it is returned as FAILED with the reason. The stage goes
            # back to where it started, since the next move is relative to wherever this stops.
            # DisconnectError and MicrocontrollerTimeout are BaseExceptions and are not caught.
            failure = str(e.detail) if isinstance(e, HTTPException) else describe(e)
            logger.opt(exception=e).warning(
                f"autofocus - approach failed, returning z to {initial_z:.4f} mm: {failure}"
            )
            try:
                qmc.move_to_mm("z", initial_z)
            except BaseException as reset_error:
                reset_error.add_note(
                    f"while returning z to {initial_z:.4f} mm after the autofocus approach "
                    f"failed: {failure}"
                )
                raise
            outcome = cmd.AutofocusOutcome.FAILED

        logger.debug(f"squid - autofocus approach ended: {outcome.value}")

        res = cmd.AutofocusApproachTargetDisplacementResult(
            num_compensating_moves=num_compensating_moves,
            uncompensated_offset_mm=last_distance_estimate_mm,
            reached_threshold=reached_threshold,
            outcome=outcome,
            failure=failure,
        )
        return res

    def _execute_AutofocusSnap(
        self,
        qmc: Microcontroller,
        command:cmd.AutofocusSnap
    ):
        if command.turn_laser_on:
            qmc.af_laser_on()
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

        # even if acquisition fails, ensure the laser is turned off again - the same guard
        # _execute_ChannelSnapshot puts around its illumination. Without it a camera that
        # raises mid-frame leaves the autofocus laser on the sample, and nothing downstream
        # turns it off: the exception unwinds past every later step, including this one.
        try:
            with self.focus_camera.locked() as focus_camera:
                if focus_camera is None:
                    error_internal("focus camera is busy")
                with self._watchdog.guard(
                    Device.FOCUS_CAMERA,
                    "snap",
                    command.exposure_time_ms / 1000 + SNAP_DEADLINE_MARGIN_S,
                    on_stall=self._af_laser_off_during_stall,
                ):
                    img = focus_camera.snap(channel_config)

            logger.debug("squid - autofocus - acquired image")
        finally:
            if command.turn_laser_off:
                qmc.af_laser_off()
                logger.debug("squid - autofocus - turned laser off")

        result = cmd.AutofocusSnapResult(
            width_px=img.shape[1],
            height_px=img.shape[0],
        )

        # Store raw image - blur is applied in _measure_laser_autofocus when needed
        result._img = img
        result._channel = channel_config
        return result

    async def execute[T](self, command: cmd.BaseCommand[T]) -> T:
        # Single owner (the worker) — no hardware lock needed. ChannelStreamEnd still
        # dispatches directly (it must run while a stream is active but the worker is idle).
        # Either way through the same boundary, so a device lost while stopping a stream is
        # dropped like one lost anywhere else.
        operation = type(command).__name__
        try:
            if isinstance(command, cmd.ChannelStreamEnd):
                self.validate_command(command)
                result = self._execute_ChannelStreamEnd(command)
            else:
                result = self._dispatch(self.microcontroller, command)
            return result # type: ignore[no-any-return]
        except CameraOfflineError as e:
            raise self._drop_connection(self._camera_device(e.camera), operation, e) from e
        except IOError as e:
            raise self._drop_connection(Device.MICROCONTROLLER, operation, e) from e

    def _dispatch(self, qmc: Microcontroller, command: cmd.BaseCommand[tp.Any]) -> tp.Any:
        """Synchronous command dispatch. Caller must already hold the microcontroller lock.

        Also invoked re-entrantly by the _execute_* helpers that issue sub-commands
        (e.g. MoveToWell -> MoveTo, AutofocusApproach -> AutofocusMeasureDisplacement),
        which is why it takes the already-held qmc rather than re-acquiring the lock.
        """
        # Validate every command exactly once, here at the single dispatch choke point.
        # This covers both the outer command and internally-issued sub-commands.
        self.validate_command(command)

        logger.debug(f"squid - executing {type(command).__qualname__}")

        if isinstance(command, cmd.EstablishHardwareConnection):
            return self._execute_EstablishHardwareConnection(qmc, command)

        elif isinstance(command, cmd.LoadingPositionEnter):
            return self._execute_LoadingPositionEnter(qmc, command)

        elif isinstance(command, cmd.LoadingPositionLeave):
            return self._execute_LoadingPositionLeave(qmc, command)

        elif isinstance(command, cmd.MoveBy):
            return self._execute_MoveBy(qmc, command)

        elif isinstance(command, cmd.MoveTo):
            return self._execute_MoveTo(qmc, command)

        elif isinstance(command, cmd.MoveToWell):
            return self._execute_MoveToWell(qmc, command)

        elif isinstance(command, cmd.AutofocusMeasureDisplacement):
            if command.config_file.machine_config is not None:
                GlobalConfigHandler.override(command.config_file.machine_config)

            conf_af_if_calibrated = ConfigRegistry.get(LaserAutofocusConfig.CALIBRATION_IS_CALIBRATED)
            conf_af_calib_x = ConfigRegistry.get(LaserAutofocusConfig.CALIBRATION_X_PEAK_POS)
            conf_af_calib_umpx = ConfigRegistry.get(LaserAutofocusConfig.CALIBRATION_UM_PER_PX)
            if not conf_af_if_calibrated.boolvalue:
                cmd.error_internal(detail="laser autofocus not calibrated")

            # get laser spot location
            # sometimes one of the two expected dots cannot be found in _get_laser_spot_centroid because the plate is so far off the focus plane though, catch that case
            displacement_um = 0
            try:
                calib_params = cmd.LaserAutofocusCalibrationData(
                    um_per_px=conf_af_calib_umpx.floatvalue,
                    x_reference=conf_af_calib_x.floatvalue,
                    calibration_position=cmd.Position.zero(),
                )

                num_images = command.override_num_images or 3
                for _ in range(num_images):
                    displacement_um += (
                        self._approximate_laser_af_displacement_um(qmc, calib_params)
                    ) / num_images

            except Exception as e:
                cmd.error_internal(
                    detail=f"failed to measure displacement (got no signal): {e!s}"
                )

            logger.debug("squid - used autofocus to measure displacement")

            # Apply configured offset to the measured displacement
            conf_af_offset_um = ConfigRegistry.get(LaserAutofocusConfig.OFFSET_UM).floatvalue
            if conf_af_offset_um != 0.0:
                logger.debug(f"adding {conf_af_offset_um} to measured offset {displacement_um} for result {displacement_um+conf_af_offset_um}")
                displacement_um += conf_af_offset_um

            return cmd.AutofocusMeasureDisplacementResult(displacement_um=displacement_um)

        elif isinstance(command, cmd.AutofocusSnap):
            return self._execute_AutofocusSnap(qmc, command)

        elif isinstance(command, cmd.AutofocusLaserWarmup):
            qmc.af_laser_on()

            # wait for the laser to warm up
            time.sleep(command.warmup_time_s)

            qmc.af_laser_off()

            logger.debug("squid - warmed up autofocus laser")

            return cmd.BasicSuccessResponse()

        elif isinstance(command, cmd.IlluminationEndAll):
            self._illumination_end_all(qmc)
            return cmd.BasicSuccessResponse()

        elif isinstance(command, cmd.ChannelSnapshot):
            return self._execute_ChannelSnapshot(qmc, command)

        elif isinstance(command, cmd.ChannelStreamBegin):
            return self._execute_ChannelStreamBegin(qmc, command)

        elif isinstance(command, cmd.LaserAutofocusCalibrate):
            result = self._execute_LaserAutofocusCalibrate(qmc, command)
            logger.debug("squid - calibrated laser autofocus")
            return result

        elif isinstance(command, cmd.AutofocusApproachTargetDisplacement):
            return self._execute_AutofocusApproachTargetDisplacement(qmc, command)

        elif isinstance(command, cmd.GetCurrentPosition):
            logger.debug("squid - fetching current calibrated stage position")
            res = self._get_current_position(qmc)
            logger.debug("squid - fetched current calibrated stage position")
            return res

        else:
            cmd.error_internal(detail=f"Unsupported command type {type(command)}")

    def _validate_parameter_range(
        self, value: float, limit: HardwareLimitValue, param_name: str
    ) -> None:
        """Validate a parameter is within the range the hardware accepts."""
        # Not a number fails every comparison below, so "is it outside the range" answers
        # no and the value goes to the hardware. An illumination percentage of NaN is a laser
        # driven by whatever the conversion happens to produce.
        if not math.isfinite(value):
            cmd.error_internal(detail=f"{param_name} value {value} is not a finite number")

        if value < limit.min:
            cmd.error_internal(detail=f"{param_name} value {value} is below minimum {limit.min}")
        if value > limit.max:
            cmd.error_internal(detail=f"{param_name} value {value} is above maximum {limit.max}")
        # step is not enforced: callers snap with round((val - min) / step) * step + min, and
        # checking compliance here would fail on floating-point representation alone

    def _validate_imaging_parameters(self, **params) -> None:
        """Validate imaging parameters against hardware limits."""
        limits = self.get_hardware_limits()

        # Validate each parameter if provided
        if 'exposure_time_ms' in params:
            self._validate_parameter_range(params['exposure_time_ms'], limits.imaging_exposure_time_ms, "exposure_time_ms")
        if 'analog_gain_db' in params:
            self._validate_parameter_range(params['analog_gain_db'], limits.imaging_analog_gain_db, "analog_gain_db")
        if 'z_offset_um' in params:
            self._validate_parameter_range(params['z_offset_um'], limits.imaging_focus_offset_um, "z_offset_um")
        if 'illum_perc' in params:
            self._validate_parameter_range(params['illum_perc'], limits.imaging_illum_perc, "illum_perc")
        if 'num_z_planes' in params:
            self._validate_parameter_range(params['num_z_planes'], limits.imaging_number_z_planes, "num_z_planes")
        if 'delta_z_um' in params:
            self._validate_parameter_range(params['delta_z_um'], limits.imaging_delta_z_um, "delta_z_um")

    def _validate_acquisition_config(self, config: sc.AcquisitionConfig) -> None:
        """Validate all channels in an acquisition config."""
        for channel in config.channels:
            if not channel.enabled:
                continue
            self._validate_imaging_parameters(
                exposure_time_ms=channel.exposure_time_ms,
                analog_gain_db=channel.analog_gain,
                z_offset_um=channel.z_offset_um,
                illum_perc=channel.illum_perc,
                num_z_planes=channel.num_z_planes,
                delta_z_um=channel.delta_z_um
            )

    def validate_command(self, command: cmd.BaseCommand[tp.Any]) -> None:
        """Validate command parameters against hardware limits."""
        if isinstance(command, cmd.ChannelSnapshot):
            self._validate_imaging_parameters(
                exposure_time_ms=command.channel.exposure_time_ms,
                analog_gain_db=command.channel.analog_gain,
                z_offset_um=command.channel.z_offset_um,
                illum_perc=command.channel.illum_perc,
                num_z_planes=command.channel.num_z_planes,
                delta_z_um=command.channel.delta_z_um
            )

        elif isinstance(command, cmd.ChannelStreamBegin):
            self._validate_imaging_parameters(
                exposure_time_ms=command.channel.exposure_time_ms,
                analog_gain_db=command.channel.analog_gain,
                z_offset_um=command.channel.z_offset_um,
                illum_perc=command.channel.illum_perc,
                num_z_planes=command.channel.num_z_planes,
                delta_z_um=command.channel.delta_z_um
            )

        elif isinstance(command, cmd.AutofocusMeasureDisplacement):
            self._validate_acquisition_config(command.config_file)

        # Other commands don't require validation (movement, connection, etc.)

    def validate_channel_for_acquisition(self, channel: sc.AcquisitionChannelConfig) -> None:
        """
        Validate that a channel can be acquired with current microscope configuration.

        For SquidAdapter, validates that if filters are configured, the channel
        has a filter selected.

        Args:
            channel: The channel configuration to validate

        Raises:
            Calls error_internal() if validation fails
        """
        if self.filters and channel.enabled:
            if channel.filter_handle is None or channel.filter_handle == '':
                cmd.error_internal(
                    detail=f"Channel '{channel.name}' has no filter selected, but filter wheel is available. "
                    "Please select a filter for all enabled channels."
                )

        if not channel.enabled:
            return

        # The same limits every snap is checked against, applied before the run starts
        # rather than when it reaches its first image. A protocol whose illumination is out
        # of range is refusable in a millisecond at the front door; the same protocol
        # accepted and started fails partway through a plate, with images already written.
        self._validate_imaging_parameters(
            exposure_time_ms=channel.exposure_time_ms,
            analog_gain_db=channel.analog_gain,
            z_offset_um=channel.z_offset_um,
            illum_perc=channel.illum_perc,
            num_z_planes=channel.num_z_planes,
            delta_z_um=channel.delta_z_um,
        )

    def extend_machine_config(self, config_items: list[ConfigItem]) -> None:
        """
        Extend machine configuration with microscope-specific options.

        Delegates to cameras to update their pixel format options based on
        actual hardware capabilities.

        Args:
            config_items: List of ConfigItem objects to modify in-place
        """
        # Before connecting there is no camera to ask; the listing then carries the configured
        # options, and connecting restricts them (update_pixel_format_options). A camera busy
        # with a command is skipped for the same reason. A connected camera that fails to answer
        # is an error for the caller, not a config listing with its options missing.
        if not self.is_connected:
            return
        with self.main_camera.locked(blocking=False) as camera:
            if camera is not None:
                camera.extend_machine_config(config_items)

        with self.focus_camera.locked(blocking=False) as camera:
            if camera is not None:
                camera.extend_machine_config(config_items)

    def estimate_acquisition(self, config: sc.AcquisitionConfig) -> cmd.AcquisitionEstimate:
        """
        Estimate storage and time requirements for an acquisition on SQUID hardware.

        Uses actual exposure times from channel configs plus hardware overhead,
        and estimates stage movement time using firmware velocity/acceleration parameters.
        """
        from seafront.hardware.firmware_config import (
            calculate_movement_time_s,
            estimate_xy_movement_time_s,
            get_firmware_config,
        )

        metrics = cmd.calculate_acquisition_metrics(config)
        firmware_config = get_firmware_config()

        # Calculate time per site: sum of exposure times for all enabled channels/z-planes
        enabled_channels = [c for c in config.channels if c.enabled]
        exposure_time_per_site_ms = sum(
            c.exposure_time_ms * c.num_z_planes for c in enabled_channels
        )

        # Hardware overhead per image: autofocus, camera readout, etc. (not including stage movement)
        OVERHEAD_PER_IMAGE_MS = 100

        num_sites_total = metrics.num_wells * metrics.num_sites * metrics.num_timepoints
        time_per_site_ms = exposure_time_per_site_ms + (OVERHEAD_PER_IMAGE_MS * metrics.num_z_planes_total)
        estimated_time_s = (num_sites_total * time_per_site_ms) / 1000

        # XY stabilization time after each XY move
        xy_stabilization_time_s = max(
            firmware_config.SCAN_STABILIZATION_TIME_MS_X,
            firmware_config.SCAN_STABILIZATION_TIME_MS_Y,
        ) / 1000.0

        # Z stabilization time after each Z move
        z_stabilization_time_s = firmware_config.SCAN_STABILIZATION_TIME_MS_Z / 1000.0

        # Z movement time per image (protocol moves Z for each image)
        # Estimate typical Z move distance as ~10um between adjacent z-positions
        typical_z_move_mm = 0.01
        z_move_time_s = calculate_movement_time_s(
            typical_z_move_mm,
            firmware_config.MAX_VELOCITY_Z_mm,
            firmware_config.MAX_ACCELERATION_Z_mm,
        )
        # Add Z overhead for each image (movement + stabilization)
        estimated_time_s += metrics.total_num_images * (z_move_time_s + z_stabilization_time_s)

        # Backlash compensation: 2 extra Z moves of 40um per site
        backlash_move_mm = 0.04
        backlash_move_time_s = calculate_movement_time_s(
            backlash_move_mm,
            firmware_config.MAX_VELOCITY_Z_mm,
            firmware_config.MAX_ACCELERATION_Z_mm,
        )
        estimated_time_s += num_sites_total * 2 * (backlash_move_time_s + z_stabilization_time_s)

        # Estimate inter-well movement time (movement + stabilization)
        if metrics.num_wells > 1:
            plate = config.wellplate_type
            # Use 2x well distance as typical inter-well movement (accounts for non-adjacent wells)
            typical_move_x = plate.Well_distance_x_mm * 2
            typical_move_y = plate.Well_distance_y_mm * 2
            time_per_well_move_s = estimate_xy_movement_time_s(typical_move_x, typical_move_y)
            num_well_moves = (metrics.num_wells - 1) * metrics.num_timepoints
            estimated_time_s += num_well_moves * (time_per_well_move_s + xy_stabilization_time_s)

        # Intra-well site-to-site movement (if multiple sites per well)
        if metrics.num_sites > 1:
            # Estimate typical site spacing based on grid
            site_spacing_mm = 1.0  # Conservative estimate for site spacing
            time_per_site_move_s = estimate_xy_movement_time_s(site_spacing_mm, site_spacing_mm)
            num_site_moves = metrics.num_wells * (metrics.num_sites - 1) * metrics.num_timepoints
            estimated_time_s += num_site_moves * (time_per_site_move_s + xy_stabilization_time_s)

        # Add time between timepoints for time series
        if metrics.num_timepoints > 1:
            delta_t_s = config.grid.delta_t.h * 3600 + config.grid.delta_t.m * 60 + config.grid.delta_t.s
            imaging_time_per_timepoint = estimated_time_s / metrics.num_timepoints
            wait_per_timepoint = max(0, delta_t_s - imaging_time_per_timepoint)
            estimated_time_s += wait_per_timepoint * (metrics.num_timepoints - 1)

        return cmd.AcquisitionEstimate(
            project_name=config.project_name,
            plate_name=config.plate_name,
            num_selected_wells=metrics.num_wells,
            num_selected_sites=metrics.num_sites,
            num_channels=metrics.num_channels,
            num_z_planes_total=metrics.num_z_planes_total,
            num_timepoints=metrics.num_timepoints,
            total_num_images=metrics.total_num_images,
            max_storage_size_GB=metrics.max_storage_size_GB,
            estimated_time_s=estimated_time_s,
        )
