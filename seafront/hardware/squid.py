import asyncio
import math
import os
import threading
import typing as tp
from contextlib import contextmanager

import cv2
import json5
import numpy as np
import scipy  # to find peaks in a signal

# import scipy.ndimage  # for guassian blur
import seaconfig as sc

from seafront.hardware.camera.galaxy_camera import GalaxyCameraOffline
from matplotlib import pyplot as plt
from pydantic import BaseModel
from scipy import stats  # for linear regression

from seafront.config.basics import (
    CameraDriver,
    ChannelConfig,
    ConfigItem,
    FilterConfig,
    GlobalConfigHandler,
    ImagingOrder,
)
from seafront.config.registry import ConfigRegistry
from seafront.config.handles import (
    CalibrationConfig,
    CameraConfig,
    FilterWheelConfig,
    ImageConfig,
    ImagingConfig,
    LaserAutofocusConfig,
    MicrocontrollerConfig,
)
from seafront.hardware import microcontroller as mc
from seafront.hardware.adapter import AdapterState
from seafront.hardware.camera import (
    Camera,
    CameraOpenRequest,
    HardwareLimitValue,
    camera_open,
)
from seafront.hardware.illumination import IlluminationController
from seafront.hardware.adapter import DeviceAlreadyInUseError
from seafront.hardware.microscope import DisconnectError, HardwareLimits, Locked, Microscope, microscope_exclusive
from seafront.logger import logger
from seafront.server import commands as cmd
from seafront.server.commands import (
    BasicSuccessResponse,
    IlluminationEndAll,
    error_device_in_use,
    error_internal,
)

from seafront.config.handles import ProtocolConfig
from seafront.hardware.forbidden_areas import ForbiddenAreaList

# utility functions


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

    cam_img_width = CameraConfig.MAIN_IMAGE_WIDTH_PX.value_item
    assert cam_img_width is not None
    target_width: int = cam_img_width.intvalue
    assert isinstance(target_width, int), f"{type(target_width) = }"
    cam_img_height = CameraConfig.MAIN_IMAGE_HEIGHT_PX.value_item
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

    flip_img_horizontal = CameraConfig.MAIN_IMAGE_FLIP_HORIZONTAL.value_item
    assert flip_img_horizontal is not None
    if flip_img_horizontal.boolvalue:
        ret = np.flip(ret, axis=1)

    flip_img_vertical = CameraConfig.MAIN_IMAGE_FLIP_VERTICAL.value_item
    assert flip_img_vertical is not None
    if flip_img_vertical.boolvalue:
        ret = np.flip(ret, axis=0)

    image_file_pad_low = ImageConfig.FILE_PAD_LOW.value_item
    assert image_file_pad_low is not None
    cambits = 0

    match camera.pixel_format:
        case "mono8":
            cambits = 8
        case "mono10":
            cambits = 10
        case "mono12":
            cambits = 12
        case "mono14":
            cambits = 14
        case "mono16":
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


class SquidAdapter(Microscope):
    """interface to squid microscope"""

    # SQUID-specific hardware components
    main_camera: Locked[Camera]
    focus_camera: Locked[Camera]
    microcontroller: mc.Microcontroller
    illumination_controller: IlluminationController

    @contextmanager
    def lock(self, blocking: bool = True, reason: str = "unknown") -> tp.Iterator[tp.Self | None]:
        "lock all hardware devices"
        if self._lock.acquire(blocking=blocking):
            self._lock_reasons.append(reason)
            try:
                with (
                    self.main_camera.locked(blocking=blocking),
                    self.focus_camera.locked(blocking=blocking),
                    self.microcontroller.locked(blocking=blocking),
                ):
                    yield self
            finally:
                self._lock_reasons.pop()
                self._lock.release()
        else:
            yield None

    @classmethod
    def make(cls) -> "SquidAdapter":
        g_dict = GlobalConfigHandler.get_dict()

        # Get device configuration (USB IDs)
        main_camera_id = g_dict[CameraConfig.MAIN_ID.value].strvalue
        main_camera_driver = g_dict[CameraConfig.MAIN_DRIVER.value].strvalue
        focus_camera_id = g_dict[LaserAutofocusConfig.CAMERA_ID.value].strvalue
        focus_camera_driver = g_dict[LaserAutofocusConfig.CAMERA_DRIVER.value].strvalue
        microcontroller_id = g_dict[MicrocontrollerConfig.ID.value].strvalue

        # Validate camera drivers
        valid_drivers = tp.get_args(CameraDriver)  # Get valid values from the Literal type
        if main_camera_driver not in valid_drivers:
            error_msg = f"invalid main camera driver '{main_camera_driver}'. Valid drivers: {valid_drivers}"
            logger.critical(f"startup - {error_msg}")
            cmd.error_internal(detail=error_msg)

        if focus_camera_driver not in valid_drivers:
            error_msg = f"invalid focus camera driver '{focus_camera_driver}'. Valid drivers: {valid_drivers}"
            logger.critical(f"startup - {error_msg}")
            cmd.error_internal(detail=error_msg)

        # Find devices by USB ID
        main_camera_request = CameraOpenRequest(driver=main_camera_driver, usb_id=main_camera_id)  # type: ignore
        focus_camera_request = CameraOpenRequest(driver=focus_camera_driver, usb_id=focus_camera_id)  # type: ignore

        main_camera = camera_open(main_camera_request)
        focus_camera = camera_open(focus_camera_request)
        microcontroller = mc.Microcontroller.find_by_usb_id(microcontroller_id)

        # Extract channel and filter configurations (native objects from config)
        channels_data = ConfigRegistry.get(ImagingConfig.CHANNELS.value).objectvalue
        channel_configs = [ChannelConfig(**ch) for ch in channels_data] #type: ignore

        # Only process filters if filter wheel is available
        filter_wheel_available = g_dict.get(FilterWheelConfig.AVAILABLE.value)
        if filter_wheel_available and filter_wheel_available.boolvalue:
            filters_data = ConfigRegistry.get(FilterWheelConfig.CONFIGURATION.value).objectvalue
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

    @microscope_exclusive
    def open_connections(self):
        """open connections to devices"""
        if self.is_connected:
            return

        with (
            self.microcontroller.locked() as mc,
            self.main_camera.locked() as main_camera,
            self.focus_camera.locked() as focus_camera,
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
            except DeviceAlreadyInUseError:
                # Don't convert to DisconnectError - let it propagate for specific handling
                raise
            except GalaxyCameraOffline as e:
                logger.critical("startup - camera offline")
                raise DisconnectError() from e
            except IOError as e:
                logger.critical("startup - microcontroller offline")
                raise DisconnectError() from e

            # Fetch actual camera capabilities and update config with runtime values
            from seafront.config.basics import GlobalConfigHandler

            try:
                main_formats = main_camera.get_supported_pixel_formats()
                focus_formats = focus_camera.get_supported_pixel_formats()
                logger.debug(f"startup - main camera supports formats: {main_formats}")
                logger.debug(f"startup - focus camera supports formats: {focus_formats}")
                GlobalConfigHandler.update_pixel_format_options(main_formats, focus_formats)
                logger.info("startup - updated config with actual camera pixel format capabilities")
            except Exception as e:
                logger.warning(f"startup - failed to update pixel format options: {e}")
                # Don't fail startup if this fails, but log the warning

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
            self.main_camera.locked() as main_camera,
            self.focus_camera.locked() as focus_camera,
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
                return # unreachable but satisfies the type checker

            try:
                logger.info("starting stage calibration (by entering loading position)")

                # reset the MCU
                logger.debug("resetting mcu")
                await qmc.send_cmd(mc.Command.reset())
                logger.debug("done")

                # reinitialize motor drivers and DAC
                logger.debug("initializing microcontroller")
                await qmc.send_cmd(mc.Command.initialize())
                logger.debug("done initializing microcontroller")

                if True:
                    # disable for testing (new firmware should have better defaults)
                    logger.debug("configure_actuators")
                    await qmc.send_cmd(mc.Command.configure_actuators())
                    logger.debug("done configuring actuators")

                logger.info("ensuring illumination is off")
                # make sure all illumination is off
                for illum_src in [
                    # turn off all fluorescence LEDs
                    mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_FLUOSLOT11,
                    mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_FLUOSLOT12,
                    mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_FLUOSLOT13,
                    mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_FLUOSLOT14,
                    mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_FLUOSLOT15,
                    # this will turn off the led matrix
                    mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_LED_ARRAY_FULL,
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

                # Only initialize filter wheel if it's available
                g_dict = GlobalConfigHandler.get_dict()
                filter_wheel_available = g_dict.get(FilterWheelConfig.AVAILABLE.value)
                if filter_wheel_available and filter_wheel_available.boolvalue:
                    # Initialize filter wheel with homing sequence (matching Squid behavior)
                    logger.info("initializing filter wheel...")
                    try:
                        await qmc.filter_wheel_init()
                        logger.info("configuring filter wheel actuator...")
                        await qmc.filter_wheel_configure_actuator()
                        logger.info("performing filter wheel homing...")
                        await qmc.filter_wheel_home()
                        logger.info("✓ Filter wheel initialized, configured, and homed")
                    except Exception as e:
                        logger.warning(f"Filter wheel initialization/configuration/homing failed: {e}")
                        logger.info("Continuing with microscope initialization...")
                else:
                    logger.info("Filter wheel not available - skipping initialization")

                logger.info("done initializing microscope")

            except IOError as e:
                logger.critical("lost connection to microcontroller")
                self.close()
                raise DisconnectError() from e

    def _sort_channels_by_imaging_order(self, channels: list, imaging_order: ImagingOrder) -> list:
        """
        Sort channels according to the specified imaging order.
        
        Args:
            channels: List of enabled AcquisitionChannelConfig objects
            imaging_order: Sorting strategy to use
            
        Returns:
            Sorted list of channels
        """
        if imaging_order == "z_order":
            # Sort by z_offset_um (lowest to highest z coordinate)
            return sorted(channels, key=lambda ch: ch.z_offset_um)

        elif imaging_order == "wavelength_order":
            # Sort by wavelength (highest to lowest), then brightfield last
            def wavelength_sort_key(ch):
                # Extract wavelength from channel name if possible
                import re
                wavelength_match = re.search(r'(\d+)\s*nm', ch.name, re.IGNORECASE)
                if wavelength_match:
                    wavelength = int(wavelength_match.group(1))
                    # Higher wavelengths first (descending order)
                    return (0, -wavelength)
                elif 'brightfield' in ch.name.lower() or 'bf' in ch.name.lower():
                    # Brightfield comes last
                    return (1, 0)
                else:
                    # Unknown wavelength, put in middle
                    return (0, -500)  # Assume ~500nm for unknown

            return sorted(channels, key=wavelength_sort_key)

        elif imaging_order == "protocol_order":
            # Keep original order from config file
            return channels

        else:
            # Default to protocol order for unknown values
            logger.warning(f"Unknown imaging order '{imaging_order}', using protocol_order")
            return channels

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

            laf_is_calibrated = g_config.get(
                LaserAutofocusConfig.CALIBRATION_IS_CALIBRATED.value
            )

            # get channels from that, filter for selected/enabled channels
            channels = [c for c in config_file.channels if c.enabled]

            # get imaging order from machine config, default to protocol_order
            imaging_order = g_config.get(ImagingConfig.ORDER.value, "protocol_order")
            if isinstance(imaging_order, str):
                imaging_order_value = imaging_order
            else:
                imaging_order_value = imaging_order.strvalue if hasattr(imaging_order, 'strvalue') else "protocol_order"

            # sort channels according to configured imaging order
            channels = self._sort_channels_by_imaging_order(channels, tp.cast(ImagingOrder, imaging_order_value))

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

        conf_af_exp_ms_item = LaserAutofocusConfig.EXPOSURE_TIME_MS.value_item
        assert conf_af_exp_ms_item is not None
        conf_af_exp_ms = conf_af_exp_ms_item.floatvalue

        conf_af_exp_ag_item = LaserAutofocusConfig.CAMERA_ANALOG_GAIN.value_item
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

    async def _execute_LaserAutofocusCalibrate(
        self,
        qmc:mc.Microcontroller,
        command:cmd.LaserAutofocusCalibrate,
    ) -> cmd.LaserAutofocusCalibrationResponse:
    
        # Hardcoded calibration parameters
        Z_MM_BACKLASH_COUNTER = 40e-3
        DEBUG_LASER_AF_CALIBRATION = os.getenv("DEBUG_LASER_AF_CALIBRATION", "").lower() == "true"
        DEBUG_LASER_AF_SHOW_REGRESSION_FIT = os.getenv("DEBUG_LASER_AF_SHOW_REGRESSION_FIT", "").lower() == "true"
        DEBUG_LASER_AF_SHOW_EVAL_FIT = os.getenv("DEBUG_LASER_AF_SHOW_EVAL_FIT", "true").lower() == "true"

        # Read number of z steps from machine config
        conf_num_steps_item = LaserAutofocusConfig.CALIBRATION_NUM_Z_STEPS.value_item
        assert conf_num_steps_item is not None
        num_z_steps_calibrate = conf_num_steps_item.intvalue

        if self.is_in_loading_position:
            cmd.error_internal(detail="now allowed while in loading position")

        # Read z_span from machine config
        conf_z_span_item = LaserAutofocusConfig.CALIBRATION_Z_SPAN_MM.value_item
        assert conf_z_span_item is not None
        z_mm_movement_range = conf_z_span_item.floatvalue
        logger.debug(f"laser autofocus calibration: z_span={z_mm_movement_range}mm, num_steps={num_z_steps_calibrate}")

        conf_af_exp_ms_item = LaserAutofocusConfig.EXPOSURE_TIME_MS.value_item
        assert conf_af_exp_ms_item is not None
        conf_af_exp_ms = conf_af_exp_ms_item.floatvalue

        conf_af_exp_ag_item = LaserAutofocusConfig.CAMERA_ANALOG_GAIN.value_item
        assert conf_af_exp_ag_item is not None
        conf_af_exp_ag = conf_af_exp_ag_item.floatvalue

        z_step_mm = z_mm_movement_range / (num_z_steps_calibrate - 1)
        half_z_mm = z_mm_movement_range / 2

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

        for i in range(num_z_steps_calibrate):
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

            half_z_mm = z_mm_movement_range / 2
            num_z_steps_eval = 51
            z_step_mm = z_mm_movement_range / (num_z_steps_eval - 1)

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

        logger.debug(f"laser autofocus calibration result: um_per_px={um_per_px:.4f}, x_reference={x_reference:.1f}")
        return cmd.LaserAutofocusCalibrationResponse(calibration_data=calibration_data)

    async def get_current_state(self) -> AdapterState:
        try:
            last_stage_position = await self.microcontroller.get_last_position()
        except IOError as e:
            self.close()
            logger.debug("microcontroller disconnected (IOError)")
            raise DisconnectError() from e

        # supposed=real-calib
        x_pos_mm = self._pos_x_measured_to_real(last_stage_position.x_pos_mm)
        y_pos_mm = self._pos_y_measured_to_real(last_stage_position.y_pos_mm)
        z_pos_mm = self._pos_z_measured_to_real(last_stage_position.z_pos_mm)

        self.last_state = AdapterState(
            is_in_loading_position=self.is_in_loading_position,
            stage_position=cmd.Position(
                x_pos_mm=x_pos_mm,
                y_pos_mm=y_pos_mm,
                z_pos_mm=z_pos_mm,
            ),
        )

        return self.last_state

    def get_hardware_limits(self) -> HardwareLimits:
        """
        Get hardware-specific limits by querying camera and combining with SQUID mechanical limits.
        """
        # Get camera-specific limits (exposure time and analog gain)
        with self.main_camera.locked() as main_camera:
            if main_camera is None:
                raise RuntimeError("Main camera not available for hardware limits query")

            # Get exposure time limits from camera hardware
            exposure_limits = main_camera.get_exposure_time_limits()

            # Get analog gain limits from camera hardware
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
            imaging_exposure_time_ms=exposure_limits.to_dict(),
            imaging_analog_gain_db=gain_limits.to_dict(),
            imaging_focus_offset_um=focus_offset_limits.to_dict(),
            imaging_illum_perc=generic_power_limits.to_dict(),
            imaging_illum_perc_fluorescence=fluorescence_power_limits.to_dict(),
            imaging_illum_perc_brightfield=brightfield_power_limits.to_dict(),
            imaging_number_z_planes=z_planes_limits.to_dict(),
            imaging_delta_z_um=z_spacing_limits.to_dict(),
        )

    def is_position_forbidden(self, x_mm: float, y_mm: float) -> tuple[bool, str]:
        """
        Check if a position is forbidden for movement.
        
        returns (<position is forbidden?>, <forbidden reason>)
        """

        try:
            data = ConfigRegistry.get(ProtocolConfig.FORBIDDEN_AREAS.value).objectvalue
        except KeyError:
            # No forbidden areas configured - position is allowed
            return False, ""

        forbidden_areas = ForbiddenAreaList.model_validate({"areas": data})
        # Check if position is forbidden
        is_forbidden, conflicting_area = forbidden_areas.is_position_forbidden(x_mm, y_mm)

        if is_forbidden and conflicting_area is not None:
            reason_text = f" ({conflicting_area.reason})" if conflicting_area.reason else ""
            error_msg = f"Movement to ({x_mm:.1f}, {y_mm:.1f}) mm is forbidden - conflicts with area '{conflicting_area.name}'{reason_text}"
            return True, error_msg

        return False, ""

    async def _execute_EstablishHardwareConnection(
        self,
        qmc:mc.Microcontroller,
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
                await self.execute(IlluminationEndAll())
                logger.debug("squid - connect - turned off illumination")
                # 2) perform home maneuver to reset stage position to known values
                logger.debug("squid - connect - calibrating stage position")

                await self.home()

                logger.info("squid - connect - calibrated stage")

            except DeviceAlreadyInUseError as e:
                logger.critical(f"squid - connect - {e}")
                logger.critical("Please close any other instances of Seafront before starting a new one.")
                error_device_in_use(e.device_type, e.device_id)
            except DisconnectError:
                error_message = "hardware connection could not be established"
                logger.critical(f"squid - connect - {error_message}")
                error_internal(error_message)

        logger.debug("squid - connected.")

        result = BasicSuccessResponse()
        return result  # type: ignore[no-any-return]

    async def _execute_LoadingPositionEnter(
        self,
        qmc:mc.Microcontroller,
        command:cmd.LoadingPositionEnter
    ):
        if self.is_in_loading_position:
            cmd.error_internal(detail="already in loading position")

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

        logger.debug("squid - entered loading position")

        result = cmd.BasicSuccessResponse()
        return result  # type: ignore[no-any-return]

    async def _execute_LoadingPositionLeave(
        self,
        qmc:mc.Microcontroller,
        command:cmd.LoadingPositionLeave
    ):
        if not self.is_in_loading_position:
            cmd.error_internal(detail="not in loading position")

        await qmc.send_cmd(mc.Command.move_to_mm("x", 30))
        await qmc.send_cmd(mc.Command.move_to_mm("y", 30))
        await qmc.send_cmd(mc.Command.move_to_mm("z", 1))

        self.is_in_loading_position = False

        logger.debug("squid - left loading position")

        result = cmd.BasicSuccessResponse()
        return result

    async def _execute_MoveBy(
        self,
        qmc:mc.Microcontroller,
        command:cmd.MoveBy
    ):
        if self.is_in_loading_position:
            cmd.error_internal(detail="now allowed while in loading position")

        # Calculate target position after relative movement
        try:
            current_stage_position = await qmc.get_last_position()
        except IOError as e:
            self.close()
            raise DisconnectError() from e

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

        await qmc.send_cmd(mc.Command.move_by_mm(command.axis, command.distance_mm))

        logger.debug("squid - moved by")

        result = cmd.MoveByResult(moved_by_mm=command.distance_mm, axis=command.axis)
        return result

    async def _execute_MoveTo(
        self,
        qmc:mc.Microcontroller,
        command:cmd.MoveTo
    ):
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

        logger.debug("squid - moved to")

        result = cmd.BasicSuccessResponse()
        return result

    async def _execute_MoveToWell(
        self,
        qmc:mc.Microcontroller,
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
        res = await self.execute(cmd.MoveTo(x_mm=x_mm, y_mm=None))
        res = await self.execute(cmd.MoveTo(x_mm=None, y_mm=y_mm))

        logger.debug("squid - moved to well")

        result = cmd.BasicSuccessResponse()
        return result

    async def _execute_ChannelStreamEnd(
        self,
        command:cmd.ChannelStreamEnd
    ):
        if self.stream_callback is None:
            cmd.error_internal(detail="not currently streaming")

        logger.debug("squid - requested stream stop")

        # If natural cleanup didn't happen, force comprehensive cleanup to prevent desync
        if self.stream_callback is not None:
            logger.warning("squid - streaming thread didn't clean up naturally, forcing comprehensive cleanup")

            # Comprehensive camera state reset
            try:
                with self.main_camera.locked(blocking=False) as cam:
                    if cam is None:
                        raise RuntimeError("failed to acquire camera to stop streaming!")

                    logger.debug("squid - stopping camera streaming")
                    cam.stop_streaming()
                    logger.debug("squid - forced camera streaming stop")
            except Exception as e:
                logger.warning(f"squid - failed to force camera streaming stop: {e}")
            
            # Find channel config to clean up illumination
            channel_config = None
            for ch in self.channels:
                if ch.handle == command.channel.handle:
                    channel_config = ch
                    break

            if channel_config is not None:
                try:
                    illum_code = mc.ILLUMINATION_CODE.from_slot(channel_config.source_slot)
                    with self.microcontroller.locked() as qmc:
                        if qmc is None:
                            logger.debug("failed to acquire microcontroller locl to turn off illumination!!")
                        else:
                            await qmc.send_cmd(mc.Command.illumination_end(illum_code))
                            logger.debug(f"squid - forced illumination cleanup for channel {command.channel.handle}")
                except Exception as e:
                    logger.warning(f"squid - failed to force illumination cleanup: {e}")

            # Force microscope state cleanup
            self.stream_callback = None
            logger.debug("squid - completed forced comprehensive cleanup")

        result = cmd.BasicSuccessResponse()
        return result  # type: ignore[no-any-return]

    async def _execute_ChannelSnapshot(
        self,
        qmc:mc.Microcontroller,
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
            illum_code = mc.ILLUMINATION_CODE.from_slot(channel_config.source_slot)
        except Exception as e:
            cmd.error_internal(
                detail=f"Invalid illumination source slot {channel_config.source_slot} for channel {command.channel.handle}: {e}"
            )

        # Debug output for channel and filter selection
        logger.debug(f"channel snap - using channel '{channel_config.name}' (handle: {channel_config.handle}, illumination slot: {channel_config.source_slot})")

        GlobalConfigHandler.override(command.machine_config)

        if self.stream_callback is not None:
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
                await qmc.filter_wheel_set_position(filter_config.slot)
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
            await qmc.send_cmd(
                mc.Command.illumination_begin(
                    illum_code,
                    100.0,  # intensity_percent is ignored for LED matrix
                    rgb_brightness,  # R
                    rgb_brightness,  # G
                    rgb_brightness   # B
                )
            )
        else:
            # For regular sources (lasers), use intensity directly
            await qmc.send_cmd(
                mc.Command.illumination_begin(illum_code, calibrated_intensity)
            )

        with self.main_camera.locked() as main_camera:
            if main_camera is None:
                error_internal("main camera is busy")

            logger.debug("channel snap - before acq")

            # even if acqusition fails, ensure light is turned off again!
            try:
                img = main_camera.snap(command.channel)
            finally:
                logger.debug("channel snap - after acq")

                await qmc.send_cmd(mc.Command.illumination_end(illum_code))
                logger.debug("channel snap - after illum off")

            img, cambits = _process_image(img, camera=main_camera)

        logger.debug("squid - took channel snapshot")

        result = cmd.ImageAcquiredResponse()
        result._img = img
        result._cambits = cambits
        return result

    async def _execute_ChannelStreamBegin(
        self,
        qmc:mc.Microcontroller,
        command:cmd.ChannelStreamBegin
    ):
        if self.stream_callback is not None:
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
            illum_code = mc.ILLUMINATION_CODE.from_slot(channel_config.source_slot)
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
                await qmc.filter_wheel_set_position(filter_config.slot)
            except Exception as e:
                cmd.error_internal(
                    detail=f"Failed to set filter wheel to position {filter_config.slot}: {e}"
                )

        # Get calibrated intensity for this channel
        calibrated_intensity = self.illumination_controller.get_calibrated_intensity(
            command.channel.handle, command.channel.illum_perc
        )

        # For LED matrix sources (brightfield), intensity is controlled via RGB values
        if illum_code.is_led_matrix:
            # Convert calibrated intensity to RGB brightness (0-1 range)
            rgb_brightness = calibrated_intensity / 100.0
            await qmc.send_cmd(
                mc.Command.illumination_begin(
                    illum_code,
                    100.0,  # intensity_percent is ignored for LED matrix
                    rgb_brightness,  # R
                    rgb_brightness,  # G
                    rgb_brightness   # B
                )
            )
        else:
            # For regular sources (lasers), use intensity directly
            await qmc.send_cmd(
                mc.Command.illumination_begin(illum_code, calibrated_intensity)
            )

        def forward_image(img: np.ndarray) -> None:
            if self.stream_callback is None:
                return

            img_np = img.copy()

            # camera must only be locked for specific image acquisition, not for the whole duration where an image may be acquired
            # there is currently no way to solve this well.. (this current solution is quite brittle)
            with self.main_camera.locked() as main_camera:
                if main_camera is None:
                    error_internal("main camera is busy")
                img_np, cambits = _process_image(img_np, camera=main_camera)

            self.stream_callback(img_np)

        with self.main_camera.locked() as main_camera:
            if main_camera is None:
                error_internal("main camera is busy")

            main_camera.start_streaming(
                command.channel,
                callback=forward_image,
            )

        logger.debug("squid - channel stream begin")

        result = cmd.StreamingStartedResponse(channel=command.channel)
        return result

    async def _execute_AutofocusApproachTargetDisplacement(
        self,
        qmc:mc.Microcontroller,
        command:cmd.AutofocusApproachTargetDisplacement
    ):
        # Extract values from command for use in nested function
        config_file = command.config_file
        target_offset_um = command.target_offset_um

        async def _estimate_offset_mm():
            res = await self.execute(
                cmd.AutofocusMeasureDisplacement(config_file=config_file)
            )

            current_displacement_um = res.displacement_um
            assert current_displacement_um is not None

            return (target_offset_um - current_displacement_um) * 1e-3

        if self.is_in_loading_position:
            cmd.error_internal(detail="now allowed while in loading position")

        # get autofocus calibration data
        conf_af_calib_x = LaserAutofocusConfig.CALIBRATION_X_PEAK_POS.value_item.floatvalue
        conf_af_calib_umpx = LaserAutofocusConfig.CALIBRATION_UM_PER_PX.value_item.floatvalue
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
            refz_item = None
            try:
                refz_item = LaserAutofocusConfig.CALIBRATION_REF_Z_MM.value_item
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
                res = await self.execute(
                    cmd.MoveTo(
                        x_mm=None,
                        y_mm=None,
                        z_mm=refz_item.floatvalue,
                    )
                )

            logger.debug("autofocus - did pre approach refz")

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

        logger.debug("squid - used autofocus to approach target displacement")

        res = cmd.AutofocusApproachTargetDisplacementResult(
            num_compensating_moves=num_compensating_moves,
            uncompensated_offset_mm=last_distance_estimate_mm,
            reached_threshold=reached_threshold,
        )
        return res

    async def _execute_AutofocusSnap(
        self,
        qmc:mc.Microcontroller,
        command:cmd.AutofocusSnap
    ):
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

        with self.focus_camera.locked() as focus_camera:
            if focus_camera is None:
                error_internal("focus camera is busy")
            img = focus_camera.snap(channel_config)

        logger.debug("squid - autofocus - acquired image")

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
        return result

    #async def _execute_(self,command:cmd.)

    async def execute[T](self, command: cmd.BaseCommand[T]) -> T:
        # Validate command against hardware limits first
        self.validate_command(command)

        # Handle ChannelStreamEnd without locking to avoid deadlock
        if isinstance(command, cmd.ChannelStreamEnd):
            result = await self._execute_ChannelStreamEnd(command)
            return result # type: ignore[no-any-return]

        with self.microcontroller.locked() as qmc:
            if qmc is None:
                error_internal("microcontroller is busy")

            try:
                logger.debug(f"squid - executing {type(command).__qualname__}")

                if isinstance(command, cmd.EstablishHardwareConnection):
                    result = await self._execute_EstablishHardwareConnection(qmc,command)
                    return result # type: ignore[no-any-return]

                elif isinstance(command, cmd.LoadingPositionEnter):
                    result = await self._execute_LoadingPositionEnter(qmc,command)
                    return result # type: ignore[no-any-return]

                elif isinstance(command, cmd.LoadingPositionLeave):
                    result = await self._execute_LoadingPositionLeave(qmc,command)
                    return result # type: ignore[no-any-return]

                elif isinstance(command, cmd.MoveBy):
                    result = await self._execute_MoveBy(qmc,command)
                    return result # type: ignore[no-any-return]

                elif isinstance(command, cmd.MoveTo):
                    result = await self._execute_MoveTo(qmc,command)
                    return result # type: ignore[no-any-return]

                elif isinstance(command, cmd.MoveToWell):
                    result = await self._execute_MoveToWell(qmc,command)
                    return result # type: ignore[no-any-return]

                elif isinstance(command, cmd.AutofocusMeasureDisplacement):
                    if command.config_file.machine_config is not None:
                        GlobalConfigHandler.override(command.config_file.machine_config)

                    g_config = LaserAutofocusConfig.CALIBRATION_IS_CALIBRATED.get_dict()

                    conf_af_if_calibrated = g_config.get(str(LaserAutofocusConfig.CALIBRATION_IS_CALIBRATED))
                    conf_af_calib_x = g_config.get(str(LaserAutofocusConfig.CALIBRATION_X_PEAK_POS))
                    conf_af_calib_umpx = g_config.get(str(LaserAutofocusConfig.CALIBRATION_UM_PER_PX))
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

                    # Apply configured offset to the measured displacement
                    conf_af_offset_um = g_config.get(str(LaserAutofocusConfig.OFFSET_UM))
                    if conf_af_offset_um is not None:
                        logger.debug(f"adding {conf_af_offset_um.floatvalue} to measured offset {displacement_um} for result {displacement_um+conf_af_offset_um.floatvalue}")
                        displacement_um += conf_af_offset_um.floatvalue

                    result = cmd.AutofocusMeasureDisplacementResult(displacement_um=displacement_um)
                    return result # type: ignore[no-any-return]

                elif isinstance(command, cmd.AutofocusSnap):
                    result = await self._execute_AutofocusSnap(qmc,command)
                    return result  # type: ignore[no-any-return]

                elif isinstance(command, cmd.AutofocusLaserWarmup):
                    await qmc.send_cmd(mc.Command.af_laser_illum_begin())

                    # wait for the laser to warm up
                    await asyncio.sleep(command.warmup_time_s)

                    await qmc.send_cmd(mc.Command.af_laser_illum_end())

                    logger.debug("squid - warmed up autofocus laser")

                    result = cmd.BasicSuccessResponse()
                    return result # type: ignore[no-any-return]

                elif isinstance(command, cmd.IlluminationEndAll):
                    # Turn off all configured illumination sources
                    for channel_config in self.channels:
                        try:
                            illum_src = mc.ILLUMINATION_CODE.from_slot(channel_config.source_slot)
                            await qmc.send_cmd(mc.Command.illumination_end(illum_src))
                            logger.debug(
                                f"squid - illumination end all - turned off illumination for {channel_config.name} (slot {channel_config.source_slot})"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to turn off illumination for channel {channel_config.name} (slot {channel_config.source_slot}): {e}"
                            )

                    logger.debug("squid - turned off all illumination")

                    ret = cmd.BasicSuccessResponse()
                    return ret # type: ignore[no-any-return]

                elif isinstance(command, cmd.ChannelSnapshot):
                    result = await self._execute_ChannelSnapshot(qmc,command)
                    return result # type: ignore[no-any-return]

                elif isinstance(command, cmd.ChannelStreamBegin):
                    result = await self._execute_ChannelStreamBegin(qmc,command)
                    return result # type: ignore[no-any-return]

                elif isinstance(command, cmd.LaserAutofocusCalibrate):
                    result = await self._execute_LaserAutofocusCalibrate(qmc,command)
                    logger.debug("squid - calibrated laser autofocus")
                    return result # type: ignore[no-any-return]

                elif isinstance(command, cmd.AutofocusApproachTargetDisplacement):
                    result = await self._execute_AutofocusApproachTargetDisplacement(qmc,command)
                    return result # type: ignore[no-any-return]

                elif isinstance(command, cmd.MC_getLastPosition):
                    logger.debug("squid - fetching last stage position")
                    res = await qmc.get_last_position()
                    logger.debug("squid - fetched last stage position")
                    return res  # type: ignore[no-any-return]

                else:
                    cmd.error_internal(detail=f"Unsupported command type {type(command)}")

            except GalaxyCameraOffline as e:
                logger.critical("squid - lost camera connection")
                raise DisconnectError() from e
            except IOError as e:
                logger.critical("squid - lost connection to microcontroller")
                self.close()
                raise DisconnectError() from e

    def _validate_parameter_range(self, value: float, limit_dict: dict, param_name: str) -> None:
        """Validate a parameter is within specified range and step."""
        min_val = limit_dict.get("min")
        max_val = limit_dict.get("max")
        step = limit_dict.get("step")

        if min_val is not None and value < min_val:
            cmd.error_internal(detail=f"{param_name} value {value} is below minimum {min_val}")
        if max_val is not None and value > max_val:
            cmd.error_internal(detail=f"{param_name} value {value} is above maximum {max_val}")
        if step is not None and step > 0:
            if min_val is not None:
                # Don't validate step compliance - just snap to nearest valid step
                # This avoids floating-point precision issues with modulo operations
                offset = value - min_val
                snapped_offset = round(offset / step) * step
                snapped_value = min_val + snapped_offset
                # We could warn or adjust the value here if needed, but for now just accept it
                # The caller should use: usedval = round((val - min_val) / step) * step + min_val

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

    def extend_machine_config(self, config_items: list[ConfigItem]) -> None:
        """
        Extend machine configuration with microscope-specific options.

        Delegates to cameras to update their pixel format options based on
        actual hardware capabilities.

        Args:
            config_items: List of ConfigItem objects to modify in-place
        """
        try:
            # Ask main camera to extend config
            with self.main_camera.locked(blocking=False) as camera:
                if camera is not None:
                    camera.extend_machine_config(config_items)

            # Ask focus camera to extend config if available
            with self.focus_camera.locked(blocking=False) as camera:
                if camera is not None:
                    camera.extend_machine_config(config_items)
        except Exception as e:
            logger.debug(f"Could not extend machine config from cameras: {e}")

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
