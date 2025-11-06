import time
import typing as tp
from functools import wraps
import subprocess
import re

import numpy as np
from gxipy import gxiapi
from seaconfig import AcquisitionChannelConfig

from seafront.config.basics import GlobalConfigHandler
from seafront.config.basics import ConfigItem, ConfigItemOption
from seafront.config.handles import CameraConfig, LaserAutofocusConfig
from seafront.hardware.camera import AcquisitionMode, Camera, HardwareLimitValue
from seafront.logger import logger

gxiapi.gx_init_lib()

def _camera_operation_with_reconnect(func):
    """
    Decorator for camera API operations that implements nested retry logic:
    - Inner loop: operation_retry_attempts on the current connection
    - Outer loop: reconnection_attempts with reconnection_delay_ms between

    Retry parameters are fetched from GlobalConfig at runtime.
    On exceptions, attempts to reconnect then retries the operation.
    Only retries on transient errors - persistent failures are immediately re-raised.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Fetch retry configuration from GlobalConfig

        g_config = GlobalConfigHandler.get_dict()

        reconnect_attempts_item = g_config.get("camera.reconnection_attempts")
        reconnect_attempts = 5 if reconnect_attempts_item is None else reconnect_attempts_item.intvalue

        reconnect_delay_ms_item = g_config.get("camera.reconnection_delay_ms")
        reconnect_delay_ms = 1000.0 if reconnect_delay_ms_item is None else reconnect_delay_ms_item.floatvalue

        retry_attempts_item = g_config.get("camera.operation_retry_attempts")
        retry_attempts = 5 if retry_attempts_item is None else retry_attempts_item.intvalue

        logger.debug(
            f"{func.__name__}: loaded retry config: "
            f"reconnect_attempts={reconnect_attempts}, "
            f"reconnect_delay_ms={reconnect_delay_ms}, "
            f"retry_attempts={retry_attempts}"
        )

        last_error: Exception | None = None

        for reconnect_attempt in range(reconnect_attempts):
            for retry_attempt in range(retry_attempts):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"{func.__name__} failed: "
                        f"reconnect_attempt={reconnect_attempt + 1}/{reconnect_attempts}, "
                        f"retry_attempt={retry_attempt + 1}/{retry_attempts}, "
                        f"error_type={type(e).__name__}, error='{e}'"
                    )
                    if retry_attempt < retry_attempts - 1:
                        # Still have retries left on this connection, just retry
                        continue
                    else:
                        # This connection failed all retries, try reconnection
                        break

            if reconnect_attempt < reconnect_attempts - 1:
                # Try to reconnect for next attempt
                logger.info(
                    f"{func.__name__}: attempting reconnection "
                    f"(attempt {reconnect_attempt + 1}/{reconnect_attempts})"
                )
                try:
                    self.close()
                    self.open(self.device_type)
                    logger.info(
                        f"{func.__name__}: successfully reconnected, retrying operation"
                    )
                except Exception as reconnect_error:
                    logger.warning(
                        f"{func.__name__}: reconnection failed: "
                        f"{type(reconnect_error).__name__}: {reconnect_error}"
                    )
                # Wait before next reconnection attempt
                time.sleep(reconnect_delay_ms / 1000.0)

        # All retries and reconnection attempts failed
        if last_error:
            logger.error(
                f"{func.__name__}: all retry/reconnection attempts exhausted, re-raising exception"
            )
            raise last_error

    return wrapper


class GalaxyCamera(Camera):
    """Galaxy Camera implementation using the gxiapi library."""

    device_manager = gxiapi.DeviceManager()
    _usb_enum_cached: bool = False
    _cached_usb_devices: list[gxiapi.GxDeviceBaseInfo] = []

    @staticmethod
    def _enumerate_via_usb() -> list[gxiapi.GxDeviceBaseInfo]:
        """
        Enumerate Galaxy cameras using raw USB device lookup (lsusb).
        This avoids calling gxiapi enumeration methods which reset USB bus.
        Results are cached to avoid repeated lsusb calls which can fail due to timing issues.

        Returns:
            List of device info dicts with vendor_name, model_name, sn, etc.
        """

        # Return cached results if available
        if GalaxyCamera._usb_enum_cached:
            return GalaxyCamera._cached_usb_devices

        try:
            result = subprocess.run(['lsusb', '-v'], capture_output=True, text=True, timeout=15)
            devices = []
            current_device = None

            for line in result.stdout.split('\n'):
                # Match Galaxy camera VID:PID (2ba2:4d55)
                bus_match = re.match(r'Bus \d+ Device \d+: ID 2ba2:4d55\s+(.*)', line)
                if bus_match:
                    current_device = {
                        'vendor_name': None,
                        'model_name': None,
                        'sn': None
                    }
                elif current_device:
                    if 'iManufacturer' in line:
                        match = re.search(r'iManufacturer\s+\d+\s+(.*)', line)
                        if match:
                            current_device['vendor_name'] = match.group(1).strip()  # type: ignore[index]
                    elif 'iProduct' in line:
                        match = re.search(r'iProduct\s+\d+\s+(.*)', line)
                        if match:
                            current_device['model_name'] = match.group(1).strip()  # type: ignore[index]
                    elif 'iSerial' in line:
                        match = re.search(r'iSerial\s+\d+\s+(.*)', line)
                        if match:
                            current_device['sn'] = match.group(1).strip()  # type: ignore[index]
                            devices.append(current_device)
                            current_device = None

            # Cache the results
            GalaxyCamera._cached_usb_devices = devices
            GalaxyCamera._usb_enum_cached = True
            return devices
        except Exception as e:
            logger.warning(f"Failed to enumerate Galaxy cameras via USB: {e}")
            # Cache empty list on error too to prevent retries
            GalaxyCamera._cached_usb_devices = []
            GalaxyCamera._usb_enum_cached = True
            return []

    @staticmethod
    def get_all() -> list["Camera"]:
        """
        Get all available Galaxy cameras by querying USB devices.
        Prefers USB enumeration to avoid USB bus resets.
        Falls back to gxiapi enumeration if USB enumeration returns no devices.
        """
        devices_from_usb = GalaxyCamera._enumerate_via_usb()

        # Fallback to gxiapi if USB enumeration returns nothing
        # we do not do this by default, because:
        # updating the device list via that function resets all usb connections on the computer,
        # and reconnection of the cameras may take longer than the function waits for, hence
        # missing cameras that are actually connected!
        # the default _enumerate_via_usb uses the existing usb query infrastructure of linux instead.
        # (this may miss devices in certain cases, hence falling back to update_device_list if none are found)
        if not devices_from_usb:
            logger.warning("USB enumeration returned no devices, falling back to gxiapi enumeration")
            try:
                dev_num, dev_info_list = GalaxyCamera.device_manager.update_device_list()
                if dev_num > 0 and dev_info_list:
                    print(f"found {len(dev_info_list)} devices via gxiapi fallback")
                    devices_from_usb = dev_info_list
                else:
                    print("found 0 devices via gxiapi fallback")
            except Exception as e:
                logger.error(f"gxiapi fallback enumeration failed: {e}")
                print("found 0 devices (USB and gxiapi both failed)")
        else:
            print(f"found {len(devices_from_usb)} devices via USB")

        ret: list[Camera] = []

        for idx, dev_info in enumerate(devices_from_usb, 1):
            # Build minimal device_info compatible with GalaxyCamera.__init__
            # Add index for compatibility with gxiapi structures
            dev_info['index'] = idx
            dev_info['display_name'] = f"{dev_info['model_name']}({dev_info['sn']})"
            dev_info['device_id'] = f"{dev_info['model_name']}({dev_info['sn']})"
            dev_info['user_id'] = ''
            dev_info['access_status'] = 0
            dev_info['device_class'] = 3  # USB3.0
            # Fill in remaining fields for compatibility
            for field in ['mac', 'ip', 'subnet_mask', 'gateway', 'nic_mac', 'nic_ip', 'nic_subnet_mask', 'nic_gateWay', 'nic_description']:
                dev_info[field] = ''

            cam = GalaxyCamera(dev_info)
            ret.append(cam)

        return ret

    def __init__(self, device_info: gxiapi.GxDeviceBaseInfo):
        super().__init__(device_info)

        self.vendor_name: str = device_info["vendor_name"]
        self.model_name: str = device_info["model_name"]
        self.sn: str = device_info["sn"]

        self.handle = None

    def open(self, device_type: tp.Literal["main", "autofocus"]):
        """
        open device for interaction
        """

        for i in range(5):
            try:
                self.handle = GalaxyCamera.device_manager.open_device_by_sn(self.sn)
            except Exception as e:
                logger.debug(f"camera - opening failed {i+1} times {e=}")

                try:
                    self.close()
                except Exception as e:
                    logger.warning(f"camera - opening failed with error {e=}. retrying. ")

                time.sleep(2)
                logger.debug("camera - done sleeping to await camera response on open")
                continue

            logger.debug("opened camera")
            break

        if self.handle is None:
            logger.critical(f"failed to open camera {self.device_info}")
            raise RuntimeError(f"failed to open camera {self.device_info}")

        self.device_type = device_type

        self.acquisition_ongoing = False

        # prints some camera hardware information
        if False:
            print(" - pixel formats:")
            pixel_formats = self.handle.PixelFormat.get_range()
            for pf in pixel_formats:
                print(f"   - {pf}")

            widths = self.handle.Width.get_range()
            print(" - width:")
            print(f"   - min {widths['min']}")
            print(f"   - max {widths['max']}")
            print(f"   - inc {widths['inc']}")

            heights = self.handle.Height.get_range()
            print(" - height:")
            print(f"   - min {heights['min']}")
            print(f"   - max {heights['max']}")
            print(f"   - inc {heights['inc']}")

            if self.handle.ExposureTime.is_readable():
                for e in self.handle.ExposureTime.get_range().items():
                    print(f" - exposure time {e}")

            if self.handle.Gain.is_readable():
                for g in self.handle.Gain.get_range().items():
                    print(f" - gain {g}")

            if self.handle.BinningHorizontal.is_readable():
                for bh in self.handle.BinningHorizontal.get_range().items():
                    print(f" - binning horizontal {bh}")

            if self.handle.BinningVertical.is_readable():
                for bv in self.handle.BinningVertical.get_range().items():
                    print(f" - binning vertical {bv}")

            if self.handle.TriggerSource.is_readable():
                for ts in self.handle.TriggerSource.get_range().items():
                    print(f" - trigger source {ts}")

        # notable features:
        #  - AcquisitionMode: both cameras support continuous and single frame mode. both
        #                     modes are broadly similar (stream on, take image, stream off),
        #                     but single frame mode only allows taking a single image with a
        #                     running stream. continuous allows any number. there is no clear
        #                     downside to continuous mode. latency to stream on/off is the same.
        #  - TriggerMode: can be ON/OFF, i.e. trigger needs to be sent to trigger image acquisition
        #  - TriggerActivation: rising/falling edge - unused in this code

        # set software trigger
        # self.camera.TriggerMode.set(gx.GxSwitchEntry.ON)
        # self.camera.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)

        # set hardware trigger
        # self.camera.TriggerMode.set(gx.GxSwitchEntry.ON)
        # self.camera.TriggerSource.set(gx.GxTriggerSourceEntry.LINE2)
        # self.camera.TriggerSource.set(gx.GxTriggerActivationEntry.RISING_EDGE) # this was commented out by hongquan, but why?

        # these features seem to have no effect, or I cannot find a way to use them properly
        #   - AcquisitionFrameRateMode
        #   - AcquisitionFrameRate

        # the following features only have one valid value for either camera:
        #   - ExposureMode = Timed
        #   - GainSelector = AnalogAll

        # the following features are not supported by either camera:
        #   - ExposureDelay
        #   - ExposureTimeMode

        # turn off device link throughput limit
        self.handle.DeviceLinkThroughputLimitMode.set(gxiapi.GxSwitchEntry.OFF)

        # set acquisition mode
        self.handle.AcquisitionMode.set(gxiapi.GxAcquisitionModeEntry.CONTINUOUS)

        # set pixel format
        self.handle.PixelFormat.set(gxiapi.GxPixelFormatEntry.MONO8)
        self.pixel_format = "mono8"

        self.is_streaming = False
        self.acq_mode = None
        self._streaming_active = False
        self._set_acquisition_mode(AcquisitionMode.ON_TRIGGER)

        # turning stream on takes 25ms for continuous mode, 20ms for single frame mode
        self.is_streaming = True
        self._api_stream_on()

    @_camera_operation_with_reconnect
    def _api_trigger(self) -> None:
        """
        Send software trigger to camera to acquire next frame.
        Decorated with nested retry logic for connection resilience.

        Raises:
            Exception: if trigger fails after all retry/reconnection attempts
        """
        if self.handle is None:
            raise RuntimeError("Camera handle is None - not connected")
        self.handle.TriggerSoftware.send_command()

    @_camera_operation_with_reconnect
    def _api_get_image(self) -> gxiapi.RawImage | None:
        """
        Retrieve image from camera after trigger.
        Decorated with nested retry logic for connection resilience.

        Returns:
            RawImage object or None if no image available

        Raises:
            Exception: if retrieval fails after all retry/reconnection attempts
        """
        if self.handle is None:
            raise RuntimeError("Camera handle is None - not connected")
        return self.handle.data_stream[0].get_image()

    @_camera_operation_with_reconnect
    def _api_stream_on(self) -> None:
        """
        Enable camera streaming.
        Decorated with nested retry logic for connection resilience.

        Raises:
            Exception: if operation fails after all retry/reconnection attempts
        """
        if self.handle is None:
            raise RuntimeError("Camera handle is None - not connected")
        self.handle.stream_on()

    @_camera_operation_with_reconnect
    def _api_stream_off(self) -> None:
        """
        Disable camera streaming.
        Decorated with nested retry logic for connection resilience.

        Raises:
            Exception: if operation fails after all retry/reconnection attempts
        """
        if self.handle is None:
            raise RuntimeError("Camera handle is None - not connected")
        self.handle.stream_off()

    @_camera_operation_with_reconnect
    def _api_set_exposure_time(self, exposure_time_native_unit: float) -> None:
        """
        Set exposure time via API.
        Decorated with nested retry logic for connection resilience.

        Args:
            exposure_time_native_unit: Exposure time in camera native units

        Raises:
            Exception: if operation fails after all retry/reconnection attempts
        """
        if self.handle is None:
            raise RuntimeError("Camera handle is None - not connected")
        self.handle.ExposureTime.set(exposure_time_native_unit)

    @_camera_operation_with_reconnect
    def _api_set_analog_gain(self, gain_db: float) -> None:
        """
        Set analog gain via API.
        Decorated with nested retry logic for connection resilience.

        Args:
            gain_db: Analog gain in decibels

        Raises:
            Exception: if operation fails after all retry/reconnection attempts
        """
        if self.handle is None:
            raise RuntimeError("Camera handle is None - not connected")
        self.handle.Gain.set(gain_db)

    def _clear_acquisition_state(self) -> None:
        """
        Reset acquisition state flags unconditionally.

        Clears both acquisition_ongoing and acq_mode to prepare for the next
        acquisition cycle or mode switch.
        """
        self.acquisition_ongoing = False
        self.acq_mode = None

    def _set_acquisition_mode(
        self,
        acq_mode: AcquisitionMode,

        with_cb: tp.Callable[[gxiapi.RawImage], None] | None = None,
        continuous_target_fps: float | None = None,
    ):
        """
        set acquisition mode
        """
        assert self.handle is not None

        if self.acq_mode == acq_mode:
            return

        # ensure stream is off, and unregister all registered callbacks
        was_streaming = self.is_streaming
        if was_streaming:
            self._api_stream_off()
        self.handle.data_stream[0].unregister_capture_callback()

        match acq_mode:
            case AcquisitionMode.ON_TRIGGER:
                if with_cb is not None:
                    raise RuntimeError("callback is not allowed for on trigger mode")

                # enable trigger
                self.handle.TriggerMode.set(gxiapi.GxSwitchEntry.ON)
                self.handle.TriggerSource.set(gxiapi.GxTriggerSourceEntry.SOFTWARE)

                # disable framerate limit
                self.handle.AcquisitionFrameRateMode.set(gxiapi.GxSwitchEntry.OFF)

            case AcquisitionMode.CONTINUOUS:
                if with_cb is None:
                    raise RuntimeError("callback must be set for continuous mode")

                # disable trigger
                self.handle.TriggerMode.set(gxiapi.GxSwitchEntry.OFF)

                def galaxy_fwd_image(img:gxiapi.RawImage):
                    with_cb(img)

                self.handle.data_stream[0].register_capture_callback(galaxy_fwd_image)

                # Force stream restart when switching from trigger mode to ensure continuous acquisition works
                if self.acq_mode == AcquisitionMode.ON_TRIGGER:
                    was_streaming = True

        # set acquisition mode
        self.handle.AcquisitionMode.set(gxiapi.GxAcquisitionModeEntry.CONTINUOUS)

        # turning stream on takes 25ms for continuous mode, 20ms for single frame mode
        if was_streaming:
            self._api_stream_on()

        self.acq_mode = acq_mode

    def close(self):
        """
        close device handle
        """
        logger.debug("camera - closing")

        if self.handle is None:
            return

        # turning stream off takes 300ms (for continuous and single frame mode)
        self.is_streaming = False

        # this may throw if the device is already offline or just not streaming
        try:
            self._api_stream_off()
        except:
            pass

        try:
            self.handle.close_device()
        except:
            pass

        self.handle = None

        logger.debug("closed camera")

    def _exposure_time_ms_to_native(self, exposure_time_ms: float):
        """
        convert exposure time from ms to the native unit of the camera
        """
        assert self.handle is not None

        exposure_time_native_unit = exposure_time_ms
        exposure_time_range = self.handle.ExposureTime.get_range()
        assert exposure_time_range is not None
        match exposure_time_range["unit"]:
            case "us":
                exposure_time_native_unit *= 1e3
            case "ms":
                pass
            case _:
                raise RuntimeError(f"unhandled unit {exposure_time_range['unit']}")

        return exposure_time_native_unit

    def _set_exposure_time(self, exposure_time_ms: float) -> None:
        """
        Set camera exposure time.

        Args:
            exposure_time_ms: Exposure time in milliseconds

        Takes ~10ms.
        """
        assert self.handle is not None
        exposure_time_native_unit = self._exposure_time_ms_to_native(exposure_time_ms)
        self._api_set_exposure_time(exposure_time_native_unit)

    def _set_analog_gain(self, gain_db: float) -> None:
        """
        Set camera analog gain.

        Args:
            gain_db: Analog gain in decibels

        Takes ~8ms.
        """
        assert self.handle is not None
        self._api_set_analog_gain(gain_db)

    def snap(self, config: AcquisitionChannelConfig) -> np.ndarray:
        """
        Acquire a single image in trigger mode.

        Args:
            config: Acquisition configuration (exposure time, gain, pixel format, etc.)

        Returns:
            np.ndarray: Image data as numpy array
        """
        assert self.handle is not None

        if self.acquisition_ongoing:
            logger.warning(
                "camera - requested acquisition while one was already ongoing. returning None."
            )
            raise RuntimeError("acquisition already in progress")

        # set pixel format (determined from device_type config)
        self._set_pixel_format()

        self._set_exposure_time(config.exposure_time_ms)
        self._set_analog_gain(config.analog_gain)

        self.acquisition_ongoing = True

        try:
            # Ensure camera is in trigger mode before sending software trigger
            self._set_acquisition_mode(AcquisitionMode.ON_TRIGGER)

            # send command to trigger acquisition (with retry logic)
            self._api_trigger()

            # wait for image to arrive (with retry logic)
            img: gxiapi.RawImage | None = self._api_get_image()

            if img is None:
                raise RuntimeError("failed to acquire image")

            match img.get_status():
                case gxiapi.GxFrameStatusList.INCOMPLETE:
                    raise RuntimeError("incomplete frame")
                case gxiapi.GxFrameStatusList.SUCCESS:
                    pass

            np_img = img.get_numpy_array()
            assert np_img is not None
            return np_img.copy()

        finally:
            self.acquisition_ongoing = False

    def start_streaming(
        self,
        config: AcquisitionChannelConfig,
        callback: tp.Callable[[np.ndarray], None],
    ) -> None:
        """
        Start continuous image streaming in continuous mode.

        Args:
            config: Acquisition configuration (exposure time, gain, pixel format, etc.)
            callback: Function to call with each image data (np.ndarray).
        """
        assert self.handle is not None

        if self._streaming_active:
            logger.warning("streaming already active")
            return

        self._streaming_active = True

        # set pixel format (determined from device_type config)
        self._set_pixel_format()

        self._set_exposure_time(config.exposure_time_ms)
        self._set_analog_gain(config.analog_gain)

        def streaming_callback(img: gxiapi.RawImage):
            """Internal callback that ignores return value and checks streaming flag."""
            if not self._streaming_active or img is None:  # type: ignore
                self._clear_acquisition_state()
                return

            img_status = img.get_status()
            match img_status:
                case gxiapi.GxFrameStatusList.INCOMPLETE:
                    raise RuntimeError("incomplete frame")
                case gxiapi.GxFrameStatusList.SUCCESS:
                    pass

            img_np = img.get_numpy_array()
            assert img_np is not None

            # Call user callback and ignore return value
            callback(img_np)

        self._set_acquisition_mode(
            AcquisitionMode.CONTINUOUS,
            with_cb=streaming_callback,
        )

    def stop_streaming(self) -> None:
        """
        Stop continuous image streaming and reset to trigger mode.
        """
        if not self._streaming_active:
            return

        self._streaming_active = False

        # Reset to trigger mode
        try:
            self._set_acquisition_mode(AcquisitionMode.ON_TRIGGER)
        except Exception as e:
            logger.warning(f"error resetting to trigger mode: {e}")

        logger.debug("galaxy camera - streaming stopped")

    def _set_pixel_format(self, pixel_format: str | None = None) -> None:
        """
        Set pixel format.

        Args:
            pixel_format: Pixel format string (e.g., "mono8", "mono10", "mono12").
                         If None, determines format from device_type configuration.

        Handles the pixel format change by stopping and restarting stream if needed.
        """
        assert self.handle is not None

        # If no format provided, determine from device type
        if pixel_format is None:
            match self.device_type:
                case "main":
                    pixel_format_item = CameraConfig.MAIN_PIXEL_FORMAT.value_item
                case "autofocus":
                    pixel_format_item = LaserAutofocusConfig.CAMERA_PIXEL_FORMAT.value_item
                case _:
                    raise RuntimeError(f"unsupported device type {self.device_type}")

            assert pixel_format_item is not None
            pixel_format = pixel_format_item.value
            assert isinstance(pixel_format, str)

        pixel_format_range: dict[str, tp.Any] | None = self.handle.PixelFormat.get_range()
        assert pixel_format_range is not None
        format_is_supported = any(
            k.lower() == pixel_format.lower() for k in pixel_format_range.keys()
        )
        if not format_is_supported:
            raise RuntimeError(f"unsupported pixel format {pixel_format}")

        # pixel format change is not possible while streaming
        # turning the stream off takes nearly half a second, so we cache the current pixel format
        # and only pause streaming to change it, if necessary
        match pixel_format:
            case "mono8":
                if self.pixel_format != "mono8":
                    logger.debug("camera - perf warning - changing pixel format to mono8")
                    self.handle.stream_off()
                    self.pixel_format = "mono8"
                    self.handle.PixelFormat.set(gxiapi.GxPixelFormatEntry.MONO8)
                    self.handle.stream_on()
            case "mono10":
                if self.pixel_format != "mono10":
                    logger.debug("camera - perf warning - changing pixel format to mono10")
                    self.handle.stream_off()
                    self.pixel_format = "mono10"
                    self.handle.PixelFormat.set(gxiapi.GxPixelFormatEntry.MONO10)
                    self.handle.stream_on()
            case "mono12":
                if self.pixel_format != "mono12":
                    logger.debug("camera - perf warning - changing pixel format to mono12")
                    self.handle.stream_off()
                    self.pixel_format = "mono12"
                    self.handle.PixelFormat.set(gxiapi.GxPixelFormatEntry.MONO12)
                    self.handle.stream_on()
            case _:
                raise RuntimeError(f"unsupported pixel format {pixel_format}")

    def get_exposure_time_limits(self) -> HardwareLimitValue:
        """
        Get camera's exposure time limits from Galaxy camera hardware.
        
        Returns:
            HardwareLimitValue with min/max/step values (all in milliseconds)
        """
        if not self.handle:
            raise RuntimeError("Camera not opened")

        # Get exposure time range from camera hardware
        exposure_range = self.handle.ExposureTime.get_range()

        # Convert from microseconds to milliseconds and return
        # Type: ignore needed because gxiapi TypedDict doesn't expose these keys to type checker
        return HardwareLimitValue(
            min=exposure_range['min'] / 1000.0,  # type: ignore[typeddict-item]
            max=exposure_range['max'] / 1000.0,  # type: ignore[typeddict-item]
            step=exposure_range['inc'] / 1000.0  # type: ignore[typeddict-item]
        )

    def get_analog_gain_limits(self) -> HardwareLimitValue:
        """
        Get camera's analog gain limits from Galaxy camera hardware.
        
        Returns:
            HardwareLimitValue with min/max/step values (all in decibels)
        """
        if not self.handle:
            raise RuntimeError("Camera not opened")

        # Get analog gain range from camera hardware
        gain_range = self.handle.Gain.get_range()

        # Galaxy cameras typically report gain in dB already
        # Type: ignore needed because gxiapi TypedDict doesn't expose these keys to type checker
        return HardwareLimitValue(
            min=gain_range['min'],  # type: ignore[typeddict-item]
            max=gain_range['max'],  # type: ignore[typeddict-item]
            step=gain_range['inc']  # type: ignore[typeddict-item]
        )

    def get_supported_pixel_formats(self) -> list[str]:
        """
        Get list of supported monochrome pixel formats from Galaxy camera.

        Returns:
            List of format strings (e.g., ["mono8", "mono10", "mono12"])
        """
        if not self.handle:
            raise RuntimeError("Camera not opened")

        pixel_format_range = self.handle.PixelFormat.get_range()

        # Filter for monochrome formats only
        supported: list[str] = []
        mono_formats = ["mono8", "mono10", "mono12", "mono14", "mono16"]
        for fmt in mono_formats:
            if any(k.lower() == fmt for k in pixel_format_range.keys()):
                supported.append(fmt)

        return supported

    def extend_machine_config(self, config_items: list[ConfigItem]) -> None:
        """
        Extend machine configuration with camera-specific pixel format options.

        Updates the pixel format options for the main or autofocus camera based
        on actual hardware capabilities.

        Args:
            config_items: List of ConfigItem objects to modify in-place
        """
        if not self.handle or not self.device_type:
            return

        supported_formats = self.get_supported_pixel_formats()

        # Determine the config key based on device type
        match self.device_type:
            case "main":
                config_key = "camera.main.pixel_format"
            case "autofocus":
                config_key = "camera.autofocus.pixel_format"
            case _:
                # Should not happen due to early return, but handle defensively
                return

        # Build the new options list
        new_options = [
            ConfigItemOption(name=fmt.capitalize(), handle=fmt)
            for fmt in sorted(supported_formats)
        ]

        # Find and update existing item, or create if missing
        found = False
        for item in config_items:
            if item.handle == config_key:
                item.options = new_options
                # Ensure current value is valid
                if item.value not in supported_formats:
                    item.value = supported_formats[0]
                found = True
                break

        # If not found, create a new config item
        if not found:
            new_item = ConfigItem(
                handle=config_key,
                name="Pixel Format",
                value=supported_formats[0],
                value_kind="option",
                options=new_options,
            )
            config_items.append(new_item)
