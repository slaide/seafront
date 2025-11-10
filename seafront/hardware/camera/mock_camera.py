"""Mock camera implementation for testing and development."""

import threading
import time
import typing as tp

import numpy as np
from seaconfig import AcquisitionChannelConfig

from seafront.config.basics import ConfigItem, ConfigItemOption, set_config_item_bool
from seafront.config.handles import CameraConfig, LaserAutofocusConfig
from seafront.hardware.camera import AcquisitionMode, Camera, HardwareLimitValue
from seafront.logger import logger


class MockCamera(Camera):
    """Mock camera implementation for testing without hardware."""

    def __init__(self):
        """Initialize mock camera."""
        # Create a dummy device info object
        class MockDeviceInfo:
            pass

        super().__init__(MockDeviceInfo())

        self.vendor_name = "Mock"
        self.model_name = "MockCamera"
        self.sn = "MOCK-001"

        self.width = 1920
        self.height = 1080
        self._streaming_active = False
        self._streaming_thread: threading.Thread | None = None
        self._streaming_stop_event = threading.Event()
        self._frame_counter = 0

    @staticmethod
    def get_all() -> list["Camera"]:
        """Get all available mock cameras."""
        return [MockCamera()]

    def open(self, device_type: tp.Literal["main", "autofocus"]):
        """Open device for interaction (mock: no-op)."""
        self.device_type = device_type
        self.acquisition_ongoing = False
        logger.debug(f"mock camera - opened as {device_type}")

    def close(self):
        """Close device handle (mock: no-op)."""
        if self._streaming_active:
            self.stop_streaming()
        logger.debug("mock camera - closed")

    def snap(self, config: AcquisitionChannelConfig) -> np.ndarray:
        """
        Acquire a single image in trigger mode.

        Args:
            config: Acquisition configuration (exposure time, gain, pixel format, etc.)

        Returns:
            np.ndarray: Synthetic image data as numpy array
        """
        if not self.device_type:
            raise RuntimeError("Camera not opened")

        if self.acquisition_ongoing:
            raise RuntimeError("acquisition already in progress")

        self.acquisition_ongoing = True
        try:
            # Generate synthetic image based on config
            img = self._generate_synthetic_image(
                config.exposure_time_ms,
                config.analog_gain,
                self._frame_counter,
                channel_handle=config.handle,
            )
            self._frame_counter += 1
            return img
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
        if not self.device_type:
            raise RuntimeError("Camera not opened")

        if self._streaming_active:
            logger.warning("mock camera - streaming already active")
            return

        self._streaming_active = True
        self._streaming_stop_event.clear()

        # Start streaming thread
        self._streaming_thread = threading.Thread(
            target=self._streaming_loop,
            args=(config, callback),
            daemon=True,
        )
        self._streaming_thread.start()
        logger.debug("mock camera - streaming started")

    def stop_streaming(self) -> None:
        """Stop continuous image streaming."""
        if not self._streaming_active:
            return

        self._streaming_active = False
        self._streaming_stop_event.set()

        # Wait for streaming thread to finish
        if self._streaming_thread is not None:
            self._streaming_thread.join(timeout=2.0)
            self._streaming_thread = None

        logger.debug("mock camera - streaming stopped")

    def get_exposure_time_limits(self) -> HardwareLimitValue:
        """Get camera's exposure time limits in milliseconds."""
        return HardwareLimitValue(
            min=0.1,  # 0.1ms
            max=1000.0,  # 1000ms
            step=0.1,
        )

    def get_analog_gain_limits(self) -> HardwareLimitValue:
        """Get camera's analog gain limits in decibels."""
        return HardwareLimitValue(
            min=0.0,  # 0 dB
            max=20.0,  # 20 dB
            step=0.1,
        )

    def get_supported_pixel_formats(self) -> list[str]:
        """Get list of supported monochrome pixel formats."""
        return ["mono8", "mono16"]

    def extend_machine_config(self, config_items: list[ConfigItem]) -> None:
        """
        Extend machine configuration with camera-specific pixel format options.

        Args:
            config_items: List of ConfigItem objects to modify in-place
        """
        if not self.device_type:
            return

        supported_formats = self.get_supported_pixel_formats()

        # Determine the config key based on device type
        match self.device_type:
            case "main":
                config_key = CameraConfig.MAIN_PIXEL_FORMAT.value
            case "autofocus":
                config_key = LaserAutofocusConfig.CAMERA_PIXEL_FORMAT.value
            case _:
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

        # Set camera-specific image flip defaults for main mock cameras only
        if self.device_type == "main":
            set_config_item_bool(config_items, CameraConfig.MAIN_IMAGE_FLIP_VERTICAL.value, False, frozen=True)
            set_config_item_bool(config_items, CameraConfig.MAIN_IMAGE_FLIP_HORIZONTAL.value, False, frozen=True)

    def _streaming_loop(
        self,
        config: AcquisitionChannelConfig,
        callback: tp.Callable[[np.ndarray], None],
    ) -> None:
        """Background thread that continuously generates streaming images."""
        try:
            while self._streaming_active:
                # Generate synthetic image
                img = self._generate_synthetic_image(
                    config.exposure_time_ms,
                    config.analog_gain,
                    self._frame_counter,
                    channel_handle=config.handle,
                )
                self._frame_counter += 1

                # Call callback with image
                try:
                    callback(img)
                except Exception as e:
                    logger.error(f"mock camera - streaming callback error: {e}")
                    break

                # Simulate frame acquisition time (exposure + small overhead)
                frame_time = config.exposure_time_ms / 1000.0 + 0.01  # 10ms overhead
                self._streaming_stop_event.wait(timeout=frame_time)

        except Exception as e:
            logger.error(f"mock camera - streaming thread error: {e}")
        finally:
            self._streaming_active = False

    def _generate_synthetic_image(
        self,
        exposure_time_ms: float = 50.0,
        analog_gain: float = 1.0,
        frame_number: int = 0,
        channel_handle: str = "generic",
    ) -> np.ndarray:
        """
        Generate a synthetic image for testing.

        Generates channel-specific patterns:
        - Brightfield (handles starting with 'bfled'): cellular-like structures
        - Fluorescence (other): spot-like blinking structures

        Args:
            exposure_time_ms: Exposure time in milliseconds
            analog_gain: Analog gain in decibels
            frame_number: Frame number for variation
            channel_handle: Channel identifier to determine pattern type

        Returns:
            np.ndarray: Synthetic image as uint16 array (1024x1024 for compatibility)
        """
        # Use fixed dimensions for consistency with MockMicroscope behavior
        height, width = 1024, 1024

        # Use frame number to create variation between frames
        np.random.seed((frame_number * 73 + hash(channel_handle)) % 2**31)

        # Add time-based noise variation
        time_factor = np.sin(frame_number * 0.1) * 0.2 + 1.0  # Varies between 0.8 and 1.2

        # Calculate exposure and gain factors
        # Exposure: linear relationship, normalized to 50ms baseline
        exposure_factor = exposure_time_ms / 50.0

        # Analog gain: convert from dB to linear scale
        # 0 dB = 1x, 10 dB = 10x, 20 dB = 100x
        gain_factor = 10.0 ** (analog_gain / 10.0)

        # Create different patterns based on channel type
        if channel_handle.startswith("bfled"):
            # Brightfield: create cellular-like structures with variation
            base_intensity = int(2000 * time_factor * exposure_factor)
            base = np.random.normal(base_intensity, 500, (height, width)).astype(np.uint16)

            # Add some circular "cells" that slightly move/change
            y, x = np.ogrid[:height, :width]
            num_cells = 20 + int(5 * np.sin(frame_number * 0.05))  # Cell count varies
            for i in range(num_cells):
                # Cells have slight movement over time
                base_x = 100 + (i * 100) % (width - 100)
                base_y = 100 + ((i * 73) % (height - 100))
                drift_x = 10 * np.sin(frame_number * 0.02 + i)
                drift_y = 10 * np.cos(frame_number * 0.03 + i)

                center_x = int(base_x + drift_x)
                center_y = int(base_y + drift_y)
                radius = 25 + int(10 * np.sin(frame_number * 0.04 + i))

                if 0 <= center_x < width and 0 <= center_y < height:
                    mask = (x - center_x) ** 2 + (y - center_y) ** 2 < radius**2
                    cell_signal = int(8000 * time_factor * exposure_factor * (0.8 + 0.4 * np.random.random()))
                    base[mask] += np.clip(
                        cell_signal + np.random.normal(0, 500, int(np.sum(mask))),
                        0,
                        50000,
                    ).astype(np.uint16)

        else:
            # Fluorescence: create spot-like structures with blinking/movement
            base_intensity = int(300 * time_factor * exposure_factor)
            base = np.random.normal(base_intensity, 100, (height, width)).astype(np.uint16)

            # Add bright fluorescent spots that blink and move
            y, x = np.ogrid[:height, :width]
            num_spots = 25 + int(10 * np.sin(frame_number * 0.08))
            for i in range(num_spots):
                # Some spots blink on/off
                blink_factor = max(0, np.sin(frame_number * 0.1 + i * 0.5))
                if blink_factor < 0.2:
                    continue

                # More irregular positioning
                base_x = (i * 347 + hash(channel_handle) * 191) % width
                base_y = (i * 431 + hash(channel_handle) * 239) % height
                drift_x = 8 * np.sin(frame_number * 0.03 + i * 0.7)
                drift_y = 8 * np.cos(frame_number * 0.04 + i * 0.7)

                center_x = int(base_x + drift_x) % width
                center_y = int(base_y + drift_y) % height
                radius = 15 + int(10 * np.sin(frame_number * 0.06 + i))

                mask = (x - center_x) ** 2 + (y - center_y) ** 2 < radius**2
                spot_signal = int(
                    12000 * time_factor * exposure_factor * blink_factor * (0.7 + 0.6 * np.random.random())
                )
                base[mask] += np.clip(
                    spot_signal + np.random.normal(0, 1000, int(np.sum(mask))),
                    0,
                    40000,
                ).astype(np.uint16)

        # Apply analog gain to entire image
        final_image = base.astype(np.float32) * gain_factor

        return np.clip(final_image, 0, 65535).astype(np.uint16)
