import re
import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
from seaconfig import AcquisitionChannelConfig

from seafront.config.basics import ConfigItem
from seafront.hardware.limits import HardwareLimitValue
from seafront.logger import logger




_MONO_FORMAT = re.compile(r"^mono(\d+)$")


def bits_for_pixel_format(pixel_format: str) -> int:
    """
    How many bits of the container a camera in this mode actually fills.

    The number every adapter needs and nobody should spell out twice: it decides the array
    dtype, whether the samples get padded up to the container, and what goes in the OME-XML
    as SignificantBits.

    Read out of the name rather than looked up in a table of the five the drivers happen to
    enumerate today (8/10/12/14/16, from the GenICam mono set and ToupCam's RAW flags). The
    depth is the camera's to choose and any step between 8 and 16 is a coherent thing for one
    to offer; a table would turn a camera the drivers grew support for into a crash here.
    """
    match = _MONO_FORMAT.match(pixel_format.lower())
    if match is None:
        raise ValueError(f"unsupported pixel format {pixel_format!r}: expected 'mono<bits>'")
    bits = int(match.group(1))
    if not 1 <= bits <= 16:
        raise ValueError(f"pixel format {pixel_format!r} claims {bits} bits, which fits no container")
    return bits


class AcquisitionMode(str, Enum):
    """
    set acquisition mode of microscope

    modes:
        ON_TRIGGER - default mode, which is expected to be most reactive -> lowest latency, lowest throughput (40-50ms overhead)

        CONTINUOUS - highest latency, highest throughput (expect 500ms latency before first image, and 500ms after last image before return)
    """

    ON_TRIGGER = "on_trigger"
    CONTINUOUS = "continuous"


CameraDriver = tp.Literal["galaxy", "toupcam"]


@dataclass
class CameraOpenRequest:
    """Request structure for opening a camera by USB ID."""

    driver: CameraDriver
    usb_id: str  # USB device ID (serial number for Galaxy, device ID for ToupCam)

class Camera(ABC):
    """
    Abstract interface for camera implementations to support different manufacturers.
    """

    def __init__(self, device_info: tp.Any):
        self.device_info = device_info
        self.vendor_name: str
        self.model_name: str
        self.sn: str
        self.device_type: tp.Literal["main", "autofocus"] | None = None
        self.acquisition_ongoing: bool = False
        self.pixel_format:tp.Literal["mono8","mono10","mono12","mono14","mono16"]="mono8"

    @staticmethod
    @abstractmethod
    def get_all() -> list["Camera"]:
        """Get all available cameras of this type."""
        pass

    @abstractmethod
    def open(self, device_type: tp.Literal["main", "autofocus"]):
        """Open device for interaction."""
        pass

    @abstractmethod
    def close(self):
        """Close device handle."""
        pass

    @abstractmethod
    def snap(self, config: AcquisitionChannelConfig) -> np.ndarray:
        """
        Acquire a single image in trigger mode.

        Args:
            config: Acquisition configuration (exposure time, gain, pixel format, etc.)

        Returns:
            np.ndarray: Image data as numpy array
        """
        pass

    @abstractmethod
    def start_streaming(
        self,
        config: AcquisitionChannelConfig,
        callback: tp.Callable[[np.ndarray], None],
    ) -> None:
        """
        Start continuous image streaming in continuous mode.

        The callback will be called for each acquired image. The callback return value
        is ignored - use stop_streaming() to end streaming.

        Args:
            config: Acquisition configuration (exposure time, gain, pixel format, etc.)
            callback: Function to call with each image data (np.ndarray). Return value is ignored.
        """
        pass

    @abstractmethod
    def stop_streaming(self) -> None:
        """
        Stop continuous image streaming and reset to trigger mode.

        Safe to call even if streaming is not active.
        """
        pass

    @abstractmethod
    def get_exposure_time_limits(self) -> HardwareLimitValue:
        """
        Get camera's exposure time limits.

        Returns:
            HardwareLimitValue with min/max/step values (all in milliseconds)
        """
        pass

    @abstractmethod
    def get_analog_gain_limits(self) -> HardwareLimitValue:
        """
        Get camera's analog gain limits.

        Returns:
            HardwareLimitValue with min/max/step values (all in decibels)
        """
        pass

    @abstractmethod
    def get_supported_pixel_formats(self) -> list[str]:
        """
        Get list of supported monochrome pixel formats.

        Returns:
            List of format strings (e.g., ["mono8", "mono10", "mono12"])
        """
        pass

    @abstractmethod
    def extend_machine_config(self, config_items: list[ConfigItem]) -> None:
        """
        Extend machine configuration with camera-specific options.

        Cameras can modify config_items in-place to add or update options
        based on their hardware capabilities (e.g., pixel formats).

        Args:
            config_items: List of ConfigItem objects to modify in-place
        """
        pass

def get_all_cameras() -> list[Camera]:
    """
    Get all available cameras from all supported manufacturers.

    Returns:
        list[Camera]: List of all available cameras
    """
    all_cameras: list[Camera] = []

    # Get Galaxy cameras
    try:
        from .galaxy_camera import GalaxyCamera

        galaxy_cameras = GalaxyCamera.get_all()
        all_cameras.extend(galaxy_cameras)
    except Exception as e:
        logger.warning(f"Failed to get Galaxy cameras: {e}")

    # Get ToupCam cameras
    try:
        from .toupcam_camera import ToupCamCamera

        toupcam_cameras = ToupCamCamera.get_all()
        all_cameras.extend(toupcam_cameras)
    except Exception as e:
        logger.warning(f"Failed to get ToupCam cameras: {e}")

    return all_cameras


def camera_open(request: CameraOpenRequest) -> Camera:
    """
    Get a camera instance for the specified USB ID (does not open the connection).

    Args:
        request: Camera opening request with driver and USB ID

    Returns:
        Camera: Camera instance ready to be opened

    Raises:
        ValueError: If driver is unsupported
        RuntimeError: If no matching camera is found
    """
    match request.driver:
        case "galaxy":
            return _get_galaxy_camera(request.usb_id)

        case "toupcam":
            return _get_toupcam_camera(request.usb_id)

        case _:
            raise ValueError(f"unsupported driver: {request.driver}")


def _get_galaxy_camera(usb_id: str) -> Camera:
    """Get a Galaxy camera instance with the specified USB ID (unopened)."""
    from .galaxy_camera import GalaxyCamera

    available_cameras = GalaxyCamera.get_all()

    if not available_cameras:
        raise RuntimeError("no galaxy cameras found")

    # Find camera matching the USB ID (serial number)
    for camera in available_cameras:
        if camera.sn == usb_id:
            return camera

    available_sns = [cam.sn for cam in available_cameras]
    raise RuntimeError(
        f"galaxy camera with usb_id='{usb_id}' not found. available: {available_sns}"
    )


def _get_toupcam_camera(usb_id: str) -> Camera:
    """Get a ToupCam camera instance with the specified USB ID (unopened)."""
    from .toupcam_camera import ToupCamCamera

    available_cameras = ToupCamCamera.get_all()

    if not available_cameras:
        raise RuntimeError("no toupcam cameras found")

    # Find camera matching the USB ID
    for camera in available_cameras:
        if camera.sn == usb_id:
            return camera

    available_ids = [cam.sn for cam in available_cameras]
    raise RuntimeError(
        f"toupcam camera with usb_id='{usb_id}' not found. available: {available_ids}"
    )
