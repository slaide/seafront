import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
from seaconfig import AcquisitionChannelConfig

from seafront.logger import logger


@dataclass(frozen=True)
class HardwareLimitValue:
    """
    Represents a hardware limit with min/max/step values.
    Mirrors the TypeScript HardwareLimitValue type.
    """
    min: float | int
    max: float | int
    step: float | int

    def to_dict(self) -> dict[str, float | int]:
        """Convert to dictionary format for API responses."""
        return {"min": self.min, "max": self.max, "step": self.step}


class AcquisitionMode(str, Enum):
    """
    set acquisition mode of microscope

    modes:
        ON_TRIGGER - default mode, which is expected to be most reactive -> lowest latency, lowest throughput (40-50ms overhead)

        CONTINUOUS - highest latency, highest throughput (expect 500ms latency before first image, and 500ms after last image before return)
    """

    ON_TRIGGER = "on_trigger"
    CONTINUOUS = "continuous"


@dataclass
class GalaxyCameraIdentifier:
    """Galaxy camera identification fields."""

    sn: str  # Serial number - primary identifier for Galaxy cameras
    vendor_name: str | None = None  # Optional vendor name filter
    model_name: str | None = None  # Optional model name filter


@dataclass
class ToupCamIdentifier:
    """ToupCam camera identification fields (placeholder for future implementation)."""

    id: str  # ToupCam uses ID strings
    # Add other ToupCam-specific fields as needed when implementing

@dataclass
class CameraOpenRequest:
    """Request structure for opening a camera."""

    driver: tp.Literal["galaxy", "toupcam"]
    galaxy: GalaxyCameraIdentifier | None = None
    toupcam: ToupCamIdentifier | None = None

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
    def acquire_with_config(
        self,
        config: AcquisitionChannelConfig,
        mode: tp.Literal["once", "until_stop"] = "once",
        callback: tp.Callable[[np.ndarray], bool] | None = None,
    ) -> np.ndarray | None:
        """
        Acquire image with given configuration.

        Args:
            config: Acquisition configuration (exposure time, gain, etc.)
            mode: "once" for single image, "until_stop" for continuous until callback returns True
            callback: Callback function for continuous mode

        Returns:
            np.ndarray of image data if mode is "once", None if mode is "until_stop"
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
    def set_acquisition_mode_trigger(
        self,
    )->None:
        """
        set acquisition to trigger mode.
        """
        pass

    @abstractmethod
    def stop_acquisition(self) -> None:
        """
        Force stop any ongoing acquisition.

        This method should interrupt continuous acquisition mode safely.
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
    Get a camera instance for the specified request (does not open the connection).

    Args:
        request: Camera opening request with driver and vendor-specific identifiers

    Returns:
        Camera: Camera instance ready to be opened

    Raises:
        ValueError: If driver is unsupported or required identifiers are missing
        RuntimeError: If no matching camera is found
    """
    match request.driver:
        case "galaxy":
            if request.galaxy is None:
                raise ValueError("galaxy identifier required for galaxy driver")

            return _get_galaxy_camera(request.galaxy)

        case "toupcam":
            if request.toupcam is None:
                raise ValueError("toupcam identifier required for toupcam driver")

            return _get_toupcam_camera(request.toupcam)

        case _:
            raise ValueError(f"unsupported driver: {request.driver}")


def _get_galaxy_camera(identifier: GalaxyCameraIdentifier) -> Camera:
    """Get a Galaxy camera instance with the specified identifier (unopened)."""
    from .galaxy_camera import GalaxyCamera

    available_cameras = GalaxyCamera.get_all()

    if not available_cameras:
        raise RuntimeError("no galaxy cameras found")

    # Find camera matching the identifier
    matching_camera = None
    for camera in available_cameras:
        if camera.sn == identifier.sn:
            # Check optional filters if provided
            if identifier.vendor_name is not None and camera.vendor_name != identifier.vendor_name:
                continue
            if identifier.model_name is not None and camera.model_name != identifier.model_name:
                continue
            matching_camera = camera
            break

    if matching_camera is None:
        available_sns = [cam.sn for cam in available_cameras]
        raise RuntimeError(
            f"galaxy camera with sn='{identifier.sn}' not found. available cameras: {available_sns}"
        )

    # Return the unopened camera
    return matching_camera


def _get_toupcam_camera(identifier: ToupCamIdentifier) -> Camera:
    """Get a ToupCam camera instance with the specified identifier (unopened)."""
    from .toupcam_camera import ToupCamCamera

    available_cameras = ToupCamCamera.get_all()

    if not available_cameras:
        raise RuntimeError("no toupcam cameras found")

    # Find camera matching the identifier
    matching_camera = None
    for camera in available_cameras:
        if camera.sn == identifier.id:  # ToupCam uses ID strings as serial numbers
            matching_camera = camera
            break

    if matching_camera is None:
        available_ids = [cam.sn for cam in available_cameras]
        raise RuntimeError(
            f"toupcam camera with id='{identifier.id}' not found. "
            f"available cameras: {available_ids}"
        )

    # Return the unopened camera
    return matching_camera
