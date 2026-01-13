"""
Abstract Microcontroller interface and dispatch mechanism.

This module defines the abstract interface that all microcontroller drivers must implement,
as well as the dispatch mechanism for selecting the appropriate driver based on configuration.
"""

import typing as tp
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from seafront.hardware.adapter import Position
from seafront.logger import logger


# Type alias for the driver literal - extend as new drivers are added
MicrocontrollerDriver = tp.Literal["teensy"]


@dataclass
class MicrocontrollerOpenRequest:
    """Request structure for opening a microcontroller by driver and USB ID."""

    driver: MicrocontrollerDriver
    usb_id: str  # USB serial number


class Microcontroller(ABC):
    """
    Abstract interface for microcontroller implementations to support different manufacturers.

    This base class defines the contract that all microcontroller drivers must implement.
    Each driver handles its own protocol details (commands, packet format, etc.) internally.
    """

    # Device info (abstract properties to be implemented by drivers)

    @property
    @abstractmethod
    def vendor_name(self) -> str:
        """Vendor/manufacturer name."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model name of the device."""
        pass

    @property
    @abstractmethod
    def sn(self) -> str:
        """USB serial number."""
        pass

    # Connection management

    @abstractmethod
    def open(self) -> None:
        """Open connection to the device."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close connection to the device."""
        pass

    @contextmanager
    @abstractmethod
    def locked(self, blocking: bool = True) -> tp.Iterator[tp.Self | None]:
        """
        Context manager for exclusive access to the microcontroller.

        Args:
            blocking: If True, wait for lock. If False, return None if lock unavailable.

        Yields:
            Self if lock acquired, None if non-blocking and lock unavailable.
        """
        pass

    # Hardware initialization (driver-specific)

    @abstractmethod
    async def reset(self) -> None:
        """Reset the microcontroller."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the microcontroller (motor drivers, DAC, etc.)."""
        pass

    @abstractmethod
    async def configure_actuators(self) -> None:
        """Configure motor drivers, leadscrew pitch, velocity/acceleration limits."""
        pass

    # Stage control

    @abstractmethod
    async def home(self, axis: tp.Literal["x", "y", "z"]) -> None:
        """
        Home the specified axis.

        Args:
            axis: The axis to home ("x", "y", or "z")
        """
        pass

    @abstractmethod
    async def move_to_mm(self, axis: tp.Literal["x", "y", "z"], position_mm: float) -> None:
        """
        Move to an absolute position on the specified axis.

        Args:
            axis: The axis to move
            position_mm: Target position in millimeters
        """
        pass

    @abstractmethod
    async def move_by_mm(self, axis: tp.Literal["x", "y", "z"], distance_mm: float) -> None:
        """
        Move by a relative distance on the specified axis.

        Args:
            axis: The axis to move
            distance_mm: Distance to move in millimeters (positive or negative)
        """
        pass

    @abstractmethod
    async def set_zero(self, axis: tp.Literal["x", "y", "z"]) -> None:
        """
        Set the current position as zero for the specified axis.

        Args:
            axis: The axis to zero
        """
        pass

    @abstractmethod
    async def set_limit_mm(
        self,
        axis: tp.Literal["x", "y", "z"],
        coord: float,
        bound: tp.Literal["upper", "lower"],
    ) -> None:
        """
        Set a software limit on the specified axis.

        Args:
            axis: The axis to set limit for
            coord: Position of the limit in millimeters
            bound: Whether this is an "upper" or "lower" limit
        """
        pass

    @abstractmethod
    async def get_position(self) -> Position:
        """
        Get the current stage position.

        Returns:
            Position object with x, y, z coordinates in millimeters
        """
        pass

    @abstractmethod
    async def get_last_position(self) -> Position:
        """
        Get the last known stage position from cached state.

        Returns:
            Position object with x, y, z coordinates in millimeters
        """
        pass

    # Illumination control

    @abstractmethod
    async def illumination_begin(
        self,
        source: int,
        intensity_percent: float,
        rgb: tuple[float, float, float] | None = None,
    ) -> None:
        """
        Turn on illumination.

        Args:
            source: Illumination source code (hardware-specific)
            intensity_percent: Intensity as percentage (0-100)
            rgb: Optional RGB color tuple for LED matrix (each 0-1)
        """
        pass

    @abstractmethod
    async def illumination_end(self, source: int | None = None) -> None:
        """
        Turn off illumination.

        Args:
            source: Optional source to specifically clear, or None for all
        """
        pass

    # Autofocus laser

    @abstractmethod
    async def af_laser_on(self) -> None:
        """Turn on the autofocus laser."""
        pass

    @abstractmethod
    async def af_laser_off(self) -> None:
        """Turn off the autofocus laser."""
        pass

    # Filter wheel

    @abstractmethod
    async def filter_wheel_init(self) -> None:
        """Initialize the filter wheel hardware."""
        pass

    @abstractmethod
    async def filter_wheel_configure_actuator(self) -> None:
        """Configure filter wheel motor parameters."""
        pass

    @abstractmethod
    async def filter_wheel_home(self) -> None:
        """Home the filter wheel to establish reference position."""
        pass

    @abstractmethod
    async def filter_wheel_set_position(self, position: int) -> None:
        """
        Set the filter wheel to a specific position.

        Args:
            position: Target position index
        """
        pass

    @abstractmethod
    def filter_wheel_get_position(self) -> int:
        """
        Get the current filter wheel position.

        Returns:
            Current position index
        """
        pass

    # Device discovery (static methods)

    @staticmethod
    @abstractmethod
    def get_all() -> Sequence["Microcontroller"]:
        """
        Get all available microcontrollers of this type.

        Returns:
            Sequence of Microcontroller instances for discovered devices
        """
        pass


def get_all_microcontrollers() -> list[Microcontroller]:
    """
    Get all available microcontrollers from all supported manufacturers.

    Returns:
        List of all discovered microcontrollers across all drivers
    """
    all_mcs: list[Microcontroller] = []

    # Get Teensy microcontrollers
    try:
        from .teensy_microcontroller import TeensyMicrocontroller

        teensy_mcs = TeensyMicrocontroller.get_all()
        all_mcs.extend(teensy_mcs)
    except Exception as e:
        logger.warning(f"Failed to get Teensy microcontrollers: {e}")

    return all_mcs


def _get_teensy_microcontroller(usb_id: str) -> Microcontroller:
    """Get a Teensy microcontroller by USB ID."""
    from .teensy_microcontroller import TeensyMicrocontroller

    return TeensyMicrocontroller.find_by_usb_id(usb_id)


def microcontroller_open(request: MicrocontrollerOpenRequest) -> Microcontroller:
    """
    Get a microcontroller instance for the specified driver and USB ID.

    Args:
        request: MicrocontrollerOpenRequest with driver type and USB ID

    Returns:
        Microcontroller instance (not yet opened - call .open() to connect)

    Raises:
        ValueError: If the driver is not supported
        RuntimeError: If no device with the given USB ID is found
    """
    match request.driver:
        case "teensy":
            return _get_teensy_microcontroller(request.usb_id)
        case _:
            raise ValueError(f"unsupported microcontroller driver: {request.driver}")
