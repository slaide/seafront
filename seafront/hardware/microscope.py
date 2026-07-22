"""
Base microscope abstraction for hardware-agnostic microscope control.

This module defines the abstract Microscope class that provides a common interface
for different microscope implementations (SquidAdapter, MockMicroscope, etc.).
"""

import abc
import threading
import typing as tp
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import seaconfig as sc
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from seafront.config.basics import ChannelConfig, ConfigItem, FilterConfig, ImagingOrder
from seafront.hardware.adapter import AdapterState
from seafront.server import commands as cmd


class DisconnectError(BaseException):
    """Indicate that the hardware was disconnected."""

    def __init__(self):
        super().__init__()


class OperationCancelledError(Exception):
    """Raised inside a hardware operation when its cancel event has been set."""


class Locked[T]:
    """Thread-safe wrapper for a value with a reentrant lock."""

    def __init__(self, t: T):
        self.lock = threading.RLock()
        self.t = t

    @property
    def value(self) -> T:
        """Quick access to the wrapped value without locking (for read-only access)."""
        return self.t

    @contextmanager
    def locked(self, blocking: bool = True) -> tp.Iterator[T | None]:
        if self.lock.acquire(blocking=blocking):
            try:
                yield self.t
            finally:
                self.lock.release()
        else:
            yield None


@dataclass(frozen=True)
class HardwareLimits:
    """
    Complete hardware limits structure that mirrors the TypeScript HardwareLimits type.
    """
    imaging_exposure_time_ms: dict[str, float | int]
    imaging_analog_gain_db: dict[str, float | int]
    imaging_focus_offset_um: dict[str, float | int]
    imaging_illum_perc: dict[str, float | int]
    imaging_illum_perc_fluorescence: dict[str, float | int]
    imaging_illum_perc_brightfield: dict[str, float | int]
    imaging_number_z_planes: dict[str, float | int]
    imaging_delta_z_um: dict[str, float | int]

    def to_dict(self) -> dict[str, dict[str, float | int]]:
        """Convert to dictionary format for API responses."""
        return {
            "imaging_exposure_time_ms": self.imaging_exposure_time_ms,
            "imaging_analog_gain_db": self.imaging_analog_gain_db,
            "imaging_focus_offset_um": self.imaging_focus_offset_um,
            "imaging_illum_perc": self.imaging_illum_perc,
            "imaging_illum_perc_fluorescence": self.imaging_illum_perc_fluorescence,
            "imaging_illum_perc_brightfield": self.imaging_illum_perc_brightfield,
            "imaging_number_z_planes": self.imaging_number_z_planes,
            "imaging_delta_z_um": self.imaging_delta_z_um,
        }


class Microscope(BaseModel, abc.ABC):
    """
    Abstract base class for microscope implementations.
    
    This class defines the common interface that all microscope implementations
    must provide, enabling hardware-agnostic operation.
    """

    # Common state attributes
    channels: list[ChannelConfig]
    filters: list[FilterConfig]
    is_connected: bool = False
    is_in_loading_position: bool = False

    stream_callback: tp.Callable[[np.ndarray | bool], bool] | None = Field(default=None)
    """
    call with either:
        image, then return if should stop or not
        or call with bool, which indicates if should stop (return value then ignored)

    Touched from three threads (the worker sets/clears it via ChannelStream begin/end,
    the camera SDK's acquisition thread reads it to forward frames), so all access goes
    through set_stream_callback()/read_stream_callback() under _stream_callback_lock.
    """

    last_state: AdapterState | None = None

    _stream_callback_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    """The one remaining cross-thread guard: serializes access to stream_callback between
    the worker and the SDK frame-callback thread."""

    _current_cancel: threading.Event | None = PrivateAttr(default=None)
    """Cancel event of the command currently executing, set by the worker. Single
    owner => at most one command runs at a time, so this is unambiguous."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def set_stream_callback(self, cb: tp.Callable[[np.ndarray | bool], bool] | None) -> None:
        """Set or clear the streaming frame callback (worker thread)."""
        with self._stream_callback_lock:
            self.stream_callback = cb

    def read_stream_callback(self) -> tp.Callable[[np.ndarray | bool], bool] | None:
        """Read the current streaming frame callback (SDK acquisition thread)."""
        with self._stream_callback_lock:
            return self.stream_callback

    def set_current_cancel(self, cancel: threading.Event | None) -> None:
        """Set (or clear) the cancel event for the in-flight command. Called by the
        worker around each command; long operations poll it via raise_if_cancelled()."""
        self._current_cancel = cancel

    def raise_if_cancelled(self) -> None:
        """Cooperative cancel checkpoint: raise OperationCancelledError if the current
        command's cancel event is set. Safe to call from any hardware operation."""
        if self.should_cancel():
            raise OperationCancelledError()

    def should_cancel(self) -> bool:
        """True if the in-flight command's cancel event is set. Callers that want a
        different exception (e.g. acquisition) check this instead of raise_if_cancelled."""
        return self._current_cancel is not None and self._current_cancel.is_set()

    @classmethod
    @abc.abstractmethod
    def make(cls) -> "Microscope":
        """
        Factory method to create and configure a microscope instance.
        
        Returns:
            An unconnected microscope instance
        """
        pass

    @abc.abstractmethod
    def open_connections(self) -> None:
        """
        Open connections to all hardware devices.
        
        This method should establish communication with cameras, microcontrollers,
        and any other hardware components.
        """
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """
        Close connections to all hardware devices.
        
        Should be safe to call even if some devices are already disconnected.
        """
        pass

    @abc.abstractmethod
    async def home(self) -> None:
        """
        Perform complete homing/calibration sequence for the microscope.
        
        This should include:
        - Resetting microcontroller
        - Initializing motors
        - Calibrating stage positions
        - Filter wheel initialization (if present)
        """
        pass

    @abc.abstractmethod
    async def get_current_state(self) -> AdapterState:
        """
        Get current microscope state including stage position.
        
        Returns:
            Current state with calibrated positions
        """
        pass

    @property
    @abc.abstractmethod
    def calibrated_stage_position(self) -> tuple[float, float, float]:
        """
        Get calibrated XY stage offset from configuration.
        
        Returns:
            (x_mm, y_mm, z_mm) calibration offset tuple
        """
        pass

    @abc.abstractmethod
    async def snap_selected_channels(self, config_file: sc.AcquisitionConfig) -> cmd.BasicSuccessResponse:
        """
        Take snapshots of all enabled channels in the acquisition config.
        
        Args:
            config_file: Acquisition configuration with enabled channels
            
        Returns:
            Success response
        """
        pass

    @abc.abstractmethod
    async def execute[T](self, command: cmd.BaseCommand[T]) -> T:
        """
        Main command dispatcher for all microscope operations.
        
        Implementations should call self.validate_command(command) at the start
        of their execute method to ensure commands are validated before execution.
        
        Args:
            command: Command object to execute
            
        Returns:
            Command execution result
        """
        pass

    @abc.abstractmethod
    def get_hardware_limits(self) -> HardwareLimits:
        """
        Get hardware-specific limits for all configurable parameters.
        
        This method should query the actual hardware components to determine
        their capabilities and combine them into a complete limits structure.
        
        Camera limits (exposure time, gain) should be obtained from the main camera.
        Mechanical limits (focus offset, z-planes) should be defined by the microscope.
        Power limits should be specific to the illumination system.
        
        Returns:
            HardwareLimits object with strongly-typed limit values
        """
        pass

    @abc.abstractmethod
    def validate_command(self, command: cmd.BaseCommand[tp.Any]) -> None:
        """
        Validate a command against hardware limits before execution.
        
        This method should validate all imaging parameters in the command against
        the current hardware limits. It should raise an HTTPException if any
        parameter is outside the valid range.
        
        Args:
            command: Command object to validate
            
        Raises:
            HTTPException: If any parameter is outside valid range
            
        Returns:
            None on success (raises exception on failure)
        """
        pass

    @abc.abstractmethod
    def is_position_forbidden(self, x_mm: float, y_mm: float) -> tuple[bool, str]:
        """
        Check if a position is forbidden for movement.

        This method validates whether the microscope can safely move to the given
        coordinates, considering any hardware-specific forbidden areas or constraints.

        Args:
            x_mm: X coordinate in mm
            y_mm: Y coordinate in mm

        Returns:
            Tuple of (is_forbidden, error_message). error_message is empty if position is allowed.
        """
        pass

    @abc.abstractmethod
    def _sort_channels_by_imaging_order(self, channels: list, imaging_order: ImagingOrder) -> list:
        """
        Sort channels according to the specified imaging order.

        Args:
            channels: List of enabled AcquisitionChannelConfig objects
            imaging_order: Sorting strategy to use

        Returns:
            Sorted list of channels
        """
        pass

    @abc.abstractmethod
    def validate_channel_for_acquisition(self, channel: sc.AcquisitionChannelConfig) -> None:
        """
        Validate that a channel can be acquired with current microscope configuration.

        Different microscope implementations may have different validation requirements.
        For example, microscopes with filter wheels may require filter selection,
        or some microscopes may disallow certain lightsource+filter combinations.

        Args:
            channel: The channel configuration to validate

        Raises:
            Calls error_internal() if validation fails
        """
        pass

    @abc.abstractmethod
    def extend_machine_config(self, config_items: list[ConfigItem]) -> None:
        """
        Extend machine configuration with microscope-specific options.

        Microscope implementations can modify config_items in-place to add or update
        options based on their hardware capabilities. This method should delegate to
        cameras and other hardware components to allow them to extend the config.

        Args:
            config_items: List of ConfigItem objects to modify in-place
        """
        pass

    @abc.abstractmethod
    def estimate_acquisition(self, config: sc.AcquisitionConfig) -> cmd.AcquisitionEstimate:
        """
        Estimate storage and time requirements for an acquisition.

        Each microscope implementation can provide its own estimates based on
        its hardware characteristics (stage speed, camera frame rate, etc.).

        Args:
            config: The acquisition configuration to estimate

        Returns:
            AcquisitionEstimate with storage and time estimates
        """
        pass
