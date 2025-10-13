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

from seafront.config.basics import ChannelConfig, FilterConfig, ImagingOrder
from seafront.hardware.adapter import AdapterState, CoreState
from seafront.hardware.camera import HardwareLimitValue
from seafront.server import commands as cmd


@dataclass(frozen=True)
class HardwareLimits:
    """
    Complete hardware limits structure that mirrors the TypeScript HardwareLimits type.
    """
    imaging_exposure_time_ms: dict[str, tp.Union[float, int]]
    imaging_analog_gain_db: dict[str, tp.Union[float, int]]
    imaging_focus_offset_um: dict[str, tp.Union[float, int]]
    imaging_illum_perc: dict[str, tp.Union[float, int]]
    imaging_illum_perc_fluorescence: dict[str, tp.Union[float, int]]
    imaging_illum_perc_brightfield: dict[str, tp.Union[float, int]]
    imaging_number_z_planes: dict[str, tp.Union[float, int]]
    imaging_delta_z_um: dict[str, tp.Union[float, int]]
    
    def to_dict(self) -> dict[str, dict[str, tp.Union[float, int]]]:
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
    state: CoreState = CoreState.Idle
    is_connected: bool = False
    is_in_loading_position: bool = False
    
    stream_callback: tp.Callable[[np.ndarray | bool], bool] | None = Field(default=None)
    """
    call with either:
        image, then return if should stop or not
        or call with bool, which indicates if should stop (return value then ignored)
    """
    
    last_state: AdapterState | None = None
    
    _lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)
    _stop_streaming_flag: bool = PrivateAttr(default=False)
    "indicate that streaming should stop, without locking hardware"
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @contextmanager
    @abc.abstractmethod
    def lock(self, blocking: bool = True) -> tp.Iterator[tp.Self | None]:
        """
        Lock all hardware devices.
        
        Args:
            blocking: Whether to block waiting for the lock
            
        Yields:
            Self if lock acquired, None if blocking=False and lock unavailable
        """
        pass
    
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
    def is_position_forbidden(self, x_mm: float, y_mm: float, safety_radius_mm: float = 0.0) -> tuple[bool, str]:
        """
        Check if a position is forbidden for movement.

        This method validates whether the microscope can safely move to the given
        coordinates, considering any hardware-specific forbidden areas or constraints.

        Args:
            x_mm: X coordinate in mm
            y_mm: Y coordinate in mm
            safety_radius_mm: Safety margin radius around the position (default: 0.0)

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


def microscope_exclusive(f):
    """
    Decorator to ensure exclusive access to microscope hardware.
    
    This is a hardware-agnostic version of the squid_exclusive decorator.
    """
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return f(self, *args, **kwargs)
    return wrapper