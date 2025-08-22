"""
Base microscope abstraction for hardware-agnostic microscope control.

This module defines the abstract Microscope class that provides a common interface
for different microscope implementations (SquidAdapter, MockMicroscope, etc.).
"""

import abc
import threading
import typing as tp
from contextlib import contextmanager

import numpy as np
import seaconfig as sc
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from seafront.config.basics import ChannelConfig, FilterConfig
from seafront.hardware.adapter import AdapterState, CoreState
from seafront.server import commands as cmd


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
        
        Args:
            command: Command object to execute
            
        Returns:
            Command execution result
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