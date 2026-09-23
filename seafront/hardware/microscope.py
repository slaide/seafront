"""
Base microscope abstraction for hardware-agnostic microscope control.

This module defines the abstract Microscope class that provides a common interface
for different microscope implementations (SquidAdapter, MockMicroscope, etc.).
"""

import abc
import threading
import time
import typing as tp
from contextlib import contextmanager

import numpy as np
import seaconfig as sc
from pydantic import BaseModel, ConfigDict, PrivateAttr

from seafront.config.basics import (
    ChannelConfig,
    ConfigItem,
    FilterConfig,
    GlobalConfigHandler,
)
from seafront.config.handles import LaserAutofocusConfig
from seafront.hardware.adapter import AdapterState
from seafront.hardware.camera import Camera
from seafront.hardware.health import Device, DeviceHealth, HealthBoard
from seafront.hardware.watchdog import Stall, StallWatchdog
from seafront.hardware.limits import HardwareLimits
from seafront.server import commands as cmd


class DisconnectError(BaseException):
    """
    Indicate that the hardware was disconnected.

    `device` is which one was lost, when the adapter knows. If dropping the connections on the
    way out did not go cleanly either, that failure rides along as a note rather than
    replacing this error: the disconnect is what happened, the untidy close is detail.
    """

    def __init__(self, device: Device | None = None):
        super().__init__(f"{device} disconnected" if device is not None else "hardware disconnected")
        self.device = device


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




class StreamState(BaseModel):
    """
    What the camera is streaming right now, or None when it is not streaming.

    Frozen, and the adapter's own: whoever reads it gets the channel and the moment it
    started as one snapshot, so a status read cannot catch a stream that has a channel but is
    not running, or the other way round.
    """

    model_config = ConfigDict(frozen=True)

    channel: sc.AcquisitionChannelConfig
    "the channel the stream was started with, and whose illumination is currently on"
    started_at: float
    "unix time, for the record; nothing decides anything from it"


type FrameSink = tp.Callable[[np.ndarray, sc.AcquisitionChannelConfig], None]
"""
what a streamed frame is handed to: the image, and the channel it belongs to

Installed once with set_frame_sink() and left there. The stream state, not the presence of a
sink, is what says whether frames should flow - installing and removing a callback around
each stream is what used to drop every frame between the camera starting and the server
getting round to registering interest in it.
"""


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

    last_state: AdapterState | None = None

    main_camera: Locked[Camera]

    _stream: StreamState | None = PrivateAttr(default=None)
    """
    the one answer to "is the camera streaming, and with what"

    Written by the worker inside ChannelStreamBegin/End, read by the SDK's frame-callback
    thread and by every status read, all under _stream_lock.
    """

    _frame_sink: FrameSink | None = PrivateAttr(default=None)
    "where streamed frames go; installed once by whoever owns this adapter"

    _stream_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    """The one remaining cross-thread guard: serializes the stream state between the worker
    and the SDK frame-callback thread. Never held while taking the camera lock - the capture
    callback would then deadlock against stop_streaming()."""

    _current_cancel: threading.Event | None = PrivateAttr(default=None)
    """Cancel event of the command currently executing, set by the worker. Single
    owner => at most one command runs at a time, so this is unambiguous."""

    _health: HealthBoard = PrivateAttr()
    "the condition of each device this adapter drives; see hardware/health.py"

    _watchdog: StallWatchdog = PrivateAttr()
    "deadlines on the hardware calls that can hang; see hardware/watchdog.py"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, context: tp.Any, /) -> None:
        super().model_post_init(context)
        self._health = HealthBoard(self.hardware_devices())
        self._watchdog = StallWatchdog(self._health)

    @classmethod
    @abc.abstractmethod
    def hardware_devices(cls) -> list[Device]:
        "the devices this adapter drives, which is what its health board reports on"

    def health(self) -> list[DeviceHealth]:
        "the condition of every device, as one snapshot. Never touches the hardware."
        return self._health.snapshot()

    def stalls(self) -> list[Stall]:
        """
        hardware calls past their deadline and still running. Never touches the hardware, and
        never waits on it - which is the point: while one of these lasts, anything that takes a
        device lock waits for as long as the stuck call does.
        """
        return self._watchdog.stalls()

    def main_camera_info(self) -> cmd.CameraInfo:
        "the main camera's identity, which its driver knows from before it is opened. No I/O."
        camera = self.main_camera.value
        return cmd.CameraInfo(vendor=camera.vendor_name, model=camera.model_name, serial_number=camera.sn)

    @abc.abstractmethod
    def illumination_state(self) -> cmd.IlluminationState:
        "which lights are on, as far as the adapter has told them. No I/O."

    # -- streaming: the adapter owns whether it is happening ----------------------------------

    def set_frame_sink(self, sink: FrameSink | None) -> None:
        """Install the one consumer of streamed frames. Done once, when the adapter is built."""
        self._frame_sink = sink

    def stream_state(self) -> StreamState | None:
        """What is streaming, or None. The single source of truth, for status and for policy."""
        with self._stream_lock:
            return self._stream

    @property
    def is_streaming(self) -> bool:
        return self.stream_state() is not None

    def _stream_started(self, channel: sc.AcquisitionChannelConfig) -> None:
        """
        Mark the stream as running. Adapters call this from ChannelStreamBegin *before*
        starting the camera, so no frame arrives while the adapter still thinks it is idle.
        A begin that then fails must call _stream_stopped() on its way out.
        """
        with self._stream_lock:
            self._stream = StreamState(channel=channel, started_at=time.time())

    def _stream_stopped(self) -> StreamState | None:
        """Mark the stream as over and hand back what it was, or None if nothing was running."""
        with self._stream_lock:
            previous, self._stream = self._stream, None
            return previous

    def _forward_frame(self, img: np.ndarray) -> None:
        """
        Hand one frame to the sink. Runs on the camera SDK's acquisition thread.

        Reads the state and the sink under the lock and calls out from outside it, so a slow
        consumer can never hold up a stream stopping.
        """
        with self._stream_lock:
            stream = self._stream
            sink = self._frame_sink
        if stream is None or sink is None:
            return
        sink(img, stream.channel)

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

        Every device is attempted and released whatever happens to the others, and calling this
        with some or all of them already closed is fine. If any device did not close cleanly,
        raises HardwareCloseError afterwards - by then everything has been let go of, so the
        error reports what state the devices may have been left in, not a close to retry.
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

    def record_laser_autofocus_calibration(self, data: cmd.LaserAutofocusCalibrationData) -> None:
        """
        Write a fresh laser autofocus calibration into the privileged machine config and
        persist it. The server computed these numbers, so it owns them; a client only
        needs to re-read the machine config afterwards.
        """
        GlobalConfigHandler.override(
            [
                ConfigItem(
                    name="", handle=LaserAutofocusConfig.CALIBRATION_IS_CALIBRATED.value,
                    value_kind="option", value="yes",
                ),
                ConfigItem(
                    name="", handle=LaserAutofocusConfig.CALIBRATION_X_PEAK_POS.value,
                    value_kind="float", value=data.x_reference,
                ),
                ConfigItem(
                    name="", handle=LaserAutofocusConfig.CALIBRATION_UM_PER_PX.value,
                    value_kind="float", value=data.um_per_px,
                ),
                ConfigItem(
                    name="", handle=LaserAutofocusConfig.CALIBRATION_REF_Z_MM.value,
                    value_kind="float", value=data.calibration_position.z_pos_mm,
                ),
                ConfigItem(
                    name="", handle=LaserAutofocusConfig.OFFSET_UM.value,
                    value_kind="float", value=data.offset_um,
                ),
            ],
            privileged=True,
        )

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
