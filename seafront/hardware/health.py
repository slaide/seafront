"""
What each hardware device is doing, and the last thing that went wrong with it.

Every failure a device has ends up here as well as wherever it was raised, because raising only
tells the caller. The caller is often a request that has since been answered, or a teardown
nobody is waiting on, or the status loop - and the operator looking at the UI is none of them.
The board is what they read: `current_state` carries it, `/api/health` summarises it.

A condition is only ever set from an observation. A device is `unresponsive` because a read
came back empty or a call overran its deadline, `disconnected` because an operation on it
failed and the connection was dropped; never because of a guess about what it is probably
doing.
"""

import threading
import time
from enum import StrEnum

from pydantic import BaseModel


class Device(StrEnum):
    MAIN_CAMERA = "main_camera"
    FOCUS_CAMERA = "focus_camera"
    MICROCONTROLLER = "microcontroller"


class Condition(StrEnum):
    NOT_CONNECTED = "not_connected"
    "never opened, or closed on purpose"
    OK = "ok"
    "open, and the last thing done with it worked"
    UNRESPONSIVE = "unresponsive"
    "open, but it has been observed not answering: a read came back empty, or a call overran"
    DISCONNECTED = "disconnected"
    "an operation on it failed and the connection was dropped; the next command reconnects"


class DeviceFault(BaseModel):
    "one thing that went wrong with a device"

    operation: str
    "what was being done, e.g. `close`, `stream_off`, `status read`"
    message: str
    at: float
    "unix time"


class DeviceHealth(BaseModel):
    device: Device
    condition: Condition
    since: float
    "unix time the device entered this condition"
    last_fault: DeviceFault | None = None
    "the most recent fault, kept after the device recovers: a glitch an hour ago still happened"
    last_heard_at: float | None = None
    """
    unix time the device last proved it was alive (the microcontroller: its last status packet).
    null for devices that have no way to say so, or that have not said anything yet
    """


def describe(error: BaseException) -> str:
    "an exception as one line an operator can act on: its type, and what it said"
    text = str(error)
    return f"{type(error).__name__}: {text}" if text else type(error).__name__


class DeviceCloseError(Exception):
    """
    A device was released, but not cleanly.

    Raised by a driver's `close()` *after* it has let go of the handle, so the driver is closed
    either way - what this reports is which of the steps on the way failed, which is what says
    whether the physical device may still be streaming, or still held open by the SDK.
    """

    def __init__(self, device: str, failures: list[tuple[str, BaseException]]):
        self.device = device
        self.failures = failures
        steps = "; ".join(f"{step}: {describe(error)}" for step, error in failures)
        super().__init__(f"{device} did not close cleanly ({steps})")


class HardwareCloseError(Exception):
    """
    The adapter closed every device, and some of them did not close cleanly.

    One error per device, keyed by which device it was. Every device is attempted whatever
    happened to the ones before it: a camera that fails to close is no reason to leave the
    stage controller open.
    """

    def __init__(self, failures: dict[Device, BaseException]):
        self.failures = failures
        detail = "; ".join(f"{device}: {describe(error)}" for device, error in failures.items())
        super().__init__(f"hardware did not close cleanly - {detail}")


class HealthBoard:
    """
    The condition of each device an adapter drives. Thread-safe: written from the worker, the
    status read, the stall watchdog and the camera SDK's threads.
    """

    def __init__(self, devices: list[Device]):
        now = time.time()
        self._lock = threading.Lock()
        self._health: dict[Device, DeviceHealth] = {
            device: DeviceHealth(device=device, condition=Condition.NOT_CONNECTED, since=now)
            for device in devices
        }

    def _set(
        self, device: Device, condition: Condition, fault: DeviceFault | None = None
    ) -> None:
        with self._lock:
            current = self._health[device]
            self._health[device] = current.model_copy(
                update={
                    "condition": condition,
                    "since": current.since if current.condition == condition else time.time(),
                    "last_fault": fault if fault is not None else current.last_fault,
                }
            )

    def connected(self, device: Device) -> None:
        self._set(device, Condition.OK)

    def closed(self, device: Device, error: BaseException | None = None) -> None:
        "closed on purpose; `error` is what went wrong on the way, if anything"
        self._set(
            device,
            Condition.NOT_CONNECTED,
            None if error is None else DeviceFault(operation="close", message=describe(error), at=time.time()),
        )

    def open_failed(self, device: Device, error: BaseException) -> None:
        self._set(
            device,
            Condition.NOT_CONNECTED,
            DeviceFault(operation="open", message=describe(error), at=time.time()),
        )

    def lost(self, device: Device, operation: str, error: BaseException) -> None:
        "an operation failed in a way that means the connection is gone"
        self._set(
            device,
            Condition.DISCONNECTED,
            DeviceFault(operation=operation, message=describe(error), at=time.time()),
        )

    def unresponsive(self, device: Device, operation: str, message: str) -> None:
        self._set(
            device,
            Condition.UNRESPONSIVE,
            DeviceFault(operation=operation, message=message, at=time.time()),
        )

    def heard(self, device: Device, at: float | None = None) -> None:
        """
        the device proved it is alive, at `at`. Only lifts `unresponsive`, and only when it was
        heard from after being marked so: a device that is not connected, or was dropped, stays
        so until it is opened again, and hearing about a packet from before it went quiet says
        nothing about now.
        """
        at = time.time() if at is None else at
        with self._lock:
            current = self._health[device]
            update: dict[str, object] = {"last_heard_at": max(at, current.last_heard_at or at)}
            if current.condition == Condition.UNRESPONSIVE and at > current.since:
                update |= {"condition": Condition.OK, "since": at}
            self._health[device] = current.model_copy(update=update)

    def get(self, device: Device) -> DeviceHealth:
        with self._lock:
            return self._health[device]

    def snapshot(self) -> list[DeviceHealth]:
        with self._lock:
            return list(self._health.values())
