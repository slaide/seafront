"""
What travels over the websockets, declared.

The HTTP routes get their schema from the request and response models FastAPI already
validates against. Sockets have no such machinery, so the message shapes are written down
here instead: every inbound message is parsed through one of these models rather than being
splatted into a call, and every outbound payload is built from one. `server/asyncapi.py`
publishes them.
"""

import typing as tp
from enum import StrEnum

import seaconfig as sc
from pydantic import BaseModel

from seafront.server.lock import LockReason
from seafront.server.commands import CapturedImageMetadata


class SocketErrorKind(StrEnum):
    TARGET_DEVICE_MISMATCH = "target_device_mismatch"
    DEVICE_LOCKED = "device_locked"
    HARDWARE_DISCONNECT = "hardware_disconnect"
    INTERNAL_ERROR = "internal_error"
    IMAGE_REPLACED = "image_replaced"


# -- inbound -----------------------------------------------------------------------------------


class SocketHandshake(BaseModel):
    """
    first message on every socket

    target_device is the same routing check the HTTP envelope does; an empty or absent value
    is the bypass a client uses before it knows the device name. lock_token is read only by
    sockets that drive the hardware.
    """

    target_device: str | None = None
    lock_token: str | None = None


class AcquisitionStatusQuery(BaseModel):
    """/ws/acquisition/status: which run to report on"""

    acquisition_id: str


class ProgressiveSnapRequest(BaseModel):
    """/ws/action/snap_selected_channels_progressive: the protocol to snap"""

    config_file: sc.AcquisitionConfig


# -- outbound ----------------------------------------------------------------------------------


class SocketMisdirected(BaseModel):
    """sent before closing with 4421: this socket reached the wrong microscope"""

    error: tp.Literal[SocketErrorKind.TARGET_DEVICE_MISMATCH] = (
        SocketErrorKind.TARGET_DEVICE_MISMATCH
    )
    message: str


class SocketLocked(BaseModel):
    """sent before closing with 4423: the websocket mirror of a 423"""

    error: tp.Literal[SocketErrorKind.DEVICE_LOCKED] = SocketErrorKind.DEVICE_LOCKED
    reason: LockReason
    owner: str | None = None
    message: str


class SocketFailure(BaseModel):
    """something went wrong mid-stream; the socket stays open and the client decides"""

    error: SocketErrorKind
    message: str
    timestamp: float
    channel_handle: str | None = None


class SocketErrorText(BaseModel):
    """a refusal that carries only its own text (bad acquisition id, and the like)"""

    error: str


class AcquiredImageInfo(BaseModel):
    """
    /ws/get_info/acquired_image: what follows this message, so the client can size the buffer

    `width` and `height` describe the transmitted (possibly downsampled) image. The pixels
    themselves arrive as the next frame, as raw bytes: monochrome, `bit_depth` bits per pixel,
    row by row, top to bottom, left to right.
    """

    channel_handle: str
    image_id: str
    width: int
    height: int
    camera_bit_depth: int
    bit_depth: int
    capture: CapturedImageMetadata


class NoImageYet(BaseModel):
    """nothing has been acquired in that channel yet (an empty object on the wire)"""
