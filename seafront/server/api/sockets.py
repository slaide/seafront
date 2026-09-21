"""
The websockets: the same operations as the HTTP routes, but pushed rather than polled.

Each socket opens with a handshake that says which microscope the client means and, for the
one socket that drives the hardware, carries the device lease. `socketspec.socket()` does all
of that before a handler here runs, so what is left in this module is only what each socket
does once it is open. After that every inbound message is parsed through a model in
`messages.py` before anything acts on it - a socket takes client JSON exactly as sceptically
as a POST body does.

Close codes mirror the HTTP refusals: 4421 for a misdirected request (421), 4423 for a
missing or foreign device lease (423).
"""

import asyncio
import time

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from seafront.logger import logger
from seafront.server.api.messages import (
    AcquiredImageInfo,
    AcquisitionStatusQuery,
    NoImageYet,
    ProgressiveSnapRequest,
    SocketErrorKind,
    SocketErrorText,
    SocketFailure,
    SocketHandshake,
    SocketLocked,
    SocketMisdirected,
)
from seafront.server.api.socketspec import Frame, send, send_bytes, socket
from seafront.server.commands import (
    AcquisitionStatusOut,
    ChannelSnapProgressiveStatus,
    CoreCurrentState,
)
from seafront.server.core import Core
from seafront.server.protocol import make_unique_acquisition_id

router = APIRouter()

_status_pushes: set[asyncio.Task] = set()
"""
strong references to in-flight status sends. A task with no reference can be collected
before it runs, which silently drops the update it was carrying.
"""

HANDSHAKE = Frame(
    "handshake",
    "which microscope this socket means, and the device lease for the sockets that need one",
    SocketHandshake,
)
MISDIRECTED = Frame(
    "misdirected", "sent before closing 4421: this socket reached the wrong microscope", SocketMisdirected
)
FAILURE = Frame("failure", "something went wrong; the socket stays open", SocketFailure)

MISDIRECTED_CLOSE = {4421: "target_device named another microscope", 4400: "malformed message"}


@socket(
    router,
    "/ws/get_info/current_state",
    receive=[
        HANDSHAKE,
        Frame("poll", "any message; its contents are ignored and one state reply is sent", schema={}),
    ],
    send=[
        Frame("state", "the microscope's current state", CoreCurrentState),
        MISDIRECTED,
        FAILURE,
    ],
    close_codes=MISDIRECTED_CLOSE,
)
async def ws_current_state(ws: WebSocket, core: Core, opening: SocketHandshake):
    """
    get current state of the microscope

    after the handshake, every message received triggers one state reply; the contents of
    that message are ignored. A websocket wrapper around /api/get_info/current_state.
    """
    while True:
        message = await ws.receive()
        if message.get("type") in {"websocket.disconnect", "websocket.close"}:
            break

        try:
            await send(ws, await core.get_current_state())
        except Exception as e:
            # hardware disconnects and the like: report and let the client decide
            await send(
                ws,
                SocketFailure(
                    error=SocketErrorKind.HARDWARE_DISCONNECT
                    if "disconnect" in str(e).lower()
                    else SocketErrorKind.INTERNAL_ERROR,
                    message=str(e),
                    timestamp=time.time(),
                ),
            )


@socket(
    router,
    "/ws/get_info/acquired_image",
    receive=[
        HANDSHAKE,
        Frame("channel_handle", "which channel to read, as a text frame", schema={"type": "string"}),
        Frame(
            "image_id",
            "the exact image returned by the request that produced this identifier",
            schema={"type": "string"},
        ),
        Frame(
            "downsample_factor",
            "take every nth pixel on both axes, as a text frame holding an integer >= 1",
            schema={"type": "string", "pattern": r"^[1-9][0-9]*$"},
        ),
    ],
    send=[
        Frame("image_info", "what the following binary frame contains", AcquiredImageInfo),
        Frame("no_image_yet", "that channel has not been acquired yet", NoImageYet),
        Frame(
            "pixels",
            "the image itself: one binary frame, monochrome, row by row, top to bottom",
            schema={"type": "string", "format": "binary"},
        ),
        MISDIRECTED,
        FAILURE,
    ],
    close_codes=MISDIRECTED_CLOSE,
)
async def ws_acquired_image(ws: WebSocket, core: Core, opening: SocketHandshake):
    """
    get acquired image data

    after the handshake the exchange repeats in three steps:
     - the client sends a channel handle as a text frame
     - the client sends the expected image identifier as a text frame
     - the client sends a downsampling factor as a text frame
     - the server replies with AcquiredImageInfo followed by the pixels as one binary frame,
       or an empty object if that channel has nothing yet
    """
    while True:
        channel_handle = await ws.receive_text()
        image_id = await ws.receive_text()
        raw_factor = await ws.receive_text()
        try:
            factor = int(raw_factor)
        except ValueError:
            await send(
                ws,
                SocketFailure(
                    error=SocketErrorKind.INTERNAL_ERROR,
                    message=f"downsampling factor must be an integer, got {raw_factor!r}",
                    timestamp=time.time(),
                ),
            )
            continue
        if factor < 1:
            await send(
                ws,
                SocketFailure(
                    error=SocketErrorKind.INTERNAL_ERROR,
                    message=f"downsampling factor must be at least 1, got {factor}",
                    timestamp=time.time(),
                ),
            )
            continue

        img = core.latest_images.get(channel_handle)
        if img is None:
            if not await send(ws, NoImageYet()):
                break
            continue

        if img.info.image_id != image_id:
            if not await send(
                ws,
                SocketFailure(
                    error=SocketErrorKind.IMAGE_REPLACED,
                    message=(
                        f"image {image_id!r} is no longer available for channel "
                        f"{channel_handle!r}"
                    ),
                    timestamp=time.time(),
                    channel_handle=channel_handle,
                ),
            ):
                break
            continue

        img_data = np.ascontiguousarray(img._img[::factor, ::factor])
        assert img.capture is not None
        info = AcquiredImageInfo(
            channel_handle=channel_handle,
            image_id=image_id,
            width=img_data.shape[1],
            height=img_data.shape[0],
            camera_bit_depth=img.bit_depth,
            bit_depth=img_data.dtype.itemsize * 8,
            capture=img.capture,
        )
        if not await send(ws, info):
            break

        if not await send_bytes(ws, img_data.tobytes()):
            break


@socket(
    router,
    "/ws/acquisition/status",
    receive=[HANDSHAKE, Frame("query", "which run to report on", AcquisitionStatusQuery)],
    send=[
        Frame("status", "one status snapshot of that run", AcquisitionStatusOut),
        Frame("refused", "the id was not one this server knows", SocketErrorText),
        MISDIRECTED,
    ],
    close_codes=MISDIRECTED_CLOSE,
)
async def ws_acquisition_status(ws: WebSocket, core: Core, opening: SocketHandshake):
    """
    watch a running acquisition

    after the handshake, each message names the acquisition to report on and is answered with
    one status.
    """
    while True:
        raw = await ws.receive_json()
        try:
            query = AcquisitionStatusQuery.model_validate(raw)
        except ValidationError as exc:
            await send(ws, {"error": f"expected an acquisition_id: {exc}"})
            continue

        try:
            await send(ws, await core.get_acquisition_status(query.acquisition_id))
        except Exception as e:
            detail = getattr(e, "detail", None)
            if detail is None:
                logger.warning(f"Error getting acquisition status: {e}")
            await send(ws, {"error": str(detail if detail is not None else e)})


@socket(
    router,
    "/ws/action/snap_selected_channels_progressive",
    requires_lease=True,
    receive=[
        HANDSHAKE,
        Frame("protocol", "the acquisition config whose channels to snap", ProgressiveSnapRequest),
    ],
    send=[
        Frame("progress", "one update per channel as it completes", ChannelSnapProgressiveStatus),
        Frame("locked", "sent before closing 4423: the websocket mirror of a 423", SocketLocked),
        MISDIRECTED,
        FAILURE,
    ],
    close_codes=MISDIRECTED_CLOSE | {4423: "device lease missing or held by another client"},
)
async def ws_snap_selected_channels_progressive(
    ws: WebSocket, core: Core, opening: SocketHandshake
):
    """
    snap every selected channel, reporting each one as it completes

    The handshake must carry lock_token: this socket drives the hardware, so it is gated
    exactly like POST /api/action/snap_selected_channels (requires_lease on the declaration
    above is what enforces that). The first message after it is the protocol to snap. Each
    completed channel image is immediately available from the normal image retrieval
    endpoints.
    """
    callback_id = make_unique_acquisition_id()

    try:
        try:
            request = ProgressiveSnapRequest.model_validate({"config_file": await ws.receive_json()})
        except ValidationError as exc:
            await send(
                ws,
                SocketFailure(
                    error=SocketErrorKind.INTERNAL_ERROR,
                    message=f"expected an acquisition config: {exc}",
                    timestamp=time.time(),
                ),
            )
            await ws.close(code=4400)
            return

        # The snap loop runs on the worker thread, so marshal each status onto this socket's
        # event loop with call_soon_threadsafe before touching the socket.
        loop = asyncio.get_running_loop()

        def send_status_update(status: ChannelSnapProgressiveStatus):
            def _send():
                try:
                    task = asyncio.create_task(send(ws, status))
                    _status_pushes.add(task)
                    task.add_done_callback(_status_pushes.discard)
                except Exception as e:
                    logger.warning(f"Failed to send progressive snap status: {e}")

            loop.call_soon_threadsafe(_send)

        core._progressive_snap_callbacks[callback_id] = send_status_update

        try:
            await core.start_progressive_channel_snap(request.config_file, callback_id)
        except Exception as e:
            send_status_update(
                ChannelSnapProgressiveStatus(
                    channel_handle="",
                    channel_name="",
                    status="error",
                    total_channels=0,
                    completed_channels=0,
                    message="Failed to start progressive snap",
                    error_detail=str(e),
                )
            )

    except WebSocketDisconnect:
        raise  # a client leaving mid-snap is not an error; the gate swallows it
    except Exception as e:
        logger.error(f"Progressive snap WebSocket error: {e}")
    finally:
        core._progressive_snap_callbacks.pop(callback_id, None)
