"""
How a websocket declares itself, and what that declaration does.

This is `routing.py` for sockets. The HTTP routes get their schema for free: FastAPI reads
the models they are annotated with. Sockets have no equivalent, so each one states its frames
here instead - what the client may send, what the server sends back, and whether it needs the
device lease - right next to the handler, the same way a route states its Policy.

The declaration is also what enforces. `socket()` wraps the handler with everything that has
to happen before a socket may do anything: accept it, read the opening message, check it
reached the right microscope, and - when the socket drives the hardware - check the device
lease. A handler is only ever called on a socket that has passed all of it, and is handed the
handshake it passed with.

That wrapping is the whole point of this module existing rather than the spec being a
docstring. The gate used to be declared here and written out by hand inside each handler, so
a socket could advertise 4423 in asyncapi.json and refuse nobody, and the test that read the
published contract would agree with it. Now there is one place, and no way to say one thing
and do another.

What a socket passes through, in order:

    handshake   4400   the first message is not a handshake object
    routing     4421   the handshake names a different microscope than this server drives
    lease       4423   the caller does not hold the device lease (gated sockets only)
    handler

`server/asyncapi.py` turns the same declarations into the published document.
"""

import time
import typing as tp
from dataclasses import dataclass, field

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

from seafront.logger import logger
from seafront.server.api.messages import (
    SocketErrorKind,
    SocketFailure,
    SocketHandshake,
    SocketLocked,
    SocketMisdirected,
)
from seafront.server.api.routing import get_core
from seafront.server.core import Core


@dataclass(frozen=True, slots=True)
class Frame:
    """one kind of message that crosses a socket"""

    name: str
    description: str
    model: type[BaseModel] | None = None
    "JSON payload, described by this model"
    schema: dict[str, tp.Any] | None = None
    "for frames that are not JSON objects: a plain text or binary frame"


@dataclass(frozen=True, slots=True)
class SocketSpec:
    path: str
    summary: str
    description: str
    requires_lease: bool
    receive: list[Frame] = field(default_factory=list)
    "client -> server"
    send: list[Frame] = field(default_factory=list)
    "server -> client"
    close_codes: dict[int, str] = field(default_factory=dict)


SOCKETS: list[SocketSpec] = []
"every declared socket, in registration order; read by server/asyncapi.py"


# -- the transport helpers every socket shares ---------------------------------------------


async def send(ws: WebSocket, payload: BaseModel | dict[str, tp.Any]) -> bool:
    "send one JSON message; False once the socket has gone away"
    try:
        await ws.send_json(
            payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
        )
        return True
    except Exception as exc:
        logger.debug(f"websocket send_json failed: {exc}")
        return False


async def send_bytes(ws: WebSocket, payload: bytes) -> bool:
    "send one binary message; False once the socket has gone away"
    try:
        await ws.send_bytes(payload)
        return True
    except Exception as exc:
        logger.debug(f"websocket send_bytes failed: {exc}")
        return False


async def _handshake(ws: WebSocket, core: Core) -> SocketHandshake | None:
    """
    Read and check the opening message. None once the socket has been closed.
    """
    try:
        opening = SocketHandshake.model_validate(await ws.receive_json())
    except (ValidationError, ValueError) as exc:
        await send(
            ws,
            SocketFailure(
                error=SocketErrorKind.INTERNAL_ERROR,
                message=f"first message must be a handshake object: {exc}",
                timestamp=time.time(),
            ),
        )
        await ws.close(code=4400)
        return None

    if opening.target_device and opening.target_device != core.microscope_name:
        await send(
            ws,
            SocketMisdirected(
                message=f"target_device mismatch: request targets '{opening.target_device}' but this server is '{core.microscope_name}'"
            ),
        )
        await ws.close(code=4421)
        return None

    return opening


async def _require_lease(ws: WebSocket, core: Core, opening: SocketHandshake) -> bool:
    """
    The lease rule of the hardware routes, in websocket clothing.

    False once the socket has been closed with 4423, mirroring the 423 a POST would have got.
    """
    decision = core.lock.check(opening.lock_token)
    if decision.permitted:
        return True

    await send(
        ws,
        SocketLocked(
            reason=decision.reason,
            owner=decision.state.owner,
            message=decision.message,
        ),
    )
    await ws.close(code=4423)
    return False


type Handler = tp.Callable[[WebSocket, Core, SocketHandshake], tp.Awaitable[None]]
"""
what a socket handler is, once the gate is in front of it

It gets a socket that is accepted, aimed at this microscope, and (for a gated socket) held by
the caller, plus the handshake that proved it.
"""


def socket(
    router: APIRouter,
    path: str,
    *,
    requires_lease: bool = False,
    receive: list[Frame],
    send: list[Frame],
    close_codes: dict[int, str] | None = None,
):
    """
    register a websocket handler, record what travels over it, and enforce that record

    The handler is called as `(ws, core, opening)` and only once the socket has been accepted
    and admitted. `requires_lease` is both what asyncapi.json publishes and what closes the
    socket with 4423, so the two cannot disagree.
    """
    declared_send = send

    def decorator(fn: Handler) -> Handler:
        doc = [line.strip() for line in (fn.__doc__ or "").splitlines()]
        doc = [line for line in doc if line]
        SOCKETS.append(
            SocketSpec(
                path=path,
                summary=doc[0] if doc else "",
                description="\n".join(doc[1:]),
                requires_lease=requires_lease,
                receive=receive,
                send=declared_send,
                close_codes=close_codes or {},
            )
        )

        # Deliberately not functools.wraps: that sets __wrapped__, and FastAPI follows it to
        # build the endpoint's signature - it would then try to inject `core` and `opening`
        # as request parameters. The socket the router registers takes exactly one argument.
        async def gated(ws: WebSocket) -> None:
            core = get_core(ws)
            await ws.accept()
            try:
                opening = await _handshake(ws, core)
                if opening is None:
                    return
                if requires_lease and not await _require_lease(ws, core, opening):
                    return
                await fn(ws, core, opening)
            except WebSocketDisconnect:
                pass

        gated.__name__ = fn.__name__
        gated.__doc__ = fn.__doc__
        router.websocket(path)(gated)
        return fn

    return decorator
