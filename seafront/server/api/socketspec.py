"""
How a websocket declares itself.

The HTTP routes get their schema for free: FastAPI reads the models they are annotated with.
Sockets have no equivalent, so each one states its frames here instead - what the client may
send, what the server sends back, and whether it needs the device lease - right next to the
handler, the same way a route states its Policy. `server/asyncapi.py` turns these
declarations into the published document; nothing about a socket is described twice.
"""

import typing as tp
from dataclasses import dataclass, field

from fastapi import APIRouter
from pydantic import BaseModel


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


def socket(
    router: APIRouter,
    path: str,
    *,
    requires_lease: bool = False,
    receive: list[Frame],
    send: list[Frame],
    close_codes: dict[int, str] | None = None,
):
    """register a websocket handler and record what travels over it"""

    def decorator(fn):
        doc = [line.strip() for line in (fn.__doc__ or "").splitlines()]
        doc = [line for line in doc if line]
        SOCKETS.append(
            SocketSpec(
                path=path,
                summary=doc[0] if doc else "",
                description="\n".join(doc[1:]),
                requires_lease=requires_lease,
                receive=receive,
                send=send,
                close_codes=close_codes or {},
            )
        )
        return router.websocket(path)(fn)

    return decorator
