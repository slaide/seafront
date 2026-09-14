"""
One place where an HTTP request becomes a call into the Core.

Every /api route is registered through `Api.post`, which states the route's policy - does it
drive the hardware, may it run during an acquisition, during a live stream - right next to
the handler. That same policy object is what enforces the checks at request time *and* what
is published in the openapi schema, so the documented contract and the enforced one cannot
drift apart.

What a request passes through, in order:

    body        422   the body does not match the route's request model (FastAPI)
    state       400   the microscope is acquiring or streaming, and this route may not run then
    routing     421   the request names a different microscope than this server drives
    lease       423   the caller does not hold the device lease (hardware routes only)
    handler           a Core method, or a command submitted to the worker
    worker      409   a command is already in flight (reject-on-busy, raised by Core.run)

`admit` reads the envelope off the already-validated body, so a malformed request is refused
before any of this is consulted.
"""

import typing as tp
from dataclasses import dataclass
from functools import wraps

from fastapi import APIRouter, Depends
from starlette.requests import HTTPConnection
from pydantic import BaseModel

from seafront.server.api.envelope import Envelope
from seafront.server.commands import (
    ConflictErrorModel,
    InternalErrorModel,
    LockedErrorModel,
    error_device_locked,
)

from seafront.server.core import Core


class RefusalModel(BaseModel):
    """
    400 and 421: the request was understood, but this server will not run it as asked.

    Distinct from the 409/423/500 models, which carry a structured `detail`. This one is flat
    because that is the shape these two refusals have always had on the wire.
    """

    status: tp.Literal["error"] = "error"
    message: str


class RefusedError(Exception):
    """a refusal that renders as RefusalModel; see the handler installed in server/app.py"""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = RefusalModel(message=message)


def get_core(connection: HTTPConnection) -> Core:
    "the Core this server drives, attached to the app at startup (works for sockets too)"
    return connection.app.state.core


CoreDep = tp.Annotated[Core, Depends(get_core)]


@dataclass(frozen=True, slots=True)
class Policy:
    """
    what a route is allowed to do, and what it demands of the caller

    requires_lease: the route drives the hardware, so the caller must hold the device lease
        and send it as `lock_token`. Default on, so a new route is gated until someone
        decides otherwise. Opt out for reads, for cancel (the safety hatch, always allowed)
        and for the lease routes themselves.
    may_refuse_locked: the route does not need the lease but can still answer 423 (acquiring
        or releasing one somebody else holds). Purely so the schema documents the 423.
    """

    requires_lease: bool = True
    may_refuse_locked: bool = False
    allow_while_acquisition_is_running: bool = True
    allow_while_streaming: bool = True


# the policies the route table is written in. Named rather than spelled out at each route, so
# the whole access model is these six lines.

HARDWARE = Policy()
"drives the microscope; safe to interleave with a run or a live stream"

HARDWARE_NOT_STREAMING = Policy(allow_while_streaming=False)
"drives the microscope and needs the camera to itself"

HARDWARE_NOT_ACQUIRING = Policy(allow_while_acquisition_is_running=False)
"drives the microscope in a way an acquisition in progress could not survive"

HARDWARE_IDLE = Policy(allow_while_acquisition_is_running=False, allow_while_streaming=False)
"drives the microscope and needs it doing nothing else"

READ = Policy(requires_lease=False)
"answers from server state: open to anyone, including a client watching someone else's run"

LEASE_ADMIN = Policy(requires_lease=False, may_refuse_locked=True)
"acts on the lease itself, so it does not need one but can still refuse 423"


def admit(core: Core, body: Envelope, policy: Policy) -> None:
    """
    Let a validated request through, or raise the refusal that stops it.

    Kept out of Core on purpose: Core knows about the microscope, not about HTTP.
    """
    if (not policy.allow_while_acquisition_is_running) and core.acquisition_is_running:
        raise RefusedError(400, "cannot run this command while acquisition is running")

    if (not policy.allow_while_streaming) and core.microscope.stream_callback is not None:
        raise RefusedError(400, "cannot run this command while live streaming is active")

    # an empty target_device is the bypass a client uses before it knows the device name
    if body.target_device and body.target_device != core.microscope_name:
        raise RefusedError(
            421,
            f"target_device mismatch: request targets '{body.target_device}' but this server is '{core.microscope_name}'",
        )

    # Every request goes through check() so that a holder's lease is renewed by whatever it
    # does, reads included; hardware routes additionally require the decision to come back
    # permitted - no lease at all is a refusal, not a free pass.
    decision = core.lock.check(body.lock_token)
    if policy.requires_lease and not decision.permitted:
        error_device_locked(decision)


class Api:
    """an APIRouter whose routes carry a Policy, and enforce it"""

    def __init__(self, tags: list[str] | None = None) -> None:
        self.router = APIRouter()
        self.tags = tags or []

    def post(
        self,
        path: str,
        *,
        policy: Policy = HARDWARE,
        tags: list[str] | None = None,
        summary: str | None = None,
    ):
        """
        register a POST handler of the shape `(body: <Envelope subclass>, core: CoreDep)`

        The handler runs only once the policy has admitted the request.
        """

        def decorator(fn):
            hints = tp.get_type_hints(fn)
            if "body" not in hints or "core" not in hints:
                raise TypeError(f"{path}: handler must take `body` and `core` ({fn.__name__})")
            if not issubclass(hints["body"], Envelope):
                raise TypeError(f"{path}: `body` must be an Envelope subclass ({fn.__name__})")

            @wraps(fn)
            async def gated(**kwargs: tp.Any):
                core: Core = kwargs["core"]

                admit(core, kwargs["body"], policy)
                result = await fn(**kwargs)
                core.mark_request_served()

                # the response says which microscope answered, so a client talking to several
                # of them cannot mix up the replies
                if isinstance(result, BaseModel) and "microscope_name" in type(result).model_fields:
                    result = result.model_copy(update={"microscope_name": core.microscope_name})

                return result

            responses: dict[int | str, dict[str, tp.Any]] = {
                400: {"model": RefusalModel, "description": "refused in the current state"},
                409: {
                    "model": ConflictErrorModel,
                    "description": "conflict (e.g., microscope busy)",
                },
                421: {"model": RefusalModel, "description": "request names another microscope"},
                500: {"model": InternalErrorModel, "description": "any failure mode"},
            }
            if policy.requires_lease or policy.may_refuse_locked:
                responses[423] = {
                    "model": LockedErrorModel,
                    "description": "device lease missing or held by another client",
                }

            self.router.add_api_route(
                path,
                gated,
                methods=["POST"],
                operation_id=path[1:].replace("/", ".") + ".post",
                summary=summary,
                tags=tags if tags is not None else self.tags,  # type: ignore[arg-type]
                responses=responses,
                # x-requires-device-lease is the machine-readable answer to "is this a
                # hardware write?". Clients read it to know what needs a lease;
                # tests/test_lock_enforcement.py reads it to prove every write route
                # actually refuses without one.
                openapi_extra={"x-requires-device-lease": policy.requires_lease},
            )
            return fn

        return decorator
