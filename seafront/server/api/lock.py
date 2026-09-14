"""
The device lease: one client at a time across requests.

None of these routes needs a lease to be called - that would be a deadlock - but acquire and
release act on one somebody else may hold, so they can still answer 423. See
seafront/server/lock.py for the lease itself.
"""

from seafront.server.api.requests import (
    LockAcquireRequest,
    LockBreakRequest,
    LockReleaseRequest,
    LockStatusRequest,
)
from seafront.server.api.routing import LEASE_ADMIN, READ, Api, CoreDep
from seafront.server.api.tags import RouteTag
from seafront.server.commands import (
    BasicSuccessResponse,
    LockAcquireResponse,
    LockStatusResponse,
)

api = Api(tags=[RouteTag.LOCK.value])


@api.post("/api/lock/acquire", policy=LEASE_ADMIN)
async def acquire(body: LockAcquireRequest, core: CoreDep) -> LockAcquireResponse:
    """
    take the device lease

    a lease is required for every hardware command (moves, snaps, streaming, acquisition
    start, machine config flush, ...): without one they are refused with 423, and while you
    hold one they are refused for everybody else. status reads and cancel are never refused.
    the lease expires `ttl_s` seconds after the last request that carried the token, but never
    while a command is running on the hardware. calling this with your own token renews (and
    re-labels) the lease.
    """
    return core.lock_acquire(owner=body.owner, ttl_s=body.ttl_s, lock_token=body.lock_token)


@api.post("/api/lock/release", policy=LEASE_ADMIN)
async def release(body: LockReleaseRequest, core: CoreDep) -> BasicSuccessResponse:
    """
    give up the device lease

    idempotent when nothing is held; 423 if someone else holds it (use /api/lock/break).
    """
    return core.lock_release(lock_token=body.lock_token)


@api.post("/api/lock/status", policy=READ)
async def status(body: LockStatusRequest, core: CoreDep) -> LockStatusResponse:
    """who holds the device lease, if anyone"""
    return core.lock_status(lock_token=body.lock_token)


@api.post("/api/lock/break", policy=READ)
async def break_lock(body: LockBreakRequest, core: CoreDep) -> BasicSuccessResponse:
    """
    take the device lease away from whoever holds it

    open to anyone (it is a coordination tool, not a permission system). does not stop a
    command already running on the hardware.
    """
    return core.lock_break(reason=body.reason)
