"""
Reads: what the microscope is doing, what it can do, and whether the server is alive.

None of these drive the hardware, so none needs the device lease - a client watching a run
somebody else owns can call all of them.
"""

import asyncio
import time

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from seaconfig.config_item import ConfigItem

from seafront.debug_introspect import format_stack_dump
from seafront.server.api.requests import (
    CurrentStateRequest,
    HardwareCapabilitiesRequest,
    MachineDefaultsRequest,
)
from seafront.server.api.routing import READ, Api, CoreDep
from seafront.server.api.tags import RouteTag
from seafront.server.commands import (
    CoreCurrentState,
    HardwareCapabilitiesResponse,
    HealthResponse,
)

api = Api(tags=[])
router = APIRouter()
"routes that carry no envelope, so they do not go through the policy gate"


@api.post("/api/get_info/current_state", policy=READ)
async def current_state(body: CurrentStateRequest, core: CoreDep) -> CoreCurrentState:
    """get current state of the microscope"""
    return await core.get_current_state()


@api.post("/api/get_features/hardware_capabilities", policy=READ)
async def hardware_capabilities(
    body: HardwareCapabilitiesRequest, core: CoreDep
) -> HardwareCapabilitiesResponse:
    """get a list of all the hardware capabilities of this microscope"""
    return core.get_hardware_capabilities()


@api.post("/api/get_features/machine_defaults", policy=READ)
async def machine_defaults(
    body: MachineDefaultsRequest, core: CoreDep
) -> list[ConfigItem]:
    """get a list of all the low level machine settings"""
    return core.get_machine_defaults()


@router.get(
    "/api/health",
    operation_id="api.health.get",
    tags=[RouteTag.INFO.value],
    summary="Health check endpoint for monitoring",
)
async def health(core: CoreDep) -> HealthResponse:
    """
    Health check endpoint for monitoring.

    Returns basic health information without requiring hardware interaction or target_device
    authentication. Always available.
    """
    now = time.time()
    time_since_last_request = None if core.last_request_time is None else now - core.last_request_time

    # health checks count as requests
    core.last_request_time = now

    return HealthResponse(
        microscope_name=core.microscope_name,
        time_since_last_request_s=time_since_last_request,
        server_time=now,
    )


@router.get(
    "/api/debug/stacks",
    operation_id="api.debug.stacks.get",
    tags=[RouteTag.DOCUMENTATION.value],
    summary="Dump all thread/task stacks and hardware lock reasons (debugging)",
    response_class=PlainTextResponse,
)
async def debug_stacks(core: CoreDep) -> str:
    """
    Where every thread and task is currently executing.

    Never takes the hardware lock, so it stays responsive while other request threads are
    wedged on it. Each request runs in its own thread, so this answers even during a hang.
    """
    return await asyncio.to_thread(format_stack_dump, core.microscope, [])
