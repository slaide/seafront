"""Starting, watching and cancelling acquisition runs, and the protocol files they run from."""

from seafront.server.api.requests import (
    AcquisitionCancelRequest,
    AcquisitionEstimateRequest,
    AcquisitionStartRequest,
    AcquisitionStatusRequest,
    ConfigFetchRequest,
    ConfigListRequest,
    ConfigStoreRequest,
)
from seafront.server.api.routing import (
    HARDWARE_NOT_STREAMING,
    READ,
    Api,
    CoreDep,
    Policy,
)
from seafront.server.api.tags import RouteTag
from seafront.server.commands import (
    AcquisitionEstimate,
    AcquisitionStartResponse,
    AcquisitionStatusOut,
    BasicSuccessResponse,
    ConfigFetchResponse,
    ConfigListResponse,
)

api = Api(tags=[RouteTag.ACQUISITION_CONTROLS.value])


@api.post("/api/acquisition/start", policy=HARDWARE_NOT_STREAMING)
async def start(body: AcquisitionStartRequest, core: CoreDep) -> AcquisitionStartResponse:
    """
    start an acquisition

    the acquisition is run in the background, i.e. this command returns after acquisition has
    begun. see /api/acquisition/status for ongoing status of the acquisition.
    """
    return await core.start_acquisition(config_file=body.config_file)


# Cancel is never gated by the lease: stopping a run may be needed to protect the hardware,
# and the client that needs to stop it is not always the one that started it.
@api.post("/api/acquisition/cancel", policy=Policy(requires_lease=False, allow_while_streaming=False))
async def cancel(body: AcquisitionCancelRequest, core: CoreDep) -> BasicSuccessResponse:
    """cancel a running acquisition"""
    return await core.cancel_acquisition(acquisition_id=body.acquisition_id)


@api.post("/api/acquisition/status", policy=READ)
async def status(body: AcquisitionStatusRequest, core: CoreDep) -> AcquisitionStatusOut:
    """get status of an acquisition"""
    return await core.get_acquisition_status(acquisition_id=body.acquisition_id)


@api.post("/api/acquisition/estimate", policy=READ)
async def estimate(body: AcquisitionEstimateRequest, core: CoreDep) -> AcquisitionEstimate:
    """estimate how long an acquisition will take, and how much storage it needs"""
    return await core.get_acquisition_estimate(config_file=body.config_file)


@api.post("/api/acquisition/config_list", policy=READ)
async def config_list(body: ConfigListRequest, core: CoreDep) -> ConfigListResponse:
    """get list of existing config files"""
    return core.get_config_list()


@api.post("/api/acquisition/config_fetch", policy=READ)
async def config_fetch(body: ConfigFetchRequest, core: CoreDep) -> ConfigFetchResponse:
    """get one config file by name"""
    return core.config_fetch(config_file=body.config_file)


@api.post("/api/acquisition/config_store", policy=READ)
async def config_store(body: ConfigStoreRequest, core: CoreDep) -> BasicSuccessResponse:
    """
    store this file locally

    stores the file on the microscope-connected computer for later retrieval.
    """
    return core.config_store(
        config_file=body.config_file,
        filename=body.filename,
        overwrite_on_conflict=body.overwrite_on_conflict,
        comment=body.comment,
    )
