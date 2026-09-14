"""
Things the microscope does right now.

Every handler here reads top to bottom: admit the request (done by the decorator), turn the
body into the domain command, run it on the worker, and - where the command produced an
image - store it. There is no callback indirection: what happens after a command is written
after the command.
"""

from seafront.server.api.envelope import command_of
from seafront.server.api.requests import (
    AutofocusApproachTargetRequest,
    AutofocusLaserWarmupRequest,
    AutofocusMeasureDisplacementRequest,
    AutofocusSnapRequest,
    CalibrateStageXYHereRequest,
    CancelCurrentRequest,
    EstablishHardwareConnectionRequest,
    IlluminationEndAllRequest,
    LaserAutofocusCalibrateRequest,
    LoadingPositionEnterRequest,
    LoadingPositionLeaveRequest,
    MachineConfigFlushRequest,
    MoveByRequest,
    MoveToRequest,
    MoveToWellRequest,
    SnapChannelRequest,
    SnapSelectedChannelsRequest,
    StreamChannelBeginRequest,
    StreamChannelEndRequest,
)
from seafront.server.api.routing import (
    HARDWARE,
    HARDWARE_IDLE,
    HARDWARE_NOT_ACQUIRING,
    HARDWARE_NOT_STREAMING,
    READ,
    Api,
    CoreDep,
    Policy,
)
from seafront.server.api.tags import RouteTag
from seafront.server.commands import (
    AutofocusApproachTargetDisplacement,
    AutofocusApproachTargetDisplacementResult,
    AutofocusLaserWarmup,
    AutofocusMeasureDisplacement,
    AutofocusMeasureDisplacementResult,
    AutofocusSnap,
    AutofocusSnapResult,
    BasicSuccessResponse,
    ChannelSnapSelectionResult,
    ChannelSnapshot,
    ChannelStreamBegin,
    ChannelStreamEnd,
    EstablishHardwareConnection,
    IlluminationEndAll,
    ImageAcquiredResponse,
    LaserAutofocusCalibrate,
    LaserAutofocusCalibrationResponse,
    LoadingPositionEnter,
    LoadingPositionLeave,
    MachineConfigFlushResponse,
    MoveBy,
    MoveByResult,
    MoveTo,
    MoveToWell,
    StreamingStartedResponse,
)

api = Api(tags=[RouteTag.ACTIONS.value])


# -- stage ---------------------------------------------------------------------------------


@api.post("/api/action/move_by", policy=HARDWARE)
async def move_by(body: MoveByRequest, core: CoreDep) -> MoveByResult:
    """move backwards or forwards by some distance along one axis"""
    return await core.run(command_of(body, MoveBy))


@api.post("/api/action/move_to", policy=HARDWARE)
async def move_to(body: MoveToRequest, core: CoreDep) -> BasicSuccessResponse:
    """
    move to target coordinates

    any of the arguments may be None, in which case the corresponding axis is not moved
    these coordinates are internally adjusted to take the calibration into account
    """
    return await core.run(command_of(body, MoveTo))


@api.post("/api/action/move_to_well", policy=HARDWARE)
async def move_to_well(body: MoveToWellRequest, core: CoreDep) -> BasicSuccessResponse:
    """move to well"""
    return await core.run(command_of(body, MoveToWell))


@api.post("/api/action/enter_loading_position", policy=HARDWARE_NOT_ACQUIRING)
async def enter_loading_position(
    body: LoadingPositionEnterRequest, core: CoreDep
) -> BasicSuccessResponse:
    """enter loading position"""
    return await core.run(command_of(body, LoadingPositionEnter))


@api.post("/api/action/leave_loading_position", policy=HARDWARE_NOT_ACQUIRING)
async def leave_loading_position(
    body: LoadingPositionLeaveRequest, core: CoreDep
) -> BasicSuccessResponse:
    """leave loading position"""
    return await core.run(command_of(body, LoadingPositionLeave))


@api.post("/api/action/calibrate_stage_xy_here", policy=HARDWARE_NOT_ACQUIRING)
async def calibrate_stage_xy_here(
    body: CalibrateStageXYHereRequest, core: CoreDep
) -> BasicSuccessResponse:
    """
    set current xy position as reference

    set current xy position as top left corner of B2, which is used as reference to calculate
    all other positions on a plate.

    this WILL lead to hardware damage (down the line) if used improperly!
    """
    return await core.calibrate_stage_xy_here(plate_model_id=body.plate_model_id)


# -- imaging -------------------------------------------------------------------------------


@api.post("/api/action/snap_channel", policy=HARDWARE_NOT_STREAMING)
async def snap_channel(body: SnapChannelRequest, core: CoreDep) -> ImageAcquiredResponse:
    """snap an image in the given channel"""
    command = command_of(body, ChannelSnapshot)
    result = await core.run(command)
    await core.store_snapped_image(command, result)
    return result


@api.post("/api/action/snap_selected_channels", policy=HARDWARE_NOT_STREAMING)
async def snap_selected_channels(
    body: SnapSelectedChannelsRequest, core: CoreDep
) -> ChannelSnapSelectionResult:
    """snap an image in each channel the config file selects"""
    return await core.snap_selected_channels(config_file=body.config_file)


@api.post("/api/action/stream_channel_begin", policy=HARDWARE_IDLE)
async def stream_channel_begin(
    body: StreamChannelBeginRequest, core: CoreDep
) -> StreamingStartedResponse:
    """begin streaming in the given channel"""
    command = command_of(body, ChannelStreamBegin)
    result = await core.run(command)
    core.stream_begin(command.channel)
    return result


@api.post("/api/action/stream_channel_end", policy=HARDWARE)
async def stream_channel_end(
    body: StreamChannelEndRequest, core: CoreDep
) -> BasicSuccessResponse:
    """stop streaming"""
    result = await core.run(command_of(body, ChannelStreamEnd))
    core.stream_end()
    return result


@api.post("/api/action/turn_off_all_illumination", policy=HARDWARE)
async def turn_off_all_illumination(
    body: IlluminationEndAllRequest, core: CoreDep
) -> BasicSuccessResponse:
    """turn off all illumination sources"""
    return await core.run(command_of(body, IlluminationEndAll))


# -- laser autofocus -----------------------------------------------------------------------


@api.post("/api/action/snap_reflection_autofocus", policy=HARDWARE_NOT_STREAMING)
async def snap_reflection_autofocus(
    body: AutofocusSnapRequest, core: CoreDep
) -> AutofocusSnapResult:
    """snap an image on the laser autofocus camera"""
    result = await core.run(command_of(body, AutofocusSnap))
    await core.store_laseraf_image(result)
    return result


@api.post("/api/action/laser_autofocus_measure_displacement", policy=HARDWARE_NOT_STREAMING)
async def laser_autofocus_measure_displacement(
    body: AutofocusMeasureDisplacementRequest, core: CoreDep
) -> AutofocusMeasureDisplacementResult:
    """measure the current displacement from the reference plane"""
    return await core.run(command_of(body, AutofocusMeasureDisplacement))


@api.post("/api/action/laser_autofocus_move_to_target_offset", policy=HARDWARE_NOT_STREAMING)
async def laser_autofocus_move_to_target_offset(
    body: AutofocusApproachTargetRequest, core: CoreDep
) -> AutofocusApproachTargetDisplacementResult:
    """move to a target offset from the reference plane"""
    return await core.run(command_of(body, AutofocusApproachTargetDisplacement))


@api.post("/api/action/laser_autofocus_calibrate", policy=HARDWARE_NOT_STREAMING)
async def laser_autofocus_calibrate(
    body: LaserAutofocusCalibrateRequest, core: CoreDep
) -> LaserAutofocusCalibrationResponse:
    """calibrate the laser autofocus system"""
    return await core.run(command_of(body, LaserAutofocusCalibrate))


@api.post("/api/action/laser_autofocus_warm_up_laser", policy=HARDWARE_NOT_STREAMING)
async def laser_autofocus_warm_up_laser(
    body: AutofocusLaserWarmupRequest, core: CoreDep
) -> BasicSuccessResponse:
    """warm up the autofocus laser"""
    return await core.run(command_of(body, AutofocusLaserWarmup))


# -- server-side state ---------------------------------------------------------------------


@api.post("/api/action/machine_config_flush", policy=HARDWARE, tags=[])
async def machine_config_flush(
    body: MachineConfigFlushRequest, core: CoreDep
) -> MachineConfigFlushResponse:
    """write client-side machine config values into the running server config"""
    return core.machine_config_flush(
        machine_config=body.machine_config, privileged=body.privileged
    )


@api.post("/api/action/establish_hardware_connection", policy=HARDWARE)
async def establish_hardware_connection(
    body: EstablishHardwareConnectionRequest, core: CoreDep
) -> BasicSuccessResponse:
    """establish the connection to the microscope hardware"""
    return await core.run(command_of(body, EstablishHardwareConnection))


# Cancel hatch: cancel whatever command is currently on the worker. Never gated - it is the
# escape from a stuck or long command, so it must work without a lease and in every state.
@api.post("/api/action/cancel_current", policy=READ)
async def cancel_current(body: CancelCurrentRequest, core: CoreDep) -> BasicSuccessResponse:
    """cancel the command currently running on the hardware"""
    return await core.cancel_current_command()


# `Policy` is re-exported for routes elsewhere that need a one-off combination
__all__ = ["Policy", "api"]
