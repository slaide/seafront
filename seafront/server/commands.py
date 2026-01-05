import asyncio
import typing as tp
from enum import Enum

import json5
import numpy as np
import seaconfig as sc
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from seafront.config.basics import GlobalConfigHandler
from seafront.config.handles import ProtocolConfig
from seafront.hardware import microcontroller as mc
from seafront.hardware.adapter import AdapterState, Position
from seafront.hardware.forbidden_areas import ForbiddenAreaList

from ..logger import logger


class ImageStoreEntry(BaseModel):
    """utility class to store camera images with some metadata"""

    _img: np.ndarray = PrivateAttr(...)  # type:ignore

    pixel_format: str

    info: "ImageStoreInfo"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def bit_depth(self) -> int:
        match self.pixel_format.lower():
            case "mono8":
                return 8
            case "mono10":
                return 10
            case "mono12":
                return 12
            case "mono14":
                return 14
            case "mono16":
                return 16
            case _unknown:
                error_internal(detail=f"unexpected pixel format {_unknown}")


# wrap error cases in http response models


class InternalErrorModel(BaseModel):
    detail: str


class ConflictErrorDetail(BaseModel):
    message: str
    busy_reasons: list[str]


class ConflictErrorModel(BaseModel):
    detail: ConflictErrorDetail


def error_internal(detail: str) -> tp.NoReturn:
    """raise an HTTPException with specified detail"""
    logger.debug(f"error_internal - {detail=}")

    import faulthandler

    faulthandler.dump_traceback(all_threads=True)

    raise HTTPException(status_code=500, detail=detail)


def error_microscope_busy(busy_reasons: list[str]) -> tp.NoReturn:
    """raise an HTTPException for microscope busy with reasons"""
    logger.debug(f"error_microscope_busy - {busy_reasons=}")

    raise HTTPException(
        status_code=409,
        detail={
            "message": "microscope is busy",
            "busy_reasons": busy_reasons,
        }
    )


def positionIsForbidden(
    x_mm: float,
    y_mm: float,
    forbidden_areas: ForbiddenAreaList | None = None,
) -> tuple[bool, str]:
    """
    Check if a position is in a forbidden area, as indicated by global config.

    Args:
        x_mm: X coordinate in mm
        y_mm: Y coordinate in mm
        forbidden_areas: Pre-parsed ForbiddenAreaList to avoid repeated parsing. If None, will be parsed from config.

    Returns:
        Tuple of (is_forbidden, error_message). error_message is empty if position is allowed.
    """
    # Use provided forbidden areas or parse from config
    if forbidden_areas is None:
        g_config = GlobalConfigHandler.get_dict()
        forbidden_areas_entry = g_config.get(ProtocolConfig.FORBIDDEN_AREAS.value)

        # If no forbidden areas config is found, allow the movement
        if forbidden_areas_entry is None:
            return False, ""

        forbidden_areas_str = forbidden_areas_entry.value
        if not isinstance(forbidden_areas_str, str):
            logger.warning("forbidden_areas entry is not a string, allowing movement")
            return False, ""

        data = json5.loads(forbidden_areas_str)
        forbidden_areas = ForbiddenAreaList.model_validate({"areas": data})

    # Check if movement is safe
    is_safe, conflicting_area = forbidden_areas.is_movement_safe(x_mm, y_mm)

    if not is_safe and conflicting_area is not None:
        reason_text = f" ({conflicting_area.reason})" if conflicting_area.reason else ""
        error_msg = f"Movement to ({x_mm:.1f}, {y_mm:.1f}) mm is forbidden - conflicts with area '{conflicting_area.name}'{reason_text}"
        return True, error_msg

    return False, ""


# command base class


# BaseCommand is a generic Pydantic model
class BaseCommand[T]:
    """

    Example
    ```
    # Example return value classes
    class MoveCommandResult:
        def __init__(self, success: bool):
            self.success = success

    class StopCommandResult:
        def __init__(self, message: str):
            self.message = message

    # Command classes inheriting from BaseCommand
    class MoveCommand(BaseCommand[MoveCommandResult]):
        _ReturnValue = MoveCommandResult

    class StopCommand(BaseCommand[StopCommandResult]):
        _ReturnValue = StopCommandResult
    ```
    """

    pass  # _ReturnValue: Type[T] = PrivateAttr(...)


# input and output parameter models (for server i/o and internal use)


class MC_getLastPosition(BaseModel, BaseCommand[mc.Position]):
    """command class to retrieve core.mc.get_last_position"""

    _ReturnValue: type = PrivateAttr(default=mc.Position)


class SitePosition(BaseModel):
    """
    location of an imaging site on the plate, including well information.

    the well name and site indices are included to keep track of the sites per well that have been imaged.

    without well type and well grid config, the actual site position on the plate cannot be calculated, and including that information
    here seems superfluous and wasteful, so the offset from the well center (to resolve the grid config) and the position on the plate
    (to resolve the plate layout) are included instead.
    """

    well_name: str = Field(
        ..., title="Well Name", description="name of the well that this site is inside of"
    )
    "name of the well that this site is inside of"

    site_x: int = Field(
        ..., title="Site X index", description="x site index in well grid config, 0-indexed"
    )
    "x site index in well grid config, 0-indexed"
    site_y: int = Field(
        ..., title="Site Y index", description="y site index in well grid config, 0-indexed"
    )
    "y site index in well grid config, 0-indexed"
    site_z: int = Field(
        ..., title="Site Z index", description="z site index in well grid config, 0-indexed"
    )
    "z site index in well grid config, 0-indexed"

    x_offset_mm: float = Field(
        ...,
        title="Site X Offset [mm]",
        description="site offset from the center of the well, in x, in mm",
    )
    "site offset from the center of the well, in x, in mm"
    y_offset_mm: float = Field(
        ...,
        title="Site Y Offset [mm]",
        description="site offset from the center of the well, in y, in mm",
    )
    "site offset from the center of the well, in y, in mm"
    z_offset_mm: float = Field(
        ...,
        title="Site Z Offset [mm]",
        description="site offset from the center of the well, in z, in mm",
    )
    "site offset from the center of the well, in z, in mm"

    position: Position = Field(..., description="position of the site on the plate")
    "position of the site on the plate"


class ImageStoreInfo(BaseModel):
    """contains information about an image that has been acquired and stored"""

    channel: sc.AcquisitionChannelConfig
    width_px: int
    height_px: int
    timestamp: float
    "time when the image was taken (finish time)"

    position: SitePosition

    storage_path: str | None = Field(default=None)
    "storage path (may be local filesystem, or object storage)"


class ConfigFileInfo(BaseModel):
    filename: str
    project_name: str
    plate_name: str
    timestamp: str | None
    comment: str | None
    cell_line: str
    plate_type: sc.Wellplate


class ConfigListResponse(BaseModel):
    configs: list[ConfigFileInfo]


class CoreCurrentState(BaseModel):
    adapter_state: AdapterState
    latest_imgs: dict[str, ImageStoreInfo]
    current_acquisition_id: str | None
    is_streaming: bool
    "True if live streaming/acquisition is currently active"
    is_busy: bool
    "True if the microscope is currently busy and cannot accept new commands"
    busy_reasons: list[str] = Field(default_factory=list)
    "Stack of reasons explaining why the microscope is busy (useful for debugging)"
    last_acquisition_error: str | None = None
    "Last acquisition error message for display in GUI"
    last_acquisition_error_timestamp: str | None = None
    "ISO timestamp of the last acquisition error"
    microscope_name: str
    "Name of the currently connected microscope configuration"


class BasicSuccessResponse(BaseModel):
    "indicates that something has succeeded without returning any value"

    pass


class ConfigFetchResponse(BaseModel):
    file: sc.AcquisitionConfig


class LaserAutofocusCalibrationData(BaseModel):
    um_per_px: float
    x_reference: float

    calibration_position: Position


class LaserAutofocusCalibrationResponse(BaseModel):
    calibration_data: LaserAutofocusCalibrationData


class LaserAutofocusCalibrate(BaseModel, BaseCommand[LaserAutofocusCalibrationResponse]):
    """
    set current position as laser autofocus reference

    calculates the conversion factor between pixels and micrometers, and sets the reference for the laser autofocus signal

    the calibration process takes dozens of measurements of the laser autofocus signal at known z positions.
    then it calculates the positions of the dots, and tracks them over time. this is expected to observe two dots,
    at constant distance to each other. one dot may only be visible for a subrange of the total z range, which is
    expected.
    one dot is known to be always visible, so its trajectory is used as reference data.

    measuring the actual offset at an unknown z then locates the dot[s] on the sensor, and uses the position of the dot
    that is known to be always visible to calculate the approximate z offset, based on the reference measurements.
    """

    _ReturnValue: type = PrivateAttr(default=LaserAutofocusCalibrationResponse)


class AcquisitionCommand(str, Enum):
    CANCEL = "cancel"


class AcquisitionStartResponse(BaseModel):
    "indicates that an acquisition has been started"

    acquisition_id: str


class AcquisitionProgressStatus(BaseModel):
    # measureable progress
    current_num_images: int
    time_since_start_s: float
    start_time_iso: str
    current_storage_usage_GB: float

    # estimated completion time information
    # estimation may be more complex than linear interpolation, hence done on server side
    estimated_remaining_time_s: float | None

    # last image that was acquired
    last_image: ImageStoreInfo | None

    # per-timepoint timing (for time series acquisitions)
    # frontend can compare estimated_timepoint_duration_s against delta_t from config
    # to warn user if imaging takes longer than the scheduled interval
    current_timepoint: int | None = None
    total_timepoints: int | None = None
    estimated_timepoint_duration_s: float | None = None


class AcquisitionMetaInformation(BaseModel):
    total_num_images: int
    max_storage_size_images_GB: float


class AcquisitionStatusStage(str, Enum):
    RUNNING = "running"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    CRASHED = "crashed"
    SCHEDULED = "scheduled"


class AcquisitionStatusOut(BaseModel):
    """acquisition thread message out"""

    acquisition_id: str
    acquisition_status: AcquisitionStatusStage
    acquisition_progress: AcquisitionProgressStatus

    # some meta information about the acquisition, derived from configuration file
    # i.e. this is not updated during acquisition
    acquisition_meta_information: AcquisitionMetaInformation

    acquisition_config: sc.AcquisitionConfig

    message: str


class AcquisitionStatus(BaseModel):
    acquisition_id: str = Field(
        ..., title="Acquition Identifier", description="unique identifier for this acquisition"
    )
    "unique identifier for this acquisition"

    queue_in: asyncio.Queue[AcquisitionCommand]
    """queue to send messages to the thread"""
    queue_out: asyncio.Queue[AcquisitionStatusOut]
    """queue to receive messages from the thread"""

    last_status: AcquisitionStatusOut | None
    thread_is_running: bool

    model_config = ConfigDict(arbitrary_types_allowed=True)


class HardwareCapabilitiesResponse(BaseModel):
    wellplate_types: list[sc.Wellplate] = Field(
        ...,
        title="Wellplate Types",
        description="list of wellplates with calibration data present to be used on this system",
    )
    "list of wellplates with calibration data present to be used on this system"
    main_camera_imaging_channels: list[sc.AcquisitionChannelConfig] = Field(
        ...,
        title="Main Camera Imaging Channels",
        description="list of imaging channels that the microscope is capable of",
    )
    "list of imaging channels that the microscope is capable of"
    hardware_limits: dict[str, dict[str, float | int]] = Field(
        ...,
        title="Hardware Limits",
        description="numerical limits of configurable parameters based on actual hardware capabilities",
    )
    "numerical limits of configurable parameters based on actual hardware capabilities"


class LoadingPositionEnter(BaseModel, BaseCommand[BasicSuccessResponse]):
    """
    enter loading position

    enters the stage loading position, and remains there until leave loading position command is executed.
    """

    _ReturnValue: type = PrivateAttr(default=BasicSuccessResponse)


class LoadingPositionLeave(BaseModel, BaseCommand[BasicSuccessResponse]):
    """
    leave loading position

    leaves the stage loading position, and moves to a default position on the plate (this position may not be inside a well, so no cells may be visibile at this time!)
    """

    _ReturnValue: type = PrivateAttr(default=BasicSuccessResponse)


class ImageAcquiredResponse(BaseModel):
    """
    indicates that an image has been acquired

    acquired image data is only present for internal use.
    """

    _img: np.ndarray = PrivateAttr(...)  # type:ignore
    _cambits: int = PrivateAttr(...)  # type:ignore

    model_config = ConfigDict(arbitrary_types_allowed=True)


class StreamingStartedResponse(BaseModel):
    channel: sc.AcquisitionChannelConfig = Field(
        ..., title="Channel", description="contains channel configuration used to start a stream"
    )
    "contains channel configuration used to start a stream"


class ChannelStreamBegin(BaseModel, BaseCommand[StreamingStartedResponse]):
    """
    the callback is called on every image, and must return true if the acquisition should stop
    the callback is called with the value True if it is requested to stop (it should then also return True)
    """

    channel: sc.AcquisitionChannelConfig = Field(
        ..., title="Channel", description="channel configuration to use for streaming"
    )
    "channel configuration to use for streaming"

    machine_config: list[sc.ConfigItem] = Field(default_factory=list)

    _ReturnValue: type = PrivateAttr(default=StreamingStartedResponse)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ChannelStreamEnd(BaseModel, BaseCommand[BasicSuccessResponse]):
    channel: sc.AcquisitionChannelConfig

    machine_config: list[sc.ConfigItem] = Field(default_factory=list)

    _ReturnValue: type = PrivateAttr(default=BasicSuccessResponse)


class ChannelSnapshot(BaseModel, BaseCommand[ImageAcquiredResponse]):
    channel: sc.AcquisitionChannelConfig

    machine_config: list[sc.ConfigItem] = Field(default_factory=list)

    _ReturnValue: type = PrivateAttr(default=ImageAcquiredResponse)


class ChannelSnapSelectionResult(BaseModel):
    channel_handles: list[str]
    _images: dict[str, np.ndarray] = PrivateAttr(default_factory=dict)


class ChannelSnapProgressiveStatus(BaseModel):
    """Progressive channel snap status update"""

    channel_handle: str
    channel_name: str
    status: tp.Literal["starting", "completed", "error", "finished"]
    total_channels: int
    completed_channels: int
    message: str = ""
    error_detail: str | None = None


class MoveByResult(BaseModel):
    axis: str
    moved_by_mm: float


class MoveBy(BaseModel, BaseCommand[MoveByResult]):
    """
    move stage by some distance

    moves stage by some distance, relative to its current location. may introduce minor errors if used frequently without intermediate absolute moveto
    """

    axis: tp.Literal["x", "y", "z"]
    distance_mm: float

    _ReturnValue: type = PrivateAttr(default=MoveByResult)


class MoveTo(BaseModel, BaseCommand[BasicSuccessResponse]):
    """
    move to target coordinates

    any of the arguments may be None, in which case the corresponding axis is not moved

    these coordinates are internally adjusted to take the calibration into account
    """

    x_mm: float | None = None
    y_mm: float | None = None
    z_mm: float | None = None

    _ReturnValue: type = PrivateAttr(default=BasicSuccessResponse)


class MoveToWell(BaseModel, BaseCommand[BasicSuccessResponse]):
    """
    move to well

    moves to center of target well (centers field of view). requires specification of plate type because measurements of plate types vary by manufacturer.
    """

    plate_type: sc.Wellplate
    well_name: str

    _ReturnValue: type = PrivateAttr(default=BasicSuccessResponse)


class AutofocusMeasureDisplacementResult(BaseModel):
    displacement_um: float


class AutofocusMeasureDisplacement(BaseModel, BaseCommand[AutofocusMeasureDisplacementResult]):
    """
    measure current displacement from reference position
    """

    config_file: sc.AcquisitionConfig = Field(
        ..., title="Config File", description="config file (file contents, not path)"
    )
    "config file (file contents, not path)"
    override_num_images: int | None = Field(
        default=None,
        title="Override Number of Images",
        description="override the number of images used to measure displacement (internal parameter, use with care)",
    )
    "override the number of images used to measure displacement (internal parameter, use with care)"

    _ReturnValue: type = PrivateAttr(default=AutofocusMeasureDisplacementResult)


class AutofocusSnapResult(BaseModel):
    width_px: int = Field(..., title="Image Width [pixels]", description="image height, in pixels")
    "image width, in pixels"
    height_px: int = Field(
        ..., title="Image Height [pixels]", description="image height, in pixels"
    )
    "image height, in pixels"

    _img: np.ndarray = PrivateAttr(...)
    "image that was snapped"
    _channel: sc.AcquisitionChannelConfig = PrivateAttr(...)
    "channel config used to take the snapshot"

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AutofocusSnap(BaseModel, BaseCommand[AutofocusSnapResult]):
    """
    snap a laser autofocus image
    """

    exposure_time_ms: float = 5
    analog_gain: float = 10
    turn_laser_on: bool = True
    turn_laser_off: bool = True

    _ReturnValue: type = PrivateAttr(default=AutofocusSnapResult)


class AutofocusLaserWarmup(BaseModel, BaseCommand[BasicSuccessResponse]):
    """
    warm up the laser for autofocus

    sometimes the laser needs to stay on for a little bit before it can be used (the short on-time of ca. 5ms is
    sometimes not enough to turn the laser on properly without a recent warmup)
    """

    warmup_time_s: float = 0.5

    _ReturnValue: type = PrivateAttr(default=BasicSuccessResponse)


class IlluminationEndAll(BaseModel, BaseCommand[BasicSuccessResponse]):
    """
    turn off all illumination sources

    rather, send signal to do so. this function does not check if any are on before sending the signals.

    this function is NOT for regular operation!
    it does not verify that the microscope is not in any acquisition state
    """

    _ReturnValue: type = PrivateAttr(default=BasicSuccessResponse)


class AutofocusApproachTargetDisplacementResult(BaseModel):
    """
    final distance to target offset is not zero, and may have taken a variable number of moves, up to the limit. return those numbers here.

    reached_threshold is set to true if the system has determined that it cannot get close to the target offset
    """

    num_compensating_moves: int
    uncompensated_offset_mm: float
    reached_threshold: bool


class AutofocusApproachTargetDisplacement(
    BaseModel, BaseCommand[AutofocusApproachTargetDisplacementResult]
):
    """
    move to target offset

    measure current offset from reference position, then moves to target offset (may perform more than one physical move to reach target position)
    """

    target_offset_um: float
    config_file: sc.AcquisitionConfig
    max_num_reps: int = 3
    pre_approach_refz: bool = True

    _ReturnValue: type = PrivateAttr(default=AutofocusApproachTargetDisplacementResult)


class EstablishHardwareConnection(BaseModel, BaseCommand[BasicSuccessResponse]):
    """
    explicitely request a hardware connection to be established

    this should immediately precede any command that requires an established connection.
    Note that this command may still fail.

    note: the hardware_identifier field is reserved for future use and currently unused.
    """

    hardware_identifier: str = Field(default="")

    _ReturnValue: type = PrivateAttr(default=BasicSuccessResponse)
