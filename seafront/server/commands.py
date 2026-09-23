import asyncio
import typing as tp
import uuid
from enum import Enum

import json5
import numpy as np
import seaconfig as sc
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from seafront.server.lock import LockDecision, LockReason, LockState

from seafront.config.basics import FilterConfig, GlobalConfigHandler
from seafront.config.handles import ProtocolConfig
from seafront.config.registry import ConfigRegistry
from seafront.hardware.adapter import AdapterState, Position
from seafront.hardware.forbidden_areas import ForbiddenArea, ForbiddenAreaList
from seafront.hardware.health import DeviceHealth
from seafront.hardware.watchdog import Stall
from seafront.hardware.limits import HardwareLimits

from ..hardware.camera.camera import bits_for_pixel_format
from ..logger import logger


class ImageStoreEntry(BaseModel):
    """utility class to store camera images with some metadata"""

    _img: np.ndarray = PrivateAttr(...)  # type:ignore

    pixel_format: str

    info: "ImageStoreInfo"
    capture: "CapturedImageMetadata | None" = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def bit_depth(self) -> int:
        "how many bits of each stored sample the camera actually filled"
        try:
            return bits_for_pixel_format(self.pixel_format)
        except ValueError:
            error_internal(detail=f"unexpected pixel format {self.pixel_format}")


# wrap error cases in http response models


class InternalErrorModel(BaseModel):
    detail: str


class ConflictErrorDetail(BaseModel):
    message: str
    busy_reasons: list[str]


class ConflictErrorModel(BaseModel):
    detail: ConflictErrorDetail


class LockedErrorDetail(BaseModel):
    "423 Locked: the request may not drive the hardware without the device lease"

    message: str
    reason: LockReason
    "`not_held` (nobody holds it: acquire one) or `held_by_other` (the fields below say who)"
    held: bool
    owner: str | None
    "null when nobody holds the lease; always present, like the rest of this detail"
    lock_id: str | None
    expires_at: str | None


class LockedErrorModel(BaseModel):
    detail: LockedErrorDetail


def error_internal(detail: str) -> tp.NoReturn:
    """raise an HTTPException with specified detail"""
    logger.debug(f"error_internal - {detail=}")

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


def error_device_locked(decision: LockDecision) -> tp.NoReturn:
    """
    raise an HTTPException (423 Locked): this request may not drive the hardware

    Either nobody holds the device lease (`not_held` - hardware commands always need one) or
    somebody else does (`held_by_other`). Clients tell the two apart by `detail.reason`: the
    first is fixed by acquiring a lease, the second by waiting or breaking it.
    """
    logger.debug(f"error_device_locked - {decision=}")

    state = decision.state
    raise HTTPException(
        status_code=423,
        detail={
            "message": decision.message,
            "reason": decision.reason,
            "held": state.held,
            "owner": state.owner,
            "lock_id": state.lock_id,
            "expires_at": state.expires_at,
        },
    )


def error_device_in_use(device_type: str, device_id: str) -> tp.NoReturn:
    """raise an HTTPException when a device is already in use by another process"""
    message = f"{device_type} '{device_id}' is already in use by another process"
    logger.critical(f"error_device_in_use - {message}")

    raise HTTPException(
        status_code=503,
        detail={
            "message": message,
            "error_type": "device_in_use",
            "device_type": device_type,
            "device_id": device_id,
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
        try:
            data = ConfigRegistry.get(ProtocolConfig.FORBIDDEN_AREAS).objectvalue
        except KeyError:
            # No forbidden areas configured - allow the movement
            return False, ""

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


class GetCurrentPosition(BaseModel, BaseCommand[Position]):
    """Read the stage position in the calibrated coordinate frame used by MoveTo and the API."""

    _ReturnValue: type = PrivateAttr(default=Position)


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

    image_id: str | None = None
    "opaque identity of a transient image exposed through the latest-image socket"

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
    microscope_name: str = ""
    "name of the device that processed this request"


class CoreCurrentState(BaseModel):
    adapter_state: AdapterState
    latest_imgs: dict[str, ImageStoreInfo]
    current_acquisition_id: str | None
    is_streaming: bool
    "True if live streaming/acquisition is currently active"
    streaming_channel_handle: str | None = None
    "Handle of the channel currently being streamed, if any"
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
    lock: LockState = Field(default_factory=LockState)
    "who holds the device lease, if anyone (see /api/lock/*)"


class LockAcquireResponse(BaseModel):
    "a freshly taken (or renewed) device lease; keep `token` secret, `lock_id` is public"

    token: str
    "send as `lock_token` with every request while you hold the lease"
    lock_id: str
    "public id of this lease; matches CoreCurrentState.lock.lock_id while it is yours"
    owner: str
    expires_at: str
    "ISO-8601 UTC; renewed by every request carrying the token"

    microscope_name: str = ""
    "name of the device that processed this request"


class LockStatusResponse(BaseModel):
    "device lease status plus whether the token you sent is the holder's"

    held: bool
    owner: str | None
    "null when nobody holds the lease; always present, so a client need not tell absent from null"
    lock_id: str | None
    expires_at: str | None
    token_matches: bool
    "True if the `lock_token` in this request is the current lease's token"

    microscope_name: str = ""
    "name of the device that processed this request"


class BasicSuccessResponse(BaseModel):
    "indicates that something has succeeded without returning any value"

    microscope_name: str = ""
    "name of the device that processed this request"


class MachineConfigFlushResponse(BaseModel):
    "outcome of a machine config flush: which handles were written, which were ignored"

    changed: list[str] = Field(default_factory=list)
    "handles whose value was written and differs from before"
    skipped_privileged: list[str] = Field(default_factory=list)
    "privileged handles left untouched because the request was not privileged"
    unknown_handles: list[str] = Field(default_factory=list)
    "handles the server does not know; ignored"
    invalid: dict[str, str] = Field(default_factory=dict)
    "handle -> reason, for values rejected by validation; left untouched"

    microscope_name: str = ""
    "name of the device that processed this request"


class ConfigFetchResponse(BaseModel):
    file: sc.AcquisitionConfig
    microscope_name: str = ""
    "name of the device that processed this request"


class LaserAutofocusCalibrationData(BaseModel):
    um_per_px: float
    x_reference: float
    offset_um: float = 0.0
    "offset correction to zero out error at calibration position"

    calibration_position: Position


class LaserAutofocusCalibrationResponse(BaseModel):
    calibration_data: LaserAutofocusCalibrationData
    microscope_name: str = ""
    "name of the device that processed this request"


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
    microscope_name: str = ""
    "name of the device that processed this request"


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


class AcquisitionMetrics(BaseModel):
    """
    Computed metrics for an acquisition configuration.

    This is the single source of truth for acquisition size calculations,
    used by both the estimate endpoint and the protocol runner.
    """
    num_wells: int
    num_sites: int
    num_channels: int
    num_z_planes_total: int
    num_timepoints: int
    total_num_images: int
    image_size_bytes: int
    max_storage_size_GB: float


def calculate_acquisition_metrics(config: sc.AcquisitionConfig) -> AcquisitionMetrics:
    """
    Calculate acquisition metrics from a config.

    This is the single source of truth for image count and storage calculations.
    Used by microscope.estimate_acquisition() and ProtocolGenerator.

    Args:
        config: The acquisition configuration

    Returns:
        AcquisitionMetrics with all computed values
    """
    from seafront.config.handles import CameraConfig

    # Count selected items
    selected_wells = [w for w in config.plate_wells if w.selected]
    selected_sites = [s for s in config.grid.mask if s.selected]
    enabled_channels = [c for c in config.channels if c.enabled]

    num_wells = len(selected_wells)
    num_sites = len(selected_sites)
    num_channels = len(enabled_channels)
    num_z_planes_total = sum(c.num_z_planes for c in enabled_channels)
    num_timepoints = config.grid.num_t

    total_num_images = num_timepoints * num_wells * num_sites * num_z_planes_total

    # Calculate image size from camera config
    cam_img_width = ConfigRegistry.get(CameraConfig.MAIN_IMAGE_WIDTH_PX)
    cam_img_height = ConfigRegistry.get(CameraConfig.MAIN_IMAGE_HEIGHT_PX)
    main_cam_pix_format = ConfigRegistry.get(CameraConfig.MAIN_PIXEL_FORMAT)

    assert cam_img_width is not None, "Camera width not configured"
    assert cam_img_height is not None, "Camera height not configured"
    assert main_cam_pix_format is not None, "Camera pixel format not configured"

    target_width = cam_img_width.intvalue
    target_height = cam_img_height.intvalue

    match main_cam_pix_format.value:
        case "mono8":
            bytes_per_pixel = 1
        case "mono10" | "mono12" | "mono14" | "mono16":
            bytes_per_pixel = 2
        case _unexpected:
            error_internal(f"unexpected pixel format: {_unexpected}")

    image_size_bytes = target_width * target_height * bytes_per_pixel
    max_storage_size_GB = total_num_images * image_size_bytes / 1024**3

    return AcquisitionMetrics(
        num_wells=num_wells,
        num_sites=num_sites,
        num_channels=num_channels,
        num_z_planes_total=num_z_planes_total,
        num_timepoints=num_timepoints,
        total_num_images=total_num_images,
        image_size_bytes=image_size_bytes,
        max_storage_size_GB=max_storage_size_GB,
    )


class AcquisitionEstimate(BaseModel):
    """
    Pre-acquisition estimate for confirmation dialog.

    Provides storage and time estimates without starting an acquisition.
    """
    project_name: str
    plate_name: str
    num_selected_wells: int
    num_selected_sites: int
    num_channels: int
    num_z_planes_total: int
    "Total z-planes across all channels (sum of each channel's num_z_planes)"
    num_timepoints: int
    total_num_images: int
    max_storage_size_GB: float
    "Upper bound storage estimate (uncompressed)"
    estimated_time_s: float | None
    "Estimated acquisition time in seconds (None if cannot be estimated)"
    microscope_name: str = ""
    "name of the device that processed this request"


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

    microscope_name: str = ""
    "name of the device that processed this request"


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


class CameraInfo(BaseModel):
    "identity of a camera as reported by its driver (empty strings when not connected)"

    vendor: str = ""
    model: str = ""
    serial_number: str = ""


class HardwareCapabilitiesResponse(BaseModel):
    main_camera: CameraInfo = Field(default_factory=CameraInfo)
    "identity of the main camera; the server puts this into OME metadata, clients can do the same"
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
    hardware_limits: HardwareLimits = Field(
        ...,
        title="Hardware Limits",
        description="numerical limits of configurable parameters based on actual hardware capabilities",
    )
    "numerical limits of configurable parameters based on actual hardware capabilities"
    filters: list[FilterConfig] = Field(
        default_factory=list,
        title="Filters",
        description="filter wheel positions this microscope has, empty if it has no filter wheel",
    )
    """
    filter wheel positions this microscope has

    Served here rather than left for a client to dig out of the machine config, where the
    value is only typed as "some object".
    """
    forbidden_areas: list[ForbiddenArea] = Field(
        default_factory=list,
        title="Forbidden Areas",
        description="stage regions the server refuses to move into",
    )
    "stage regions the server refuses to move into; same reason as `filters` for being here"
    microscope_name: str = ""
    "name of the device that processed this request"


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


class CapturedImageMetadata(BaseModel):
    """Facts fixed when a camera exposure completes, independent of storage policy."""

    image_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    channel: sc.AcquisitionChannelConfig
    width_px: int
    height_px: int
    timestamp: float
    position: Position
    pixel_format: str
    camera_bit_depth: int
    bit_depth: int
    ome_xml: str | None = None


class OmeMetadataContext(BaseModel):
    """Optional acquisition-owned facts used to enrich server-generated OME-XML."""

    project_name: str = ""
    plate_name: str = ""
    cell_line: str = ""
    well_name: str = ""
    site_x: int = 0
    site_y: int = 0
    site_z: int = 0
    x_offset_mm: float = 0
    y_offset_mm: float = 0
    z_offset_mm: float = 0
    plane_index: int = 0
    autofocus_succeeded: bool = False


class ImageAcquiredResponse(CapturedImageMetadata):
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
    microscope_name: str = ""
    "name of the device that processed this request"


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
    """
    stop whatever is streaming

    Carries no channel on purpose: the adapter lit the illumination and knows which channel
    it is running, so there is nothing for a client to get wrong (or to go stale) between
    starting a stream and stopping it.
    """

    machine_config: list[sc.ConfigItem] = Field(default_factory=list)

    _ReturnValue: type = PrivateAttr(default=BasicSuccessResponse)


class ChannelSnapshot(BaseModel, BaseCommand[ImageAcquiredResponse]):
    channel: sc.AcquisitionChannelConfig
    ome_context: OmeMetadataContext = Field(default_factory=OmeMetadataContext)

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
    microscope_name: str = ""
    "name of the device that processed this request"


class MoveByResult(BaseModel):
    axis: str
    moved_by_mm: float
    microscope_name: str = ""
    "name of the device that processed this request"


Millimetres = tp.Annotated[float, Field(allow_inf_nan=False)]
"""
a distance or coordinate in mm, which must be an actual number

NaN and infinity are rejected here because every safety check downstream is a comparison,
and every comparison with NaN is False: `x_mm < 0` says it is not negative, and
`min_x <= x <= max_x` says it is not inside any forbidden area. A NaN coordinate is therefore
reported safe by each check in turn and handed to the stage. Infinity passes the same way.
"""


class MoveBy(BaseModel, BaseCommand[MoveByResult]):
    """
    move stage by some distance

    moves stage by some distance, relative to its current location. may introduce minor errors if used frequently without intermediate absolute moveto
    """

    axis: tp.Literal["x", "y", "z"]
    distance_mm: Millimetres

    _ReturnValue: type = PrivateAttr(default=MoveByResult)


class MoveTo(BaseModel, BaseCommand[BasicSuccessResponse]):
    """
    move to target coordinates

    any of the arguments may be None, in which case the corresponding axis is not moved

    these coordinates are internally adjusted to take the calibration into account
    """

    x_mm: Millimetres | None = None
    y_mm: Millimetres | None = None
    z_mm: Millimetres | None = None

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
    microscope_name: str = ""
    "name of the device that processed this request"


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
    image_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    width_px: int = Field(..., title="Image Width [pixels]", description="image height, in pixels")
    "image width, in pixels"
    height_px: int = Field(
        ..., title="Image Height [pixels]", description="image height, in pixels"
    )
    "image height, in pixels"
    microscope_name: str = ""
    "name of the device that processed this request"

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
    microscope_name: str = ""
    "name of the device that processed this request"


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


class HealthResponse(BaseModel):
    """
    /api/health: answered without touching the hardware, and without a target_device.

    Always 200 while the server runs: that it answered is the liveness. Whether the microscope
    can do its job is `status`, and `reasons` says why when it cannot.
    """

    status: tp.Literal["ok", "degraded", "down"]
    """
    ok: every device connected and answering. degraded: a device has been seen not answering
    but is still connected. down: a device is not connected, or a hardware call is stuck
    """
    reasons: list[str] = Field(default_factory=list)
    "one line per thing that keeps the status from ok"
    hardware: list["DeviceHealth"]
    stalls: list["Stall"] = Field(default_factory=list)
    last_acquisition_error: str | None = None
    last_acquisition_error_timestamp: str | None = None
    microscope_name: str
    time_since_last_request_s: float | None
    "seconds since the last request this server answered, or null if it has answered none"
    server_time: float
    "server clock at the time of this reply, as a unix timestamp"
