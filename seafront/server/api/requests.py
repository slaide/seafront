"""
The request body of every /api route, declared once.

Each model is `Envelope` + the payload, so the body stays flat on the wire while routing and
access (the envelope) stay a separate type from what the microscope is being asked to do.
Where a route runs a hardware command, the payload half *is* the command class from
`server/commands.py`, so a field is defined in exactly one place and the route cannot drift
from the command it submits.

These models are what FastAPI validates incoming bodies against and what it derives
`/openapi.json` from, so the published schema is the enforced schema by construction.
"""

import seaconfig as sc
from seaconfig.acquisition import AcquisitionConfig

from seaconfig.config_item import ConfigItem
from seafront.server.api.envelope import Envelope
from seafront.server.commands import (
    AutofocusApproachTargetDisplacement,
    AutofocusLaserWarmup,
    AutofocusMeasureDisplacement,
    AutofocusSnap,
    ChannelSnapshot,
    ChannelStreamBegin,
    ChannelStreamEnd,
    EstablishHardwareConnection,
    IlluminationEndAll,
    LaserAutofocusCalibrate,
    LoadingPositionEnter,
    LoadingPositionLeave,
    MoveBy,
    MoveTo,
    MoveToWell,
)

# -- requests that carry nothing but the envelope ---------------------------------------------


class CurrentStateRequest(Envelope):
    pass


class HardwareCapabilitiesRequest(Envelope):
    pass


class MachineDefaultsRequest(Envelope):
    pass


class ConfigListRequest(Envelope):
    pass


class CancelCurrentRequest(Envelope):
    pass


# -- hardware commands: envelope + the command that will reach the microscope ------------------


class MoveByRequest(Envelope, MoveBy):
    pass


class MoveToRequest(Envelope, MoveTo):
    pass


class MoveToWellRequest(Envelope, MoveToWell):
    pass


class LoadingPositionEnterRequest(Envelope, LoadingPositionEnter):
    pass


class LoadingPositionLeaveRequest(Envelope, LoadingPositionLeave):
    pass


class SnapChannelRequest(Envelope, ChannelSnapshot):
    pass


class StreamChannelBeginRequest(Envelope, ChannelStreamBegin):
    pass


class StreamChannelEndRequest(Envelope, ChannelStreamEnd):
    pass


class AutofocusSnapRequest(Envelope, AutofocusSnap):
    pass


class AutofocusMeasureDisplacementRequest(Envelope, AutofocusMeasureDisplacement):
    pass


class AutofocusApproachTargetRequest(Envelope, AutofocusApproachTargetDisplacement):
    pass


class LaserAutofocusCalibrateRequest(Envelope, LaserAutofocusCalibrate):
    pass


class AutofocusLaserWarmupRequest(Envelope, AutofocusLaserWarmup):
    pass


class IlluminationEndAllRequest(Envelope, IlluminationEndAll):
    pass


class EstablishHardwareConnectionRequest(Envelope, EstablishHardwareConnection):
    pass


# -- requests the server answers itself --------------------------------------------------------


class MachineConfigFlushRequest(Envelope):
    machine_config: list[ConfigItem]
    privileged: bool = False


class AcquisitionStartRequest(Envelope):
    config_file: AcquisitionConfig


class AcquisitionCancelRequest(Envelope):
    acquisition_id: str


class AcquisitionStatusRequest(Envelope):
    acquisition_id: str


class AcquisitionEstimateRequest(Envelope):
    config_file: AcquisitionConfig


class ConfigFetchRequest(Envelope):
    config_file: str


class ConfigStoreRequest(Envelope):
    config_file: AcquisitionConfig
    filename: str
    overwrite_on_conflict: None | bool = None
    comment: None | str = None


class SnapSelectedChannelsRequest(Envelope):
    config_file: sc.AcquisitionConfig


class CalibrateStageXYHereRequest(Envelope):
    plate_model_id: str = "revvity-384-6057800"


# -- the device lease ---------------------------------------------------------------------------
# these read `lock_token` off the envelope like every other route; it is the token they act on


class LockAcquireRequest(Envelope):
    owner: str
    ttl_s: float | None = None


class LockReleaseRequest(Envelope):
    pass


class LockStatusRequest(Envelope):
    pass


class LockBreakRequest(Envelope):
    reason: str = ""
