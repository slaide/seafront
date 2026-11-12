"""
What the hardware will accept, as data.

These are published: they reach the browser inside HardwareCapabilitiesResponse, and the
generated client types are only as precise as the declaration here. They live in their own
leaf module because both the camera drivers and the API response models need them, and
`hardware/microscope.py` already imports `server/commands.py`.
"""

from pydantic import BaseModel, ConfigDict, Field

from seafront.hardware.firmware_config import get_firmware_config


class HardwareLimitValue(BaseModel):
    """the range a single hardware parameter accepts, and the granularity it accepts it in"""

    model_config = ConfigDict(frozen=True)

    min: float | int
    max: float | int
    step: float | int


class HardwareLimits(BaseModel):
    """
    What this microscope will accept for each configurable imaging parameter.

    Every field is a named range rather than an entry in a loose dict, so a client knows which
    limits exist without probing for them, and a limit renamed here reaches the browser's type
    checker rather than failing silently at runtime.
    """

    model_config = ConfigDict(frozen=True)

    stage_z_travel_mm: HardwareLimitValue | None = Field(
        default=None,
        description="Motor Z travel before calibration offsets; add calibration.offset.z_mm "
        "for stage coordinates. Unknown for adapters that do not report it.",
    )

    imaging_exposure_time_ms: HardwareLimitValue
    imaging_analog_gain_db: HardwareLimitValue
    imaging_focus_offset_um: HardwareLimitValue
    imaging_illum_perc: HardwareLimitValue
    imaging_illum_perc_fluorescence: HardwareLimitValue
    imaging_illum_perc_brightfield: HardwareLimitValue
    imaging_number_z_planes: HardwareLimitValue
    imaging_delta_z_um: HardwareLimitValue


# The post-homing motor envelope used by SQUID firmware. Coordinates precede calibration.
SQUID_Z_MIN_MM = 0.0
SQUID_Z_MAX_MM = 6.7


def squid_z_travel_limits() -> HardwareLimitValue:
    return HardwareLimitValue(
        min=SQUID_Z_MIN_MM,
        max=SQUID_Z_MAX_MM,
        step=get_firmware_config().mm_per_ustep_z,
    )
