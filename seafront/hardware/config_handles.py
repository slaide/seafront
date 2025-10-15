"""
Hardware-specific configuration handle enums.

This module contains config handles specific to hardware implementations,
organized by device type.
"""

from seafront.config.handles import ConfigHandle


class SquidConfig(ConfigHandle):
    """SQUID microscope-specific configuration handles."""

    # SQUID-specific imaging settings
    SQUID_IMAGING_MODE = "squid.imaging.mode"
    SQUID_STAGE_SPEED = "squid.stage.speed"
    SQUID_AUTOFOCUS_ALGORITHM = "squid.autofocus.algorithm"

    # SQUID hardware limits and capabilities
    SQUID_MAX_EXPOSURE_MS = "squid.limits.max_exposure_ms"
    SQUID_MIN_EXPOSURE_MS = "squid.limits.min_exposure_ms"
    SQUID_STAGE_LIMITS_X_MM = "squid.limits.stage_x_mm"
    SQUID_STAGE_LIMITS_Y_MM = "squid.limits.stage_y_mm"
    SQUID_STAGE_LIMITS_Z_MM = "squid.limits.stage_z_mm"


class GalaxyCameraConfig(ConfigHandle):
    """Galaxy camera-specific configuration handles."""

    # Galaxy camera specific settings
    GALAXY_GAIN_MODE = "camera.galaxy.gain_mode"
    GALAXY_TRIGGER_DELAY = "camera.galaxy.trigger_delay"
    GALAXY_BINNING = "camera.galaxy.binning"
    GALAXY_ROI_OFFSET_X = "camera.galaxy.roi.offset_x"
    GALAXY_ROI_OFFSET_Y = "camera.galaxy.roi.offset_y"


class ToupCamConfig(ConfigHandle):
    """ToupCam camera-specific configuration handles."""

    # ToupCam specific settings
    TOUPCAM_TEMPERATURE = "camera.toupcam.temperature"
    TOUPCAM_GAIN = "camera.toupcam.gain"
    TOUPCAM_GAMMA = "camera.toupcam.gamma"
    TOUPCAM_CONTRAST = "camera.toupcam.contrast"


# Hardware-specific convenience functions
def get_squid_config_item(handle: SquidConfig):
    """Get SQUID-specific config item."""
    return handle.get_item()


def get_camera_config_item(handle):
    """Get camera-specific config item (works with Galaxy or ToupCam)."""
    return handle.get_item()
