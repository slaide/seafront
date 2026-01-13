"""
Config handle enums for type-safe, organized access to machine configuration.

This module provides domain-specific enums for config handles, replacing hard-coded strings
throughout the codebase with organized, type-safe enum values.

Usage:
    from seafront.config.registry import ConfigRegistry
    from seafront.config.handles import CameraConfig

    # Get a config item
    camera_id = ConfigRegistry.get(CameraConfig.MAIN_ID).strvalue
"""

from enum import Enum


class ConfigHandle(Enum):
    """Base class for config handle enums."""

    def __str__(self) -> str:
        """Return the handle string value."""
        return self.value


class CameraConfig(ConfigHandle):
    """Camera-related configuration handles."""

    # Main camera configuration
    MAIN_ID = "camera.main.id"
    MAIN_DRIVER = "camera.main.driver"
    MAIN_OBJECTIVE = "camera.main.objective"
    MAIN_TRIGGER = "camera.main.trigger"
    MAIN_PIXEL_FORMAT = "camera.main.pixel_format"

    # Main camera image settings
    MAIN_IMAGE_WIDTH_PX = "camera.main.image.width_px"
    MAIN_IMAGE_HEIGHT_PX = "camera.main.image.height_px"
    MAIN_IMAGE_FLIP_HORIZONTAL = "camera.main.image.flip_horizontal"
    MAIN_IMAGE_FLIP_VERTICAL = "camera.main.image.flip_vertical"

    # Camera communication and retry settings
    RECONNECTION_ATTEMPTS = "camera.reconnection_attempts"
    RECONNECTION_DELAY_MS = "camera.reconnection_delay_ms"
    OPERATION_RETRY_ATTEMPTS = "camera.operation_retry_attempts"


class MicrocontrollerConfig(ConfigHandle):
    """Microcontroller communication and retry configuration handles."""

    # Device identification
    ID = "microcontroller.id"
    DRIVER = "microcontroller.driver"

    # Connection grace period and retry settings
    RECONNECTION_GRACE_PERIOD_MS = "microcontroller.reconnection_grace_period_ms"
    RECONNECTION_ATTEMPTS = "microcontroller.reconnection_attempts"
    RECONNECTION_DELAY_MS = "microcontroller.reconnection_delay_ms"
    OPERATION_RETRY_ATTEMPTS = "microcontroller.operation_retry_attempts"
    OPERATION_RETRY_DELAY_MS = "microcontroller.operation_retry_delay_ms"


class LaserAutofocusConfig(ConfigHandle):
    """Laser autofocus system configuration handles."""

    # System availability and basic settings
    AVAILABLE = "laser.autofocus.available"
    EXPOSURE_TIME_MS = "laser.autofocus.exposure_time_ms"
    USE_GLASS_TOP = "laser.autofocus.use_glass_top"
    WARM_UP_LASER = "laser.autofocus.warm_up_laser"
    OFFSET_UM = "laser.autofocus.offset_um"

    # Camera configuration
    CAMERA_ID = "laser.autofocus.camera.id"
    CAMERA_DRIVER = "laser.autofocus.camera.driver"
    CAMERA_ANALOG_GAIN = "laser.autofocus.camera.analog_gain"
    CAMERA_PIXEL_FORMAT = "laser.autofocus.camera.pixel_format"

    # Calibration settings
    CALIBRATION_Z_SPAN_MM = "laser.autofocus.calibration.z_span_mm"
    CALIBRATION_NUM_Z_STEPS = "laser.autofocus.calibration.num_z_steps"
    CALIBRATION_IS_CALIBRATED = "laser.autofocus.calibration.is_calibrated"
    CALIBRATION_X_PEAK_POS = "laser.autofocus.calibration.x_peak_pos"
    CALIBRATION_UM_PER_PX = "laser.autofocus.calibration.um_per_px"
    CALIBRATION_REF_Z_MM = "laser.autofocus.calibration.ref_z_mm"


class CalibrationConfig(ConfigHandle):
    """System calibration configuration handles."""

    CALIBRATE_B2_HERE = "calibration.calibrate_B2_here"
    OFFSET_X_MM = "calibration.offset.x_mm"
    OFFSET_Y_MM = "calibration.offset.y_mm"
    OFFSET_Z_MM = "calibration.offset.z_mm"


class ImagingConfig(ConfigHandle):
    """Imaging and acquisition configuration handles."""

    CHANNELS = "imaging.channels"
    ORDER = "imaging.order"


class ImageConfig(ConfigHandle):
    """Image processing and storage configuration handles."""

    FILE_PAD_LOW = "image.file.pad_low"
    FILENAME_USE_CHANNEL_NAME = "image.filename.use_channel_name"
    FILENAME_XY_INDEX_START = "image.filename.xy_index_start"
    FILENAME_Z_INDEX_START = "image.filename.z_index_start"
    FILENAME_SITE_INDEX_START = "image.filename.site_index_start"
    FILENAME_ZERO_PAD_COLUMN = "image.filename.zero_pad_column"


class StorageConfig(ConfigHandle):
    """Storage and output configuration handles."""

    BASE_IMAGE_OUTPUT_DIR = "storage.base_image_output_dir"


class FilterWheelConfig(ConfigHandle):
    """Filter wheel system configuration handles."""

    AVAILABLE = "filter.wheel.available"
    CONFIGURATION = "filter.wheel.configuration"


class ProtocolConfig(ConfigHandle):
    """Protocol and acquisition workflow configuration handles."""

    FORBIDDEN_AREAS = "protocol.forbidden_areas"


class SystemConfig(ConfigHandle):
    """System-level configuration handles."""

    MICROSCOPE_NAME = "system.microscope_name"
    MICROSCOPE_TYPE = "system.microscope_type"


class IlluminationConfig(ConfigHandle):
    """Illumination control configuration handles."""

    TURN_OFF_ALL = "illumination.turn_off_all"

