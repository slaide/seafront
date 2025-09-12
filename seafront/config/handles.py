"""
Config handle enums for type-safe, organized access to machine configuration.

This module provides domain-specific enums for config handles, replacing hard-coded strings
throughout the codebase with organized, type-safe enum values.
"""

from enum import Enum
from typing import Dict, Any, Union

from seafront.config.basics import GlobalConfigHandler

class ConfigHandle(Enum):
    """Base class for config handle enums with utility methods."""
    
    def __str__(self) -> str:
        """Return the handle string value."""
        return self.value
    
    @classmethod
    def get_dict(cls) -> Dict[str, Any]:
        """Get the global config dict for convenient access."""
        return GlobalConfigHandler.get_dict()
    
    def get_item(self) -> Any:
        """Get the config item for this handle."""
        g_config = self.get_dict()
        return g_config[self.value]
    
    @property
    def value_item(self) -> Any:
        """Convenient property to get the config item."""
        return self.get_item()


class CameraConfig(ConfigHandle):
    """Camera-related configuration handles."""
    
    # Main camera configuration
    MAIN_MODEL = "camera.main.model"
    MAIN_DRIVER = "camera.main.driver"
    MAIN_OBJECTIVE = "camera.main.objective"
    MAIN_TRIGGER = "camera.main.trigger"
    MAIN_PIXEL_FORMAT = "camera.main.pixel_format"
    
    # Main camera image settings
    MAIN_IMAGE_WIDTH_PX = "camera.main.image.width_px"
    MAIN_IMAGE_HEIGHT_PX = "camera.main.image.height_px"
    MAIN_IMAGE_FLIP_HORIZONTAL = "camera.main.image.flip_horizontal"
    MAIN_IMAGE_FLIP_VERTICAL = "camera.main.image.flip_vertical"


class LaserAutofocusConfig(ConfigHandle):
    """Laser autofocus system configuration handles."""
    
    # System availability and basic settings
    AVAILABLE = "laser.autofocus.available"
    EXPOSURE_TIME_MS = "laser.autofocus.exposure_time_ms"
    USE_GLASS_TOP = "laser.autofocus.use_glass_top"
    WARM_UP_LASER = "laser.autofocus.warm_up_laser"
    
    # Camera configuration
    CAMERA_MODEL = "laser.autofocus.camera.model"
    CAMERA_DRIVER = "laser.autofocus.camera.driver"
    CAMERA_ANALOG_GAIN = "laser.autofocus.camera.analog_gain"
    CAMERA_PIXEL_FORMAT = "laser.autofocus.camera.pixel_format"
    
    # Calibration settings
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
    
    FORBIDDEN_WELLS = "protocol.forbidden_wells"


class SystemConfig(ConfigHandle):
    """System-level configuration handles."""
    
    MICROSCOPE_NAME = "system.microscope_name"


class IlluminationConfig(ConfigHandle):
    """Illumination control configuration handles."""
    
    TURN_OFF_ALL = "illumination.turn_off_all"


# Convenience function to get config item by enum
def get_config_item(handle: Union[ConfigHandle, str]) -> Any:
    """
    Get a config item by handle enum or string.
    
    Args:
        handle: ConfigHandle enum or string handle
        
    Returns:
        The config item
    """
    g_config = GlobalConfigHandler.get_dict()
    
    if isinstance(handle, ConfigHandle):
        return g_config[handle.value]
    else:
        return g_config[handle]


# Legacy compatibility mappings for gradual migration
LEGACY_HANDLE_MAPPING = {
    # Old handle -> New enum
    "main_camera_model": CameraConfig.MAIN_MODEL,
    "main_camera_driver": CameraConfig.MAIN_DRIVER,
    "main_camera_objective": CameraConfig.MAIN_OBJECTIVE,
    "main_camera_trigger": CameraConfig.MAIN_TRIGGER,
    "main_camera_pixel_format": CameraConfig.MAIN_PIXEL_FORMAT,
    "main_camera_image_width_px": CameraConfig.MAIN_IMAGE_WIDTH_PX,
    "main_camera_image_height_px": CameraConfig.MAIN_IMAGE_HEIGHT_PX,
    "main_camera_image_flip_horizontal": CameraConfig.MAIN_IMAGE_FLIP_HORIZONTAL,
    "main_camera_image_flip_vertical": CameraConfig.MAIN_IMAGE_FLIP_VERTICAL,
    
    "laser_autofocus_available": LaserAutofocusConfig.AVAILABLE,
    "laser_autofocus_exposure_time_ms": LaserAutofocusConfig.EXPOSURE_TIME_MS,
    "laser_autofocus_analog_gain": LaserAutofocusConfig.CAMERA_ANALOG_GAIN,
    "laser_autofocus_use_glass_top": LaserAutofocusConfig.USE_GLASS_TOP,
    "laser_af_warm_up_laser": LaserAutofocusConfig.WARM_UP_LASER,
    "laser_autofocus_camera_model": LaserAutofocusConfig.CAMERA_MODEL,
    "laser_autofocus_camera_driver": LaserAutofocusConfig.CAMERA_DRIVER,
    "laser_autofocus_pixel_format": LaserAutofocusConfig.CAMERA_PIXEL_FORMAT,
    "laser_autofocus_is_calibrated": LaserAutofocusConfig.CALIBRATION_IS_CALIBRATED,
    "laser_autofocus_calibration_x": LaserAutofocusConfig.CALIBRATION_X_PEAK_POS,
    "laser_autofocus_calibration_umpx": LaserAutofocusConfig.CALIBRATION_UM_PER_PX,
    "laser_autofocus_calibration_refzmm": LaserAutofocusConfig.CALIBRATION_REF_Z_MM,
    
    "calibrate_B2_here": CalibrationConfig.CALIBRATE_B2_HERE,
    "calibration_offset_x_mm": CalibrationConfig.OFFSET_X_MM,
    "calibration_offset_y_mm": CalibrationConfig.OFFSET_Y_MM,
    "calibration_offset_z_mm": CalibrationConfig.OFFSET_Z_MM,
    
    "channels": ImagingConfig.CHANNELS,
    "imaging_order": ImagingConfig.ORDER,
    
    "image_file_pad_low": ImageConfig.FILE_PAD_LOW,
    "image_filename_use_channel_name": ImageConfig.FILENAME_USE_CHANNEL_NAME,
    "image_filename_xy_index_start": ImageConfig.FILENAME_XY_INDEX_START,
    "image_filename_z_index_start": ImageConfig.FILENAME_Z_INDEX_START,
    "image_filename_site_index_start": ImageConfig.FILENAME_SITE_INDEX_START,
    "image_filename_zero_pad_column": ImageConfig.FILENAME_ZERO_PAD_COLUMN,
    
    "base_image_output_dir": StorageConfig.BASE_IMAGE_OUTPUT_DIR,
    "filter_wheel_available": FilterWheelConfig.AVAILABLE,
    "filters": FilterWheelConfig.CONFIGURATION,
    "forbidden_wells": ProtocolConfig.FORBIDDEN_WELLS,
    "microscope_name": SystemConfig.MICROSCOPE_NAME,
    "illumination_off": IlluminationConfig.TURN_OFF_ALL,
}


def get_config_item_legacy(handle: str) -> Any:
    """
    Get config item with automatic legacy handle translation.
    
    This function allows gradual migration by automatically translating old handles
    to new enum-based handles.
    
    Args:
        handle: Old or new handle string
        
    Returns:
        The config item
    """
    g_config = GlobalConfigHandler.get_dict()
    
    # Try new handle first, then legacy mapping
    if handle in g_config:
        return g_config[handle]
    elif handle in LEGACY_HANDLE_MAPPING:
        enum_handle = LEGACY_HANDLE_MAPPING[handle]
        return g_config[enum_handle.value]
    else:
        # Fallback to original handle (will raise KeyError if not found)
        return g_config[handle]