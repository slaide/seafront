"""
Core system configuration items.

This module defines the config items that are always present.
Other modules (cameras, hardware) can define their own config items.
"""

from seaconfig import ConfigItemOption

from seafront.config.registry import ConfigRegistry, config_item, bool_config_item


# =============================================================================
# Handle constants - define handles where they're used
# =============================================================================

# System
MICROSCOPE_NAME = "system.microscope_name"
MICROSCOPE_TYPE = "system.microscope_type"

# Camera
CAMERA_MAIN_ID = "camera.main.id"
CAMERA_MAIN_DRIVER = "camera.main.driver"
CAMERA_MAIN_OBJECTIVE = "camera.main.objective"
CAMERA_MAIN_TRIGGER = "camera.main.trigger"
CAMERA_MAIN_PIXEL_FORMAT = "camera.main.pixel_format"
CAMERA_MAIN_IMAGE_WIDTH_PX = "camera.main.image.width_px"
CAMERA_MAIN_IMAGE_HEIGHT_PX = "camera.main.image.height_px"
CAMERA_MAIN_IMAGE_FLIP_HORIZONTAL = "camera.main.image.flip_horizontal"
CAMERA_MAIN_IMAGE_FLIP_VERTICAL = "camera.main.image.flip_vertical"
CAMERA_RECONNECTION_ATTEMPTS = "camera.reconnection_attempts"
CAMERA_RECONNECTION_DELAY_MS = "camera.reconnection_delay_ms"
CAMERA_OPERATION_RETRY_ATTEMPTS = "camera.operation_retry_attempts"

# Microcontroller
MICROCONTROLLER_ID = "microcontroller.id"
MICROCONTROLLER_RECONNECTION_GRACE_PERIOD_MS = "microcontroller.reconnection_grace_period_ms"
MICROCONTROLLER_RECONNECTION_ATTEMPTS = "microcontroller.reconnection_attempts"
MICROCONTROLLER_RECONNECTION_DELAY_MS = "microcontroller.reconnection_delay_ms"
MICROCONTROLLER_OPERATION_RETRY_ATTEMPTS = "microcontroller.operation_retry_attempts"
MICROCONTROLLER_OPERATION_RETRY_DELAY_MS = "microcontroller.operation_retry_delay_ms"

# Calibration
CALIBRATION_CALIBRATE_B2_HERE = "calibration.calibrate_B2_here"
CALIBRATION_OFFSET_X_MM = "calibration.offset.x_mm"
CALIBRATION_OFFSET_Y_MM = "calibration.offset.y_mm"
CALIBRATION_OFFSET_Z_MM = "calibration.offset.z_mm"

# Storage
STORAGE_BASE_IMAGE_OUTPUT_DIR = "storage.base_image_output_dir"

# Imaging
IMAGING_CHANNELS = "imaging.channels"
IMAGING_ORDER = "imaging.order"

# Image output
IMAGE_FILE_PAD_LOW = "image.file.pad_low"
IMAGE_FILENAME_USE_CHANNEL_NAME = "image.filename.use_channel_name"
IMAGE_FILENAME_XY_INDEX_START = "image.filename.xy_index_start"
IMAGE_FILENAME_Z_INDEX_START = "image.filename.z_index_start"
IMAGE_FILENAME_SITE_INDEX_START = "image.filename.site_index_start"
IMAGE_FILENAME_ZERO_PAD_COLUMN = "image.filename.zero_pad_column"

# Laser autofocus
LASER_AUTOFOCUS_AVAILABLE = "laser.autofocus.available"
LASER_AUTOFOCUS_CAMERA_ID = "laser.autofocus.camera.id"
LASER_AUTOFOCUS_CAMERA_DRIVER = "laser.autofocus.camera.driver"
LASER_AUTOFOCUS_EXPOSURE_TIME_MS = "laser.autofocus.exposure_time_ms"
LASER_AUTOFOCUS_CAMERA_ANALOG_GAIN = "laser.autofocus.camera.analog_gain"
LASER_AUTOFOCUS_USE_GLASS_TOP = "laser.autofocus.use_glass_top"
LASER_AUTOFOCUS_CAMERA_PIXEL_FORMAT = "laser.autofocus.camera.pixel_format"
LASER_AUTOFOCUS_WARM_UP_LASER = "laser.autofocus.warm_up_laser"
LASER_AUTOFOCUS_OFFSET_UM = "laser.autofocus.offset_um"
LASER_AUTOFOCUS_CALIBRATION_Z_SPAN_MM = "laser.autofocus.calibration.z_span_mm"
LASER_AUTOFOCUS_CALIBRATION_NUM_Z_STEPS = "laser.autofocus.calibration.num_z_steps"
LASER_AUTOFOCUS_CALIBRATION_IS_CALIBRATED = "laser.autofocus.calibration.is_calibrated"
LASER_AUTOFOCUS_CALIBRATION_X_PEAK_POS = "laser.autofocus.calibration.x_peak_pos"
LASER_AUTOFOCUS_CALIBRATION_UM_PER_PX = "laser.autofocus.calibration.um_per_px"
LASER_AUTOFOCUS_CALIBRATION_REF_Z_MM = "laser.autofocus.calibration.ref_z_mm"

# Filter wheel
FILTER_WHEEL_AVAILABLE = "filter.wheel.available"
FILTER_WHEEL_CONFIGURATION = "filter.wheel.configuration"

# Protocol
PROTOCOL_FORBIDDEN_AREAS = "protocol.forbidden_areas"

# Illumination
ILLUMINATION_TURN_OFF_ALL = "illumination.turn_off_all"


def register_core_config(default_image_dir: str, default_channels: list | str, default_forbidden_areas: list | str) -> None:
    """
    Register core system config items.

    Call this after ConfigRegistry.init() with config file values.
    """

    # System
    ConfigRegistry.register(
        config_item(
            handle=MICROSCOPE_NAME,
            name="microscope name",
            value_kind="text",
            default="squid",
            frozen=True,
            persistent=True,
        ),
        config_item(
            handle=MICROSCOPE_TYPE,
            name="microscope type",
            value_kind="option",
            default="squid",
            options=[
                ConfigItemOption(name="SQUID", handle="squid"),
                ConfigItemOption(name="Mock", handle="mock"),
            ],
            frozen=True,
            persistent=True,
        ),
    )

    # Main camera
    ConfigRegistry.register(
        config_item(
            handle=CAMERA_MAIN_ID,
            name="main camera USB ID",
            value_kind="text",
            default="CHANGE_ME",
            frozen=True,
            persistent=True,
        ),
        config_item(
            handle=CAMERA_MAIN_DRIVER,
            name="main camera driver",
            value_kind="option",
            default="galaxy",
            options=[
                ConfigItemOption(name="Galaxy (Daheng)", handle="galaxy"),
                ConfigItemOption(name="ToupCam", handle="toupcam"),
            ],
            frozen=True,
            persistent=True,
        ),
        config_item(
            handle=CAMERA_MAIN_OBJECTIVE,
            name="main camera objective",
            value_kind="option",
            default="20xolympus",
            options=[
                ConfigItemOption(name="4x Olympus", handle="4xolympus", info={"magnification": 4}),
                ConfigItemOption(name="10x Olympus", handle="10xolympus", info={"magnification": 10}),
                ConfigItemOption(name="20x Olympus", handle="20xolympus", info={"magnification": 20}),
            ],
        ),
        config_item(
            handle=CAMERA_MAIN_TRIGGER,
            name="main camera trigger",
            value_kind="option",
            default="software",
            options=[
                ConfigItemOption(name="Software", handle="software"),
                ConfigItemOption(name="Hardware", handle="hardware"),
            ],
        ),
        config_item(
            handle=CAMERA_MAIN_PIXEL_FORMAT,
            name="main camera pixel format",
            value_kind="option",
            default="mono8",
            options=[
                ConfigItemOption(name="8 Bit", handle="mono8"),
                ConfigItemOption(name="12 Bit", handle="mono12"),
                ConfigItemOption(name="16 Bit", handle="mono16"),
            ],
        ),
        config_item(
            handle=CAMERA_MAIN_IMAGE_WIDTH_PX,
            name="main camera image width [px]",
            value_kind="int",
            default=2500,
        ),
        config_item(
            handle=CAMERA_MAIN_IMAGE_HEIGHT_PX,
            name="main camera image height [px]",
            value_kind="int",
            default=2500,
        ),
        bool_config_item(
            handle=CAMERA_MAIN_IMAGE_FLIP_HORIZONTAL,
            name="main camera flip image horizontally",
            default=False,
            frozen=True,
        ),
        bool_config_item(
            handle=CAMERA_MAIN_IMAGE_FLIP_VERTICAL,
            name="main camera flip image vertically",
            default=True,
            frozen=True,
        ),
        config_item(
            handle=CAMERA_RECONNECTION_ATTEMPTS,
            name="camera reconnection attempts",
            value_kind="int",
            default=5,
        ),
        config_item(
            handle=CAMERA_RECONNECTION_DELAY_MS,
            name="camera reconnection delay [ms]",
            value_kind="float",
            default=1000.0,
        ),
        config_item(
            handle=CAMERA_OPERATION_RETRY_ATTEMPTS,
            name="camera operation retry attempts",
            value_kind="int",
            default=5,
        ),
    )

    # Microcontroller
    ConfigRegistry.register(
        config_item(
            handle=MICROCONTROLLER_ID,
            name="microcontroller USB ID",
            value_kind="text",
            default="",
            frozen=True,
            persistent=True,
        ),
        config_item(
            handle=MICROCONTROLLER_RECONNECTION_GRACE_PERIOD_MS,
            name="microcontroller reconnection grace period [ms]",
            value_kind="float",
            default=200.0,
        ),
        config_item(
            handle=MICROCONTROLLER_RECONNECTION_ATTEMPTS,
            name="microcontroller reconnection attempts",
            value_kind="int",
            default=5,
        ),
        config_item(
            handle=MICROCONTROLLER_RECONNECTION_DELAY_MS,
            name="microcontroller reconnection delay [ms]",
            value_kind="float",
            default=1000.0,
        ),
        config_item(
            handle=MICROCONTROLLER_OPERATION_RETRY_ATTEMPTS,
            name="microcontroller operation retry attempts",
            value_kind="int",
            default=5,
        ),
        config_item(
            handle=MICROCONTROLLER_OPERATION_RETRY_DELAY_MS,
            name="microcontroller operation retry delay [ms]",
            value_kind="float",
            default=5.0,
        ),
    )

    # Calibration
    ConfigRegistry.register(
        config_item(
            handle=CALIBRATION_CALIBRATE_B2_HERE,
            name="calibrate top left of B2 here",
            value_kind="action",
            default="/api/action/calibrate_stage_xy_here",
        ),
        config_item(
            handle=CALIBRATION_OFFSET_X_MM,
            name="calibration offset x [mm]",
            value_kind="float",
            default=0.0,
            persistent=True,
        ),
        config_item(
            handle=CALIBRATION_OFFSET_Y_MM,
            name="calibration offset y [mm]",
            value_kind="float",
            default=0.0,
            persistent=True,
        ),
        config_item(
            handle=CALIBRATION_OFFSET_Z_MM,
            name="calibration offset z [mm]",
            value_kind="float",
            default=0.0,
            persistent=True,
        ),
    )

    # Storage
    ConfigRegistry.register(
        config_item(
            handle=STORAGE_BASE_IMAGE_OUTPUT_DIR,
            name="base output storage directory",
            value_kind="text",
            default=default_image_dir,
            persistent=True,
        ),
    )

    # Imaging
    ConfigRegistry.register(
        config_item(
            handle=IMAGING_CHANNELS,
            name="imaging channels configuration",
            value_kind="object",
            default=default_channels,
            frozen=True,
            persistent=True,
        ),
        config_item(
            handle=IMAGING_ORDER,
            name="imaging order",
            value_kind="option",
            default="protocol_order",
            options=[
                ConfigItemOption(name="Protocol Order (config file order)", handle="protocol_order"),
                ConfigItemOption(name="Z-Order (bottom to top)", handle="z_order"),
                ConfigItemOption(name="Wavelength Order (high to low, then brightfield)", handle="wavelength_order"),
            ],
        ),
    )

    # Image output settings
    ConfigRegistry.register(
        bool_config_item(
            handle=IMAGE_FILE_PAD_LOW,
            name="image file pad low",
            default=True,
        ),
        bool_config_item(
            handle=IMAGE_FILENAME_USE_CHANNEL_NAME,
            name="image filename use channel name",
            default=True,
        ),
        config_item(
            handle=IMAGE_FILENAME_XY_INDEX_START,
            name="image filename xy index start",
            value_kind="int",
            default=0,
        ),
        config_item(
            handle=IMAGE_FILENAME_Z_INDEX_START,
            name="image filename z index start",
            value_kind="int",
            default=0,
        ),
        config_item(
            handle=IMAGE_FILENAME_SITE_INDEX_START,
            name="image filename site index start",
            value_kind="int",
            default=1,
        ),
        bool_config_item(
            handle=IMAGE_FILENAME_ZERO_PAD_COLUMN,
            name="image filename zero pad column index",
            default=True,
        ),
    )

    # Illumination
    ConfigRegistry.register(
        config_item(
            handle=ILLUMINATION_TURN_OFF_ALL,
            name="turn off all illumination",
            value_kind="action",
            default="/api/action/turn_off_all_illumination",
        ),
    )

    # Protocol
    ConfigRegistry.register(
        config_item(
            handle=PROTOCOL_FORBIDDEN_AREAS,
            name="forbidden areas",
            value_kind="object",
            default=default_forbidden_areas,
            persistent=True,
        ),
    )

    # Filter wheel (base availability - config details registered if available)
    ConfigRegistry.register(
        bool_config_item(
            handle=FILTER_WHEEL_AVAILABLE,
            name="filter wheel system available",
            default=False,
            frozen=True,
            persistent=True,
        ),
    )

    # Laser autofocus (base availability)
    ConfigRegistry.register(
        bool_config_item(
            handle=LASER_AUTOFOCUS_AVAILABLE,
            name="laser autofocus system available",
            default=False,
            frozen=True,
            persistent=True,
        ),
    )


def register_laser_autofocus_config() -> None:
    """Register laser autofocus config items (call only if LAF is available)."""
    ConfigRegistry.register(
        config_item(
            handle=LASER_AUTOFOCUS_CAMERA_ID,
            name="laser autofocus camera USB ID",
            value_kind="text",
            default="CHANGE_ME",
            frozen=True,
            persistent=True,
        ),
        config_item(
            handle=LASER_AUTOFOCUS_CAMERA_DRIVER,
            name="laser autofocus camera driver",
            value_kind="option",
            default="galaxy",
            options=[
                ConfigItemOption(name="Galaxy (Daheng)", handle="galaxy"),
                ConfigItemOption(name="ToupCam", handle="toupcam"),
            ],
            frozen=True,
            persistent=True,
        ),
        config_item(
            handle=LASER_AUTOFOCUS_EXPOSURE_TIME_MS,
            name="laser autofocus exposure time [ms]",
            value_kind="float",
            default=5.0,
        ),
        config_item(
            handle=LASER_AUTOFOCUS_CAMERA_ANALOG_GAIN,
            name="laser autofocus camera analog gain",
            value_kind="float",
            default=0.0,
        ),
        bool_config_item(
            handle=LASER_AUTOFOCUS_USE_GLASS_TOP,
            name="laser autofocus use glass top",
            default=False,
        ),
        config_item(
            handle=LASER_AUTOFOCUS_CAMERA_PIXEL_FORMAT,
            name="laser autofocus camera pixel format",
            value_kind="option",
            default="mono8",
            options=[
                ConfigItemOption(name="8 Bit", handle="mono8"),
                ConfigItemOption(name="10 Bit", handle="mono10"),
            ],
        ),
        config_item(
            handle=LASER_AUTOFOCUS_WARM_UP_LASER,
            name="laser autofocus warm up laser",
            value_kind="action",
            default="/api/action/laser_autofocus_warm_up_laser",
        ),
        config_item(
            handle=LASER_AUTOFOCUS_OFFSET_UM,
            name="laser autofocus offset [um]",
            value_kind="float",
            default=0.0,
        ),
        config_item(
            handle=LASER_AUTOFOCUS_CALIBRATION_Z_SPAN_MM,
            name="laser autofocus calibration z span [mm]",
            value_kind="float",
            default=0.3,
        ),
        config_item(
            handle=LASER_AUTOFOCUS_CALIBRATION_NUM_Z_STEPS,
            name="laser autofocus calibration number of z steps",
            value_kind="int",
            default=7,
        ),
        bool_config_item(
            handle=LASER_AUTOFOCUS_CALIBRATION_IS_CALIBRATED,
            name="laser autofocus is calibrated",
            default=False,
        ),
        config_item(
            handle=LASER_AUTOFOCUS_CALIBRATION_X_PEAK_POS,
            name="laser autofocus calibration: x peak pos",
            value_kind="float",
            default=0.0,
        ),
        config_item(
            handle=LASER_AUTOFOCUS_CALIBRATION_UM_PER_PX,
            name="laser autofocus calibration: um per px",
            value_kind="float",
            default=0.0,
        ),
        config_item(
            handle=LASER_AUTOFOCUS_CALIBRATION_REF_Z_MM,
            name="laser autofocus calibration: ref z in mm",
            value_kind="float",
            default=0.0,
        ),
    )


def register_filter_wheel_config() -> None:
    """Register filter wheel config items (call only if filter wheel is available)."""
    ConfigRegistry.register(
        config_item(
            handle=FILTER_WHEEL_CONFIGURATION,
            name="filter wheel configuration",
            value_kind="object",
            default=[],
            frozen=True,
            persistent=True,
        ),
    )
