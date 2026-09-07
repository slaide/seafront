"""
Core system configuration items.

This module defines the config items that are always present.
Other modules (cameras, hardware) can define their own config items.
"""

from seaconfig import ConfigItemOption

from seafront.config.registry import ConfigRegistry, config_item, bool_config_item
from seafront.config.handles import (
    SystemConfig,
    CameraConfig,
    MicrocontrollerConfig,
    CalibrationConfig,
    StorageConfig,
    ImagingConfig,
    ImageConfig,
    LaserAutofocusConfig,
    FilterWheelConfig,
    ProtocolConfig,
    IlluminationConfig,
)


def register_core_config(default_image_dir: str, default_channels: list | str, default_forbidden_areas: list | str) -> None:
    """
    Register core system config items.

    Call this after ConfigRegistry.init() with config file values.
    """

    # System
    ConfigRegistry.register(
        config_item(
            handle=SystemConfig.MICROSCOPE_NAME.value,
            name="microscope name",
            value_kind="text",
            default="squid",
            privileged=True,
            persistent=True,
        ),
        config_item(
            handle=SystemConfig.MICROSCOPE_TYPE.value,
            name="microscope type",
            value_kind="option",
            default="squid",
            options=[
                ConfigItemOption(name="SQUID", handle="squid"),
                ConfigItemOption(name="Mock", handle="mock"),
            ],
            privileged=True,
            persistent=True,
        ),
    )

    # Main camera
    ConfigRegistry.register(
        config_item(
            handle=CameraConfig.MAIN_ID.value,
            name="main camera USB ID",
            value_kind="text",
            default="CHANGE_ME",
            privileged=True,
            persistent=True,
        ),
        config_item(
            handle=CameraConfig.MAIN_DRIVER.value,
            name="main camera driver",
            value_kind="option",
            default="galaxy",
            options=[
                ConfigItemOption(name="Galaxy (Daheng)", handle="galaxy"),
                ConfigItemOption(name="ToupCam", handle="toupcam"),
            ],
            privileged=True,
            persistent=True,
        ),
        config_item(
            handle=CameraConfig.MAIN_OBJECTIVE.value,
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
            handle=CameraConfig.MAIN_TRIGGER.value,
            name="main camera trigger",
            value_kind="option",
            default="software",
            options=[
                ConfigItemOption(name="Software", handle="software"),
                ConfigItemOption(name="Hardware", handle="hardware"),
            ],
        ),
        config_item(
            handle=CameraConfig.MAIN_PIXEL_FORMAT.value,
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
            handle=CameraConfig.MAIN_IMAGE_WIDTH_PX.value,
            name="main camera image width [px]",
            value_kind="int",
            default=2500,
        ),
        config_item(
            handle=CameraConfig.MAIN_IMAGE_HEIGHT_PX.value,
            name="main camera image height [px]",
            value_kind="int",
            default=2500,
        ),
        bool_config_item(
            handle=CameraConfig.MAIN_IMAGE_FLIP_HORIZONTAL.value,
            name="main camera flip image horizontally",
            default=False,
            privileged=True,
        ),
        bool_config_item(
            handle=CameraConfig.MAIN_IMAGE_FLIP_VERTICAL.value,
            name="main camera flip image vertically",
            default=True,
            privileged=True,
        ),
        config_item(
            handle=CameraConfig.RECONNECTION_ATTEMPTS.value,
            name="camera reconnection attempts",
            value_kind="int",
            default=5,
        ),
        config_item(
            handle=CameraConfig.RECONNECTION_DELAY_MS.value,
            name="camera reconnection delay [ms]",
            value_kind="float",
            default=1000.0,
        ),
        config_item(
            handle=CameraConfig.OPERATION_RETRY_ATTEMPTS.value,
            name="camera operation retry attempts",
            value_kind="int",
            default=5,
        ),
    )

    # Microcontroller
    ConfigRegistry.register(
        config_item(
            handle=MicrocontrollerConfig.ID.value,
            name="microcontroller USB ID",
            value_kind="text",
            default="",
            privileged=True,
            persistent=True,
        ),
        config_item(
            handle=MicrocontrollerConfig.DRIVER.value,
            name="microcontroller driver",
            value_kind="option",
            default="teensy",
            options=[
                ConfigItemOption(name="Teensy/Arduino", handle="teensy"),
            ],
            privileged=True,
            persistent=True,
        ),
        config_item(
            handle=MicrocontrollerConfig.RECONNECTION_GRACE_PERIOD_MS.value,
            name="microcontroller reconnection grace period [ms]",
            value_kind="float",
            default=200.0,
        ),
        config_item(
            handle=MicrocontrollerConfig.RECONNECTION_ATTEMPTS.value,
            name="microcontroller reconnection attempts",
            value_kind="int",
            default=5,
        ),
        config_item(
            handle=MicrocontrollerConfig.RECONNECTION_DELAY_MS.value,
            name="microcontroller reconnection delay [ms]",
            value_kind="float",
            default=1000.0,
        ),
        config_item(
            handle=MicrocontrollerConfig.OPERATION_RETRY_ATTEMPTS.value,
            name="microcontroller operation retry attempts",
            value_kind="int",
            default=5,
        ),
        config_item(
            handle=MicrocontrollerConfig.OPERATION_RETRY_DELAY_MS.value,
            name="microcontroller operation retry delay [ms]",
            value_kind="float",
            default=5.0,
        ),
    )

    # Calibration
    ConfigRegistry.register(
        config_item(
            handle=CalibrationConfig.CALIBRATE_B2_HERE.value,
            name="calibrate top left of B2 here",
            value_kind="action",
            default="/api/action/calibrate_stage_xy_here",
        ),
        config_item(
            handle=CalibrationConfig.OFFSET_X_MM.value,
            name="calibration offset x [mm]",
            value_kind="float",
            default=0.0,
            persistent=True,
            privileged=True,
        ),
        config_item(
            handle=CalibrationConfig.OFFSET_Y_MM.value,
            name="calibration offset y [mm]",
            value_kind="float",
            default=0.0,
            persistent=True,
            privileged=True,
        ),
        config_item(
            handle=CalibrationConfig.OFFSET_Z_MM.value,
            name="calibration offset z [mm]",
            value_kind="float",
            default=0.0,
            persistent=True,
            privileged=True,
        ),
    )

    # Device lock (expert escape hatch; the UI otherwise only breaks a lock via the cancel dialog)
    ConfigRegistry.register(
        config_item(
            handle=SystemConfig.BREAK_LOCK.value,
            name="break the device lock (frees another client's lease; stops nothing)",
            value_kind="action",
            default="/api/lock/break",
        ),
    )

    # Storage
    ConfigRegistry.register(
        config_item(
            handle=StorageConfig.BASE_IMAGE_OUTPUT_DIR.value,
            name="base output storage directory (s3: spill directory)",
            value_kind="text",
            default=default_image_dir,
            persistent=True,
        ),
        config_item(
            handle=StorageConfig.WRITER_THREADS.value,
            name="image writer threads (1 for a single HDD without RAID; raise for SSD/RAID/network)",
            value_kind="int",
            default=1,
            privileged=True,
            persistent=True,
        ),
        config_item(
            handle=StorageConfig.BACKEND.value,
            name="image storage backend",
            value_kind="option",
            default="disk",
            options=[
                ConfigItemOption(name="local disk", handle="disk"),
                ConfigItemOption(name="S3 bucket (direct upload, spill to disk when behind)", handle="s3"),
            ],
            privileged=True,
            persistent=True,
        ),
        config_item(
            handle=StorageConfig.S3_BUCKET.value,
            name="s3: bucket",
            value_kind="text",
            default="",
            privileged=True,
            persistent=True,
        ),
        config_item(
            handle=StorageConfig.S3_PREFIX.value,
            name="s3: key prefix",
            value_kind="text",
            default="",
            privileged=True,
            persistent=True,
        ),
        config_item(
            handle=StorageConfig.S3_ENDPOINT_URL.value,
            name="s3: endpoint URL (empty for AWS; set for MinIO/Ceph)",
            value_kind="text",
            default="",
            privileged=True,
            persistent=True,
        ),
        config_item(
            handle=StorageConfig.S3_REGION.value,
            name="s3: region (empty for default)",
            value_kind="text",
            default="",
            privileged=True,
            persistent=True,
        ),
        config_item(
            handle=StorageConfig.S3_UPLOAD_THREADS.value,
            name="s3: upload threads (tune against the link)",
            value_kind="int",
            default=1,
            privileged=True,
            persistent=True,
        ),
        config_item(
            handle=StorageConfig.S3_SPILL_THRESHOLD.value,
            name="s3: images held in memory before spilling to the spill directory",
            value_kind="int",
            default=16,
            privileged=True,
            persistent=True,
        ),
    )

    # Imaging
    ConfigRegistry.register(
        config_item(
            handle=ImagingConfig.CHANNELS.value,
            name="imaging channels configuration",
            value_kind="object",
            default=default_channels,
            privileged=True,
            persistent=True,
        ),
        config_item(
            handle=ImagingConfig.ORDER.value,
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
            handle=ImageConfig.FILE_PAD_LOW.value,
            name="image file pad low",
            default=True,
        ),
        bool_config_item(
            handle=ImageConfig.FILENAME_USE_CHANNEL_NAME.value,
            name="image filename use channel name",
            default=True,
        ),
        config_item(
            handle=ImageConfig.FILENAME_XY_INDEX_START.value,
            name="image filename xy index start",
            value_kind="int",
            default=0,
        ),
        config_item(
            handle=ImageConfig.FILENAME_Z_INDEX_START.value,
            name="image filename z index start",
            value_kind="int",
            default=0,
        ),
        config_item(
            handle=ImageConfig.FILENAME_SITE_INDEX_START.value,
            name="image filename site index start",
            value_kind="int",
            default=1,
        ),
        bool_config_item(
            handle=ImageConfig.FILENAME_ZERO_PAD_COLUMN.value,
            name="image filename zero pad column index",
            default=True,
        ),
    )

    # Illumination
    ConfigRegistry.register(
        config_item(
            handle=IlluminationConfig.TURN_OFF_ALL.value,
            name="turn off all illumination",
            value_kind="action",
            default="/api/action/turn_off_all_illumination",
        ),
    )

    # Protocol
    ConfigRegistry.register(
        config_item(
            handle=ProtocolConfig.FORBIDDEN_AREAS.value,
            name="forbidden areas",
            value_kind="object",
            default=default_forbidden_areas,
            persistent=True,
        ),
    )

    # Filter wheel (base availability - config details registered if available)
    ConfigRegistry.register(
        bool_config_item(
            handle=FilterWheelConfig.AVAILABLE.value,
            name="filter wheel system available",
            default=False,
            privileged=True,
            persistent=True,
        ),
    )

    # Laser autofocus (base availability)
    ConfigRegistry.register(
        bool_config_item(
            handle=LaserAutofocusConfig.AVAILABLE.value,
            name="laser autofocus system available",
            default=False,
            privileged=True,
            persistent=True,
        ),
    )


def register_laser_autofocus_config() -> None:
    """Register laser autofocus config items (call only if LAF is available)."""
    ConfigRegistry.register(
        config_item(
            handle=LaserAutofocusConfig.CAMERA_ID.value,
            name="laser autofocus camera USB ID",
            value_kind="text",
            default="CHANGE_ME",
            privileged=True,
            persistent=True,
        ),
        config_item(
            handle=LaserAutofocusConfig.CAMERA_DRIVER.value,
            name="laser autofocus camera driver",
            value_kind="option",
            default="galaxy",
            options=[
                ConfigItemOption(name="Galaxy (Daheng)", handle="galaxy"),
                ConfigItemOption(name="ToupCam", handle="toupcam"),
            ],
            privileged=True,
            persistent=True,
        ),
        config_item(
            handle=LaserAutofocusConfig.EXPOSURE_TIME_MS.value,
            name="laser autofocus exposure time [ms]",
            value_kind="float",
            default=5.0,
        ),
        config_item(
            handle=LaserAutofocusConfig.CAMERA_ANALOG_GAIN.value,
            name="laser autofocus camera analog gain",
            value_kind="float",
            default=0.0,
        ),
        bool_config_item(
            handle=LaserAutofocusConfig.USE_RIGHT_DOT.value,
            name="laser autofocus use right dot",
            default=True,
        ),
        config_item(
            handle=LaserAutofocusConfig.CAMERA_PIXEL_FORMAT.value,
            name="laser autofocus camera pixel format",
            value_kind="option",
            default="mono8",
            options=[
                ConfigItemOption(name="8 Bit", handle="mono8"),
                ConfigItemOption(name="10 Bit", handle="mono10"),
            ],
        ),
        config_item(
            handle=LaserAutofocusConfig.WARM_UP_LASER.value,
            name="laser autofocus warm up laser",
            value_kind="action",
            default="/api/action/laser_autofocus_warm_up_laser",
        ),
        config_item(
            handle=LaserAutofocusConfig.OFFSET_UM.value,
            name="laser autofocus offset [um]",
            value_kind="float",
            default=0.0,
            privileged=True,
            persistent=True,
        ),
        config_item(
            handle=LaserAutofocusConfig.CALIBRATION_Z_SPAN_MM.value,
            name="laser autofocus calibration z span [mm]",
            value_kind="float",
            default=0.3,
        ),
        config_item(
            handle=LaserAutofocusConfig.CALIBRATION_NUM_Z_STEPS.value,
            name="laser autofocus calibration number of z steps",
            value_kind="int",
            default=7,
        ),
        bool_config_item(
            handle=LaserAutofocusConfig.CALIBRATION_IS_CALIBRATED.value,
            name="laser autofocus is calibrated",
            default=False,
            privileged=True,
            persistent=True,
        ),
        config_item(
            handle=LaserAutofocusConfig.CALIBRATION_X_PEAK_POS.value,
            name="laser autofocus calibration: x peak pos",
            value_kind="float",
            default=0.0,
            privileged=True,
            persistent=True,
        ),
        config_item(
            handle=LaserAutofocusConfig.CALIBRATION_UM_PER_PX.value,
            name="laser autofocus calibration: um per px",
            value_kind="float",
            default=0.0,
            privileged=True,
            persistent=True,
        ),
        config_item(
            handle=LaserAutofocusConfig.CALIBRATION_REF_Z_MM.value,
            name="laser autofocus calibration: ref z in mm",
            value_kind="float",
            default=0.0,
            privileged=True,
            persistent=True,
        ),
    )


def register_filter_wheel_config() -> None:
    """Register filter wheel config items (call only if filter wheel is available)."""
    ConfigRegistry.register(
        config_item(
            handle=FilterWheelConfig.CONFIGURATION.value,
            name="filter wheel configuration",
            value_kind="object",
            default=[],
            privileged=True,
            persistent=True,
        ),
    )
