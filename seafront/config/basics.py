import hashlib
import json
import os
import re
import typing as tp
from pathlib import Path

import json5
from pydantic import BaseModel, Field
from seaconfig import AcquisitionConfig, ConfigItem, ConfigItemOption

# Import handle enums for config defaults
from seafront.config.handles import (
    CameraConfig,
    CalibrationConfig,
    FilterWheelConfig,
    ImagingConfig,
    ImageConfig,
    IlluminationConfig,
    LaserAutofocusConfig,
    MicrocontrollerConfig,
    ProtocolConfig,
    StorageConfig,
    SystemConfig,
)

CameraDriver = tp.Literal["galaxy", "toupcam"]
MicroscopeType = tp.Literal["squid", "mock"]
ImagingOrder = tp.Literal["z_order", "wavelength_order", "protocol_order"]


def set_config_item_int(
    config_items: list[ConfigItem],
    handle: str,
    value: int,
    name: str | None = None,
    frozen: bool = False,
) -> None:
    """
    Set or update an integer config item in the list.

    If an item with the given handle exists, updates its value.
    If not found, creates and appends a new ConfigItem.

    Args:
        config_items: List of ConfigItem objects to modify in-place
        handle: Unique identifier for the config item
        value: Integer value to set
        name: Display name (used only when creating new item)
        frozen: Whether item can be modified by user
    """
    for item in config_items:
        if item.handle == handle and item.value_kind == "int":
            item.value = value
            return
    # Item not found, create and append
    config_items.append(
        ConfigItem(
            handle=handle,
            name=name or handle,
            value_kind="int",
            value=value,
            frozen=frozen,
        )
    )


def set_config_item_float(
    config_items: list[ConfigItem],
    handle: str,
    value: float,
    name: str | None = None,
    frozen: bool = False,
) -> None:
    """
    Set or update a float config item in the list.

    If an item with the given handle exists, updates its value.
    If not found, creates and appends a new ConfigItem.

    Args:
        config_items: List of ConfigItem objects to modify in-place
        handle: Unique identifier for the config item
        value: Float value to set
        name: Display name (used only when creating new item)
        frozen: Whether item can be modified by user
    """
    for item in config_items:
        if item.handle == handle and item.value_kind == "float":
            item.value = value
            return
    # Item not found, create and append
    config_items.append(
        ConfigItem(
            handle=handle,
            name=name or handle,
            value_kind="float",
            value=value,
            frozen=frozen,
        )
    )


def set_config_item_text(
    config_items: list[ConfigItem],
    handle: str,
    value: str,
    name: str | None = None,
    frozen: bool = False,
) -> None:
    """
    Set or update a text config item in the list.

    If an item with the given handle exists, updates its value.
    If not found, creates and appends a new ConfigItem.

    Args:
        config_items: List of ConfigItem objects to modify in-place
        handle: Unique identifier for the config item
        value: String value to set
        name: Display name (used only when creating new item)
        frozen: Whether item can be modified by user
    """
    for item in config_items:
        if item.handle == handle and item.value_kind == "text":
            item.value = value
            return
    # Item not found, create and append
    config_items.append(
        ConfigItem(
            handle=handle,
            name=name or handle,
            value_kind="text",
            value=value,
            frozen=frozen,
        )
    )


def set_config_item_option(
    config_items: list[ConfigItem],
    handle: str,
    value: str,
    options: list[ConfigItemOption],
    name: str | None = None,
    frozen: bool = False,
) -> None:
    """
    Set or update an option config item in the list.

    If an item with the given handle exists, updates its value and options.
    If not found, creates and appends a new ConfigItem.

    Args:
        config_items: List of ConfigItem objects to modify in-place
        handle: Unique identifier for the config item
        value: Option value to set (must be a valid handle from options list)
        options: List of ConfigItemOption to set as available options
        name: Display name (used only when creating new item)
        frozen: Whether item can be modified by user
    """
    for item in config_items:
        if item.handle == handle and item.value_kind == "option":
            item.value = value
            item.options = options
            return
    # Item not found, create and append
    config_items.append(
        ConfigItem(
            handle=handle,
            name=name or handle,
            value_kind="option",
            value=value,
            options=options,
            frozen=frozen,
        )
    )


def set_config_item_bool(
    config_items: list[ConfigItem],
    handle: str,
    value: bool,
    name: str | None = None,
    frozen: bool = False,
) -> None:
    """
    Set or update a boolean option config item in the list.

    Convenience wrapper for option-type boolean values (yes/no).
    If an item with the given handle exists, updates its value.
    If not found, creates and appends a new ConfigItem.

    Args:
        config_items: List of ConfigItem objects to modify in-place
        handle: Unique identifier for the config item
        value: Boolean value to set
        name: Display name (used only when creating new item)
        frozen: Whether item can be modified by user
    """
    str_value = "yes" if value else "no"
    set_config_item_option(
        config_items,
        handle,
        str_value,
        ConfigItemOption.get_bool_options(),
        name=name,
        frozen=frozen,
    )


class PowerCalibration(BaseModel):
    """Power calibration data for an illumination source"""
    dac_percent: list[float]
    "DAC percentage values (0-100)"
    optical_power_mw: list[float]
    "Corresponding optical power in milliwatts (used for calibration curve generation)"

    def validate_data(self) -> None:
        """Validate that calibration data is consistent"""
        if len(self.dac_percent) != len(self.optical_power_mw):
            raise ValueError("DAC percent and optical power arrays must have same length")
        if len(self.dac_percent) < 2:
            raise ValueError("Calibration data must have at least 2 points")
        if not all(0 <= p <= 100 for p in self.dac_percent):
            raise ValueError("DAC percent values must be between 0 and 100")
        if not all(p >= 0 for p in self.optical_power_mw):
            raise ValueError("Optical power values must be non-negative")


class ChannelConfig(BaseModel):
    """Configuration for a single imaging channel"""
    name: str
    "Display name for the channel (e.g. 'Fluorescence 405 nm Ex')"
    handle: str
    "Internal handle for the channel (e.g. 'fluo405')"
    source_slot: int
    "Illumination source slot number (e.g. 11 for ILLUMINATION_SOURCE_SLOT_11)"
    use_power_calibration: bool = False
    "Whether to use power calibration for this channel"
    power_calibration: PowerCalibration | None = None
    "Power calibration data (required if use_power_calibration is True)"

    def validate_calibration(self) -> None:
        """Validate that calibration settings are consistent"""
        if self.use_power_calibration and self.power_calibration is None:
            raise ValueError(f"Channel {self.handle}: use_power_calibration=True but no power_calibration provided")
        if self.power_calibration is not None:
            self.power_calibration.validate_data()


class FilterConfig(BaseModel):
    """Configuration for a single filter wheel position"""
    name: str
    "Display name for the filter (e.g. 'DAPI Filter')"
    handle: str
    "Internal handle for the filter (e.g. 'dapi')"
    slot: int
    "Filter wheel slot position (1-8)"


class CriticalMachineConfig(BaseModel):
    microscope_name: str
    microscope_type: MicroscopeType

    main_camera_id: str
    main_camera_driver: CameraDriver = "galaxy"

    microcontroller_id: str | None = None
    "USB serial number for the microcontroller (required for squid microscope type)"

    base_image_output_dir: str
    calibration_offset_x_mm: float
    calibration_offset_y_mm: float
    calibration_offset_z_mm: float

    forbidden_areas: str | None = None
    "JSON string defining forbidden areas as AABBs in physical coordinates (mm)"

    laser_autofocus_available: tp.Literal["yes", "no"] | None = None
    laser_autofocus_camera_id: str | None = None
    "if laser_autofocus_available is yes, then this must be present"
    laser_autofocus_camera_driver: CameraDriver = "galaxy"

    filter_wheel_available: tp.Literal["yes", "no"] | None = None

    channels: str = Field(default="[]")
    "Available imaging channels with their illumination source slots (JSON-encoded string)"

    filters: str = Field(default="[]")
    "Available filters with their wheel positions (JSON-encoded string)"


class ServerConfig(BaseModel):
    port: int = 5000
    microscopes: list[CriticalMachineConfig] = Field(
        default_factory=lambda: [GlobalConfigHandler.CRITICAL_MACHINE_DEFAULTS()]
    )


class GlobalConfigHandler:
    _seafront_home: Path | None = None

    _config_list: list[ConfigItem] | None = None
    _current_microscope_name: str | None = None

    @staticmethod
    def home() -> Path:
        "get path to seafront home directory"

        if GlobalConfigHandler._seafront_home is not None:
            SEAFRONT_HOME = GlobalConfigHandler._seafront_home
        else:
            # construct SEAFRONT_HOME, from $SEAFRONT_HOME or $HOME/seafront
            env_seafront_home = os.getenv("SEAFRONT_HOME")
            if env_seafront_home is not None:
                SEAFRONT_HOME = Path(env_seafront_home)

            else:
                # get home dir of user
                home_dir = os.environ.get("HOME")
                if home_dir is None:
                    raise ValueError("could not find home directory")

                SEAFRONT_HOME = Path(home_dir) / "seafront"

            # ensure home path exists
            if not SEAFRONT_HOME.exists():
                SEAFRONT_HOME.mkdir(parents=True)

            # cache directory path
            GlobalConfigHandler._seafront_home = SEAFRONT_HOME

            # create default image output dir
            DEFAULT_IMAGE_STORAGE_DIR = SEAFRONT_HOME / "images"
            if not DEFAULT_IMAGE_STORAGE_DIR.exists():
                DEFAULT_IMAGE_STORAGE_DIR.mkdir(parents=True)

            # ensure config file is present in home
            _ = GlobalConfigHandler.home_config()
            # Note: acquisition config dir is created per-microscope, not during home() init

        return SEAFRONT_HOME

    @staticmethod
    def home_config() -> Path:
        """
        get path to [default] machine config.

        creates a default file if none is present.
        """

        CONFIG_FILE_PATH = GlobalConfigHandler.home() / "config.json"  # type: ignore
        if not CONFIG_FILE_PATH.exists():
            # create config file
            with CONFIG_FILE_PATH.open("w") as f:
                json.dump(ServerConfig().model_dump(), f, indent=4)

        return CONFIG_FILE_PATH

    @staticmethod
    def store():
        """
        write current global config to disk
        """

        CONFIG_FILE_PATH = GlobalConfigHandler.home_config()

        assert GlobalConfigHandler._config_list is not None
        critical_machine_config = GlobalConfigHandler.CRITICAL_MACHINE_DEFAULTS().model_dump()

        with CONFIG_FILE_PATH.open("r") as config_file:
            current_file_contents = json5.load(config_file)
            server_config = ServerConfig(**current_file_contents)

        # store critical config items from current config
        store_dict = {}
        current_config = GlobalConfigHandler.get_dict()

        from seafront.config.handles import (
            CalibrationConfig,
            CameraConfig,
            FilterWheelConfig,
            ImagingConfig,
            LaserAutofocusConfig,
            MicrocontrollerConfig,
            ProtocolConfig,
            StorageConfig,
            SystemConfig,
        )

        handle_lookup: dict[str, str] = {
            "microscope_name": SystemConfig.MICROSCOPE_NAME.value,
            "main_camera_id": CameraConfig.MAIN_ID.value,
            "main_camera_driver": CameraConfig.MAIN_DRIVER.value,
            "microcontroller_id": MicrocontrollerConfig.ID.value,
            "base_image_output_dir": StorageConfig.BASE_IMAGE_OUTPUT_DIR.value,
            "calibration_offset_x_mm": CalibrationConfig.OFFSET_X_MM.value,
            "calibration_offset_y_mm": CalibrationConfig.OFFSET_Y_MM.value,
            "calibration_offset_z_mm": CalibrationConfig.OFFSET_Z_MM.value,
            "laser_autofocus_available": LaserAutofocusConfig.AVAILABLE.value,
            "laser_autofocus_camera_id": LaserAutofocusConfig.CAMERA_ID.value,
            "laser_autofocus_camera_driver": LaserAutofocusConfig.CAMERA_DRIVER.value,
            "filter_wheel_available": FilterWheelConfig.AVAILABLE.value,
            "filters": FilterWheelConfig.CONFIGURATION.value,
            "channels": ImagingConfig.CHANNELS.value,
            "forbidden_areas": ProtocolConfig.FORBIDDEN_AREAS.value,
        }

        def _get_config_item(handle_key: str):
            lookup_key = handle_lookup.get(handle_key, handle_key)
            return current_config.get(lookup_key)

        # Find the existing microscope config to use as fallback
        current_microscope_name_item = _get_config_item("microscope_name")
        current_microscope_name = (
            current_microscope_name_item.value if current_microscope_name_item else None
        )
        existing_microscope_config = None
        if current_microscope_name:
            for microscope_config in server_config.microscopes:
                if microscope_config.microscope_name == current_microscope_name:
                    existing_microscope_config = microscope_config
                    break

        for key in critical_machine_config.keys():
            config_item = _get_config_item(key)
            if config_item is not None:
                store_dict[key] = config_item.value
            elif existing_microscope_config and hasattr(existing_microscope_config, key):
                # Fallback to existing config value for missing keys
                store_dict[key] = getattr(existing_microscope_config, key)
            else:
                store_dict[key] = critical_machine_config[key]

        store_config = CriticalMachineConfig(**store_dict)

        # update target microscope config
        for microscope_i, microscope_config in enumerate(server_config.microscopes):
            if microscope_config.microscope_name == store_config.microscope_name:
                server_config.microscopes[microscope_i] = store_config
                break

        with CONFIG_FILE_PATH.open("w") as f:
            json.dump(server_config.model_dump(), f, indent=4)

    @staticmethod
    def get_config_list() -> list[Path]:
        "get list of all config files in home_acquisition_config_dir"
        ret = []
        for config_file in GlobalConfigHandler.home_acquisition_config_dir().glob("*.json"):
            ret.append(config_file)

        return ret

    @staticmethod
    def add_config(config: AcquisitionConfig, filename: str, overwrite_on_conflict: bool = False):
        if len(filename) == 0:
            raise ValueError("config filename must not be empty")

        filepath = (GlobalConfigHandler.home_acquisition_config_dir() / filename).with_suffix(
            ".json"
        )

        file_already_exists = filepath in GlobalConfigHandler.get_config_list()
        if (not overwrite_on_conflict) and file_already_exists:
            raise RuntimeError(
                f"error - saving config with filename {filepath!s}, which already exists"
            )

        with filepath.open("w+") as f:
            json.dump(config.dict(), f, indent=4)

    @staticmethod
    def _sanitize_microscope_name_for_directory(microscope_name: str) -> str:
        """
        Convert microscope name to safe directory name format.
        Process: lowercase, replace spaces with dashes, remove non-alphanumeric chars,
        append 6-digit hash of original name.
        
        Raises ValueError for empty or whitespace-only names.
        """
        # Validate input - reject empty or whitespace-only names
        if not microscope_name or not microscope_name.strip():
            raise ValueError("Microscope name cannot be empty or contain only whitespace")

        # Generate 6-digit hash of original name
        name_hash = hashlib.sha256(microscope_name.encode()).hexdigest()[:6]

        # Convert to lowercase and replace spaces with dashes
        sanitized = microscope_name.lower().replace(' ', '-')

        # Remove non-alphanumeric characters (keep dashes)
        sanitized = re.sub(r'[^a-z0-9\-]', '', sanitized)

        # Remove consecutive dashes and trim dashes from ends
        sanitized = re.sub(r'-+', '-', sanitized).strip('-')

        # If sanitization resulted in empty string (e.g., name was only special chars),
        # use 'microscope' as base name
        if not sanitized:
            sanitized = 'microscope'

        return f"{sanitized}-{name_hash}"

    @staticmethod
    def home_acquisition_config_dir(microscope_name: str | None = None) -> Path:
        """
        get path to directory containing [user-defined] acquisition configurations

        will create the directory it not already present.
        if microscope_name is provided, returns microscope-specific subdirectory.
        if no microscope_name provided, uses current microscope from GlobalConfigHandler.
        
        Raises ValueError if no microscope name is available (shared directory fallback removed).
        """
        base_dir = GlobalConfigHandler.home() / "acquisition_configs"  # type: ignore

        # Use provided microscope name or fall back to current microscope
        # Empty strings are treated as None (falsy)
        target_microscope = (microscope_name if microscope_name and microscope_name.strip()
                           else GlobalConfigHandler._current_microscope_name)

        if target_microscope is None:
            raise ValueError("Microscope name is required - shared directory fallback has been removed")

        # Use microscope-specific subdirectory with safe directory name
        safe_dir_name = GlobalConfigHandler._sanitize_microscope_name_for_directory(target_microscope)
        config_dir = base_dir / safe_dir_name

        if not config_dir.exists():
            config_dir.mkdir(parents=True)

        return config_dir

    @staticmethod
    def CRITICAL_MACHINE_DEFAULTS() -> CriticalMachineConfig:
        # Example power calibration for brightfield LED - non-linear response
        bfled_calibration = PowerCalibration(
            dac_percent=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
            optical_power_mw=[0.0, 0.2, 0.8, 1.8, 3.2, 5.0, 7.2, 9.8, 12.8, 16.2, 20.0, 24.2, 28.8, 33.8, 39.2, 45.0, 51.2, 57.8, 64.8, 72.2, 80.0]
        )

        # Default channel configuration with traditional illumination sources using proper constructors
        default_channels = [
            ChannelConfig(name="Fluorescence 405 nm Ex", handle="fluo405", source_slot=11),
            ChannelConfig(name="Fluorescence 488 nm Ex", handle="fluo488", source_slot=12),
            ChannelConfig(name="Fluorescence 561 nm Ex", handle="fluo561", source_slot=14),
            ChannelConfig(name="Fluorescence 638 nm Ex", handle="fluo638", source_slot=13),
            ChannelConfig(name="Fluorescence 730 nm Ex", handle="fluo730", source_slot=15),
            ChannelConfig(name="BF LED matrix full", handle="bfledfull", source_slot=0,
                         use_power_calibration=True, power_calibration=bfled_calibration),
            ChannelConfig(name="BF LED matrix left half", handle="bfledleft", source_slot=1),
            ChannelConfig(name="BF LED matrix right half", handle="bfledright", source_slot=2)
        ]

        # Default filter configuration - empty by default since filter wheel is optional
        default_filters = []

        # Default forbidden areas - typical problematic areas for SQUID microscope
        default_forbidden_areas = [
            {
                "name": "Top-left plate corner",
                "min_x_mm": 0.0,
                "max_x_mm": 8.0,
                "min_y_mm": 77.0,
                "max_y_mm": 85.0,
                "reason": "Plate holder interference risk"
            },
            {
                "name": "Top-right plate corner",
                "min_x_mm": 119.0,
                "max_x_mm": 127.0,
                "min_y_mm": 77.0,
                "max_y_mm": 85.0,
                "reason": "Plate holder interference risk"
            },
            {
                "name": "Bottom-left plate corner",
                "min_x_mm": 0.0,
                "max_x_mm": 8.0,
                "min_y_mm": 0.0,
                "max_y_mm": 8.0,
                "reason": "Plate holder interference risk"
            },
            {
                "name": "Bottom-right plate corner",
                "min_x_mm": 119.0,
                "max_x_mm": 127.0,
                "min_y_mm": 0.0,
                "max_y_mm": 8.0,
                "reason": "Plate holder interference risk"
            }
        ]

        # Convert to JSON strings for storage
        channels_json = json.dumps([ch.model_dump() for ch in default_channels])
        filters_json = json.dumps([f.model_dump() for f in default_filters])
        forbidden_areas_json = json.dumps(default_forbidden_areas)

        return CriticalMachineConfig(
            main_camera_id="CHANGE_ME",
            laser_autofocus_camera_id="CHANGE_ME",
            microcontroller_id="CHANGE_ME",
            microscope_name="squid",
            microscope_type="squid",
            base_image_output_dir=str(GlobalConfigHandler.home() / "images"),
            laser_autofocus_available="yes",
            filter_wheel_available="no",
            calibration_offset_x_mm=0.0,
            calibration_offset_y_mm=0.0,
            calibration_offset_z_mm=0.0,
            forbidden_areas=forbidden_areas_json,
            channels=channels_json,
            filters=filters_json,
        )

    @staticmethod
    def _defaults(microscope_name: str | None = None) -> list[ConfigItem]:
        """
        get a list of all the low level machine settings

        these settings may be changed on the client side, for individual acquisitions
        (though clearly, this is highly advanced stuff, and may cause irreperable hardware damage!)
        """

        # load config file
        with GlobalConfigHandler.home_config().open("r") as f:
            server_config_json=json5.load(f)
            server_config = ServerConfig(**server_config_json)
            if len(server_config.microscopes)==0:
                raise ValueError("no microscope found in server config")

            # Select microscope by name or default to first
            critical_machine_config: CriticalMachineConfig
            if microscope_name is not None:
                # Find microscope by name
                matching_microscopes = [m for m in server_config.microscopes if m.microscope_name == microscope_name]
                if len(matching_microscopes) == 0:
                    available_names = [m.microscope_name for m in server_config.microscopes]
                    raise ValueError(f"microscope '{microscope_name}' not found. Available: {available_names}")
                critical_machine_config = matching_microscopes[0]
            else:
                # Default to first microscope
                critical_machine_config = server_config.microscopes[0]

        main_camera_attributes = [
            ConfigItem(
                name="main camera USB ID",
                handle=CameraConfig.MAIN_ID.value,
                value_kind="text",
                value=critical_machine_config.main_camera_id,
                frozen=True,
            ),
            ConfigItem(
                name="main camera driver",
                handle=CameraConfig.MAIN_DRIVER.value,
                value_kind="option",
                value=critical_machine_config.main_camera_driver,
                options=[
                    ConfigItemOption(
                        name="Galaxy (Daheng)",
                        handle="galaxy",
                    ),
                    ConfigItemOption(
                        name="ToupCam",
                        handle="toupcam",
                    ),
                ],
                frozen=True,
            ),
            ConfigItem(
                name="main camera objective",
                handle=CameraConfig.MAIN_OBJECTIVE.value,
                value_kind="option",
                value="20xolympus",
                options=[
                    ConfigItemOption(
                        name="4x Olympus",
                        handle="4xolympus",
                        info={
                            "magnification": 4,
                        },
                    ),
                    ConfigItemOption(
                        name="10x Olympus",
                        handle="10xolympus",
                        info={
                            "magnification": 10,
                        },
                    ),
                    ConfigItemOption(
                        name="20x Olympus",
                        handle="20xolympus",
                        info={
                            "magnification": 20,
                        },
                    ),
                ],
            ),
            ConfigItem(
                name="main camera trigger",
                handle=CameraConfig.MAIN_TRIGGER.value,
                value_kind="option",
                value="software",
                options=[
                    ConfigItemOption(
                        name="Software",
                        handle="software",
                    ),
                    ConfigItemOption(
                        name="Hardware",
                        handle="hardware",
                    ),
                ],
            ),
            ConfigItem(
                name="main camera pixel format",
                handle=CameraConfig.MAIN_PIXEL_FORMAT.value,
                value_kind="option",
                value="mono8",
                options=[
                    ConfigItemOption(
                        name="8 Bit",
                        handle="mono8",
                    ),
                    ConfigItemOption(
                        name="12 Bit",
                        handle="mono12",
                    ),
                    ConfigItemOption(
                        name="16 Bit",
                        handle="mono16",
                    ),
                ],
            ),
            ConfigItem(
                name="main camera image width [px]",
                handle=CameraConfig.MAIN_IMAGE_WIDTH_PX.value,
                value_kind="int",
                value=2500,
            ),
            ConfigItem(
                name="main camera image height [px]",
                handle=CameraConfig.MAIN_IMAGE_HEIGHT_PX.value,
                value_kind="int",
                value=2500,
            ),
            ConfigItem(
                name="main camera flip image horizontally",
                handle=CameraConfig.MAIN_IMAGE_FLIP_HORIZONTAL.value,
                value_kind="option",
                value="no",
                options=ConfigItemOption.get_bool_options(),
                frozen=True,
            ),
            ConfigItem(
                name="main camera flip image vertically",
                handle=CameraConfig.MAIN_IMAGE_FLIP_VERTICAL.value,
                value_kind="option",
                value="yes",
                options=ConfigItemOption.get_bool_options(),
                frozen=True,
            ),
        ]

        laser_autofocus_system_available_attribute = ConfigItem(
            name="laser autofocus system available",
            handle=LaserAutofocusConfig.AVAILABLE.value,
            value_kind="option",
            value=critical_machine_config.laser_autofocus_available or "no",
            options=ConfigItemOption.get_bool_options(),
            frozen=True,
        )

        filter_wheel_system_available_attribute = ConfigItem(
            name="filter wheel system available",
            handle=FilterWheelConfig.AVAILABLE.value,
            value_kind="option",
            value=critical_machine_config.filter_wheel_available or "no",
            options=ConfigItemOption.get_bool_options(),
            frozen=True,
        )

        if laser_autofocus_system_available_attribute.boolvalue:
            if critical_machine_config.laser_autofocus_camera_id is None:
                raise ValueError("laser autofocus available but no autofocus camera USB ID provided")

            laser_autofocus_system_attributes = [
                ConfigItem(
                    name="laser autofocus camera USB ID",
                    handle=LaserAutofocusConfig.CAMERA_ID.value,
                    value_kind="text",
                    value=critical_machine_config.laser_autofocus_camera_id,
                    frozen=True,
                ),
                ConfigItem(
                    name="laser autofocus camera driver",
                    handle=LaserAutofocusConfig.CAMERA_DRIVER.value,
                    value_kind="option",
                    value=critical_machine_config.laser_autofocus_camera_driver,
                    options=[
                        ConfigItemOption(
                            name="Galaxy (Daheng)",
                            handle="galaxy",
                        ),
                        ConfigItemOption(
                            name="ToupCam",
                            handle="toupcam",
                        ),
                    ],
                    frozen=True,
                ),
                ConfigItem(
                    name="laser autofocus exposure time [ms]",
                    handle=LaserAutofocusConfig.EXPOSURE_TIME_MS.value,
                    value_kind="float",
                    value=5.0,
                ),
                ConfigItem(
                    name="laser autofocus camera analog gain",
                    handle=LaserAutofocusConfig.CAMERA_ANALOG_GAIN.value,
                    value_kind="float",
                    value=0.0,
                ),
                ConfigItem(
                    name="laser autofocus use glass top",
                    handle=LaserAutofocusConfig.USE_GLASS_TOP.value,
                    value_kind="option",
                    value="no",
                    options=ConfigItemOption.get_bool_options(),
                ),
                ConfigItem(
                    name="laser autofocus camera pixel format",
                    handle=LaserAutofocusConfig.CAMERA_PIXEL_FORMAT.value,
                    value_kind="option",
                    value="mono8",
                    options=[
                        ConfigItemOption(
                            name="8 Bit",
                            handle="mono8",
                        ),
                        ConfigItemOption(
                            name="10 Bit",
                            handle="mono10",
                        ),
                    ],
                ),
                ConfigItem(
                    name="laser autofocus warm up laser",
                    handle=LaserAutofocusConfig.WARM_UP_LASER.value,
                    value_kind="action",
                    value="/api/action/laser_autofocus_warm_up_laser",
                ),
                # offset added to measured z displacement
                ConfigItem(
                    name="laser autofocus offset [um]",
                    handle=LaserAutofocusConfig.OFFSET_UM.value,
                    value_kind="float",
                    value=0.0,
                ),
                # calibration z span
                ConfigItem(
                    name="laser autofocus calibration z span [mm]",
                    handle=LaserAutofocusConfig.CALIBRATION_Z_SPAN_MM.value,
                    value_kind="float",
                    value=0.3,
                ),
                # calibration number of z steps
                ConfigItem(
                    name="laser autofocus calibration number of z steps",
                    handle=LaserAutofocusConfig.CALIBRATION_NUM_Z_STEPS.value,
                    value_kind="int",
                    value=7,
                ),
                # is calibrated flag
                ConfigItem(
                    name="laser autofocus is calibrated",
                    handle=LaserAutofocusConfig.CALIBRATION_IS_CALIBRATED.value,
                    value_kind="option",
                    value="no",
                    options=ConfigItemOption.get_bool_options(),
                ),
                # calibrated x on sensor
                ConfigItem(
                    name="laser autofocus calibration: x peak pos",
                    handle=LaserAutofocusConfig.CALIBRATION_X_PEAK_POS.value,
                    value_kind="float",
                    value=0.0,
                ),
                # calibrated um/px on sensor
                ConfigItem(
                    name="laser autofocus calibration: um per px",
                    handle=LaserAutofocusConfig.CALIBRATION_UM_PER_PX.value,
                    value_kind="float",
                    value=0.0,
                ),
                # z coordinate at time of calibration
                ConfigItem(
                    name="laser autofocus calibration: ref z in mm",
                    handle=LaserAutofocusConfig.CALIBRATION_REF_Z_MM.value,
                    value_kind="float",
                    value=0.0,
                ),
            ]
        else:
            laser_autofocus_system_attributes = []

        if filter_wheel_system_available_attribute.boolvalue:
            filter_wheel_system_attributes = [
                ConfigItem(
                    name="filter wheel configuration",
                    handle=FilterWheelConfig.CONFIGURATION.value,
                    value_kind="text",
                    value=critical_machine_config.filters,
                    frozen=True,
                ),
            ]
        else:
            filter_wheel_system_attributes = []

        ret = [
            ConfigItem(
                name="microscope name",
                handle=SystemConfig.MICROSCOPE_NAME.value,
                value_kind="text",
                value=critical_machine_config.microscope_name,
                frozen=True,
            ),
            ConfigItem(
                name="calibrate top left of B2 here",
                handle=CalibrationConfig.CALIBRATE_B2_HERE.value,
                value_kind="action",
                value="/api/action/calibrate_stage_xy_here",
            ),
            ConfigItem(
                name="calibration offset x [mm]",
                handle=CalibrationConfig.OFFSET_X_MM.value,
                value_kind="float",
                value=critical_machine_config.calibration_offset_x_mm,
            ),
            ConfigItem(
                name="calibration offset y [mm]",
                handle=CalibrationConfig.OFFSET_Y_MM.value,
                value_kind="float",
                value=critical_machine_config.calibration_offset_y_mm,
            ),
            ConfigItem(
                name="calibration offset z [mm]",
                handle=CalibrationConfig.OFFSET_Z_MM.value,
                value_kind="float",
                value=critical_machine_config.calibration_offset_z_mm,
            ),
            ConfigItem(
                name="turn off all illumination",
                handle=IlluminationConfig.TURN_OFF_ALL.value,
                value_kind="action",
                value="/api/action/turn_off_all_illumination",
            ),
            ConfigItem(
                name="base output storage directory",
                handle=StorageConfig.BASE_IMAGE_OUTPUT_DIR.value,
                value_kind="text",
                value=critical_machine_config.base_image_output_dir,
            ),
            # images with bit depth not a multiple of 8 (e.g. 12) use the lowest n bits of the bytes used to store them, which is an issue, because
            # most image file formats cannot handle bit depth that is not a multiple of 8. padding that data to preserve correct interpretation requires
            # padding the lowest bits of the next largest multiple of 8 with zeros, e.g. 12 bits -> shift 12 actual bits left to occupy top 12 bits,
            # then set lowest (2*8)-12=4 bits to zero.
            # this flag indicates if the lower bits should be padded (and value bits shifted into upper bits)
            ConfigItem(
                name="image file pad low",
                handle=ImageConfig.FILE_PAD_LOW.value,
                value_kind="option",
                value="yes",
                options=ConfigItemOption.get_bool_options(),
            ),
            ConfigItem(
                name="image filename use channel name",
                handle=ImageConfig.FILENAME_USE_CHANNEL_NAME.value,
                value_kind="option",
                value="yes",
                options=ConfigItemOption.get_bool_options(),
            ),
            ConfigItem(
                name="image filename xy index start",
                handle=ImageConfig.FILENAME_XY_INDEX_START.value,
                value_kind="int",
                value=0,
            ),
            ConfigItem(
                name="image filename z index start",
                handle=ImageConfig.FILENAME_Z_INDEX_START.value,
                value_kind="int",
                value=0,
            ),
            ConfigItem(
                name="image filename site index start",
                handle=ImageConfig.FILENAME_SITE_INDEX_START.value,
                value_kind="int",
                value=1,
            ),
            ConfigItem(
                name="image filename zero pad column index",
                handle=ImageConfig.FILENAME_ZERO_PAD_COLUMN.value,
                value_kind="option",
                value="yes",
                options=ConfigItemOption.get_bool_options(),
            ),
            ConfigItem(
                name="forbidden areas",
                handle=ProtocolConfig.FORBIDDEN_AREAS.value,
                value_kind="text",
                value=critical_machine_config.forbidden_areas or '{"forbidden_areas":[]}',
            ),
            ConfigItem(
                name="imaging channels configuration",
                handle=ImagingConfig.CHANNELS.value,
                value_kind="text",
                value=critical_machine_config.channels,
                frozen=True,
            ),
            laser_autofocus_system_available_attribute,
            filter_wheel_system_available_attribute,
            ConfigItem(
                name="imaging order",
                handle=ImagingConfig.ORDER.value,
                value_kind="option",
                value="protocol_order",
                options=[
                    ConfigItemOption(
                        name="Protocol Order (config file order)",
                        handle="protocol_order",
                    ),
                    ConfigItemOption(
                        name="Z-Order (bottom to top)",
                        handle="z_order",
                    ),
                    ConfigItemOption(
                        name="Wavelength Order (high to low, then brightfield)",
                        handle="wavelength_order",
                    ),
                ],
            ),
            ConfigItem(
                name="microcontroller USB ID",
                handle=MicrocontrollerConfig.ID.value,
                value_kind="text",
                value=critical_machine_config.microcontroller_id or "",
                frozen=True,
            ),
            ConfigItem(
                name="microcontroller reconnection grace period [ms]",
                handle=MicrocontrollerConfig.RECONNECTION_GRACE_PERIOD_MS.value,
                value_kind="float",
                value=200.0,
            ),
            ConfigItem(
                name="microcontroller reconnection attempts",
                handle=MicrocontrollerConfig.RECONNECTION_ATTEMPTS.value,
                value_kind="int",
                value=5,
            ),
            ConfigItem(
                name="microcontroller reconnection delay [ms]",
                handle=MicrocontrollerConfig.RECONNECTION_DELAY_MS.value,
                value_kind="float",
                value=1000.0,
            ),
            ConfigItem(
                name="microcontroller operation retry attempts",
                handle=MicrocontrollerConfig.OPERATION_RETRY_ATTEMPTS.value,
                value_kind="int",
                value=5,
            ),
            ConfigItem(
                name="microcontroller operation retry delay [ms]",
                handle=MicrocontrollerConfig.OPERATION_RETRY_DELAY_MS.value,
                value_kind="float",
                value=5.0,
            ),
            ConfigItem(
                name="camera reconnection attempts",
                handle=CameraConfig.RECONNECTION_ATTEMPTS.value,
                value_kind="int",
                value=5,
            ),
            ConfigItem(
                name="camera reconnection delay [ms]",
                handle=CameraConfig.RECONNECTION_DELAY_MS.value,
                value_kind="float",
                value=1000.0,
            ),
            ConfigItem(
                name="camera operation retry attempts",
                handle=CameraConfig.OPERATION_RETRY_ATTEMPTS.value,
                value_kind="int",
                value=5,
            ),
            *laser_autofocus_system_attributes,
            *filter_wheel_system_attributes,
            *main_camera_attributes,
        ]

        return ret

    @staticmethod
    def get(microscope_name: str | None = None) -> list[ConfigItem]:
        """
        get list of all global config items
        """

        if GlobalConfigHandler._config_list is None:
            GlobalConfigHandler.reset(microscope_name)
        ret = GlobalConfigHandler._config_list
        assert ret is not None

        return ret


    @staticmethod
    def override(new_config_items: dict[str, ConfigItem] | list[ConfigItem]):
        g_list = GlobalConfigHandler.get()

        def get_index(handle: str) -> int | None:
            """
            get index of the item in the existing configs with input handle as handle

            returns None of no such item exists
            """

            for i, item in enumerate(g_list):
                assert isinstance(item, ConfigItem)

                if item.handle == handle:
                    return i

            return None

        assert GlobalConfigHandler._config_list is not None

        if isinstance(new_config_items, dict):
            for handle, item in new_config_items.items():
                index = get_index(handle)

                if index is None:
                    GlobalConfigHandler._config_list.append(item)
                else:
                    GlobalConfigHandler._config_list[index].override(item)

        else:
            for item in new_config_items:
                handle = item.handle

                index = get_index(handle)

                if index is None:
                    GlobalConfigHandler._config_list.append(item)
                else:
                    GlobalConfigHandler._config_list[index].override(item)

    @staticmethod
    def get_dict() -> dict[str, ConfigItem]:
        ret = {}
        for c in GlobalConfigHandler.get():
            assert isinstance(c, ConfigItem)

            # make sure it does not exist already
            if c.handle in ret:
                raise ValueError("duplicate handle found in config list")

            ret[c.handle] = c

        return ret

    @staticmethod
    def update_pixel_format_options(
        main_camera_formats: list[str],
        autofocus_camera_formats: list[str] | None = None,
    ) -> None:
        """
        Update pixel format option lists with actual camera capabilities.

        Replaces hardcoded pixel format options with runtime-fetched camera capabilities.
        This ensures the config always reflects what the actual hardware supports.

        Args:
            main_camera_formats: List of supported formats from main camera (e.g., ["mono8", "mono10", "mono12"])
            autofocus_camera_formats: List of supported formats from autofocus camera, or None if autofocus unavailable
        """
        config_dict = GlobalConfigHandler.get_dict()

        # Update main camera pixel format options
        main_format_item = config_dict.get(CameraConfig.MAIN_PIXEL_FORMAT.value)
        if main_format_item is not None:
            # Create new options list from actual camera capabilities
            new_options = [
                ConfigItemOption(name=fmt.capitalize(), handle=fmt)
                for fmt in sorted(main_camera_formats)
            ]
            main_format_item.options = new_options

            # Ensure current value is still valid, reset to first if not
            if main_format_item.value not in main_camera_formats:
                main_format_item.value = main_camera_formats[0]

        # Update autofocus camera pixel format options if available
        if autofocus_camera_formats is not None:
            autofocus_format_item = config_dict.get(LaserAutofocusConfig.CAMERA_PIXEL_FORMAT.value)
            if autofocus_format_item is not None:
                # Create new options list from actual camera capabilities
                new_options = [
                    ConfigItemOption(name=fmt.capitalize(), handle=fmt)
                    for fmt in sorted(autofocus_camera_formats)
                ]
                autofocus_format_item.options = new_options

                # Ensure current value is still valid, reset to first if not
                if autofocus_format_item.value not in autofocus_camera_formats:
                    autofocus_format_item.value = autofocus_camera_formats[0]

    @staticmethod
    def reset(microscope_name: str | None = None):
        GlobalConfigHandler._current_microscope_name = microscope_name
        GlobalConfigHandler._config_list = GlobalConfigHandler._defaults(microscope_name)
