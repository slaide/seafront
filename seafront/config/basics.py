import hashlib
import json
import os
import re
import typing as tp
from pathlib import Path

import json5
from pydantic import BaseModel, Field
from seaconfig import AcquisitionConfig, ConfigItem, ConfigItemOption

from seafront.config.registry import ConfigRegistry

CameraDriver = tp.Literal["galaxy", "toupcam"]
MicroscopeType = tp.Literal["squid", "mock"]
ImagingOrder = tp.Literal["z_order", "wavelength_order", "protocol_order"]

# Type alias for microscope config - a flat dict of handle -> value
MicroscopeConfig = dict[str, tp.Any]


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


class ServerConfig(BaseModel):
    port: int = 5000
    microscopes: list[MicroscopeConfig] = Field(
        default_factory=lambda: [GlobalConfigHandler.DEFAULT_MICROSCOPE_CONFIG()]
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
        Write current global config to disk.

        Uses handle keys directly - no mapping needed.
        """
        from seafront.config.handles import SystemConfig

        CONFIG_FILE_PATH = GlobalConfigHandler.home_config()

        with CONFIG_FILE_PATH.open("r") as config_file:
            current_file_contents = json5.load(config_file)
            server_config = ServerConfig(**current_file_contents)

        current_config = ConfigRegistry.get_dict()

        # Get current microscope name
        current_microscope_name = current_config[SystemConfig.MICROSCOPE_NAME.value].value

        # Find existing microscope config to preserve non-persistent values
        existing_microscope_config: MicroscopeConfig = {}
        existing_index: int | None = None
        for i, m in enumerate(server_config.microscopes):
            if m.get(SystemConfig.MICROSCOPE_NAME.value) == current_microscope_name:
                existing_microscope_config = m
                existing_index = i
                break

        # Build new microscope config from persistent handles
        new_microscope_config: MicroscopeConfig = dict(existing_microscope_config)
        for handle in ConfigRegistry.get_persistent_handles():
            # get_value returns the correct type (objects for object types, primitives otherwise)
            new_microscope_config[handle] = ConfigRegistry.get_value(handle)

        # Update or append microscope config
        if existing_index is not None:
            server_config.microscopes[existing_index] = new_microscope_config
        else:
            server_config.microscopes.append(new_microscope_config)

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
    def DEFAULT_MICROSCOPE_CONFIG() -> MicroscopeConfig:
        """Generate default microscope config using handle keys."""
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
        default_filters: list[FilterConfig] = []

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

        # Use handle enums from handles module
        from seafront.config.handles import (
            SystemConfig,
            CameraConfig,
            MicrocontrollerConfig,
            StorageConfig,
            CalibrationConfig,
            LaserAutofocusConfig,
            FilterWheelConfig,
            ImagingConfig,
            ProtocolConfig,
        )

        return {
            SystemConfig.MICROSCOPE_NAME.value: "squid",
            SystemConfig.MICROSCOPE_TYPE.value: "squid",
            CameraConfig.MAIN_ID.value: "CHANGE_ME",
            CameraConfig.MAIN_DRIVER.value: "galaxy",
            MicrocontrollerConfig.ID.value: "CHANGE_ME",
            MicrocontrollerConfig.DRIVER.value: "teensy",
            StorageConfig.BASE_IMAGE_OUTPUT_DIR.value: str(GlobalConfigHandler.home() / "images"),
            CalibrationConfig.OFFSET_X_MM.value: 0.0,
            CalibrationConfig.OFFSET_Y_MM.value: 0.0,
            CalibrationConfig.OFFSET_Z_MM.value: 0.0,
            LaserAutofocusConfig.AVAILABLE.value: "yes",
            LaserAutofocusConfig.CAMERA_ID.value: "CHANGE_ME",
            LaserAutofocusConfig.CAMERA_DRIVER.value: "galaxy",
            FilterWheelConfig.AVAILABLE.value: "no",
            # Native objects for "object" type configs
            FilterWheelConfig.CONFIGURATION.value: [f.model_dump() for f in default_filters],
            ImagingConfig.CHANNELS.value: [ch.model_dump() for ch in default_channels],
            ProtocolConfig.FORBIDDEN_AREAS.value: default_forbidden_areas,
        }

    @staticmethod
    def get(microscope_name: str | None = None) -> list[ConfigItem]:
        """Get list of all global config items."""
        # Delegate to ConfigRegistry
        return ConfigRegistry.get_all()

    @staticmethod
    def override(new_config_items: dict[str, ConfigItem] | list[ConfigItem]):
        """Override config items with new values."""
        if isinstance(new_config_items, dict):
            for handle, item in new_config_items.items():
                ConfigRegistry.set_value(handle, item.value)
        else:
            for item in new_config_items:
                ConfigRegistry.set_value(item.handle, item.value)

    @staticmethod
    def get_dict() -> dict[str, ConfigItem]:
        """Get all config items as a dict."""
        return ConfigRegistry.get_dict()

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
        from seafront.config.handles import CameraConfig, LaserAutofocusConfig

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
        """
        Initialize config with values from config file for the selected microscope.

        This loads the config file, finds the selected microscope's config,
        and registers all config items with values from the file overriding defaults.
        """
        from seafront.config.handles import SystemConfig, LaserAutofocusConfig, FilterWheelConfig
        from seafront.config.core_config import (
            register_core_config,
            register_laser_autofocus_config,
            register_filter_wheel_config,
        )

        GlobalConfigHandler._current_microscope_name = microscope_name

        # Reset the registry
        ConfigRegistry.reset()

        # Load config file and find target microscope
        with GlobalConfigHandler.home_config().open("r") as f:
            server_config_json = json5.load(f)
            server_config = ServerConfig(**server_config_json)
            if len(server_config.microscopes) == 0:
                raise ValueError("no microscope found in server config")

            # Select microscope by name or default to first
            microscope_config: MicroscopeConfig = {}
            if microscope_name is not None:
                for m in server_config.microscopes:
                    if m.get(SystemConfig.MICROSCOPE_NAME.value) == microscope_name:
                        microscope_config = m
                        break
                if not microscope_config:
                    available_names = [
                        m.get(SystemConfig.MICROSCOPE_NAME.value, "<unnamed>")
                        for m in server_config.microscopes
                    ]
                    raise ValueError(f"microscope '{microscope_name}' not found. Available: {available_names}")
            else:
                microscope_config = server_config.microscopes[0]

        # Initialize registry with config file values
        ConfigRegistry.init(microscope_config)

        # Get default values for complex configs
        default_config = GlobalConfigHandler.DEFAULT_MICROSCOPE_CONFIG()
        default_image_dir = str(GlobalConfigHandler.home() / "images")
        default_channels = default_config.get("imaging.channels", [])
        default_forbidden_areas = default_config.get("protocol.forbidden_areas", [])

        # Register core config items
        register_core_config(default_image_dir, default_channels, default_forbidden_areas)

        # Register optional subsystem configs based on availability
        if ConfigRegistry.get(LaserAutofocusConfig.AVAILABLE).boolvalue:
            register_laser_autofocus_config()

        if ConfigRegistry.get(FilterWheelConfig.AVAILABLE).boolvalue:
            register_filter_wheel_config()
