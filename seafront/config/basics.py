import json
import json5
import os
import typing as tp
from pathlib import Path

from pydantic import BaseModel, Field
from seaconfig import AcquisitionConfig, ConfigItem, ConfigItemOption

CameraDriver = tp.Literal["galaxy", "toupcam"]
MicroscopeType = tp.Literal["squid", "mock"]


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

    main_camera_model: str
    main_camera_driver: CameraDriver = "galaxy"

    base_image_output_dir: str
    calibration_offset_x_mm: float
    calibration_offset_y_mm: float
    calibration_offset_z_mm: float

    forbidden_wells: str | None = None
    "must be json-like string"

    laser_autofocus_available: tp.Literal["yes", "no"] | None = None
    laser_autofocus_camera_model: str | None = None
    "if laser_autofocus_available is yes, then this must be present"
    laser_autofocus_camera_driver: CameraDriver = "galaxy"
    
    filter_wheel_available: tp.Literal["yes", "no"] | None = None

    channels: str = Field(default="[]")
    "Available imaging channels with their illumination source slots (JSON-encoded string)"
    
    filters: str = Field(default="[]")
    "Available filters with their wheel positions (JSON-encoded string)"

    # needs some post-init hook to check forbidden_wells for json-like-ness

class ServerConfig(BaseModel):
    port: int = 5000
    microscopes: list[CriticalMachineConfig] = Field(
        default_factory=lambda: [GlobalConfigHandler.CRITICAL_MACHINE_DEFAULTS()]
    )


class GlobalConfigHandler:
    _seafront_home: Path | None = None

    _config_list: list[ConfigItem] | None = None

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
            # ensure default acquisition dir is present
            _ = GlobalConfigHandler.home_acquisition_config_dir()

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
        
        # Find the existing microscope config to use as fallback
        current_microscope_name = current_config.get("microscope_name")
        existing_microscope_config = None
        if current_microscope_name:
            for microscope_config in server_config.microscopes:
                if microscope_config.microscope_name == current_microscope_name.value:
                    existing_microscope_config = microscope_config
                    break
        
        for key in critical_machine_config.keys():
            if key in current_config:
                store_dict[key] = current_config[key].value
            elif existing_microscope_config and hasattr(existing_microscope_config, key):
                # Fallback to existing config value for missing keys
                store_dict[key] = getattr(existing_microscope_config, key)

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
    def home_acquisition_config_dir() -> Path:
        """
        get path to directory containing [user-defined] acquisition configurations

        will create the directory it not already present.
        """

        DEFAULT_CONFIG_STORAGE_DIR = GlobalConfigHandler.home() / "acquisition_configs"  # type: ignore
        if not DEFAULT_CONFIG_STORAGE_DIR.exists():
            DEFAULT_CONFIG_STORAGE_DIR.mkdir(parents=True)

        return DEFAULT_CONFIG_STORAGE_DIR

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
        
        # Convert to JSON strings for storage
        channels_json = json.dumps([ch.model_dump() for ch in default_channels])
        filters_json = json.dumps([f.model_dump() for f in default_filters])
        
        return CriticalMachineConfig(
            main_camera_model="MER2-1220-32U3M",
            laser_autofocus_camera_model="MER2-630-60U3M",
            microscope_name="unnamed HCS SQUID",
            microscope_type="squid",
            base_image_output_dir=str(GlobalConfigHandler.home() / "images"),
            laser_autofocus_available="yes",
            filter_wheel_available="no",
            calibration_offset_x_mm=0.0,
            calibration_offset_y_mm=0.0,
            calibration_offset_z_mm=0.0,
            forbidden_wells="""{"1":[],"4":[],"96":[],"384":["A01","A24","P01","P24"],"1536":[]}""",
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
                name="main camera model",
                handle="main_camera_model",
                value_kind="text",
                value=critical_machine_config.main_camera_model,
                frozen=True,
            ),
            ConfigItem(
                name="main camera driver",
                handle="main_camera_driver",
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
                handle="main_camera_objective",
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
                handle="main_camera_trigger",
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
                handle="main_camera_pixel_format",
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
                handle="main_camera_image_width_px",
                value_kind="int",
                value=2500,
            ),
            ConfigItem(
                name="main camera image height [px]",
                handle="main_camera_image_height_px",
                value_kind="int",
                value=2500,
            ),
            ConfigItem(
                name="main camera flip image horizontally",
                handle="main_camera_image_flip_horizontal",
                value_kind="option",
                value="no",
                options=ConfigItemOption.get_bool_options(),
                frozen=True,
            ),
            ConfigItem(
                name="main camera flip image vertically",
                handle="main_camera_image_flip_vertical",
                value_kind="option",
                value="yes",
                options=ConfigItemOption.get_bool_options(),
                frozen=True,
            ),
        ]

        laser_autofocus_system_available_attribute = ConfigItem(
            name="laser autofocus system available",
            handle="laser_autofocus_available",
            value_kind="option",
            value=critical_machine_config.laser_autofocus_available or "no",
            options=ConfigItemOption.get_bool_options(),
            frozen=True,
        )

        filter_wheel_system_available_attribute = ConfigItem(
            name="filter wheel system available",
            handle="filter_wheel_available",
            value_kind="option",
            value=critical_machine_config.filter_wheel_available or "no",
            options=ConfigItemOption.get_bool_options(),
            frozen=True,
        )

        if laser_autofocus_system_available_attribute.boolvalue:
            if critical_machine_config.laser_autofocus_camera_model is None:
                raise ValueError("laser autofocus available but no autofocus camera model provided")

            laser_autofocus_system_attributes = [
                ConfigItem(
                    name="laser autofocus camera model",
                    handle="laser_autofocus_camera_model",
                    value_kind="text",
                    value=critical_machine_config.laser_autofocus_camera_model,
                    frozen=True,
                ),
                ConfigItem(
                    name="laser autofocus camera driver",
                    handle="laser_autofocus_camera_driver",
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
                    handle="laser_autofocus_exposure_time_ms",
                    value_kind="float",
                    value=5.0,
                ),
                ConfigItem(
                    name="laser autofocus camera analog gain",
                    handle="laser_autofocus_analog_gain",
                    value_kind="float",
                    value=0.0,
                ),
                ConfigItem(
                    name="laser autofocus use glass top",
                    handle="laser_autofocus_use_glass_top",
                    value_kind="option",
                    value="no",
                    options=ConfigItemOption.get_bool_options(),
                ),
                ConfigItem(
                    name="laser autofocus camera pixel format",
                    handle="laser_autofocus_pixel_format",
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
                    handle="laser_af_warm_up_laser",
                    value_kind="action",
                    value="/api/action/laser_autofocus_warm_up_laser",
                ),
                # is calibrated flag
                ConfigItem(
                    name="laser autofocus is calibrated",
                    handle="laser_autofocus_is_calibrated",
                    value_kind="option",
                    value="no",
                    options=ConfigItemOption.get_bool_options(),
                ),
                # calibrated x on sensor
                ConfigItem(
                    name="laser autofocus calibration: x peak pos",
                    handle="laser_autofocus_calibration_x",
                    value_kind="float",
                    value=0.0,
                ),
                # calibrated um/px on sensor
                ConfigItem(
                    name="laser autofocus calibration: um per px",
                    handle="laser_autofocus_calibration_umpx",
                    value_kind="float",
                    value=0.0,
                ),
                # z coordinate at time of calibration
                ConfigItem(
                    name="laser autofocus calibration: ref z in mm",
                    handle="laser_autofocus_calibration_refzmm",
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
                    handle="filters",
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
                handle="microscope_name",
                value_kind="text",
                value=critical_machine_config.microscope_name,
                frozen=True,
            ),
            ConfigItem(
                name="calibrate top left of B2 here",
                handle="calibrate_B2_here",
                value_kind="action",
                value="/api/action/calibrate_stage_xy_here",
            ),
            ConfigItem(
                name="calibration offset x [mm]",
                handle="calibration_offset_x_mm",
                value_kind="float",
                value=critical_machine_config.calibration_offset_x_mm,
            ),
            ConfigItem(
                name="calibration offset y [mm]",
                handle="calibration_offset_y_mm",
                value_kind="float",
                value=critical_machine_config.calibration_offset_y_mm,
            ),
            ConfigItem(
                name="calibration offset z [mm]",
                handle="calibration_offset_z_mm",
                value_kind="float",
                value=critical_machine_config.calibration_offset_z_mm,
            ),
            ConfigItem(
                name="turn off all illumination",
                handle="illumination_off",
                value_kind="action",
                value="/api/action/turn_off_all_illumination",
            ),
            ConfigItem(
                name="base output storage directory",
                handle="base_image_output_dir",
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
                handle="image_file_pad_low",
                value_kind="option",
                value="yes",
                options=ConfigItemOption.get_bool_options(),
            ),
            ConfigItem(
                name="image filename use channel name",
                handle="image_filename_use_channel_name",
                value_kind="option", 
                value="yes",
                options=ConfigItemOption.get_bool_options(),
            ),
            ConfigItem(
                name="forbidden wells",
                handle="forbidden_wells",
                value_kind="text",
                value=critical_machine_config.forbidden_wells or "{}",
            ),
            ConfigItem(
                name="imaging channels configuration",
                handle="channels",
                value_kind="text",
                value=critical_machine_config.channels,
                frozen=True,
            ),
            laser_autofocus_system_available_attribute,
            filter_wheel_system_available_attribute,
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
    def reset(microscope_name: str | None = None):
        GlobalConfigHandler._config_list = GlobalConfigHandler._defaults(microscope_name)
