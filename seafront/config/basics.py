import json
import os
import typing as tp
from pathlib import Path

from pydantic import BaseModel, Field
from seaconfig import AcquisitionConfig, ConfigItem, ConfigItemOption


class CriticalMachineConfig(BaseModel):
    microscope_name: str

    main_camera_model: str

    base_image_output_dir: str
    calibration_offset_x_mm: float
    calibration_offset_y_mm: float
    calibration_offset_z_mm: float

    forbidden_wells: str | None = None
    "must be json-like string"

    laser_autofocus_camera_model: str | None = None
    "if laser_autofocus_available is yes, then this must be present"
    laser_autofocus_available: tp.Literal["yes", "no"] | None = None

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
            current_file_contents = json.load(config_file)
            server_config = ServerConfig(**current_file_contents)

        # store critical config items from current config
        store_dict = {}
        current_config = GlobalConfigHandler.get_dict()
        for key in critical_machine_config.keys():
            if key in current_config:
                store_dict[key] = current_config[key].value

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
        return CriticalMachineConfig(
            main_camera_model="MER2-1220-32U3M",
            laser_autofocus_camera_model="MER2-630-60U3M",
            microscope_name="unnamed HCS SQUID",
            base_image_output_dir=str(GlobalConfigHandler.home() / "images"),
            laser_autofocus_available="yes",
            calibration_offset_x_mm=0.0,
            calibration_offset_y_mm=0.0,
            calibration_offset_z_mm=0.0,
            forbidden_wells="""{"1":[],"4":[],"96":[],"384":["A01","A24","P01","P24"],"1536":[]}""",
        )

    @staticmethod
    def _defaults() -> list[ConfigItem]:
        """
        get a list of all the low level machine settings

        these settings may be changed on the client side, for individual acquisitions
        (though clearly, this is highly advanced stuff, and may cause irreperable hardware damage!)
        """

        # load config file
        with GlobalConfigHandler.home_config().open("r") as f:
            server_config_json=json.load(f)
            server_config = ServerConfig(**server_config_json)
            if len(server_config.microscopes)==0:
                raise ValueError("no microscope found in server config")
            critical_machine_config: CriticalMachineConfig = server_config.microscopes[0]

        main_camera_attributes = [
            ConfigItem(
                name="main camera model",
                handle="main_camera_model",
                value_kind="text",
                value=critical_machine_config.main_camera_model,
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
                value="mono12",
                options=[
                    ConfigItemOption(
                        name="8 Bit",
                        handle="mono8",
                    ),
                    ConfigItemOption(
                        name="12 Bit",
                        handle="mono12",
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
                name="forbidden wells",
                handle="forbidden_wells",
                value_kind="text",
                value=critical_machine_config.forbidden_wells or "{}",
            ),
            laser_autofocus_system_available_attribute,
            *laser_autofocus_system_attributes,
            *main_camera_attributes,
        ]

        return ret

    @staticmethod
    def get() -> list[ConfigItem]:
        """
        get list of all global config items
        """

        if GlobalConfigHandler._config_list is None:
            GlobalConfigHandler.reset()
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
    def reset():
        GlobalConfigHandler._config_list = GlobalConfigHandler._defaults()
