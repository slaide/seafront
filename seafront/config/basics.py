import typing as tp
from dataclasses import dataclass
import dataclasses as dc
import os
from pathlib import Path
import json

@dataclass
class ConfigItemOption:
    name:str
    handle:str
    info:tp.Optional[tp.Any]=None

    def to_dict(self)->dict:
        return dc.asdict(self)
    
    @staticmethod
    def get_bool_options()->tp.List["ConfigItemOption"]:
        return [
            ConfigItemOption(
                name="Yes",
                handle="yes",
            ),
            ConfigItemOption(
                name="No",
                handle="no",
            ),
        ]

@dataclass
class ConfigItem:
    name:str
    handle:str
    value_kind:tp.Literal["number","text","option","action"]
    value:tp.Union[int,float,str]
    frozen:bool=False
    options:tp.Optional[tp.List[ConfigItemOption]]=None

    @property
    def intvalue(self)->int:
        assert isinstance(self.value,int), f"{self.value = } ; {type(self.value) = }"
        return self.value
    
    @property
    def boolvalue(self)->bool:
        # from ConfigItemOption.get_bool_options()
        assert isinstance(self.value,str), f"{self.value = } ; {type(self.value) = }"
        return self.value=="yes"

    def override(self,other:"ConfigItem"):
        """
            update value from other item
        """
        assert self.handle==other.handle, f"{self.handle = } ; {other.handle = }"
        match self.value_kind:
            case "number":
                if isinstance(self.value,int):
                    self.value=int(other.value)
                else:
                    self.value=float(other.value)
            case _:
                self.value=other.value

    def to_dict(self)->dict:
        ret=dc.asdict(self)
        if self.options is not None:
            ret['options']=[o.to_dict() for o in self.options]
        return ret

class GlobalConfigHandler:
    seafront_home:tp.Optional[Path]=None

    @staticmethod
    def CRITICAL_MACHINE_DEFAULTS():
        assert GlobalConfigHandler.seafront_home is not None
        return {
            "main_camera_model":"MER2-1220-32U3M",
            "laser_autofocus_camera_model":"MER2-630-60U3M",
            "microscope_name":"unnamed HCS SQUID",
            "base_image_output_dir":str(GlobalConfigHandler.seafront_home/"images"),
            "laser_autofocus_available":"yes",
            "calibration_offset_x_mm":0.0,
            "calibration_offset_y_mm":0.0,
            "calibration_offset_z_mm":0.0,
            "forbidden_wells":"96:;384:A01,A24,P01,P24;1536:",
        }

    @staticmethod
    def _defaults()->tp.List[ConfigItem]:
        """
        get a list of all the low level machine settings

        these settings may be changed on the client side, for individual acquisitions
        (though clearly, this is highly advanced stuff, and may cause irreperable hardware damage!)
        """

        if GlobalConfigHandler.seafront_home is not None:
            SEAFRONT_HOME=GlobalConfigHandler.seafront_home
        else:
            # construct SEAFRONT_HOME, from $SEAFRONT_HOME or $HOME/seafront
            env_seafront_home=os.getenv("SEAFRONT_HOME")
            if env_seafront_home is not None:
                SEAFRONT_HOME=Path(env_seafront_home)

            else:
                # get home dir of user
                home_dir=os.environ.get("HOME")
                if home_dir is None:
                    raise ValueError("could not find home directory")
                
                SEAFRONT_HOME=Path(home_dir)/"seafront"

            GlobalConfigHandler.seafront_home=SEAFRONT_HOME

        if not SEAFRONT_HOME.exists():
            SEAFRONT_HOME.mkdir(parents=True)
        DEFAULT_IMAGE_STORAGE_DIR=SEAFRONT_HOME/"images"
        if not DEFAULT_IMAGE_STORAGE_DIR.exists():
            DEFAULT_IMAGE_STORAGE_DIR.mkdir(parents=True)
        
        CONFIG_FILE_PATH=SEAFRONT_HOME/"config.json"
        if not CONFIG_FILE_PATH.exists():
            # create config file
            with open(CONFIG_FILE_PATH,"w") as f:
                json.dump(GlobalConfigHandler.CRITICAL_MACHINE_DEFAULTS(),f,indent=4)
    
        # load config file
        with open(CONFIG_FILE_PATH,"r") as f:
            critical_machine_config=json.load(f)

        main_camera_attributes=[
            ConfigItem(
                name="main camera model",
                handle="main_camera_model",
                value_kind="text",
                value=critical_machine_config["main_camera_model"],
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
                            "magnification":4,
                        }
                    ),
                    ConfigItemOption(
                        name="10x Olympus",
                        handle="10xolympus",
                        info={
                            "magnification":10,
                        }
                    ),
                    ConfigItemOption(
                        name="20x Olympus",
                        handle="20xolympus",
                        info={
                            "magnification":20,
                        }
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
                value_kind="number",
                value=2500,
            ),
            ConfigItem(
                name="main camera image height [px]",
                handle="main_camera_image_height_px",
                value_kind="number",
                value=2500,
            ),
            ConfigItem(
                name="main camera flip image horizontally",
                handle="main_camera_image_flip_horizontal",
                value_kind="option",
                value="no",
                options=ConfigItemOption.get_bool_options(),
                frozen=True, # unfrozen for debugging!
            ),
            ConfigItem(
                name="main camera flip image vertically",
                handle="main_camera_image_flip_vertical",
                value_kind="option",
                value="yes",
                options=ConfigItemOption.get_bool_options(),
                frozen=True, # unfrozen for debugging!
            ),
        ]

        laser_autofocus_system_available_attribute=ConfigItem(
            name="laser autofocus system available",
            handle="laser_autofocus_available",
            value_kind="option",
            value=critical_machine_config["laser_autofocus_available"],
            options=ConfigItemOption.get_bool_options(),
            frozen=True,
        )

        if laser_autofocus_system_available_attribute.boolvalue:
            laser_autofocus_system_attributes=[
                ConfigItem(
                    name="laser autofocus camera model",
                    handle="laser_autofocus_camera_model",
                    value_kind="text",
                    value=critical_machine_config["laser_autofocus_camera_model"],
                    frozen=True,
                ),
                ConfigItem(
                    name="laser autofocus exposure time [ms]",
                    handle="laser_autofocus_exposure_time_ms",
                    value_kind="number",
                    value=5.0,
                ),
                ConfigItem(
                    name="laser autofocus camera analog gain",
                    handle="laser_autofocus_analog_gain",
                    value_kind="number",
                    value=0,
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
                    value="/api/action/laser_af_warm_up_laser",
                ),
            ]
        else:
            laser_autofocus_system_attributes=[]

        ret=[
            ConfigItem(
                name="microscope name",
                handle="microscope_name",
                value_kind="text",
                value=critical_machine_config["microscope_name"],
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
                value_kind="number",
                value=critical_machine_config["calibration_offset_x_mm"],
            ),
            ConfigItem(
                name="calibration offset y [mm]",
                handle="calibration_offset_y_mm",
                value_kind="number",
                value=critical_machine_config["calibration_offset_y_mm"],
            ),
            ConfigItem(
                name="calibration offset z [mm]",
                handle="calibration_offset_z_mm",
                value_kind="number",
                value=critical_machine_config["calibration_offset_z_mm"],
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
                value=critical_machine_config["base_image_output_dir"],
            ),

            # preview settings are performance sensitive
            # e.g. on an rpi5, the max streaming framerate for jpeg at scale=2 resolution is 5fps
            #      but for png, to reach 5fps the resolution must be at least scale=5
            ConfigItem(
                name="streaming preview resolution scaling",
                handle="streaming_preview_resolution_scaling",
                value_kind="number",
                value=2,
            ),
            ConfigItem(
                name="streaming preview image format",
                handle="streaming_preview_format",
                value_kind="option",
                value="jpeg",
                options=[
                    ConfigItemOption(
                        name="JPEG",
                        handle="jpeg",
                    ),
                    ConfigItemOption(
                        name="PNG",
                        handle="png",
                    ),
                ]
            ),

            ConfigItem(
                name="full image display format",
                handle="image_display_format",
                value_kind="option",
                value="jpeg",
                options=[
                    ConfigItemOption(
                        name="JPEG",
                        handle="jpeg",
                    ),
                    ConfigItemOption(
                        name="PNG",
                        handle="png",
                    ),
                ]
            ),

            ConfigItem(
                name="forbidden wells",
                handle="forbidden_wells",
                value_kind="text",
                value=critical_machine_config["forbidden_wells"]
            ),

            laser_autofocus_system_available_attribute,
            *laser_autofocus_system_attributes,

            *main_camera_attributes,
        ]

        return ret

    _config_list:tp.Optional[tp.List[ConfigItem]]=None

    @staticmethod
    def get(as_dict:bool=False)->tp.Union[tp.List[ConfigItem],tp.List[dict]]:
        """
            get list of all global config items
        """

        if GlobalConfigHandler._config_list is None:
            GlobalConfigHandler.reset()
        ret=GlobalConfigHandler._config_list
        assert ret is not None

        if as_dict:
            items_json=[i.to_dict() for i in ret]
            return items_json

        return ret
    
    @staticmethod
    def get_dict()->tp.Dict[str,ConfigItem]:
        ret={}
        for c in GlobalConfigHandler.get():
            assert isinstance(c,ConfigItem)

            # make sure it does not exist already
            if c.handle in ret:
                raise ValueError("duplicate handle found in config list")
            
            ret[c.handle]=c
        return ret
    
    @staticmethod
    def override(new_config_items:tp.Union[tp.Dict[str,ConfigItem],tp.List[ConfigItem]]):
        g_list=GlobalConfigHandler.get()

        def get_index(handle:str)->tp.Optional[int]:
            for i,item in enumerate(g_list):
                assert isinstance(item,ConfigItem)

                if item.handle==handle:
                    return i
                
            return None

        assert GlobalConfigHandler._config_list is not None
        if isinstance(new_config_items,dict):
            for handle,item in new_config_items.items():
                if not isinstance(item,ConfigItem):
                    assert isinstance(item,dict), f"{type(item) = } ; {item = }"
                    item=ConfigItem(**item)
                index=get_index(handle)
                assert index is not None
                GlobalConfigHandler._config_list[index].override(item)
        elif isinstance(new_config_items,list):
            for item in new_config_items:
                if not isinstance(item,ConfigItem):
                    assert isinstance(item,dict), f"{type(item) = } ; {item = }"
                    item=ConfigItem(**item)
                handle=item.handle
                index=get_index(handle)
                assert index is not None
                GlobalConfigHandler._config_list[index].override(item)

    @staticmethod
    def store():
        """
        write current config to disk
        """

        assert GlobalConfigHandler.seafront_home is not None
        SEAFRONT_HOME=GlobalConfigHandler.seafront_home
        CONFIG_FILE_PATH=SEAFRONT_HOME/"config.json"
        assert GlobalConfigHandler._config_list is not None
        critical_machine_config=GlobalConfigHandler.CRITICAL_MACHINE_DEFAULTS()

        # store items from current config if their key is present in critical defaults
        store_dict={}
        current_config=GlobalConfigHandler.get_dict()
        for key,value in critical_machine_config.items():
            if key in current_config:
                store_dict[key]=current_config[key].value

        with open(CONFIG_FILE_PATH,"w") as f:
            json.dump(store_dict,f,indent=4)

    @staticmethod
    def reset():
        GlobalConfigHandler._config_list=GlobalConfigHandler._defaults()
