import typing as tp
from dataclasses import dataclass
import dataclasses as dc

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
        print(f"{self.name=} {self.value = }")
        return self.value=="yes"

    def override(self,other:"ConfigItem"):
        """
            update value from other item
        """
        assert self.handle==other.handle
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
    @staticmethod
    def _defaults()->tp.List[ConfigItem]:
        """
        get a list of all the low level machine settings

        these settings may be changed on the client side, for individual acquisitions
        (though clearly, this is highly advanced stuff, and may cause irreperable hardware damage!)
        """

        main_camera_attributes=[
            ConfigItem(
                name="main camera model",
                handle="main_camera_model",
                value_kind="text",
                value="MER2-1220-32U3M",
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

        laser_autofocus_system_attributes=[
            ConfigItem(
                name="laser autofocus system available",
                handle="laser_autofocus_available",
                value_kind="option",
                value="yes",
                options=ConfigItemOption.get_bool_options(),
                frozen=True,
            ),
            ConfigItem(
                name="laser autofocus camera model",
                handle="laser_autofocus_camera_model",
                value_kind="text",
                value="MER2-630-60U3M",
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
                name="laser autofocus image width [px]",
                handle="laser_autofocus_image_width_px",
                value_kind="number",
                value=3000,
            ),
            ConfigItem(
                name="laser autofocus image height [px]",
                handle="laser_autofocus_image_height_px",
                value_kind="number",
                value=400,
            ),
        ]

        ret=[
            ConfigItem(
                name="microscope name",
                handle="microscope_name",
                value_kind="text",
                value="HCS SQUID main #3",
                frozen=True,
            ),

            ConfigItem(
                name="calibrate top left of B2 here",
                handle="calibrate_B2_here",
                value_kind="action",
                value="/api/action/calibrate_stage_xy_here",
            ),

            ConfigItem(
                name="base output storage directory",
                handle="base_image_output_dir",
                value_kind="text",
                value="/mnt/squid/",
            ),

            ConfigItem(
                name="preview resolution scaling",
                handle="preview_resolution_scaling",
                value_kind="number",
                value=3, # performance wise, on an rpi5 value=3 drops about every 15th frame -> good compromise
            ),

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
    def reset():
        GlobalConfigHandler._config_list=GlobalConfigHandler._defaults()
