
def config_list_as_dict(config_list):
    ret={}
    for c in config_list:
        # make sure it does not exist already
        if c["handle"] in ret:
            raise ValueError("duplicate handle found in config list")
        
        ret[c["handle"]]=c
    return ret

def _get_machine_defaults():
    """
    get a list of all the low level machine settings

    these settings may be changed on the client side, for individual acquisitions
    (though clearly, this is highly advanced stuff, and may cause irreperable hardware damage!)
    """

    bool_options=[
        {
            "name":"Yes",
            "handle":"yes",
        },
        {
            "name":"No",
            "handle":"no",
        },
    ]

    main_camera_attributes=[
        {
            "name":"main camera model",
            "handle":"main_camera_model",
            "value_kind":"text",
            "value":"MER2-1220-32U3M",
            "frozen":True,
        },
        {
            "name":"main camera objective",
            "handle":"main_camera_objective",
            "value_kind":"option",
            "value":"20xolympus",
            "options":[
                {
                    "name":"4x Olympus",
                    "handle":"4xolympus",
                    "magnification":4,
                },
                {
                    "name":"10x Olympus",
                    "handle":"10xolympus",
                    "magnification":10,
                },
                {
                    "name":"20x Olympus",
                    "handle":"20xolympus",
                    "magnification":20,
                },
            ],
        },
        {
            "name":"main camera trigger",
            "handle":"main_camera_trigger",
            "value_kind":"option",
            "value":"software",
            "options":[
                {
                    "name":"Software",
                    "handle":"software",
                },
                {
                    "name":"Hardware",
                    "handle":"hardware",
                },
            ],
        },
        {
            "name":"main camera pixel format",
            "handle":"main_camera_pixel_format",
            "value_kind":"option",
            "value":"mono12",
            "options":[
                {
                    "name":"8 Bit",
                    "handle":"mono8",
                },
                {
                    "name":"12 Bit",
                    "handle":"mono12",
                },
            ],
        },
        {
            "name":"main camera image width [px]",
            "handle":"main_camera_image_width_px",
            "value_kind":"number",
            "value":2500,
        },
        {
            "name":"main camera image height [px]",
            "handle":"main_camera_image_height_px",
            "value_kind":"number",
            "value":2500,
        },
    ]

    laser_autofocus_system_attributes=[
        {
            "name":"laser autofocus system available",
            "handle":"laser_autofocus_available",
            "value_kind":"option",
            "value":"yes",
            "options":bool_options,
            "frozen":True,
        },
        {
            "name":"laser autofocus camera model",
            "handle":"laser_autofocus_camera_model",
            "value_kind":"text",
            "value":"MER2-630-60U3M",
            "frozen":True,
        },
        {
            "name":"laser autofocus exposure time [ms]",
            "handle":"laser_autofocus_exposure_time_ms",
            "value_kind":"number",
            "value":5.0,
        },
        {
            "name":"laser autofocus camera analog gain",
            "handle":"laser_autofocus_analog_gain",
            "value_kind":"number",
            "value":0,
        },
        {
            "name":"laser autofocus camera pixel format",
            "handle":"laser_autofocus_pixel_format",
            "value_kind":"option",
            "value":"mono8",
            "options":[
                {
                    "name":"8 Bit",
                    "handle":"mono8",
                },
                {
                    "name":"10 Bit",
                    "handle":"mono10",
                },
            ],
        },
        {
            "name":"laser autofocus image width [px]",
            "handle":"laser_autofocus_image_width_px",
            "value_kind":"number",
            "value":3000,
        },
        {
            "name":"laser autofocus image height [px]",
            "handle":"laser_autofocus_image_height_px",
            "value_kind":"number",
            "value":400,
        },
    ]

    ret=[
        {
            "name":"microscope name",
            "handle":"microscope_name",
            "value_kind":"text",
            "value":"HCS SQUID main #3",
            "frozen":True,
        },

        {
            "name":"base output storage directory",
            "handle":"base_image_output_dir",
            "value_kind":"text",
            "value":"/mnt/squid/",
        },

        {
            "name":"move in x by",
            "handle":"action_move_x_by",
            "value_kind":"action",
            "value":"/api/action/move_x_by",
        },

        *laser_autofocus_system_attributes,
        *main_camera_attributes,
    ]

    return ret


class GlobalConfigHandler:
    config_list=_get_machine_defaults()

    @staticmethod
    def get():
        return GlobalConfigHandler.config_list
    
    @staticmethod
    def override(new_config_list:list):
        GlobalConfigHandler.config_list=new_config_list

    @staticmethod
    def reset():
        GlobalConfigHandler.config_list=_get_machine_defaults()
