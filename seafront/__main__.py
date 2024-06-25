import json, time, os, io, sys
from flask import Flask, send_from_directory, request, send_file
import numpy as np
from PIL import Image
import typing as tp
from dataclasses import dataclass
from enum import Enum
import scipy
import glob
import gc
import threading as th
import queue as q
import datetime as dt
from pathlib import Path
import tifffile
import re

from seaconfig import *
from .config.basics import ConfigItem, GlobalConfigHandler
from .hardware.camera import Camera, gxiapi
from .hardware.microcontroller import Microcontroller, Command, ILLUMINATION_CODE

_DEBUG_P2JS=True

# precompile regex for performance
name_validity_regex=re.compile(r"^[a-zA-Z0-9_\-]+$")

app = Flask(__name__, static_folder='src')

@app.route('/')
def index():
    # send local file "index.html" as response
    return send_from_directory('.', 'index.html')

@app.route(rule='/img/<path:path>')
def send_img_from_local_path(path):
    return send_from_directory('img', path)

@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('css', path)

@app.route('/src/<path:path>')
def send_js(path):
    return send_from_directory('src', path)

# dummy data
remote_image_path_map:dict[str,dict]={
    "/server/data/acquisition/a65914/D05_2_3_1_1_fluo405.tiff":{
        "dir":"img",
        "filename":"cells.webp"
    },
    "/server/data/acquisition/a65914/D05_3_5_1_1_fluo405.tiff":{
        "dir":"img",
        "filename":"cells2.webp"
    },
}

@app.route("/api/img_remote")
def send_img_from_remote_storage():
    """
    send an image from remote storage
    """

    # get query parameters
    image_path=request.args.get("image_path",type=str)
    if image_path is None:
        return json.dumps({"status":"error","message":"no image_path provided"})

    if image_path not in remote_image_path_map:
        return json.dumps({"status":"error","message":"image_path not found"})

    return send_from_directory(remote_image_path_map[image_path]["dir"], remote_image_path_map[image_path]["filename"])

somemap:dict={}
"""
actual type is:
tp.Dict[str,{
    "acquisition_id":str,
    "queue_in":q.Queue,
        "queue to send messages to the thread"
    "queue_out":q.Queue,
        "queue to receive messages from the thread"
    "last_status":tp.Optional[dict],
    "thread_is_running":boolean,
}]
"""

# get objectives
@app.route("/api/get_features/hardware_capabilities", methods=["GET","POST"])
def get_hardware_capabilities():
    """
    get a list of all the hardware capabilities of the system

    these are the high-level configuration options that the user can select from
    """

    ret={
        # these are mocked values (should actually originate from seafront-config)
        "wellplate_types":[
            {
                "name":"Perkin Elmer 96",
                "handle":"pe96",
                "num_rows":8,
                "num_cols":12,
            },
            {
                "name":"Falcon 96",
                "handle":"fa96",
                "num_rows":8,
                "num_cols":12,
            },
            {
                "name":"Perkin Elmer 384",
                "handle":"pe384",
                "num_rows":16,
                "num_cols":24,
            },
            {
                "name":"Falcon 384",
                "handle":"fa384",
                "num_rows":16,
                "num_cols":24,
            },
            {
                "name":"Thermo Fischer 384",
                "handle":"tf384",
                "num_rows":16,
                "num_cols":24,
            },
            {
                "name":"Thermo Fischer 1536",
                "handle":"tf1536",
                "num_rows":32,
                "num_cols":48,
            }
        ],
        "main_camera_imaging_channels":[c.to_dict() for c in [
            AcquisitionChannelConfig(
                name="Fluo 405 nm Ex",
                handle="fluo405",
                illum_perc=100,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0
            ),
            AcquisitionChannelConfig(
                name="Fluo 488 nm Ex",
                handle="fluo488",
                illum_perc=100,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0
            ),
            AcquisitionChannelConfig(
                name="Fluo 561 nm Ex",
                handle="fluo561",
                illum_perc=100,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0
            ),
            AcquisitionChannelConfig(
                name="Fluo 638 nm Ex",
                handle="fluo638",
                illum_perc=100,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0
            ),
            AcquisitionChannelConfig(
                name="Fluo 730 nm Ex",
                handle="fluo730",
                illum_perc=100,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0
            ),
            AcquisitionChannelConfig(
                name="BF LED Full",
                handle="bfledfull",
                illum_perc=20,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0
            ),
            AcquisitionChannelConfig(
                name="BF LED Right Half",
                handle="bfledright",
                illum_perc=20,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0
            ),
            AcquisitionChannelConfig(
                name="BF LED Left Half",
                handle="bfledleft",
                illum_perc=20,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0
            ),
        ]]
    }
    return json.dumps(ret)

from .config.basics import GlobalConfigHandler
GlobalConfigHandler.reset()

@app.route("/api/get_features/machine_defaults", methods=["GET","POST"])
def get_machine_defaults():
    """
    get a list of all the low level machine settings (api wrapper to return json-as-string)

    these settings may be changed on the client side, for individual acquisitions
    (though clearly, this is highly advanced stuff, and may cause irreperable hardware damage!)
    """

    items_json=GlobalConfigHandler.get(as_dict=True)

    return json.dumps(items_json)

plate=Wellplate(
    Manufacturer="perkin elmer",
    Model_name="384 well plate",
    Model_id_manufacturer="pe-384",
    Model_id="",
    Offset_A1_x_mm=12.1,
    Offset_A1_y_mm=9.0,
    Offset_bottom_mm=14.35-10.4,
    Well_distance_x_mm=4.5,
    Well_distance_y_mm=4.5,
    Well_size_x_mm=3.65,
    Well_size_y_mm=3.65,
    Num_wells_x=24,
    Num_wells_y=16,
    Length_mm=127.76,
    Width_mm=85.48,
)

# indicate if this is the first boot of the server (this only triggers when the flask is in debug mode)
is_first_start=os.environ.get("WERKZEUG_RUN_MAIN") != "true"

class CoreLock:
    """
        basic utility to generate a token that can be used to access mutating core functions (e.g. actions)
    """

    def __init__(self):
        self._current_key:tp.Optional[str]=None
        self._key_gen_time=None
        """ timestamp when key was generated """
        self._last_key_use=None
        """ timestamp of last key use """
    def gen_key(self,invalidate_old:bool=False)->tp.Optional[str]:
        """
            generate a new key, if there is no currently valid key

            if invalidate_old is True, then the old key is discarded
        """

        if not (invalidate_old or self._current_key_is_expired()):
            return None
        
        # use 32 bytes (256 bits) for key, i.e. 64 hex chars
        # length may change at any point
        self._current_key=os.urandom(32).hex()
        self._key_gen_time=time.time()
        self._last_key_use=self._key_gen_time

        return self._current_key
    
    def _current_key_is_expired(self)->bool:
        """
            indicates if the current key is expired (i.e. returns True if key has expired)

            if no key has yet been generated, also returns True

            key expires 15 minutes after last use
        """

        if not self._last_key_use:
            return True
        return time.time()>(self._last_key_use+15*60)
    
    def key_is_valid(self,key:str)->bool:
        key_is_current_key=key==self._current_key
        if not key_is_current_key:
            return False
        # should be unreachable, but better here to reject than crash
        if self._last_key_use is None:
            return False
        
        key_has_expired=self._current_key_is_expired()
        if key_has_expired:
            return False
        self._last_key_use=time.time()
        return True

class CoreState(str,Enum):
    Idle="idle"
    ChannelSnap="channel_snap"
    ChannelStream="channel_stream"
    LoadingPosition="loading_position"
    Moving="moving"

@dataclass
class LaserAutofocusCalibrationData:
    um_per_px:float
    x_reference:float

class CoreStreamHandler:
    """
        class used to control a streaming microscope

        i.e. image acquisition is running in streaming mode
    """
    def __init__(self,core:"Core",channel_config:AcquisitionChannelConfig):
        self.core=core
        self.should_stop=False
        self.channel_config=channel_config
    def __call__(self,img:gxiapi.RawImage):
        if self.should_stop:
            return True
        
        match img.get_status():
            case gxiapi.GxFrameStatusList.INCOMPLETE:
                raise RuntimeError("incomplete frame")
            case gxiapi.GxFrameStatusList.SUCCESS:
                pass

        if img is None:
            raise RuntimeError("no image received")
        
        img_np=img.get_numpy_array()
        assert img_np is not None
        img_np=img_np.copy()

        img_np=Core._process_image(img_np)
        
        self.core._store_new_image(img_np,self.channel_config)

        return self.should_stop

class CoreCommand:
    """ virtual base class for core commands """
    def __init__(*args,**kwargs):
        pass
    def run(self,core:"Core")->str:
        raise NotImplementedError("run not implemented for basic CoreCommand")
    
CoreCommandDerived=tp.TypeVar("CoreCommandDerived",bound=CoreCommand)

class MoveBy(CoreCommand):
    def __init__(self,axis:tp.Literal["x","y","z"],distance_mm:float):
        super()
        self.axis:tp.Literal["x","y","z"]=axis
        self.distance_mm=distance_mm

    def run(self,core:"Core")->str:
        core.state=CoreState.Moving

        core.mc.send_cmd(Command.move_by_mm(self.axis,self.distance_mm))

        core.state=CoreState.Idle

        return json.dumps({"status": "success","moved_by_mm":self.distance_mm,"axis":self.axis})

class LoadingPositionEnter(CoreCommand):
    def __init__(self):
        super()
    def run(self,core:"Core")->str:
        if core.is_in_loading_position:
            return json.dumps({"status":"error","message":"already in loading position"})
        
        core.state=CoreState.Moving
        
        # home z
        core.mc.send_cmd(Command.home("z"))

        # clear clamp in y first
        core.mc.send_cmd(Command.move_to_mm("y",30))
        # then clear clamp in x
        core.mc.send_cmd(Command.move_to_mm("x",30))

        # then home y, x
        core.mc.send_cmd(Command.home("y"))
        core.mc.send_cmd(Command.home("x"))
        
        core.is_in_loading_position=True

        core.state=CoreState.LoadingPosition

        return json.dumps({"status":"success","message":"entered loading position"})

class LoadingPositionLeave(CoreCommand):
    def __init__(self):
        super()
    def run(self,core:"Core")->str:
        if not core.is_in_loading_position:
            return json.dumps({"status":"error","message":"not in loading position"})
        
        core.state=CoreState.Moving

        core.mc.send_cmd(Command.move_to_mm("x",30))
        core.mc.send_cmd(Command.move_to_mm("y",30))
        core.mc.send_cmd(Command.move_to_mm("z",1))
        
        core.is_in_loading_position=False

        core.state=CoreState.Idle

        return json.dumps({"status":"success","message":"left loading position"})

class _ChannelAcquisitionControl(CoreCommand):
    """
        control imaging in a channel

        return value fields:
            - img_handle:str (only for snap mode)
            - channel:dict (only for stream_begin mode)
    """

    def __init__(
        self,
        mode:tp.Literal['snap','stream_begin','stream_end'],
        channel:dict,
        machine_config:tp.Optional[list]=None,
        framerate_hz:tp.Optional[float]=None
    ):
        super()
        self.mode=mode
        self.channel=channel
        self.machine_config=machine_config
        self.framerate_hz=framerate_hz

    def run(self,core:"Core")->str:
        cam=core.main_cam
        
        channel_config=AcquisitionChannelConfig(**self.channel)

        try:
            illum_code=ILLUMINATION_CODE.from_handle(channel_config.handle)
        except Exception as e:
            return json.dumps({"status":"error","message":"invalid channel handle"})
        
        if self.machine_config is not None:
            GlobalConfigHandler.override(self.machine_config)

        match self.mode:
            case "snap":
                if core.is_streaming or core.stream_handler is not None:
                    return json.dumps({"status":"error","message":"already streaming"})
                
                core.state=CoreState.ChannelSnap

                core.mc.send_cmd(Command.illumination_begin(illum_code,channel_config.illum_perc))
                img=cam.acquire_with_config(channel_config)
                core.mc.send_cmd(Command.illumination_end(illum_code))
                if img is None:
                    core.state=CoreState.Idle
                    return json.dumps({"status":"error","message":"failed to acquire image"})
                

                img=Core._process_image(img)
                img_handle=core._store_new_image(img,channel_config)

                core.state=CoreState.Idle

                return json.dumps({"status":"success","img_handle":img_handle})
            
            case "stream_begin":
                if core.is_streaming or core.stream_handler is not None:
                    return json.dumps({"status":"error","message":"already streaming"})

                if self.framerate_hz is None:
                    return json.dumps({"status":"error","message":"no framerate_hz in json data"})

                core.state=CoreState.ChannelStream
                core.is_streaming=True

                core.stream_handler=CoreStreamHandler(core,channel_config)

                core.mc.send_cmd(Command.illumination_begin(illum_code,channel_config.illum_perc))

                cam.acquire_with_config(
                    channel_config,
                    mode="until_stop",
                    callback=core.stream_handler,
                    target_framerate_hz=self.framerate_hz
                )

                return json.dumps({"status":"success","channel":self.channel})
            
            case "stream_end":
                if not core.is_streaming or core.stream_handler is None:
                    return json.dumps({"status":"error","message":"not currently streaming"})
                
                core.stream_handler.should_stop=True
                core.mc.send_cmd(Command.illumination_end(illum_code))

                core.stream_handler=None
                core.is_streaming=False

                core.state=CoreState.Idle
                return json.dumps({"status":"success","message":"successfully stopped streaming"})
            
            case _o:
                raise ValueError(f"invalid mode {_o}")

class ChannelSnapshot(_ChannelAcquisitionControl):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,mode="snap",**kwargs)
class ChannelStreamBegin(_ChannelAcquisitionControl):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,mode="stream_begin",**kwargs)
class ChannelStreamEnd(_ChannelAcquisitionControl):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,mode="stream_end",**kwargs)

class MoveTo(CoreCommand):
    """
    move to target coordinates

    any of the arguments may be None, in which case the corresponding axis is not moved

    these coordinates are internally adjusted to take the calibration into account
    """
    def __init__(self,x_mm:tp.Optional[float],y_mm:tp.Optional[float],z_mm:tp.Optional[float]=None):
        super()

        self.x_mm=x_mm
        self.y_mm=y_mm
        self.z_mm=z_mm

    def run(self,core:"Core")->str:
        if core.is_in_loading_position:
            return json.dumps({"status":"error","message":"cannot move to well while in loading position"})

        if self.x_mm is not None and self.x_mm<0:
            return json.dumps({"status":"error","message":f"x coordinate out of bounds {self.x_mm = }"})
        if self.y_mm is not None and self.y_mm<0:
            return json.dumps({"status":"error","message":f"y coordinate out of bounds {self.y_mm = }"})
        if self.z_mm is not None and self.z_mm<0:
            return json.dumps({"status":"error","message":f"z coordinate out of bounds {self.z_mm = }"})

        prev_state=core.state
        core.state=CoreState.Moving

        if self.x_mm is not None:
            x_mm=core.pos_x_real_to_measured(self.x_mm)
            if x_mm<0:
                return json.dumps({"status":"error","message":f"calibrated x coordinate out of bounds {x_mm = }"})
            core.mc.send_cmd(Command.move_to_mm("x",x_mm))

        if self.y_mm is not None:
            y_mm=core.pos_y_real_to_measured(self.y_mm)
            if y_mm<0:
                return json.dumps({"status":"error","message":f"calibrated y coordinate out of bounds {y_mm = }"})
            core.mc.send_cmd(Command.move_to_mm("y",y_mm))

        if self.z_mm is not None:
            z_mm=core.pos_z_real_to_measured(self.z_mm)
            if z_mm<0:
                return json.dumps({"status":"error","message":f"calibrated z coordinate out of bounds {z_mm = }"})
            core.mc.send_cmd(Command.move_to_mm("z",z_mm))

        core.state=prev_state

        return json.dumps({"status":"success","message":"moved to x={self.x_mm:.2f} y={self.y_mm:.2f}mm"})

class MoveToWell(CoreCommand):
    def __init__(self,well_name:str):
        super()
        self.well_name=well_name

    def run(self,core:"Core")->str:
        # fov offset, because the microcontroller does not move the center of the FOV to the target position
        FOV_OFFSET_X_MM=0.3
        FOV_OFFSET_Y_MM=0.3

        x_mm=plate.get_well_offset_x(self.well_name)+plate.Well_size_x_mm/2-FOV_OFFSET_X_MM
        y_mm=plate.get_well_offset_y(self.well_name)+plate.Well_size_y_mm/2-FOV_OFFSET_Y_MM

        res_str=MoveTo(x_mm,y_mm).run(core)
        res=json.loads(res_str)
        if res["status"]!="success":
            return res_str

        return json.dumps({"status":"success","message":"moved to well "+self.well_name})

class AutofocusMeasureDisplacement(CoreCommand):
    """
        measure current displacement from reference position

        return value fields:
            - displacement_um:float

        returns json-like string
    """

    def __init__(self,override_num_images:tp.Optional[int]=None):
        super()
        self.override_num_images=override_num_images
        
    def run(self,core:"Core")->str:
        if core.laser_af_calibration_data is None:
            return json.dumps({"status":"error","message":"laser autofocus not calibrated"})

        # get laser spot location
        # sometimes one of the two expected dots cannot be found in _get_laser_spot_centroid because the plate is so far off the focus plane though, catch that case
        try:
            res=json.loads(AutofocusSnap().run(core))
            if res["status"]!="success":
                return json.dumps({"status":"error","message":"failed to snap autofocus image"})
            if core.laser_af_image is None:
                return json.dumps({"status":"error","message":"no laser autofocus image found"})
            x,y=core._get_peak_coords(core.laser_af_image.img)

            # calculate displacement
            displacement_um = (x - core.laser_af_calibration_data.x_reference)*core.laser_af_calibration_data.um_per_px
        except:
            return json.dumps({"status":"error","message":"failed to measure displacement (got no signal)"})

        return json.dumps({"status":"success","displacement_um":displacement_um})

class AutofocusSnap(CoreCommand):
    """
        snap a laser autofocus image

        returns json-like string
    """

    def __init__(self,exposure_time_ms:float=5,analog_gain:float=10,turn_laser_on:bool=True,turn_laser_off:bool=True):
        super()
        self.exposure_time_ms=exposure_time_ms
        self.analog_gain=analog_gain
        self.turn_laser_on=turn_laser_on
        self.turn_laser_off=turn_laser_off

    def run(self,core:"Core")->str:
        if self.turn_laser_on:
            core.mc.send_cmd(Command.af_laser_illum_begin())
        
        channel_config=AcquisitionChannelConfig(
            name="laser autofocus acquisition",
            handle="__invalid_laser_autofocus_channel__", # unused
            illum_perc=100, # this is not actually used
            exposure_time_ms=self.exposure_time_ms,
            analog_gain=self.analog_gain,
            z_offset_um=0 # just for clarity in code (not used)
        )
        print(f"acquiring laser autofocus image with config {channel_config}")

        img=core.focus_cam.acquire_with_config(channel_config)
        if img is None:
            self.state=CoreState.Idle
            return json.dumps({"status":"error","message":"failed to acquire image"})
        
        img_handle=core._store_new_laseraf_image(img,channel_config)

        if self.turn_laser_off:
            core.mc.send_cmd(Command.af_laser_illum_end())

        return json.dumps({
            "status":"success",
            "img_handle":img_handle,
            "width_px":img.shape[1],
            "height_px":img.shape[0],
        })

class AutofocusLaserWarmup(CoreCommand):
    """
        warm up the laser for autofocus

        sometimes the laser needs to stay on for a little bit before it can be used (the short on-time of ca. 5ms is
        sometimes not enough to turn the laser on properly without a recent warmup)
    """

    def __init__(self,warmup_time_s:float=0.5):
        super()
        self.warmup_time_s=warmup_time_s

    def run(self,core:"Core")->str:
        core.mc.send_cmd(Command.af_laser_illum_begin())

        # wait for the laser to warm up
        time.sleep(self.warmup_time_s)

        core.mc.send_cmd(Command.af_laser_illum_end())

        return json.dumps({"status":"success"})

class IlluminationEndAll(CoreCommand):
    """
        turn off all illumination sources (rather, send signal to do so. this function does not check if any are on before sending the signals)

        this function is NOT for regular operation!
        it does not verify that the microscope is not in any acquisition state

        returns json-like string
    """

    def __init__(self):
        super()

    def run(self,core:"Core")->str:
        # make sure all illumination is off
        for illum_src in [
            ILLUMINATION_CODE.ILLUMINATION_SOURCE_405NM,
            ILLUMINATION_CODE.ILLUMINATION_SOURCE_488NM,
            ILLUMINATION_CODE.ILLUMINATION_SOURCE_561NM,
            ILLUMINATION_CODE.ILLUMINATION_SOURCE_638NM,
            ILLUMINATION_CODE.ILLUMINATION_SOURCE_730NM,

            ILLUMINATION_CODE.ILLUMINATION_SOURCE_LED_ARRAY_FULL, # this will turn off the led matrix
        ]:
            core.mc.send_cmd(Command.illumination_end(illum_src))

        return json.dumps({"status":"success"})

class AutofocusApproachTargetDisplacement(CoreCommand):
    """
        move to target offset, using laser autofocus

            returns json-like string
    """

    def __init__(self,target_offset_um:float):
        super()
        self.target_offset_um=target_offset_um

    def run(self,core:"Core")->str:
        if core.laser_af_calibration_data is None:
            return json.dumps({"status":"error","message":"laser autofocus not calibrated"})
        
        if core.state!=CoreState.Idle:
            return json.dumps({"status":"error","message":"cannot move while in non-idle state"})
        
        res=json.loads(AutofocusMeasureDisplacement().run(core))
        if res["status"]!="success":
            return json.dumps({"status":"error","message":"failed to measure displacement"})
        current_displacement_um=res["displacement_um"]

        distance_to_move_to_target_mm=(self.target_offset_um-current_displacement_um)*1e-3

        old_state=core.state

        core.state=CoreState.Moving

        core.mc.send_cmd(Command.move_by_mm("z",distance_to_move_to_target_mm))

        core.state=old_state

        return json.dumps({"status":"success"})

@dataclass
class ImageStoreEntry:
    """ utility class to store camera images with some metadata """
    img:np.ndarray
    timestamp:float
    channel_config:AcquisitionChannelConfig
    bit_depth:int

class Core:
    @property
    def main_cam(self):
        defaults=GlobalConfigHandler.get_dict()
        main_camera_model_name=defaults["main_camera_model"].value
        _main_cameras=[c for c in self.cams if c.model_name==main_camera_model_name]
        if len(_main_cameras)==0:
            raise RuntimeError(f"no camera with model name {main_camera_model_name} found")
        main_camera=_main_cameras[0]
        return main_camera

    @property
    def focus_cam(self):
        defaults=GlobalConfigHandler.get_dict()
        focus_camera_model_name=defaults["laser_autofocus_camera_model"].value
        _focus_cameras=[c for c in self.cams if c.model_name==focus_camera_model_name]
        if len(_focus_cameras)==0:
            raise RuntimeError(f"no camera with model name {focus_camera_model_name} found")
        focus_camera=_focus_cameras[0]
        return focus_camera

    @property
    def calibrated_stage_position(self)->tp.Tuple[float,float,float]:
        """
        return calibrated XY stage offset from GlobalConfigHandler in order (x_mm,y_mm)
        """

        off_x_mm=GlobalConfigHandler.get_dict()["calibration_offset_x_mm"].value
        assert type(off_x_mm) is float, f"off_x_mm is {off_x_mm} of type {type(off_x_mm)}"
        off_y_mm=GlobalConfigHandler.get_dict()["calibration_offset_y_mm"].value
        assert type(off_y_mm) is float, f"off_y_mm is {off_y_mm} of type {type(off_y_mm)}"
        # TODO this is currently unused
        off_z_mm=0

        return (off_x_mm,off_y_mm,off_z_mm)

    # real/should position = measured/is position + calibrated offset

    def pos_x_measured_to_real(self,x_mm:float)->float:
        """
        convert measured x position to real position
        """
        return x_mm+self.calibrated_stage_position[0]
    def pos_y_measured_to_real(self,y_mm:float)->float:
        """
        convert measured y position to real position
        """
        return y_mm+self.calibrated_stage_position[1]
    def pos_z_measured_to_real(self,z_mm:float)->float:
        """
        convert measured z position to real position
        """
        return z_mm+self.calibrated_stage_position[2]
    def pos_x_real_to_measured(self,x_mm:float)->float:
        """
        convert real x position to measured position
        """
        return x_mm-self.calibrated_stage_position[0]
    def pos_y_real_to_measured(self,y_mm:float)->float:
        """
        convert real y position to measured position
        """
        return y_mm-self.calibrated_stage_position[1]
    def pos_z_real_to_measured(self,z_mm:float)->float:
        """
        convert real z position to measured position
        """
        return z_mm-self.calibrated_stage_position[2]

    def __init__(self):
        self.lock=CoreLock()

        self.microcontrollers=Microcontroller.get_all()
        self.cams=Camera.get_all()

        abort_startup=False

        if len(self.microcontrollers)==0:
            print("error - no microcontrollers found.")
            abort_startup=True
        if len(self.cams)<2:
            print(f"error - found less than two cameras (found {len(self.cams)})")
            abort_startup=True

        if abort_startup:
            raise RuntimeError("did not find microscope hardware")
        
        self.mc=self.microcontrollers[0]

        print(f"found {len(self.microcontrollers)} microcontrollers")
        for mc in self.microcontrollers:
            mc.open()

            # if flask server is restarted, we can skip this initialization (the microscope retains its state across reconnects)
            if is_first_start:
                print("initializing microcontroller")

                # reset the MCU
                mc.send_cmd(Command.reset())

                # reinitialize motor drivers and DAC
                mc.send_cmd(Command.initialize())
                mc.send_cmd(Command.configure_actuators())

                print("ensuring illumination is off")
                # make sure all illumination is off
                for illum_src in [
                    ILLUMINATION_CODE.ILLUMINATION_SOURCE_405NM,
                    ILLUMINATION_CODE.ILLUMINATION_SOURCE_488NM,
                    ILLUMINATION_CODE.ILLUMINATION_SOURCE_561NM,
                    ILLUMINATION_CODE.ILLUMINATION_SOURCE_638NM,
                    ILLUMINATION_CODE.ILLUMINATION_SOURCE_730NM,

                    ILLUMINATION_CODE.ILLUMINATION_SOURCE_LED_ARRAY_FULL, # this will turn off the led matrix
                ]:
                    mc.send_cmd(Command.illumination_end(illum_src))

                print("calibrating xy stage")

                # when starting up the microscope, the initial position is considered (0,0,0)
                # even homing considers the limits, so before homing, we need to disable the limits
                mc.send_cmd(Command.set_limit_mm("z",-10.0,"lower"))
                mc.send_cmd(Command.set_limit_mm("z",10.0,"upper"))

                # move objective out of the way
                mc.send_cmd(Command.home("z"))
                mc.send_cmd(Command.set_zero("z"))
                # set z limit to (or below) 6.7mm, because above that, the motor can get stuck
                mc.send_cmd(Command.set_limit_mm("z",0.0,"lower"))
                mc.send_cmd(Command.set_limit_mm("z",6.7,"upper"))
                # home x to set x reference
                mc.send_cmd(Command.home("x"))
                mc.send_cmd(Command.set_zero("x"))
                # clear clamp in x
                mc.send_cmd(Command.move_by_mm("x",30))
                # then move in position to properly apply clamp
                mc.send_cmd(Command.home("y"))
                mc.send_cmd(Command.set_zero("y"))
                # home x again to engage clamp
                mc.send_cmd(Command.home("x"))

                # move to an arbitrary position to disengage the clamp
                mc.send_cmd(Command.move_by_mm("x",30))
                mc.send_cmd(Command.move_by_mm("y",30))

                # and move objective up, slightly
                mc.send_cmd(Command.move_by_mm("z",1))

                print("warming up autofocus laser")
                AutofocusLaserWarmup().run(self)

                print("done initializing microscope")

        defaults=GlobalConfigHandler.get_dict()
        main_camera_model_name=defaults["main_camera_model"].value
        focus_camera_model_name=defaults["laser_autofocus_camera_model"].value

        print(f"found {len(self.cams)} cameras")
        for cam in self.cams:
            if cam.model_name==main_camera_model_name:
                device_type="main"
            elif cam.model_name==focus_camera_model_name:
                device_type="autofocus"
            else:
                raise RuntimeError(f"unknown camera model name {cam.model_name} is neither main nor autofocus camera")
            
            cam.open(device_type=device_type)

        if _DEBUG_P2JS:
            def sendp2():
                print("sending p2.js",list(glob.glob("../../web-pjs/p2.js")))
                return send_file("../../web-pjs/p2.js")

            app.add_url_rule(f"/p2.js", f"returnp2fromparentprojdir", sendp2,methods=["GET","POST"])

        def wrap_command(cls: tp.Type[CoreCommandDerived])->str:
            """
                wrap a command to be used as flask route

                handles: authentication, request json data presence, command verification

                to run a command:
                    - the request must contain a json object that includes some data
                    - the data mus contain information related to verification of the user to run the command
                    - the data must be valid for the requested command
            """

            # get json data
            json_data=None
            try:
                json_data=request.get_json()
            except Exception as e:
                pass

            if json_data is None:
                return json.dumps({"status": "error", "message": "no json data"})

            ret=None
            try:
                ret=cls(**json_data)
            except Exception as e:
                return json.dumps({"status": "error", "detail":"error constructing command", "message": str(e)})

            if ret is None:
                return json.dumps({"status": "error", "message": "command construction failed unexpectedly"})

            try:
                return ret.run(self)
            except Exception as e:
                return json.dumps({"status": "error", "detail":"error running command", "message": str(e)})

        # register url rules requiring machine interaction
        app.add_url_rule(
            f"/api/get_info/current_state", f"get_current_state",
            self.get_current_state,methods=["GET","POST"])

        # register urls for immediate moves
        app.add_url_rule(
            f"/api/action/move_by", f"action_move_by",
            lambda:wrap_command(MoveBy),methods=["POST"])

        # register url for start_acquisition
        app.add_url_rule(
            f"/api/acquisition/start", f"start_acquisition",
            self.start_acquisition,methods=["POST"])
        # for get_acquisition_status
        app.add_url_rule(
            f"/api/acquisition/status", f"get_acquisition_status",
            self.get_acquisition_status,methods=["GET","POST"])

        # for move_to_well
        app.add_url_rule(
            f"/api/action/move_to_well", f"move_to_well",
            lambda:wrap_command(MoveToWell),methods=["POST"])

        # send image by handle
        app.add_url_rule(
            f"/img/get_by_handle", f"send_image_by_handle",
            lambda:self.send_image_by_handle(quick_preview=False),methods=["GET"])
        app.add_url_rule(
            f"/img/get_by_handle_preview", f"send_image_by_handle_preview",
            lambda:self.send_image_by_handle(quick_preview=True),methods=["GET"])

        # loading position enter/leave
        self.is_in_loading_position=False
        app.add_url_rule(
            "/api/action/enter_loading_position", "action_enter_loading_position", 
            lambda:wrap_command(LoadingPositionEnter),methods=["POST"])
        app.add_url_rule(
            "/api/action/leave_loading_position", "action_leave_loading_position", 
            lambda:wrap_command(LoadingPositionLeave),methods=["POST"])

        # snap channel
        app.add_url_rule(
            "/api/action/snap_channel", "snap_channel", 
            lambda:wrap_command(ChannelSnapshot),methods=["POST"])

        # start streaming (i.e. acquire x images per sec, until stopped)
        self.is_streaming=False
        self.stream_handler:tp.Optional[CoreStreamHandler]=None
        app.add_url_rule(
            "/api/action/stream_channel_begin", "stream_channel_begin", 
            lambda:wrap_command(ChannelStreamBegin),methods=["POST"])
        app.add_url_rule(
            "/api/action/stream_channel_end", "stream_channel_end", 
            lambda:wrap_command(ChannelStreamEnd),methods=["POST"])

        # laser autofocus system
        self.laser_af_calibration_data:tp.Optional[LaserAutofocusCalibrationData]=None
        app.add_url_rule(
            "/api/action/snap_reflection_autofocus", "snap_reflection_autofocus", 
            lambda:wrap_command(AutofocusSnap),methods=["POST"])
        app.add_url_rule(
            "/api/action/measure_displacement", "measure_displacement", 
            lambda:wrap_command(AutofocusMeasureDisplacement),methods=["POST"])
        app.add_url_rule(
            "/api/action/laser_autofocus_move_to_target_offset", "laser_autofocus_move_to_target_offset", 
            lambda:wrap_command(AutofocusApproachTargetDisplacement),methods=["POST"])
        app.add_url_rule(
            "/api/action/laser_af_calibrate", "laser_af_calibrate", 
            self.laser_af_calibrate_here,methods=["POST"])
        app.add_url_rule(
            "/api/action/laser_af_warm_up_laser", "laser_af_warm_up_laser",
            lambda:wrap_command(AutofocusLaserWarmup),methods=["POST"])

        app.add_url_rule(
            "/api/action/calibrate_stage_xy_here", "calibrate_stage_xy_here", 
            self.calibrate_stage_xy_here,methods=["POST"])

        # for turn_off_all_illumination
        app.add_url_rule(
            "/api/action/turn_off_all_illumination", "turn_off_all_illumination", 
            lambda:wrap_command(IlluminationEndAll),methods=["POST"])

        # store last few images acquired with main imaging camera
        # TODO store these per channel, up to e.g. 3 images per channel (for a total max of 3*num_channels)
        self.latest_image_handle:tp.Optional[str]=None
        self.images:tp.Dict[str,"ImageStoreEntry"]={}

        # only store the latest laser af image
        self.laser_af_image_handle:tp.Optional[str]=None
        self.laser_af_image:tp.Optional["ImageStoreEntry"]=None

        self.state=CoreState.Idle

        self.acquisition_thread=None

    def calibrate_stage_xy_here(self)->str:
        """
            set current xy position as top left corner of B2, which is used as reference to calculate all other positions on a plate
        """

        current_pos=self.mc.get_last_position()

        # real/should position = measured/is position + calibrated offset
        # i.e. calibrated offset = real/should position - measured/is position
        ref_x_mm=plate.get_well_offset_x("B02")-current_pos.x_pos_mm
        ref_y_mm=plate.get_well_offset_y("B02")-current_pos.y_pos_mm

        # new_config_items:tp.Union[tp.Dict[str,ConfigItem
        GlobalConfigHandler.override({
            "calibration_offset_x_mm":ConfigItem(
                name="ignored",
                handle="calibration_offset_x_mm",
                value_kind="number",
                value=ref_x_mm,
            ),
            "calibration_offset_y_mm":ConfigItem(
                name="ignored",
                handle="calibration_offset_y_mm",
                value_kind="number",
                value=ref_y_mm,
            )
        })

        return json.dumps({"status":"success"})

    def _get_peak_coords(self,img:np.ndarray,use_glass_top:bool=False)->tp.Tuple[float,float]:
        """
            get peak coordinates of signal (two separate 2D gaussians) in this laser reflection autofocus image

            for air-glass-water, the smaller peak corresponds to the glass-water interface, i.e. use_glass_top -> use smaller peak

            expect some noise in this signal. for best results, use average of a few images recorded in quick succession
        """

        I = img
        # get the y position of the spots
        tmp = np.sum(I,axis=1)
        y0 = np.argmax(tmp)
        # crop along the y axis
        I = I[y0-96:y0+96,:]
        # signal along x
        tmp = np.sum(I,axis=0)
        # find peaks
        peak_locations,_ = scipy.signal.find_peaks(tmp,distance=100)

        idx = np.argsort(tmp[peak_locations])
        peak_0_location=None
        peak_1_location=None
        if len(idx)==0:
            raise Exception("did not find any peaks in Laser Reflection Autofocus signal. this is a major problem.")
        
        if use_glass_top:
            assert len(idx)>1, "only found a single peak in the Laser Reflection Autofocus signal, but trying to use the second one."
            peak_1_location = peak_locations[idx[-2]]
            x1 = peak_1_location
        else:
            peak_0_location = peak_locations[idx[-1]]
            x1 = peak_0_location

        # find centroid
        h,w = I.shape
        x,y = np.meshgrid(range(w),range(h))
        I = I[:,max(0,x1-64):min(w-1,x1+64)]
        x = x[:,max(0,x1-64):min(w-1,x1+64)]
        y = y[:,max(0,x1-64):min(w-1,x1+64)]
        I = I.astype(float)
        I = I - np.amin(I)
        I[I/np.amax(I)<0.1] = 0
        x1 = np.sum(x*I)/np.sum(I)
        y1 = np.sum(y*I)/np.sum(I)

        peak_x=x1
        peak_y=y0-96+y1

        peak_coords=(peak_x,peak_y)

        return peak_coords

    def laser_af_calibrate_here(self,z_mm_movement_range_mm:float=0.1,z_mm_backlash_counter:tp.Optional[float]=None)->str:
        """
            calibrate the laser autofocus

            calculates the conversion factor between pixels and micrometers, and sets the reference for the laser autofocus signal

            returns json-like string
        """

        # move down by half z range
        if z_mm_backlash_counter:
            self.mc.send_cmd(Command.move_by_mm("z",-(z_mm_movement_range_mm/2+z_mm_backlash_counter)))
            self.mc.send_cmd(Command.move_by_mm("z",z_mm_backlash_counter))
        else:
            self.mc.send_cmd(Command.move_by_mm("z",-z_mm_movement_range_mm/2))

        # measure pos
        res=json.loads(AutofocusSnap().run(self))
        if res["status"]!="success":
            return json.dumps({"status":"error","message":"failed to snap autofocus image [1]"})
        if self.laser_af_image is None:
            return json.dumps({"status":"error","message":"no laser autofocus image found [1]"})
        x0,y0 = self._get_peak_coords(self.laser_af_image.img)

        # move up by half z range to get position at original position, but moved to from fixed direction to counter backlash
        self.mc.send_cmd(Command.move_by_mm("z",z_mm_movement_range_mm/2))

        res=json.loads(AutofocusSnap().run(self))
        if res["status"]!="success":
            return json.dumps({"status":"error","message":"failed to snap autofocus image [2]"})
        if self.laser_af_image is None:
            return json.dumps({"status":"error","message":"no laser autofocus image found [2]"})
        x1,y1 = self._get_peak_coords(self.laser_af_image.img)

        # move up by half range again
        self.mc.send_cmd(Command.move_by_mm("z",z_mm_movement_range_mm/2))

        # measure position
        res=json.loads(AutofocusSnap().run(self))
        if res["status"]!="success":
            return json.dumps({"status":"error","message":"failed to snap autofocus image [3]"})
        if self.laser_af_image is None:
            return json.dumps({"status":"error","message":"no laser autofocus image found [3]"})
        x2,y2 = self._get_peak_coords(self.laser_af_image.img)

        # move to original position
        self.mc.send_cmd(Command.move_by_mm("z",-z_mm_movement_range_mm/2))

        self.laser_af_calibration_data=LaserAutofocusCalibrationData(
            # calculate the conversion factor, based on lowest and highest measured position
            um_per_px=z_mm_movement_range_mm*1e3/(x2-x0),
            # set reference position
            x_reference=x1
        )

        return json.dumps({"status":"success"})

    @staticmethod
    def _process_image(img:np.ndarray)->np.ndarray:
        """
            crop to size
        """

        g_config=GlobalConfigHandler.get_dict()

        cam_img_width=g_config['main_camera_image_width_px']
        assert cam_img_width is not None
        target_width:int=cam_img_width.intvalue
        assert isinstance(target_width,int), f"{type(target_width) = }"
        cam_img_height=g_config['main_camera_image_height_px']
        assert cam_img_height is not None
        target_height:int=cam_img_height.intvalue
        assert isinstance(target_height,int), f"{type(target_height) = }"
        
        current_height=img.shape[0]
        current_width=img.shape[1]

        assert target_width<=current_width, f"{target_width = } ; {current_width = }"
        assert target_height<=current_height, f"{target_height = } ; {current_height = }"

        x_offset=(current_width-target_width)//2
        y_offset=(current_height-target_height)//2

        # seemingly swap x and y because of numpy's row-major order
        ret=img[y_offset:y_offset+target_height,x_offset:x_offset+target_width]

        flip_img_horizontal=g_config['main_camera_image_flip_horizontal']
        assert flip_img_horizontal is not None
        if flip_img_horizontal.boolvalue:
            ret=np.flip(ret,axis=1)

        flip_img_vertical=g_config['main_camera_image_flip_vertical']
        assert flip_img_vertical is not None
        if flip_img_vertical.boolvalue:
            ret=np.flip(ret,axis=0)

        return ret

    def _store_new_laseraf_image(self,img:np.ndarray,channel_config:AcquisitionChannelConfig)->str:
        """
            store a new laser autofocus image, return the handle
        """

        # TODO better image buffer handle generation
        if self.laser_af_image_handle is None:
            self.laser_af_image_handle=f"laseraf_{0}"
        else:
            self.laser_af_image_handle=f"laseraf_{int(self.laser_af_image_handle.split('_')[1])+1}"
            
        pxfmt_int,pxfmt_str=self.focus_cam.handle.PixelFormat.get() # type: ignore
        match pxfmt_int:
            case gxiapi.GxPixelFormatEntry.MONO8:
                pixel_depth=8
            case gxiapi.GxPixelFormatEntry.MONO10:
                pixel_depth=10
            case gxiapi.GxPixelFormatEntry.MONO12:
                pixel_depth=12
            case _unknown:
                raise ValueError(f"unexpected pixel format {pxfmt_int = } {pxfmt_str = }")

        self.laser_af_image=ImageStoreEntry(img,time.time(),channel_config,pixel_depth)

        print(f"saved image of shape {img.shape} and dtype {img.dtype} with handle {self.laser_af_image_handle}")

        return self.laser_af_image_handle

    def _store_new_image(self,img:np.ndarray,channel_config:AcquisitionChannelConfig)->str:
        """
            store a new image, return the handle
        """

        # TODO better image buffer handle generation
        if self.latest_image_handle is None:
            self.latest_image_handle=f"{0}"
        else:
            self.latest_image_handle=f"{int(self.latest_image_handle)+1}"

        pxfmt_int,pxfmt_str=self.main_cam.handle.PixelFormat.get() # type: ignore
        match pxfmt_int:
            case gxiapi.GxPixelFormatEntry.MONO8:
                pixel_depth=8
            case gxiapi.GxPixelFormatEntry.MONO10:
                pixel_depth=10
            case gxiapi.GxPixelFormatEntry.MONO12:
                pixel_depth=12
            case _unknown:
                raise ValueError(f"unexpected pixel format {pxfmt_int = } {pxfmt_str = }")
            
        self.images[self.latest_image_handle]=ImageStoreEntry(img,time.time(),channel_config,pixel_depth)

        # remove oldest images to keep buffer length capped (at 8)
        while len(self.images)>8:
            oldest_key=min(self.images,key=lambda k:self.images[k].timestamp)
            del self.images[oldest_key].img # delete this explicitely to free ndarray memory
            del self.images[oldest_key]
            gc.collect() # force gc collection because these images really take up a lot of storage

        return self.latest_image_handle

    def send_image_by_handle(self,quick_preview:bool=False):
        """
            send image by handle, as get request to allow using this a img src
        """

        img_handle:tp.Optional[str]=None
        try:
            img_handle=request.args.get("img_handle")
        except Exception as e:
            pass

        img_container=None
        if img_handle==self.laser_af_image_handle:
            img_container=self.laser_af_image

        else:
            if img_handle is None:
                return json.dumps({"status":"error","message":"no img_handle provided"})
            
            if img_handle not in self.images:
                return json.dumps({"status":"error","message":f"img_handle {img_handle} not found"})

            img_container=self.images[img_handle]

        if img_container is None:
            return json.dumps({"status":"error","message":"no image found"})

        img_raw=img_container.img

        # convert from u12 to u8
        match img_raw.dtype:
            case np.uint16:
                match img_container.bit_depth:
                    case 12:
                        img=(img_raw>>(16-12)).astype(np.uint8)
                    case 10:
                        img=(img_raw>>(16-10)).astype(np.uint8)
                    case _:
                        raise ValueError(f"unexpected bit depth {img_container.bit_depth}")
            case np.uint8:
                assert img_container.bit_depth==8, f"unexpected {img_container.bit_depth = }"
                img=img_raw
            case _:
                raise ValueError(f"unexpected dtype {img_raw.dtype}")

        if quick_preview:
            preview_resolution_scaling=GlobalConfigHandler.get_dict()["streaming_preview_resolution_scaling"].intvalue

            img=img[::preview_resolution_scaling,::preview_resolution_scaling]

        img_pil=Image.fromarray(img,mode="L")

        img_io=io.BytesIO()
        
        if quick_preview:
            g_config=GlobalConfigHandler.get_dict()
            streaming_format_item=g_config["streaming_preview_format"]
            assert streaming_format_item is not None
            streaming_format=streaming_format_item.value
            match streaming_format:
                case "jpeg":
                    pil_kwargs,mimetype={"format":"JPEG","quality":90},"image/jpeg"
                case "png":
                    pil_kwargs,mimetype={"format":"PNG","compress_level":3},"image/png"
                case _other:
                    raise ValueError(f"unexpected streaming_preview_format format {streaming_format}")
                
        else:
            g_config=GlobalConfigHandler.get_dict()
            streaming_format_item=g_config["image_display_format"]
            assert streaming_format_item is not None
            streaming_format=streaming_format_item.value
            match streaming_format:
                case "jpeg":
                    pil_kwargs,mimetype={"format":"JPEG","quality":95},"image/jpeg"
                case "png":
                    pil_kwargs,mimetype={"format":"PNG","compress_level":6},"image/png"
                case _other:
                    raise ValueError(f"unexpected image_display_format format {streaming_format}")
            
        img_pil.save(img_io,**pil_kwargs)
        img_io.seek(0)

        return send_file(img_io, mimetype=mimetype)

    def get_current_state(self):
        last_stage_position=self.mc.get_last_position()

        img_handle=self.latest_image_handle
        latest_img_info=None
        if img_handle is not None and img_handle in self.images:
            latest_img_info={
                "handle":img_handle,
                "channel":self.images[img_handle].channel_config.to_dict(),
                "width_px":self.images[img_handle].img.shape[1],
                "height_px":self.images[img_handle].img.shape[0],
            }

        # supposed=real-calib
        x_pos_mm=self.pos_x_measured_to_real(last_stage_position.x_pos_mm)
        y_pos_mm=self.pos_y_measured_to_real(last_stage_position.y_pos_mm)
        
        return json.dumps({
            "status":"success",
            "state":self.state.value,
            "stage_position":{
                "x_pos_mm":x_pos_mm,
                "y_pos_mm":y_pos_mm,
                "z_pos_mm":last_stage_position.z_pos_mm,
            },
            "latest_img":latest_img_info
        })

    def start_acquisition(self):
        """
        start an acquisition with the given configuration
        """

        # get post data with key "config_file"
        json_data=None
        try:
            json_data=request.get_json()
        except Exception as e:
            pass

        if json_data is None:
            return json.dumps({"status": "error", "message": "no json data"})
        
        if self.acquisition_thread is not None:
            if self.acquisition_thread.is_alive():
                return json.dumps({"status": "error", "message": "acquisition already running"})
            else:
                self.acquisition_thread=None
        
        # get machine config
        if "machine_config" not in json_data:
            return json.dumps({"status": "error", "message": "no machine_config in json data"})
        
        GlobalConfigHandler.override(json_data["machine_config"])
        
        # get acquisition config
        if "config_file" not in json_data:
            return json.dumps({"status": "error", "message": "no config_file in json data"})
        
        config = AcquisitionConfig.from_json(json_data["config_file"])

        print("starting acquisition with config:",config)

        # TODO generate some unique acqisition id to identify this acquisition by
        # must be robust against server restarts, and async requests
        # must also cache previous run results, to allow fetching information from past acquisitions
        RANDOM_ACQUISITION_ID="a65914"
        # actual acquisition id has some unique identifier, plus a timestamp for legacy reasons
        # TODO remove the timestamp from the directory name in the future, because this is a horribly hacky way
        # to handle duplicate directory names, which should be avoided anyway
        acquisition_id=f"{RANDOM_ACQUISITION_ID}_{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}"

        print(f"acquiring '{acquisition_id}':")

        well_sites=config.grid.mask
        # the grid is centered around the center of the well
        site_topleft_x_mm=plate.Well_size_x_mm / 2 - (config.grid.num_x * config.grid.delta_x_mm) / 2
        "offset of top left site from top left corner of the well, in x, in mm"
        site_topleft_y_mm=plate.Well_size_y_mm / 2 - (config.grid.num_y * config.grid.delta_y_mm) / 2
        "offset of top left site from top left corner of the well, in y, in mm"

        num_wells=len(config.plate_wells)
        num_sites=len(well_sites)
        num_channels=len(config.channels)
        num_images_total=num_wells*num_sites*num_channels
        print(f"acquiring {num_wells} wells, {num_sites} sites, {num_channels} channels, i.e. {num_images_total} images")

        if num_images_total==0:
            return json.dumps({"status":"error","message":"no images to acquire"})

        # calculate meta information about acquisition
        g_config=GlobalConfigHandler.get_dict()
        cam_img_width=g_config['main_camera_image_width_px']
        assert cam_img_width is not None
        target_width:int=cam_img_width.intvalue
        cam_img_height=g_config['main_camera_image_height_px']
        assert cam_img_height is not None
        target_height:int=cam_img_height.intvalue
        # get byte size per pixel from config main camera pixel format
        main_cam_pix_format=g_config['main_camera_pixel_format']
        assert main_cam_pix_format is not None
        match main_cam_pix_format.value:
            case "mono8":
                bytes_per_pixel=1
            case "mono10":
                bytes_per_pixel=2
            case "mono12":
                bytes_per_pixel=2
            case _unexpected:
                raise ValueError(f"unexpected main camera pixel format '{_unexpected}' in {main_cam_pix_format}")
            
        image_size_bytes=target_width*target_height*bytes_per_pixel
        max_storage_size_images_GB=num_images_total*image_size_bytes

        # 1) sort wells
        # 2) sort channels (by z order)

        DISPLACEMENT_THRESHOLD_UM=0.5
        "z movement below this threshold is not performed"

        base_storage_path_item=g_config["base_image_output_dir"]
        assert base_storage_path_item is not None
        assert type(base_storage_path_item.value) is str
        base_storage_path=Path(base_storage_path_item.value)
        assert base_storage_path.exists(), f"{base_storage_path = } does not exist"

        project_name_is_acceptable=len(config.project_name) and name_validity_regex.match(config.project_name)
        if not project_name_is_acceptable:
            return json.dumps({
                "status":"error",
                "message":"project name is not acceptable: 1) must not be empty and 2) only contain alphanumeric characters, underscores, dashes"
            })
        plate_name_is_acceptable=len(config.plate_name)>0 and name_validity_regex.match(config.plate_name)
        if not plate_name_is_acceptable:
            return json.dumps({
                "status":"error",
                "message":"plate name is not acceptable: 1) must not be empty and 2) only contain alphanumeric characters, underscores, dashes"
            })

        project_output_path=base_storage_path/config.project_name/config.plate_name/acquisition_id
        project_output_path.mkdir(parents=True)

        channels:tp.List[AcquisitionChannelConfig]=[]
        for channel in config.channels:
            if channel.enabled:
                channels.append(channel)

        def run_acquisition(q_in:q.Queue,q_out:q.Queue):
            try:
                # counters on acquisition progress
                start_time=time.time()
                start_time_iso_str=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
                last_image_information=None

                num_images_acquired=0
                storage_usage_bytes=0

                # get current z coordinate as z reference
                reference_z_mm=self.mc.get_last_position().z_pos_mm

                # run acquisition:

                # 3) for well in wells:
                for well in config.plate_wells:
                # 4)for site in well:
                    for site in well_sites:
                # 5)    go to site
                        site_x_mm=plate.get_well_offset_x(well.well_name)+site_topleft_x_mm+site.col*config.grid.delta_x_mm
                        site_y_mm=plate.get_well_offset_y(well.well_name)+site_topleft_y_mm+site.row*config.grid.delta_y_mm

                        res_str=MoveTo(site_x_mm,site_y_mm).run(self)
                        res=json.loads(res_str)
                        if res["status"]!="success":
                            q_out.put({"status":"error","message":f"failed to move to site {site} in well {well} because {res['message']}"})
                            return

                # 6)    run autofocus
                        if config.autofocus_enabled:
                            for num_autofocus_attempts in range(3):
                                current_displacement_res=AutofocusMeasureDisplacement().run(self)
                                current_displacement=json.loads(current_displacement_res)
                                if current_displacement["status"]!="success":
                                    q_out.put({"status":"error","message":f"failed to measure autofocus displacement at site {site} in well {well} because {current_displacement['message']}"})
                                    return
                                
                                current_displacement_um=current_displacement["displacement_um"]
                                if current_displacement_um<DISPLACEMENT_THRESHOLD_UM:
                                    break
                                
                                res_str=AutofocusApproachTargetDisplacement(0).run(self)
                                res=json.loads(res_str)
                                if res["status"]!="success":
                                    q_out.put({"status":"error","message":f"failed to autofocus at site {site} in well {well} because {res['message']}"})
                                    return
                            
                            reference_z_mm=self.mc.get_last_position().z_pos_mm
                        
                # 7)    for channel in channels:
                        for channel in channels:
                # 8)        move to channel offset
                            current_z_mm=self.mc.get_last_position().z_pos_mm
                            channel_z_mm=channel.z_offset_um*1e-3+reference_z_mm
                            distance_z_to_move_mm=channel_z_mm-current_z_mm
                            if np.abs(distance_z_to_move_mm)>DISPLACEMENT_THRESHOLD_UM:
                                res_str=MoveTo(None,None,channel_z_mm).run(self)

                # 9)        snap image
                            res_str=_ChannelAcquisitionControl("snap",channel.to_dict()).run(self)
                            res=json.loads(res_str)
                            if res["status"]!="success":
                                q_out.put({"status":"error","message":f"failed to snap image at site {site} in well {well} because {res['message']}"})
                                return                           

                            img_handle=res["img_handle"]
                            if not img_handle in self.images:
                                q_out.put({"status":"error","message":f"image handle {img_handle} not found in image buffer"})
                                return
                            
                            img=self.images[img_handle].img
                            
                # 10)       store image
                            image_storage_path=f"{str(project_output_path)}/{well.well_name}_1_{site.col+1}_{site.row+1}_{site.plane+1}_{channel.handle}.tiff"
                            # add metadata to the file, based on tags from https://exiftool.org/TagNames/EXIF.html
                            PIXEL_SIZE_UM=900/2500 # 2500px wide fov covers 0.9mm

                            if channel.handle[:3]=="BF ":
                                light_source_type=4 # Flash (seems like the best fit)
                            else:
                                light_source_type=2 # Fluorescent


                            # store img as .tif file
                            tifffile.imwrite(
                                str(image_storage_path),
                                img,

                                compression='LZW', # LZW or zlib (are lossless formats)
                                compressionargs={},# for zlib: {'level': 8},
                                maxworkers=3,

                                # this metadata is just embedded as a comment in the file
                                metadata={
                                    # non-standard tag, because the standard bitsperpixel has a real meaning in the image image data, but
                                    # values that are not multiple of 8 cannot be used with compression
                                    "BitsPerPixel":self.images[img_handle].bit_depth,
                                    "BitPaddingInfo":"lower bits are padding with 0s",

                                    # non-standard tag
                                    "LightSourceName":channel.name,

                                    "ExposureTimeMS":f"{channel.exposure_time_ms:.2f}",
                                    "AnalogGainDB":f"{channel.analog_gain:.2f}",
                                },

                                photometric="minisblack",
                                resolutionunit=3, # 3 = cm
                                resolution=(1/(PIXEL_SIZE_UM*1e-6),1/(PIXEL_SIZE_UM*1e-6)),
                                extratags=[
                                    # tuples as accepted at https://github.com/blink1073/tifffile/blob/master/tifffile/tifffile.py#L856
                                    # tags may be specified as string interpretable by https://github.com/blink1073/tifffile/blob/master/tifffile/tifffile.py#L5000
                                    ("Make", 's', 0, self.main_cam.vendor_name, True),
                                    ("Model", 's', 0, self.main_cam.model_name, True),

                                    # apparently these are not widely supported
                                    ("ExposureTime", 'q', 1, int(channel.exposure_time_ms*1e3), True),
                                    ("LightSource", 'h', 1, light_source_type, True),
                                ]
                            )
                            # also update the image buffer
                            img_handle=self._store_new_image(img,channel)

                            num_images_acquired+=1
                            # get storage size from filesystem because tiff compression may reduce size below size in memory
                            try:
                                file_size_on_disk=Path(image_storage_path).stat().st_size
                                storage_usage_bytes+=file_size_on_disk
                            except:
                                # ignore any errors here, because this is not an essential feature
                                pass

                            # status items
                            last_image_information={
                                "well":well.well_name,
                                "site":{
                                    "x":site.col,
                                    "y":site.row,
                                    "z":0,
                                },
                                "timepoint":1,
                                "channel_name":channel.name,
                                "full_path":image_storage_path,
                                "handle":img_handle,
                            }
                            time_since_start_s=time.time()-start_time
                            if num_images_acquired>0:
                                estimated_total_time_s=time_since_start_s*(num_images_total-num_images_acquired)/num_images_acquired
                            else:
                                estimated_total_time_s=None

                            status={
                                "status":"success",

                                "acquisition_id":acquisition_id,
                                "acquisition_status":"running",
                                "acquisition_progress":{
                                    # measureable progress
                                    "current_num_images":num_images_acquired,
                                    "time_since_start_s":time_since_start_s,
                                    "start_time_iso":start_time_iso_str,
                                    "current_storage_usage_GB":storage_usage_bytes/(1024**3),

                                    # estimated completion time information
                                    # estimation may be more complex than linear interpolation, hence done on server side
                                    "estimated_total_time_s":estimated_total_time_s,

                                    # last image that was acquired
                                    "last_image":last_image_information,
                                },

                                # some meta information about the acquisition, derived from configuration file
                                # i.e. this is not updated during acquisition
                                "acquisition_meta_information":{
                                    "total_num_images":num_images_total,
                                    "max_storage_size_images_GB":max_storage_size_images_GB,
                                },

                                "acquisition_config":config.to_dict(),

                                "message":f"Acquisition is {(100*num_images_acquired/num_images_total):.2f}% complete"
                            }

                            # make sure the queue does not exceed empty queue before we put the update
                            MAX_QUEUE_SIZE_SOFT=8
                            while q_out.qsize()>MAX_QUEUE_SIZE_SOFT:
                                try:
                                    q_out.get_nowait()
                                except q.Empty:
                                    break
                            q_out.put(status)

            except Exception as e:
                import traceback
                full_error=traceback.format_exc()
                q_out.put({"status":"error","message":f"acquisition thread failed because {str(e)}, more specifically: {full_error}"})
                return

        queue_in=q.Queue()
        queue_out=q.Queue()
        self.acquisition_thread=th.Thread(target=run_acquisition,args=(queue_in,queue_out))
        self.acquisition_thread.start()

        somemap[acquisition_id]={
            "acquisition_id":acquisition_id,
            "queue_in":queue_in,
            "queue_out":queue_out,
            "last_status":None,
            "thread_is_running":True,
        }

        return json.dumps({"status": "success","acquisition_id":acquisition_id})

    def get_acquisition_status(self):
        """
        get status of an acquisition
        """

        acquisition_id=None
        try:
            acquisition_id=request.get_json()["acquisition_id"]
        except Exception as e:
            pass

        if acquisition_id is None:
            return json.dumps({"status":"error","message":"no acquisition_id provided"})

        if acquisition_id not in somemap:
            return json.dumps({"status":"error","message":"acquisition_id is invalid"})

        acq_res=somemap[acquisition_id]

        # actual code to check if acquisition is running
        msg=None
        while True:
            try:
                msg=acq_res["queue_out"].get_nowait()
            except q.Empty:
                break

        if msg is not None:
            acq_res["last_status"]=msg
        else:
            msg=acq_res["last_status"]

        if msg is None:
            return json.dumps({"status":"error","message":"no status available"})

        if msg["status"]!="success":
            if self.acquisition_thread is not None:
                print("waiting for acquisition thread to terminate")
                self.acquisition_thread.join()
                print("acquisition thread terminated")
                self.acquisition_thread=None

        return json.dumps(msg)
    
    def close(self):
        for mc in self.microcontrollers:
            mc.close()

        for cam in self.cams:
            cam.close()

        GlobalConfigHandler.store()


if __name__ == "__main__":
    core=Core()
    
    try:
        # running in debug mode doesnt work because then I am unable to properly handle the close event
        app.run(debug=False, port=5002)
    except Exception:
        pass

    print("shutting down")
    core.close()
