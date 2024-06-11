import json, time, os, io, sys
from flask import Flask, send_from_directory, request, send_file
import numpy as np
from PIL import Image
import typing as tp
from dataclasses import dataclass
from enum import Enum
import scipy

from seaconfig import *
from .hardware.camera import Camera, gxiapi
from .hardware.microcontroller import Microcontroller, ILLUMINATION_CODE
Command=Microcontroller.Command

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

@app.route('/p/<path:path>')
def send_p(path):
    return send_from_directory('p', path)

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
    image_path=request.args.get("image_path")
    if image_path is None:
        return json.dumps({"status":"error","message":"no image_path provided"})

    if image_path not in remote_image_path_map:
        return json.dumps({"status":"error","message":"image_path not found"})

    return send_from_directory(remote_image_path_map[image_path]["dir"], remote_image_path_map[image_path]["filename"])

somemap:dict[str,dict]={}

# get objectives
@app.route("/api/get_features/hardware_capabilities", methods=["GET","POST"])
def get_hardware_capabilities():
    """
    get a list of all the hardware capabilities of the system

    these are the high-level configuration options that the user can select from
    """

    ret={
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
                # reset the MCU
                mc.send_cmd(Command.reset())

                # reinitialize motor drivers and DAC
                mc.send_cmd(Command.initialize())
                mc.send_cmd(Command.configure_actuators())

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
                print("done initializing microcontroller")

        print(f"found {len(self.cams)} cameras")
        for cam in self.cams:
            cam.open()

        # register url rules requiring machine interaction
        app.add_url_rule(f"/api/get_info/current_state", f"get_current_state", self.get_current_state,methods=["GET","POST"])
        app.add_url_rule(f"/api/action/move_x_by", f"action_move_x_by", lambda:self.action_move_by("x"),methods=["POST"])
        app.add_url_rule(f"/api/action/move_y_by", f"action_move_y_by", lambda:self.action_move_by("y"),methods=["POST"])
        app.add_url_rule(f"/api/action/move_z_by", f"action_move_z_by", lambda:self.action_move_by("z"),methods=["POST"])
        # register url for start_acquisition
        app.add_url_rule(f"/api/acquisition/start", f"start_acquisition", self.start_acquisition,methods=["POST"])
        # for get_acquisition_status
        app.add_url_rule(f"/api/acquisition/status", f"get_acquisition_status", self.get_acquisition_status,methods=["GET","POST"])
        # for move_to_well
        app.add_url_rule(f"/api/action/move_to_well", f"move_to_well", self.move_to_well,methods=["POST"])
        # send image by handle
        app.add_url_rule(f"/img/get_by_handle", f"send_image_by_handle", self.send_image_by_handle,methods=["GET"])

        # loading position enter/leave
        self.is_in_loading_position=False
        app.add_url_rule("/api/action/leave_loading_position", "action_leave_loading_position", self.action_leave_loading_position,methods=["POST"])
        app.add_url_rule("/api/action/enter_loading_position", "action_enter_loading_position", self.action_enter_loading_position,methods=["POST"])

        # snap channel
        app.add_url_rule("/api/action/snap_channel", "snap_channel", lambda:self.image_channel(mode='snap'),methods=["POST"])

        # start streaming (i.e. acquire x images per sec, until stopped)
        self.laser_af_calibration_data:tp.Optional[LaserAutofocusCalibrationData]=None
        app.add_url_rule("/api/action/stream_channel_begin", "stream_channel_begin", lambda:self.image_channel(mode='stream_begin'),methods=["POST"])
        app.add_url_rule("/api/action/stream_channel_end", "stream_channel_end", lambda:self.image_channel(mode='stream_end'),methods=["POST"])
        self.is_streaming=False
        self.stream_handler=None

        # laser autofocus system
        app.add_url_rule("/api/action/snap_reflection_autofocus", "snap_reflection_autofocus", self.snap_autofocus,methods=["POST"])
        app.add_url_rule("/api/action/laser_af_calibrate", "laser_af_calibrate", self.laser_af_calibrate,methods=["POST"])
        app.add_url_rule("/api/action/measure_displacement", "measure_displacement", self.measure_displacement,methods=["POST"])
        app.add_url_rule("/api/action/laser_autofocus_move_to_target_offset", "laser_autofocus_move_to_target_offset", self.laser_autofocus_move_to_target_offset,methods=["POST"])

        # store last few images acquired with main imaging camera
        # TODO store these per channel, up to e.g. 3 images per channel (for a total max of 3*num_channels)
        self.latest_image_handle:tp.Optional[str]=None
        self.images:tp.Dict[str,"Core.ImageStoreEntry"]={}

        # only store the latest laser af image
        self.laser_af_image_handle:tp.Optional[str]=None
        self.laser_af_image:tp.Optional["Core.ImageStoreEntry"]=None

        self.state=CoreState.Idle

    def laser_autofocus_move_to_target_offset(self)->str:
        """
            move to target offset, using laser autofocus

             returns json-like string
        """

        if self.laser_af_calibration_data is None:
            return json.dumps({"status":"error","message":"laser autofocus not calibrated"})
        
        if self.state!=CoreState.Idle:
            return json.dumps({"status":"error","message":"cannot move while in non-idle state"})
        
        data=None
        try:
            data=request.get_json()
        except Exception as e:
            pass

        if data is None:
            return json.dumps({"status":"error","message":"no json data"})
        
        res=json.loads(self.measure_displacement())
        if res["status"]!="success":
            return json.dumps({"status":"error","message":"failed to measure displacement"})
        
        if "target_offset_um" not in data:
            return json.dumps({"status":"error","message":"no target_offset_um in json data"})
        target_displacement_um=data["target_offset_um"]

        current_displacement_um=res["displacement_um"]
        distance_to_move_to_target_mm=(target_displacement_um-current_displacement_um)*1e-3

        old_state=self.state
        self.state=CoreState.Moving
        self.mc.send_cmd(Command.move_by_mm("z",distance_to_move_to_target_mm))
        self.state=old_state

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

    def laser_af_calibrate(self,z_mm_movement_range_mm:float=0.1,z_mm_backlash_counter:tp.Optional[float]=None)->str:
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
        res=json.loads(self.snap_autofocus())
        if res["status"]!="success":
            return json.dumps({"status":"error","message":"failed to snap autofocus image [1]"})
        if self.laser_af_image is None:
            return json.dumps({"status":"error","message":"no laser autofocus image found [1]"})
        x0,y0 = self._get_peak_coords(self.laser_af_image.img)

        # move up by half z range to get position at original position, but moved to from fixed direction to counter backlash
        self.mc.send_cmd(Command.move_by_mm("z",z_mm_movement_range_mm/2))

        res=json.loads(self.snap_autofocus())
        if res["status"]!="success":
            return json.dumps({"status":"error","message":"failed to snap autofocus image [2]"})
        if self.laser_af_image is None:
            return json.dumps({"status":"error","message":"no laser autofocus image found [2]"})
        x1,y1 = self._get_peak_coords(self.laser_af_image.img)

        # move up by half range again
        self.mc.send_cmd(Command.move_by_mm("z",z_mm_movement_range_mm/2))

        # measure position
        res=json.loads(self.snap_autofocus())
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

    def measure_displacement(self,override_num_images:tp.Optional[int]=None)->str:
        """
            measure current displacement from reference position

            returns json-like string
        """
        if self.laser_af_calibration_data is None:
            return json.dumps({"status":"error","message":"laser autofocus not calibrated"})

        # get laser spot location
        # sometimes one of the two expected dots cannot be found in _get_laser_spot_centroid because the plate is so far off the focus plane though, catch that case
        try:
            res=json.loads(self.snap_autofocus())
            if res["status"]!="success":
                return json.dumps({"status":"error","message":"failed to snap autofocus image"})
            if self.laser_af_image is None:
                return json.dumps({"status":"error","message":"no laser autofocus image found"})
            x,y=self._get_peak_coords(self.laser_af_image.img)

            # calculate displacement
            displacement_um = (x - self.laser_af_calibration_data.x_reference)*self.laser_af_calibration_data.um_per_px
        except:
            return json.dumps({"status":"error","message":"failed to measure displacement (got no signal)"})

        return json.dumps({"status":"success","displacement_um":displacement_um})

    def snap_autofocus(self)->str:
        """
            snap a laser autofocus image

            returns json-like string
        """

        self.mc.send_cmd(Command.af_laser_illum_begin())
        
        channel_config=AcquisitionChannelConfig(name="whatever",handle="whatever",illum_perc=100,exposure_time_ms=5.0,analog_gain=10,z_offset_um=0)

        cam=self.focus_cam
        img=cam.acquire_with_config(channel_config)
        if img is None:
            self.state=CoreState.Idle
            return json.dumps({"status":"error","message":"failed to acquire image"})
        
        #img=Core._process_image(img)
        img_handle=self._store_new_laseraf_image(img,channel_config)

        self.mc.send_cmd(Command.af_laser_illum_end())

        return json.dumps({"status":"success","img_handle":img_handle})

    def _create_stream_handler(self,channel_config:AcquisitionChannelConfig):
        class StreamHandler:
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

        return StreamHandler(self,channel_config)

    @dataclass
    class ImageStoreEntry:
        img:np.ndarray
        timestamp:float
        channel_config:AcquisitionChannelConfig

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
        print(f"cropping from (WxH) {current_width}x{current_height} to {target_width}x{target_height}")

        assert target_width<=current_width, f"{target_width = } ; {current_width = }"
        assert target_height<=current_height, f"{target_height = } ; {current_height = }"

        x_offset=(current_width-target_width)//2
        y_offset=(current_height-target_height)//2

        # seemingly swap x and y because of numpy's row-major order
        ret=img[y_offset:y_offset+target_height,x_offset:x_offset+target_width]

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
            
        self.laser_af_image=Core.ImageStoreEntry(img,time.time(),channel_config)

        print(f"saved image of shape {img.shape} and dtype {img.dtype} with handle {self.latest_image_handle}")

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
            
        self.images[self.latest_image_handle]=Core.ImageStoreEntry(img,time.time(),channel_config)

        print(f"saved image of shape {img.shape} and dtype {img.dtype} with handle {self.latest_image_handle}")

        return self.latest_image_handle

    def image_channel(self,mode:tp.Literal['snap','stream_begin','stream_end']):
        """
            control imaging in a channel
        """

        cam=self.main_cam

        # get json data
        json_data=None
        try:
            json_data=request.get_json()
        except Exception as e:
            pass

        if json_data is None:
            return json.dumps({"status": "error", "message": "no json data"})
        
        if "channel" not in json_data:
            return json.dumps({"status": "error", "message": "no channel in json data"})
        
        channel_config=AcquisitionChannelConfig(**json_data["channel"])

        try:
            illum_code=ILLUMINATION_CODE.from_handle(channel_config.handle)
        except Exception as e:
            return json.dumps({"status":"error","message":"invalid channel handle"})
        
        if "machine_config" in json_data:
            GlobalConfigHandler.override(json_data["machine_config"])

        match mode:
            case "snap":
                if self.is_streaming or self.stream_handler is not None:
                    return json.dumps({"status":"error","message":"already streaming"})
                
                self.state=CoreState.ChannelSnap

                self.mc.send_cmd(Command.illumination_begin(illum_code,channel_config.illum_perc))
                img=cam.acquire_with_config(channel_config)
                self.mc.send_cmd(Command.illumination_end(illum_code))
                if img is None:
                    self.state=CoreState.Idle
                    return json.dumps({"status":"error","message":"failed to acquire image"})
                

                img=Core._process_image(img)
                img_handle=self._store_new_image(img,channel_config)

                self.state=CoreState.Idle
                return json.dumps({"status":"success","img_handle":img_handle})
            case "stream_begin":
                if self.is_streaming or self.stream_handler is not None:
                    return json.dumps({"status":"error","message":"already streaming"})

                self.state=CoreState.ChannelStream
                self.is_streaming=True

                self.stream_handler=self._create_stream_handler(channel_config)
                self.mc.send_cmd(Command.illumination_begin(illum_code,channel_config.illum_perc))
                cam.acquire_with_config(channel_config,mode="until_stop",callback=self.stream_handler)

                return json.dumps({"status":"success","channel":json_data["channel"]})
            case "stream_end":
                if not self.is_streaming or self.stream_handler is None:
                    return json.dumps({"status":"error","message":"not currently streaming"})
                
                self.stream_handler.should_stop=True
                self.mc.send_cmd(Command.illumination_end(illum_code))

                self.stream_handler=None
                self.is_streaming=False

                self.state=CoreState.Idle
                return json.dumps({"status":"success","message":"successfully stopped streaming"})
            case _o:
                raise ValueError(f"invalid mode {_o}")

    def send_image_by_handle(self):
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
            # remove oldest images to keep buffer length capped (at 8)
            while len(self.images)>8:
                oldest_key=min(self.images,key=lambda k:self.images[k].timestamp)
                del self.images[oldest_key]

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
                print(f"img_raw shape: {img_raw.shape}, dtype: {img_raw.dtype}")
                print(f"img min max: {img_raw.min()} {img_raw.max()} (max u16 is {2**16-1}, max u12 is {2**12-1})")
                img=(img_raw>>4).astype(np.uint8)
            case np.uint8:
                img=img_raw
            case _:
                raise ValueError(f"unexpected dtype {img_raw.dtype}")

        preview_resolution_scaling=GlobalConfigHandler.get_dict()["preview_resolution_scaling"].intvalue

        img_downres=img[::preview_resolution_scaling,::preview_resolution_scaling]
        img_pil=Image.fromarray(img_downres,mode="L")

        img_io=io.BytesIO()
        img_pil.save(img_io,format="PNG",compress_level=2)
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    def move_to_well(self):
        if self.is_in_loading_position:
            return json.dumps({"status":"error","message":"cannot move to well while in loading position"})
        
        # get well_name from request
        json_data=None
        try:
            json_data=request.get_json()
        except Exception as e:
            pass

        if json_data is None:
            return json.dumps({"status": "error", "message": "no json data"})
        
        well_name=json_data['well_name']

        x_mm=plate.get_well_offset_x(well_name)+plate.Well_size_x_mm/2
        y_mm=plate.get_well_offset_y(well_name)+plate.Well_size_y_mm/2

        self.state=CoreState.Moving
        self.mc.send_cmd(Command.move_to_mm("x",x_mm))
        self.mc.send_cmd(Command.move_to_mm("y",y_mm))

        self.state=CoreState.Idle
        return json.dumps({"status":"success","message":"moved to well "+well_name})

    def get_current_state(self):
        last_stage_position=self.mc.get_last_position()

        img_handle=self.latest_image_handle
        latest_img_info=None
        if img_handle is not None:
            latest_img_info={
                "handle":img_handle,
                "channel":self.images[img_handle].channel_config.to_dict()
            }
        
        return json.dumps({
            "status":"success",
            "state":self.state.value,
            "stage_position":{
                "x_pos_mm":last_stage_position.x_pos_mm,
                "y_pos_mm":last_stage_position.y_pos_mm,
                "z_pos_mm":last_stage_position.z_pos_mm,
            },
            "latest_img":latest_img_info
        })

    def action_move_by(self,axis:tp.Literal["x","y","z"]):
        if self.is_in_loading_position:
            return json.dumps({"status":"error","message":"cannot move while in loading position"})
        
        json_data=None
        try:
            json_data=request.get_json()
        except Exception as e:
            pass

        if json_data is None:
            return json.dumps({"status": "error", "message": "no json data"})
        
        distance_mm=json_data['dist_mm']
        
        self.state=CoreState.Moving
        self.mc.send_cmd(Command.move_by_mm(axis,distance_mm))

        self.state=CoreState.Idle
        return json.dumps({"status": "success","moved_by_mm":distance_mm,"axis":axis})

    def action_enter_loading_position(self):
        if self.is_in_loading_position:
            return json.dumps({"status":"error","message":"already in loading position"})
        
        self.state=CoreState.Moving
        
        # home z
        self.mc.send_cmd(Command.home("z"))

        # clear clamp in y first
        self.mc.send_cmd(Command.move_to_mm("y",30))
        # then clear clamp in x
        self.mc.send_cmd(Command.move_to_mm("x",30))

        # then home y, x
        self.mc.send_cmd(Command.home("y"))
        self.mc.send_cmd(Command.home("x"))
        
        self.is_in_loading_position=True
        self.state=CoreState.LoadingPosition
        return json.dumps({"status":"success","message":"entered loading position"})

    def action_leave_loading_position(self):
        if not self.is_in_loading_position:
            return json.dumps({"status":"error","message":"not in loading position"})
        
        self.state=CoreState.Moving
        self.mc.send_cmd(Command.move_to_mm("x",30))
        self.mc.send_cmd(Command.move_to_mm("y",30))
        self.mc.send_cmd(Command.move_to_mm("z",1))
        
        self.is_in_loading_position=False
        self.state=CoreState.Idle
        return json.dumps({"status":"success","message":"left loading position"})

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
        
        config = AcquisitionConfig.from_json(json_data["config_file"])
        print("starting acquisition with config:",config)

        # TODO generate some unique acqisition id to identify this acquisition by
        # must be robust against server restarts, and async requests
        # must also cache previous run results, to allow fetching information from past acquisitions
        acquisition_id="a65914"

        somemap[acquisition_id]={
            "acquisition_id":acquisition_id,
            "acquisition_config":config,
            "num_images_acquired":0,
            "num_images_total":1000,
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
        
        # TODO get status of acquisition with id "acquisition_id"
        # just mock something for now

        if acquisition_id not in somemap:
            return json.dumps({"status":"error","message":"acquisition_id is invalid"})

        acq_res=somemap[acquisition_id]
        acq_res["num_images_acquired"]+=1
        if acq_res["num_images_acquired"]>=acq_res["num_images_total"]:
            acq_res["num_images_acquired"]=acq_res["num_images_total"]

        ret={
            # request success
            "status":"success",

            "acquisition_id":acquisition_id,
            "acquisition_status":"running",
            "acquisition_progress":{
                # measureable progress
                "current_num_images":acq_res["num_images_acquired"],
                "time_since_start_s":50,
                "start_time_iso":"2021-01-01T12:00:00",
                "current_storage_usage_GB":4.3,

                # estimated completion time information
                # estimation may be more complex than linear interpolation, hence done on server side
                "estimated_total_time_s":100,

                # last image that was acquired
                "last_image":{
                    "well":"D05",
                    "site":{
                        "x":2,
                        "y":3,
                        "z":1,
                    },
                    "timepoint":1,
                    "channel":"fluo405",
                    "full_path":"/server/data/acquisition/a65914/D05_2_3_1_1_fluo405.tiff"
                },
            },

            # some meta information about the acquisition, derived from configuration file
            "acquisition_meta_information":{
                "total_num_images":acq_res["num_images_total"],
                "max_storage_size_images_GB":10,
            },

            "acquisition_config":{
                # full acquisition config copy here
            },

            "message":f"Acquisition is running, {100*acq_res['num_images_acquired']/acq_res['num_images_total']}% complete"
        }
        if acq_res["num_images_acquired"]==acq_res["num_images_total"]:
            ret["acquisition_status"]="finished"
            ret["message"]="Acquisition is finished"

        return json.dumps(ret)
    
    def close(self):
        for mc in self.microcontrollers:
            mc.close()

        for cam in self.cams:
            cam.close()



if __name__ == "__main__":
    core=Core()
    
    try:
        # running in debug mode doesnt work because then I am unable to properly handle the close event
        app.run(debug=False, port=5002)
    except Exception:
        pass

    core.close()
