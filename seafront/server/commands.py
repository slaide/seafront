import typing as tp
import asyncio
from enum import Enum
import time

from pydantic import BaseModel, Field, ConfigDict
import numpy as np

import seaconfig as sc
from seaconfig.acquisition import AcquisitionConfig
from ..config.basics import ConfigItem, GlobalConfigHandler
from ..hardware.camera import Camera, gxiapi
from ..hardware import microcontroller as mc

from fastapi import HTTPException

# wrap error cases in http response models

class InternalErrorModel(BaseModel):
    detail: str

def error_internal(detail:str)->tp.NoReturn:
    """raise an HTTPException with specified detail """
    raise HTTPException(status_code=500,detail=detail)

_LAST_TIMESTAMP=time.time()
def print_time(msg:str):
    """essentially a logging function"""
    global _LAST_TIMESTAMP

    if 0:
        new_time=time.time()
        time_since_last_timestamp=new_time-_LAST_TIMESTAMP
        _LAST_TIMESTAMP=new_time

        # only print if the time taken is above a threshold
        TIME_THRESHOLD=1e-3 # 1ms
        if time_since_last_timestamp>TIME_THRESHOLD:
            print(f"{(time_since_last_timestamp*1e3):21.1f}ms : {msg}")

def wellIsForbidden(well_name:str,plate_type:sc.Wellplate)->bool:
    """check if a well if forbidden, as indicated by global config"""
    g_config=GlobalConfigHandler.get_dict()
    forbidden_wells_entry=g_config["forbidden_wells"]
    if forbidden_wells_entry is None:
        error_internal(detail="forbidden_wells entry not found in global config")
    
    forbidden_wells_str=forbidden_wells_entry.value
    if not isinstance(forbidden_wells_str,str):
        error_internal(detail="forbidden_wells entry is not a string")

    for s in forbidden_wells_str.split(";"):
        num_wells,well_names=s.split(":")
        if plate_type.Num_total_wells==int(num_wells):
            for well in well_names.split(","):
                if well==well_name:
                    return True
                
    return False

def _process_image(img:np.ndarray)->np.ndarray:
    """
        center crop main camera image to target size
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

    assert target_height>0, f"target height must be positive"
    assert target_width>0, f"target width must be positive"

    assert target_height<=current_height, f"target height {target_height} is larger than max {current_height}"
    assert target_width<=current_width, f"target width {target_width} is larger than max {current_width}"

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

# command base class (unused.. should not be unused, but unsure how to actually utilize this, because the return type varies)

class CoreCommand(BaseModel):
    """ virtual base class for core commands """

    async def run(self,core:"Core")->BaseModel:
        raise NotImplementedError("run not implemented for basic CoreCommand")
    
CoreCommandDerived=tp.TypeVar("CoreCommandDerived",bound=CoreCommand)

# input and output parameter models (for server i/o and internal use)

class CoreState(str,Enum):
    Idle="idle"
    ChannelSnap="channel_snap"
    ChannelStream="channel_stream"
    LoadingPosition="loading_position"
    Moving="moving"

class CoreStreamHandler(BaseModel):
    """
        class used to control a streaming microscope

        i.e. image acquisition is running in streaming mode
    """

    core:"Core"
    should_stop:bool=False
    channel_config:sc.AcquisitionChannelConfig

    # allow field of non-basemodel Core
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __call__(self,img:gxiapi.RawImage):
        if self.should_stop:
            return True
        
        match img.get_status():
            case gxiapi.GxFrameStatusList.INCOMPLETE:
                error_internal(detail="incomplete frame")
            case gxiapi.GxFrameStatusList.SUCCESS:
                pass
        
        img_np=img.get_numpy_array()
        assert img_np is not None
        img_np=img_np.copy()

        img_np=_process_image(img_np)
        
        self.core._store_new_image(img_np,self.channel_config)

        return self.should_stop

class StagePosition(BaseModel):
    x_pos_mm:float
    y_pos_mm:float
    z_pos_mm:float

    @staticmethod
    def zero()->"StagePosition":
        return StagePosition(x_pos_mm=0,y_pos_mm=0,z_pos_mm=0)

class ImageStoreInfo(BaseModel):
    handle:str
    channel:sc.AcquisitionChannelConfig
    width_px:int
    height_px:int
    timestamp:float

class ConfigFileInfo(BaseModel):
    filename:str
    timestamp:tp.Optional[str]
    comment:tp.Optional[str]
    cell_line:str
    plate_type:str

class ConfigListResponse(BaseModel):
    status:tp.Literal["success"]
    configs:tp.List[ConfigFileInfo]

class CoreCurrentState(BaseModel):
    status:tp.Literal["success"]
    state:str
    is_in_loading_position:bool
    stage_position:StagePosition
    latest_imgs:tp.Dict[str,ImageStoreInfo]
    current_acquisition_id:tp.Optional[str]

class BasicSuccessResponse(BaseModel):
    status:tp.Literal["success"]

class HistogramResponse(BaseModel):
    status:tp.Literal["success"]
    channel_name:str
    hist_values:tp.List[int]

class ConfigFetchResponse(BaseModel):
    status:tp.Literal["success"]
    file:sc.AcquisitionConfig

class LaserAutofocusCalibrationData(BaseModel):
    um_per_px:float
    x_reference:float

    calibration_position:StagePosition

class LaserAutofocusCalibrationResponse(BaseModel):
    status:tp.Literal["success"]
    calibration_data:LaserAutofocusCalibrationData

class AcquisitionCommand(str,Enum):
    CANCEL="cancel"

class AcquisitionStartResponse(BaseModel):
    status:tp.Literal["success"]
    acquisition_id:str

class WellSite(BaseModel):
    x:int
    y:int
    z:int

class LastImageInformation(BaseModel):
    well:str
    site:WellSite
    timepoint:int
    channel_name:str
    full_path:str
    handle:str

class AcquisitionProgressStatus(BaseModel):
    # measureable progress
    current_num_images:int
    time_since_start_s:float
    start_time_iso:str
    current_storage_usage_GB:float

    # estimated completion time information
    # estimation may be more complex than linear interpolation, hence done on server side
    estimated_total_time_s:tp.Optional[float]

    # last image that was acquired
    last_image:LastImageInformation

class AcquisitionMetaInformation(BaseModel):
    total_num_images:int
    max_storage_size_images_GB:float

class AcquisitionStatusOut(BaseModel):
    """acquisition thread message out"""

    status:tp.Literal["success"]

    acquisition_id:str
    acquisition_status:tp.Literal["running","cancelled","completed","crashed"]
    acquisition_progress:AcquisitionProgressStatus

    # some meta information about the acquisition, derived from configuration file
    # i.e. this is not updated during acquisition
    acquisition_meta_information:AcquisitionMetaInformation

    acquisition_config:sc.AcquisitionConfig

    message:str

class AcquisitionStatus(BaseModel):
    acquisition_id:str
    queue_in:asyncio.Queue[AcquisitionCommand]
    """queue to send messages to the thread"""
    queue_out:asyncio.Queue[AcquisitionStatusOut]
    """queue to receive messages from the thread"""
    last_status:tp.Optional[AcquisitionStatusOut]
    thread_is_running:bool

    model_config = ConfigDict(arbitrary_types_allowed=True)

class HardwareCapabilitiesResponse(BaseModel):
    wellplate_types:list[sc.Wellplate]
    main_camera_imaging_channels:list[sc.AcquisitionChannelConfig]

class LoadingPositionEnter(BaseModel):
    """
    enter loading position

    enters the stage loading position, and remains there until leave loading position command is executed.
    """

    async def run(self,core:"Core")->BasicSuccessResponse:
        if core.is_in_loading_position:
            error_internal(detail="already in loading position")
        
        core.state=CoreState.Moving
        
        # home z
        await core.mc.send_cmd(mc.Command.home("z"))

        # clear clamp in y first
        await core.mc.send_cmd(mc.Command.move_to_mm("y",30))
        # then clear clamp in x
        await core.mc.send_cmd(mc.Command.move_to_mm("x",30))

        # then home y, x
        await core.mc.send_cmd(mc.Command.home("y"))
        await core.mc.send_cmd(mc.Command.home("x"))
        
        core.is_in_loading_position=True

        core.state=CoreState.LoadingPosition

        return BasicSuccessResponse(status="success")

class LoadingPositionLeave(BaseModel):
    """
    leave loading position

    leaves the stage loading position, and moves to a default position on the plate (this position may not be inside a well, so no cells may be visibile at this time!)
    """

    async def run(self,core:"Core")->BasicSuccessResponse:
        if not core.is_in_loading_position:
            error_internal(detail="not in loading position")
        
        core.state=CoreState.Moving

        await core.mc.send_cmd(mc.Command.move_to_mm("x",30))
        await core.mc.send_cmd(mc.Command.move_to_mm("y",30))
        await core.mc.send_cmd(mc.Command.move_to_mm("z",1))
        
        core.is_in_loading_position=False

        core.state=CoreState.Idle

        return BasicSuccessResponse(status="success")
    

class ImageAcquiredResponse(BaseModel):
    status:tp.Literal["success"]
    img_handle:str

class StreamingStartedResponse(BaseModel):
    status:tp.Literal["success"]
    channel:sc.AcquisitionChannelConfig

class _ChannelAcquisitionControl(BaseModel):
    """
        control imaging in a channel

        for mode=="snap", calls _store_new_image internally
    """

    mode:tp.Literal['snap','stream_begin','stream_end']
    channel:sc.AcquisitionChannelConfig
    framerate_hz:float=5
    machine_config:list[sc.ConfigItem]=Field(default_factory=list)

    async def run(self,core:"Core")->BasicSuccessResponse|ImageAcquiredResponse|StreamingStartedResponse:
        """
        returns:
            BasicSuccessResponse on stream_end
            ImageAcquiredResponse on snap
            StreamingStartedResponse on stream_begin
        """

        cam=core.main_cam

        try:
            illum_code=mc.ILLUMINATION_CODE.from_handle(self.channel.handle)
        except Exception as e:
            error_internal(detail=f"invalid channel handle: {self.channel.handle}")
        
        if self.machine_config is not None:
            GlobalConfigHandler.override(self.machine_config)

        match self.mode:
            case "snap":
                if core.is_streaming or core.stream_handler is not None:
                    error_internal(detail="already streaming")
                
                core.state=CoreState.ChannelSnap

                print_time("before illum on")
                await core.mc.send_cmd(mc.Command.illumination_begin(illum_code,self.channel.illum_perc))
                print_time("before acq")
                img=cam.acquire_with_config(self.channel)
                print_time("after acq")
                await core.mc.send_cmd(mc.Command.illumination_end(illum_code))
                print_time("after illum off")
                if img is None:
                    core.state=CoreState.Idle
                    error_internal(detail="failed to acquire image")                

                img=_process_image(img)
                print_time("after img proc")
                img_handle=core._store_new_image(img,self.channel)
                print_time("after img store")

                core.state=CoreState.Idle

                return ImageAcquiredResponse(status="success",img_handle=img_handle)
            
            case "stream_begin":
                if core.is_streaming or core.stream_handler is not None:
                    error_internal(detail="already streaming")

                if self.framerate_hz is None:
                    error_internal(detail="no framerate_hz")

                core.state=CoreState.ChannelStream
                core.is_streaming=True

                core.stream_handler=CoreStreamHandler(core=core,channel_config=self.channel)

                await core.mc.send_cmd(mc.Command.illumination_begin(illum_code,self.channel.illum_perc))

                cam.acquire_with_config(
                    self.channel,
                    mode="until_stop",
                    callback=core.stream_handler,
                    target_framerate_hz=self.framerate_hz
                )

                return StreamingStartedResponse(status="success",channel=self.channel)
            
            case "stream_end":
                if (not core.is_streaming or core.stream_handler is None) or (not cam.acquisition_ongoing):
                    error_internal(detail="not currently streaming")
                
                core.stream_handler.should_stop=True
                await core.mc.send_cmd(mc.Command.illumination_end(illum_code))
                # cancel ongoing acquisition
                from seafront.hardware.camera import AcquisitionMode
                cam.acquisition_ongoing=False
                cam._set_acquisition_mode(AcquisitionMode.ON_TRIGGER)

                core.stream_handler=None
                core.is_streaming=False

                core.state=CoreState.Idle
                return BasicSuccessResponse(status="success")
            
            case _o:
                error_internal(detail=f"invalid mode {_o}")

class ChannelSnapshot(_ChannelAcquisitionControl):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,mode="snap",**kwargs)

class MoveByResult(BaseModel):
    status:tp.Literal["success"]
    moved_by_mm:float
    axis:str

class MoveBy(BaseModel):
    """
    move stage by some distance

    moves stage by some distance, relative to its current location. may introduce minor errors if used frequently without intermediate absolute moveto
    """

    axis:tp.Literal["x","y","z"]
    distance_mm:float

    async def run(self,core:"Core")->MoveByResult:
        if core.is_in_loading_position:
            error_internal(detail="now allowed while in loading position")

        core.state=CoreState.Moving

        await core.mc.send_cmd(mc.Command.move_by_mm(self.axis,self.distance_mm))

        core.state=CoreState.Idle

        return MoveByResult(status= "success",moved_by_mm=self.distance_mm,axis=self.axis)

class MoveTo(BaseModel):
    """
    move to target coordinates

    any of the arguments may be None, in which case the corresponding axis is not moved

    these coordinates are internally adjusted to take the calibration into account
    """

    x_mm:tp.Optional[float]
    y_mm:tp.Optional[float]
    z_mm:tp.Optional[float]=None

    async def run(self,core:"Core")->BasicSuccessResponse:
        if core.is_in_loading_position:
            error_internal(detail="now allowed while in loading position")

        if self.x_mm is not None and self.x_mm<0:
            error_internal(detail=f"x coordinate out of bounds {self.x_mm = }")
        if self.y_mm is not None and self.y_mm<0:
            error_internal(detail=f"y coordinate out of bounds {self.y_mm = }")
        if self.z_mm is not None and self.z_mm<0:
            error_internal(detail=f"z coordinate out of bounds {self.z_mm = }")

        prev_state=core.state
        core.state=CoreState.Moving

        approach_x_before_y=True

        if self.x_mm is not None and self.y_mm is not None:
            current_state=await core.get_current_state()

            # plate center is (very) rougly at x=61mm, y=40mm
            # we have: start position, target position, and two possible edges to move across

            center=61.0,40.0
            start=current_state.stage_position.x_pos_mm,current_state.stage_position.y_pos_mm
            target=self.x_mm,self.y_mm

            # if edge1 is closer to center, then approach_x_before_y=True, else approach_x_before_y=False
            edge1=self.x_mm,current_state.stage_position.y_pos_mm
            edge2=current_state.stage_position.x_pos_mm,self.y_mm

            def dist(p1:tp.Tuple[float,float],p2:tp.Tuple[float,float])->float:
                return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5

            approach_x_before_y=dist(edge1,center)<dist(edge2,center)

            # we want to choose the edge that is closest to the center, because this avoid moving through the forbidden plate corners

        if approach_x_before_y:
            if self.x_mm is not None:
                x_mm=core.pos_x_real_to_measured(self.x_mm)
                if x_mm<0:
                    error_internal(detail=f"calibrated x coordinate out of bounds {x_mm = }")
                await core.mc.send_cmd(mc.Command.move_to_mm("x",x_mm))

            if self.y_mm is not None:
                y_mm=core.pos_y_real_to_measured(self.y_mm)
                if y_mm<0:
                    error_internal(detail=f"calibrated y coordinate out of bounds {y_mm = }")
                await core.mc.send_cmd(mc.Command.move_to_mm("y",y_mm))
        else:
            if self.y_mm is not None:
                y_mm=core.pos_y_real_to_measured(self.y_mm)
                if y_mm<0:
                    error_internal(detail=f"calibrated y coordinate out of bounds {y_mm = }")
                await core.mc.send_cmd(mc.Command.move_to_mm("y",y_mm))

            if self.x_mm is not None:
                x_mm=core.pos_x_real_to_measured(self.x_mm)
                if x_mm<0:
                    error_internal(detail=f"calibrated x coordinate out of bounds {x_mm = }")
                await core.mc.send_cmd(mc.Command.move_to_mm("x",x_mm))

        if self.z_mm is not None:
            z_mm=core.pos_z_real_to_measured(self.z_mm)
            if z_mm<0:
                error_internal(detail=f"calibrated z coordinate out of bounds {z_mm = }")
            await core.mc.send_cmd(mc.Command.move_to_mm("z",z_mm))

        core.state=prev_state

        return BasicSuccessResponse(status="success")

class MoveToWell(BaseModel):
    """
    move to well

    moves to center of target well (centers field of view). requires specification of plate type because measurements of plate types vary by manufacturer.
    """

    plate_type:str
    well_name:str

    async def run(self,core:"Core")->BasicSuccessResponse:
        if core.is_in_loading_position:
            error_internal(detail="now allowed while in loading position")

        plates=[p for p in sc.Plates if p.Model_id==self.plate_type]
        if len(plates)==0:
            error_internal(detail="plate type not found")

        assert len(plates)==1, f"found multiple plates with id {self.plate_type}"

        plate=plates[0]

        if wellIsForbidden(self.well_name,plate):
            error_internal(detail="well is forbidden")

        x_mm=plate.get_well_offset_x(self.well_name) + plate.Well_size_x_mm/2
        y_mm=plate.get_well_offset_y(self.well_name) + plate.Well_size_y_mm/2

        res=await MoveTo(x_mm=x_mm,y_mm=y_mm).run(core)
        if res.status!="success":
            return res

        return BasicSuccessResponse(status="success")

class AutofocusMeasureDisplacementResult(BaseModel):
    status:tp.Literal["success"]
    displacement_um:float

class AutofocusMeasureDisplacement(BaseModel):
    """
        measure current displacement from reference position
    """

    config_file:sc.AcquisitionConfig
    override_num_images:tp.Optional[int]=None
        
    async def run(self,core:"Core")->AutofocusMeasureDisplacementResult:
        if self.config_file.machine_config is not None:
            GlobalConfigHandler.override(self.config_file.machine_config)

        g_config=GlobalConfigHandler.get_dict()

        conf_af_exp_ms_item=g_config["laser_autofocus_exposure_time_ms"]
        assert conf_af_exp_ms_item is not None
        conf_af_exp_ms=conf_af_exp_ms_item.floatvalue

        conf_af_exp_ag_item=g_config["laser_autofocus_analog_gain"]
        assert conf_af_exp_ag_item is not None
        conf_af_exp_ag=conf_af_exp_ag_item.floatvalue

        conf_af_pix_fmt_item=g_config["laser_autofocus_pixel_format"]
        assert conf_af_pix_fmt_item is not None
        # todo also use conf_af_pix_fmt

        conf_af_if_calibrated=g_config["laser_autofocus_is_calibrated"]
        conf_af_calib_x=g_config["laser_autofocus_calibration_x"]
        conf_af_calib_umpx=g_config["laser_autofocus_calibration_umpx"]
        if conf_af_if_calibrated is None or conf_af_calib_x is None or conf_af_calib_umpx is None or not conf_af_if_calibrated.boolvalue:
            error_internal(detail="laser autofocus not calibrated")

        # get laser spot location
        # sometimes one of the two expected dots cannot be found in _get_laser_spot_centroid because the plate is so far off the focus plane though, catch that case
        try:
            calib_params=LaserAutofocusCalibrationData(um_per_px=conf_af_calib_umpx.floatvalue,x_reference=conf_af_calib_x.floatvalue,calibration_position=StagePosition.zero())
            displacement_um=0

            num_images=3 or self.override_num_images
            for i in range(num_images):
                latest_esimated_z_offset_mm=await core._approximate_laser_af_z_offset_mm(calib_params)
                displacement_um+=latest_esimated_z_offset_mm*1e3/num_images

        except Exception as e:
            error_internal(detail="failed to measure displacement (got no signal): {str(e)}")

        return AutofocusMeasureDisplacementResult(status="success",displacement_um=displacement_um)

class AutofocusSnapResult(BaseModel):
    status:tp.Literal["success"]
    img_handle:str
    width_px:int
    height_px:int

class AutofocusSnap(BaseModel):
    """
        snap a laser autofocus image
    """

    exposure_time_ms:float=5
    analog_gain:float=10
    turn_laser_on:bool=True
    turn_laser_off:bool=True

    async def run(self,core:"Core")->AutofocusSnapResult:
        if self.turn_laser_on:
            await core.mc.send_cmd(mc.Command.af_laser_illum_begin())
        
        channel_config=sc.AcquisitionChannelConfig(
            name="laser autofocus acquisition", # unused
            handle="__invalid_laser_autofocus_channel__", # unused
            illum_perc=100, # unused
            exposure_time_ms=self.exposure_time_ms,
            analog_gain=self.analog_gain,
            z_offset_um=0, # unused
            num_z_planes=0, # unused
            delta_z_um=0, # unused
        )

        img=core.focus_cam.acquire_with_config(channel_config)
        if img is None:
            core.state=CoreState.Idle
            error_internal(detail="failed to acquire image")
        
        img_handle=core._store_new_laseraf_image(img,channel_config)

        if self.turn_laser_off:
            await core.mc.send_cmd(mc.Command.af_laser_illum_end())

        return AutofocusSnapResult(
            status="success",
            img_handle=img_handle,
            width_px=img.shape[1],
            height_px=img.shape[0],
        )

class AutofocusLaserWarmup(BaseModel):
    """
        warm up the laser for autofocus

        sometimes the laser needs to stay on for a little bit before it can be used (the short on-time of ca. 5ms is
        sometimes not enough to turn the laser on properly without a recent warmup)
    """

    warmup_time_s:float=0.5

    async def run(self,core:"Core")->BasicSuccessResponse:
        await core.mc.send_cmd(mc.Command.af_laser_illum_begin())

        # wait for the laser to warm up
        await asyncio.sleep(self.warmup_time_s)

        await core.mc.send_cmd(mc.Command.af_laser_illum_end())

        return BasicSuccessResponse(status="success")

class IlluminationEndAll(BaseModel):
    """
        turn off all illumination sources

        rather, send signal to do so. this function does not check if any are on before sending the signals.

        this function is NOT for regular operation!
        it does not verify that the microscope is not in any acquisition state
    """

    async def run(self,core:"Core")->BasicSuccessResponse:
        # make sure all illumination is off
        for illum_src in [
            mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_405NM,
            mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_488NM,
            mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_561NM,
            mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_638NM,
            mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_730NM,

            mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_LED_ARRAY_FULL, # this will turn off the led matrix
        ]:
            await core.mc.send_cmd(mc.Command.illumination_end(illum_src))

        return BasicSuccessResponse(status="success")

class AutofocusApproachTargetDisplacement(BaseModel):
    """
        move to target offset

        measure current offset from reference position, then moves to target offset (may perform more than one physical move to reach target position)
    """

    target_offset_um:float
    config_file:sc.AcquisitionConfig
    max_num_reps:int=3
    pre_approach_refz:bool=True

    async def estimate_offset_mm(self,core:"Core"):
        res=await AutofocusMeasureDisplacement(config_file=self.config_file).run(core)

        current_displacement_um=res.displacement_um
        assert current_displacement_um is not None

        return (self.target_offset_um-current_displacement_um)*1e-3

    async def current_z_mm(self,core:"Core"):
        current_state=await core.get_current_state()
        return current_state.stage_position.z_pos_mm

    async def run(self,core:"Core")->BasicSuccessResponse:
        if core.is_in_loading_position:
            error_internal(detail="now allowed while in loading position")

        if core.state!=CoreState.Idle:
            error_internal(detail="cannot move while in non-idle state")

        g_config=GlobalConfigHandler.get_dict()

        # get autofocus calibration data
        conf_af_calib_x=g_config["laser_autofocus_calibration_x"].floatvalue
        conf_af_calib_umpx=g_config["laser_autofocus_calibration_umpx"].floatvalue
        autofocus_calib=LaserAutofocusCalibrationData(um_per_px=conf_af_calib_umpx,x_reference=conf_af_calib_x,calibration_position=StagePosition.zero())

        # we are looking for a z coordinate where the measured dot_x is equal to this target_x.
        # we can estimate the current z offset based on the currently measured dot_x.
        # then we loop:
        #   we move by the estimated offset to reduce the difference between target_x and dot_x.
        #   then we check if we are at target_x.
        #     if we have not reached it, we move further in that direction, based on another estimate.
        #     if have overshot (moved past) it, we move back by some estimate.
        #     terminate when dot_x is within a margin of target_x.

        target_x=conf_af_calib_x

        current_state=await core.get_current_state()
        if current_state.status!="success":
            return current_state
        current_z=current_state.stage_position.z_pos_mm
        initial_z=current_z

        if self.pre_approach_refz:
            gconfig_refzmm_item=g_config["laser_autofocus_calibration_refzmm"]
            if gconfig_refzmm_item is None:
                error_internal(detail="laser_autofocus_calibration_refzmm is not available when AutofocusApproachTargetDisplacement had pre_approach_refz set")

            res=await MoveTo(x_mm=None,y_mm=None,z_mm=gconfig_refzmm_item.floatvalue).run(core)
            if res.status!="success":
                error_internal(detail=f"failed to approach ref z: {res.dict()}")

        old_state=core.state
        core.state=CoreState.Moving

        try:
            # TODO : make this better, utilizing these value pairs
            # (
            #    physz = (await core.get_current_state()).stage_position.z_pos_mm,
            #    dotx = await core._approximate_laser_af_z_offset_mm(autofocus_calib,_leftmostxinsteadofestimatedz=True)
            # )

            last_distance_estimate_mm=await self.estimate_offset_mm(core)
            MAX_MOVEMENT_RANGE_MM=0.3 # should be derived from the calibration data, but this value works fine in practice
            if np.abs(last_distance_estimate_mm)>MAX_MOVEMENT_RANGE_MM:
                error_internal(detail="measured autofocus focal plane offset too large")

            for rep_i in range(self.max_num_reps):
                distance_estimate_mm=await self.estimate_offset_mm(core)

                # stop if the new estimate indicates a larger distance to the focal plane than the previous estimate
                # (since this indicates that we have moved away from the focal plane, which should not happen)
                if rep_i>0 and np.abs(last_distance_estimate_mm)<np.abs(distance_estimate_mm):
                    break

                last_distance_estimate_mm=distance_estimate_mm

                await core.mc.send_cmd(mc.Command.move_by_mm("z",distance_estimate_mm))

        except:
            # if any interaction failed, attempt to reset z position to known somewhat-good position
            await core.mc.send_cmd(mc.Command.move_to_mm("z",initial_z))
        finally:
            core.state=old_state

        return BasicSuccessResponse(status="success")

class SendImageHistogram(BaseModel):
    """
    calculate histogram for an image

    calculates a 256 bucket histogram, regardless of pixel depth.

    may internally downsample the image to reduce calculation time.
    """

    img_handle:str

    async def run(self,core:"Core")->HistogramResponse:
        res=await core.get_image_histogram(self.img_handle)

        return res
