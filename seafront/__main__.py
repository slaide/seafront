#!/usr/bin/env python3

# system deps

import json, time, os, io
import typing as tp
from typing import Callable, Optional
from types import MethodType
from enum import Enum

import gc
import datetime as dt
import pathlib as path
import re
import cv2
import random
import traceback
from functools import wraps
import inspect
import asyncio

# math and image dependencies

import numpy as np
from PIL import Image
import tifffile
import scipy
from scipy import stats

# for robust type safety at runtime

from pydantic import create_model, BaseModel, Field, ConfigDict

# for debugging

import matplotlib.pyplot as plt
_DEBUG_P2JS=True

# http server dependencies

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse, Response
from fastapi.exceptions import RequestValidationError

import uvicorn

# microscope dependencies

from .server.commands import *
from .server.commands import _ChannelAcquisitionControl

import seaconfig as sc
from seaconfig.acquisition import AcquisitionConfig
from .config.basics import ConfigItem, GlobalConfigHandler
from .hardware.camera import Camera, gxiapi
from .hardware import microcontroller as mc

# utility functions

def linear_regression(x:list[float]|np.ndarray,y:list[float]|np.ndarray)->tuple[float,float]:
    "returns (slope,intercept)"
    slope, intercept, _,_,_ = stats.linregress(x, y)
    return slope,intercept #type:ignore

# Set the working directory to the script's directory as reference for static file paths

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# file name utility code

# precompile regex for performance
name_validity_regex=re.compile(r"^[a-zA-Z0-9_\-]+$")

def generate_random_number_string(num_digits:int=9)->str:
    "used to generate new image handles"
    max_val:float=10**num_digits
    max_val-=2
    return f"{(int(random.random()*max_val)+1):0{num_digits}d}"

app = FastAPI(debug=True)
app.mount("/src", StaticFiles(directory="src"), name="static")

@app.get("/")
async def index():
    return FileResponse("index.html")

@app.get("/css/{path:path}")
async def send_css(path: str):
    return FileResponse(f"css/{path}")

@app.get("/src/{path:path}")
async def send_js(path: str):
    return FileResponse(f"src/{path}")

@app.api_route("/api/get_features/hardware_capabilities", methods=["GET"], summary="Get available imaging channels and plate types.")
def get_hardware_capabilities()->HardwareCapabilitiesResponse:
    """
    get a list of all the hardware capabilities of the system

    these are the high-level configuration options that the user can select from
    """

    return HardwareCapabilitiesResponse(
        wellplate_types=sc.Plates,

        main_camera_imaging_channels=[c for c in [
            sc.AcquisitionChannelConfig(
                name="Fluo 405 nm Ex",
                handle="fluo405",
                illum_perc=100,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0,
                num_z_planes=1,
                delta_z_um=1,
            ),
            sc.AcquisitionChannelConfig(
                name="Fluo 488 nm Ex",
                handle="fluo488",
                illum_perc=100,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0,
                num_z_planes=1,
                delta_z_um=1,
            ),
            sc.AcquisitionChannelConfig(
                name="Fluo 561 nm Ex",
                handle="fluo561",
                illum_perc=100,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0,
                num_z_planes=1,
                delta_z_um=1,
            ),
            sc.AcquisitionChannelConfig(
                name="Fluo 638 nm Ex",
                handle="fluo638",
                illum_perc=100,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0,
                num_z_planes=1,
                delta_z_um=1,
            ),
            sc.AcquisitionChannelConfig(
                name="Fluo 730 nm Ex",
                handle="fluo730",
                illum_perc=100,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0,
                num_z_planes=1,
                delta_z_um=1,
            ),
            sc.AcquisitionChannelConfig(
                name="BF LED Full",
                handle="bfledfull",
                illum_perc=20,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0,
                num_z_planes=1,
                delta_z_um=1,
            ),
            sc.AcquisitionChannelConfig(
                name="BF LED Right Half",
                handle="bfledright",
                illum_perc=20,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0,
                num_z_planes=1,
                delta_z_um=1,
            ),
            sc.AcquisitionChannelConfig(
                name="BF LED Left Half",
                handle="bfledleft",
                illum_perc=20,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0,
                num_z_planes=1,
                delta_z_um=1,
            ),
        ]]
    )

from .config.basics import GlobalConfigHandler
GlobalConfigHandler.reset()

@app.api_route("/api/get_features/machine_defaults", methods=["GET"], summary="Get default low-level machine parameters.")
def get_machine_defaults()->list[ConfigItem]:
    """
    get a list of all the low level machine settings

    these settings may be changed on the client side, for individual acquisitions
    (though clearly, this is highly advanced stuff, and may cause irreperable hardware damage!)
    """

    return GlobalConfigHandler.get()

class CoreLock(BaseModel):
    """
        basic utility to generate a token that can be used to access mutating core functions (e.g. actions)
    """

    _current_key:tp.Optional[str]=None
    _key_gen_time=None
    """ timestamp when key was generated """
    _last_key_use=None
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

class ImageStoreEntry(BaseModel):
    """ utility class to store camera images with some metadata """

    img: np.ndarray = Field(..., repr=False)
    timestamp: float
    channel_config: sc.AcquisitionChannelConfig
    bit_depth: int

    model_config = ConfigDict(arbitrary_types_allowed=True)

class MC_getLastPosition(BaseModel):
    """ command class to retrieve core.mc.get_last_position """
    async def run(self,core:"Core")->mc.Position:
        return await core.mc.get_last_position()

class Core_getMainCam(BaseModel):
    async def run(self,core:"Core")->Camera:
        return core.main_cam

class Core_getImageByHandle(BaseModel):
    img_handle:str

    async def run(self,core:"Core")->tp.Union[ImageStoreEntry,None]:
        return core._get_imageinfo_by_handle(self.img_handle)

# store img as .tiff file
async def store_image(
    latest_channel_image:ImageStoreEntry,
    channel:sc.AcquisitionChannelConfig,
    core_main_cam:Camera,
    image_storage_path:str,
    img_compression_algorithm:str,
    PIXEL_SIZE_UM:float,
    light_source_type:int
):
    # takes 70-250ms
    tifffile.imwrite(
        image_storage_path,
        latest_channel_image.img,

        compression=img_compression_algorithm,
        compressionargs={},# for zlib: {'level': 8},
        maxworkers=1,#3,

        # this metadata is just embedded as a comment in the file
        metadata={
            # non-standard tag, because the standard bitsperpixel has a real meaning in the image image data, but
            # values that are not multiple of 8 cannot be used with compression
            "BitsPerPixel":latest_channel_image.bit_depth, # type:ignore
            "BitPaddingInfo":"lower bits are padding with 0s",

            "LightSourceName":channel.name,

            "ExposureTimeMS":f"{channel.exposure_time_ms:.2f}",
            "AnalogGainDB":f"{channel.analog_gain:.2f}",
        },

        photometric="minisblack", # zero means black
        resolutionunit=3, # 3 = cm
        resolution=(1/(PIXEL_SIZE_UM*1e-6),1/(PIXEL_SIZE_UM*1e-6)),
        extratags=[
            # tuples as accepted at https://github.com/blink1073/tifffile/blob/master/tifffile/tifffile.py#L856
            # tags may be specified as string interpretable by https://github.com/blink1073/tifffile/blob/master/tifffile/tifffile.py#L5000
            ("Make", 's', 0, core_main_cam.vendor_name, True),
            ("Model", 's', 0, core_main_cam.model_name, True),

            # apparently these are not widely supported
            ("ExposureTime", 'q', 1, int(channel.exposure_time_ms*1e3), True),
            ("LightSource", 'h', 1, light_source_type, True),
        ]
    )

    print(f"stored image to {image_storage_path}")

class ProtocolGenerator(BaseModel):
    """
    params:
        img_compression_algorithm: tp.Literal["LZW","zlib"] - lossless compression algorithms only
    """

    config_file:sc.AcquisitionConfig
    handle_q_in:tp.Callable[[],None]
    plate:sc.Wellplate
    acquisition_status:AcquisitionStatus
    acquisition_id:str
    ongoing_image_store_tasks:list

    img_compression_algorithm:tp.Literal["LZW","zlib"]="LZW"

    # the values below are initialized during post init hook
    num_wells:int=-1
    num_sites:int=-1
    num_channels:int=-1
    num_channel_z_combinations:int=-1
    num_images_total:int=-1
    site_topleft_x_mm:float=-1
    site_topleft_y_mm:float=-1
    image_size_bytes:int=-1
    max_storage_size_images_GB:float=-1
    project_output_path:path.Path=Field(default_factory=path.Path)

    @property
    def well_sites(self):
        "selected sites in the grid mask"
        return [s for s in self.config_file.grid.mask if s.selected]
    @property
    def channels(self):
        "selected channels"
        return [c for c in self.config_file.channels if c.enabled]

    # pydantics version of dataclass.__post_init__
    def model_post_init(self,__context):
        self.num_wells=len(self.config_file.plate_wells)
        self.num_sites=len(self.well_sites)
        self.num_channels=len([c for c in self.config_file.channels if c.enabled])
        self.num_channel_z_combinations=sum((c.num_z_planes for c in self.config_file.channels))
        self.num_images_total=self.num_wells*self.num_sites*self.num_channel_z_combinations

        # the grid is centered around the center of the well
        self.site_topleft_x_mm=self.plate.Well_size_x_mm / 2 - ((self.config_file.grid.num_x-1) * self.config_file.grid.delta_x_mm) / 2
        "offset of top left site from top left corner of the well, in x, in mm"
        self.site_topleft_y_mm=self.plate.Well_size_y_mm / 2 - ((self.config_file.grid.num_y-1) * self.config_file.grid.delta_y_mm) / 2
        "offset of top left site from top left corner of the well, in y, in mm"

        g_config=GlobalConfigHandler.get_dict()

        # calculate meta information about acquisition
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
                error_internal(detail=f"unexpected main camera pixel format '{_unexpected}' in {main_cam_pix_format}")

        self.image_size_bytes=target_width*target_height*bytes_per_pixel
        self.max_storage_size_images_GB=self.num_images_total*self.image_size_bytes/1024**3

        base_storage_path_item=g_config["base_image_output_dir"]
        assert base_storage_path_item is not None
        assert type(base_storage_path_item.value) is str
        base_storage_path=path.Path(base_storage_path_item.value)
        assert base_storage_path.exists(), f"{base_storage_path = } does not exist"

        project_name_is_acceptable=len(self.config_file.project_name) and name_validity_regex.match(self.config_file.project_name)
        if not project_name_is_acceptable:
            error_internal(detail="project name is not acceptable: 1) must not be empty and 2) only contain alphanumeric characters, underscores, dashes")
        
        plate_name_is_acceptable=len(self.config_file.plate_name)>0 and name_validity_regex.match(self.config_file.plate_name)
        if not plate_name_is_acceptable:
            error_internal(detail="plate name is not acceptable: 1) must not be empty and 2) only contain alphanumeric characters, underscores, dashes")

        self.project_output_path=base_storage_path/self.config_file.project_name/self.config_file.plate_name/self.acquisition_id
        self.project_output_path.mkdir(parents=True)

        # write config file to output directory
        with (self.project_output_path/"config.json").open("w") as file:
            file.write(self.config_file.json())

    def generate(self)->tp.Generator[
        # yielded types: None means done, str is returned on first iter
        tp.Union[ None, str, MoveTo, AutofocusMeasureDisplacement, AutofocusApproachTargetDisplacement, _ChannelAcquisitionControl, MC_getLastPosition, Core_getMainCam, Core_getImageByHandle],
        # received types (at runtime must match return type of <yielded type>.run()."resulttype")
        tp.Union[ None, AutofocusMeasureDisplacementResult, BasicSuccessResponse, ImageAcquiredResponse, StreamingStartedResponse, mc.Position, Camera, ImageStoreEntry]
        # generator return value
        ,None
    ]:
        # print(f"acquiring {num_wells} wells, {num_sites} sites, {num_channels} channels, i.e. {num_images_total} images, taking up to {max_storage_size_images_GB}GB")

        # 
        Z_STACK_COUNTER_BACKLASH_MM=40e-3 # 40um
        PIXEL_SIZE_UM=900/3000 # 3000px wide fov covers 0.9mm
        # movement below this threshold is not performed
        DISPLACEMENT_THRESHOLD_UM: float=0.5

        # counters on acquisition progress
        start_time=time.time()
        start_time_iso_str=sc.datetime2str(dt.datetime.now(dt.timezone.utc))
        last_image_information=None

        num_images_acquired=0
        storage_usage_bytes=0

        g_config=GlobalConfigHandler.get_dict()
        
        # first yield indicates that this generator is ready to produce commands
        # the value from the consumer on the first yield is None
        # i.e. this MUST be the first yield !!
        yield "ready"

        # get current z coordinate as z reference
        last_position=yield MC_getLastPosition()
        assert isinstance(last_position,mc.Position), f"{type(last_position)=}"
        reference_z_mm=last_position.z_pos_mm

        # if laser autofocus is enabled, use autofocus z reference as initial z reference
        gconfig_refzmm_item=g_config["laser_autofocus_calibration_refzmm"]
        if self.config_file.autofocus_enabled and gconfig_refzmm_item is not None:
            reference_z_mm=gconfig_refzmm_item.floatvalue

            res=yield MoveTo(x_mm=None,y_mm=None,z_mm=reference_z_mm)
            assert isinstance(res,BasicSuccessResponse), f"{type(res)=}"
            if res.status!="success": error_internal(detail=f"failed to move to laser autofocus ref z because {res.message}")

        # run acquisition:

        # used to store into metadata
        core_main_cam=yield Core_getMainCam()
        assert type(core_main_cam)==Camera

        for well in self.config_file.plate_wells:
            # these are xy sites
            print_time("before next site")
            for site_index,site in enumerate(self.well_sites):
                self.handle_q_in()

                # go to site
                site_x_mm=self.plate.get_well_offset_x(well.well_name) + self.site_topleft_x_mm + site.col * self.config_file.grid.delta_x_mm
                site_y_mm=self.plate.get_well_offset_y(well.well_name) + self.site_topleft_y_mm + site.row * self.config_file.grid.delta_y_mm

                res=yield MoveTo(x_mm=site_x_mm,y_mm=site_y_mm)
                assert isinstance(res,BasicSuccessResponse), f"{type(res)=}"

                print_time("moved to site")

                # run autofocus
                if self.config_file.autofocus_enabled:
                    for autofocus_attempt_num in range(3):
                        current_displacement=yield AutofocusMeasureDisplacement(config_file=self.config_file)
                        assert isinstance(current_displacement,AutofocusMeasureDisplacementResult), f"{type(current_displacement)=}"
                        if current_displacement.status!="success": error_internal(detail=f"failed to measure autofocus displacement at site {site} in well {well} because {current_displacement.message}")
                        
                        current_displacement_um=current_displacement.displacement_um
                        assert current_displacement_um is not None
                        # print(f"measured offset of {current_displacement_um:.2f}um on attempt {autofocus_attempt_num}")
                        if np.abs(current_displacement_um)<DISPLACEMENT_THRESHOLD_UM:
                            break
                        
                        res=yield AutofocusApproachTargetDisplacement(target_offset_um=0,config_file=self.config_file,pre_approach_refz=False)
                        assert isinstance(res,BasicSuccessResponse), f"{type(res)=}"
                        if res.status!="success": error_internal(detail=f"failed to autofocus at site {site} in well {well} because {res.message}")

                # reference for channel z offsets
                # (this position may have been adjusted by the autofocus system, but even without autofocus the current position must be the reference)
                last_position=yield MC_getLastPosition()
                assert isinstance(last_position,mc.Position), f"{type(last_position)=}"
                reference_z_mm=last_position.z_pos_mm

                print_time("af performed")
                
                # z stack may be different for each channel, hence:
                # 1. get list of (channel_z_index,channel,z_relative_to_reference), which may contain each channel more than once
                # 2. order by z, in ascending (low to high)
                # 3. move to lowest, start imaging while moving up
                # 4. move to reference z again in preparation for next site

                image_pos_z_list:list[tuple[int,float,sc.AcquisitionChannelConfig]]=[]
                for channel in self.channels:
                    channel_delta_z_mm=channel.delta_z_um*1e-3

                    # <channel reference> is <site reference>+<channel z offset>
                    base_z=reference_z_mm+channel.z_offset_um*1e-3

                    # lower z base is <channel reference> adjusted for z stack, where \
                    # n-1 z movements are performed, half of those below, half above <channel ref>
                    base_z-=((channel.num_z_planes-1)/2)*channel_delta_z_mm

                    for i in range(channel.num_z_planes):
                        i_offset_mm=i*channel_delta_z_mm
                        target_z_mm=base_z+i_offset_mm
                        image_pos_z_list.append((i,target_z_mm,channel))

                # sort in z
                image_pos_z_list=sorted(image_pos_z_list,key=lambda v:v[1])

                res=yield MoveTo(x_mm=None,y_mm=None,z_mm=image_pos_z_list[0][1]-Z_STACK_COUNTER_BACKLASH_MM)
                assert isinstance(res,BasicSuccessResponse), f"{type(res)=}"
                if res.status!="success": error_internal(detail=f"failed, because: {res.message}")
                res=yield MoveTo(x_mm=None,y_mm=None,z_mm=image_pos_z_list[0][1])
                assert isinstance(res,BasicSuccessResponse), f"{type(res)=}"
                if res.status!="success": error_internal(detail=f"failed, because: {res.message}")

                print_time("moved to init z")

                for plane_index,channel_z_mm,channel in image_pos_z_list:
                    self.handle_q_in()

                    # move to channel offset
                    last_position=yield MC_getLastPosition()
                    assert isinstance(last_position,mc.Position), f"{type(last_position)=}"
                    current_z_mm=last_position.z_pos_mm

                    distance_z_to_move_mm=channel_z_mm-current_z_mm
                    if np.abs(distance_z_to_move_mm)>DISPLACEMENT_THRESHOLD_UM:
                        res=yield MoveTo(x_mm=None,y_mm=None,z_mm=channel_z_mm)
                        assert isinstance(res,BasicSuccessResponse), f"{type(res)=}"
                        assert res.status=="success"

                    print_time("moved to channel z")

                    # snap image
                    res=yield _ChannelAcquisitionControl(mode="snap",channel=channel)
                    assert isinstance(res,ImageAcquiredResponse), f"{type(res)=}"
                    assert res.status=="success"
                    if not isinstance(res,ImageAcquiredResponse):error_internal(detail=f"failed to snap image at site {site} in well {well} (invalid result type {type(res)})")

                    print_time("snapped image")

                    img_handle=res.img_handle
                    latest_channel_image=None
                    if img_handle is not None:
                        latest_channel_image=yield Core_getImageByHandle(img_handle=img_handle)
                        assert isinstance(latest_channel_image,(ImageStoreEntry,None)), f"{type(latest_channel_image)=}" # type:ignore
                    if latest_channel_image is None:
                        error_internal(detail=f"image handle {img_handle} not found in image buffer")
                    
                    # store image
                    image_storage_path=f"{str(self.project_output_path)}/{well.well_name}_s{site_index}_x{site.col+1}_y{site.row+1}_z{plane_index+1}_{channel.handle}.tiff"
                    # add metadata to the file, based on tags from https://exiftool.org/TagNames/EXIF.html

                    if channel.handle[:3]=="BF ":
                        light_source_type=4 # Flash (seems like the best fit)
                    else:
                        light_source_type=2 # Fluorescent

                    # improve system responsiveness while compressing and writing to disk
                    self.ongoing_image_store_tasks.append(asyncio.create_task(store_image(
                        latest_channel_image=latest_channel_image, #type:ignore
                        channel=channel,
                        core_main_cam=core_main_cam,
                        image_storage_path=str(image_storage_path),
                        img_compression_algorithm=self.img_compression_algorithm,
                        PIXEL_SIZE_UM=PIXEL_SIZE_UM,
                        light_source_type=light_source_type
                    )))

                    # pop finished tasks off queue (in-place, hence pop()+push())
                    running_tasks=[]
                    num_done=0
                    while len(self.ongoing_image_store_tasks)>0:
                        task=self.ongoing_image_store_tasks.pop()
                        if task.done():
                            num_done+=1
                        else:
                            running_tasks.append(task)
                    print_time(f"image store tasks: {len(running_tasks)} running, {num_done} finished")
                    for task in running_tasks:
                        self.ongoing_image_store_tasks.append(task)

                    print_time("scheduled image store")

                    num_images_acquired+=1
                    # get storage size from filesystem because tiff compression may reduce size below size in memory
                    try:
                        file_size_on_disk=path.Path(image_storage_path).stat().st_size
                        storage_usage_bytes+=file_size_on_disk
                    except:
                        # ignore any errors here, because this is not an essential feature
                        pass

                    # status items
                    last_image_information=LastImageInformation(
                        well=well.well_name,
                        site=WellSite(
                            x=site.col,
                            y=site.row,
                            z=plane_index,
                        ),
                        timepoint=1,
                        channel_name=channel.name,
                        full_path=image_storage_path,
                        handle=img_handle,
                    )
                    time_since_start_s=time.time()-start_time
                    if num_images_acquired>0:
                        estimated_total_time_s=time_since_start_s*(self.num_images_total-num_images_acquired)/num_images_acquired
                    else:
                        estimated_total_time_s=None

                    print(f"{num_images_acquired}/{self.num_images_total} images acquired")
                    
                    self.acquisition_status.last_status=AcquisitionStatusOut(
                        status="success",

                        acquisition_id=self.acquisition_id,
                        acquisition_status="running",
                        acquisition_progress=AcquisitionProgressStatus(
                            # measureable progress
                            current_num_images=num_images_acquired,
                            time_since_start_s=time_since_start_s,
                            start_time_iso=start_time_iso_str,
                            current_storage_usage_GB=storage_usage_bytes/(1024**3),

                            # estimated completion time information
                            # estimation may be more complex than linear interpolation, hence done on server side
                            estimated_total_time_s=estimated_total_time_s,

                            # last image that was acquired
                            last_image=last_image_information,
                        ),

                        # some meta information about the acquisition, derived from configuration file
                        # i.e. this is not updated during acquisition
                        acquisition_meta_information=AcquisitionMetaInformation(
                            total_num_images=self.num_images_total,
                            max_storage_size_images_GB=self.max_storage_size_images_GB,
                        ),

                        acquisition_config=self.config_file,

                        message=f"Acquisition is {(100*num_images_acquired/self.num_images_total):.2f}% complete"
                    )

                    print(f"last message: {self.acquisition_status.last_status.message}")

        # done -> yield None and return
        print(f"last message before yield: {self.acquisition_status.last_status.message}")
        yield None

class Core:
    @property
    def main_cam(self):
        defaults=GlobalConfigHandler.get_dict()
        main_camera_model_name=defaults["main_camera_model"].value
        _main_cameras=[c for c in self.cams if c.model_name==main_camera_model_name]
        if len(_main_cameras)==0:
            error_internal(detail=f"no camera with model name {main_camera_model_name} found")
        main_camera=_main_cameras[0]
        return main_camera

    @property
    def focus_cam(self):
        defaults=GlobalConfigHandler.get_dict()
        focus_camera_model_name=defaults["laser_autofocus_camera_model"].value
        _focus_cameras=[c for c in self.cams if c.model_name==focus_camera_model_name]
        if len(_focus_cameras)==0:
            error_internal(detail=f"no camera with model name {focus_camera_model_name} found")
        focus_camera=_focus_cameras[0]
        return focus_camera

    @property
    def calibrated_stage_position(self)->tp.Tuple[float,float,float]:
        """
        return calibrated XY stage offset from GlobalConfigHandler in order (x_mm,y_mm)
        """

        off_x_mm=GlobalConfigHandler.get_dict()["calibration_offset_x_mm"].value
        assert isinstance(off_x_mm,float), f"off_x_mm is {off_x_mm} of type {type(off_x_mm)}"
        off_y_mm=GlobalConfigHandler.get_dict()["calibration_offset_y_mm"].value
        assert isinstance(off_y_mm,float), f"off_y_mm is {off_y_mm} of type {type(off_y_mm)}"
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

    async def home(self):

        # reset the MCU
        await self.mc.send_cmd(mc.Command.reset())

        # reinitialize motor drivers and DAC
        await self.mc.send_cmd(mc.Command.initialize())
        await self.mc.send_cmd(mc.Command.configure_actuators())

        print("ensuring illumination is off")
        # make sure all illumination is off
        for illum_src in [
            mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_405NM,
            mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_488NM,
            mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_561NM,
            mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_638NM,
            mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_730NM,

            mc.ILLUMINATION_CODE.ILLUMINATION_SOURCE_LED_ARRAY_FULL, # this will turn off the led matrix
        ]:
            await self.mc.send_cmd(mc.Command.illumination_end(illum_src))

        print("calibrating xy stage")

        # when starting up the microscope, the initial position is considered (0,0,0)
        # even homing considers the limits, so before homing, we need to disable the limits
        await self.mc.send_cmd(mc.Command.set_limit_mm("z",-10.0,"lower"))
        await self.mc.send_cmd(mc.Command.set_limit_mm("z",10.0,"upper"))

        # move objective out of the way
        await self.mc.send_cmd(mc.Command.home("z"))
        await self.mc.send_cmd(mc.Command.set_zero("z"))
        # set z limit to (or below) 6.7mm, because above that, the motor can get stuck
        await self.mc.send_cmd(mc.Command.set_limit_mm("z",0.0,"lower"))
        await self.mc.send_cmd(mc.Command.set_limit_mm("z",6.7,"upper"))
        # home x to set x reference
        await self.mc.send_cmd(mc.Command.home("x"))
        await self.mc.send_cmd(mc.Command.set_zero("x"))
        # clear clamp in x
        await self.mc.send_cmd(mc.Command.move_by_mm("x",30))
        # then move in position to properly apply clamp
        await self.mc.send_cmd(mc.Command.home("y"))
        await self.mc.send_cmd(mc.Command.set_zero("y"))
        # home x again to engage clamp
        await self.mc.send_cmd(mc.Command.home("x"))

        # move to an arbitrary position to disengage the clamp
        await self.mc.send_cmd(mc.Command.move_by_mm("x",30))
        await self.mc.send_cmd(mc.Command.move_by_mm("y",30))

        # and move objective up, slightly
        await self.mc.send_cmd(mc.Command.move_by_mm("z",1))

        print("done initializing microscope")
        self.homing_performed=True

    def __init__(self):
        self.lock=CoreLock()

        self.microcontrollers=mc.Microcontroller.get_all()
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
        
        defaults=GlobalConfigHandler.get_dict()
        main_camera_model_name=defaults["main_camera_model"].value
        focus_camera_model_name=defaults["laser_autofocus_camera_model"].value

        found_main_cam=False
        found_focus_cam=False
        for cam in self.cams:
            if cam.model_name==main_camera_model_name:
                device_type="main"
                found_main_cam=True
            elif cam.model_name==focus_camera_model_name:
                device_type="autofocus"
                found_focus_cam=True
            else:
                # skip other cameras
                continue
            
            cam.open(device_type=device_type)

        if not found_main_cam:
            raise RuntimeError(f"error - did not find main camera with model name {main_camera_model_name}")
        if not found_focus_cam:
            raise RuntimeError(f"error - did not find autofocus camera with model name {focus_camera_model_name}")
        
        self.mc=self.microcontrollers[0]
        self.mc.open()

        print("initializing microcontroller")
        self.homing_performed=False

        self.acquisition_map:tp.Dict[str,AcquisitionStatus]={}
        """
        map containing information on past and current acquisitions
        """

        # set up routes to member functions

        if _DEBUG_P2JS:
            def sendp2():
                return FileResponse("../../web-pjs/p2.js")

            app.add_api_route("/p2.js", sendp2, methods=["GET"])

        # store request_models for re-use (works around issues with fastapi)
        request_models=dict()

        # Utility function to wrap the shared logic by including handlers for GET requests
        def route_wrapper(path: str, target_func: Callable, methods:list[str]=["GET"], allow_while_acquisition_is_running: bool = True, summary:str|None=None, **kwargs_static):
            async def callfunc(request_data):
                # Call the target function
                if inspect.iscoroutinefunction(target_func):
                    return await target_func(**request_data)
                elif inspect.isfunction(target_func) or isinstance(target_func, MethodType):
                    return target_func(**request_data)
                elif inspect.isclass(target_func):
                    instance = target_func(**request_data)
                    if hasattr(instance, 'run'):
                        if inspect.iscoroutinefunction(instance.run):
                            return await instance.run(self)
                        else:
                            return instance.run(self)
                    else:
                        raise AttributeError(f"Provided class {target_func} has no method 'run'")
                else:
                    raise TypeError(f"Unsupported target_func type: {type(target_func)}")

            def get_return_type():
                """
                Determines the return type of the target function.
                If the target function is a class, retrieves the return type of the 'run()' method.
                """
                # Case 1: target_func is a coroutine function or a standard function/method
                if inspect.iscoroutinefunction(target_func) or inspect.isfunction(target_func) or isinstance(target_func, MethodType):
                    #print(f"returning {target_func.__name__} {inspect.signature(target_func).return_annotation}")
                    return inspect.signature(target_func).return_annotation
                
                # Case 2: target_func is a class, get return type of the 'run()' method if it exists
                elif inspect.isclass(target_func):
                    if hasattr(target_func, 'run'):
                        run_method = getattr(target_func, 'run')
                        return_type=inspect.signature(run_method).return_annotation
                        #print(f"returning {target_func.__name__}.run {return_type}")
                        return return_type

                # Default case: if none of the above matches
                print(f"{target_func=} has unknown return type (hasattr run {hasattr(target_func, 'run')})")
                return tp.Any

            return_type=get_return_type()

            @wraps(target_func)
            async def handler_logic_get(**kwargs:Optional[tp.Any])->return_type: # type: ignore
                # Perform verification
                if (not allow_while_acquisition_is_running) and self.acquisition_is_running:
                    return JSONResponse(content={"status": "error", "message": "cannot run this command while acquisition is running"}, status_code=400)

                request_data=kwargs.copy()
                request_data.update(kwargs_static)

                return await callfunc(request_data)

            # Dynamically create a Pydantic model for the POST request body if the target function has parameters
            model_fields = {}
            for key, value in inspect.signature(target_func).parameters.items():
                if key != "request" and value.annotation is not inspect._empty:
                    default_value = kwargs_static.get(key, value.default)
                    if default_value is inspect._empty:
                        model_fields[key] = (value.annotation, ...)
                    else:
                        model_fields[key] = (value.annotation, Field(default=default_value))

            RequestModel=None
            if model_fields:
                # Dynamically create the Pydantic model
                model_name = f"{target_func.__name__.capitalize()}RequestModel"
                RequestModel=request_models.get(model_name)
                if RequestModel is None:
                    RequestModel = create_model(
                        model_name,
                        **model_fields,
                        __base__=BaseModel
                    )
                    request_models[model_name]=RequestModel

                async def handler_logic_post(request_body: RequestModel): # type:ignore
                    # Perform verification
                    if (not allow_while_acquisition_is_running) and self.acquisition_is_running:
                        return JSONResponse(content={"status": "error", "message": "cannot run this command while acquisition is running"}, status_code=400)

                    request_data = kwargs_static.copy()
                    if RequestModel and request_body:
                        request_body_as_toplevel_dict=dict()
                        for key in request_body.dict(exclude_unset=True).keys():
                            request_body_as_toplevel_dict[key]=getattr(request_body,key)
                        request_data.update(request_body_as_toplevel_dict)

                    return await callfunc(request_data)
            else:
                async def handler_logic_post(): # type:ignore
                    # Perform verification
                    if (not allow_while_acquisition_is_running) and self.acquisition_is_running:
                        return JSONResponse(content={"status": "error", "message": "cannot run this command while acquisition is running"}, status_code=400)

                    request_data = kwargs_static.copy()

                    return await callfunc(request_data)

            # copy annotation and fix return type
            handler_logic_post.__doc__ = target_func.__doc__
            handler_logic_post.__annotations__['return'] = return_type

            if summary is None and target_func.__doc__ is not None:
                docstring_lines=[line.lstrip().rstrip() for line in target_func.__doc__.split("\n")]
                docstring_lines=[line for line in docstring_lines if len(line)>0]
                if len(docstring_lines)>=1:
                    summary=docstring_lines[0]

            for m in methods:
                # Extract return annotation if present
                return_annotation = inspect.signature(target_func).return_annotation

                if m == "GET":
                    app.add_api_route(
                        path,
                        handler_logic_get,
                        methods=["GET"],
                        operation_id=path[1:].replace("/",".")+".get",
                        summary=summary,
                        responses={
                            500:{"model":InternalErrorModel},
                        },
                    )
                if m == "POST":
                    app.add_api_route(
                        path,
                        handler_logic_post,
                        methods=["POST"],
                        operation_id=path[1:].replace("/",".")+".post",
                        summary=summary,
                        responses={
                            500:{"model":InternalErrorModel},
                        },
                    )

        # Register URL rules requiring machine interaction
        route_wrapper(
            "/api/get_info/current_state",
            self.get_current_state,
            methods=["GET", "POST"],
        )

        # Register URLs for immediate moves
        route_wrapper(
            "/api/action/move_by",
            MoveBy,
            methods=["POST"],
        )

        # Register URL for start_acquisition
        route_wrapper(
            "/api/acquisition/start",
            self.start_acquisition,
            methods=["POST"],
        )

        # Register URL for cancel_acquisition
        route_wrapper(
            "/api/acquisition/cancel",
            self.cancel_acquisition,
            allow_while_acquisition_is_running=True,
            methods=["POST"],
        )

        # Register URL for get_acquisition_status
        route_wrapper(
            "/api/acquisition/status",
            self.get_acquisition_status,
            allow_while_acquisition_is_running=True,
            methods=["GET", "POST"],
        )

        # Retrieve config list
        route_wrapper(
            "/api/acquisition/config_list",
            self.get_config_list,
            methods=["POST"],
        )

        # Fetch acquisition config
        route_wrapper(
            "/api/acquisition/config_fetch",
            self.config_fetch,
            methods=["POST"],
        )

        # Save/load config
        route_wrapper(
            "/api/acquisition/config_store",
            self.config_store,
            methods=["POST"],
        )

        # Move to well
        route_wrapper(
            "/api/action/move_to_well",
            MoveToWell,
            methods=["POST"],
        )

        # Send image by handle
        route_wrapper(
            "/img/get_by_handle",
            self.send_image_by_handle,
            quick_preview=False,
            allow_while_acquisition_is_running=True,
            methods=["GET"],
        )

        # Send image by handle preview
        route_wrapper(
            "/img/get_by_handle_preview",
            self.send_image_by_handle,
            quick_preview=True,
            allow_while_acquisition_is_running=True,
            methods=["GET"],
        )

        # Send image histogram
        route_wrapper(
            "/api/action/get_histogram_by_handle",
            SendImageHistogram,
            allow_while_acquisition_is_running=True,
            methods=["POST"],
        )

        # Loading position enter/leave
        self.is_in_loading_position:bool = False
        route_wrapper(
            "/api/action/enter_loading_position",
            LoadingPositionEnter,
            methods=["POST"],
        )
        route_wrapper(
            "/api/action/leave_loading_position",
            LoadingPositionLeave,
            methods=["POST"],
        )

        # Snap channel
        route_wrapper(
            "/api/action/snap_channel",
            _ChannelAcquisitionControl,
            mode="snap",
            methods=["POST"],
        )

        route_wrapper(
            "/api/action/snap_selected_channels",
            self.snap_selected_channels,
            methods=["POST"],
        )

        # Start streaming (i.e., acquire x images per sec, until stopped)
        self.is_streaming:bool = False
        self.stream_handler: tp.Optional[CoreStreamHandler] = None
        route_wrapper(
            "/api/action/stream_channel_begin",
            _ChannelAcquisitionControl,
            mode="stream_begin",
            methods=["POST"],
        )
        route_wrapper(
            "/api/action/stream_channel_end",
            _ChannelAcquisitionControl,
            mode="stream_end",
            methods=["POST"],
        )

        # Laser autofocus system
        route_wrapper(
            "/api/action/snap_reflection_autofocus",
            AutofocusSnap,
            methods=["POST"],
        )
        route_wrapper(
            "/api/action/measure_displacement",
            AutofocusMeasureDisplacement,
            methods=["POST"],
        )
        route_wrapper(
            "/api/action/laser_autofocus_move_to_target_offset",
            AutofocusApproachTargetDisplacement,
            methods=["POST"],
        )
        route_wrapper(
            "/api/action/laser_af_calibrate",
            self.laser_af_calibrate_here,
            methods=["POST"],
        )
        route_wrapper(
            "/api/action/laser_af_warm_up_laser",
            AutofocusLaserWarmup,
            methods=["POST"],
        )

        # Calibrate stage position
        route_wrapper(
            "/api/action/calibrate_stage_xy_here",
            self.calibrate_stage_xy_here,
            methods=["POST"],
        )

        # Turn off all illumination
        route_wrapper(
            "/api/action/turn_off_all_illumination",
            IlluminationEndAll,
            methods=["POST"],
        )

        # init some self.fields

        main_camera_imaging_channels=get_hardware_capabilities().main_camera_imaging_channels
        self.latest_image_handle:dict[str,str]={
            channel.handle:""
            for channel
            in main_camera_imaging_channels
        }
        """
        store last image acquired with main imaging camera.
        key is channel handle, value is actual handle
        """
        self.images:dict[str,dict[str,ImageStoreEntry]]={
            channel.handle:{}
            for channel
            in main_camera_imaging_channels
        }
        """
        contain the actual image data for each handle.
        the image handle code is responsible for cleaning up old images
        """

        # only store the latest laser af image
        self.laser_af_image_latest_handle:str=""
        self.laser_af_images:dict[str,ImageStoreEntry]={}

        self.state=CoreState.Idle

        self.acquisition_thread=None

    def get_config_list(self)->ConfigListResponse:
        """
        get list of existing config files

        these files are already stored on the machine, and can be loaded on request.
        """

        def map_filepath_to_info(c:path.Path)->Optional[ConfigFileInfo]:
            filename=c.name
            timestamp=None
            comment=None

            with c.open("r") as f:
                contents=json.load(f)
                config=AcquisitionConfig(**contents)

                timestamp=config.timestamp
                comment=config.comment

                cell_line=config.cell_line
                
                plate_type=None
                for plate in sc.Plates:
                    if plate.Model_id==config.wellplate_type:
                        plate_type=f"{plate.Manufacturer} {plate.Model_name}"

                if plate_type is None:
                    print(f"error - plate type {config.wellplate_type} in config file {c.name} not found in seaconfig plate list")
                    return None

            return ConfigFileInfo(
                filename=filename,
                timestamp=timestamp,
                comment=comment,
                cell_line=cell_line,
                plate_type=plate_type,
            )

        config_list_str=[]
        for c in GlobalConfigHandler.get_config_list():
            next_config=None
            try:
                next_config=map_filepath_to_info(c)
            except:
                pass

            if next_config is None:
                continue

            config_list_str.append(next_config)

        ret=ConfigListResponse(
            status="success",
            configs=config_list_str,
        )

        return ret

    def config_fetch(self,config_file:str)->ConfigFetchResponse:
        """
        get contents of specific config file

        retrieves the whole file. this return value can be submitted as config file with an acquisition start command.
        """

        filename=config_file

        filepath=None
        for c_path in GlobalConfigHandler.get_config_list():
            if c_path.name==filename:
                filepath=c_path

        if filepath is None:
            error_internal(detail=f"config file with name {filename} not found")

        with filepath.open("r") as f:
            config_json=json.load(f)

        config=sc.AcquisitionConfig(**config_json)

        return ConfigFetchResponse(
            status="success",
            file=config
        )

    def config_store(self,config_file:sc.AcquisitionConfig,filename:str,comment:None|str=None)->BasicSuccessResponse:
        """
        store this file locally

        stores the file on the microscope-connected computer for later retrieval.
        the optional comment provided is stored with the config file to quickly identify its purpose/function.
        """

        config_file.timestamp=sc.datetime2str(dt.datetime.now(dt.timezone.utc))

        # get machine config
        if config_file.machine_config is not None:
            GlobalConfigHandler.override(config_file.machine_config)

        if comment is not None:
            config_file.comment=comment

        try:
            GlobalConfigHandler.add_config(config_file,filename,overwrite_on_conflict=False)
        except Exception as e:
            error_internal(detail=f"failed storing config to file because {e}")

        return BasicSuccessResponse(status="success")

    async def snap_selected_channels(self,config_file:sc.AcquisitionConfig)->BasicSuccessResponse:
        """
        take a snapshot of all selected channels

        these images will be stored into the local buffer for immediate retrieval, i.e. NOT stored to disk.

        if autofocus is calibrated, this will automatically run the autofocus and take channel z offsets into account
        """

        if self.is_in_loading_position:
            error_internal(detail="now allowed while in loading position")

        # get machine config
        if config_file.machine_config is not None:
            GlobalConfigHandler.override(config_file.machine_config)

        g_config=GlobalConfigHandler.get_dict()

        laf_is_calibrated=g_config["laser_autofocus_is_calibrated"]

        # get channels from that, filter for selected/enabled channels
        channels=[c for c in config_file.channels if c.enabled]

        # then:
        # if autofocus is available, measure and approach 0 in a loop up to 5 times
        current_state=await self.get_current_state()
        if current_state.status!="success":
            return current_state
        reference_z_mm=current_state.stage_position.z_pos_mm

        if laf_is_calibrated is not None and laf_is_calibrated.boolvalue:
            for i in range(5):
                displacement_measure_data=await AutofocusMeasureDisplacement(config_file=config_file).run(self)
                if displacement_measure_data.status!="success":
                    error_internal(detail=f"failed to measure displacement for reference z: {displacement_measure_data.message}")

                current_displacement_um=displacement_measure_data.displacement_um
                assert current_displacement_um is not None
                if np.abs(current_displacement_um)<0.5:
                    break

                move_data=await MoveBy(axis="z",distance_mm=-1e-3*current_displacement_um).run(self)
                if move_data.status!="success":
                    error_internal(detail=f"failed to move into focus: {move_data.message}")

            # then store current z coordinate as reference z
            current_state=await self.get_current_state()
            if current_state.status!="success":
                return current_state
            reference_z_mm=current_state.stage_position.z_pos_mm

        # then go through list of channels, and approach each channel with offset relative to reference z 
        for channel in channels:
            move_to_data=await MoveTo(x_mm=None,y_mm=None,z_mm=reference_z_mm+channel.z_offset_um*1e-3).run(self)
            if move_to_data.status!="success":
                error_internal(detail=f"failed to move to channel offset: {move_to_data.message}")

            channel_snap_data=await ChannelSnapshot(channel=channel.dict()).run(self)
            if channel_snap_data.status!="success":
                error_internal(detail=f"failed to take channel snapshot: {channel_snap_data.message}")

        return BasicSuccessResponse(status="success")

    async def calibrate_stage_xy_here(self)->BasicSuccessResponse:
        """
        set current xy position as reference

        set current xy position as top left corner of B2, which is used as reference to calculate all other positions on a plate.

        this WILL lead to hardware damage (down the line) if used improperly!
        due to the delay between improper calibration and actual cause of the damage, this function should be treat with appropriate care.
        """

        if self.is_in_loading_position:
            error_internal(detail="now allowed while in loading position")

        # TODO this needs a better solution where the plate type can be configured
        plate=[p for p in sc.Plates if p.Model_id=="revvity-phenoplate-384"][0]

        current_pos=await self.mc.get_last_position()

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

        return BasicSuccessResponse(
            status="success"
        )

    def _get_peak_coords(self,img:np.ndarray,use_glass_top:bool=False,TOP_N_PEAKS:int=2)->tuple[float,list[float]]:
        """
        get peaks in laser autofocus signal

        used to derive actual location information. by itself not terribly useful.

        returns rightmost_peak_x, distances_between_peaks
        """

        # 8 bit signal -> max value 255
        I = img
        I_1d:np.ndarray=I.max(axis=0) # use max to avoid issues with noise (sum is another option, but prone to issues with noise)
        x=np.array(range(len(I_1d)))
        y=I_1d

        # locate peaks == locate dots
        peak_locations,_ = scipy.signal.find_peaks(I_1d,distance=300,height=10)

        # order by height to pick top N
        tallestpeaks_x=list(sorted(peak_locations.tolist(),key=lambda x:float(I_1d[x])))[-TOP_N_PEAKS:]
        # then order peaks by x again
        tallestpeaks_x=list(sorted(tallestpeaks_x))
        assert len(tallestpeaks_x)>0, "no signal found"

        # Find the rightmost (largest x) peak
        rightmost_peak:float = max(tallestpeaks_x)

        # Compute distances between consecutive peaks
        distances_between_peaks:list[float] = [tallestpeaks_x[i+1] - tallestpeaks_x[i] for i in range(len(tallestpeaks_x)-1)]

        # Output rightmost peak and distances between consecutive peaks
        return rightmost_peak, distances_between_peaks

    async def laser_af_calibrate_here(self,
        z_mm_movement_range_mm:float=0.3,

        Z_MM_BACKLASH_COUNTER:float=40e-3,
        NUM_Z_STEPS_CALIBRATE:int=13
    )->LaserAutofocusCalibrationResponse:
        """
            set current position as laser autofocus reference

            calculates the conversion factor between pixels and micrometers, and sets the reference for the laser autofocus signal

            the calibration process takes dozens of measurements of the laser autofocus signal at known z positions.
            then it calculates the positions of the dots, and tracks them over time. this is expected to observe two dots, 
            at constant distance to each other. one dot may only be visible for a subrange of the total z range, which is
            expected.
            one dot is known to be always visible, so its trajectory is used as reference data.

            measuring the actual offset at an unknown z then locates the dot[s] on the sensor, and uses the position of the dot
            that is known to be always visible to calculate the approximate z offset, based on the reference measurements.
        """

        if self.is_in_loading_position:
            error_internal(detail="now allowed while in loading position")

        DEBUG_LASER_AF_CALIBRATION=bool(0)
        DEBUG_LASER_AF_SHOW_REGRESSION_FIT=False
        DEBUG_LASER_AF_SHOW_EVAL_FIT=True

        if DEBUG_LASER_AF_CALIBRATION:
            z_mm_movement_range_mm=0.3
            NUM_Z_STEPS_CALIBRATE=13
        else:
            z_mm_movement_range_mm=0.05
            NUM_Z_STEPS_CALIBRATE=7

        async def get_current_z_mm()->float:
            current_state=await self.get_current_state()
            assert current_state.status=="success"
            return current_state.stage_position.z_pos_mm

        g_config=GlobalConfigHandler.get_dict()

        conf_af_exp_ms_item=g_config["laser_autofocus_exposure_time_ms"]
        assert conf_af_exp_ms_item is not None
        conf_af_exp_ms=conf_af_exp_ms_item.floatvalue

        conf_af_exp_ag_item=g_config["laser_autofocus_analog_gain"]
        assert conf_af_exp_ag_item is not None
        conf_af_exp_ag=conf_af_exp_ag_item.floatvalue

        z_step_mm=z_mm_movement_range_mm/(NUM_Z_STEPS_CALIBRATE-1)
        half_z_mm=z_mm_movement_range_mm/2

        start_z_mm:float=await get_current_z_mm()

        # move down by half z range
        if Z_MM_BACKLASH_COUNTER is not None:
            await self.mc.send_cmd(mc.Command.move_by_mm("z",-(half_z_mm+Z_MM_BACKLASH_COUNTER)))
            await self.mc.send_cmd(mc.Command.move_by_mm("z",Z_MM_BACKLASH_COUNTER))
        else:
            await self.mc.send_cmd(mc.Command.move_by_mm("z",-half_z_mm))

        # Display each peak's height and width
        class CalibrationData(BaseModel):
            z_mm:float
            p:tuple[float,list[float]]

        async def measure_dot_params():
            # measure pos
            res=await AutofocusSnap(exposure_time_ms=conf_af_exp_ms,analog_gain=conf_af_exp_ag).run(self)
            if res.status!="success":
                error_internal(detail="failed to snap autofocus image [1]")

            assert self.laser_af_image_latest_handle is not None
            latest_laser_af_image=self.laser_af_images.get(self.laser_af_image_latest_handle)
            if latest_laser_af_image is None:
                error_internal(detail="no laser autofocus image found [1]")

            params = self._get_peak_coords(latest_laser_af_image.img)
            return params

        peak_info:list[CalibrationData] = []

        for i in range(NUM_Z_STEPS_CALIBRATE):
            if i>0:
                # move up by half z range to get position at original position, but moved to from fixed direction to counter backlash
                await self.mc.send_cmd(mc.Command.move_by_mm("z",z_step_mm))

            params = await measure_dot_params()

            peak_info.append(CalibrationData(z_mm=-half_z_mm+i*z_step_mm,p=params))

        # move to original position
        await self.mc.send_cmd(mc.Command.move_by_mm("z",-half_z_mm))

        class DomainInfo(BaseModel):
            lowest_x:float
            highest_x:float
            peak_xs:list[tuple[list[float],list[float]]]

        # in the two peak domain, both dots are visible
        two_peak_domain=DomainInfo(lowest_x=3000,highest_x=0,peak_xs=[])
        # in the one peak domain, only one dot is visible (the one with the lower x in the two dot domain)
        one_peak_domain=DomainInfo(lowest_x=3000,highest_x=0,peak_xs=[])

        distances=[]
        distance_x=[]
        for i in peak_info:
            rightmost_x,p_distances=i.p

            new_distances=[rightmost_x]
            for p_d in p_distances:
                new_distances.append(new_distances[-1]-p_d)

            new_z=[i.z_mm]*len(new_distances)

            distances.extend(new_distances)
            distance_x.extend(new_z)

            if len(p_distances)==0:
                target_domain=one_peak_domain
            elif len(p_distances)==1:
                target_domain=two_peak_domain
            else:
                assert False

            for x in new_distances:
                target_domain.lowest_x=min(target_domain.lowest_x,x)
                target_domain.highest_x=max(target_domain.highest_x,x)
                target_domain.peak_xs.append((new_distances,new_z))

        if DEBUG_LASER_AF_CALIBRATION and DEBUG_LASER_AF_SHOW_REGRESSION_FIT:
            plt.figure(figsize=(8, 6))
            plt.scatter(distance_x,distances, color='blue',label="all peaks")

        # x is dot x
        # y is z coordinate
        left_dot_x=[]
        left_dot_y=[]
        right_dot_x=[]
        right_dot_y=[]

        for peak_y,peak_x in one_peak_domain.peak_xs:
            left_dot_x.append(peak_x[0])
            left_dot_y.append(peak_y[0])

        for peak_y,peak_x in two_peak_domain.peak_xs:
            right_dot_x.append(peak_x[0])
            right_dot_y.append(peak_y[0])
            left_dot_x.append(peak_x[1])
            left_dot_y.append(peak_y[1])

        left_dot_regression=linear_regression(left_dot_x,left_dot_y)
        right_dot_regression=linear_regression(right_dot_x,right_dot_y)

        if DEBUG_LASER_AF_CALIBRATION and DEBUG_LASER_AF_SHOW_REGRESSION_FIT:
            # plot one peak domain
            domain_x=[]
            domain_y=[]
            for py,pz in one_peak_domain.peak_xs:
                domain_x.extend(pz)
                domain_y.extend(py)

            plt.scatter(domain_x,domain_y,color="green",marker="x",label="one peak domain")

            # plot two peak domain
            domain_x=[]
            domain_y=[]
            for py,pz in two_peak_domain.peak_xs:
                domain_x.extend(pz)
                domain_y.extend(py)

            plt.scatter(domain_x,domain_y,color="red",marker="x",label="two peak domain")

            #plot left dot regression
            slope,intercept=left_dot_regression
            plt.axline((0,intercept),slope=slope,color="purple",label="left dot regression")

            # plot right dot regression
            slope,intercept=right_dot_regression
            plt.axline((0,intercept),slope=slope,color="black",label="right dot regression")

            plt.xlabel('physical z coordinate')
            plt.ylabel('sensor x coordinate')
            plt.legend()
            plt.grid(True)
            plt.show()

        # -- eval performance, display with pyplot
        if DEBUG_LASER_AF_CALIBRATION and DEBUG_LASER_AF_SHOW_EVAL_FIT:

            start_z=await get_current_z_mm()
            # z_mm_movement_range_mm*=0.8
            half_z_mm=z_mm_movement_range_mm/2
            num_z_steps_eval=51
            z_step_mm=z_mm_movement_range_mm/(num_z_steps_eval-1)

            if Z_MM_BACKLASH_COUNTER is not None:
                await self.mc.send_cmd(mc.Command.move_by_mm("z",-(half_z_mm+Z_MM_BACKLASH_COUNTER)))
                await self.mc.send_cmd(mc.Command.move_by_mm("z",Z_MM_BACKLASH_COUNTER))
            else:
                await self.mc.send_cmd(mc.Command.move_by_mm("z",-half_z_mm))

            approximated_z:list[tuple[float,float]]=[]
            for i in range(num_z_steps_eval):
                if i>0:
                    # move up by half z range to get position at original position, but moved to from fixed direction to counter backlash
                    await self.mc.send_cmd(mc.Command.move_by_mm("z",z_step_mm))

                approx=await self._approximate_laser_af_z_offset_mm(LaserAutofocusCalibrationData(um_per_px=left_dot_regression[0],x_reference=left_dot_regression[1],calibration_position=StagePosition.zero()))
                current_z_mm=await get_current_z_mm()
                approximated_z.append((current_z_mm-start_z,approx))

            # move to original position
            await self.mc.send_cmd(mc.Command.move_by_mm("z",-half_z_mm))

            plt.figure(figsize=(8, 6))
            plt.scatter([v[0]*1e3 for v in approximated_z],[v[0]*1e3 for v in approximated_z], color='green',label="real/expected", marker='o', linestyle='-')
            plt.scatter([v[0]*1e3 for v in approximated_z],[v[1]*1e3 for v in approximated_z], color='blue',label="estimated", marker='o', linestyle='-')
            plt.scatter([v[0]*1e3 for v in approximated_z],[(v[1]-v[0])*1e3 for v in approximated_z], color='red',label="error", marker='o', linestyle='-')
            plt.xlabel('real z [um]')
            plt.ylabel('measured z [um]')
            plt.legend()
            plt.grid(True)
            plt.show()

        um_per_px,_x_reference=left_dot_regression
        print(f"y = {_x_reference} + x * {um_per_px}")
        x_reference=(await measure_dot_params())[0]
        print(f"{_x_reference=} {x_reference=}")

        current_state=await self.get_current_state()
        if current_state.status!="success":
            return current_state

        calibration_position=current_state.stage_position

        calibration_data=LaserAutofocusCalibrationData(
            # calculate the conversion factor, based on lowest and highest measured position
            um_per_px=um_per_px,
            # set reference position
            x_reference=x_reference,
            calibration_position=calibration_position
        )

        return LaserAutofocusCalibrationResponse(status="success",calibration_data=calibration_data)

    async def _approximate_laser_af_z_offset_mm(self,calib_params:LaserAutofocusCalibrationData,_leftmostxinsteadofestimatedz:bool=False)->float:
        """
        approximate current z offset (distance from current imaging plane to focus plane)

        args:
            calib_params:
            _leftmostxinsteadofestimatedz: if True, return the coordinate of the leftmost dot in the laser autofocus signal instead of the estimated z value that is based on this coordinate
        """

        g_config=GlobalConfigHandler.get_dict()

        conf_af_exp_ms_item=g_config["laser_autofocus_exposure_time_ms"]
        assert conf_af_exp_ms_item is not None
        conf_af_exp_ms=conf_af_exp_ms_item.floatvalue

        conf_af_exp_ag_item=g_config["laser_autofocus_analog_gain"]
        assert conf_af_exp_ag_item is not None
        conf_af_exp_ag=conf_af_exp_ag_item.floatvalue

        # get params to describe current signal
        res=await AutofocusSnap(exposure_time_ms=conf_af_exp_ms,analog_gain=conf_af_exp_ag).run(self)
        assert res.status=="success"

        assert self.laser_af_image_latest_handle is not None
        latest_laser_af_image=self.laser_af_images.get(self.laser_af_image_latest_handle)
        assert latest_laser_af_image is not None

        new_params = self._get_peak_coords(latest_laser_af_image.img)
        rightmost_x,interpeakdistances=new_params

        if len(interpeakdistances)==0:
            leftmost_x=rightmost_x
        elif len(interpeakdistances)==1:
            leftmost_x=rightmost_x-interpeakdistances[0]
        else: assert False

        if _leftmostxinsteadofestimatedz:
            return leftmost_x

        def find_x_for_y(y_measured,regression_params):
            "find x (input value, z position) for given y (measured value, dot location on sensor)"
            slope,intercept=regression_params
            return (y_measured - intercept) / slope

        regression_params=(calib_params.um_per_px,calib_params.x_reference)

        return find_x_for_y(leftmost_x,regression_params)

    def _store_new_laseraf_image(self,img:np.ndarray,channel_config:sc.AcquisitionChannelConfig)->str:
        """
            store a new laser autofocus image, return the handle
        """

        # apply a bit of blur to solve some noise related issues
        img = cv2.GaussianBlur(img,(3,3),cv2.BORDER_DEFAULT)

        # generate new handle
        self.laser_af_image_latest_handle=f"laseraf_{generate_random_number_string()}"
            
        pxfmt_int,pxfmt_str=self.focus_cam.handle.PixelFormat.get() # type: ignore
        match pxfmt_int:
            case gxiapi.GxPixelFormatEntry.MONO8:
                pixel_depth=8
            case gxiapi.GxPixelFormatEntry.MONO10:
                pixel_depth=10
            case gxiapi.GxPixelFormatEntry.MONO12:
                pixel_depth=12
            case _unknown:
                error_internal(detail=f"unexpected pixel format {pxfmt_int = } {pxfmt_str = }")

        # store new image
        self.laser_af_images[self.laser_af_image_latest_handle]=ImageStoreEntry(img=img,timestamp=time.time(),channel_config=channel_config,bit_depth=pixel_depth)

        # remove oldest images if more than 8 are stored
        MAX_NUM_IMAGES=8
        if len(self.laser_af_images)>MAX_NUM_IMAGES:
            channel_image_entries=sorted(list(self.laser_af_images.items()),key=lambda i:i[1].timestamp)[:-MAX_NUM_IMAGES]
            for channel_image_key,channel_image_entry in channel_image_entries:
                del self.laser_af_images[channel_image_key]

        return self.laser_af_image_latest_handle

    def _store_new_image(self,img:np.ndarray,channel_config:sc.AcquisitionChannelConfig)->str:
        """
            store a new image, return the handle
        """

        # remove last image in channel from self.images

        channel_images=self.images[channel_config.handle]

        image_timestamps=sorted([(k,v) for (k,v) in channel_images.items()],key=lambda i:i[1].timestamp)
        # only keep last N-1 (then add new one for a total of N)
        MAX_NUM_IMAGES_PER_CHANNEL=8
        for image_handle,image in image_timestamps[:-(MAX_NUM_IMAGES_PER_CHANNEL-1)]:
            del image.img # specifically delete this to free memory
            del channel_images[image_handle]

        # generate new handle
        new_image_handle=channel_config.handle+generate_random_number_string()
        self.latest_image_handle[channel_config.handle]=new_image_handle

        pxfmt_int,pxfmt_str=self.main_cam.handle.PixelFormat.get() # type: ignore
        match pxfmt_int:
            case gxiapi.GxPixelFormatEntry.MONO8:
                pixel_depth=8
            case gxiapi.GxPixelFormatEntry.MONO10:
                pixel_depth=10
            case gxiapi.GxPixelFormatEntry.MONO12:
                pixel_depth=12
            case _unknown:
                error_internal(detail=f"unexpected pixel format {pxfmt_int = } {pxfmt_str = }")
            
        # store new image
        channel_images[new_image_handle]=ImageStoreEntry(img=img,timestamp=time.time(),channel_config=channel_config,bit_depth=pixel_depth)

        return new_image_handle

    def _get_imageinfo_by_handle(self,img_handle:str)->Optional[ImageStoreEntry]:
        "get image story entry for image handle, regardless of channel (only images though, not laser af)"
        img_container=None
        for channel_images in self.images.values():
            img_container=channel_images.get(img_handle)
            if img_container is not None:
                break

        if img_container is None:
            img_container=self.laser_af_images.get(img_handle)
            
        return img_container

    async def send_image_by_handle(self,img_handle:str,quick_preview:bool=False)->Response:
        """
            send image with given handle

            allows use in <img src='...'> tags.

            args:
                quick_preview: if true, reduces image quality to minimize camera->display latency
        """

        img_container=self._get_imageinfo_by_handle(img_handle)
            
        if img_container is None:
            error_internal(detail=f"img_handle {img_handle} not found")

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
                        error_internal(detail=f"unexpected bit depth {img_container.bit_depth}")
            case np.uint8:
                assert img_container.bit_depth==8, f"unexpected {img_container.bit_depth = }"
                img=img_raw
            case _:
                error_internal(detail=f"unexpected dtype {img_raw.dtype}")

        if quick_preview:
            preview_resolution_scaling=GlobalConfigHandler.get_dict()["streaming_preview_resolution_scaling"].intvalue

            img=img[::preview_resolution_scaling,::preview_resolution_scaling]

        if quick_preview:
            g_config=GlobalConfigHandler.get_dict()
            streaming_format_item=g_config["streaming_preview_format"]
            assert streaming_format_item is not None
            streaming_format=streaming_format_item.value
            match streaming_format:
                case "jpeg":
                    pil_kwargs,mimetype={"format":"JPEG","quality":50},"image/jpeg"
                case "png":
                    pil_kwargs,mimetype={"format":"PNG","compress_level":0},"image/png"
                case _other:
                    error_internal(detail=f"unexpected streaming_preview_format format {streaming_format}")

        else:
            g_config=GlobalConfigHandler.get_dict()
            streaming_format_item=g_config["image_display_format"]
            assert streaming_format_item is not None
            streaming_format=streaming_format_item.value
            match streaming_format:
                case "jpeg":
                    pil_kwargs,mimetype={"format":"JPEG","quality":95},"image/jpeg"
                case "png":
                    pil_kwargs,mimetype={"format":"PNG","compress_level":3},"image/png"
                case _other:
                    error_internal(detail=f"unexpected image_display_format format {streaming_format}")

        # compress image async
        def compress_image():
            img_pil=Image.fromarray(img,mode="L")

            img_io=io.BytesIO()

            img_pil.save(img_io,**pil_kwargs)

            img_io.seek(0)

            return img_io

        img_io=await asyncio.get_running_loop().run_in_executor(None,compress_image)

        headers = {"Content-Disposition": f"inline; filename=image.{streaming_format}"}

        return StreamingResponse(img_io, media_type=mimetype, headers=headers)

    async def get_image_histogram(self,handle:str)->HistogramResponse:
        """
        calculate image histogram

        returns 256 value bucket of image histogram from handle, no matter the pixel depth.
        """

        img_info=self._get_imageinfo_by_handle(handle)

        if img_info is None:
            error_internal(detail="image not found")
        
        img=img_info.img

        # calc time scales with DOWNSAMPLE_FACTOR squared! (i.e. factor=2 -> 1/4th the time)
        DOWNSAMPLE_FACTOR=4
        rescaled_img=img[::DOWNSAMPLE_FACTOR,::DOWNSAMPLE_FACTOR]

        def calc_hist()->list[int]:
            if img.dtype==np.uint16:
                hist,edges=np.histogram((rescaled_img>>(16-img_info.bit_depth)).astype(np.uint8),bins=256,range=(0,255))
            else:
                hist,edges=np.histogram(rescaled_img,bins=256,range=(0,255))

            return hist.tolist()

        hist=await asyncio.get_running_loop().run_in_executor(None,calc_hist)

        return HistogramResponse(status="success",channel_name=img_info.channel_config.name,hist_values=hist)

    async def get_current_state(self)->CoreCurrentState:
        """
        get current state of the microscope

        for details see fields of return value
        """

        last_stage_position=await self.mc.get_last_position()

        latest_img_info={
            channel_handle:ImageStoreInfo(
                handle=img_handle,
                channel=img_info.channel_config,
                width_px=img_info.img.shape[1],
                height_px=img_info.img.shape[0],
                timestamp=img_info.timestamp,
            )
            for (channel_handle,img_handle,img_info)
            in (
                (channel_handle,img_handle,self.images[channel_handle][img_handle])
                for (channel_handle,img_handle)
                in self.latest_image_handle.items()
                if img_handle in self.images[channel_handle]
            )
        }

        # supposed=real-calib
        x_pos_mm=self.pos_x_measured_to_real(last_stage_position.x_pos_mm)
        y_pos_mm=self.pos_y_measured_to_real(last_stage_position.y_pos_mm)

        current_acquisition_id=None
        if self.acquisition_is_running:
            for acq_id,acquisition_status in self.acquisition_map.items():
                if acquisition_status.thread_is_running==True:
                    if current_acquisition_id is not None:
                        print(f"warning - more than one acquisition is running at a time?! {current_acquisition_id} and {acq_id}")
                    current_acquisition_id=acq_id
        
        return CoreCurrentState(
            status="success",
            state=self.state.value,
            is_in_loading_position=self.is_in_loading_position,
            stage_position=StagePosition(
                x_pos_mm=x_pos_mm,
                y_pos_mm=y_pos_mm,
                z_pos_mm=last_stage_position.z_pos_mm,
            ),
            latest_imgs=latest_img_info,
            current_acquisition_id=current_acquisition_id,
        )

    async def cancel_acquisition(self,acquisition_id:str)->BasicSuccessResponse:
        """
        cancel the ongoing acquisition
        """
        
        if acquisition_id not in self.acquisition_map:
            error_internal(detail= "acquisition_id not found")
        
        acq=self.acquisition_map[acquisition_id]

        if acq.thread_is_running:
            if self.acquisition_thread is None:
                acq.thread_is_running=False    
            else:
                acq.thread_is_running=not self.acquisition_thread.done()

        if not acq.thread_is_running:
            error_internal(detail=f"acquisition with id {acquisition_id} is not running")

        await acq.queue_in.put(AcquisitionCommand.CANCEL)

        return BasicSuccessResponse(status="success")

    @property
    def acquisition_is_running(self)->bool:
        if self.acquisition_thread is None:
            return False

        if self.acquisition_thread.done():
            self.acquisition_thread=None
            return False

        return True

    async def start_acquisition(self,config_file:sc.AcquisitionConfig)->AcquisitionStartResponse:
        """
        start an acquisition

        the acquisition is run in the background, i.e. this command returns after acquisition bas begun. see /api/acquisition/status for ongoing status of the acquisition.
        """

        if self.is_in_loading_position:
            error_internal(detail="now allowed while in loading position")
        
        if self.acquisition_thread is not None:
            if not self.acquisition_thread.done():
                error_internal(detail="acquisition already running")
            else:
                self.acquisition_thread=None

        # get machine config
        if config_file.machine_config is not None:
            GlobalConfigHandler.override(config_file.machine_config)

        g_config=GlobalConfigHandler.get_dict()

        # TODO generate some unique acqisition id to identify this acquisition by
        # must be robust against server restarts, and async requests
        # must also cache previous run results, to allow fetching information from past acquisitions
        RANDOM_ACQUISITION_ID="a65914"
        # actual acquisition id has some unique identifier, plus a timestamp for legacy reasons
        # TODO remove the timestamp from the directory name in the future, because this is a horribly hacky way
        # to handle duplicate directory names, which should be avoided anyway
        acquisition_id=f"{RANDOM_ACQUISITION_ID}_{sc.datetime2str(dt.datetime.now(dt.timezone.utc))}"

        plates=[p for p in sc.Plates if p.Model_id==config_file.wellplate_type]
        if len(plates)==0:
            error_internal(detail="unknown wellplate type")
        assert len(plates)==1, f"multiple wellplate types with id {config_file.wellplate_type}"

        plate=plates[0]

        for well in config_file.plate_wells:
            if wellIsForbidden(well.well_name,plate):
                error_internal(detail=f"well {well.well_name} is not allowed on this plate")

        if config_file.autofocus_enabled:
            laser_autofocus_is_calibrated_item=g_config["laser_autofocus_is_calibrated"]

            laser_autofocus_is_calibrated=laser_autofocus_is_calibrated_item is not None and laser_autofocus_is_calibrated_item.boolvalue

            if not laser_autofocus_is_calibrated:
                error_internal(detail="laser autofocus is enabled, but not calibrated")

        queue_in=asyncio.Queue()
        queue_out=asyncio.Queue()
        acquisition_status=AcquisitionStatus(
            acquisition_id=acquisition_id,
            queue_in=queue_in,
            queue_out=queue_out,
            last_status=None,
            thread_is_running=False,
        )
        self.acquisition_map[acquisition_id]=acquisition_status
            

        ongoing_image_store_tasks:list=[]
        def handle_q_in(q_in=queue_in):
            """ if there is something in q_in, fetch it and handle it (e.g. terminae on cancel command) """
            if not q_in.empty():
                q_in_item=q_in.get_nowait()
                if q_in_item==AcquisitionCommand.CANCEL:
                    error_internal(detail="acquisition cancelled")

                print(f"warning - command unhandled: {q_in_item}")
        protocol=ProtocolGenerator(
            config_file=config_file,
            handle_q_in=handle_q_in,
            plate=plate,
            acquisition_status=acquisition_status,
            acquisition_id=acquisition_id,
            ongoing_image_store_tasks=ongoing_image_store_tasks
        )

        if protocol.num_images_total==0:
            # TODO set acquisition_status here
            error_internal(detail=f"no images to acquire ({protocol.num_wells = },{protocol.num_sites = },{protocol.num_channels = },{protocol.num_channel_z_combinations = })")

        async def run_acquisition(
            q_in:asyncio.Queue[AcquisitionCommand],
            q_out:asyncio.Queue[InternalErrorModel|AcquisitionStatusOut],
        ):
            """
            acquisition execution

            may be run in another thread

            arguments:
                - q_in:q.Queue send messages into the acquisition logic, mainly for cancellation message
                - q_out:q.Queue acquisition status updates are posted at regular logic intervals. The
                    queue length is capped to a low number, so long times without reading an update to not
                    consume large amounts of memory. The oldest messages are evicted first.

            """

            try:
                # initiate generation
                protocol_generator=protocol.generate()

                # send none on first yield
                next_step=protocol_generator.send(None)
                while next_step is not None:
                    if isinstance(next_step,str):
                        result=None
                    elif next_step is None:
                        break
                    else:
                        result=await next_step.run(self)
                    next_step=protocol_generator.send(result)

                # finished regularly, set status accordingly (there must have been at least one image, so a status has been set)
                assert acquisition_status.last_status is not None
                acquisition_status.last_status.acquisition_status="completed"

            except Exception as e:
                print(f"error during acquisition {e}\n{traceback.format_exc()}")
                
                if isinstance(e,HTTPException):
                    print("exception is httpexcepton")
                    if acquisition_status.last_status is not None:
                        print("exception is httpexcepton")
                        acquisition_status.last_status.acquisition_status="cancelled"

                else:
                    full_error=traceback.format_exc()
                    await q_out.put(InternalErrorModel(detail=f"acquisition thread failed because {str(e)}, more specifically: {full_error}"))

                    if acquisition_status.last_status is not None:
                        acquisition_status.last_status.acquisition_status="crashed"

            finally:
                # ensure no dangling image store task threads
                await asyncio.gather(*ongoing_image_store_tasks)

            # indicate that this thread has stopped running (no matter the cause)
            acquisition_status.thread_is_running=False

            print("acquisition thread is done")
            return

        self.acquisition_thread=asyncio.create_task(coro=run_acquisition(queue_in,queue_out))
        acquisition_status.thread_is_running=True

        return AcquisitionStartResponse(status="success",acquisition_id=acquisition_id)

    async def get_acquisition_status(self,acquisition_id:str)->AcquisitionStatusOut:
        """
        get status of an acquisition
        """

        acq_res=self.acquisition_map.get(acquisition_id)
        if acq_res is None:
            error_internal(detail="acquisition_id is invalid")

        if acq_res.last_status is None:
            error_internal(detail="no status available")
        else:
            return acq_res.last_status
    
    def close(self):
        for mc in self.microcontrollers:
            mc.close()

        for cam in self.cams:
            cam.close()

        GlobalConfigHandler.store()

# handle validation errors with ability to print to terminal for debugging

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"Validation error at {request.url}: {exc.errors()}",flush=True)
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )

# -- fix issue in tp.Optional annotation with pydantic
# (from https://github.com/fastapi/fastapi/pull/9873#issuecomment-1997105091)

from fastapi.openapi.utils import get_openapi

def handle_anyof_nullable(schema: dict|list):
    """Recursively modifies the schema to handle anyOf with null for OpenAPI 3.0 compatibility."""

    if isinstance(schema, dict):
        for key, value in list(
            schema.items()
        ):  # Iterate over a copy to avoid modification errors
            if key == "anyOf" and isinstance(value, list):
                non_null_types = [item for item in value if item.get("type") != "null"]
                if len(value) > len(non_null_types):  # Found 'null' in anyOf
                    if len(non_null_types) == 1:
                        schema.update(non_null_types[0])  # Replace with non-null type
                        schema["nullable"] = True
                        del schema[key]  # Remove anyOf
            else:
                handle_anyof_nullable(value)
    elif isinstance(schema, list):
        for item in schema:
            handle_anyof_nullable(item)

def downgrade_openapi_schema_to_3_0(schema: dict) -> dict:
    """Downgrades an OpenAPI schema from 3.1 to 3.0, handling anyOf with null."""

    handle_anyof_nullable(schema)
    return schema

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="my API",
        openapi_version="3.0.1",
        version="1.0.0",
        description="Seafront OpenAPI schema",
        routes=app.routes,
    )
    # Here, modify the openapi_schema as needed to ensure 3.0.0 compatibility
    # For example, adjust for `nullable` fields compatibility
    openapi_schema = downgrade_openapi_schema_to_3_0(openapi_schema)
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# -- end fix

import asyncio

def main():
    core = Core()

    asyncio.run(core.home())
    
    try:
        # Start FastAPI using uvicorn
        uvicorn.run(app, host="127.0.0.1", port=5002, log_level="debug")
    except Exception as e:
        print(f"error running uvicorn: {e=}")
        pass

    print("shutting down")
    core.close()

if __name__=="__main__":
    main()
