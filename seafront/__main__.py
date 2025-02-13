#!/usr/bin/env python3

# system deps

import json, time, os, json
import typing as tp
from typing import Callable, Optional
from types import MethodType

import datetime as dt
import pathlib as path
import re
import random
import traceback
from functools import wraps
import inspect
import asyncio

# math and image dependencies

import numpy as np

# for robust type safety at runtime

from pydantic import create_model, BaseModel, Field
from pydantic.fields import FieldInfo

# for debugging

_DEBUG_P2JS=True

# http server dependencies

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIWebSocketRoute

import uvicorn

# microscope dependencies

from .server.commands import *

import seaconfig as sc
from seaconfig.acquisition import AcquisitionConfig

from .config.basics import ConfigItem, GlobalConfigHandler
from .server.protocol import ProtocolGenerator, AsyncThreadPool, make_unique_acquisition_id
from .hardware.squid import SquidAdapter

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

app = FastAPI(debug=False)
app.mount("/src", StaticFiles(directory="src"), name="static")

# route tags to structure swagger interface (/docs,/redoc)

class RouteTag(str,Enum):
    STATIC_FILES="Static Files"
    ACTIONS="Microscope Actions"
    ACQUISITION_CONTROLS="Acquisition Controls"
    DOCUMENTATION="Documentation"

openapi_tags=[
    {"name":RouteTag.STATIC_FILES.value,"description":"serve static files of all sorts"},
    {"name":RouteTag.ACTIONS.value,"description":"actions for the microscope to perform immediately"},
    {"name":RouteTag.ACQUISITION_CONTROLS.value,"description":"acquisition related controls and information i/o"},
    {"name":RouteTag.DOCUMENTATION.value,"description":"documentation on software and api"},
]

@app.get("/",tags=[RouteTag.STATIC_FILES.value])
async def index():
    return FileResponse("index.html")

@app.get("/css/{path:path}",tags=[RouteTag.STATIC_FILES.value])
async def send_css(path: str):
    return FileResponse(f"css/{path}")

@app.get("/src/{path:path}",tags=[RouteTag.STATIC_FILES.value])
async def send_js(path: str):
    return FileResponse(f"src/{path}")

@app.api_route("/api/get_features/hardware_capabilities", methods=["POST"], summary="Get available imaging channels and plate types.")
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

@app.api_route("/api/get_features/machine_defaults", methods=["POST"], summary="Get default low-level machine parameters.")
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

class HistogramResponse(BaseModel):
    channel_name:str
    hist_values:tp.List[int]

class CustomRoute(BaseModel):
    handler:tp.Union[tp.Type[BaseCommand],tp.Callable]
    tags:tp.List[str]=Field(default_factory=list)

    callback:tp.Optional[tp.Union[tp.Callable[[tp.Any,tp.Any],None],tp.Callable[[tp.Any,tp.Any],tp.Coroutine[tp.Any,tp.Any,None]]]]=None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

custom_route_handlers:tp.Dict[str,CustomRoute]={}

class Core:
    """application core, contains server capabilities and microcontroller interaction"""

    async def home(self):
        """perform homing maneuver"""

        return await self.squid.home()

    @property
    def main_cam(self):
        return self.squid.main_camera
    @property
    def focus_cam(self):
        return self.squid.focus_camera
    @property
    def mc(self):
        return self.squid.microcontroller

    def __init__(self):
        self.lock=CoreLock()

        self.squid=SquidAdapter.make()

        print("initializing microcontroller")

        self.acquisition_map:tp.Dict[str,AcquisitionStatus]={}
        """ map containing information on past and current acquisitions """

        # set up routes to member functions

        if _DEBUG_P2JS:
            def sendp2():
                return FileResponse("../../web-pjs/p2.js")

            app.add_api_route("/p2.js", sendp2, methods=["GET"])

        # store request_models for re-use (works around issues with fastapi)
        request_models=dict()

        # Utility function to wrap the shared logic by including handlers for GET requests
        def route_wrapper(
            path: str,
            route: CustomRoute,
            methods:list[str]=["GET"],
            allow_while_acquisition_is_running: bool = True,
            summary:str|None=None,
            **kwargs_static
        ):
            custom_route_handlers[path]=route

            target_func=route.handler

            async def callfunc(request_data):
                arg=None
                # Call the target function
                if inspect.isclass(target_func):
                    instance = target_func(**request_data)
                    arg=instance
                    if isinstance(instance, BaseCommand):
                        result=await self.squid.execute(instance)
                    else:
                        raise AttributeError(f"Provided class {target_func} {type(target_func)=} is no BaseCommand")
                elif inspect.iscoroutinefunction(target_func):
                    arg=request_data
                    result=await target_func(**request_data)
                elif inspect.isfunction(target_func) or isinstance(target_func, MethodType):
                    arg=request_data
                    result=target_func(**request_data)
                else:
                    raise TypeError(f"Unsupported target_func type: {type(target_func)}")

                if route.callback is not None:
                    if inspect.iscoroutinefunction(route.callback):
                        await route.callback(arg,result)
                    else:
                        route.callback(arg,result)

                return result

            def get_return_type():
                """
                Determines the return type of the target function.
                If the target function is a class, retrieves the return type of the 'run()' method.
                """
                # Case 1: target_func is a coroutine function or a standard function/method
                if inspect.iscoroutinefunction(target_func) or inspect.isfunction(target_func) or isinstance(target_func, MethodType):
                    #print(f"returning {target_func.__name__} {inspect.signature(target_func).return_annotation}")
                    return inspect.signature(target_func).return_annotation

                if issubclass(target_func,BaseCommand):#type:ignore
                    return target_func.__private_attributes__["_ReturnValue"].default#type:ignore
                
                # Case 2: target_func is a class, get return type of the 'run()' method if it exists
                if inspect.isclass(target_func):
                    if hasattr(target_func, 'run'):
                        run_method = getattr(target_func, 'run')
                        return_type=inspect.signature(run_method).return_annotation
                        #print(f"returning {target_func.__name__}.run {return_type}")
                        return return_type

                # Default case: if none of the above matches
                print(f"{target_func=} has unknown return type")
                return tp.Any

            return_type=get_return_type()

            @wraps(target_func)
            async def handler_logic_get(**kwargs:Optional[tp.Any])->return_type: # type: ignore
                # Perform verification
                if (not allow_while_acquisition_is_running) and self.acquisition_is_running:
                    return JSONResponse(content={"status": "error", "message": "cannot run this command while acquisition is running"}, status_code=400)

                request_data=kwargs.copy()
                request_data.update(kwargs_static)

                result=await callfunc(request_data)
                return result

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

                    result=await callfunc(request_data)
                    return result
            else:
                async def handler_logic_post(): # type:ignore
                    # Perform verification
                    if (not allow_while_acquisition_is_running) and self.acquisition_is_running:
                        return JSONResponse(content={"status": "error", "message": "cannot run this command while acquisition is running"}, status_code=400)

                    request_data = kwargs_static.copy()

                    result=await callfunc(request_data)
                    return result

            # copy annotation and fix return type
            handler_logic_post.__doc__ = target_func.__doc__
            handler_logic_post.__annotations__['return'] = return_type

            if summary is None and target_func.__doc__ is not None:
                docstring_lines=[line.lstrip().rstrip() for line in target_func.__doc__.split("\n")]
                docstring_lines=[line for line in docstring_lines if len(line)>0]
                if len(docstring_lines)>=1:
                    summary=docstring_lines[0]

            for m in methods:
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
                        tags=route.tags,#type:ignore
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
                        tags=route.tags,#type:ignore
                    )

        # Register URL rules requiring machine interaction
        route_wrapper(
            "/api/get_info/current_state",
            CustomRoute(handler=self.get_current_state),
            methods=["POST"],
        )
        
        @app.websocket("/ws/get_info/current_state")
        async def ws_get_info_current_state(ws:WebSocket):
            await ws.accept()
            try:
                while True:
                    # await message, but ignore its contents
                    await ws.receive()
                    await ws.send_json((await self.get_current_state()).json())
            except WebSocketDisconnect:
                pass

        @app.websocket("/ws/get_info/acquired_image")
        async def getacquiredimage(ws:WebSocket):
            await ws.accept()
            try:
                while True:
                    channel_name=await ws.receive_text()

                    img=self.latest_images[channel_name]
                    if img is not None:
                        await ws.send_json({"width":img._img.shape[0],"height":img._img.shape[1],"bit_depth":img.bit_depth})

                        img_bytes=np.ascontiguousarray(img._img).tobytes()

                        await ws.send_bytes(img_bytes)

            except WebSocketDisconnect:
                pass

        # Register URLs for immediate moves
        route_wrapper(
            "/api/action/move_by",
            CustomRoute(handler=MoveBy,tags=[RouteTag.ACTIONS.value]),
            methods=["POST"],
        )

        # Register URL for start_acquisition
        route_wrapper(
            "/api/acquisition/start",
            CustomRoute(handler=self.start_acquisition,tags=[RouteTag.ACQUISITION_CONTROLS.value]),
            methods=["POST"],
        )

        # Register URL for cancel_acquisition
        route_wrapper(
            "/api/acquisition/cancel",
            CustomRoute(handler=self.cancel_acquisition,tags=[RouteTag.ACQUISITION_CONTROLS.value]),
            allow_while_acquisition_is_running=True,
            methods=["POST"],
        )

        # Register URL for get_acquisition_status
        route_wrapper(
            "/api/acquisition/status",
            CustomRoute(handler=self.get_acquisition_status,tags=[RouteTag.ACQUISITION_CONTROLS.value]),
            allow_while_acquisition_is_running=True,
            methods=["POST"],
        )

        @app.websocket("/ws/acquisition/status")
        async def ws_acquisition_status(ws:WebSocket):
            await ws.accept()
            try:
                while True:
                    # await message, but ignore its contents
                    args=await ws.receive_json()
                    await ws.send_json((await self.get_acquisition_status(**args)).json())
            except WebSocketDisconnect:
                pass

        # Retrieve config list
        route_wrapper(
            "/api/acquisition/config_list",
            CustomRoute(handler=self.get_config_list,tags=[RouteTag.ACQUISITION_CONTROLS.value]),
            methods=["POST"],
        )

        # Fetch acquisition config
        route_wrapper(
            "/api/acquisition/config_fetch",
            CustomRoute(handler=self.config_fetch,tags=[RouteTag.ACQUISITION_CONTROLS.value]),
            methods=["POST"],
        )

        # Save/load config
        route_wrapper(
            "/api/acquisition/config_store",
            CustomRoute(handler=self.config_store,tags=[RouteTag.ACQUISITION_CONTROLS.value]),
            methods=["POST"],
        )

        # Move to well
        route_wrapper(
            "/api/action/move_to_well",
            CustomRoute(handler=MoveToWell,tags=[RouteTag.ACTIONS.value]),
            methods=["POST"],
        )

        # Loading position enter/leave
        route_wrapper(
            "/api/action/enter_loading_position",
            CustomRoute(handler=LoadingPositionEnter,tags=[RouteTag.ACTIONS.value]),
            methods=["POST"],
        )
        route_wrapper(
            "/api/action/leave_loading_position",
            CustomRoute(handler=LoadingPositionLeave,tags=[RouteTag.ACTIONS.value]),
            methods=["POST"],
        )

        async def write_image(cmd:ChannelSnapshot,res:ImageAcquiredResponse):
            await self._store_new_image(img=res._img,channel_config=cmd.channel)

        # Snap channel
        route_wrapper(
            "/api/action/snap_channel",
            CustomRoute(handler=ChannelSnapshot,tags=[RouteTag.ACTIONS.value],callback=write_image),
            methods=["POST"],
        )

        route_wrapper(
            "/api/action/snap_selected_channels",
            CustomRoute(handler=self.snap_selected_channels,tags=[RouteTag.ACTIONS.value]),
            methods=["POST"],
        )

        # Start streaming (i.e., acquire x images per sec, until stopped)
        self.image_store_threadpool=AsyncThreadPool()
        stream_info:tp.Dict[str,None|sc.AcquisitionChannelConfig]={"channel":None}
        def handle_image(arg:np.ndarray|bool)->bool:
            if isinstance(arg,bool):
                should_stop=arg
                if should_stop:return True
            else:
                img=arg
                assert stream_info["channel"] is not None
                self.image_store_threadpool.run(self._store_new_image(img=img,channel_config=stream_info["channel"]))
            return False

        def register_stream_begin(begin:ChannelStreamBegin,res):
            # register callback on microscope
            self.squid.stream_callback=handle_image
            # store channel info, to be used inside the streaming callback to store the images in the server properly
            stream_info["channel"]=begin.channel
        def register_stream_end(a,b):
            self.squid.stream_callback=None

        route_wrapper(
            "/api/action/stream_channel_begin",
            CustomRoute(handler=ChannelStreamBegin,tags=[RouteTag.ACTIONS.value],callback=register_stream_begin),
            methods=["POST"],
        )
        route_wrapper(
            "/api/action/stream_channel_end",
            CustomRoute(handler=ChannelStreamEnd,tags=[RouteTag.ACTIONS.value],callback=register_stream_end),
            methods=["POST"],
        )

        # Laser autofocus system
        route_wrapper(
            "/api/action/snap_reflection_autofocus",
            CustomRoute(handler=AutofocusSnap,tags=[RouteTag.ACTIONS.value]),
            methods=["POST"],
        )
        route_wrapper(
            "/api/action/measure_displacement",
            CustomRoute(handler=AutofocusMeasureDisplacement,tags=[RouteTag.ACTIONS.value]),
            methods=["POST"],
        )
        route_wrapper(
            "/api/action/laser_autofocus_move_to_target_offset",
            CustomRoute(handler=AutofocusApproachTargetDisplacement,tags=[RouteTag.ACTIONS.value]),
            methods=["POST"],
        )
        route_wrapper(
            "/api/action/laser_af_calibrate",
            CustomRoute(handler=LaserAutofocusCalibrate,tags=[RouteTag.ACTIONS.value]),
            methods=["POST"],
        )
        route_wrapper(
            "/api/action/laser_af_warm_up_laser",
            CustomRoute(handler=AutofocusLaserWarmup,tags=[RouteTag.ACTIONS.value]),
            methods=["POST"],
        )

        # Calibrate stage position
        route_wrapper(
            "/api/action/calibrate_stage_xy_here",
            CustomRoute(handler=self.calibrate_stage_xy_here,tags=[RouteTag.ACTIONS.value]),
            methods=["POST"],
        )

        # Turn off all illumination
        route_wrapper(
            "/api/action/turn_off_all_illumination",
            CustomRoute(handler=IlluminationEndAll,tags=[RouteTag.ACTIONS.value]),
            methods=["POST"],
        )

        self.latest_images:tp.Dict[str,ImageStoreEntry]={}
        "latest image acquired in each channel"

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

        return BasicSuccessResponse()

    async def snap_selected_channels(self,config_file:sc.AcquisitionConfig)->BasicSuccessResponse:
        """
        take a snapshot of all selected channels

        these images will be stored into the local buffer for immediate retrieval, i.e. NOT stored to disk.

        if autofocus is calibrated, this will automatically run the autofocus and take channel z offsets into account
        """

        return await self.squid.snap_selected_channels(config_file)

    async def calibrate_stage_xy_here(self)->BasicSuccessResponse:
        """
        set current xy position as reference

        set current xy position as top left corner of B2, which is used as reference to calculate all other positions on a plate.

        this WILL lead to hardware damage (down the line) if used improperly!
        due to the delay between improper calibration and actual cause of the damage, this function should be treat with appropriate care.
        """

        if self.squid.is_in_loading_position:
            error_internal(detail="now allowed while in loading position")

        # TODO this needs a better solution where the plate type can be configured
        plate=[p for p in sc.Plates if p.Model_id=="revvity-phenoplate-384"][0]

        current_pos=(await self.squid.get_current_state()).stage_position

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

        return BasicSuccessResponse()

    async def _store_new_image(self,img:np.ndarray,channel_config:sc.AcquisitionChannelConfig)->str:
        """
            store a new image, return the channel handle (into self.latest_images)

            note: this stores regular images, as well as autofocus images
        """

        g_config=GlobalConfigHandler.get_dict()

        adapter_state=await self.squid.get_current_state()

        # store new image
        new_image_store_entry=ImageStoreEntry(
            pixel_format=g_config["laser_autofocus_pixel_format"].strvalue,
            info=ImageStoreInfo(
                channel=channel_config,
                width_px=0,
                height_px=0,
                timestamp=time.time(),
                position=SitePosition(
                    well_name="",
                    site_x=0,site_y=0,site_z=0,
                    x_offset_mm=0,y_offset_mm=0,z_offset_mm=0,
                    position=adapter_state.stage_position,
                ),
            )
        )
        new_image_store_entry._img=img
        self.latest_images[channel_config.name]=new_image_store_entry

        return channel_config.name

    """async def _send_image_by_handle(self,img_handle:str,quick_preview:bool=False)->Response:
        """"""
            send image with given handle

            allows use in <img src='...'> tags.

            args:
                quick_preview: if true, reduces image quality to minimize camera->display latency
        """"""

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
    """
    """async def _calculate_histogram(self,handle:str)->HistogramResponse:
        """"""
        calculate image histogram

        returns 256 value bucket of image histogram from handle, no matter the pixel depth.

        may internally downsample the image to reduce calculation time.
        """"""

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

        return HistogramResponse(channel_name=img_info.info.channel.name,hist_values=hist)
    """
    async def get_current_state(self)->CoreCurrentState:
        """
        get current state of the microscope

        for details see fields of return value
        """

        current_acquisition_id=None
        if self.acquisition_is_running:
            for acq_id,acquisition_status in self.acquisition_map.items():
                if acquisition_status.thread_is_running==True:
                    if current_acquisition_id is not None:
                        print(f"warning - more than one acquisition is running at a time?! {current_acquisition_id} and {acq_id}")
                    current_acquisition_id=acq_id

        # blur laser autofocus image (this code is super out of place here)
        # img = cv2.GaussianBlur(img,(3,3),cv2.BORDER_DEFAULT)
        
        return CoreCurrentState(
            adapter_state=await self.squid.get_current_state(),
            latest_imgs={key:entry.info for key,entry in self.latest_images.items()},
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

        return BasicSuccessResponse()

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

        if self.squid.is_in_loading_position:
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

        acquisition_id=make_unique_acquisition_id()

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
            acquisition_id=acquisition_id
        )

        project_name_is_acceptable=len(protocol.config_file.project_name) and name_validity_regex.match(protocol.config_file.project_name)
        if not project_name_is_acceptable:
            error_internal(detail="project name is not acceptable: 1) must not be empty and 2) only contain alphanumeric characters, underscores, dashes")
        
        plate_name_is_acceptable=len(protocol.config_file.plate_name)>0 and name_validity_regex.match(protocol.config_file.plate_name)
        if not plate_name_is_acceptable:
            error_internal(detail="plate name is not acceptable: 1) must not be empty and 2) only contain alphanumeric characters, underscores, dashes")


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
                        result=await self.squid.execute(next_step)

                        if isinstance(next_step,ChannelSnapshot):
                            await self._store_new_image(result._img,next_step.channel)

                    next_step=protocol_generator.send(result)

                # finished regularly, set status accordingly (there must have been at least one image, so a status has been set)
                assert acquisition_status.last_status is not None
                acquisition_status.last_status.acquisition_status="completed"

            except Exception as e:
                if isinstance(e,HTTPException):
                    if acquisition_status.last_status is not None:
                        acquisition_status.last_status.acquisition_status="cancelled"

                else:
                    print(f"error during acquisition {e}\n{traceback.format_exc()}")
                
                    full_error=traceback.format_exc()
                    await q_out.put(InternalErrorModel(detail=f"acquisition thread failed because {str(e)}, more specifically: {full_error}"))

                    if acquisition_status.last_status is not None:
                        acquisition_status.last_status.acquisition_status="crashed"

            finally:
                # ensure no dangling image store task threads
                protocol.image_store_pool.join()

            # indicate that this thread has stopped running (no matter the cause)
            acquisition_status.thread_is_running=False

            print("acquisition thread is done")
            return

        self.acquisition_thread=asyncio.create_task(coro=run_acquisition(queue_in,queue_out))
        acquisition_status.thread_is_running=True

        return AcquisitionStartResponse(acquisition_id=acquisition_id)

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
        self.squid.close()

        GlobalConfigHandler.store()

        self.image_store_threadpool.join()

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

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = {
        "openapi":"3.0.1",
        "info":{
            "title":"seafront api",
            "version":"0.2.0"
        },
        "paths":{},
        "description":"Seafront OpenAPI schema",
        "tags":openapi_tags,
    }

    def register_pydantic_schema(t):
        assert issubclass(t,BaseModel)

        for field in t.model_fields.values():
            # register field types
            type_to_schema(field)

        # make schema as json
        model_schema=t.model_json_schema(mode="serialization")

        # pydantic uses $defs to reference the type of internal fields, but openapi uses components/schemas
        # which we swap via stringify-replace-reparse to write as little code as possible to do what is
        # otherwise recursion
        schema_str=json.dumps(model_schema)
        schema_str=schema_str.replace("#/$defs/","#/components/schemas/")
        model_schema=json.loads(schema_str)

        # the json schema has a top level field called defs, which contains internal fields, which we
        # embed into the openapi schema here (the path replacement is separate from this)
        defs=model_schema.pop("$defs",{})
        openapi_schema.setdefault("components",{}).setdefault("schemas",{}).update(defs)

        # unsure if this works with the new code (written for 0.1, untested in 0.2)
        handle_anyof_nullable(model_schema)

        # finally, add the actual model we have handled to the openapi schema
        openapi_schema.setdefault("components",{}).setdefault("schemas",{})[t.__name__]=model_schema

        return {"$ref":f"#/components/schemas/{t.__name__}"}

    def type_to_schema(t):
        if isinstance(t,FieldInfo):
            t=t.annotation

        if t is int:
            return {"type":"integer"}
        elif t is float:
            return {"type":"number","format":"float"}
        elif t is bool:
            return {"type":"boolean"}
        elif t is dict:
            return {"type":"object"}
        elif inspect.isclass(t) and issubclass(t,BaseModel):#type:ignore
            return register_pydantic_schema(t)#type:ignore
        else:
            origin=tp.get_origin(t)
            if origin in (list,):
                item_type=tp.get_args(t)[0]
                return {"type":"array","items":type_to_schema(item_type)}
            return {"type":"string"}

    for route in app.routes:
        if not hasattr(route,"endpoint"):
            continue

        route_path:str=route.path#type:ignore

        tags:tp.List[str]
        if hasattr(route,"tags"):
            tags=route.tags#type:ignore
        else:
            tags=[]

        responses={
            "200":{
                "description":"Success",
                # actual content type is filled in during return type annotation inspection below
                "content":{"application/json":{"schema":None}}
            },
            "500":{
                "description":"any failure mode",
                "content":{"application/json":{"schema":register_pydantic_schema(InternalErrorModel)}}
            },
        }

        parameters=[]

        endpoint=route.endpoint#type:ignore
        if customroute:=custom_route_handlers.get(route_path):
            if inspect.isclass(customroute.handler) and issubclass(customroute.handler,BaseCommand):#type:ignore
                assert issubclass(customroute.handler,BaseModel), f"{customroute.handler.__name__} does not inherit from basemodel, even though it inherits from basecommand"

                # register 
                type_to_schema(customroute.handler)
                
                responses["200"]["content"]["application/json"]["schema"]=type_to_schema(customroute.handler.__private_attributes__["_ReturnValue"].default)#type:ignore

                for name,field in customroute.handler.model_fields.items():
                    parameters.append({
                        "name":name,
                        "in":"query",
                        "required":field.is_required(),
                        "schema":type_to_schema(field),
                    })
                
        if responses["200"]["content"]["application/json"]["schema"] is None:
            if customroute:=custom_route_handlers.get(route_path):
                endpoint=customroute.handler

            sig=inspect.signature(endpoint)
            hints=tp.get_type_hints(endpoint)
            
            for name,param in sig.parameters.items():
                if name in {"request","background_tasks"}:
                    continue

                if inspect.isclass(endpoint) and issubclass(endpoint,BaseModel) and ((not endpoint.model_fields[name].repr) or (endpoint.model_fields[name].exclude)):
                    continue

                param_schema={
                    "name":name,
                    "in":"query",
                    "required":param.default is inspect.Parameter.empty,
                }

                if name in hints:
                    param_schema["schema"]=type_to_schema(hints[name])

                parameters.append(param_schema)
                
            ret=sig.return_annotation
            if ret is not inspect.Signature.empty:
                responses["200"]["content"]["application/json"]["schema"]=type_to_schema(ret)

        doc=endpoint.__doc__ or ""
        doc_lines=[l.strip() for l in doc.splitlines() if l.strip()]
        summary=doc_lines[0] if doc_lines else ""
        description="\n".join(doc_lines[1:]) if len(doc_lines)>1 else ""

        if isinstance(route,APIWebSocketRoute):
            method="get"
            responses["101"]={
                "description":"switch protocol (to websocket)"
            }
        else:
            method=list(route.methods)[0].lower()#type:ignore

        if route_path not in openapi_schema["paths"]:
            openapi_schema["paths"][route_path]={}

        if route_path in ("/docs","/redoc","/openapi.json","/docs/oauth2-redirect") and len(tags)==0:
            tags=[RouteTag.DOCUMENTATION.value]

        openapi_schema["paths"][route_path][method]={
            "summary":summary,
            "description":description,
            "parameters":parameters,
            "responses":responses,
            "tags":tags,
        }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# -- end fix

import asyncio

# disable compression of websocket connections..
# because compression time is unpredictable (takes 70ms to send an all-white image, and 1400ms, twenty times as long (!!!!) for all-black images, which are not a rare occurence in practice)
import websockets.server
_orig_init=websockets.server.WebSocketServerProtocol.__init__
def _no_comp_init(self,*args,**kwargs):
    kwargs["extensions"]=[]
    return _orig_init(self,*args,**kwargs)
websockets.server.WebSocketServerProtocol.__init__=_no_comp_init

def main():
    core = Core()

    asyncio.run(core.home())
    
    try:
        # Start FastAPI using uvicorn
        uvicorn.run(app, host="127.0.0.1", port=5002)#, log_level="debug")
    except Exception as e:
        print(f"error running uvicorn: {e=}")
        pass

    print("shutting down")
    core.close()

if __name__=="__main__":
    main()
