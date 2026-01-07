#!/usr/bin/env python3

# system deps
import argparse
import asyncio
import datetime as dt
import faulthandler
import inspect
import json
import os
import pathlib as path
import re
import signal
import threading
import time
import traceback
import typing as tp
from concurrent.futures import Future as CCFuture
from enum import Enum
from functools import wraps
from types import MethodType

import json5

# math and image dependencies
import numpy as np

# microscope dependencies
import seaconfig as sc

# http server dependencies
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.routing import APIWebSocketRoute
from fastapi.staticfiles import StaticFiles

# for robust type safety at runtime
from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic.fields import FieldInfo
from seaconfig.acquisition import AcquisitionConfig

from seafront.config.basics import (
    ChannelConfig,
    ConfigItem,
    CriticalMachineConfig,
    GlobalConfigHandler,
    ImagingOrder,
    ServerConfig,
)
from seafront.config.handles import (
    CalibrationConfig,
    CameraConfig,
    ImagingConfig,
    LaserAutofocusConfig,
    ProtocolConfig,
)
from seafront.hardware.microscope import DisconnectError, HardwareLimits, Microscope
from seafront.hardware.mock_microscope import MockMicroscope
from seafront.logger import logger
from seafront.hardware.forbidden_areas import ForbiddenAreaList
from seafront.server.commands import (
    AcquisitionCommand,
    AcquisitionEstimate,
    AcquisitionMetaInformation,
    AcquisitionProgressStatus,
    AcquisitionStartResponse,
    AcquisitionStatus,
    AcquisitionStatusOut,
    AcquisitionStatusStage,
    AutofocusApproachTargetDisplacement,
    AutofocusLaserWarmup,
    AutofocusMeasureDisplacement,
    AutofocusSnap,
    AutofocusSnapResult,
    BaseCommand,
    BasicSuccessResponse,
    ChannelSnapProgressiveStatus,
    ChannelSnapSelectionResult,
    ChannelSnapshot,
    ChannelStreamBegin,
    ChannelStreamEnd,
    ConfigFetchResponse,
    ConfigFileInfo,
    ConfigListResponse,
    CoreCurrentState,
    EstablishHardwareConnection,
    HardwareCapabilitiesResponse,
    IlluminationEndAll,
    ImageAcquiredResponse,
    ImageStoreEntry,
    ImageStoreInfo,
    InternalErrorModel,
    ConflictErrorModel,
    LaserAutofocusCalibrate,
    LoadingPositionEnter,
    LoadingPositionLeave,
    MoveBy,
    MoveTo,
    MoveToWell,
    SitePosition,
    StreamingStartedResponse,
    error_internal,
    error_microscope_busy,
    positionIsForbidden,
)
from seafront.server.protocol import (
    AsyncThreadPool,
    ProtocolGenerator,
    build_ome_instrument,
    make_unique_acquisition_id,
)


# Custom exception for acquisition cancellation
class AcquisitionCancelledError(Exception):
    """Custom exception raised when an acquisition is cancelled by user request"""
    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail

# Set the working directory to the script's directory as reference for static file paths

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# create server enty point

app = FastAPI(debug=False)

# setup tags for /docs structure

class RouteTag(str, Enum):
    STATIC_FILES = "Static Files"
    ACTIONS = "Microscope Actions"
    ACQUISITION_CONTROLS = "Acquisition Controls"
    DOCUMENTATION = "Documentation"


openapi_tags = [
    {
        "name": RouteTag.STATIC_FILES.value,
        "description": "serve static files of all sorts",
    },
    {
        "name": RouteTag.ACTIONS.value,
        "description": "actions for the microscope to perform immediately",
    },
    {
        "name": RouteTag.ACQUISITION_CONTROLS.value,
        "description": "acquisition related controls and information i/o",
    },
    {
        "name": RouteTag.DOCUMENTATION.value,
        "description": "documentation on software and api",
    },
]

# register static file paths (for front end)

@app.get("/", tags=[RouteTag.STATIC_FILES.value])
async def index():
    return FileResponse("web-static/index.html")

app.mount("/css", StaticFiles(directory="web-static/css"), name="web-css")
app.mount("/src", StaticFiles(directory="web-static/src"), name="web-src")
app.mount("/resources", StaticFiles(directory="web-static/resources"), name="web-resources")
app.mount("/vendor", StaticFiles(directory="web-static/vendor"), name="web-vendor")

# Register handler so that, if you send SIGUSR1 to this process,
# it will print all thread backtraces to stderr.
faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)

# precompile regex for performance
name_validity_regex = re.compile(r"^[a-zA-Z0-9_\-\.]+$")
""" name_validity_regex only permits: lower case latin letter, upper case latin letters, digits, underscore, dash, dot """


class CustomRoute(BaseModel):
    handler: type[BaseCommand] | tp.Callable
    tags: list[str] = Field(default_factory=list)

    callback: (
        None
        | tp.Callable[[tp.Any, tp.Any], None]
        | tp.Callable[[tp.Any, tp.Any], tp.Coroutine[tp.Any, tp.Any, None]]
    ) = None

    require_hardware_lock:bool=True
    """
    not all commands require a hardware lock (i.e. do not strictly require hardware connection either)"

    e.g. streamend (just sets a flag)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)


custom_route_handlers: dict[str, CustomRoute] = {}


class SquidError(Exception):
    """
    msg can be any of:
      MSG_BUSY
    """

    MSG_BUSY = "already busy"

    def __init__(self, msg: str):
        super().__init__(msg)


def name_check(name: str) -> str | None:
    """
    check name for validity.

    returns None if valid, otherwise returns description of the error.

    used to check plate name and project name for validity.
    """
    if len(name) == 0:
        return "name must not be empty"

    if not name_validity_regex.match(name):
        # from name_validity_regex doc:
        return "name invalid. it only permits: lower case latin letter, upper case latin letters, digits, underscore, dash, dot"

    return None


def filename_check(name: str) -> str | None:
    """
    check filename for validity.

    returns None if valid, otherwise returns description of the error.

    used to check config filename for validity.
    """
    if len(name) == 0:
        return "name must not be empty"

    if not name_validity_regex.match(name):
        # from name_validity_regex doc:
        return "name invalid. it only permits: lower case latin letter, upper case latin letters, digits, underscore, dash, dot"

    return None


class Core:
    """application core, contains server capabilities and microcontroller interaction"""

    def __init__(self, selected_microscope: CriticalMachineConfig):

        def make_acquisition_event_loop():
            worker_loop = asyncio.new_event_loop()

            # make daemon so that it automatically terminates on program shutdown
            t = threading.Thread(target=worker_loop.run_forever, daemon=True)
            t.start()

            return worker_loop

        self.acqusition_eventloop = make_acquisition_event_loop()

        # Store microscope name for status reporting
        self.microscope_name = selected_microscope.microscope_name

        # Use the passed microscope configuration
        microscope_type = selected_microscope.microscope_type

        if microscope_type == "mock":
            logger.info("Creating mock microscope adapter")
            self.microscope: Microscope = MockMicroscope.make()
        else:
            logger.info("Creating SQUID microscope adapter")
            # Deferred import: squid.py requires hardware drivers that must be externally installed.
            # By importing here instead of at module level, mock mode works without drivers installed.
            from seafront.hardware.squid import SquidAdapter
            self.microscope = SquidAdapter.make()

        self.acquisition_map: dict[str, AcquisitionStatus] = {}
        """ map containing information on past and current acquisitions """

        self._hardware_limits_cache: HardwareLimits | None = None
        """ cached hardware limits to serve when microscope is busy """

        # Progressive channel snap state
        self._progressive_snap_callbacks: dict[str, tp.Callable[[ChannelSnapProgressiveStatus], None]] = {}
        """ callback functions for progressive channel snap status updates """

        # Acquisition error buffer
        self._last_acquisition_error: str | None = None
        """ stores the last acquisition error message for display in GUI """
        self._last_acquisition_error_timestamp: str | None = None
        """ stores the ISO timestamp of the last acquisition error """

        # set up routes to member functions

        # store request_models for re-use (works around issues with fastapi)
        request_models = {}

        # Check if command is allowed to run given current acquisition/streaming state
        def check_operation_allowed(allow_while_acquisition_is_running: bool, allow_while_streaming: bool) -> JSONResponse | None:
            """
            Verify that the operation is allowed given current microscope state.
            Returns a JSONResponse error if not allowed, None if allowed.
            """
            if (not allow_while_acquisition_is_running) and self.acquisition_is_running:
                return JSONResponse(
                    content={
                        "status": "error",
                        "message": "cannot run this command while acquisition is running",
                    },
                    status_code=400,
                )

            if (not allow_while_streaming) and (self.microscope.stream_callback is not None):
                return JSONResponse(
                    content={
                        "status": "error",
                        "message": "cannot run this command while live streaming is active",
                    },
                    status_code=400,
                )

            return None

        # Utility function to wrap the shared logic by including handlers for GET requests
        def route_wrapper(
            path: str,
            route: CustomRoute,
            methods: list[str] | None = None,
            allow_while_acquisition_is_running: bool = True,
            allow_while_streaming: bool = True,
            summary: str | None = None,
            **kwargs_static,
        ):
            if methods is None:
                methods = ["GET"]
            custom_route_handlers[path] = route

            target_func = route.handler

            async def callfunc(request_data):
                arg = None
                # Call the target function
                if inspect.isclass(target_func):
                    instance = target_func(**request_data)
                    arg = instance
                    if isinstance(instance, BaseCommand):
                        if route.require_hardware_lock:
                            try:
                                # ensure a connection is established, before running any other command
                                command_name = instance.__class__.__name__
                                with self.microscope.lock(blocking=False, reason=f"executing command: {command_name}") as microscope:
                                    if microscope is None:
                                        error_microscope_busy(self.microscope.get_lock_reasons())
                                    _ = await microscope.execute(EstablishHardwareConnection())
                                    result = await microscope.execute(instance)
                            except DisconnectError:
                                error_internal("hardware disconnected")
                        else:
                            _ = await self.microscope.execute(EstablishHardwareConnection())
                            result = await self.microscope.execute(instance)
                    else:
                        raise AttributeError(
                            f"Provided class {target_func} {type(target_func)=} is no BaseCommand"
                        )

                elif inspect.iscoroutinefunction(target_func):
                    arg = request_data
                    result = await target_func(**request_data)
                elif inspect.isfunction(target_func) or isinstance(target_func, MethodType):
                    arg = request_data
                    result = target_func(**request_data)
                else:
                    raise TypeError(f"Unsupported target_func type: {type(target_func)}")

                if route.callback is not None:
                    if inspect.iscoroutinefunction(route.callback):
                        await route.callback(arg, result)
                    else:
                        route.callback(arg, result)

                return result

            def get_return_type():
                """
                Determines the return type of the target function.
                If the target function is a class, retrieves the return type of the 'run()' method.
                """
                # Case 1: target_func is a coroutine function or a standard function/method
                if (
                    inspect.iscoroutinefunction(target_func)
                    or inspect.isfunction(target_func)
                    or isinstance(target_func, MethodType)
                ):
                    # print(f"returning {target_func.__name__} {inspect.signature(target_func).return_annotation}")
                    return inspect.signature(target_func).return_annotation

                if issubclass(target_func, BaseCommand):  # type:ignore
                    return target_func.__private_attributes__["_ReturnValue"].default  # type:ignore

                # Case 2: target_func is a class, get return type of the 'run()' method if it exists
                if inspect.isclass(target_func):
                    if hasattr(target_func, "run"):
                        return_type = inspect.signature(target_func.run).return_annotation
                        # print(f"returning {target_func.__name__}.run {return_type}")
                        return return_type

                # Default case: if none of the above matches
                logger.warning(f"{target_func=} has unknown return type")
                return tp.Any

            return_type = get_return_type()

            @wraps(target_func)
            async def handler_logic_get(**kwargs: tp.Any | None) -> return_type:  # type: ignore
                # Perform verification
                error_response = check_operation_allowed(allow_while_acquisition_is_running, allow_while_streaming)
                if error_response is not None:
                    return error_response

                request_data = kwargs.copy()
                request_data.update(kwargs_static)

                # Use asyncio.to_thread to run each request in its own system thread
                # This ensures RLock works properly with different thread IDs
                result = await asyncio.to_thread(asyncio.run, callfunc(request_data))
                return result

            # Dynamically create a Pydantic model for the POST request body if the target function has parameters
            model_fields = {}
            for key, value in inspect.signature(target_func).parameters.items():
                if key != "request" and value.annotation is not inspect._empty:
                    default_value = kwargs_static.get(key, value.default)
                    if default_value is inspect._empty:
                        model_fields[key] = (value.annotation, ...)
                    else:
                        model_fields[key] = (
                            value.annotation,
                            Field(default=default_value),
                        )

            RequestModel = None
            if model_fields:
                # Dynamically create the Pydantic model
                model_name = f"{target_func.__name__.capitalize()}RequestModel"
                RequestModel = request_models.get(model_name)
                if RequestModel is None:
                    RequestModel = create_model(model_name, **model_fields, __base__=BaseModel)
                    request_models[model_name] = RequestModel

                async def handler_logic_post(request_body: RequestModel):  # type:ignore
                    # Perform verification
                    error_response = check_operation_allowed(allow_while_acquisition_is_running, allow_while_streaming)
                    if error_response is not None:
                        return error_response

                    request_data = kwargs_static.copy()
                    if RequestModel and request_body:
                        request_body_as_toplevel_dict = {}
                        for key in request_body.dict(exclude_unset=True).keys():
                            request_body_as_toplevel_dict[key] = getattr(request_body, key)
                        request_data.update(request_body_as_toplevel_dict)

                    # microscope code expects rlock to function as expected, but fastapi serving two requests in parallel
                    # through async will technically run in the same thread, so rlock will function improperly.
                    # we start a new thread just to run this async code in it, to work around this issue.
                    logger.debug(f"about to start thread to generate answer {callfunc}")
                    result = await asyncio.to_thread(asyncio.run, callfunc(request_data))
                    logger.debug("answer thread done")
                    return result
            else:

                async def handler_logic_post():  # type:ignore
                    # Perform verification
                    error_response = check_operation_allowed(allow_while_acquisition_is_running, allow_while_streaming)
                    if error_response is not None:
                        return error_response

                    request_data = kwargs_static.copy()

                    # microscope code expects rlock to function as expected, but fastapi serving two requests in parallel
                    # through async will technically run in the same thread, so rlock will function improperly.
                    # we start a new thread just to run this async code in it, to work around this issue.
                    result = await asyncio.to_thread(asyncio.run, callfunc(request_data))
                    return result

            # copy annotation and fix return type
            handler_logic_post.__doc__ = target_func.__doc__
            handler_logic_post.__annotations__["return"] = return_type

            if summary is None and target_func.__doc__ is not None:
                docstring_lines = [
                    line.lstrip().rstrip() for line in target_func.__doc__.split("\n")
                ]
                docstring_lines = [line for line in docstring_lines if len(line) > 0]
                if len(docstring_lines) >= 1:
                    summary = docstring_lines[0]

            for m in methods:
                if m == "GET":
                    app.add_api_route(
                        path,
                        handler_logic_get,
                        methods=["GET"],
                        operation_id=path[1:].replace("/", ".") + ".get",
                        summary=summary,
                        responses={
                            409: {"model": ConflictErrorModel},
                            500: {"model": InternalErrorModel},
                        },
                        tags=route.tags,  # type:ignore
                    )
                if m == "POST":
                    app.add_api_route(
                        path,
                        handler_logic_post,
                        methods=["POST"],
                        operation_id=path[1:].replace("/", ".") + ".post",
                        summary=summary,
                        responses={
                            409: {"model": ConflictErrorModel},
                            500: {"model": InternalErrorModel},
                        },
                        tags=route.tags,  # type:ignore
                    )

        # Register URL rules requiring machine interaction
        route_wrapper(
            "/api/get_info/current_state",
            CustomRoute(handler=self.get_current_state),
            methods=["POST"],
        )

        # Register hardware capabilities route
        route_wrapper(
            "/api/get_features/hardware_capabilities",
            CustomRoute(handler=self.get_hardware_capabilities),
            methods=["POST"],
        )

        # Register machine defaults route
        route_wrapper(
            "/api/get_features/machine_defaults",
            CustomRoute(handler=self.get_machine_defaults),
            methods=["POST"],
        )

        async def _safe_send_json(ws: WebSocket, payload: tp.Any) -> None:
            try:
                await ws.send_json(payload)
            except Exception as exc:
                logger.debug(f"websocket send_json failed: {exc}")

        async def _safe_send_bytes(ws: WebSocket, payload: bytes) -> None:
            try:
                await ws.send_bytes(payload)
            except Exception as exc:
                logger.debug(f"websocket send_bytes failed: {exc}")

        @app.websocket("/ws/get_info/current_state")
        async def ws_get_info_current_state(ws: WebSocket):
            """
            get current state of the microscope

            whenever any message is sent to this websocket it returns a currentstate object.
            simple websocket wrapper around repeated calls to /api/get_info/current_state.
            """
            await ws.accept()
            try:
                while True:
                    # await message, but ignore its contents
                    message = await ws.receive()
                    msg_type = message.get("type")
                    if msg_type in {"websocket.disconnect", "websocket.close"}:
                        break

                    try:
                        current_state = await self.get_current_state()
                        await _safe_send_json(ws, current_state.model_dump())
                    except Exception as e:
                        # Handle hardware disconnects and other errors gracefully
                        error_msg = {
                            "error": "hardware_disconnect" if "disconnect" in str(e).lower() else "internal_error",
                            "message": str(e),
                            "timestamp": time.time()
                        }
                        await _safe_send_json(ws, error_msg)
                        # Don't close the connection - let client decide

            except WebSocketDisconnect:
                pass

        @app.websocket("/ws/get_info/acquired_image")
        async def getacquiredimage(ws: WebSocket):
            """
            get acquired image data

            interaction takes multiple stages:
             - user send channel handle to request image
             - (if there is no image data for this channel, returns empty object)
             - 1) sends object with image metadata {width:number,height:number,bit_depth:number}
             - 2) sends image data in bytes (monochrome, bytes per pixel indicated by bit_depth, row by row, top to bottom, left to right)
            """
            await ws.accept()
            try:
                while True:
                    channel_handle = await ws.receive_text()

                    img = self.latest_images.get(channel_handle)
                    if img is None:
                        await _safe_send_json(ws, {})
                    else:
                        # downsample image for preview
                        img_data = img._img

                        await _safe_send_json(ws,
                            {
                                "channel_handle": channel_handle,
                                "width": img_data.shape[1],
                                "height": img_data.shape[0],
                                "camera_bit_depth": img.bit_depth,
                                "bit_depth": img_data.dtype.itemsize * 8,
                            }
                        )

                        # await downsample factor
                        factor = int(await ws.receive_text())

                        img_bytes = np.ascontiguousarray(img_data[::factor, ::factor]).tobytes()

                        await _safe_send_bytes(ws, img_bytes)

            except WebSocketDisconnect:
                pass

        # Register URLs for immediate moves
        route_wrapper(
            "/api/action/move_by",
            CustomRoute(handler=MoveBy, tags=[RouteTag.ACTIONS.value]),
            methods=["POST"],
            allow_while_streaming=True,
        )
        route_wrapper(
            "/api/action/move_to",
            CustomRoute(handler=MoveTo, tags=[RouteTag.ACTIONS.value]),
            methods=["POST"],
            allow_while_streaming=True,
        )

        route_wrapper(
            "/api/action/machine_config_flush",
            CustomRoute(handler=self.machine_config_flush),
            methods=["POST"],
        )

        # Register URL for start_acquisition
        route_wrapper(
            "/api/acquisition/start",
            CustomRoute(
                handler=self.start_acquisition,
                tags=[RouteTag.ACQUISITION_CONTROLS.value],
            ),
            methods=["POST"],
            allow_while_streaming=False,
        )

        # Register URL for cancel_acquisition
        route_wrapper(
            "/api/acquisition/cancel",
            CustomRoute(
                handler=self.cancel_acquisition,
                tags=[RouteTag.ACQUISITION_CONTROLS.value],
            ),
            allow_while_acquisition_is_running=True,
            allow_while_streaming=False,
            methods=["POST"],
        )

        # Register URL for get_acquisition_status
        route_wrapper(
            "/api/acquisition/status",
            CustomRoute(
                handler=self.get_acquisition_status,
                tags=[RouteTag.ACQUISITION_CONTROLS.value],
            ),
            allow_while_acquisition_is_running=True,
            methods=["POST"],
        )

        # Register URL for get_acquisition_estimate
        route_wrapper(
            "/api/acquisition/estimate",
            CustomRoute(
                handler=self.get_acquisition_estimate,
                tags=[RouteTag.ACQUISITION_CONTROLS.value],
            ),
            methods=["POST"],
        )

        @app.websocket("/ws/acquisition/status")
        async def ws_acquisition_status(ws: WebSocket):
            await ws.accept()
            try:
                while True:
                    # await message, but ignore its contents
                    args = await ws.receive_json()
                    try:
                        status = await self.get_acquisition_status(**args)
                        await _safe_send_json(ws, status.model_dump())
                    except HTTPException as e:
                        # Send error as JSON instead of crashing the WebSocket
                        await _safe_send_json(ws, {"error": str(e.detail)})
                    except Exception as e:
                        logger.warning(f"Error getting acquisition status: {e}")
                        await _safe_send_json(ws, {"error": str(e)})
            except WebSocketDisconnect:
                pass

        @app.websocket("/ws/action/snap_selected_channels_progressive")
        async def ws_snap_selected_channels_progressive(ws: WebSocket):
            """
            Progressive channel snapping with real-time updates.
            
            Client sends AcquisitionConfig, server responds with status updates
            as each channel completes. Each completed channel image is immediately
            available via the normal image retrieval endpoints.
            """
            await ws.accept()
            callback_id = make_unique_acquisition_id()  # Define outside try block

            try:
                # Receive the configuration
                config_data = await ws.receive_json()
                config_file = sc.AcquisitionConfig(**config_data)

                # Register callback to send updates via WebSocket
                def send_status_update(status: ChannelSnapProgressiveStatus):
                    try:
                        # Create a task to send the WebSocket message
                        asyncio.create_task(_safe_send_json(ws, status.model_dump()))
                    except Exception as e:
                        logger.warning(f"Failed to send progressive snap status: {e}")

                self._progressive_snap_callbacks[callback_id] = send_status_update

                # Start progressive snapping
                try:
                    await self.start_progressive_channel_snap(config_file, callback_id)
                except Exception as e:
                    # Send error status if startup fails
                    send_status_update(ChannelSnapProgressiveStatus(
                        channel_handle="",
                        channel_name="",
                        status="error",
                        total_channels=0,
                        completed_channels=0,
                        message="Failed to start progressive snap",
                        error_detail=str(e)
                    ))

            except WebSocketDisconnect:
                pass  # Client disconnected
            except Exception as e:
                logger.error(f"Progressive snap WebSocket error: {e}")
            finally:
                # Always clean up callback
                if callback_id in self._progressive_snap_callbacks:
                    del self._progressive_snap_callbacks[callback_id]

        # Retrieve config list
        route_wrapper(
            "/api/acquisition/config_list",
            CustomRoute(handler=self.get_config_list, tags=[RouteTag.ACQUISITION_CONTROLS.value]),
            methods=["POST"],
        )

        # Fetch acquisition config
        route_wrapper(
            "/api/acquisition/config_fetch",
            CustomRoute(handler=self.config_fetch, tags=[RouteTag.ACQUISITION_CONTROLS.value]),
            methods=["POST"],
        )

        # Save/load config
        route_wrapper(
            "/api/acquisition/config_store",
            CustomRoute(handler=self.config_store, tags=[RouteTag.ACQUISITION_CONTROLS.value]),
            methods=["POST"],
        )

        # Move to well
        route_wrapper(
            "/api/action/move_to_well",
            CustomRoute(handler=MoveToWell, tags=[RouteTag.ACTIONS.value]),
            methods=["POST"],
            allow_while_streaming=True,
        )

        # Loading position enter/leave
        route_wrapper(
            "/api/action/enter_loading_position",
            CustomRoute(handler=LoadingPositionEnter, tags=[RouteTag.ACTIONS.value]),
            methods=["POST"],
            allow_while_acquisition_is_running=False,
        )
        route_wrapper(
            "/api/action/leave_loading_position",
            CustomRoute(handler=LoadingPositionLeave, tags=[RouteTag.ACTIONS.value]),
            methods=["POST"],
            allow_while_acquisition_is_running=False,
        )

        async def write_image_laseraf(cmd: AutofocusSnap, res: AutofocusSnapResult):
            "store new laser autofocus image"

            pixel_format = LaserAutofocusConfig.CAMERA_PIXEL_FORMAT.value_item.strvalue

            await self._store_new_image(
                img=res._img, pixel_format=pixel_format, channel_config=res._channel
            )

        async def write_image(cmd: ChannelSnapshot, res: ImageAcquiredResponse):
            "store new regular image"

            pixel_format = CameraConfig.MAIN_PIXEL_FORMAT.value_item.strvalue

            logger.debug(f"storing new image for {cmd.channel.handle}")

            await self._store_new_image(
                img=res._img, pixel_format=pixel_format, channel_config=cmd.channel
            )

        # Snap channel
        route_wrapper(
            "/api/action/snap_channel",
            CustomRoute(
                handler=ChannelSnapshot,
                tags=[RouteTag.ACTIONS.value],
                callback=write_image,
            ),
            methods=["POST"],
            allow_while_streaming=False,
        )

        # Snap selected channels (server orchestrates individual channel snaps)
        route_wrapper(
            "/api/action/snap_selected_channels",
            CustomRoute(
                handler=self.snap_selected_channels,
                tags=[RouteTag.ACTIONS.value],
            ),
            methods=["POST"],
            allow_while_streaming=False,
        )

        # Start streaming (i.e., acquire x images per sec, until stopped)
        self.image_store_threadpool = AsyncThreadPool()
        stream_info: dict[str, None | sc.AcquisitionChannelConfig] = {"channel": None}

        def handle_image(arg: np.ndarray | bool) -> bool:
            if isinstance(arg, bool):
                should_stop = arg
                if should_stop:
                    return True
            else:
                img = arg
                assert stream_info["channel"] is not None
                self.image_store_threadpool.run(
                    self._store_new_image(
                        img=img,
                        pixel_format=CameraConfig.MAIN_PIXEL_FORMAT.value_item.strvalue,
                        channel_config=stream_info["channel"],
                    )
                )
            return False

        def register_stream_begin(begin: ChannelStreamBegin, res: StreamingStartedResponse):
            # register callback on microscope
            with self.microscope.lock(reason=f"starting stream: {begin.channel.name}") as microscope:
                if microscope is None:
                    error_microscope_busy(self.microscope.get_lock_reasons())

                microscope.stream_callback = handle_image

                # store channel info, to be used inside the streaming callback to store the images in the server properly
                stream_info["channel"] = begin.channel

        route_wrapper(
            "/api/action/stream_channel_begin",
            CustomRoute(
                handler=ChannelStreamBegin,
                tags=[RouteTag.ACTIONS.value],
                callback=register_stream_begin,
            ),
            methods=["POST"],
            allow_while_acquisition_is_running=False,
            allow_while_streaming=False,
        )
        route_wrapper(
            "/api/action/stream_channel_end",
            CustomRoute(
                handler=ChannelStreamEnd,
                tags=[RouteTag.ACTIONS.value],
                require_hardware_lock=False,
            ),
            methods=["POST"],
        )

        # Laser autofocus system
        route_wrapper(
            "/api/action/snap_reflection_autofocus",
            CustomRoute(
                handler=AutofocusSnap,
                tags=[RouteTag.ACTIONS.value],
                callback=write_image_laseraf,
            ),
            methods=["POST"],
            allow_while_streaming=False,
        )
        route_wrapper(
            "/api/action/laser_autofocus_measure_displacement",
            CustomRoute(handler=AutofocusMeasureDisplacement, tags=[RouteTag.ACTIONS.value]),
            methods=["POST"],
            allow_while_streaming=False,
        )
        route_wrapper(
            "/api/action/laser_autofocus_move_to_target_offset",
            CustomRoute(
                handler=AutofocusApproachTargetDisplacement,
                tags=[RouteTag.ACTIONS.value],
            ),
            methods=["POST"],
            allow_while_streaming=False,
        )
        route_wrapper(
            "/api/action/laser_autofocus_calibrate",
            CustomRoute(handler=LaserAutofocusCalibrate, tags=[RouteTag.ACTIONS.value]),
            methods=["POST"],
            allow_while_streaming=False,
        )
        route_wrapper(
            "/api/action/laser_autofocus_warm_up_laser",
            CustomRoute(handler=AutofocusLaserWarmup, tags=[RouteTag.ACTIONS.value]),
            methods=["POST"],
            allow_while_streaming=False,
        )

        # Calibrate stage position
        route_wrapper(
            "/api/action/calibrate_stage_xy_here",
            CustomRoute(handler=self.calibrate_stage_xy_here, tags=[RouteTag.ACTIONS.value]),
            methods=["POST"],
            allow_while_acquisition_is_running=False,
        )

        # Turn off all illumination
        route_wrapper(
            "/api/action/turn_off_all_illumination",
            CustomRoute(handler=IlluminationEndAll, tags=[RouteTag.ACTIONS.value]),
            methods=["POST"],
        )

        # Establish hardware connection
        route_wrapper(
            "/api/action/establish_hardware_connection",
            CustomRoute(handler=EstablishHardwareConnection, tags=[RouteTag.ACTIONS.value]),
            methods=["POST"],
        )

        self.latest_images: dict[str, ImageStoreEntry] = {}
        "latest image acquired in each channel, key is channel handle"

        # self.acquisition_future: asyncio.Future[None] | None = None
        self.acquisition_future: CCFuture[None] | None = None

    def get_config_list(self) -> ConfigListResponse:
        """
        get list of existing config files

        these files are already stored on the machine, and can be loaded on request.
        """

        def map_filepath_to_info(c: path.Path) -> ConfigFileInfo | None:
            filename = c.name
            timestamp = None
            comment = None

            with c.open("r") as f:
                contents = json5.load(f)
                config = AcquisitionConfig(**contents)

                timestamp = config.timestamp
                comment = config.comment

                cell_line = config.cell_line

                plate_type = config.wellplate_type

                project = config.project_name
                platename = config.plate_name

            return ConfigFileInfo(
                filename=filename,
                project_name=project,
                plate_name=platename,
                timestamp=timestamp,
                comment=comment,
                cell_line=cell_line,
                plate_type=plate_type,
            )

        config_list_str = []
        for c in GlobalConfigHandler.get_config_list():
            next_config = None
            try:
                next_config = map_filepath_to_info(c)
            except:
                pass

            if next_config is None:
                continue

            config_list_str.append(next_config)

        ret = ConfigListResponse(
            configs=config_list_str,
        )

        return ret

    def config_fetch(self, config_file: str) -> ConfigFetchResponse:
        """
        get contents of specific config file

        retrieves the whole file. this return value can be submitted as config file with an acquisition start command.
        """

        filename = config_file

        filepath = None
        for c_path in GlobalConfigHandler.get_config_list():
            if c_path.name == filename:
                filepath = c_path

        if filepath is None:
            error_internal(detail=f"config file with name {filename} not found")

        with filepath.open("r") as f:
            config_json = json5.load(f)

        config = sc.AcquisitionConfig(**config_json)

        return ConfigFetchResponse(file=config)

    def config_store(
        self,
        config_file: sc.AcquisitionConfig,
        filename: str,
        overwrite_on_conflict: None | bool = None,
        comment: None | str = None,
    ) -> BasicSuccessResponse:
        """
        store this file locally

        stores the file on the microscope-connected computer for later retrieval.
        the optional comment provided is stored with the config file to quickly identify its purpose/function.
        """

        filename_check_issue = filename_check(filename)
        if filename_check_issue is not None:
            error_internal(
                detail=f"failed storing config because filename contains an issue: {filename_check_issue}"
            )

        config_file.timestamp = sc.datetime2str(dt.datetime.now(dt.UTC))

        # get machine config
        if config_file.machine_config is not None:
            GlobalConfigHandler.override(config_file.machine_config)

        if comment is not None:
            config_file.comment = comment

        try:
            GlobalConfigHandler.add_config(
                config_file, filename, overwrite_on_conflict=overwrite_on_conflict or False
            )
        except Exception as e:
            error_internal(detail=f"failed storing config to file because {e}")

        return BasicSuccessResponse()

    async def calibrate_stage_xy_here(
        self, plate_model_id: str = "revvity-384-6057800"
    ) -> BasicSuccessResponse:
        """
        set current xy position as reference

        set current xy position as top left corner of B2, which is used as reference to calculate all other positions on a plate.

        this WILL lead to hardware damage (down the line) if used improperly!
        due to the delay between improper calibration and actual cause of the damage, this function should be treat with appropriate care.
        """

        with self.microscope.lock(reason="calibrating stage XY position") as microscope:
            if microscope is None:
                error_microscope_busy(self.microscope.get_lock_reasons())

            if microscope.is_in_loading_position:
                error_internal(detail="now allowed while in loading position")

            _plates = [p for p in sc.Plates if p.Model_id == plate_model_id]
            if len(_plates) == 0:
                error_internal(f"{plate_model_id=} is not a known plate model")
            plate = _plates[0]

            current_pos = (await microscope.get_current_state()).stage_position

        # real/should position = measured/is position + calibrated offset
        # i.e. calibrated offset = real/should position - measured/is position
        old_x_mm=CalibrationConfig.OFFSET_X_MM.get_item().floatvalue
        old_y_mm=CalibrationConfig.OFFSET_Y_MM.get_item().floatvalue
        old_z_mm=CalibrationConfig.OFFSET_Z_MM.get_item().floatvalue
        # new offset is relative to existing calibration, hence add old to new offsets for new reference
        ref_x_mm = old_x_mm + plate.get_well_offset_x("B02") - current_pos.x_pos_mm
        ref_y_mm = old_y_mm + plate.get_well_offset_y("B02") - current_pos.y_pos_mm
        ref_z_mm = old_z_mm + 0.0  # TODO currently unused

        logger.debug(f"old plate calibration: {old_x_mm:.2} {old_y_mm:.2} {old_z_mm:.2}")
        logger.debug(f"new plate calibration: {ref_x_mm:.2} {ref_y_mm:.2} {ref_z_mm:.2}")

        # new_config_items:tp.Union[tp.Dict[str,ConfigItem
        GlobalConfigHandler.override(
            {
                CalibrationConfig.OFFSET_X_MM.value: ConfigItem(
                    name="ignored",
                    handle=CalibrationConfig.OFFSET_X_MM.value,
                    value_kind="float",
                    value=ref_x_mm,
                ),
                CalibrationConfig.OFFSET_Y_MM.value: ConfigItem(
                    name="ignored",
                    handle=CalibrationConfig.OFFSET_Y_MM.value,
                    value_kind="float",
                    value=ref_y_mm,
                ),
                CalibrationConfig.OFFSET_Z_MM.value: ConfigItem(
                    name="ignored",
                    handle=CalibrationConfig.OFFSET_Z_MM.value,
                    value_kind="float",
                    value=ref_z_mm,
                ),
            }
        )

        return BasicSuccessResponse()

    async def _store_new_image(
        self, img: np.ndarray, pixel_format: str, channel_config: sc.AcquisitionChannelConfig
    ) -> str:
        """
        store a new image, return the channel handle (into self.latest_images)

        note: this stores regular images, as well as autofocus images
        """

        try:
            adapter_state = await self.microscope.get_current_state()
        except DisconnectError:
            error_internal("hardware disconnect")

        # store new image
        new_image_store_entry = ImageStoreEntry(
            pixel_format=pixel_format,
            info=ImageStoreInfo(
                channel=channel_config,
                width_px=0,
                height_px=0,
                timestamp=time.time(),
                position=SitePosition(
                    well_name="",
                    site_x=0,
                    site_y=0,
                    site_z=0,
                    x_offset_mm=0,
                    y_offset_mm=0,
                    z_offset_mm=0,
                    position=adapter_state.stage_position,
                ),
            ),
        )
        new_image_store_entry._img = img
        logger.debug(f"stored new image for {channel_config.handle} at {time.time()}")
        self.latest_images[channel_config.handle] = new_image_store_entry

        return channel_config.name

    async def snap_selected_channels(self, config_file: sc.AcquisitionConfig) -> ChannelSnapSelectionResult:
        """
        Take snapshots of all selected channels with real-time image updates.

        The server orchestrates individual channel snapshots, allowing each image
        to be stored and displayed as soon as it's acquired, rather than waiting
        for all channels to complete.
        """
        pixel_format = CameraConfig.MAIN_PIXEL_FORMAT.value_item.strvalue

        try:
            with self.microscope.lock(blocking=False, reason="snapping all selected channels") as microscope:
                if microscope is None:
                    error_microscope_busy(self.microscope.get_lock_reasons())

                # Establish hardware connection
                await microscope.execute(EstablishHardwareConnection())

                # Validate all enabled channels BEFORE starting to image any of them
                for channel in config_file.channels:
                    if channel.enabled:
                        microscope.validate_channel_for_acquisition(channel)

                channel_handles: list[str] = []
                channel_images: dict[str, np.ndarray] = {}

                # Execute individual channel snapshots
                for channel in config_file.channels:
                    if not channel.enabled:
                        continue

                    # Create and execute individual snapshot command
                    cmd_snap = ChannelSnapshot(
                        channel=channel,
                        machine_config=config_file.machine_config or [],
                    )
                    res_snap = await microscope.execute(cmd_snap)

                    # Store image immediately (allows incremental UI updates)
                    await self._store_new_image(
                        img=res_snap._img,
                        pixel_format=pixel_format,
                        channel_config=channel
                    )

                    # Accumulate results
                    channel_images[channel.handle] = res_snap._img
                    channel_handles.append(channel.handle)

                logger.debug(f"server - took snapshot in {len(channel_handles)} channels")

                result = ChannelSnapSelectionResult(channel_handles=channel_handles)
                result._images = channel_images
                return result
        except DisconnectError:
            error_internal("hardware disconnected")

    async def get_current_state(self) -> CoreCurrentState:
        """
        get current state of the microscope

        for details see fields of return value
        """

        current_acquisition_id = None
        if self.acquisition_is_running:
            for acq_id, acquisition_status in self.acquisition_map.items():
                if acquisition_status.thread_is_running:
                    if current_acquisition_id is not None:
                        logger.warning(
                            f"more than one acquisition is running at a time?! {current_acquisition_id} and {acq_id}"
                        )

                    current_acquisition_id = acq_id

        try:
            microscope_adapter_state = await self.microscope.get_current_state()
        except DisconnectError:
            error_internal("hardware disconnect")

        # Check if streaming is currently active by looking at the stream callback
        is_streaming = self.microscope.stream_callback is not None

        # Check if microscope is busy by attempting a non-blocking lock
        is_busy = False
        busy_reasons: list[str] = []
        try:
            with self.microscope.lock(blocking=False, reason="checking microscope status") as microscope:
                if microscope is None:
                    is_busy = True
                    # Get the reasons why the lock is held
                    busy_reasons = self.microscope.get_lock_reasons()
        except Exception:
            is_busy = True
            busy_reasons = self.microscope.get_lock_reasons()

        return CoreCurrentState(
            adapter_state=microscope_adapter_state,
            latest_imgs={key: entry.info for key, entry in self.latest_images.items()},
            current_acquisition_id=current_acquisition_id,
            is_streaming=is_streaming,
            is_busy=is_busy,
            busy_reasons=busy_reasons,
            last_acquisition_error=self._last_acquisition_error,
            last_acquisition_error_timestamp=self._last_acquisition_error_timestamp,
            microscope_name=self.microscope_name,
        )

    async def cancel_acquisition(self, acquisition_id: str) -> BasicSuccessResponse:
        """
        cancel the ongoing acquisition
        """

        if acquisition_id not in self.acquisition_map:
            error_internal(detail="acquisition_id not found")

        acq = self.acquisition_map[acquisition_id]

        if acq.thread_is_running:
            if self.acquisition_future is None:
                acq.thread_is_running = False
            else:
                acq.thread_is_running = not self.acquisition_future.done()

        if not acq.thread_is_running:
            error_internal(detail=f"acquisition with id {acquisition_id} is not running")

        await acq.queue_in.put(AcquisitionCommand.CANCEL)

        return BasicSuccessResponse()

    @property
    def acquisition_is_running(self) -> bool:
        if self.acquisition_future is None:
            return False

        if self.acquisition_future.done():
            self.acquisition_future = None
            return False

        return True

    def machine_config_flush(self, machine_config: list[ConfigItem]) -> BasicSuccessResponse:
        GlobalConfigHandler.override(machine_config)

        return BasicSuccessResponse()

    def get_hardware_capabilities(self) -> HardwareCapabilitiesResponse:
        """
        Get hardware capabilities including real hardware limits from the microscope.
        """
        # Access global configuration
        g_dict = GlobalConfigHandler.get_dict()

        # Get channels JSON string and parse it
        channels_json = g_dict["imaging.channels"].strvalue
        channels_data = json5.loads(channels_json)

        if channels_data is None:
            raise ValueError("Parsed channels configuration is None - invalid JSON structure")

        if not isinstance(channels_data, list):
            raise ValueError("Parsed channels configuration must be a list")

        # Convert to ChannelConfig objects
        channel_configs = []
        for ch in channels_data:
            if not isinstance(ch, dict):
                raise ValueError("Each channel configuration must be a dictionary")
            channel_configs.append(ChannelConfig(**ch))

        # Convert ChannelConfig to AcquisitionChannelConfig with sensible defaults
        acquisition_channels = []
        for ch in channel_configs:
            # Use different default illumination based on channel type
            default_illum_perc = 20.0 if ch.handle.startswith('bfled') else 100.0

            acquisition_channels.append(sc.AcquisitionChannelConfig(
                name=ch.name,
                handle=ch.handle,
                illum_perc=default_illum_perc,   # Lower default for brightfield LEDs
                exposure_time_ms=5.0,           # Default exposure time
                analog_gain=0.0,                # Default analog gain
                z_offset_um=0.0,                # Default z offset
                num_z_planes=1,                 # Default single z plane
                delta_z_um=1.0,                 # Default z spacing
                filter_handle=None,             # No filter by default
                enabled=True                    # Enabled by default
            ))

        # Get real hardware limits from the microscope
        with self.microscope.lock(blocking=False, reason="querying hardware capabilities") as microscope:
            if microscope is None:
                # Microscope is busy - use cached limits if available
                if self._hardware_limits_cache is None:
                    error_internal(f"Internal error: hardware limits cache not initialized and microscope is busy: {self.microscope.get_lock_reasons()}")
                hardware_limits_obj = self._hardware_limits_cache
            else:
                # Get actual hardware limits from microscope and cache them
                hardware_limits_obj = microscope.get_hardware_limits()
                self._hardware_limits_cache = hardware_limits_obj

        return HardwareCapabilitiesResponse(
            wellplate_types=sc.Plates,
            main_camera_imaging_channels=acquisition_channels,
            hardware_limits=hardware_limits_obj.to_dict(),
        )

    def get_machine_defaults(self) -> list[ConfigItem]:
        """
        Get a list of all the low level machine settings with dynamic pixel format injection.

        These settings may be changed on the client side, for individual acquisitions
        (though clearly, this is highly advanced stuff, and may cause irreperable hardware damage!)
        """
        config_items = GlobalConfigHandler.get()

        # Let the microscope extend config with hardware-specific options
        try:
            self.microscope.extend_machine_config(config_items)
        except Exception as e:
            logger.debug(f"Could not extend machine config from microscope: {e}")
            # Continue with default options if hardware unavailable

        # Convert JSON5 strings to standard JSON for browser compatibility
        for item in config_items:
            if item.handle in ['channels', 'filters'] and item.value_kind == 'text':
                try:
                    # Parse JSON5 and re-serialize as standard JSON
                    parsed_data = json5.loads(item.strvalue)
                    item.value = json.dumps(parsed_data)
                except Exception as e:
                    # If parsing fails, leave the original value
                    print(f"Warning: Failed to convert {item.handle} from JSON5 to JSON: {e}")

        return config_items

    async def start_acquisition(
        self, config_file: sc.AcquisitionConfig
    ) -> AcquisitionStartResponse:
        """
        start an acquisition

        the acquisition is run in the background, i.e. this command returns after acquisition bas begun. see /api/acquisition/status for ongoing status of the acquisition.
        """

        # Clear any previous acquisition error
        self._last_acquisition_error = None
        self._last_acquisition_error_timestamp = None

        # Note: Protocol compatibility is validated in config_fetch, but we could add additional
        # validation here if protocols are submitted directly via API without going through config_fetch

        # check if microscope is even is connected
        with self.microscope.lock(reason=f"starting acquisition: {config_file.plate_name}") as microscope:
            if microscope is None:
                error_microscope_busy(self.microscope.get_lock_reasons())

            try:
                _ = await microscope.get_current_state()
            except DisconnectError:
                error_internal("device not connected")

            if microscope.is_in_loading_position:
                error_internal(detail="now allowed while in loading position")

        if self.acquisition_future is not None:
            if not self.acquisition_future.done():
                error_internal(detail="acquisition already running")
            else:
                self.acquisition_future = None

        # get machine config
        if config_file.machine_config is not None:
            GlobalConfigHandler.override(config_file.machine_config)

        g_config = GlobalConfigHandler.get_dict()

        acquisition_id = make_unique_acquisition_id()

        plate = config_file.wellplate_type

        if config_file.autofocus_enabled:
            laser_autofocus_is_calibrated_item = g_config.get(
                LaserAutofocusConfig.CALIBRATION_IS_CALIBRATED.value
            )

            laser_autofocus_is_calibrated = (
                laser_autofocus_is_calibrated_item is not None
                and laser_autofocus_is_calibrated_item.boolvalue
            )

            if not laser_autofocus_is_calibrated:
                error_internal(detail="laser autofocus is enabled, but not calibrated")

        queue_in = asyncio.Queue()
        queue_out = asyncio.Queue()
        acquisition_status = AcquisitionStatus(
            acquisition_id=acquisition_id,
            queue_in=queue_in,
            queue_out=queue_out,
            last_status=None,
            thread_is_running=False,
        )
        self.acquisition_map[acquisition_id] = acquisition_status

        def handle_q_in(q_in=queue_in):
            """if there is something in q_in, fetch it and handle it (e.g. terminae on cancel command)"""
            if not q_in.empty():
                q_in_item = q_in.get_nowait()
                if q_in_item == AcquisitionCommand.CANCEL:
                    raise AcquisitionCancelledError("acquisition cancelled")

                logger.warning(f"command unhandled: {q_in_item}")

        project_name_issue = name_check(config_file.project_name)
        if project_name_issue is not None:
            error_internal(detail=f"project name is not acceptable: {project_name_issue}")

        plate_name_issue = name_check(config_file.plate_name)
        if plate_name_issue is not None:
            error_internal(detail=f"plate name is not acceptable: {plate_name_issue}")

        # Get camera info for OME metadata from microscope
        camera_vendor = ""
        camera_model = ""
        camera_sn = ""
        try:
            if hasattr(self.microscope, 'main_camera') and self.microscope.main_camera is not None:  # type: ignore
                # main_camera is a Locked[Camera], need to access the camera object
                camera = self.microscope.main_camera.value  # type: ignore
                camera_vendor = getattr(camera, 'vendor_name', '')
                camera_model = getattr(camera, 'model_name', '')
                camera_sn = getattr(camera, 'sn', '')
        except Exception as e:
            logger.warning(f"Could not get camera info for OME metadata: {e}")

        # Extract objective magnification from configuration
        objective_magnification: int | None = None
        try:
            g_config = GlobalConfigHandler.get_dict()
            objective_config_item = g_config.get(CameraConfig.MAIN_OBJECTIVE.value)
            if objective_config_item:
                # Parse objective value (e.g., "20xolympus" -> 20)
                objective_value = objective_config_item.strvalue
                # Extract magnification from objective string
                import re
                mag_match = re.match(r'(\d+)x', objective_value)
                if mag_match:
                    objective_magnification = int(mag_match.group(1))
        except Exception as e:
            logger.debug(f"Could not extract objective magnification: {e}")

        # Build OME template with microscope-wide metadata
        ome_template = build_ome_instrument(
            microscope_name=self.microscope_name,
            camera_vendor=camera_vendor,
            camera_model=camera_model,
            camera_sn=camera_sn,
            hardware_channels=self.microscope.channels,
            objective_magnification=objective_magnification,
        )

        protocol = ProtocolGenerator(
            config_file=config_file,
            handle_q_in=handle_q_in,
            plate=plate,
            acquisition_status=acquisition_status,
            acquisition_id=acquisition_id,
            ome_template=ome_template,
            hardware_channels=self.microscope.channels,
        )

        if protocol.num_images_total == 0:
            error_detail = f"no images to acquire ({protocol.num_wells = },{protocol.num_sites = },{protocol.num_channels = },{protocol.num_channel_z_combinations = })"

            assert protocol.acquisition_status.last_status is not None
            protocol.acquisition_status.last_status.acquisition_status = (
                AcquisitionStatusStage.CRASHED
            )
            protocol.acquisition_status.last_status.message = error_detail

            error_internal(detail=error_detail)

        # Validate that no site positions fall within forbidden areas
        # This catches forbidden positions during preparation instead of during execution
        # Parse forbidden areas once and reuse for all positions to avoid repeated parsing
        g_config = GlobalConfigHandler.get_dict()
        forbidden_areas_entry = g_config.get(ProtocolConfig.FORBIDDEN_AREAS.value)
        forbidden_areas = None
        if forbidden_areas_entry is not None and isinstance(forbidden_areas_entry.value, str):
            data = json5.loads(forbidden_areas_entry.value)
            forbidden_areas = ForbiddenAreaList.model_validate({"areas": data})

        for position_info in protocol.iter_positions():
            site_x_mm, site_y_mm = position_info.physical_position

            # Check if this position is forbidden
            is_forbidden, error_message = positionIsForbidden(site_x_mm, site_y_mm, forbidden_areas=forbidden_areas)
            if is_forbidden:
                error_internal(detail=f"Site position in well {position_info.well.well_name} (site {position_info.site.col},{position_info.site.row}) is forbidden: {error_message}")

        # Validate that all enabled channels have valid configurations
        # This catches channel configuration issues during preparation instead of during execution
        for channel_info in protocol.iter_channels():
            self.microscope.validate_channel_for_acquisition(channel_info.channel)

        # this function internally locks the microscope
        async def run_acquisition(
            q_in: asyncio.Queue[AcquisitionCommand],
            q_out: asyncio.Queue[
                tp.Literal["disconnected"] | InternalErrorModel | AcquisitionStatusOut
            ],
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

            async def inner(
                q_in: asyncio.Queue[AcquisitionCommand],
                q_out: asyncio.Queue[
                    tp.Literal["disconnected"] | InternalErrorModel | AcquisitionStatusOut
                ],
            ):
                logger.debug("protocol - started. awaiting microscope lock.")

                with self.microscope.lock(reason="running acquisition protocol") as microscope:
                    if microscope is None:
                        error_microscope_busy(self.microscope.get_lock_reasons())

                    logger.debug("protocol - acquired microscope lock")

                    try:
                        # initiate generation
                        protocol_generator = protocol.generate()

                        # send none on first yield
                        result = None
                        # protocol generates None to indicate that protocol is finished
                        while (next_step := await protocol_generator.asend(result)) is not None:
                            logger.debug(f"protocol - next step {type(next_step)}")
                            if isinstance(next_step, str):
                                result = None
                            else:
                                result = None
                                try:
                                    result = await microscope.execute(next_step)
                                except DisconnectError as e:
                                    logger.debug(f"executing protocol step generated error {e}")
                                    await protocol_generator.athrow(e)
                                    # TODO what should we do here?! break?

                                if result is not None and isinstance(next_step, ChannelSnapshot):
                                    await self._store_new_image(
                                        img=result._img,
                                        pixel_format=CameraConfig.MAIN_PIXEL_FORMAT.value_item.strvalue,
                                        channel_config=next_step.channel,
                                    )

                        logger.debug("protocol done")

                        # finished regularly, set status accordingly (there must have been at least one image, so a status has been set)
                        assert acquisition_status.last_status is not None
                        acquisition_status.last_status.acquisition_status = (
                            AcquisitionStatusStage.COMPLETED
                        )

                    except AcquisitionCancelledError:
                        # User requested cancellation
                        assert acquisition_status.last_status is not None
                        acquisition_status.last_status.acquisition_status = (
                            AcquisitionStatusStage.CANCELLED
                        )
                    except HTTPException as e:
                        # Validation errors or other internal errors should crash the acquisition
                        assert acquisition_status.last_status is not None
                        acquisition_status.last_status.acquisition_status = (
                            AcquisitionStatusStage.CRASHED
                        )
                        acquisition_status.last_status.message = str(e.detail)

                        # Store error in buffer for GUI to display
                        self._last_acquisition_error = str(e.detail)
                        self._last_acquisition_error_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

                    except DisconnectError:
                        await q_out.put("disconnected")
                        logger.debug("microscope disconnected during acquisition - closing connections.")
                        microscope.close()

                        assert acquisition_status.last_status is not None
                        acquisition_status.last_status.acquisition_status = (
                            AcquisitionStatusStage.CRASHED
                        )
                        acquisition_status.last_status.message = "hardware disconnect"

                    except Exception as e:
                        logger.exception(f"error during acquisition {e}\n{traceback.format_exc()}")

                        full_error = traceback.format_exc()
                        await q_out.put(
                            InternalErrorModel(
                                detail=f"acquisition thread failed because {e!s}, more specifically: {full_error}"
                            )
                        )

                        if acquisition_status.last_status is not None:
                            acquisition_status.last_status.acquisition_status = (
                                AcquisitionStatusStage.CRASHED
                            )

                    finally:
                        # ensure no dangling image store task threads
                        protocol.image_store_pool.join()

            # wrap async execution in sync function
            def sync_inner(q_in, q_out):
                asyncio.run(inner(q_in, q_out))

            # then run sync in a real thread, but async awaitable
            # (an async task, which would behave the same in practice, does NOT work with RLock, which is essential!)
            await asyncio.to_thread(sync_inner, q_in, q_out)

            # indicate that this thread has stopped running (no matter the cause)
            acquisition_status.thread_is_running = False

            logger.debug("acquisition thread is done")
            return

        # create future object for acquisition that allows status polling.
        self.acquisition_future = asyncio.run_coroutine_threadsafe(
            run_acquisition(queue_in, queue_out),
            # run coroutine in dedicated acquisition event loop
            self.acqusition_eventloop,
        )

        acquisition_status.thread_is_running = True

        return AcquisitionStartResponse(acquisition_id=acquisition_id)

    async def get_acquisition_status(self, acquisition_id: str) -> AcquisitionStatusOut:
        """
        get status of an acquisition
        """

        acq_res = self.acquisition_map.get(acquisition_id)
        if acq_res is None:
            error_internal(detail="acquisition_id is invalid")

        if acq_res.last_status is None:
            error_internal(detail="no status available")
        else:
            return acq_res.last_status

    async def get_acquisition_estimate(self, config_file: sc.AcquisitionConfig) -> AcquisitionEstimate:
        """
        Get storage and time estimates for an acquisition without starting it.

        This endpoint allows the frontend to show a confirmation dialog with
        estimates before the user commits to starting an acquisition.
        """
        return self.microscope.estimate_acquisition(config_file)

    async def start_progressive_channel_snap(
        self, config_file: sc.AcquisitionConfig, callback_id: str
    ) -> BasicSuccessResponse:
        """
        Start progressive channel snapping with real-time status updates.
        
        Each channel is acquired sequentially, with status updates sent via callback
        as soon as each channel completes. This allows the UI to show results immediately
        instead of waiting for all channels to complete.
        """

        if callback_id not in self._progressive_snap_callbacks:
            error_internal(detail="no callback registered for progressive snap")

        callback = self._progressive_snap_callbacks[callback_id]

        # Get enabled channels
        enabled_channels = [c for c in config_file.channels if c.enabled]

        if len(enabled_channels) == 0:
            error_internal(detail="no channels selected")

        # Get imaging order from machine config and sort channels
        g_config = GlobalConfigHandler.get_dict()
        imaging_order = g_config.get(ImagingConfig.ORDER.value, "protocol_order")
        if isinstance(imaging_order, str):
            imaging_order_value = imaging_order
        else:
            imaging_order_value = imaging_order.strvalue if hasattr(imaging_order, 'strvalue') else "protocol_order"

        # Use microscope's sorting function to maintain consistent ordering logic
        with self.microscope.lock(reason="sorting channels for progressive snap") as microscope:
            if microscope is None:
                error_microscope_busy(self.microscope.get_lock_reasons())
            enabled_channels = microscope._sort_channels_by_imaging_order(enabled_channels, tp.cast(ImagingOrder, imaging_order_value))

        total_channels = len(enabled_channels)

        # Start background task for progressive snapping
        async def progressive_snap_task():
            completed_channels = 0

            try:
                for channel in enabled_channels:
                    # Send starting status
                    callback(ChannelSnapProgressiveStatus(
                        channel_handle=channel.handle,
                        channel_name=channel.name,
                        status="starting",
                        total_channels=total_channels,
                        completed_channels=completed_channels,
                        message=f"Starting acquisition for {channel.name}"
                    ))

                    try:
                        # Execute channel snapshot
                        with self.microscope.lock(reason=f"snapping channel: {channel.name}") as microscope:
                            if microscope is None:
                                error_microscope_busy(self.microscope.get_lock_reasons())

                            result = await microscope.execute(ChannelSnapshot(
                                channel=channel,
                                machine_config=config_file.machine_config or []
                            ))

                        # Store the image
                        g_dict = GlobalConfigHandler.get_dict()
                        pixel_format = g_dict[CameraConfig.MAIN_PIXEL_FORMAT.value].strvalue
                        await self._store_new_image(
                            img=result._img,
                            pixel_format=pixel_format,
                            channel_config=channel
                        )

                        completed_channels += 1

                        # Send completed status
                        callback(ChannelSnapProgressiveStatus(
                            channel_handle=channel.handle,
                            channel_name=channel.name,
                            status="completed",
                            total_channels=total_channels,
                            completed_channels=completed_channels,
                            message=f"Completed {channel.name} ({completed_channels}/{total_channels})"
                        ))

                    except Exception as e:
                        # Send error status
                        callback(ChannelSnapProgressiveStatus(
                            channel_handle=channel.handle,
                            channel_name=channel.name,
                            status="error",
                            total_channels=total_channels,
                            completed_channels=completed_channels,
                            message=f"Error acquiring {channel.name}",
                            error_detail=str(e)
                        ))
                        break

                # Send finished status
                callback(ChannelSnapProgressiveStatus(
                    channel_handle="",
                    channel_name="",
                    status="finished",
                    total_channels=total_channels,
                    completed_channels=completed_channels,
                    message=f"All channels complete ({completed_channels}/{total_channels})"
                ))

            except Exception as e:
                # Send error status for unexpected errors
                callback(ChannelSnapProgressiveStatus(
                    channel_handle="",
                    channel_name="",
                    status="error",
                    total_channels=total_channels,
                    completed_channels=completed_channels,
                    message="Unexpected error during progressive snap",
                    error_detail=str(e)
                ))
            finally:
                # Clean up callback
                if callback_id in self._progressive_snap_callbacks:
                    del self._progressive_snap_callbacks[callback_id]

        # Start the task in background
        asyncio.create_task(progressive_snap_task())

        return BasicSuccessResponse()

    def close(self):
        logger.debug("shutdown - closing microscope")
        try:
            with self.microscope.lock(blocking=True, reason="shutting down microscope") as microscope:
                assert microscope is not None

                microscope.close()
        except:
            pass

        logger.debug("shutdown - storing global config")
        GlobalConfigHandler.store()

        logger.debug("shutdown - closing image store threadpool")
        self.image_store_threadpool.join()

        logger.debug("shutdown - done")


# handle validation errors with ability to print to terminal for debugging
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.debug(
        f"Validation error at {request.url}: {json.dumps(exc.errors(), indent=2, ensure_ascii=False)}"
    )

    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({"detail": exc.errors()}),
    )


# -- fix issue in tp.Optional annotation with pydantic
# (from https://github.com/fastapi/fastapi/pull/9873#issuecomment-1997105091)

type ApiSchema = dict[str, ApiSchema] | list[ApiSchema] | tp.Any


def handle_anyof_nullable(schema: ApiSchema):
    """Recursively modifies the schema to handle anyOf with null for OpenAPI 3.0 compatibility."""

    if isinstance(schema, dict):
        for key, value in list(schema.items()):  # Iterate over a copy to avoid modification errors
            if key == "anyOf" and isinstance(value, list):
                non_null_types = [item for item in value if item.get("type") != "null"]  # type: ignore
                if len(value) > len(non_null_types):  # Found 'null' in anyOf
                    if len(non_null_types) == 1:
                        # Replace with non-null type
                        schema.update(non_null_types[0])  # type: ignore
                        schema["nullable"] = True
                        del schema[key]  # Remove anyOf
            else:
                # if value is a schema:
                if isinstance(value, list | dict):
                    handle_anyof_nullable(value)

    elif isinstance(schema, list):
        # this can lead to recursion under certain circumstances.. ???
        for item in schema:
            handle_anyof_nullable(item)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = {
        "openapi": "3.0.1",
        "info": {"title": "seafront api", "version": "0.2.0"},
        "paths": {},
        "description": "Seafront OpenAPI schema",
        "tags": openapi_tags,
    }

    def register_pydantic_schema(t):
        assert issubclass(t, BaseModel)

        for field in t.model_fields.values():
            # register field types
            type_to_schema(field)

        # make schema as json
        model_schema = t.model_json_schema(mode="serialization")

        # pydantic uses $defs to reference the type of internal fields, but openapi uses components/schemas
        # which we swap via stringify-replace-reparse to write as little code as possible to do what is
        # otherwise recursion
        schema_str = json.dumps(model_schema)
        schema_str = schema_str.replace("#/$defs/", "#/components/schemas/")
        model_schema = json5.loads(schema_str)

        # the json schema has a top level field called defs, which contains internal fields, which we
        # embed into the openapi schema here (the path replacement is separate from this)
        defs = model_schema.pop("$defs", {}) # type: ignore
        openapi_schema.setdefault("components", {}).setdefault("schemas", {}).update(defs)

        # unsure if this works with the new code (written for 0.1, untested in 0.2)
        handle_anyof_nullable(model_schema)

        # finally, add the actual model we have handled to the openapi schema
        openapi_schema.setdefault("components", {}).setdefault("schemas", {})[t.__name__] = (
            model_schema
        )

        return {"$ref": f"#/components/schemas/{t.__name__}"}

    def type_to_schema(t):
        if isinstance(t, FieldInfo):
            t = t.annotation

        if t is int:
            return {"type": "integer"}
        elif t is float:
            return {"type": "number", "format": "float"}
        elif t is bool:
            return {"type": "boolean"}
        elif t is dict:
            return {"type": "object"}
        elif inspect.isclass(t) and issubclass(t, BaseModel):  # type:ignore
            return register_pydantic_schema(t)  # type:ignore
        else:
            origin = tp.get_origin(t)
            if origin in (list,):
                item_type = tp.get_args(t)[0]
                return {"type": "array", "items": type_to_schema(item_type)}
            return {"type": "string"}

    for route in app.routes:
        if not hasattr(route, "endpoint"):
            continue

        route_path: str = route.path  # type:ignore

        tags: list[str]
        if hasattr(route, "tags"):
            tags = route.tags  # type:ignore
        else:
            tags = []

        responses = {
            "200": {
                "description": "Success",
                # actual content type is filled in during return type annotation inspection below
                "content": {"application/json": {"schema": None}},
            },
            "409": {
                "description": "conflict (e.g., microscope busy)",
                "content": {
                    "application/json": {"schema": register_pydantic_schema(ConflictErrorModel)}
                },
            },
            "500": {
                "description": "any failure mode",
                "content": {
                    "application/json": {"schema": register_pydantic_schema(InternalErrorModel)}
                },
            },
        }

        parameters = []

        endpoint = route.endpoint  # type:ignore
        if customroute := custom_route_handlers.get(route_path):
            if inspect.isclass(customroute.handler) and issubclass(
                customroute.handler, BaseCommand
            ):  # type:ignore
                assert issubclass(customroute.handler, BaseModel), (
                    f"{customroute.handler.__name__} does not inherit from basemodel, even though it inherits from basecommand"
                )

                # register
                type_to_schema(customroute.handler)

                responses["200"]["content"]["application/json"]["schema"] = type_to_schema(
                    customroute.handler.__private_attributes__["_ReturnValue"].default
                )  # type:ignore

                for name, field in customroute.handler.model_fields.items():
                    parameters.append(
                        {
                            "name": name,
                            "in": "query",
                            "required": field.is_required(),
                            "schema": type_to_schema(field),
                        }
                    )

        if responses["200"]["content"]["application/json"]["schema"] is None:
            if customroute := custom_route_handlers.get(route_path):
                endpoint = customroute.handler

            sig = inspect.signature(endpoint)
            hints = tp.get_type_hints(endpoint)

            for name, param in sig.parameters.items():
                if name in {"request", "background_tasks"}:
                    continue

                if (
                    inspect.isclass(endpoint)
                    and issubclass(endpoint, BaseModel)
                    and (
                        (not endpoint.model_fields[name].repr)
                        or (endpoint.model_fields[name].exclude)
                    )
                ):
                    continue

                param_schema = {
                    "name": name,
                    "in": "query",
                    "required": param.default is inspect.Parameter.empty,
                }

                if name in hints:
                    param_schema["schema"] = type_to_schema(hints[name])

                parameters.append(param_schema)

            ret = sig.return_annotation
            if ret is not inspect.Signature.empty:
                responses["200"]["content"]["application/json"]["schema"] = type_to_schema(ret)

        doc = endpoint.__doc__ or ""
        doc_lines = [docline.strip() for docline in doc.splitlines() if docline.strip()]
        summary = doc_lines[0] if doc_lines else ""
        description = "\n".join(doc_lines[1:]) if len(doc_lines) > 1 else ""

        if isinstance(route, APIWebSocketRoute):
            method = "get"
            responses["101"] = {"description": "switch protocol (to websocket)"}
        else:
            method = next(iter(route.methods)).lower()  # type:ignore

        if route_path not in openapi_schema["paths"]:
            openapi_schema["paths"][route_path] = {}

        if (
            route_path in ("/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect")
            and len(tags) == 0
        ):
            tags = [RouteTag.DOCUMENTATION.value]

        openapi_schema["paths"][route_path][method] = {
            "summary": summary,
            "description": description,
            "parameters": parameters,
            "responses": responses,
            "tags": tags,
        }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# -- end fix


# --- websocket compression disabled via uvicorn configuration
# compression time is unpredictable:
#    takes 70ms to send an all-white image, and 1400ms (twenty times as long !!!!) for all-black images, which are not a rare occurence in practice

# --- begin allow cross origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- end allow cross origin requests


@logger.catch
def main():
    # Load server config to get available microscopes and port for help message
    with GlobalConfigHandler.home_config().open("r") as f:
        server_config = ServerConfig(**json5.load(f))

    available_microscopes = [m.microscope_name for m in server_config.microscopes]
    microscope_list = ", ".join(f'"{name}"' for name in available_microscopes)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Seafront Microscope Server")
    parser.add_argument(
        "--microscope",
        type=str,
        help=f"Name of microscope configuration to use. Available: {microscope_list} (defaults to first)"
    )
    parser.add_argument(
        "--port",
        type=int,
        help=f"Port to run the server on (default: {server_config.port} from config)"
    )
    args = parser.parse_args()

    # Override port from config if specified on command line
    if args.port:
        server_config.port = args.port

    if len(server_config.microscopes) == 0:
        logger.critical("No microscope configurations found in config file")
        return

    # Determine which microscope to use
    selected_microscope = None
    if args.microscope:
        # Find microscope by name
        matching_microscopes = [m for m in server_config.microscopes if m.microscope_name == args.microscope]
        if len(matching_microscopes) == 0:
            available_names = [m.microscope_name for m in server_config.microscopes]
            logger.critical(f"Microscope '{args.microscope}' not found. Available: {available_names}")
            return
        selected_microscope = matching_microscopes[0]
    else:
        # Default to first microscope
        selected_microscope = server_config.microscopes[0]

    # Log startup information
    logger.info(f"🔬 Selected microscope: {selected_microscope.microscope_name}")
    logger.info(f"📷 Main camera: {selected_microscope.main_camera_id} (driver: {selected_microscope.main_camera_driver})")
    if selected_microscope.laser_autofocus_available == "yes" and selected_microscope.laser_autofocus_camera_id:
        logger.info(f"🎯 Autofocus camera: {selected_microscope.laser_autofocus_camera_id} (driver: {selected_microscope.laser_autofocus_camera_driver})")
    logger.info(f"🌐 Server port: {server_config.port}")

    logger.info(f"initializing core server with microscope: {selected_microscope.microscope_name}")

    # Initialize global config with selected microscope
    GlobalConfigHandler.reset(selected_microscope.microscope_name)

    # Check for default protocol file
    default_protocol_file = GlobalConfigHandler.home_acquisition_config_dir() / "default.json"

    if not default_protocol_file.exists():
        logger.critical(f"Default protocol file not found: {default_protocol_file}")
        logger.critical(f"Please run: uv run python scripts/generate_default_protocol.py --microscope \"{selected_microscope.microscope_name}\"")
        return

    logger.info(f"✓ Default protocol found: {default_protocol_file}")

    core = Core(selected_microscope)

    # Define hardware connection function (but don't start it yet)
    def establish_hardware_connection():
        import urllib.error
        import urllib.request

        try:
            server_base_url=f"http://127.0.0.1:{server_config.port}"

            # Wait for server to be ready by polling a simple endpoint
            server_ready = False
            max_attempts = 30  # 30 seconds max wait
            for _ in range(max_attempts):
                try:
                    with urllib.request.urlopen(server_base_url+"/", timeout=0.5) as response:
                        if response.status == 200:
                            server_ready = True
                            break
                except (TimeoutError, urllib.error.URLError):
                    pass
                time.sleep(0.5)

            if not server_ready:
                logger.warning("server did not become ready within 30 seconds, skipping hardware connection")
                return

            logger.info("establishing hardware connection at startup")
            req = urllib.request.Request(
                server_base_url + "/api/action/establish_hardware_connection",
                data=b'{}',
                headers={'Content-Type': 'application/json'},
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                if response.status == 200:
                    logger.info("hardware connection established")

                    # Initialize hardware limits cache
                    try:
                        logger.info("initializing hardware limits cache")
                        req_limits = urllib.request.Request(
                            server_base_url + "/api/get_features/hardware_capabilities",
                            data=b'{}',
                            headers={'Content-Type': 'application/json'},
                            method='POST'
                        )

                        with urllib.request.urlopen(req_limits, timeout=30) as limits_response:
                            if limits_response.status == 200:
                                logger.info("hardware limits cache initialized")
                            else:
                                logger.warning(f"hardware limits cache initialization failed: {limits_response.status}")
                    except Exception as e_limits:
                        logger.warning(f"failed to initialize hardware limits cache: {e_limits}")
                else:
                    logger.warning(f"hardware connection failed: {response.status}")
        except urllib.error.HTTPError as e:
            if e.code == 503:
                # Device is already in use by another process - must exit
                logger.critical(f"hardware connection failed: {e}")
                logger.critical("A hardware device is already in use by another process.")
                logger.critical("Shutting down server...")
                # Send SIGINT to gracefully shut down uvicorn
                os.kill(os.getpid(), signal.SIGINT)
            elif e.code == 500:
                # General hardware error - log but don't exit (user can retry via UI)
                logger.warning(f"hardware connection failed: {e}")
                logger.warning("Hardware connection could not be established. You may retry via the UI.")
            else:
                logger.warning(f"failed to establish hardware connection at startup: {e}")
        except Exception as e:
            logger.warning(f"failed to establish hardware connection at startup: {e}")

    logger.info("starting http server")
    logger.info("Starting Uvicorn server")
    logger.info("🧵 Request handlers run via asyncio.to_thread for real threading!")

    # Test if port is available before starting uvicorn
    try:
        import socket
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        test_socket.bind(('127.0.0.1', server_config.port))
        test_socket.close()
    except OSError as e:
        if e.errno == 98:  # Address already in use
            logger.critical(f"Failed to bind to port {server_config.port}. Port is already in use by another process.")
            logger.critical("Please ensure no other instance of Seafront is running, or change the port in your configuration.")
            logger.critical("Exiting...")
            return
        else:
            logger.critical(f"Failed to test port {server_config.port}: {e}")
            logger.critical("Exiting...")
            return

    try:
        # Start hardware connection thread only after we know the port is available
        connection_thread = threading.Thread(target=establish_hardware_connection, daemon=True)
        connection_thread.start()

        # Start uvicorn server
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=server_config.port,
            ws_per_message_deflate=False
        )
        logger.info("http server initialised")

    except Exception as e:
        # This should now rarely be reached due to pre-flight port check
        logger.critical(f"Unexpected server error: {e}")
        logger.critical("Exiting...")
        return
        
    finally:

        logger.info("shutting down")

        core.close()

        logger.info("shutdown complete")


if __name__ == "__main__":
    main()
