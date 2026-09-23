"""
The Core: everything the server can do to the microscope, with no HTTP in sight.

A Core owns the hardware adapter, the single worker that serialises access to it, the device
lease above that worker, and the bookkeeping an acquisition needs. Its methods are the whole
vocabulary of the machine; `server/api/` is the only thing that turns requests into calls on
them, and nothing here knows that HTTP exists.
"""

import asyncio
import datetime as dt
import pathlib as path
import re
import time
import threading
import typing as tp
import uuid
from concurrent.futures import Future as CCFuture

import json5
import numpy as np
import seaconfig as sc
from fastapi import HTTPException
from pydantic import ValidationError
from seaconfig.acquisition import AcquisitionConfig

from seafront.config.basics import (
    ChannelConfig,
    ConfigItem,
    FilterConfig,
    GlobalConfigHandler,
    ImagingOrder,
    MicroscopeConfig,
)
from seafront.config.handles import (
    CalibrationConfig,
    CameraConfig,
    FilterWheelConfig,
    ImagingConfig,
    LaserAutofocusConfig,
    ProtocolConfig,
    SystemConfig,
)
from seafront.config.registry import ConfigRegistry
from seafront.debug_introspect import register_signal_dumper
from seafront.hardware.camera.camera import bits_for_pixel_format
from seafront.hardware.forbidden_areas import ForbiddenArea, ForbiddenAreaList
from seafront.hardware.health import Condition, HardwareCloseError, describe
from seafront.hardware.microscope import (
    DisconnectError,
    HardwareLimits,
    Microscope,
    OperationCancelledError,
)
from seafront.hardware.mock_microscope import MockMicroscope
from seafront.logger import logger
from seafront.server.commands import (
    AcquisitionEstimate,
    AcquisitionStartResponse,
    AcquisitionStatus,
    AcquisitionStatusOut,
    AcquisitionStatusStage,
    AutofocusSnapResult,
    BaseCommand,
    BasicSuccessResponse,
    HealthResponse,
    ChannelSnapProgressiveStatus,
    ChannelSnapSelectionResult,
    ChannelSnapshot,
    CapturedImageMetadata,
    ConfigFetchResponse,
    ConfigFileInfo,
    ConfigListResponse,
    UnreadableConfigFile,
    CoreCurrentState,
    EstablishHardwareConnection,
    HardwareCapabilitiesResponse,
    ImageAcquiredResponse,
    ImageStoreEntry,
    ImageStoreInfo,
    LockAcquireResponse,
    LockStatusResponse,
    MachineConfigFlushResponse,
    SitePosition,
    error_device_locked,
    error_internal,
    error_microscope_busy,
    positionIsForbidden,
)
from seafront.server.acquisitions import AcquisitionRegistry
from seafront.storage import InsufficientStorageError, StorageFailedError, check_capacity
from seafront.server.lock import DeviceLock, LockDecision, LockReason
from seafront.server.protocol import (
    AsyncThreadPool,
    ProtocolGenerator,
    build_ome_instrument,
    make_unique_acquisition_id,
)
from seafront.server.worker import MicroscopeWorker

ACQUISITION_LABEL_PREFIX = "acquisition:"
"""
how an acquisition job is labelled on the worker. The label is what status reports as the
busy reason AND what it reports as the running acquisition, so both come from one read and
cannot disagree about whether a run is in progress.
"""


name_validity_regex = re.compile(r"^[a-zA-Z0-9_\-\.]+$")
""" only permits: lower case latin letters, upper case latin letters, digits, underscore, dash, dot """


MAX_NAME_LENGTH = 255
"""
the longest a single path component may be

ext4, xfs, apfs and ntfs all stop at 255 bytes. A longer name is not a naming preference, it
is an OSError on the first write - which for an acquisition means one that has already been
accepted, scheduled and reported as running.
"""

DOT_NAMES = {".", ".."}
"""
the two names that are instructions to the filesystem rather than names

A project name is a path component on its own (`<base>/<project>/<plate>_<timestamp>`), and
the disk sink creates it with mkdir(parents=True). `..` therefore writes outside the output
directory the operator configured. No accepted name can contain a separator, so this is the
whole of the exposure: one level, and only via these two spellings.
"""


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

    if name in DOT_NAMES:
        return f"name must not be '{name}': it names a directory rather than being one"

    if len(name.encode()) > MAX_NAME_LENGTH:
        return f"name must be at most {MAX_NAME_LENGTH} bytes, got {len(name.encode())}"

    return None


def filename_check(name: str) -> str | None:
    """
    check filename for validity.

    returns None if valid, otherwise returns description of the error.

    used to check config filename for validity. Same rules as name_check: a stored protocol
    filename is a path component under the config directory, so the dot names walk out of it
    the same way a project name does.
    """
    return name_check(name)


class AcquisitionCancelledError(Exception):
    """raised when an acquisition is cancelled by user request"""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


class SquidError(Exception):
    """
    msg can be any of:
      MSG_BUSY
    """

    MSG_BUSY = "already busy"

    def __init__(self, msg: str):
        super().__init__(msg)


class Core:
    """application core, contains server capabilities and microcontroller interaction"""

    def __init__(self, selected_microscope: MicroscopeConfig):
        self.microscope_name: str = selected_microscope.get(
            SystemConfig.MICROSCOPE_NAME.value, "unknown"
        )

        self.last_request_time: float | None = None
        "when the last request was served, for health monitoring"

        microscope_type = selected_microscope.get(SystemConfig.MICROSCOPE_TYPE.value, "squid")
        if microscope_type == "mock":
            logger.info("Creating mock microscope adapter")
            self.microscope: Microscope = MockMicroscope.make()
        else:
            logger.info("Creating SQUID microscope adapter")
            # Deferred import: squid.py requires hardware drivers that must be externally
            # installed. Importing here instead of at module level keeps mock mode working
            # without them.
            from seafront.hardware.squid import SquidAdapter

            self.microscope = SquidAdapter.make()

        # Single-owner hardware worker (reject-on-busy).
        self.worker = MicroscopeWorker(self.microscope)
        self.worker.start()

        # Streamed frames go straight into the image store. Installed once, here: the
        # adapter's stream state decides whether frames flow, so there is no callback to
        # install and remove around each stream (and no window where the camera is running
        # but nobody is listening).
        self.microscope.set_frame_sink(self._handle_streamed_image)

        # Client-held lease above the worker: one client at a time across requests. Routes
        # registered with a lease-requiring policy refuse 423 while another client holds it.
        # "In use" is a command on the worker OR a running camera stream - a stream hands the
        # worker slot straight back, so watching the worker alone would let the lease lapse
        # under a live camera.
        self.lock = DeviceLock(
            device_in_use=lambda: self.worker.is_busy or self.microscope.is_streaming
        )

        self.acquisitions = AcquisitionRegistry()
        "the running acquisition and the most recent finished ones"

        self.acquisition_future: CCFuture[None] | None = None

        self.latest_images: dict[str, ImageStoreEntry] = {}
        "latest image acquired in each channel, key is channel handle"

        self.image_store_threadpool = AsyncThreadPool()

        self._hardware_limits_cache: HardwareLimits | None = None
        """ cached hardware limits to serve when microscope is busy """

        self._progressive_snap_callbacks: dict[
            str, tp.Callable[[ChannelSnapProgressiveStatus], None]
        ] = {}
        """ callback functions for progressive channel snap status updates """

        self._last_acquisition_error: str | None = None
        """ stores the last acquisition error message for display in GUI """
        self._last_acquisition_error_timestamp: str | None = None
        """ stores the ISO timestamp of the last acquisition error """

        # Install SIGUSR2 -> timestamped rich stack dump (thread stacks + lock reasons +
        # asyncio tasks). Complements the SIGUSR1 faulthandler dump registered in server/app.py.
        # See seafront/debug_introspect.py for how to use both.
        try:
            register_signal_dumper(
                self.microscope,
                GlobalConfigHandler.home() / "logs",
                extra_loops=[],
            )
        except Exception as e:
            logger.warning(f"could not install SIGUSR2 stack dumper: {e}")

    # -- what the api layer calls into -------------------------------------------------------

    def health_report(self, *, time_since_last_request_s: float | None, now: float) -> HealthResponse:
        "what /api/health answers; reads the health board, never the hardware"
        hardware = self.microscope.health()
        stalls = self.microscope.stalls()

        down: list[str] = [f"hardware stalled: {stall.describe(now)}" for stall in stalls]
        degraded: list[str] = []
        for device in hardware:
            fault = f": {device.last_fault.message}" if device.last_fault is not None else ""
            match device.condition:
                case Condition.OK:
                    pass
                case Condition.UNRESPONSIVE:
                    degraded.append(f"{device.device} is not answering{fault}")
                case Condition.DISCONNECTED:
                    down.append(f"{device.device} is disconnected{fault}")
                case Condition.NOT_CONNECTED:
                    down.append(f"{device.device} is not connected")

        return HealthResponse(
            status="down" if down else "degraded" if degraded else "ok",
            reasons=down + degraded,
            hardware=hardware,
            stalls=stalls,
            last_acquisition_error=self._last_acquisition_error,
            last_acquisition_error_timestamp=self._last_acquisition_error_timestamp,
            microscope_name=self.microscope_name,
            time_since_last_request_s=time_since_last_request_s,
            server_time=now,
        )

    def mark_request_served(self) -> None:
        "record that a request was answered, for /api/health"
        self.last_request_time = time.time()

    async def run[T](self, command: BaseCommand[T]) -> T:
        """
        Run one hardware command and return its result.

        The worker is claimed once for the whole command: submit() connects and runs it inside
        a single hold, and answers None when something else is already in flight
        (reject-on-busy) - which is the 409 a caller sees.
        """
        submitted = self.worker.submit(command)
        if submitted is None:
            error_microscope_busy(self._busy_reasons())

        try:
            return await asyncio.wrap_future(submitted[0])
        except DisconnectError:
            error_internal("hardware disconnected")
        except OperationCancelledError:
            error_internal("operation cancelled")

    async def store_snapped_image(
        self, cmd: ChannelSnapshot, res: ImageAcquiredResponse
    ) -> None:
        "store the image a snap_channel produced"
        pixel_format = ConfigRegistry.get(CameraConfig.MAIN_PIXEL_FORMAT).strvalue
        logger.debug(f"storing new image for {cmd.channel.handle}")
        await self._store_new_image(
            img=res._img,
            pixel_format=pixel_format,
            channel_config=cmd.channel,
            image_id=res.image_id,
            capture=res,
        )

    async def store_laseraf_image(self, res: AutofocusSnapResult) -> None:
        "store the image a reflection-autofocus snap produced"
        pixel_format = ConfigRegistry.get(LaserAutofocusConfig.CAMERA_PIXEL_FORMAT).strvalue
        await self._store_new_image(
            img=res._img,
            pixel_format=pixel_format,
            channel_config=res._channel,
            image_id=res.image_id,
        )

    def _handle_streamed_image(
        self, img: np.ndarray, channel: sc.AcquisitionChannelConfig
    ) -> None:
        """
        one streamed frame, on the camera SDK's acquisition thread

        The channel comes with the frame rather than from server-side bookkeeping, so a frame
        can never be filed under a channel the stream is no longer running.
        """
        self.image_store_threadpool.run(
            self._store_new_image(
                img=img,
                pixel_format=ConfigRegistry.get(CameraConfig.MAIN_PIXEL_FORMAT).strvalue,
                channel_config=channel,
            )
        )

    def get_config_list(self) -> ConfigListResponse:
        """
        get list of existing config files

        these files are already stored on the machine, and can be loaded on request.
        """

        def map_filepath_to_info(c: path.Path) -> ConfigFileInfo:
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

        configs: list[ConfigFileInfo] = []
        unreadable: list[UnreadableConfigFile] = []
        for c in GlobalConfigHandler.get_config_list():
            # One bad file must not hide the rest, nor disappear itself: a protocol that
            # silently drops out of the list looks deleted, when it is only broken.
            try:
                configs.append(map_filepath_to_info(c))
            except Exception as e:
                logger.warning(f"protocol list - cannot read {c}: {describe(e)}")
                unreadable.append(UnreadableConfigFile(filename=c.name, error=describe(e)))

        return ConfigListResponse(configs=configs, unreadable=unreadable)

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

        # Lock-free: reads a field + the status hatch (get_current_state), no hardware mutation.
        if self.microscope.is_in_loading_position:
            error_internal(detail="now allowed while in loading position")

        _plates = [p for p in sc.Plates if p.Model_id == plate_model_id]
        if len(_plates) == 0:
            error_internal(f"{plate_model_id=} is not a known plate model")
        plate = _plates[0]

        current_pos = (await self.microscope.get_current_state()).stage_position

        # real/should position = measured/is position + calibrated offset
        # i.e. calibrated offset = real/should position - measured/is position
        old_x_mm=ConfigRegistry.get(CalibrationConfig.OFFSET_X_MM).floatvalue
        old_y_mm=ConfigRegistry.get(CalibrationConfig.OFFSET_Y_MM).floatvalue
        old_z_mm=ConfigRegistry.get(CalibrationConfig.OFFSET_Z_MM).floatvalue
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
            },
            privileged=True,
        )

        return BasicSuccessResponse()

    async def _store_new_image(
        self,
        img: np.ndarray,
        pixel_format: str,
        channel_config: sc.AcquisitionChannelConfig,
        image_id: str | None = None,
        capture: CapturedImageMetadata | None = None,
    ) -> ImageStoreEntry:
        """
        store a new image, return the channel handle (into self.latest_images)

        note: this stores regular images, as well as autofocus images
        """

        image_id = image_id or str(uuid.uuid4())

        if capture is None:
            try:
                adapter_state = await self.microscope.get_current_state()
            except DisconnectError:
                error_internal("hardware disconnect")
            capture = CapturedImageMetadata(
                image_id=image_id,
                channel=channel_config,
                width_px=img.shape[1],
                height_px=img.shape[0],
                timestamp=time.time(),
                position=adapter_state.stage_position,
                pixel_format=pixel_format,
                camera_bit_depth=bits_for_pixel_format(pixel_format),
                bit_depth=img.dtype.itemsize * 8,
            )

        # store new image
        new_image_store_entry = ImageStoreEntry(
            pixel_format=pixel_format,
            capture=capture,
            info=ImageStoreInfo(
                channel=channel_config,
                width_px=capture.width_px,
                height_px=capture.height_px,
                timestamp=capture.timestamp,
                image_id=image_id,
                position=SitePosition(
                    well_name="",
                    site_x=0,
                    site_y=0,
                    site_z=0,
                    x_offset_mm=0,
                    y_offset_mm=0,
                    z_offset_mm=0,
                    position=capture.position,
                ),
            ),
        )
        new_image_store_entry._img = img
        logger.debug(f"stored new image for {channel_config.handle} at {time.time()}")
        self.latest_images[channel_config.handle] = new_image_store_entry

        return new_image_store_entry

    def _cache_captured_image(self, capture: ImageAcquiredResponse) -> ImageStoreEntry:
        """Keep an acquired frame available to viewers without assigning it storage policy."""
        entry = ImageStoreEntry(
            pixel_format=capture.pixel_format,
            capture=capture,
            info=ImageStoreInfo(
                channel=capture.channel,
                width_px=capture.width_px,
                height_px=capture.height_px,
                timestamp=capture.timestamp,
                image_id=capture.image_id,
                position=SitePosition(
                    well_name="",
                    site_x=0,
                    site_y=0,
                    site_z=0,
                    x_offset_mm=0,
                    y_offset_mm=0,
                    z_offset_mm=0,
                    position=capture.position,
                ),
            ),
        )
        entry._img = capture._img
        self.latest_images[capture.channel.handle] = entry
        return entry

    async def snap_selected_channels(self, config_file: sc.AcquisitionConfig) -> ChannelSnapSelectionResult:
        """
        Take snapshots of all selected channels with real-time image updates.

        The server orchestrates individual channel snapshots, allowing each image
        to be stored and displayed as soon as it's acquired, rather than waiting
        for all channels to complete. The whole multi-channel snap runs as one job
        on the worker (single owner), so it is atomic w.r.t. other commands.
        """
        submitted = self.worker.submit_job(
            lambda: self._snap_selected_channels_impl(config_file), "snap_selected_channels"
        )
        if submitted is None:
            error_microscope_busy(self._busy_reasons())
        try:
            return await asyncio.wrap_future(submitted[0])
        except DisconnectError:
            error_internal("hardware disconnected")
        except OperationCancelledError:
            error_internal("snap cancelled")

    async def _snap_selected_channels_impl(self, config_file: sc.AcquisitionConfig) -> ChannelSnapSelectionResult:
        """Worker-thread body of snap_selected_channels: establish, then snap each
        enabled channel in turn, storing each image as it arrives."""
        pixel_format = ConfigRegistry.get(CameraConfig.MAIN_PIXEL_FORMAT).strvalue

        # Establish hardware connection
        await self.microscope.execute(EstablishHardwareConnection())

        # Validate all enabled channels BEFORE starting to image any of them
        for channel in config_file.channels:
            if channel.enabled:
                self.microscope.validate_channel_for_acquisition(channel)

        channel_handles: list[str] = []
        channel_images: dict[str, np.ndarray] = {}

        # Execute individual channel snapshots
        for channel in config_file.channels:
            if not channel.enabled:
                continue

            self.microscope.raise_if_cancelled()  # cancel between channels

            # Create and execute individual snapshot command
            cmd_snap = ChannelSnapshot(
                channel=channel,
                machine_config=config_file.machine_config or [],
            )
            res_snap = await self.microscope.execute(cmd_snap)

            # Store image immediately (allows incremental UI updates)
            await self._store_new_image(
                img=res_snap._img,
                pixel_format=pixel_format,
                channel_config=channel,
                image_id=res_snap.image_id,
                capture=res_snap,
            )

            # Accumulate results
            channel_images[channel.handle] = res_snap._img
            channel_handles.append(channel.handle)

        logger.debug(f"server - took snapshot in {len(channel_handles)} channels")

        result = ChannelSnapSelectionResult(channel_handles=channel_handles)
        result._images = channel_images
        return result

    async def get_current_state(self) -> CoreCurrentState:
        """
        get current state of the microscope

        for details see fields of return value
        """


        try:
            microscope_adapter_state = await self.microscope.get_current_state()
        except DisconnectError:
            error_internal("hardware disconnect")

        # One locked read of the adapter for both answers, so a client can never be handed a
        # stream with no channel or a channel that is not streaming.
        stream = self.microscope.stream_state()

        # Busy state is owned by the single-owner worker (reject-on-busy): a command
        # in flight == busy. The status hatch reads it directly, never queuing.
        # One read of the worker for all three: whether the microscope is busy, what with,
        # and - when that is an acquisition - which one. Derived separately they disagree at
        # the edges of a run, because the bookkeeping flags are set inside the job while the
        # worker holds it for a moment longer either side.
        label = self.worker.current_label()
        is_busy = self.worker.is_busy
        busy_reasons: list[str] = self._busy_reasons(label) if is_busy else []
        current_acquisition_id = (
            label[len(ACQUISITION_LABEL_PREFIX) :]
            if is_busy and label is not None and label.startswith(ACQUISITION_LABEL_PREFIX)
            else None
        )

        return CoreCurrentState(
            adapter_state=microscope_adapter_state,
            # dict() first: this map is written from the worker thread and the image-store
            # pool, and walking it directly raises if a new channel appears mid-iteration -
            # in a status read, which must never fail
            latest_imgs={key: entry.info for key, entry in dict(self.latest_images).items()},
            current_acquisition_id=current_acquisition_id,
            is_streaming=stream is not None,
            streaming_channel_handle=stream.channel.handle if stream is not None else None,
            is_busy=is_busy,
            busy_reasons=busy_reasons,
            last_acquisition_error=self._last_acquisition_error,
            last_acquisition_error_timestamp=self._last_acquisition_error_timestamp,
            microscope_name=self.microscope_name,
            lock=self.lock.state(),
        )

    def lock_acquire(
        self, owner: str, ttl_s: float | None = None, lock_token: str | None = None
    ) -> LockAcquireResponse:
        """
        take the device lease

        a lease is required for every hardware command (moves, snaps, streaming, acquisition
        start, machine config flush, ...): without one they are refused with 423, and while
        you hold one they are refused for everybody else. status reads and cancel are never
        refused. the lease expires `ttl_s` seconds after the last request that carried the
        token, but never while a command is running on the hardware. calling this with your
        own token renews (and re-labels) the lease.
        """
        # the refusal comes back from acquire() itself: reading the lock again to find out who
        # refused could name a client that took the lease in the meantime
        outcome = self.lock.acquire(owner=owner, ttl_s=ttl_s, token=lock_token)
        if isinstance(outcome, LockDecision):
            error_device_locked(outcome)
        logger.info(f"device lock acquired by '{outcome.owner}' (lock {outcome.lock_id})")
        return LockAcquireResponse(
            token=outcome.token,
            lock_id=outcome.lock_id,
            owner=outcome.owner,
            expires_at=outcome.expires_at,
        )

    def lock_release(self, lock_token: str | None = None) -> BasicSuccessResponse:
        """
        give the device lease back

        idempotent when nothing is held; 423 if someone else holds it (use /api/lock/break).
        """
        outcome = self.lock.release(lock_token)
        match outcome.reason:
            case LockReason.HELD_BY_OTHER:
                error_device_locked(outcome)
            case LockReason.OK:
                released = outcome.state
                logger.info(
                    f"device lock released by '{released.owner}' (lock {released.lock_id})"
                )
            case LockReason.NOT_HELD:
                pass  # nothing was held, so there is nothing to give back
        return BasicSuccessResponse()

    def lock_status(self, lock_token: str | None = None) -> LockStatusResponse:
        """
        who holds the device lease, and whether the `lock_token` you sent is the holder's
        """
        # one read, not two: who holds it and whether it is yours have to be the same snapshot
        decision = self.lock.check(lock_token, renew=False)
        state = decision.state
        return LockStatusResponse(
            held=state.held,
            owner=state.owner,
            lock_id=state.lock_id,
            expires_at=state.expires_at,
            token_matches=decision.permitted,
        )

    def lock_break(self, reason: str = "") -> BasicSuccessResponse:
        """
        force-free the device lease, whoever holds it

        open to anyone (it is a coordination tool, not a permission system). does not stop
        anything that is running; use cancel for that. logged with the previous owner and
        your reason.
        """
        previous = self.lock.break_lock()
        if previous.held:
            logger.warning(
                f"device lock held by '{previous.owner}' (lock {previous.lock_id}) was broken; reason: {reason or '(none given)'}"
            )
        return BasicSuccessResponse()

    async def cancel_acquisition(self, acquisition_id: str) -> BasicSuccessResponse:
        """
        cancel the ongoing acquisition
        """

        if self.acquisitions.get(acquisition_id) is None:
            error_internal(detail=self.acquisitions.describe_unknown(acquisition_id))

        # The worker is what decides, and it decides both halves at once: an acquisition
        # holds the worker for its whole duration under the label `acquisition:<id>`, so a
        # job with this run's label IS this run, and the cancel event that came with it is
        # this run's. Setting it makes the protocol loop's next handle_q_in checkpoint raise
        # AcquisitionCancelledError.
        #
        # Deciding from `AcquisitionStatus.thread_is_running` instead is what made this
        # dangerous: it was refreshed against `self.acquisition_future`, the *latest* run's
        # future rather than this one's, so cancelling a run that had already finished
        # reported it as still going and then cancelled whatever the worker held - the next
        # acquisition, or somebody's move.
        running = self.worker.current_job()
        if running is None or running.label != f"{ACQUISITION_LABEL_PREFIX}{acquisition_id}":
            error_internal(detail=f"acquisition with id {acquisition_id} is not running")

        running.cancel.set()

        return BasicSuccessResponse()

    async def cancel_current_command(self) -> BasicSuccessResponse:
        """Cancel hatch: request cancellation of the command currently on the worker.

        Never rejected/queued — sets the in-flight command's cooperative cancel event.
        Long operations abort at their next checkpoint, freeing the worker. A no-op if
        nothing is running.
        """
        cancel = self.worker.current_cancel()
        if cancel is not None:
            cancel.set()
        return BasicSuccessResponse()

    def _busy_reasons(self, label: str | None = None) -> list[str]:
        """Busy (409) reason: name what the hardware worker is currently running, so the
        rejection says WHY (e.g. "hardware busy: acquisition:<id>") instead of a generic string.

        Pass `label` to describe a worker read the caller has already taken, so that what a
        status response says is busy and what it says is running come from the same instant."""
        if label is None:
            label = self.worker.current_label()
        reasons = [f"hardware busy: {label}"] if label is not None else ["hardware command in progress"]
        # a stall is why the hardware is busy, and why it will stay busy: without it the
        # rejection reads as a wait-and-retry when retrying will never get through
        reasons += [
            f"hardware stalled: {stall.describe()}; it will stay busy until that call returns, "
            f"which may need a restart of the server"
            for stall in self.microscope.stalls()
        ]
        return reasons

    @property
    def acquisition_is_running(self) -> bool:
        if self.acquisition_future is None:
            return False

        if self.acquisition_future.done():
            self.acquisition_future = None
            return False

        return True

    def machine_config_flush(
        self, machine_config: list[ConfigItem], privileged: bool = False
    ) -> MachineConfigFlushResponse:
        """
        write client-side machine config values into the running server config

        privileged handles (calibration, device identity, ...) are only written when `privileged`
        is set; such a change is persisted to the config file at once (and may need a server
        restart to take effect). without it, a flush that would change a privileged handle is
        rejected (400) listing the offending handles, so a stale client cannot overwrite them by
        accident. values failing type/option validation are rejected the same way; unknown
        handles are ignored and reported.
        """
        if not privileged:
            mismatches = ConfigRegistry.privileged_mismatches(machine_config)
            if len(mismatches) > 0:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "flush would change privileged config without privilege",
                        "mismatched_handles": mismatches,
                    },
                )

        # validate everything before writing anything: an explicit flush is all-or-nothing
        known = ConfigRegistry.get_dict()
        invalid = {
            item.handle: reason
            for item in machine_config
            if item.handle in known
            and (reason := ConfigRegistry.validate_value(item.handle, item.value)) is not None
        }
        if len(invalid) > 0:
            raise HTTPException(
                status_code=400,
                detail={"message": "flush contains invalid config values", "invalid": invalid},
            )

        result = GlobalConfigHandler.override(machine_config, privileged=privileged)
        return MachineConfigFlushResponse(
            changed=result.changed,
            skipped_privileged=result.skipped_privileged,
            unknown_handles=result.unknown_handles,
            invalid=result.invalid,
        )

    def get_hardware_capabilities(self) -> HardwareCapabilitiesResponse:
        """
        Get hardware capabilities including real hardware limits from the microscope.
        """
        # Get channels config (native list, .objectvalue asserts type)
        channels_data = ConfigRegistry.get(ImagingConfig.CHANNELS).objectvalue

        # Convert to ChannelConfig objects
        channel_configs = [ChannelConfig(**ch) for ch in channels_data]  # type: ignore

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

        # Hardware limits are static camera capabilities: query once (briefly takes the
        # camera lock), then always serve the cache — no hardware access on later calls.
        if self._hardware_limits_cache is None:
            self._hardware_limits_cache = self.microscope.get_hardware_limits()
        hardware_limits_obj = self._hardware_limits_cache

        main_camera = self.microscope.main_camera_info()

        # Filter positions and forbidden areas are stored as config objects. Parsing them here
        # means a client reads a typed list instead of digging through machine config for a
        # value the config system can only describe as "some object".
        filters: list[FilterConfig] = []
        if ConfigRegistry.get(FilterWheelConfig.AVAILABLE).boolvalue:
            filters = [
                FilterConfig(**f)
                for f in ConfigRegistry.get(FilterWheelConfig.CONFIGURATION).objectvalue  # type: ignore[misc]
            ]

        forbidden_areas: list[ForbiddenArea] = []
        try:
            forbidden_areas = ForbiddenAreaList.model_validate(
                {"areas": ConfigRegistry.get(ProtocolConfig.FORBIDDEN_AREAS).objectvalue}
            ).areas
        except (KeyError, ValidationError) as e:
            logger.warning(f"forbidden areas could not be read for hardware capabilities: {e}")

        return HardwareCapabilitiesResponse(
            main_camera=main_camera,
            wellplate_types=sc.Plates,
            main_camera_imaging_channels=acquisition_channels,
            hardware_limits=hardware_limits_obj,
            filters=filters,
            forbidden_areas=forbidden_areas,
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
        # Pre-flight checks (lock-free: status read + a field read). The acquisition
        # itself runs on the worker, which provides exclusivity.
        try:
            _ = await self.microscope.get_current_state()
        except DisconnectError:
            error_internal("device not connected")

        if self.microscope.is_in_loading_position:
            error_internal(detail="now allowed while in loading position")

        if self.acquisition_future is not None:
            if not self.acquisition_future.done():
                # a run in progress is busy hardware, not a server fault: 409 like every other
                # command that arrives while something else has the microscope
                error_microscope_busy(self._busy_reasons())
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

        acquisition_status = AcquisitionStatus(acquisition_id=acquisition_id)

        def handle_q_in():
            """protocol cancel checkpoint: raise if this acquisition's worker cancel
            event has been set (see cancel_acquisition)."""
            if self.microscope.should_cancel():
                raise AcquisitionCancelledError("acquisition cancelled")

        project_name_issue = name_check(config_file.project_name)
        if project_name_issue is not None:
            error_internal(detail=f"project name is not acceptable: {project_name_issue}")

        plate_name_issue = name_check(config_file.plate_name)
        if plate_name_issue is not None:
            error_internal(detail=f"plate name is not acceptable: {plate_name_issue}")

        # Get camera info for OME metadata from microscope
        camera = self.microscope.main_camera_info()

        # Extract objective magnification from configuration
        objective_magnification: int | None = None
        try:
            g_config = GlobalConfigHandler.get_dict()
            objective_config_item = g_config.get(CameraConfig.MAIN_OBJECTIVE.value)
            if objective_config_item:
                # Parse objective value (e.g., "20xolympus" -> 20)
                objective_value = objective_config_item.strvalue
                # Extract magnification from objective string
                mag_match = re.match(r'(\d+)x', objective_value)
                if mag_match:
                    objective_magnification = int(mag_match.group(1))
        except Exception as e:
            logger.debug(f"Could not extract objective magnification: {e}")

        # Build OME template with microscope-wide metadata
        ome_template = build_ome_instrument(
            microscope_name=self.microscope_name,
            camera_vendor=camera.vendor,
            camera_model=camera.model,
            camera_sn=camera.serial_number,
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
            objective_magnification=objective_magnification,
        )

        if protocol.num_images_total == 0:
            error_internal(
                detail=f"no images to acquire ({protocol.num_wells = },{protocol.num_sites = },{protocol.num_channels = },{protocol.num_channel_z_combinations = })"
            )

        # Validate that no site positions fall within forbidden areas
        # This catches forbidden positions during preparation instead of during execution
        # Parse forbidden areas once and reuse for all positions to avoid repeated parsing
        try:
            forbidden_areas_data = ConfigRegistry.get(ProtocolConfig.FORBIDDEN_AREAS).objectvalue
            forbidden_areas = ForbiddenAreaList.model_validate({"areas": forbidden_areas_data})
        except KeyError:
            forbidden_areas = None

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

        # Room for the images. Checked here with everything else: a run that fills the disk
        # partway has already cost the sample the time and light of the part it did take.
        try:
            check_capacity(int(protocol.max_storage_size_images_GB * 1024**3))
        except InsufficientStorageError as e:
            error_internal(detail=f"cannot start the acquisition: {e}")

        # Run the whole acquisition as one worker job (single owner). It holds the worker
        # for its entire duration, so every other command is rejected (409) meanwhile —
        # which is exactly what allow_while_acquisition_is_running=False expects. Cancel via
        # the worker's cancel event (cancel_acquisition sets it; handle_q_in checks it).
        # Registered BEFORE the job is handed over: the moment the worker takes it, status
        # reports the microscope busy with this acquisition, and a client asking about
        # `current_acquisition_id` in that instant must find the run that is making it busy
        # rather than nothing at all. Undone again if the handover is refused.
        self.acquisitions.add(acquisition_status)
        submitted = self.worker.submit_job(
            lambda: self._run_acquisition_impl(protocol, acquisition_status),
            f"{ACQUISITION_LABEL_PREFIX}{acquisition_id}",
        )
        if submitted is None:
            self.acquisitions.discard(acquisition_id)
            error_microscope_busy(self._busy_reasons())
        self.acquisition_future = submitted[0]

        return AcquisitionStartResponse(acquisition_id=acquisition_id)

    async def _run_acquisition_impl(
        self, protocol: ProtocolGenerator, acquisition_status: AcquisitionStatus
    ) -> None:
        """Worker-thread body of an acquisition: drive the protocol generator, execute each
        hardware step, store each channel image. Updates acquisition_status.last_status in
        place (the status endpoint reads that). No lock — the worker is the single owner."""
        logger.debug("protocol - started")
        try:
            # initiate generation
            protocol_generator = protocol.generate()

            # send none on first yield; protocol yields None to indicate it is finished
            result = None
            while (next_step := await protocol_generator.asend(result)) is not None:
                logger.debug(f"protocol - next step {type(next_step)}")
                if isinstance(next_step, str):
                    result = None
                else:
                    result = None
                    try:
                        result = await self.microscope.execute(next_step)
                    except DisconnectError as e:
                        logger.debug(f"executing protocol step generated error {e}")
                        await protocol_generator.athrow(e)

                    if isinstance(result, ImageAcquiredResponse) and isinstance(
                        next_step, ChannelSnapshot
                    ):
                        self._cache_captured_image(result)

            logger.debug("protocol done")
            self._acquisition_ended(acquisition_status, AcquisitionStatusStage.COMPLETED)

        except (AcquisitionCancelledError, OperationCancelledError):
            # user requested cancellation (via cancel_acquisition -> worker cancel event)
            self._acquisition_ended(acquisition_status, AcquisitionStatusStage.CANCELLED)
        except HTTPException as e:
            # validation / internal errors crash the acquisition
            self._acquisition_crashed(acquisition_status, str(e.detail))
        except StorageFailedError as e:
            self._acquisition_crashed(acquisition_status, f"image storage failed: {e}")
        except DisconnectError as e:
            # The adapter has already dropped the hardware and recorded which device went; what
            # is left is to say so on the acquisition, in words the operator can act on.
            reason = "; ".join([str(e), *getattr(e, "__notes__", [])])
            self._acquisition_crashed(acquisition_status, f"hardware disconnect: {reason}")
        except Exception as e:
            # not one of the ways a run is expected to fail, so the traceback goes in the log
            self._acquisition_crashed(
                acquisition_status, "; ".join([describe(e), *getattr(e, "__notes__", [])]), e
            )
        finally:
            self.acquisitions.finish(acquisition_status.acquisition_id)
        logger.debug("acquisition done")

    def _acquisition_ended(
        self, acquisition_status: AcquisitionStatus, stage: AcquisitionStatusStage
    ) -> AcquisitionStatusOut:
        # The protocol publishes a status when it is built, before the run is handed to the
        # worker, so every run that can end has one to end.
        last = acquisition_status.last_status
        assert last is not None, "the protocol publishes a status before the run starts"
        last.acquisition_status = stage
        return last

    def _acquisition_crashed(
        self,
        acquisition_status: AcquisitionStatus,
        message: str,
        unexpected: BaseException | None = None,
    ) -> None:
        "the run stopped on an error: say so on it, in the log, and on the status clients poll"
        logger.opt(exception=unexpected).error(
            f"acquisition {acquisition_status.acquisition_id} crashed: {message}"
        )
        last = self._acquisition_ended(acquisition_status, AcquisitionStatusStage.CRASHED)
        last.message = message
        self._last_acquisition_error = message
        self._last_acquisition_error_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    async def get_acquisition_status(self, acquisition_id: str) -> AcquisitionStatusOut:
        """
        get status of an acquisition
        """

        acq_res = self.acquisitions.get(acquisition_id)
        if acq_res is None:
            error_internal(detail=self.acquisitions.describe_unknown(acquisition_id))

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

        # Sort channels (pure computation, no hardware -> no lock).
        enabled_channels = self.microscope._sort_channels_by_imaging_order(
            enabled_channels, tp.cast(ImagingOrder, imaging_order_value)
        )

        # Run the whole per-channel loop as one worker job (single owner). Await it so the
        # WebSocket stays open and status updates (marshaled onto its loop by the callback)
        # flow until the snap finishes.
        submitted = self.worker.submit_job(
            lambda: self._progressive_snap_impl(enabled_channels, config_file, callback),
            "progressive_snap",
        )
        if submitted is None:
            error_microscope_busy(self._busy_reasons())
        await asyncio.wrap_future(submitted[0])

        return BasicSuccessResponse()

    async def _progressive_snap_impl(
        self,
        enabled_channels: list[sc.AcquisitionChannelConfig],
        config_file: sc.AcquisitionConfig,
        callback: tp.Callable[[ChannelSnapProgressiveStatus], None],
    ) -> None:
        """Worker-thread body of progressive snap: snap each channel in turn, reporting
        status via `callback` (which marshals onto the WebSocket's loop) as each completes."""
        total_channels = len(enabled_channels)
        completed_channels = 0

        try:
            for channel in enabled_channels:
                self.microscope.raise_if_cancelled()

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
                    snap_command = ChannelSnapshot(
                        channel=channel,
                        machine_config=config_file.machine_config or []
                    )
                    result = await self.microscope.execute(snap_command)

                    # Store the image
                    g_dict = GlobalConfigHandler.get_dict()
                    pixel_format = g_dict[CameraConfig.MAIN_PIXEL_FORMAT.value].strvalue
                    await self._store_new_image(
                        img=result._img,
                        pixel_format=pixel_format,
                        channel_config=channel,
                        image_id=result.image_id,
                        capture=result,
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

                except OperationCancelledError:
                    raise
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

        except OperationCancelledError:
            callback(ChannelSnapProgressiveStatus(
                channel_handle="",
                channel_name="",
                status="error",
                total_channels=total_channels,
                completed_channels=completed_channels,
                message="Progressive snap cancelled",
                error_detail="cancelled"
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

    SHUTDOWN_CLOSE_DEADLINE_S: tp.ClassVar[float] = 90.0
    "how long shutdown waits for the hardware to close: past the adapter's own close deadlines"

    def _close_microscope(self) -> bool:
        """
        Close the hardware, without waiting forever on it. Returns whether it closed cleanly.

        A hardware call stuck in an SDK holds its device lock for as long as it is stuck, and
        closing takes every device lock, so closing behind a stall would never return and the
        process would never exit. With a stall known up front there is nothing to wait for;
        otherwise the close runs on its own thread, with a deadline in case one begins.
        """
        stalls = self.microscope.stalls()
        if stalls:
            for stall in stalls:
                logger.error(
                    f"shutdown - not closing the hardware: {stall.describe()}, and closing would "
                    f"wait on it. The SDK still holds the device; power-cycle it if the next "
                    f"start reports it in use"
                )
            return False

        outcome: list[BaseException | None] = []

        def close() -> None:
            try:
                self.microscope.close()
                outcome.append(None)
            except BaseException as e:
                outcome.append(e)

        closer = threading.Thread(target=close, name="shutdown-close", daemon=True)
        closer.start()
        closer.join(self.SHUTDOWN_CLOSE_DEADLINE_S)

        if closer.is_alive():
            stalled = "; ".join(stall.describe() for stall in self.microscope.stalls())
            logger.error(
                f"shutdown - the hardware did not close within {self.SHUTDOWN_CLOSE_DEADLINE_S:.0f} s"
                + (f": {stalled}" if stalled else "")
                + ". Exiting with it still open; power-cycle it if the next start reports it in use"
            )
            return False

        error = outcome[0]
        if error is None:
            return True
        if isinstance(error, HardwareCloseError):
            for device, failure in error.failures.items():
                logger.error(f"shutdown - {device} did not close cleanly: {describe(failure)}")
        else:
            logger.opt(exception=error).error("shutdown - closing the microscope failed")
        return False

    def close(self) -> bool:
        """
        Shut down: release the hardware, save the config, finish writing images.

        Every step runs whatever happened to the ones before it - hardware that would not
        close is no reason to lose the config or the images still queued. Each failure is
        logged as it happens. Returns whether everything went cleanly, so the process can exit
        non-zero when it did not: a device left open is what makes the next start fail with
        "device in use", and that start needs to be able to find out why.
        """
        clean = True

        logger.debug("shutdown - closing microscope")
        if not self._close_microscope():
            clean = False

        logger.debug("shutdown - storing global config")
        try:
            GlobalConfigHandler.store()
        except Exception:
            clean = False
            logger.exception("shutdown - storing the global config failed")

        logger.debug("shutdown - closing image store threadpool")
        try:
            self.image_store_threadpool.join()
        except Exception:
            clean = False
            logger.exception("shutdown - finishing queued image writes failed")

        logger.debug("shutdown - done")
        return clean
