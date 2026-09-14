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
import traceback
import time
import typing as tp
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
from seafront.hardware.forbidden_areas import ForbiddenArea, ForbiddenAreaList
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
    CameraInfo,
    ChannelSnapProgressiveStatus,
    ChannelSnapSelectionResult,
    ChannelSnapshot,
    ConfigFetchResponse,
    ConfigFileInfo,
    ConfigListResponse,
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

        # Client-held lease above the worker: one client at a time across requests. Routes
        # registered with a lease-requiring policy refuse 423 while another client holds it.
        self.lock = DeviceLock(worker_is_busy=lambda: self.worker.is_busy)

        self.acquisition_map: dict[str, AcquisitionStatus] = {}
        """ map containing information on past and current acquisitions """

        self.acquisition_future: CCFuture[None] | None = None

        self.latest_images: dict[str, ImageStoreEntry] = {}
        "latest image acquired in each channel, key is channel handle"

        self.image_store_threadpool = AsyncThreadPool()

        self._streaming_channel: sc.AcquisitionChannelConfig | None = None

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
            img=res._img, pixel_format=pixel_format, channel_config=cmd.channel
        )

    async def store_laseraf_image(self, res: AutofocusSnapResult) -> None:
        "store the image a reflection-autofocus snap produced"
        pixel_format = ConfigRegistry.get(LaserAutofocusConfig.CAMERA_PIXEL_FORMAT).strvalue
        await self._store_new_image(
            img=res._img, pixel_format=pixel_format, channel_config=res._channel
        )

    def stream_begin(self, channel: sc.AcquisitionChannelConfig) -> None:
        """
        start feeding streamed frames into the image store

        Channel first, then the callback: status reports the channel only while the callback
        is installed (see get_current_state), so setting them in this order is what keeps a
        status read from ever catching a stream with no channel.
        """
        self._streaming_channel = channel
        # Wire the server-side frame handler under the narrow stream-callback lock (the SDK
        # acquisition thread reads it). The ChannelStreamBegin command already ran on the
        # worker and started the camera stream; frames are dropped until this is set.
        self.microscope.set_stream_callback(self._handle_streamed_image)

    def stream_end(self) -> None:
        "stop attributing streamed frames to a channel"
        self._streaming_channel = None

    def _handle_streamed_image(self, arg: "np.ndarray | bool") -> bool:
        if isinstance(arg, bool):
            should_stop = arg
            if should_stop:
                return True
        else:
            img = arg
            assert self._streaming_channel is not None
            self.image_store_threadpool.run(
                self._store_new_image(
                    img=img,
                    pixel_format=ConfigRegistry.get(CameraConfig.MAIN_PIXEL_FORMAT).strvalue,
                    channel_config=self._streaming_channel,
                )
            )
        return False

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
                channel_config=channel
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

        # Check if streaming is currently active by looking at the stream callback. Read once
        # and report the channel from that same read: the two are set together on the way in,
        # but on the way out the stream is stopped by the command on the worker while the
        # channel is cleared by the route callback that follows it. A status read landing in
        # between must not hand a client a channel that is no longer streaming.
        is_streaming = self.microscope.stream_callback is not None
        streaming_channel = self._streaming_channel if is_streaming else None

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
            is_streaming=is_streaming,
            streaming_channel_handle=streaming_channel.handle if streaming_channel is not None else None,
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

        # The acquisition holds the worker for its whole duration, so the in-flight
        # cancel event IS this acquisition's. Setting it makes the protocol loop's next
        # handle_q_in checkpoint raise AcquisitionCancelledError.
        cancel = self.worker.current_cancel()
        if cancel is not None:
            cancel.set()

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
        if label is not None:
            return [f"hardware busy: {label}"]
        return ["hardware command in progress"]

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

        main_camera = CameraInfo()
        try:
            camera = self.microscope.main_camera.value  # type: ignore[attr-defined]
            main_camera = CameraInfo(
                vendor=str(getattr(camera, "vendor_name", "")),
                model=str(getattr(camera, "model_name", "")),
                serial_number=str(getattr(camera, "sn", "")),
            )
        except Exception as e:
            logger.debug(f"camera identity unavailable for hardware capabilities: {e}")

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

        # Run the whole acquisition as one worker job (single owner). It holds the worker
        # for its entire duration, so every other command is rejected (409) meanwhile —
        # which is exactly what allow_while_acquisition_is_running=False expects. Cancel via
        # the worker's cancel event (cancel_acquisition sets it; handle_q_in checks it).
        # Flagged as running BEFORE the job is handed over: the moment the worker takes it,
        # status reports the microscope busy with this acquisition, and a client reading
        # `current_acquisition_id` in that instant must find the run that is making it busy
        # rather than nothing at all. Undone again if the handover is refused.
        acquisition_status.thread_is_running = True
        submitted = self.worker.submit_job(
            lambda: self._run_acquisition_impl(protocol, acquisition_status),
            f"{ACQUISITION_LABEL_PREFIX}{acquisition_id}",
        )
        if submitted is None:
            acquisition_status.thread_is_running = False
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

                    if result is not None and isinstance(next_step, ChannelSnapshot):
                        await self._store_new_image(
                            img=result._img,
                            pixel_format=ConfigRegistry.get(CameraConfig.MAIN_PIXEL_FORMAT).strvalue,
                            channel_config=next_step.channel,
                        )

            logger.debug("protocol done")

            # finished regularly (there was at least one image, so a status exists)
            assert acquisition_status.last_status is not None
            acquisition_status.last_status.acquisition_status = AcquisitionStatusStage.COMPLETED

        except (AcquisitionCancelledError, OperationCancelledError):
            # user requested cancellation (via cancel_acquisition -> worker cancel event)
            assert acquisition_status.last_status is not None
            acquisition_status.last_status.acquisition_status = AcquisitionStatusStage.CANCELLED
        except HTTPException as e:
            # validation / internal errors crash the acquisition
            assert acquisition_status.last_status is not None
            acquisition_status.last_status.acquisition_status = AcquisitionStatusStage.CRASHED
            acquisition_status.last_status.message = str(e.detail)
            self._last_acquisition_error = str(e.detail)
            self._last_acquisition_error_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        except DisconnectError:
            logger.debug("microscope disconnected during acquisition - closing connections.")
            self.microscope.close()
            assert acquisition_status.last_status is not None
            acquisition_status.last_status.acquisition_status = AcquisitionStatusStage.CRASHED
            acquisition_status.last_status.message = "hardware disconnect"
        except Exception as e:
            logger.exception(f"error during acquisition {e}\n{traceback.format_exc()}")
            full_error = traceback.format_exc()
            self._last_acquisition_error = f"acquisition failed because {e!s}, more specifically: {full_error}"
            self._last_acquisition_error_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if acquisition_status.last_status is not None:
                acquisition_status.last_status.acquisition_status = AcquisitionStatusStage.CRASHED
        finally:
            # ensure no dangling image-writer threads, and mark the run finished (close is idempotent)
            if protocol.image_store is not None:
                late = protocol.image_store.close()
                if late.failed:
                    logger.error(f"acquisition {protocol.acquisition_id}: {len(late.failed)} images failed to store: {late.failed[:5]}")
            acquisition_status.thread_is_running = False
        logger.debug("acquisition done")

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
                    result = await self.microscope.execute(ChannelSnapshot(
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

    def close(self):
        logger.debug("shutdown - closing microscope")
        try:
            self.microscope.close()
        except:
            pass

        logger.debug("shutdown - storing global config")
        GlobalConfigHandler.store()

        logger.debug("shutdown - closing image store threadpool")
        self.image_store_threadpool.join()

        logger.debug("shutdown - done")

