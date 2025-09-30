import asyncio as aio
import datetime as dt
import math
import pathlib as path
import random
import string
import time
import typing as tp
from concurrent.futures import Future as ConcurrentFuture
from threading import Thread
import asyncio

import seaconfig as sc
import tifffile
from pydantic import BaseModel, ConfigDict, Field

import seafront.server.commands as cmds
from seafront.config import basics
from seafront.config.basics import ImagingOrder
from seafront.config.handles import CameraConfig, ImagingConfig, LaserAutofocusConfig, StorageConfig
from seafront.hardware import microcontroller as mc
from seafront.logger import logger


class AsyncThreadPool(BaseModel):
    """
    Thread pool for running async future/coroutines

    Example:
    ```
    async def work():
        print("async work")

    # initialize pool
    pool = AsyncThreadPool()
    # submit to pool
    future = pool.run(work())
    # block waiting for results
    future.result()
    # shutdown pool
    pool.join()
    ```

    from https://stackoverflow.com/a/77682889
    """

    handle_disconnect: bool = Field(default=False)
    "indicate if hardware disconnect errors should be handled by this protocol"

    workers: int = Field(default=1)
    """ Number of worker threads in the pool """
    threads: list[Thread] = Field(default_factory=list)
    """ Running threads in the pool """
    loops: list[aio.AbstractEventLoop] = Field(default_factory=list)
    """ Event loops for each thread """
    roundrobin: int = Field(default=0)
    """ Next thread to run something in, for round-robin scheduling """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # pydantics version of dataclass.__post_init__
    def model_post_init(self, __context):
        # initialize threads
        for _ in range(self.workers):
            loop = aio.new_event_loop()
            self.loops.append(loop)
            thread = Thread(target=loop.run_forever)
            thread.start()
            self.threads.append(thread)

    def run(self, target: "tp.Coroutine", worker: int | None = None) -> "ConcurrentFuture":
        """Run async future/coroutine in the thread pool

        :param target: the future/coroutine to execute
        :param worker: worker thread you want to run the callable; None for round-robin
            selection of worker thread
        :return: future with result
        """
        if worker is None:
            worker = self.roundrobin
            self.roundrobin = (worker + 1) % self.workers

        return aio.run_coroutine_threadsafe(target, self.loops[worker])

    def join(self, timeout: float = 0.5):
        """Blocking call to close the thread pool

        :param timeout: timeout for polling a thread to check if its async tasks are all finished
        """
        for i in range(self.workers):
            # wait for completion of pending tasks before stopping;
            # stop only waits for current batch of callbacks to complete
            loop = self.loops[i]
            while len(aio.all_tasks(loop)):
                time.sleep(timeout)
            loop.call_soon_threadsafe(loop.stop)
            self.threads[i].join(timeout=timeout)


def _format_well_name_with_padding(well_name: str) -> str:
    """
    Format well name to ensure column index is zero-padded to 2 digits.
    
    Examples:
        F8 -> F08
        A10 -> A10 (already padded)
        H1 -> H01
    
    Args:
        well_name: Original well name (e.g., "F8", "A10")
        
    Returns:
        Formatted well name with zero-padded column index
    """
    # Well name format: single letter + number (e.g., F8, A10)
    row_letter = well_name[0]
    col_number = well_name[1:]
    
    # Zero-pad column to 2 digits
    return f"{row_letter}{col_number.zfill(2)}"


def make_unique_acquisition_id(length: tp.Literal[16, 32] = 16) -> str:
    """
    Generates a random microscope image acquisition protocol name.

    The name:
    - Starts with an uppercase letter.
    - Uses only uppercase letters and digits (i.e. no mix of cases).
    - Is either 16 or 32 characters long.

    Parameters:
        length (int): The desired length of the ID (must be 16 or 32).

    Returns:
        str: A randomly generated protocol name.
    """

    # Ensure the first character is an uppercase letter.
    first_char = random.choice(string.ascii_uppercase)

    # The remaining characters can be uppercase letters or digits.
    allowed_chars = string.ascii_uppercase + string.digits
    remaining = "".join(random.choices(allowed_chars, k=length - 1))

    return first_char + remaining


async def store_image(
    image_entry: cmds.ImageStoreEntry,
    img_compression_algorithm: tp.Literal["LZW", "zlib"],
    metadata: dict[str, str],
):
    """
    store img as .tiff file, with compression and some metadata

    params:
        metadata: embedded into the file
            e.g.:
                {
                    "BitsPerPixel":latest_channel_image.bit_depth, # type:ignore
                    "BitPaddingInfo":"lower bits are padding with 0s",

                    "LightSourceName":channel.name,
                    "PixelSizeUM":f"{PIXEL_SIZE_UM:.3f}",

                    "ExposureTime":f"{int(channel.exposure_time_ms*1e3)}",
                    "ExposureTimeMS":f"{channel.exposure_time_ms:.3f}",
                    "AnalogGainDB":f"{channel.analog_gain:.1f}",

                    "Make":core_main_cam.vendor_name,
                    "Model":core_main_cam.model_name,
                }
    """

    image_storage_path = image_entry.info.storage_path
    assert isinstance(image_storage_path, str), f"{image_entry.info.storage_path} is not str"

    # ensure parent dir exists (primarily for time series)
    storage_path = path.Path(image_storage_path)
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    # takes 70-250ms
    tifffile.imwrite(
        storage_path,
        image_entry._img,
        compression=img_compression_algorithm,
        compressionargs={},  # for zlib: {'level': 8},
        maxworkers=1,
        # this metadata is just embedded as a comment in the file
        metadata=metadata,
        photometric="minisblack",  # zero means black
    )

    logger.debug(f"stored image to {image_storage_path}")


class ProtocolGenerator(BaseModel):
    """
    microscope independent protocol that generates commands to be executed (generates on the fly, based on results of previous commands, e.g. based on measured z offset)
    params:
        img_compression_algorithm: tp.Literal["LZW","zlib"] - lossless compression algorithms only
    """

    config_file: sc.AcquisitionConfig
    handle_q_in: tp.Callable[[], None]
    plate: sc.Wellplate
    acquisition_status: cmds.AcquisitionStatus

    acquisition_id: str = Field(default_factory=make_unique_acquisition_id)

    image_store_pool: AsyncThreadPool = Field(default_factory=lambda: AsyncThreadPool())

    img_compression_algorithm: tp.Literal["LZW", "zlib"] = "LZW"

    # the values below are initialized during post init hook
    num_wells: int = -1
    num_sites: int = -1
    num_channels: int = -1
    num_channel_z_combinations: int = -1
    num_images_total: int = -1
    num_timepoints: int = 1
    site_topleft_x_mm: float = -1
    site_topleft_y_mm: float = -1
    image_size_bytes: int = -1
    max_storage_size_images_GB: float = -1
    project_output_path: path.Path = Field(default_factory=path.Path)

    latest_channel_images: dict[str, cmds.ImageStoreEntry] = Field(default_factory=dict)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def plate_wells(self):
        "selected wells"
        return [w for w in self.config_file.plate_wells if w.selected]

    @property
    def well_sites(self):
        "selected sites in the grid mask"
        return [s for s in self.config_file.grid.mask if s.selected]

    @property
    def channels(self):
        "selected channels"
        return [c for c in self.config_file.channels if c.enabled]

    def _sort_channels_by_imaging_order(self, channels: list, imaging_order: ImagingOrder) -> list:
        """
        Sort channels according to the specified imaging order.
        
        Args:
            channels: List of enabled AcquisitionChannelConfig objects
            imaging_order: Sorting strategy to use
            
        Returns:
            Sorted list of channels
        """
        if imaging_order == "protocol_order":
            # Keep original order from config file
            return channels
                
        elif imaging_order == "wavelength_order":
            # Sort by emission wavelength (high to low), then brightfield last
            def wavelength_key(channel):
                # Brightfield channels go last
                if hasattr(channel, 'is_brightfield') and channel.is_brightfield:
                    return (0, channel.name)  # Sort brightfield last with lowest priority
                # Sort by emission wavelength descending (highest first)
                emission_nm = getattr(channel, 'emission_wavelength_nm', 500)  # Default 500nm
                return (emission_nm, channel.name)  # Positive for descending order when reversed
                    
            return sorted(channels, key=wavelength_key, reverse=True)
                
        elif imaging_order == "z_order":
            # This case is handled in the z-stacking logic below
            # For channel ordering, use protocol order as fallback
            return channels
        
        else:
            # Default fallback
            return channels

    def get_greedy_well_order(self) -> list[sc.PlateWellConfig]:
        """
        Get wells in greedy nearest-neighbor traversal order for maximum efficiency.
        Always moves to the nearest unvisited well.
        """
        wells = self.plate_wells.copy()
        if len(wells) <= 1:
            return wells
        
        def distance(w1: sc.PlateWellConfig, w2: sc.PlateWellConfig) -> float:
            return math.sqrt((w1.col - w2.col)**2 + (w1.row - w2.row)**2)
        
        # Start with the first well (top-left-most)
        wells.sort(key=lambda w: (w.row, w.col))
        current_well = wells.pop(0)
        optimized_wells = [current_well]
        
        # Greedily pick the nearest remaining well
        while wells:
            nearest_well = min(wells, key=lambda w: distance(current_well, w))
            wells.remove(nearest_well)
            optimized_wells.append(nearest_well)
            current_well = nearest_well
        
        logger.info(f"protocol - greedy well traversal: {len(optimized_wells)} wells")
        return optimized_wells

    def get_optimized_site_order_for_well(self, 
                                          current_well: sc.PlateWellConfig,
                                          last_position: tuple[float, float]) -> list[sc.AcquisitionWellSiteConfigurationSiteSelectionItem]:
        """
        Get sites in optimal traversal order for maximum efficiency within a single well.
        Starts with the site closest to the current physical position and uses greedy traversal.
        
        Args:
            current_well: The well we're about to enter
            last_position: Current physical position (x_mm, y_mm)
        """
        sites = self.well_sites.copy()
        if len(sites) <= 1:
            return sites
        
        def site_distance(s1: sc.AcquisitionWellSiteConfigurationSiteSelectionItem, 
                         s2: sc.AcquisitionWellSiteConfigurationSiteSelectionItem) -> float:
            return math.sqrt((s1.col - s2.col)**2 + (s1.row - s2.row)**2)
        
        def physical_distance_to_site(site: sc.AcquisitionWellSiteConfigurationSiteSelectionItem, 
                                     position: tuple[float, float]) -> float:
            """Calculate physical distance from current position to a site in the target well"""
            # Calculate the physical position of this site
            site_x_mm = (
                self.plate.get_well_offset_x(current_well.well_name)
                + self.site_topleft_x_mm
                + site.col * self.config_file.grid.delta_x_mm
            )
            site_y_mm = (
                self.plate.get_well_offset_y(current_well.well_name)
                + self.site_topleft_y_mm
                + site.row * self.config_file.grid.delta_y_mm
            )
            
            # Calculate distance from current position to this site
            return math.sqrt((site_x_mm - position[0])**2 + (site_y_mm - position[1])**2)
        
        # Find the site closest to our current physical position
        start_site = min(sites, key=lambda s: physical_distance_to_site(s, last_position))
        start_distance = physical_distance_to_site(start_site, last_position)
        
        sites.remove(start_site)
        optimized_sites = [start_site]
        current_site = start_site
        
        # Greedily pick the nearest remaining site within this well
        while sites:
            nearest_site = min(sites, key=lambda s: site_distance(current_site, s))
            sites.remove(nearest_site)
            optimized_sites.append(nearest_site)
            current_site = nearest_site
        
        logger.debug(f"protocol - optimized site order for well {current_well.well_name}: {len(optimized_sites)} sites, starting with site ({start_site.col},{start_site.row}) at distance {start_distance:.2f}mm")
        return optimized_sites

    # pydantics version of dataclass.__post_init__
    def model_post_init(self, __context):
        self.num_wells = len(self.plate_wells)
        self.num_sites = len(self.well_sites)
        self.num_channels = len(self.channels)
        self.num_channel_z_combinations = sum(c.num_z_planes for c in self.channels)
        self.num_timepoints = self.config_file.grid.num_t

        self.num_images_total = self.num_timepoints * self.num_wells * self.num_sites * self.num_channel_z_combinations

        # the grid is centered around the center of the well
        self.site_topleft_x_mm = (
            self.plate.Well_size_x_mm / 2
            - ((self.config_file.grid.num_x - 1) * self.config_file.grid.delta_x_mm) / 2
        )
        "offset of top left site from top left corner of the well, in x, in mm"
        self.site_topleft_y_mm = (
            self.plate.Well_size_y_mm / 2
            - ((self.config_file.grid.num_y - 1) * self.config_file.grid.delta_y_mm) / 2
        )
        "offset of top left site from top left corner of the well, in y, in mm"

        # calculate meta information about acquisition
        cam_img_width = CameraConfig.MAIN_IMAGE_WIDTH_PX.value_item
        assert cam_img_width is not None
        target_width: int = cam_img_width.intvalue
        cam_img_height = CameraConfig.MAIN_IMAGE_HEIGHT_PX.value_item
        assert cam_img_height is not None
        target_height: int = cam_img_height.intvalue
        # get byte size per pixel from config main camera pixel format
        main_cam_pix_format = CameraConfig.MAIN_PIXEL_FORMAT.value_item
        assert main_cam_pix_format is not None
        match main_cam_pix_format.value:
            case "mono8":
                bytes_per_pixel = 1
            case "mono10":
                bytes_per_pixel = 2
            case "mono12":
                bytes_per_pixel = 2
            case "mono14":
                bytes_per_pixel = 2
            case "mono16":
                bytes_per_pixel = 2

            case _unexpected:
                cmds.error_internal(
                    detail=f"unexpected main camera pixel format '{_unexpected}' in {main_cam_pix_format}"
                )

        self.image_size_bytes = target_width * target_height * bytes_per_pixel
        self.max_storage_size_images_GB = self.num_images_total * self.image_size_bytes / 1024**3

        base_storage_path_item = StorageConfig.BASE_IMAGE_OUTPUT_DIR.value_item
        assert base_storage_path_item is not None
        assert type(base_storage_path_item.value) is str
        base_storage_path = path.Path(base_storage_path_item.value)
        assert base_storage_path.exists(), f"{base_storage_path = } does not exist"

        self.project_output_path = (
            base_storage_path
            / self.config_file.project_name
            / f"{self.config_file.plate_name!s}_{sc.datetime2str(dt.datetime.now(dt.UTC))}"
        )
        self.project_output_path.mkdir(parents=True)

        # write config file to output directory
        with (self.project_output_path / "config.json").open("w") as file:
            file.write(self.config_file.json())

        self.acquisition_status.last_status = cmds.AcquisitionStatusOut(
            acquisition_id=self.acquisition_id,
            acquisition_status=cmds.AcquisitionStatusStage.SCHEDULED,
            acquisition_progress=cmds.AcquisitionProgressStatus(
                current_num_images=0,
                time_since_start_s=0,
                start_time_iso="scheduled",
                current_storage_usage_GB=0,
                estimated_remaining_time_s=0,
                last_image=None,
            ),
            acquisition_meta_information=cmds.AcquisitionMetaInformation(
                total_num_images=self.num_images_total,
                max_storage_size_images_GB=self.max_storage_size_images_GB,
            ),
            acquisition_config=self.config_file,
            message="scheduled",
        )

    @logger.catch
    async def generate(
        self,
    ) -> tp.AsyncGenerator[
        # yielded types: None means done, str is returned on first iter, other types are BaseCommands
        None | tp.Literal["ready"] | cmds.BaseCommand,
        # received types (at runtime must match return type of <yielded type>.run().ResultType)
        None | tp.Any,
    ]:
        logger.info(
            f"protocol - initialised. acquiring {self.num_wells} wells, {self.num_sites} sites per well, {self.num_channel_z_combinations} channel+z combinations, i.e. {self.num_images_total} images, taking up to {self.max_storage_size_images_GB:.2f}GB"
        )

        Z_STACK_COUNTER_BACKLASH_MM = 40e-3  # 40um
        PIXEL_SIZE_UM = 900 / 3000  # 3000px wide fov covers 0.9mm
        # movement below this threshold is not performed (0.5um)
        DISPLACEMENT_THRESHOLD_MM: float = 0.5e-3

        # 10um
        UNCOMPENSATED_Z_FAILURE_THRESHOLD_MM = 10e-3

        g_config = basics.GlobalConfigHandler.get_dict()

        # first yield indicates that this generator is ready to produce commands
        # the value from the consumer on the first yield is None
        # i.e. this MUST be the first yield !!
        yield "ready"

        logger.info("protocol - ready for execution")

        yield cmds.EstablishHardwareConnection()

        logger.info("protocol - established hardware connection")

        # counters on acquisition progress
        start_time = time.time()
        start_time_iso_str = sc.datetime2str(dt.datetime.now(dt.UTC))
        last_image_information = None

        num_images_acquired = 0
        storage_usage_bytes = 0

        # get current z coordinate as z reference
        last_position = yield cmds.MC_getLastPosition()
        assert isinstance(last_position, mc.Position), f"{type(last_position)=}"
        reference_z_mm = last_position.z_pos_mm

        # if laser autofocus is enabled, use autofocus z reference as initial z reference
        gconfig_refzmm_item = g_config.get(LaserAutofocusConfig.CALIBRATION_REF_Z_MM.value)
        if self.config_file.autofocus_enabled and gconfig_refzmm_item is not None:
            reference_z_mm = gconfig_refzmm_item.floatvalue

            res = yield cmds.MoveTo(x_mm=None, y_mm=None, z_mm=reference_z_mm)
            assert isinstance(res, cmds.BasicSuccessResponse), f"{type(res)=}"

        # run acquisition:

        # Get optimized well traversal order once (doesn't change between timepoints)
        # Site order will be calculated per-well for maximum efficiency
        optimized_wells = self.get_greedy_well_order()

        # Initialize position tracking (start from reference position if autofocus enabled)
        current_position = (0.0, 0.0)

        is_timeseries_acquisition=self.config_file.grid.num_t>1

        # for each timepoint, starting at 1
        for timepoint in range(1, self.config_file.grid.num_t + 1):
            logger.info(f"protocol - started timepoint {timepoint}/{self.config_file.grid.num_t}")

            # keep track of time
            start_time=time.time()

            self.handle_q_in()

            for well_index, well in enumerate(optimized_wells):
                # these are xy sites
                logger.info(
                    f"protocol - handling next well: {well.well_name} {well_index + 1}/{len(optimized_wells)} (row {well.row}, col {well.col})"
                )

                # Calculate optimal site order for this well based on current physical position
                optimized_sites = self.get_optimized_site_order_for_well(well, current_position)

                for site_index, site in enumerate(optimized_sites):
                    logger.info(f"protocol - handling site {site_index + 1}/{len(optimized_sites)} (row {site.row}, col {site.col})")

                    self.handle_q_in()

                    # go to site
                    site_x_mm = (
                        self.plate.get_well_offset_x(well.well_name)
                        + self.site_topleft_x_mm
                        + site.col * self.config_file.grid.delta_x_mm
                    )
                    site_y_mm = (
                        self.plate.get_well_offset_y(well.well_name)
                        + self.site_topleft_y_mm
                        + site.row * self.config_file.grid.delta_y_mm
                    )

                    res = yield cmds.MoveTo(x_mm=site_x_mm, y_mm=site_y_mm)
                    assert isinstance(res, cmds.BasicSuccessResponse), f"{type(res)=}"
                    
                    # Update current position for distance calculations
                    current_position = (site_x_mm, site_y_mm)

                    logger.debug("protocol - moved to site")

                    # indicate if autofocus was performed, and if it succeeded
                    autofocus_succeeded = False

                    # run autofocus
                    if self.config_file.autofocus_enabled:
                        logger.debug("protocol - performing autofocus")

                        res = None
                        total_compensating_moves = 0
                        AUTOFOCUS_NUM_ATTEMPTS_MAX = 3
                        for autofocus_attempt_num in range(AUTOFOCUS_NUM_ATTEMPTS_MAX):
                            logger.debug(f"protocol - autofocus attempt {autofocus_attempt_num}")

                            # approach target offset
                            res = yield cmds.AutofocusApproachTargetDisplacement(
                                target_offset_um=0,
                                config_file=self.config_file,
                                pre_approach_refz=False,
                            )
                            assert isinstance(
                                res, cmds.AutofocusApproachTargetDisplacementResult
                            ), f"{type(res)=}"

                            logger.debug(
                                f"protocol - autofocus results: {res.num_compensating_moves=} {res.uncompensated_offset_mm=:.3f}"
                            )

                            total_compensating_moves += res.num_compensating_moves

                            # if the final estimated offset is larger than some small threshold, assume it has failed
                            # and reset to some known good-ish z position
                            if (
                                math.fabs(res.uncompensated_offset_mm)
                                > UNCOMPENSATED_Z_FAILURE_THRESHOLD_MM
                            ):
                                mres = yield cmds.MoveTo(x_mm=None, y_mm=None, z_mm=reference_z_mm)
                                assert isinstance(mres, cmds.BasicSuccessResponse), f"{type(mres)=}"
                                logger.debug(
                                    f"protocol - autofocus done after exceeding uncompensated threshold: {res.uncompensated_offset_mm=:.4f} {UNCOMPENSATED_Z_FAILURE_THRESHOLD_MM=:.4f}"
                                )
                                break

                            # if no moves have been performed to compensate for offset, assume we have reached the target offset
                            # (may in practice be still offset, but at least we cannot do better than we have so far)
                            if res.num_compensating_moves == 0 or res.reached_threshold:
                                autofocus_succeeded = True
                                logger.debug(f"protocol - autofocus done {res.reached_threshold=}")
                                break

                        if res is not None:
                            logger.debug(
                                f"{total_compensating_moves=} ; {res.uncompensated_offset_mm=:.4f}"
                            )

                        # reference for channel z offsets
                        last_position = yield cmds.MC_getLastPosition()
                        assert isinstance(last_position, mc.Position), f"{type(last_position)=}"
                        reference_z_mm = last_position.z_pos_mm

                        logger.debug("autofocus performed")
                    else:
                        res = yield cmds.MoveTo(x_mm=None, y_mm=None, z_mm=reference_z_mm)
                        assert isinstance(res, cmds.BasicSuccessResponse), f"{type(res)=}"

                        logger.debug("approached reference z")

                    # Get imaging order configuration
                    imaging_order_item = ImagingConfig.ORDER.value_item
                    assert imaging_order_item is not None
                    imaging_order = tp.cast(ImagingOrder, imaging_order_item.value)
                    
                    # Sort channels according to imaging order configuration
                    ordered_channels = self._sort_channels_by_imaging_order(self.channels, imaging_order)
                    
                    # z stack may be different for each channel, hence:
                    # 1. get list of (channel_z_index,channel,z_relative_to_reference), which may contain each channel more than once  
                    # 2. order by imaging order configuration (z_order/wavelength_order/protocol_order)
                    # 3. move to appropriate starting position, execute imaging sequence
                    # 4. move to reference z again in preparation for next site

                    image_pos_z_list: list[tuple[int, float, sc.AcquisitionChannelConfig]] = []
                    for channel in ordered_channels:
                        channel_delta_z_mm = channel.delta_z_um * 1e-3

                        # <channel reference> is <site reference>+<channel z offset>
                        base_z = reference_z_mm + channel.z_offset_um * 1e-3

                        # lower z base is <channel reference> adjusted for z stack, where \
                        # n-1 z movements are performed, half of those below, half above <channel ref>
                        base_z -= ((channel.num_z_planes - 1) / 2) * channel_delta_z_mm

                        for i in range(channel.num_z_planes):
                            i_offset_mm = i * channel_delta_z_mm
                            target_z_mm = base_z + i_offset_mm
                            image_pos_z_list.append((i, target_z_mm, channel))

                    # Apply final sorting based on imaging order
                    if imaging_order == "z_order":
                        # Sort by z coordinate (fastest - minimizes z movement)
                        image_pos_z_list = sorted(image_pos_z_list, key=lambda v: v[1])
                    else:  # wavelength_order or protocol_order
                        # Keep channel-based ordering (best image quality - minimizes photobleaching)
                        # No additional sorting needed as channels are already ordered correctly
                        pass

                    res = yield cmds.MoveTo(
                        x_mm=None,
                        y_mm=None,
                        z_mm=image_pos_z_list[0][1] - Z_STACK_COUNTER_BACKLASH_MM,
                    )
                    assert isinstance(res, cmds.BasicSuccessResponse), f"{type(res)=}"
                    res = yield cmds.MoveTo(x_mm=None, y_mm=None, z_mm=image_pos_z_list[0][1])
                    assert isinstance(res, cmds.BasicSuccessResponse), f"{type(res)=}"

                    logger.debug("protocol - moved to z order bottom")

                    for plane_index, channel_z_mm, channel in image_pos_z_list:
                        logger.debug(
                            f"protocol - imaging plane {plane_index} at site with {channel_z_mm=:.4f} {channel.name=}"
                        )

                        site_x = site.col
                        site_y = site.row
                        site_z = plane_index

                        self.handle_q_in()

                        # move to channel offset
                        last_position = yield cmds.MC_getLastPosition()
                        assert isinstance(last_position, mc.Position), f"{type(last_position)=}"
                        current_z_mm = last_position.z_pos_mm

                        distance_z_to_move_mm = channel_z_mm - current_z_mm
                        if math.fabs(distance_z_to_move_mm) > DISPLACEMENT_THRESHOLD_MM:
                            res = yield cmds.MoveTo(x_mm=None, y_mm=None, z_mm=channel_z_mm)
                            assert isinstance(res, cmds.BasicSuccessResponse), f"{type(res)=}"

                        last_position = yield cmds.MC_getLastPosition()
                        assert isinstance(last_position, mc.Position), f"{type(last_position)=}"

                        logger.debug(
                            f"protocol - moved to channel z (should be {channel_z_mm:.4f}mm, is {last_position.z_pos_mm:.4f}mm)"
                        )

                        # snap image
                        res = yield cmds.ChannelSnapshot(channel=channel)
                        if not isinstance(res, cmds.ImageAcquiredResponse):
                            cmds.error_internal(
                                detail=f"failed to snap image at site {site} in well {well} (invalid result type {type(res)})"
                            )

                        logger.debug("protocol - took image snapshot")

                        # store image
                        # Choose channel identifier for filename based on configuration
                        use_channel_name = g_config.get("image_filename_use_channel_name")
                        if use_channel_name is None or use_channel_name.boolvalue:
                            # Use channel name with spaces replaced by underscores (default behavior)
                            channel_identifier = channel.name.replace(" ", "_")
                        else:
                            # Use channel handle when explicitly set to "no"
                            channel_identifier = channel.handle
                        
                        # Get index starting values from configuration (with defaults if not present)
                        xy_start = g_config.get("image_filename_xy_index_start")
                        xy_start_value = xy_start.intvalue if xy_start else 0
                        
                        z_start = g_config.get("image_filename_z_index_start") 
                        z_start_value = z_start.intvalue if z_start else 0
                        
                        site_start = g_config.get("image_filename_site_index_start")
                        site_start_value = site_start.intvalue if site_start else 1
                        
                        # Generate filename with configurable index starting values
                        site_num = site_index + site_start_value
                        x_num = site.col + xy_start_value
                        y_num = site.row + xy_start_value  
                        z_num = plane_index + z_start_value
                        
                        # Apply zero-padding to well name if configured
                        zero_pad_column = g_config.get("image_filename_zero_pad_column")
                        if zero_pad_column is None or zero_pad_column.boolvalue:
                            # Default behavior: apply zero-padding (e.g., F8 -> F08)
                            formatted_well_name = _format_well_name_with_padding(well.well_name)
                        else:
                            # Use well name as-is without zero-padding
                            formatted_well_name = well.well_name
                        
                        image_filename=f"{formatted_well_name}_s{site_num}_x{x_num}_y{y_num}_z{z_num}_{channel_identifier}.tiff"
                        if is_timeseries_acquisition:
                            #timepoints start at 1, but we use 0 for the first one in dir
                            assert timepoint>=1
                            timepoint_name=str(timepoint-1)
                            image_storage_path = f"{self.project_output_path!s}/{timepoint_name}/{image_filename}"
                        else:
                            image_storage_path = f"{self.project_output_path!s}/{image_filename}"

                        image_store_entry = cmds.ImageStoreEntry(
                            pixel_format=CameraConfig.MAIN_PIXEL_FORMAT.value_item.strvalue,
                            info=cmds.ImageStoreInfo(
                                channel=channel,
                                width_px=res._img.shape[1],
                                height_px=res._img.shape[0],
                                timestamp=time.time(),
                                position=cmds.SitePosition(
                                    well_name=well.well_name,
                                    site_x=site_x,
                                    site_y=site_y,
                                    site_z=site_z,
                                    x_offset_mm=0,
                                    y_offset_mm=0,
                                    z_offset_mm=0,
                                    position=last_position.pos,
                                ),
                                storage_path=str(image_storage_path),
                            ),
                        )
                        image_store_entry._img = res._img.copy()

                        self.latest_channel_images[channel.name] = image_store_entry

                        # improve system responsiveness while compressing and writing to disk
                        image_store_task = store_image(
                            image_entry=image_store_entry,
                            img_compression_algorithm=self.img_compression_algorithm,
                            metadata={
                                "BitsPerPixel": f"{image_store_entry.bit_depth}",
                                "Make": "CameraMaker",
                                "Model": "CameraModel",
                                "LightSourceName": channel.name,
                                "ExposureTimeMS": f"{channel.exposure_time_ms:.2f}",
                                "AnalogGainDB": f"{channel.analog_gain:.2f}",
                                "PixelSizeUM": f"{PIXEL_SIZE_UM:.3f}",
                                "Position_x_mm": f"{image_store_entry.info.position.position.x_pos_mm:.3f}",
                                "Position_y_mm": f"{image_store_entry.info.position.position.y_pos_mm:.3f}",
                                "Position_z_mm": f"{image_store_entry.info.position.position.z_pos_mm:.3f}",
                                "autofocus_succeeded": f"{autofocus_succeeded}",
                            },
                        )
                        self.image_store_pool.run(image_store_task)

                        logger.debug(
                            f"protocol - scheduled image store to {image_store_entry.info.storage_path}"
                        )

                        num_images_acquired += 1
                        # get storage size from filesystem because tiff compression may reduce size below size in memory
                        try:
                            file_size_on_disk = path.Path(image_storage_path).stat().st_size
                            storage_usage_bytes += file_size_on_disk
                        except:
                            # ignore any errors here, because this is not an essential feature
                            pass

                        # status items
                        last_image_information = image_store_entry.info

                        time_since_start_s = time.time() - start_time
                        if num_images_acquired > 0:
                            estimated_remaining_time_s = (
                                time_since_start_s
                                * (self.num_images_total - num_images_acquired)
                                / num_images_acquired
                            )
                        else:
                            estimated_remaining_time_s = None

                        logger.debug(
                            f"protocol - {num_images_acquired}/{self.num_images_total} images acquired"
                        )

                        self.acquisition_status.last_status = cmds.AcquisitionStatusOut(
                            acquisition_id=self.acquisition_id,
                            acquisition_status=cmds.AcquisitionStatusStage.RUNNING,
                            acquisition_progress=cmds.AcquisitionProgressStatus(
                                # measureable progress
                                current_num_images=num_images_acquired,
                                time_since_start_s=time_since_start_s,
                                start_time_iso=start_time_iso_str,
                                current_storage_usage_GB=storage_usage_bytes / (1024**3),
                                # estimated completion time information
                                # estimation may be more complex than linear interpolation, hence done on server side
                                estimated_remaining_time_s=estimated_remaining_time_s,
                                # last image that was acquired
                                last_image=last_image_information,
                            ),
                            # some meta information about the acquisition, derived from configuration file
                            # i.e. this is not updated during acquisition
                            acquisition_meta_information=cmds.AcquisitionMetaInformation(
                                total_num_images=self.num_images_total,
                                max_storage_size_images_GB=self.max_storage_size_images_GB,
                            ),
                            acquisition_config=self.config_file,
                            message=f"Acquisition is {(100 * num_images_acquired / self.num_images_total):.2f}% complete",
                        )

            # sleep for additional time, maybe
            end_time=time.time()
            if self.config_file.grid.num_t>timepoint:
                time_elapsed=end_time-start_time

                delta_t_s=self.config_file.grid.delta_t.h*3600+self.config_file.grid.delta_t.m*60+self.config_file.grid.delta_t.s

                # sleep through remaining time in small intervals to process inputs (e.g. cancel acquisition)
                time_remaining=delta_t_s-time_elapsed

                logger.debug(f"protocol - time remaining before starting next time series acquisition: {time_remaining}s")

                timeslice_s=0.5
                while time_remaining>0:
                    self.handle_q_in()

                    await asyncio.sleep(min(time_remaining,timeslice_s))
                    time_remaining-=timeslice_s

        logger.debug("protocol - finished protocol steps")

        # wait for image storage tasks to finish
        self.image_store_pool.join()

        logger.debug("protocol - finished image storage stasks")

        logger.info("protocol - done")

        # done -> yield None and return
        yield None
