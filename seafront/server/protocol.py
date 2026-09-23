import asyncio
import asyncio as aio
import datetime as dt
import collections
import functools
import math
import pathlib as path
import random
import re
import string
import time
import typing as tp
import uuid
from concurrent.futures import Future as ConcurrentFuture
from threading import Thread

import seaconfig as sc
from ome_types import OME
from ome_types.model import (
    Channel,
    Detector,
    Image,
    Instrument,
    Laser,
    LightEmittingDiode,
    LightSourceSettings,
    Microscope,
    Microscope_Type,
    Objective,
    Objective_Correction,
    Objective_Immersion,
    ObjectiveSettings,
    Pixels,
    Plane,
    PixelType,
    Pixels_DimensionOrder,
    StageLabel,
    UnitsLength,
    Detector_Type,
    TiffData,
)
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

import seafront.server.commands as cmds
from seafront.config import basics
from seafront.config.basics import ImagingOrder
from seafront.config.handles import (
    CameraConfig,
    ImageConfig,
    ImagingConfig,
    LaserAutofocusConfig,
    StorageConfig,
    SystemConfig,
)
from seafront.config.registry import ConfigRegistry
from seafront.hardware.adapter import Position as McPosition
from seafront.logger import logger
from seafront.storage import ImageStore, encode_ome_tiff, image_store_from_config
from seafront.storage.base import CODEC_THREADS, STRIP_ROWS


# Supporting data classes for protocol iterators
class WellInfo(BaseModel):
    """Information about a well with metadata for protocol iteration"""
    well: sc.PlateWellConfig
    well_index: int
    physical_center: tuple[float, float]
    "Physical coordinates of well center (x_mm, y_mm)"


class PositionInfo(BaseModel):
    """Information about a specific well+site position with metadata"""
    well: sc.PlateWellConfig
    site: sc.AcquisitionWellSiteConfigurationSiteSelectionItem
    well_index: int
    site_index: int
    physical_position: tuple[float, float]
    "Physical coordinates of this site position (x_mm, y_mm)"

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ChannelInfo(BaseModel):
    """Information about a channel with imaging order metadata"""
    channel: sc.AcquisitionChannelConfig
    channel_index: int
    imaging_order: ImagingOrder

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ZPositionInfo(BaseModel):
    """Information about a z-position with channel metadata"""
    plane_index: int
    z_position_mm: float
    channel: sc.AcquisitionChannelConfig
    channel_info: ChannelInfo

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AcquisitionStatusReporter:
    """Build and publish status snapshots; independent of protocol traversal and storage."""

    def __init__(
        self,
        target: cmds.AcquisitionStatus,
        acquisition_id: str,
        config: sc.AcquisitionConfig,
        total_num_images: int,
        max_storage_size_images_gb: float,
    ) -> None:
        self._target = target
        self._acquisition_id = acquisition_id
        self._config = config
        self._total_num_images = total_num_images
        self._max_storage_size_images_gb = max_storage_size_images_gb
        self._num_autofocus_misses = 0
        self._recent_autofocus_misses: collections.deque[cmds.AutofocusMiss] = collections.deque(
            maxlen=self.KEEP_RECENT_AUTOFOCUS_MISSES
        )

    KEEP_RECENT_AUTOFOCUS_MISSES = 10
    "each status carries these, and a status is polled continuously: the count covers the rest"

    def autofocus_missed(self, miss: cmds.AutofocusMiss) -> None:
        "record a site imaged out of focus, and show it now rather than with the next image"
        self._num_autofocus_misses += 1
        self._recent_autofocus_misses.append(miss)
        last = self._target.last_status
        if last is not None:
            last.acquisition_progress = last.acquisition_progress.model_copy(
                update=self._autofocus_fields()
            )

    def _autofocus_fields(self) -> dict[str, tp.Any]:
        return {
            "num_autofocus_misses": self._num_autofocus_misses,
            "recent_autofocus_misses": list(self._recent_autofocus_misses),
        }

    def publish(
        self,
        *,
        status: cmds.AcquisitionStatusStage,
        current_num_images: int,
        time_since_start_s: float,
        start_time_iso: str,
        storage_usage_bytes: int,
        estimated_remaining_time_s: float | None,
        last_image: cmds.ImageStoreInfo | None,
        message: str,
        current_timepoint: int | None = None,
        total_timepoints: int | None = None,
        estimated_timepoint_duration_s: float | None = None,
    ) -> None:
        self._target.last_status = cmds.AcquisitionStatusOut(
            acquisition_id=self._acquisition_id,
            acquisition_status=status,
            acquisition_progress=cmds.AcquisitionProgressStatus(
                current_num_images=current_num_images,
                time_since_start_s=time_since_start_s,
                start_time_iso=start_time_iso,
                current_storage_usage_GB=storage_usage_bytes / (1024**3),
                estimated_remaining_time_s=estimated_remaining_time_s,
                last_image=last_image,
                current_timepoint=current_timepoint,
                total_timepoints=total_timepoints,
                estimated_timepoint_duration_s=estimated_timepoint_duration_s,
                **self._autofocus_fields(),
            ),
            acquisition_meta_information=cmds.AcquisitionMetaInformation(
                total_num_images=self._total_num_images,
                max_storage_size_images_GB=self._max_storage_size_images_gb,
            ),
            acquisition_config=self._config,
            message=message,
        )

    def scheduled(self) -> None:
        self.publish(
            status=cmds.AcquisitionStatusStage.SCHEDULED,
            current_num_images=0,
            time_since_start_s=0,
            start_time_iso="scheduled",
            storage_usage_bytes=0,
            estimated_remaining_time_s=0,
            last_image=None,
            message="scheduled",
        )


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


class TimeEstimate(tp.NamedTuple):
    """
    what the progress display needs: how much longer, and how long one timepoint takes

    `remaining_s` is None until the first image, because there is no rate to extrapolate
    from. `timepoint_duration_s` is None for a run that is not a time series, where the idea
    does not apply.
    """

    remaining_s: float | None
    timepoint_duration_s: float | None


def estimate_remaining_time(
    *,
    num_images_acquired: int,
    num_images_total: int,
    time_since_start_s: float,
    num_timepoints: int,
    current_timepoint: int,
    delta_t_s: float,
    completed_timepoint_imaging_times: list[float],
) -> TimeEstimate:
    """
    How much longer the run has, from how long it has taken so far.

    This is the number on the operator's screen for the hours a plate takes, so it is worth
    it being roughly right: it decides whether somebody waits or goes home.

    None until the first image, because there is no rate to extrapolate from yet.

    A time series is not just more images. Imaging a timepoint and then waiting for the next
    one are separate costs, and the wait is whatever is left of `delta_t` after the imaging -
    nothing, if imaging already takes longer than the interval. Once a timepoint has
    completed, its measured time is a better estimate for the rest than anything derived from
    a partial one, so it takes over.
    """
    if num_images_acquired <= 0:
        return TimeEstimate(remaining_s=None, timepoint_duration_s=None)

    images_remaining = num_images_total - num_images_acquired
    time_per_image = time_since_start_s / num_images_acquired

    if num_timepoints <= 1:
        # Single timepoint: simple linear interpolation
        return TimeEstimate(
            remaining_s=time_per_image * images_remaining, timepoint_duration_s=None
        )

    images_per_tp = num_images_total / num_timepoints

    if completed_timepoint_imaging_times:
        # Use average of completed timepoint times for all remaining imaging
        est_imaging_per_tp = sum(completed_timepoint_imaging_times) / len(
            completed_timepoint_imaging_times
        )
        tps_remaining = images_remaining / images_per_tp
        estimate = est_imaging_per_tp * tps_remaining
    else:
        # No completed TPs yet: estimate from current progress
        est_imaging_per_tp = time_per_image * images_per_tp
        estimate = time_per_image * images_remaining

    # Add wait periods
    waits_remaining = num_timepoints - current_timepoint
    wait_per_cycle = max(0, delta_t_s - est_imaging_per_tp)
    return TimeEstimate(
        remaining_s=estimate + waits_remaining * wait_per_cycle,
        timepoint_duration_s=est_imaging_per_tp,
    )


def image_filename_for(
    *,
    well_name: str,
    site: sc.AcquisitionWellSiteConfigurationSiteSelectionItem,
    site_index: int,
    plane_index: int,
    channel: sc.AcquisitionChannelConfig,
    g_config: dict[str, basics.ConfigItem],
) -> str:
    """
    The name one acquired frame is stored under, and the well name used in it.

    Each OME-TIFF here holds a single plane and declares SizeZ=SizeC=SizeT=1, with no
    TiffData UUID tying it to its siblings - so this name is the only thing connecting the
    files of one acquisition to each other. Whatever reassembles a z-stack downstream is
    parsing it, which is why the indices are configurable and why changing the shape is a
    breaking change for somebody.

    """
    # Choose channel identifier for filename based on configuration
    use_channel_name = g_config.get(ImageConfig.FILENAME_USE_CHANNEL_NAME.value)
    if use_channel_name is None or use_channel_name.boolvalue:
        # Use channel name with spaces replaced by underscores (default behavior)
        channel_identifier = channel.name.replace(" ", "_")
    else:
        # Use channel handle when explicitly set to "no"
        channel_identifier = channel.handle

    # Get index starting values from configuration (with defaults if not present)
    xy_start = g_config.get(ImageConfig.FILENAME_XY_INDEX_START.value)
    xy_start_value = xy_start.intvalue if xy_start else 0

    z_start = g_config.get(ImageConfig.FILENAME_Z_INDEX_START.value)
    z_start_value = z_start.intvalue if z_start else 0

    site_start = g_config.get(ImageConfig.FILENAME_SITE_INDEX_START.value)
    site_start_value = site_start.intvalue if site_start else 1

    site_num = site_index + site_start_value
    x_num = site.col + xy_start_value
    y_num = site.row + xy_start_value
    z_num = plane_index + z_start_value

    # Apply zero-padding to well name if configured
    zero_pad_column = g_config.get(ImageConfig.FILENAME_ZERO_PAD_COLUMN.value)
    if zero_pad_column is None or zero_pad_column.boolvalue:
        # Default behavior: apply zero-padding (e.g., F8 -> F08)
        formatted_well_name = _format_well_name_with_padding(well_name)
    else:
        # Use well name as-is without zero-padding
        formatted_well_name = well_name

    filename = (
        f"{formatted_well_name}_s{site_num}_x{x_num}_y{y_num}_z{z_num}"
        f"_{channel_identifier}.ome.tiff"
    )
    return filename


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


def extract_wavelength_nm(channel_name: str) -> int | None:
    """
    Extract wavelength in nm from channel name using regex.

    Examples:
        "Fluorescence 405 nm Ex" -> 405
        "BF LED matrix full" -> None
    """
    match = re.search(r'(\d+)\s*nm', channel_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def build_ome_instrument(
    microscope_name: str,
    camera_vendor: str,
    camera_model: str,
    camera_sn: str,
    hardware_channels: list[basics.ChannelConfig],
    objective_magnification: int | None = None,
) -> OME:
    """
    Build OME instrument structure with microscope-wide metadata.

    This function creates the reusable portions of the OME structure that are
    identical across all images in an acquisition: Microscope, Detector, Objective,
    and all light sources. The returned OME object can be reused to build multiple
    image OME-TIFF files efficiently.

    Args:
        microscope_name: Name of the microscope
        camera_vendor: Camera manufacturer name
        camera_model: Camera model name
        camera_sn: Camera serial number
        hardware_channels: List of available hardware channels for wavelength lookup
        objective_magnification: Objective magnification (e.g., 4, 10, 20)

    Returns:
        OME object with UUID and Instrument pre-populated
    """
    # Create OME root with unique UUID for this template
    # Each OME-TIFF file will get its own UUID when serialized
    ome_uuid = f"urn:uuid:{uuid.uuid4()}"
    ome = OME(creator="Seafront", uuid=ome_uuid)

    # Add light sources for all channels and separate by type
    lasers = []
    leds = []
    for i, hw_channel in enumerate(hardware_channels):
        # Determine light source type from slot
        source_slot = hw_channel.source_slot
        wavelength_nm = extract_wavelength_nm(hw_channel.name)

        if 0 <= source_slot <= 10:
            # LED matrix
            light_source = LightEmittingDiode(
                id=f"LightSource:{i}",
                manufacturer=None,
                model=None,
                serial_number=None,
            )
            leds.append(light_source)
        elif 11 <= source_slot <= 15:
            # Laser
            light_source = Laser(
                id=f"LightSource:{i}",
                manufacturer=None,
                model=None,
                serial_number=None,
                wavelength=wavelength_nm,  # Laser has wavelength field
            )
            lasers.append(light_source)
        else:
            # Generic light source (treat as LED)
            light_source = LightEmittingDiode(
                id=f"LightSource:{i}",
                manufacturer=None,
                model=None,
                serial_number=None,
            )
            leds.append(light_source)

    # Create instrument with detector and light sources
    instrument = Instrument(id="Instrument:0")

    # Add microscope information
    # SQUID microscope is inverted (sample on top, objective below)
    microscope = Microscope(
        manufacturer="Cephla",
        model=microscope_name,
        type=Microscope_Type.INVERTED,
    )
    instrument.microscope = microscope

    # Add objective if magnification is available
    if objective_magnification is not None:
        objective = Objective(
            id="Objective:0",
            manufacturer="Olympus",
            nominal_magnification=float(objective_magnification),
            correction=Objective_Correction.PLAN_APO,  # Olympus objectives are typically plan-apochromatic
            immersion=Objective_Immersion.AIR,         # SQUID uses air objectives
        )
        instrument.objectives = [objective]

    # Add detector (main camera)
    detector = Detector(
        id="Detector:0",
        manufacturer=camera_vendor if camera_vendor else None,
        model=camera_model if camera_model else None,
        serial_number=camera_sn if camera_sn else None,
        type=Detector_Type.CCD,
    )
    instrument.detectors = [detector]

    # Add light sources to instrument
    if lasers:
        instrument.lasers = lasers
    if leds:
        instrument.light_emitting_diodes = leds

    ome.instruments = [instrument]

    return ome


MAIN_CAMERA_PIXEL_PITCH_UM = 6.0
"""
The main camera's pixel pitch projected back through the optics, in micrometres.

Anchored on the one magnification this was ever measured at: with the 20x objective a
3000 px wide field covers 0.9 mm, i.e. 0.3 um per pixel at the sample, so the pitch is
0.3 * 20. Every other objective follows from it by `pixel_size_um_at`.
"""


def pixel_size_um_at(objective_magnification: int | None) -> float | None:
    """
    Pixel size at the sample, in micrometres, for the objective currently in the light path.

    None when the magnification is unknown, and the caller must then leave PhysicalSize out
    of the metadata rather than guess. This used to be a single hardcoded 0.3, which is the
    20x value: a 4x acquisition wrote files claiming a 4x objective and 0.3 um pixels at the
    same time, and every distance measured off them in micrometres was out by a factor of
    five.
    """
    if objective_magnification is None or objective_magnification <= 0:
        return None
    return MAIN_CAMERA_PIXEL_PITCH_UM / objective_magnification


def significant_bits_for(camera_bits: int, container_bits: int, pad_low: bool) -> int:
    """
    What SignificantBits should say for samples this camera produced.

    OME gives no way to state where the significant bits sit inside the container, so a reader
    has only one interpretation available: the value fits in N bits, i.e. the range is
    0..2**N-1, low-aligned. That is true of the camera's own samples, so normally this is just
    the camera's bit depth.

    With `image.file.pad_low` the adapter shifts the samples up to fill the container. They
    then really do span the container's range - in coarse steps, but a viewer wanting a
    display range is right to use all of it - so the honest answer becomes the container
    width. Claiming the camera's depth there would tell a reader to stretch 0..4095 across a
    frame whose values reach 65520, and everything would clip to white.
    """
    return container_bits if pad_low and container_bits != camera_bits else camera_bits


def build_ome_metadata(
    ome_template: OME,
    image_entry: cmds.ImageStoreEntry,
    channel_config: sc.AcquisitionChannelConfig,
    position: cmds.SitePosition,
    plane_index: int,
    pixel_size_um: float | None,
    significant_bits: int | None,
    project_name: str,
    plate_name: str,
    cell_line: str,
    hardware_channels: list[basics.ChannelConfig],
    autofocus_succeeded: bool = False,
) -> str:
    """
    Build OME-XML metadata string for an acquired image.

    Takes a pre-constructed OME template with microscope-wide metadata
    (Instrument, Microscope, Detector, Objectives, LightSources) and adds
    image-specific metadata (Image, Pixels, Channel, Plane, TiffData).

    Args:
        ome_template: Pre-built OME object with Instrument populated
        image_entry: Image data and metadata
        channel_config: Channel acquisition settings
        position: Stage position and well information
        plane_index: Z-plane index (0-based)
        pixel_size_um: Physical pixel size at the sample in micrometers, or None when the
            objective is unknown - in which case PhysicalSize is left out rather than guessed
        significant_bits: how many bits of each stored sample carry data (see
            significant_bits_for); None leaves the attribute out, which OME reads as the
            full width of the type
        project_name: Project/experiment name
        plate_name: Plate identifier
        cell_line: Cell line identifier
        hardware_channels: List of available hardware channels for wavelength lookup
        autofocus_succeeded: Whether autofocus succeeded for this plane

    Returns:
        OME-XML string with complete metadata hierarchy
    """
    # Start with the template and add image-specific elements
    # Generate new UUID for this specific OME-TIFF file
    ome_uuid = f"urn:uuid:{uuid.uuid4()}"
    ome = OME(creator="Seafront", uuid=ome_uuid)
    # Copy instrument from template
    ome.instruments = ome_template.instruments

    # Find hardware channel info for current channel
    hw_channel_info = next(
        (ch for ch in hardware_channels if ch.handle == channel_config.handle),
        None
    )
    wavelength_nm = (
        extract_wavelength_nm(hw_channel_info.name)
        if hw_channel_info else None
    )

    # Determine pixel type for OME
    # The container the samples ride in, which follows from their depth. This was a table of
    # the five formats the drivers enumerate today, defaulting to UINT8 - so a depth nobody
    # had listed was silently declared 8-bit while the array beside it was 16, and a reader
    # trusting the declaration would have read every image wrong.
    pixel_type = PixelType.UINT8 if image_entry.bit_depth <= 8 else PixelType.UINT16

    # Create pixels element
    # Note: Each TIFF file contains a single 2D image plane, so size_z=1
    # (not the full Z-stack size). The Z-position is indicated in the Plane element
    # and the file name, not in the Pixels dimensions.
    pixels = Pixels(
        id="Pixels:0",
        dimension_order=Pixels_DimensionOrder.XYCZT,
        size_x=image_entry.info.width_px,
        size_y=image_entry.info.height_px,
        size_c=1,  # Single channel per image
        size_z=1,  # Single plane per file
        size_t=1,  # No time dimension in single acquisition
        type=pixel_type,
        physical_size_x=pixel_size_um,
        physical_size_y=pixel_size_um,
        physical_size_z=channel_config.delta_z_um if channel_config.delta_z_um > 0 else None,
        significant_bits=significant_bits,
    )

    # Create channel
    channel = Channel(
        id="Channel:0:0",
        name=channel_config.name,
        samples_per_pixel=1,
    )

    # Add light source settings if we found the hardware channel
    if hw_channel_info:
        light_source_index = hardware_channels.index(hw_channel_info)
        light_source_settings = LightSourceSettings(
            id=f"LightSource:{light_source_index}",
            wavelength=wavelength_nm,
            attenuation=channel_config.illum_perc / 100.0,  # Convert 0-100 to 0-1
        )
        channel.light_source_settings = light_source_settings

    pixels.channels = [channel]

    # Create plane with detailed acquisition information
    # Note: the_z=0 because this file contains only one plane (local index 0)
    # The global Z-position is stored in position_z and in the image name
    plane = Plane(
        the_z=0,  # Local index in this file (always 0 for single-plane TIFF)
        the_c=0,  # Local channel index
        the_t=0,  # Local time index
        exposure_time=channel_config.exposure_time_ms / 1000.0,  # Convert ms to seconds
        position_x=position.position.x_pos_mm,
        position_y=position.position.y_pos_mm,
        position_z=position.position.z_pos_mm,
        # The driver works in millimetres and these come straight from it. Saying so is not
        # optional: OME defaults an unset position unit to "reference frame", which declares
        # an undefined coordinate system rather than a length, and there is no way back to
        # millimetres from that. Anything stitching a plate from these was reading bare
        # numbers.
        position_x_unit=UnitsLength.MILLIMETER,
        position_y_unit=UnitsLength.MILLIMETER,
        position_z_unit=UnitsLength.MILLIMETER,
    )

    # Create TiffData block to explicitly link Plane to TIFF IFD
    # This helps ImageJ and other OME tools properly associate metadata with image data
    tiffdata = TiffData(
        ifd=0,  # Point to the first (and only) TIFF IFD in this file
        first_z=0,  # Local index in this file (always 0 for single-plane TIFF)
        first_c=0,  # First channel index
        first_t=0,  # First time index
        plane_count=1,
    )

    pixels.tiff_data_blocks = [tiffdata]
    pixels.planes = [plane]

    # Extract microscope information from template
    microscope_name = ""
    has_objective = False
    if ome_template.instruments and len(ome_template.instruments) > 0:
        instrument = ome_template.instruments[0]
        if instrument.microscope:
            microscope_name = instrument.microscope.model or ""
        has_objective = bool(instrument.objectives) and len(instrument.objectives) > 0

    # Create image with pixels
    image = Image(
        id="Image:0",
        name=f"{position.well_name}_{channel_config.handle}_z{plane_index:03d}",
        acquisition_date=dt.datetime.fromtimestamp(image_entry.info.timestamp, dt.UTC),
        pixels=pixels,
    )

    # Create stage label with well and site information
    stage_label = StageLabel(
        name=position.well_name,
        x=position.position.x_pos_mm,
        y=position.position.y_pos_mm,
        z=position.position.z_pos_mm,
        # same defaulting trap as the Plane above
        x_unit=UnitsLength.MILLIMETER,
        y_unit=UnitsLength.MILLIMETER,
        z_unit=UnitsLength.MILLIMETER,
    )
    image.stage_label = stage_label

    # Link objective settings to image if objective was defined
    if has_objective:
        image.objective_settings = ObjectiveSettings(id="Objective:0")

    # Add image description with experiment context
    image.description = (
        f"Project: {project_name}\n"
        f"Plate: {plate_name}\n"
        f"Cell Line: {cell_line}\n"
        f"Well: {position.well_name}\n"
        f"Site: ({position.site_x}, {position.site_y}, {position.site_z})\n"
        f"Channel: {channel_config.handle}\n"
        f"Z-Plane: {plane_index}/{channel_config.num_z_planes}\n"
        f"Microscope: {microscope_name}\n"
        f"Analog Gain: {channel_config.analog_gain:.2f} dB\n"
        f"Autofocus: {'Success' if autofocus_succeeded else 'N/A'}"
    )

    ome.images = [image]

    # Convert OME object to XML string
    # TIFF ASCII descriptions do not preserve trailing whitespace on readback. Normalize at
    # the sole construction point so HTTP, WebSocket, and stored TIFF retrieval are identical.
    return ome.to_xml().rstrip()


def finalize_capture_ome_xml(
    capture: cmds.CapturedImageMetadata,
    context: cmds.OmeMetadataContext,
    camera: tp.Any,
    hardware_channels: list[basics.ChannelConfig],
) -> str:
    """Finalize OME-XML beside the pixels, while every capture fact is authoritative."""
    objective_magnification: int | None = None
    objective = ConfigRegistry.get(CameraConfig.MAIN_OBJECTIVE).strvalue
    if match := re.match(r"(\d+)x", objective):
        objective_magnification = int(match.group(1))

    position = cmds.SitePosition(
        well_name=context.well_name,
        site_x=context.site_x,
        site_y=context.site_y,
        site_z=context.site_z,
        x_offset_mm=context.x_offset_mm,
        y_offset_mm=context.y_offset_mm,
        z_offset_mm=context.z_offset_mm,
        position=capture.position,
    )
    entry = cmds.ImageStoreEntry(
        pixel_format=capture.pixel_format,
        info=cmds.ImageStoreInfo(
            channel=capture.channel,
            width_px=capture.width_px,
            height_px=capture.height_px,
            timestamp=capture.timestamp,
            image_id=capture.image_id,
            position=position,
        ),
        capture=capture,
    )
    template = build_ome_instrument(
        microscope_name=ConfigRegistry.get(SystemConfig.MICROSCOPE_NAME).strvalue,
        camera_vendor=str(getattr(camera, "vendor_name", "")),
        camera_model=str(getattr(camera, "model_name", "")),
        camera_sn=str(getattr(camera, "sn", "")),
        hardware_channels=hardware_channels,
        objective_magnification=objective_magnification,
    )
    return build_ome_metadata(
        ome_template=template,
        image_entry=entry,
        channel_config=capture.channel,
        position=position,
        plane_index=context.plane_index,
        pixel_size_um=pixel_size_um_at(objective_magnification),
        significant_bits=significant_bits_for(
            camera_bits=capture.camera_bit_depth,
            container_bits=capture.bit_depth,
            pad_low=ConfigRegistry.get(ImageConfig.FILE_PAD_LOW).boolvalue,
        ),
        project_name=context.project_name,
        plate_name=context.plate_name,
        cell_line=context.cell_line,
        hardware_channels=hardware_channels,
        autofocus_succeeded=context.autofocus_succeeded,
    )


def encode_image(
    image_entry: cmds.ImageStoreEntry,
    img_compression_algorithm: tp.Literal["LZW", "zlib"],
    ome_xml: str,
    strip_rows: int = STRIP_ROWS,
    codec_threads: int = CODEC_THREADS,
) -> bytes:
    """
    OME-TIFF bytes for one acquired image (OME-XML in ImageDescription, lossless compression).

    Pure: runs on an image-writer thread; the ImageStore's sink decides where the bytes go.
    Shared with scripts/acquire_via_api.py so both write identical files.
    """
    # takes 10-250ms depending on image size, compression and strip layout
    return encode_ome_tiff(image_entry._img, ome_xml, img_compression_algorithm, strip_rows, codec_threads)


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

    image_store: ImageStore | None = None
    "built from machine config in model_post_init (backend, writer threads); closed at the end of generate()"

    img_compression_algorithm: tp.Literal["LZW", "zlib"] = "LZW"

    # Microscope context for OME metadata
    ome_template: OME = Field(...)
    "Pre-built OME template with Instrument, Microscope, Detector, Objectives, LightSources"
    hardware_channels: list[basics.ChannelConfig] = Field(default_factory=list)
    "List of available hardware channels with wavelength info"
    objective_magnification: int | None = None
    "the objective in the light path, which is what sets the pixel size at the sample"

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
    project_relpath: path.Path = Field(default_factory=path.Path)
    "<project>/<plate>_<timestamp>, relative to the storage backend's base (directory or bucket prefix)"

    latest_channel_images: dict[str, cmds.ImageStoreEntry] = Field(default_factory=dict)
    completed_timepoint_imaging_times: list[float] = Field(default_factory=list)
    "Actual imaging times for completed timepoints (used to estimate remaining ones)"

    _status_reporter: AcquisitionStatusReporter = PrivateAttr()

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

    def get_physical_position_for_well_site(self, well: sc.PlateWellConfig, site: sc.AcquisitionWellSiteConfigurationSiteSelectionItem) -> tuple[float, float]:
        """
        Get physical coordinates for a specific well+site combination.

        This is the single source of truth for position calculation, used by both
        the optimization algorithms and the iterator methods.

        Args:
            well: The well configuration
            site: The site within the well

        Returns:
            Tuple of (x_mm, y_mm) physical coordinates
        """
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
        return (site_x_mm, site_y_mm)

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
            # Use the single source of truth for position calculation
            site_x_mm, site_y_mm = self.get_physical_position_for_well_site(current_well, site)

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
        # Use shared metrics calculation
        metrics = cmds.calculate_acquisition_metrics(self.config_file)

        self.num_wells = metrics.num_wells
        self.num_sites = metrics.num_sites
        self.num_channels = metrics.num_channels
        self.num_channel_z_combinations = metrics.num_z_planes_total
        self.num_timepoints = metrics.num_timepoints
        self.num_images_total = metrics.total_num_images
        self.image_size_bytes = metrics.image_size_bytes
        self.max_storage_size_images_GB = metrics.max_storage_size_GB

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

        self.project_relpath = path.Path(self.config_file.project_name) / (
            f"{self.config_file.plate_name!s}_{sc.datetime2str(dt.datetime.now(dt.UTC))}"
        )
        self._status_reporter = AcquisitionStatusReporter(
            target=self.acquisition_status,
            acquisition_id=self.acquisition_id,
            config=self.config_file,
            total_num_images=self.num_images_total,
            max_storage_size_images_gb=self.max_storage_size_images_GB,
        )
        self._status_reporter.scheduled()

    # Iterator methods for protocol traversal
    def iter_wells(self) -> tp.Iterator[WellInfo]:
        """Iterator over selected wells with metadata"""
        for well_index, well in enumerate(self.plate_wells):
            yield WellInfo(
                well=well,
                well_index=well_index,
                physical_center=(
                    self.plate.get_well_offset_x(well.well_name),
                    self.plate.get_well_offset_y(well.well_name)
                )
            )

    def iter_positions(self) -> tp.Iterator[PositionInfo]:
        """Iterator over all well+site combinations with physical positions"""
        for well_info in self.iter_wells():
            for site_index, site in enumerate(self.well_sites):
                # Use single source of truth for position calculation
                physical_position = self.get_physical_position_for_well_site(well_info.well, site)
                yield PositionInfo(
                    well=well_info.well,
                    site=site,
                    well_index=well_info.well_index,
                    site_index=site_index,
                    physical_position=physical_position
                )

    def iter_channels(self) -> tp.Iterator[ChannelInfo]:
        """Iterator over channels with imaging order applied"""
        imaging_order_item = ConfigRegistry.get(ImagingConfig.ORDER)
        assert imaging_order_item is not None
        imaging_order = tp.cast(ImagingOrder, imaging_order_item.value)

        ordered_channels = self._sort_channels_by_imaging_order(self.channels, imaging_order)
        for channel_index, channel in enumerate(ordered_channels):
            yield ChannelInfo(
                channel=channel,
                channel_index=channel_index,
                imaging_order=imaging_order
            )

    def iter_z_positions(self, reference_z_mm: float) -> tp.Iterator[ZPositionInfo]:
        """Iterator over Z positions with channel information and imaging order"""
        # Encapsulate the complex z-stacking logic from the generate method
        image_pos_z_list = []

        for channel_info in self.iter_channels():
            channel = channel_info.channel
            channel_delta_z_mm = channel.delta_z_um * 1e-3

            # <channel reference> is <site reference>+<channel z offset>
            base_z = reference_z_mm + channel.z_offset_um * 1e-3

            # lower z base is <channel reference> adjusted for z stack, where
            # n-1 z movements are performed, half of those below, half above <channel ref>
            base_z -= ((channel.num_z_planes - 1) / 2) * channel_delta_z_mm

            for plane_index in range(channel.num_z_planes):
                i_offset_mm = plane_index * channel_delta_z_mm
                target_z_mm = base_z + i_offset_mm
                image_pos_z_list.append((plane_index, target_z_mm, channel, channel_info))

        # Apply final sorting based on imaging order
        if len(image_pos_z_list) > 0 and image_pos_z_list[0][3].imaging_order == "z_order":
            # Sort by z coordinate (fastest - minimizes z movement)
            image_pos_z_list = sorted(image_pos_z_list, key=lambda v: v[1])

        for plane_index, z_position_mm, channel, channel_info in image_pos_z_list:
            yield ZPositionInfo(
                plane_index=plane_index,
                z_position_mm=z_position_mm,
                channel=channel,
                channel_info=channel_info
            )

    async def generate(
        self,
    ) -> tp.AsyncGenerator[
        None | tp.Literal["ready"] | cmds.BaseCommand,
        None | tp.Any,
    ]:
        """Yield hardware commands while retaining ownership of persistent storage.

        Core is deliberately only the command executor.  In particular, closing an async
        generator (the cancellation/error path) must also drain and close this protocol's
        writers; callers do not need access to the ImageStore to clean up correctly.
        """
        commands = self._generate_commands()
        send_value: tp.Any = None
        pending_error: BaseException | None = None
        try:
            while True:
                try:
                    if pending_error is None:
                        command = await commands.asend(send_value)
                    else:
                        error = pending_error
                        pending_error = None
                        command = await commands.athrow(error)
                except StopAsyncIteration:
                    return

                try:
                    send_value = yield command
                except BaseException as error:
                    send_value = None
                    pending_error = error
        finally:
            await commands.aclose()
            if self.image_store is not None:
                self.image_store.close()

    @logger.catch
    async def _generate_commands(
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
        pixel_size_um = pixel_size_um_at(self.objective_magnification)
        if pixel_size_um is None:
            logger.warning(
                "protocol - objective magnification is unknown, so the images will carry no "
                "PhysicalSize. Better a missing scale than a wrong one, but check "
                f"{CameraConfig.MAIN_OBJECTIVE.value} in the config."
            )
        else:
            logger.info(
                f"protocol - {self.objective_magnification}x objective, "
                f"{pixel_size_um:.4f} um/px at the sample"
            )
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

        # Pre-flight validation: check all planned positions against forbidden areas
        logger.info("protocol - performing pre-flight forbidden area validation")
        # Parse forbidden areas once and pass to positionIsForbidden to avoid repeated parsing
        forbidden_areas = cmds.ForbiddenAreaList.from_config()

        forbidden_positions = []
        for pos_info in self.iter_positions():
            is_forbidden, error_message = cmds.positionIsForbidden(
                pos_info.physical_position[0], pos_info.physical_position[1],
                forbidden_areas=forbidden_areas
            )
            if is_forbidden:
                forbidden_positions.append({
                    "well": pos_info.well.well_name,
                    "site": f"({pos_info.site.col}, {pos_info.site.row})",
                    "position": f"({pos_info.physical_position[0]:.1f}, {pos_info.physical_position[1]:.1f}) mm",
                    "error": error_message
                })

        if forbidden_positions:
            error_detail = "Acquisition contains positions in forbidden areas:\n"
            for pos in forbidden_positions:
                error_detail += f"  - Well {pos['well']}, Site {pos['site']} at {pos['position']}: {pos['error']}\n"
            logger.error(f"protocol - {error_detail}")
            cmds.error_internal(detail=error_detail)

        logger.info("protocol - pre-flight validation passed, no forbidden areas detected")

        # counters on acquisition progress
        start_time = time.time()
        start_time_iso_str = sc.datetime2str(dt.datetime.now(dt.UTC))
        last_image_information = None

        num_images_acquired = 0
        # Opened here, when the run starts, rather than when the protocol is built: a protocol
        # is built to be checked, and one that is refused must not leave a project directory
        # and a pool of writer threads behind. Closed by generate(), however the run ends.
        # Backend and writer threads come from machine config (storage.*).
        image_store = self.image_store = image_store_from_config()
        # the protocol itself travels with the images
        image_store.put_bytes(str(self.project_relpath / "config.json"), self.config_file.json().encode())
        strip_rows = ConfigRegistry.get(StorageConfig.STRIP_ROWS).intvalue
        codec_threads = ConfigRegistry.get(StorageConfig.CODEC_THREADS).intvalue

        # get current z coordinate as z reference
        last_position = yield cmds.GetCurrentPosition()
        assert isinstance(last_position, McPosition), f"{type(last_position)=}"
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

            # keep track of time for this timepoint
            timepoint_start_time=time.time()

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
                    site_x_mm, site_y_mm = self.get_physical_position_for_well_site(well, site)

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
                        miss: str | None = None
                        "why this site is imaged without having reached focus"
                        for autofocus_attempt_num in range(AUTOFOCUS_NUM_ATTEMPTS_MAX):
                            # each attempt is several frames and moves: a cancel should not
                            # have to wait for all of them
                            self.handle_q_in()
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
                                f"protocol - autofocus results: {res.outcome.value} {res.num_compensating_moves=} {res.uncompensated_offset_mm=:.3f}"
                            )

                            total_compensating_moves += res.num_compensating_moves

                            if res.outcome == cmds.AutofocusOutcome.IN_FOCUS:
                                autofocus_succeeded = True
                                break

                            # a failed measurement, or an estimate too far off to trust: the z
                            # it left is no better than the reference, so image at that
                            if res.outcome == cmds.AutofocusOutcome.FAILED:
                                miss = f"autofocus failed ({res.failure})"
                            elif (
                                math.fabs(res.uncompensated_offset_mm)
                                > UNCOMPENSATED_Z_FAILURE_THRESHOLD_MM
                            ):
                                miss = (
                                    f"autofocus {res.outcome.value} with "
                                    f"{res.uncompensated_offset_mm * 1e3:.1f} um left, beyond the "
                                    f"{UNCOMPENSATED_Z_FAILURE_THRESHOLD_MM * 1e3:.0f} um it may be off by"
                                )
                            if miss is not None:
                                mres = yield cmds.MoveTo(x_mm=None, y_mm=None, z_mm=reference_z_mm)
                                assert isinstance(mres, cmds.BasicSuccessResponse), f"{type(mres)=}"
                                miss += f"; imaged at the reference z {reference_z_mm:.4f} mm"
                                break

                            # close, but not in focus: another attempt, from where this one stopped
                        else:
                            assert res is not None
                            miss = (
                                f"autofocus did not reach focus in {AUTOFOCUS_NUM_ATTEMPTS_MAX} "
                                f"attempts ({res.outcome.value}, {res.uncompensated_offset_mm * 1e3:.1f} um "
                                f"left); imaged where the last attempt stopped"
                            )

                        logger.debug(f"protocol - autofocus {total_compensating_moves=}")

                        # reference for channel z offsets
                        last_position = yield cmds.GetCurrentPosition()
                        assert isinstance(last_position, McPosition), f"{type(last_position)=}"
                        reference_z_mm = last_position.z_pos_mm

                        if miss is not None:
                            assert res is not None
                            logger.warning(
                                f"protocol - {well.well_name} site ({site.col}, {site.row}): {miss}"
                            )
                            self._status_reporter.autofocus_missed(
                                cmds.AutofocusMiss(
                                    well_name=well.well_name,
                                    site_x=site.col,
                                    site_y=site.row,
                                    timepoint=timepoint,
                                    outcome=res.outcome,
                                    detail=miss,
                                    imaged_at_z_mm=reference_z_mm,
                                )
                            )

                        logger.debug("autofocus performed")
                    else:
                        res = yield cmds.MoveTo(x_mm=None, y_mm=None, z_mm=reference_z_mm)
                        assert isinstance(res, cmds.BasicSuccessResponse), f"{type(res)=}"

                        logger.debug("approached reference z")

                    # Use iterator for z-positions with proper imaging order and channel information
                    z_positions_list = list(self.iter_z_positions(reference_z_mm))

                    if len(z_positions_list) == 0:
                        continue  # No z-positions to acquire for this site

                    # Move to starting z position with backlash compensation
                    first_z_mm = z_positions_list[0].z_position_mm
                    res = yield cmds.MoveTo(
                        x_mm=None,
                        y_mm=None,
                        z_mm=first_z_mm - Z_STACK_COUNTER_BACKLASH_MM,
                    )
                    assert isinstance(res, cmds.BasicSuccessResponse), f"{type(res)=}"
                    res = yield cmds.MoveTo(x_mm=None, y_mm=None, z_mm=first_z_mm)
                    assert isinstance(res, cmds.BasicSuccessResponse), f"{type(res)=}"

                    logger.debug("protocol - moved to z order bottom")

                    for z_pos_info in z_positions_list:
                        plane_index = z_pos_info.plane_index
                        channel_z_mm = z_pos_info.z_position_mm
                        channel = z_pos_info.channel

                        logger.debug(
                            f"protocol - imaging plane {plane_index} at site with {channel_z_mm=:.4f} {channel.name=}"
                        )

                        site_x = site.col
                        site_y = site.row
                        site_z = plane_index

                        self.handle_q_in()

                        # move to channel offset
                        last_position = yield cmds.GetCurrentPosition()
                        assert isinstance(last_position, McPosition), f"{type(last_position)=}"
                        current_z_mm = last_position.z_pos_mm

                        distance_z_to_move_mm = channel_z_mm - current_z_mm
                        if math.fabs(distance_z_to_move_mm) > DISPLACEMENT_THRESHOLD_MM:
                            res = yield cmds.MoveTo(x_mm=None, y_mm=None, z_mm=channel_z_mm)
                            assert isinstance(res, cmds.BasicSuccessResponse), f"{type(res)=}"

                        last_position = yield cmds.GetCurrentPosition()
                        assert isinstance(last_position, McPosition), f"{type(last_position)=}"

                        logger.debug(
                            f"protocol - moved to channel z (should be {channel_z_mm:.4f}mm, is {last_position.z_pos_mm:.4f}mm)"
                        )

                        # snap image
                        res = yield cmds.ChannelSnapshot(
                            channel=channel,
                            ome_context=cmds.OmeMetadataContext(
                                project_name=self.config_file.project_name,
                                plate_name=self.config_file.plate_name,
                                cell_line=self.config_file.cell_line,
                                well_name=well.well_name,
                                site_x=site_x,
                                site_y=site_y,
                                site_z=site_z,
                                plane_index=plane_index,
                                autofocus_succeeded=autofocus_succeeded,
                            ),
                        )
                        if not isinstance(res, cmds.ImageAcquiredResponse):
                            cmds.error_internal(
                                detail=f"failed to snap image at site {site} in well {well} (invalid result type {type(res)})"
                            )

                        logger.debug("protocol - took image snapshot")

                        # store image
                        image_filename = image_filename_for(
                            well_name=well.well_name,
                            site=site,
                            site_index=site_index,
                            plane_index=plane_index,
                            channel=channel,
                            g_config=g_config,
                        )

                        if is_timeseries_acquisition:
                            #timepoints start at 1, but we use 0 for the first one in dir
                            assert timepoint>=1
                            image_relpath = self.project_relpath / f"t{timepoint - 1}" / image_filename
                        else:
                            image_relpath = self.project_relpath / image_filename
                        image_storage_path = image_store.describe(str(image_relpath))

                        image_store_entry = cmds.ImageStoreEntry(
                            pixel_format=ConfigRegistry.get(CameraConfig.MAIN_PIXEL_FORMAT).strvalue,
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
                                    position=last_position,
                                ),
                                storage_path=str(image_storage_path),
                            ),
                        )
                        image_store_entry._img = res._img.copy()
                        assert res.ome_xml is not None

                        self.latest_channel_images[channel.name] = image_store_entry

                        # An image already lost means the storage has failed, and it stays
                        # failed (a full disk, a gone share): stop now, not after every image
                        # still to come has gone the same way.
                        image_store.raise_if_failed()
                        # encode + store off the acquisition path (image-writer threads, then the sink)
                        image_store.put(
                            str(image_relpath),
                            functools.partial(
                                encode_image,
                                image_entry=image_store_entry,
                                img_compression_algorithm=self.img_compression_algorithm,
                                ome_xml=res.ome_xml,
                                strip_rows=strip_rows,
                                codec_threads=codec_threads,
                            ),
                        )

                        logger.debug(
                            f"protocol - scheduled image store to {image_store_entry.info.storage_path}"
                        )

                        num_images_acquired += 1

                        # status items
                        last_image_information = image_store_entry.info

                        # Time since overall acquisition started (for progress display)
                        time_since_start_s = time.time() - start_time

                        # Estimate remaining time from imaging progress
                        time_estimate = estimate_remaining_time(
                            num_images_acquired=num_images_acquired,
                            num_images_total=self.num_images_total,
                            time_since_start_s=time_since_start_s,
                            num_timepoints=self.num_timepoints,
                            current_timepoint=timepoint,
                            delta_t_s=(
                                self.config_file.grid.delta_t.h * 3600
                                + self.config_file.grid.delta_t.m * 60
                                + self.config_file.grid.delta_t.s
                            ),
                            completed_timepoint_imaging_times=self.completed_timepoint_imaging_times,
                        )
                        estimated_remaining_time_s = time_estimate.remaining_s

                        logger.debug(
                            f"protocol - {num_images_acquired}/{self.num_images_total} images acquired"
                        )

                        self._status_reporter.publish(
                            status=cmds.AcquisitionStatusStage.RUNNING,
                            current_num_images=num_images_acquired,
                            time_since_start_s=time_since_start_s,
                            start_time_iso=start_time_iso_str,
                            storage_usage_bytes=image_store.bytes_written,
                            estimated_remaining_time_s=estimated_remaining_time_s,
                            last_image=last_image_information,
                            message=f"Acquisition is {(100 * num_images_acquired / self.num_images_total):.2f}% complete",
                            current_timepoint=timepoint if self.num_timepoints > 1 else None,
                            total_timepoints=self.num_timepoints if self.num_timepoints > 1 else None,
                            estimated_timepoint_duration_s=time_estimate.timepoint_duration_s,
                        )

            # sleep for additional time, maybe
            end_time=time.time()
            if self.config_file.grid.num_t>timepoint:
                time_elapsed_s=end_time-timepoint_start_time
                # Record imaging time for this completed timepoint (for estimates of remaining ones)
                self.completed_timepoint_imaging_times.append(time_elapsed_s)

                delta_t_s=self.config_file.grid.delta_t.h*3600+self.config_file.grid.delta_t.m*60+self.config_file.grid.delta_t.s

                # sleep through remaining time in small intervals to process inputs (e.g. cancel acquisition)
                time_remaining=delta_t_s-time_elapsed_s

                logger.debug(f"protocol - time remaining before starting next time series acquisition: {time_remaining}s")

                if time_remaining<0:
                    logger.warning(f"protocol - took longer to acquire one time point than the specified delta time between acquisitions (took {time_elapsed_s}s but delta time is {delta_t_s}s )")

                # Calculate how long each timepoint takes (imaging + waiting)
                time_per_timepoint = max(delta_t_s, time_elapsed_s)

                timeslice_s=0.5
                while time_remaining>0:
                    self.handle_q_in()

                    # Update status with live countdown during wait
                    # Current wait + remaining timepoints
                    remaining_full_cycles = self.num_timepoints - timepoint - 1
                    estimated_remaining_during_wait = time_remaining + (remaining_full_cycles * time_per_timepoint) + time_elapsed_s
                    # Use average of completed timepoint times for the estimate
                    avg_timepoint_duration = sum(self.completed_timepoint_imaging_times) / len(self.completed_timepoint_imaging_times)
                    self._status_reporter.publish(
                        status=cmds.AcquisitionStatusStage.RUNNING,
                        current_num_images=num_images_acquired,
                        time_since_start_s=time.time() - start_time,
                        start_time_iso=start_time_iso_str,
                        storage_usage_bytes=image_store.bytes_written,
                        estimated_remaining_time_s=estimated_remaining_during_wait,
                        last_image=last_image_information,
                        message=f"Waiting for next timepoint ({time_remaining:.1f}s remaining)...",
                        current_timepoint=timepoint,
                        total_timepoints=self.num_timepoints,
                        estimated_timepoint_duration_s=avg_timepoint_duration,
                    )

                    await asyncio.sleep(min(time_remaining,timeslice_s))
                    time_remaining-=timeslice_s


        logger.debug("protocol - finished protocol steps")

        # wait for image storage to finish; a lost image is an error, not a silent gap
        report = image_store.close()
        logger.debug(f"protocol - finished image storage ({report.stored} stored, {report.spilled_to_disk} spilled)")
        if report.failed:
            shown = "\n  ".join(report.failed[:10])
            more = f"\n  ... and {len(report.failed) - 10} more" if len(report.failed) > 10 else ""
            cmds.error_internal(
                detail=f"acquisition finished but {len(report.failed)} of {num_images_acquired} images failed to store:\n  {shown}{more}"
            )

        total_time_s = time.time() - start_time
        total_storage_gb = report.bytes_written / (1024**3)
        logger.info(f"protocol - done (total time: {total_time_s:.1f}s, total storage: {total_storage_gb:.2f} GB)")

        # done -> yield None and return
        yield None
