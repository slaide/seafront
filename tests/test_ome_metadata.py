"""
The per-image half of the OME-XML: where on the plate, how long, which light.

`tests/test_ome_pixel_size.py` covers the scale of a pixel and how many of its bits are real.
This covers everything else `build_ome_metadata` writes about one acquired frame, none of
which had a test: the stage coordinates and their units, the exposure, the channel's
wavelength and illumination, and the experiment context in the description.

The units matter as much as the numbers. OME defaults an unset position unit to "reference
frame", which is a statement that the file is using some coordinate system it does not
define - so a stage coordinate written without its unit is not a length at all, and anything
reconstructing a plate montage from these positions is working with bare numbers.
"""

import datetime as dt

import pytest
import seaconfig as sc
from ome_types import from_xml

from seafront.config import basics
from seafront.server import commands as cmds
from seafront.server.protocol import (
    _format_well_name_with_padding,
    build_ome_instrument,
    build_ome_metadata,
    extract_wavelength_nm,
)

HARDWARE_CHANNELS = [
    basics.ChannelConfig(name="Fluorescence 405 nm Ex", handle="fluo405", source_slot=11),
    basics.ChannelConfig(name="Fluorescence 488 nm Ex", handle="fluo488", source_slot=12),
    basics.ChannelConfig(name="BF LED matrix full", handle="bfledfull", source_slot=0),
]

# a site somewhere in the middle of a plate, in mm, with a non-zero z so a dropped
# coordinate cannot pass by being zero anyway
SITE_X_MM, SITE_Y_MM, SITE_Z_MM = 66.13, 44.99, 1.234


def _channel(
    handle: str = "fluo405",
    name: str = "Fluorescence 405 nm Ex",
    exposure_ms: float = 12.5,
    illum_perc: float = 40.0,
    gain: float = 3.5,
    num_z_planes: int = 3,
) -> sc.AcquisitionChannelConfig:
    return sc.AcquisitionChannelConfig(
        name=name,
        handle=handle,
        illum_perc=illum_perc,
        exposure_time_ms=exposure_ms,
        analog_gain=gain,
        z_offset_um=0.0,
        num_z_planes=num_z_planes,
        delta_z_um=1.0,
        enabled=True,
    )


def _ome(
    channel: sc.AcquisitionChannelConfig | None = None,
    plane_index: int = 2,
    well_name: str = "B02",
    timestamp: float = 1_700_000_000.0,
    autofocus_succeeded: bool = False,
    project_name: str = "my-project",
    plate_name: str = "plate-7",
    cell_line: str = "U2OS",
):
    """the Image element of a real acquisition's OME-XML, parsed back"""
    channel = channel or _channel()
    template = build_ome_instrument(
        microscope_name="test-scope",
        camera_vendor="vendor",
        camera_model="cam-model",
        camera_sn="SN123",
        hardware_channels=HARDWARE_CHANNELS,
        objective_magnification=20,
    )
    position = cmds.SitePosition(
        well_name=well_name,
        site_x=1, site_y=2, site_z=3,
        x_offset_mm=0.0, y_offset_mm=0.0, z_offset_mm=0.0,
        position=cmds.Position(x_pos_mm=SITE_X_MM, y_pos_mm=SITE_Y_MM, z_pos_mm=SITE_Z_MM),
    )
    entry = cmds.ImageStoreEntry(
        pixel_format="mono12",
        info=cmds.ImageStoreInfo(
            channel=channel,
            width_px=2500,
            height_px=2000,
            timestamp=timestamp,
            position=position,
        ),
    )
    xml = build_ome_metadata(
        ome_template=template,
        image_entry=entry,
        channel_config=channel,
        position=position,
        plane_index=plane_index,
        pixel_size_um=0.3,
        significant_bits=12,
        project_name=project_name,
        plate_name=plate_name,
        cell_line=cell_line,
        hardware_channels=HARDWARE_CHANNELS,
        autofocus_succeeded=autofocus_succeeded,
    )
    return from_xml(xml).images[0]


class TestTheStageCoordinatesAreLengths:
    """
    a number without a unit is not a position

    The driver works in millimetres and these fields are written straight from it. OME will
    not infer that: an unset unit means "reference frame", i.e. an undeclared coordinate
    system, and a reader has no way back to millimetres from there.
    """

    def test_the_plane_carries_the_site_position(self):
        plane = _ome().pixels.planes[0]
        assert plane.position_x == pytest.approx(SITE_X_MM)
        assert plane.position_y == pytest.approx(SITE_Y_MM)
        assert plane.position_z == pytest.approx(SITE_Z_MM)

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_the_plane_position_says_it_is_in_millimetres(self, axis):
        plane = _ome().pixels.planes[0]
        assert getattr(plane, f"position_{axis}_unit").value == "mm"

    def test_the_stage_label_carries_the_same_position(self):
        """the stage label is what a viewer shows; disagreeing with the plane would be worse"""
        image = _ome()
        label, plane = image.stage_label, image.pixels.planes[0]
        assert label.x == pytest.approx(plane.position_x)
        assert label.y == pytest.approx(plane.position_y)
        assert label.z == pytest.approx(plane.position_z)

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_the_stage_label_says_it_is_in_millimetres(self, axis):
        assert getattr(_ome().stage_label, f"{axis}_unit").value == "mm"

    def test_the_stage_label_is_named_for_the_well(self):
        assert _ome(well_name="H11").stage_label.name == "H11"


class TestTheExposure:
    def test_milliseconds_become_seconds(self):
        plane = _ome(_channel(exposure_ms=12.5)).pixels.planes[0]
        assert plane.exposure_time == pytest.approx(0.0125)

    def test_and_the_unit_says_so(self):
        """
        this one is right by luck rather than by statement

        OME defaults ExposureTimeUnit to seconds and the code divides by 1000, so the pair
        agrees. Pinned because the division and the default have to keep agreeing.
        """
        assert _ome().pixels.planes[0].exposure_time_unit.value == "s"

    @pytest.mark.parametrize("ms", [0.1, 1.0, 100.0, 5000.0])
    def test_the_conversion_holds_across_the_range(self, ms):
        plane = _ome(_channel(exposure_ms=ms)).pixels.planes[0]
        assert plane.exposure_time == pytest.approx(ms / 1000.0)


class TestTheChannel:
    def test_the_channel_is_named_as_the_protocol_named_it(self):
        image = _ome(_channel(handle="fluo488", name="Fluorescence 488 nm Ex"))
        assert image.pixels.channels[0].name == "Fluorescence 488 nm Ex"

    @pytest.mark.parametrize(
        ("name", "expected_nm"),
        [
            ("Fluorescence 405 nm Ex", 405),
            ("Fluorescence 488 nm Ex", 488),
            ("Fluorescence 561nm Ex", 561),
            ("BF LED matrix full", None),
            ("", None),
        ],
    )
    def test_the_wavelength_is_read_out_of_the_channel_name(self, name, expected_nm):
        assert extract_wavelength_nm(name) == expected_nm

    def test_the_wavelength_reaches_the_light_source_settings(self):
        image = _ome(_channel(handle="fluo488", name="Fluorescence 488 nm Ex"))
        settings = image.pixels.channels[0].light_source_settings
        assert settings is not None
        assert settings.wavelength == pytest.approx(488)

    def test_a_brightfield_channel_has_no_wavelength_to_give(self):
        """not every light source is a laser line; absent beats invented"""
        image = _ome(_channel(handle="bfledfull", name="BF LED matrix full"))
        settings = image.pixels.channels[0].light_source_settings
        assert settings is not None
        assert settings.wavelength is None

    @pytest.mark.parametrize(("illum_perc", "expected"), [(100.0, 1.0), (40.0, 0.4), (2.5, 0.025)])
    def test_illumination_percent_becomes_a_fraction(self, illum_perc, expected):
        """OME attenuation is 0-1; the protocol speaks 0-100"""
        image = _ome(_channel(illum_perc=illum_perc))
        assert image.pixels.channels[0].light_source_settings.attenuation == pytest.approx(expected)

    def test_a_channel_the_hardware_does_not_have_gets_no_light_source(self):
        image = _ome(_channel(handle="does-not-exist", name="Nothing 999 nm Ex"))
        assert image.pixels.channels[0].light_source_settings is None


class TestTheImageIdentity:
    def test_the_acquisition_date_is_the_moment_the_frame_was_taken(self):
        image = _ome(timestamp=1_700_000_000.0)
        assert image.acquisition_date == dt.datetime.fromtimestamp(1_700_000_000.0, dt.UTC)

    def test_the_acquisition_date_carries_its_timezone(self):
        """a bare local timestamp is ambiguous a few months later, and across a fleet"""
        assert _ome().acquisition_date.tzinfo is not None

    def test_the_name_locates_the_frame(self):
        """well, channel and z-plane, which is all that distinguishes one file from the next"""
        image = _ome(_channel(handle="fluo488"), plane_index=2, well_name="C07")
        assert "C07" in image.name
        assert "fluo488" in image.name
        assert "002" in image.name

    @pytest.mark.parametrize(
        ("field", "value"),
        [("Project", "my-project"), ("Plate", "plate-7"), ("Cell Line", "U2OS"), ("Well", "B02")],
    )
    def test_the_description_carries_the_experiment_context(self, field, value):
        assert f"{field}: {value}" in _ome().description

    def test_the_description_records_the_analog_gain(self):
        assert "Analog Gain: 3.50 dB" in _ome(_channel(gain=3.5)).description

    def test_the_description_says_where_in_the_stack_this_plane_sits(self):
        assert "Z-Plane: 2/3" in _ome(_channel(num_z_planes=3), plane_index=2).description

    @pytest.mark.parametrize(("succeeded", "expected"), [(True, "Success"), (False, "N/A")])
    def test_the_description_says_whether_autofocus_worked(self, succeeded, expected):
        """
        the only place a site imaged out of focus is recorded

        The acquisition status does not report it and the log mentions it at debug, so this
        string is what tells anyone afterwards that a frame was taken on a fallback z.
        """
        assert f"Autofocus: {expected}" in _ome(autofocus_succeeded=succeeded).description


class TestTheSinglePlaneLayout:
    """
    every file holds one plane, and its indices are local to that file

    The z and channel a frame belongs to live in its name and its PositionZ, not in these
    indices - so all of them are 0 and SizeZ/C/T are 1, in every file.
    """

    def test_the_plane_indices_are_local(self):
        plane = _ome(plane_index=7).pixels.planes[0]
        assert (plane.the_z, plane.the_c, plane.the_t) == (0, 0, 0)

    def test_the_dimensions_describe_one_plane(self):
        pixels = _ome().pixels
        assert (pixels.size_z, pixels.size_c, pixels.size_t) == (1, 1, 1)

    def test_the_tiffdata_points_at_the_only_ifd(self):
        block = _ome().pixels.tiff_data_blocks[0]
        assert block.ifd == 0
        assert block.plane_count == 1
        assert (block.first_z, block.first_c, block.first_t) == (0, 0, 0)

    def test_the_pixel_dimensions_are_the_frame_that_was_stored(self):
        pixels = _ome().pixels
        assert (pixels.size_x, pixels.size_y) == (2500, 2000)


class TestWellNamesAreZeroPadded:
    """
    so that a directory of them sorts the way a plate is laid out

    F8 next to F10 sorts wrong in every tool that sorts names; F08 does not.
    """

    @pytest.mark.parametrize(
        ("raw", "padded"), [("F8", "F08"), ("A1", "A01"), ("A10", "A10"), ("H12", "H12"), ("B2", "B02")]
    )
    def test_the_column_is_two_digits(self, raw, padded):
        assert _format_well_name_with_padding(raw) == padded
