"""
What an acquired frame is called, which is the only thing holding an acquisition together.

Every OME-TIFF seafront writes holds a single plane and declares SizeZ=SizeC=SizeT=1, with no
TiffData UUID pointing at its siblings. A reader opening one of them sees an unrelated
single-plane image. The z-stack, the channels and the sites exist only in the filenames, so
whatever reassembles a dataset downstream is parsing this string - and the config knobs that
change it are user-facing.

It had no tests at all, because it was thirty lines inline in the middle of a 480-line
generator. Extracting `image_filename_for` is what made this file possible.
"""

import pytest
import seaconfig as sc

from seafront.config import basics
from seafront.config.handles import ImageConfig
from seafront.server.protocol import image_filename_for

WELL = "F8"


def _site(row: int = 0, col: int = 0):
    return sc.AcquisitionWellSiteConfigurationSiteSelectionItem(row=row, col=col, selected=True)


def _channel(name: str = "Fluorescence 405 nm Ex", handle: str = "fluo405"):
    return sc.AcquisitionChannelConfig(
        name=name,
        handle=handle,
        illum_perc=100.0,
        exposure_time_ms=5.0,
        analog_gain=0.0,
        z_offset_um=0.0,
        num_z_planes=1,
        delta_z_um=1.0,
        enabled=True,
    )


def _config(**overrides) -> dict[str, basics.ConfigItem]:
    """
    the filename-shaping config items, as `generate` reads them

    An absent item means "take the default", which is a case worth being able to express
    because a config file written before a knob existed will not have it.
    """
    items = {}
    for handle, value in overrides.items():
        key = getattr(ImageConfig, handle).value
        if isinstance(value, bool):
            items[key] = basics.ConfigItem(
                name=key,
                handle=key,
                value_kind="option",
                value="yes" if value else "no",
                options=basics.ConfigItemOption.get_bool_options(),
            )
        else:
            items[key] = basics.ConfigItem(
                name=key, handle=key, value_kind="int", value=value
            )
    return items


def _name(**kwargs) -> str:
    return image_filename_for(
        well_name=kwargs.pop("well_name", WELL),
        site=kwargs.pop("site", _site()),
        site_index=kwargs.pop("site_index", 0),
        plane_index=kwargs.pop("plane_index", 0),
        channel=kwargs.pop("channel", _channel()),
        g_config=kwargs.pop("g_config", _config()),
    )


class TestTheShapeOfTheName:
    """
    <well>_s<site>_x<col>_y<row>_z<plane>_<channel>.ome.tiff

    Pinned field by field because the string is a public interface: anything that reads these
    directories parses it, and a field that moves silently breaks them all at once.
    """

    def test_the_default_name(self):
        assert _name() == "F08_s1_x0_y0_z0_Fluorescence_405_nm_Ex.ome.tiff"

    def test_it_is_an_ome_tiff(self):
        """the double extension is what makes readers treat it as OME rather than plain TIFF"""
        assert _name().endswith(".ome.tiff")

    def test_the_well_comes_first(self):
        assert _name(well_name="B02").startswith("B02_")

    def test_the_site_index_is_marked_s(self):
        assert "_s3_" in _name(site_index=2)

    def test_the_grid_column_is_x_and_the_row_is_y(self):
        """
        which way round these go decides whether a montage is transposed

        col -> x and row -> y. Swapping them produces a plausible-looking name and a
        reconstruction that is a reflection of the sample.
        """
        assert "_x4_y7_" in _name(site=_site(row=7, col=4))

    def test_the_z_plane_is_marked_z(self):
        assert "_z6_" in _name(plane_index=6)

    def test_the_channel_comes_last(self):
        assert _name().endswith("_Fluorescence_405_nm_Ex.ome.tiff")

    def test_every_field_is_present_at_once(self):
        assert _name(
            well_name="C11", site=_site(row=2, col=3), site_index=4, plane_index=5
        ) == "C11_s5_x3_y2_z5_Fluorescence_405_nm_Ex.ome.tiff"


class TestTheChannelPart:
    def test_spaces_become_underscores(self):
        """a filename with spaces in it is a shell quoting bug waiting downstream"""
        assert "Fluorescence_405_nm_Ex" in _name()

    def test_the_handle_can_be_used_instead(self):
        assert _name(g_config=_config(FILENAME_USE_CHANNEL_NAME=False)).endswith(
            "_fluo405.ome.tiff"
        )

    def test_the_name_is_the_default_when_the_setting_is_absent(self):
        """a config file written before the knob existed keeps the old behaviour"""
        assert "Fluorescence_405_nm_Ex" in _name(g_config={})

    def test_two_channels_produce_two_names(self):
        """the channel is what distinguishes otherwise identical frames at one site"""
        first = _name(channel=_channel("BF LED full", "bfledfull"))
        second = _name(channel=_channel("Fluorescence 488 nm Ex", "fluo488"))
        assert first != second


class TestWhereTheIndicesStart:
    """
    sites count from 1, x/y/z from 0 - and all of them are configurable

    The defaults are not arbitrary: they are what existing datasets on disk were written
    with, so changing them silently would split a project's files into two naming schemes.
    """

    def test_sites_count_from_one_by_default(self):
        assert "_s1_" in _name(site_index=0)

    def test_xy_and_z_count_from_zero_by_default(self):
        assert "_x0_y0_z0_" in _name(site=_site(0, 0), plane_index=0)

    def test_the_site_start_can_be_moved(self):
        assert "_s0_" in _name(site_index=0, g_config=_config(FILENAME_SITE_INDEX_START=0))

    def test_the_xy_start_can_be_moved(self):
        name = _name(site=_site(row=1, col=2), g_config=_config(FILENAME_XY_INDEX_START=1))
        assert "_x3_y2_" in name

    def test_the_z_start_can_be_moved(self):
        assert "_z1_" in _name(plane_index=0, g_config=_config(FILENAME_Z_INDEX_START=1))

    def test_the_starts_are_offsets_not_replacements(self):
        """the index still counts up from wherever it was told to start"""
        config = _config(FILENAME_SITE_INDEX_START=10)
        assert "_s12_" in _name(site_index=2, g_config=config)

    def test_absent_settings_give_the_documented_defaults(self):
        assert _name(g_config={}) == "F08_s1_x0_y0_z0_Fluorescence_405_nm_Ex.ome.tiff"


class TestTheWellName:
    """
    zero-padded by default, so a directory listing sorts the way the plate is laid out

    F8 sorts after F10 in every tool that sorts strings; F08 does not. A 384-well plate has
    columns into the twenties, so this is the difference between a plate reading in order and
    one reading 1, 10, 11, ..., 2, 20.
    """

    @pytest.mark.parametrize(
        ("raw", "expected"), [("F8", "F08"), ("A1", "A01"), ("A10", "A10"), ("P24", "P24")]
    )
    def test_the_column_is_padded_to_two_digits(self, raw, expected):
        assert _name(well_name=raw).startswith(f"{expected}_")

    def test_padding_can_be_turned_off(self):
        assert _name(well_name="F8", g_config=_config(FILENAME_ZERO_PAD_COLUMN=False)).startswith(
            "F8_"
        )

    def test_padding_is_on_when_the_setting_is_absent(self):
        assert _name(well_name="F8", g_config={}).startswith("F08_")

    def test_padded_names_sort_into_plate_order(self):
        """the property the padding exists for, rather than the mechanism"""
        names = sorted(_name(well_name=f"A{col}") for col in (1, 2, 10, 11, 20))
        columns = [n.split("_")[0] for n in names]
        assert columns == ["A01", "A02", "A10", "A11", "A20"]

    def test_unpadded_names_do_not(self):
        """
        which is what the default is protecting against

        The exact wrong order is not the point and is not worth pinning - it falls out of
        where '_' sits against the digits in ASCII. That it is not plate order is the point.
        """
        config = _config(FILENAME_ZERO_PAD_COLUMN=False)
        columns = [1, 2, 10]
        names = sorted(_name(well_name=f"A{col}", g_config=config) for col in columns)
        assert [n.split("_")[0] for n in names] != [f"A{col}" for col in columns]


class TestNamesAreUniqueWithinARun:
    """
    the property everything above exists to serve

    Two frames of one acquisition that share a name overwrite each other, and the loss is
    silent - the run reports the number of images it meant to write.
    """

    def test_planes_of_one_stack_differ(self):
        names = {_name(plane_index=i) for i in range(20)}
        assert len(names) == 20

    def test_sites_of_one_well_differ(self):
        names = {
            _name(site=_site(row=r, col=c), site_index=r * 3 + c)
            for r in range(3)
            for c in range(3)
        }
        assert len(names) == 9

    def test_a_whole_well_of_frames_is_distinct(self):
        names = {
            _name(
                well_name=well,
                site=_site(row=r, col=c),
                site_index=r * 2 + c,
                plane_index=z,
                channel=channel,
            )
            for well in ("A01", "B02")
            for r in range(2)
            for c in range(2)
            for z in range(3)
            for channel in (_channel("BF", "bf"), _channel("Fluorescence 488 nm Ex", "fluo488"))
        }
        assert len(names) == 2 * 2 * 2 * 3 * 2

    def test_a_site_index_alone_does_not_carry_the_grid_position(self):
        """
        s, x and y are not redundant

        The site index counts sites in visit order, which the greedy route reorders; x and y
        are the grid cell. Two frames with the same x,y from different wells are different
        frames, and the well name is what separates them.
        """
        first = _name(well_name="A01", site=_site(row=1, col=1), site_index=0)
        second = _name(well_name="B02", site=_site(row=1, col=1), site_index=0)
        assert first != second
