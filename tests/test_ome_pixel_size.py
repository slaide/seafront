"""
The scale bar in the metadata, which is the objective's to set.

`PhysicalSizeX`/`PhysicalSizeY` in an OME-TIFF is what anyone measuring a distance off these
images divides by. It used to be the literal 0.3 um for every acquisition - the value that
holds at 20x - while the same file recorded whichever objective was actually configured. A 4x
run therefore claimed a 4x objective and 20x pixels at once, and every micrometre anybody
derived from it was out by five.

Nothing covered `build_ome_metadata` at all, so there was nothing to notice.
"""

import seaconfig as sc
from ome_types import from_xml

from seafront.config import basics
from seafront.server import commands as cmds
from seafront.server.protocol import (
    MAIN_CAMERA_PIXEL_PITCH_UM,
    build_ome_instrument,
    build_ome_metadata,
    pixel_size_um_at,
    significant_bits_for,
)

import pytest


class TestPixelSizeFollowsTheObjective:
    """
    the derivation itself

    Pixel size at the sample is the sensor's pitch divided by the magnification. The pitch is
    anchored on the one measurement there is - 0.3 um/px at 20x - so 20x must still come out
    at exactly the historical value, and the others must scale from it.
    """

    def test_twenty_x_is_the_value_the_constant_was_measured_at(self):
        """0.9 mm across 3000 px, which is where the old hardcoded 0.3 came from"""
        assert pixel_size_um_at(20) == pytest.approx(900 / 3000)

    @pytest.mark.parametrize(
        ("magnification", "expected_um"),
        [(4, 1.5), (10, 0.6), (20, 0.3), (40, 0.15)],
    )
    def test_the_size_scales_inversely_with_magnification(self, magnification, expected_um):
        assert pixel_size_um_at(magnification) == pytest.approx(expected_um)

    def test_a_lower_magnification_sees_more_sample_per_pixel(self):
        """the direction of the relationship, which an inverted formula would flip"""
        assert pixel_size_um_at(4) > pixel_size_um_at(10) > pixel_size_um_at(20)

    def test_the_pitch_is_what_ties_them_together(self):
        assert pixel_size_um_at(20) * 20 == pytest.approx(MAIN_CAMERA_PIXEL_PITCH_UM)

    @pytest.mark.parametrize("magnification", [None, 0, -1])
    def test_an_unusable_magnification_yields_no_size_rather_than_a_wrong_one(
        self, magnification
    ):
        """
        the whole point of the change

        A missing scale is honest and visible. A confidently wrong one is the bug this
        replaces, and it is undetectable in the finished data.
        """
        assert pixel_size_um_at(magnification) is None


def _channel() -> sc.AcquisitionChannelConfig:
    return sc.AcquisitionChannelConfig(
        name="Fluorescence 405 nm Ex",
        handle="fluo405",
        illum_perc=100.0,
        exposure_time_ms=10.0,
        analog_gain=0.0,
        z_offset_um=0.0,
        num_z_planes=1,
        delta_z_um=1.0,
        enabled=True,
    )


def _ome_xml(magnification: int | None, significant_bits: int | None = None,
             pixel_format: str = "mono12") -> str:
    """a real OME-TIFF description string, built the way an acquisition builds one"""
    hardware_channels = [
        basics.ChannelConfig(name="Fluorescence 405 nm Ex", handle="fluo405", source_slot=11)
    ]
    template = build_ome_instrument(
        microscope_name="test-scope",
        camera_vendor="vendor",
        camera_model="model",
        camera_sn="sn",
        hardware_channels=hardware_channels,
        objective_magnification=magnification,
    )
    entry = cmds.ImageStoreEntry(
        pixel_format=pixel_format,
        info=cmds.ImageStoreInfo(
            channel=_channel(),
            width_px=2500,
            height_px=2500,
            timestamp=0.0,
            position=cmds.SitePosition(
                well_name="B02",
                site_x=0, site_y=0, site_z=0,
                x_offset_mm=0.0, y_offset_mm=0.0, z_offset_mm=0.0,
                position=cmds.Position(x_pos_mm=10.0, y_pos_mm=20.0, z_pos_mm=1.0),
            ),
        ),
    )
    return build_ome_metadata(
        ome_template=template,
        image_entry=entry,
        channel_config=_channel(),
        position=entry.info.position,
        plane_index=0,
        pixel_size_um=pixel_size_um_at(magnification),
        significant_bits=significant_bits,
        project_name="project",
        plate_name="plate",
        cell_line="line",
        hardware_channels=hardware_channels,
    )


class TestWhatReachesTheFile:
    """
    the derivation is only worth anything if it survives into the OME-XML

    Parsed back with ome_types rather than string-matched, so these check the document a
    downstream reader actually sees.
    """

    @pytest.mark.parametrize(
        ("magnification", "expected_um"), [(4, 1.5), (10, 0.6), (20, 0.3)]
    )
    def test_the_recorded_pixel_size_matches_the_objective(self, magnification, expected_um):
        pixels = from_xml(_ome_xml(magnification)).images[0].pixels
        assert pixels.physical_size_x == pytest.approx(expected_um)
        assert pixels.physical_size_y == pytest.approx(expected_um)

    @pytest.mark.parametrize("magnification", [4, 10, 20])
    def test_the_objective_and_the_pixel_size_agree_in_the_same_file(self, magnification):
        """
        the contradiction that was there before

        Both numbers were already written; only one of them moved with the configuration, so
        any objective other than 20x produced a file that disagreed with itself.
        """
        ome = from_xml(_ome_xml(magnification))
        objective = ome.instruments[0].objectives[0]
        assert objective.nominal_magnification == pytest.approx(magnification)
        assert ome.images[0].pixels.physical_size_x == pytest.approx(
            MAIN_CAMERA_PIXEL_PITCH_UM / objective.nominal_magnification
        )

    def test_an_unknown_objective_leaves_the_size_out(self):
        pixels = from_xml(_ome_xml(None)).images[0].pixels
        assert pixels.physical_size_x is None
        assert pixels.physical_size_y is None


class TestSignificantBitsDescribesTheSamples:
    """
    the other half of an honest file: how much of each 16-bit word is real

    The cameras are 12-bit (or 8/10/14) and there is no container for that, so the samples
    ride in a 16-bit type. OME has one attribute for saying so, and nothing was setting it: a
    reader saw uint16 and had no way to know the values only ever reach 4095.

    OME gives no way to say *where* in the word the bits sit, so the only interpretation
    available to a reader is low-aligned, 0..2**N-1. That is what the camera produces, which
    is why `image.file.pad_low` now defaults off.
    """

    @pytest.mark.parametrize("camera_bits", [8, 10, 12, 14, 16])
    def test_unpadded_samples_report_the_cameras_own_depth(self, camera_bits):
        container = 8 if camera_bits <= 8 else 16
        assert significant_bits_for(camera_bits, container, pad_low=False) == camera_bits

    def test_padding_shifts_the_samples_so_the_container_width_is_the_honest_answer(self):
        """
        12 bits shifted up to fill 16 really do span the container

        Saying 12 here would tell a reader to stretch 0..4095 over a frame whose values reach
        65520, and the whole image would clip to white - worse than the darkness the shift was
        added to avoid.
        """
        assert significant_bits_for(12, 16, pad_low=True) == 16

    def test_padding_changes_nothing_when_the_camera_already_fills_the_container(self):
        assert significant_bits_for(16, 16, pad_low=True) == 16
        assert significant_bits_for(8, 8, pad_low=True) == 8

    @pytest.mark.parametrize(("pixel_format", "expected"), [("mono8", 8), ("mono12", 12), ("mono16", 16)])
    def test_the_depth_reaches_the_file(self, pixel_format, expected):
        pixels = from_xml(
            _ome_xml(20, significant_bits=expected, pixel_format=pixel_format)
        ).images[0].pixels
        assert pixels.significant_bits == expected

    def test_twelve_bit_data_is_not_described_as_sixteen(self):
        """the specific lie the file used to tell by omission"""
        pixels = from_xml(_ome_xml(20, significant_bits=12)).images[0].pixels
        assert pixels.type.value == "uint16"
        assert pixels.significant_bits == 12


class TestEveryDepthTheCameraCouldOffer:
    """
    the depth is the camera's to choose, and the chain has to carry whatever it picks

    The drivers enumerate 8/10/12/14/16 today - the GenICam mono set, and ToupCam's RAW
    flags - but the depth is a camera capability, not a fact about this codebase, and the
    config options are replaced at runtime by whatever `get_supported_pixel_formats` reports.
    Four separate places used to hold their own list of those five; the two that mattered
    both had a silent fallback for anything absent, so a camera the drivers grew support for
    would have written an image labelled 8-bit with 16-bit samples in it.
    """

    @pytest.mark.parametrize("bits", list(range(1, 17)))
    def test_the_depth_is_read_out_of_the_format_name(self, bits):
        from seafront.hardware.camera import bits_for_pixel_format

        assert bits_for_pixel_format(f"mono{bits}") == bits

    @pytest.mark.parametrize("pixel_format", ["", "mono", "rgb8", "mono8packed", "bayer12", "MONO", "mono0", "mono17"])
    def test_a_format_that_is_not_a_mono_depth_is_refused(self, pixel_format):
        """loudly, rather than falling back to a depth nobody chose"""
        from seafront.hardware.camera import bits_for_pixel_format

        with pytest.raises(ValueError):
            bits_for_pixel_format(pixel_format)

    def test_the_name_is_case_insensitive(self):
        from seafront.hardware.camera import bits_for_pixel_format

        assert bits_for_pixel_format("Mono12") == 12

    @pytest.mark.parametrize("bits", list(range(8, 17)))
    def test_the_container_and_the_declared_depth_agree_at_every_step(self, bits):
        """
        the pairing a reader needs: a type wide enough to hold the samples, and a count of
        how many of its bits are real
        """
        pixels = from_xml(
            _ome_xml(20, significant_bits=bits, pixel_format=f"mono{bits}")
        ).images[0].pixels
        assert pixels.significant_bits == bits
        assert pixels.type.value == ("uint8" if bits <= 8 else "uint16")

    @pytest.mark.parametrize("bits", [9, 11, 13, 15])
    def test_an_odd_depth_is_carried_rather_than_mislabelled(self, bits):
        """
        the case the tables could not express

        No driver reports these today. The point is that adding one would be a driver change,
        not a hunt through four lists in three modules.
        """
        pixels = from_xml(
            _ome_xml(20, significant_bits=bits, pixel_format=f"mono{bits}")
        ).images[0].pixels
        assert pixels.significant_bits == bits
        assert pixels.type.value == "uint16"
