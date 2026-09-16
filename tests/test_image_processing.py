"""
What the driver does to a frame between the camera and the file.

`_process_image` is the whole of it on real hardware: centre crop to the configured size,
optional flips, and the bit shift that `image.file.pad_low` asks for. Every image seafront
has ever written went through it and nothing tested it - the mock takes a different path, so
the tests that do run an acquisition never touch this function at all.

The crop is the part with teeth. It decides which part of the sensor becomes the image, so
getting the offset wrong moves every recorded stage position off the pixels it describes, and
nothing downstream can tell.
"""

import numpy as np
import pytest

from seafront.config.handles import CameraConfig, ImageConfig
from seafront.config.registry import ConfigRegistry
from seafront.hardware.squid import _process_image


@pytest.fixture
def cam_config(mock_microscope_env):
    """
    the config items _process_image reads, at known values

    Builds on mock_microscope_env because the registry has to be populated before anything
    can be overridden; the fixture resets it afterwards.
    """

    def configure(
        *,
        width: int = 4,
        height: int = 4,
        flip_h: bool = False,
        flip_v: bool = False,
        pad_low: bool = False,
    ) -> None:
        # the flags are option items over "yes"/"no", not booleans
        yn = {True: "yes", False: "no"}
        ConfigRegistry.set_value(CameraConfig.MAIN_IMAGE_WIDTH_PX, width)
        ConfigRegistry.set_value(CameraConfig.MAIN_IMAGE_HEIGHT_PX, height)
        ConfigRegistry.set_value(CameraConfig.MAIN_IMAGE_FLIP_HORIZONTAL, yn[flip_h])
        ConfigRegistry.set_value(CameraConfig.MAIN_IMAGE_FLIP_VERTICAL, yn[flip_v])
        ConfigRegistry.set_value(ImageConfig.FILE_PAD_LOW, yn[pad_low])

    return configure


def _ramp(height: int, width: int, dtype=np.uint16) -> np.ndarray:
    """each pixel numbered by position, so a crop or flip is visible in the values"""
    return np.arange(height * width, dtype=dtype).reshape(height, width)


class TestTheCropIsCentred:
    """
    the configured size is taken from the middle of the sensor, not a corner

    An off-centre crop shifts the whole field relative to the stage coordinate recorded with
    it, consistently, in a way no downstream check would notice.
    """

    def test_the_result_has_the_configured_size(self, cam_config):
        cam_config(width=4, height=6)
        out, _ = _process_image(_ramp(10, 10), pixel_format="mono16")
        assert out.shape == (6, 4)

    def test_an_even_inset_takes_the_middle(self, cam_config):
        cam_config(width=2, height=2)
        # 4x4 ramp; the centre 2x2 is rows 1-2, cols 1-2
        out, _ = _process_image(_ramp(4, 4), pixel_format="mono16")
        assert out.tolist() == [[5, 6], [9, 10]]

    def test_an_odd_inset_biases_the_same_way_every_time(self, cam_config):
        """
        (5-2)//2 == 1, so one extra row goes above and one extra column to the left

        The direction matters less than it being fixed: a bias that changed with the size
        would move the field between configurations.
        """
        cam_config(width=2, height=2)
        out, _ = _process_image(_ramp(5, 5), pixel_format="mono16")
        assert out.tolist() == [[6, 7], [11, 12]]

    def test_a_crop_the_size_of_the_sensor_changes_nothing(self, cam_config):
        cam_config(width=4, height=4)
        source = _ramp(4, 4)
        out, _ = _process_image(source, pixel_format="mono16")
        assert out.tolist() == source.tolist()

    @pytest.mark.parametrize(("width", "height"), [(11, 4), (4, 11), (11, 11)])
    def test_asking_for_more_than_the_sensor_has_is_refused(self, cam_config, width, height):
        """loudly, rather than returning a smaller frame than the config promised"""
        cam_config(width=width, height=height)
        with pytest.raises(AssertionError):
            _process_image(_ramp(10, 10), pixel_format="mono16")


class TestTheFlips:
    """
    which way round the sample is

    A flipped image with unflipped stage coordinates is a mirror image of the plate, and it
    reads as real until somebody tries to go back to a feature they found.
    """

    def test_horizontal_reverses_the_columns(self, cam_config):
        cam_config(width=4, height=2, flip_h=True)
        out, _ = _process_image(_ramp(2, 4), pixel_format="mono16")
        assert out.tolist() == [[3, 2, 1, 0], [7, 6, 5, 4]]

    def test_vertical_reverses_the_rows(self, cam_config):
        cam_config(width=4, height=2, flip_v=True)
        out, _ = _process_image(_ramp(2, 4), pixel_format="mono16")
        assert out.tolist() == [[4, 5, 6, 7], [0, 1, 2, 3]]

    def test_both_together_are_a_half_turn(self, cam_config):
        cam_config(width=4, height=2, flip_h=True, flip_v=True)
        out, _ = _process_image(_ramp(2, 4), pixel_format="mono16")
        assert out.tolist() == [[7, 6, 5, 4], [3, 2, 1, 0]]

    def test_neither_leaves_it_alone(self, cam_config):
        cam_config(width=4, height=2)
        out, _ = _process_image(_ramp(2, 4), pixel_format="mono16")
        assert out.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]


class TestTheBitDepthItReports:
    @pytest.mark.parametrize("bits", [8, 10, 12, 14, 16])
    def test_the_depth_comes_from_the_pixel_format(self, cam_config, bits):
        cam_config(width=2, height=2)
        _, cambits = _process_image(_ramp(4, 4), pixel_format=f"mono{bits}")
        assert cambits == bits

    def test_a_format_that_is_not_a_depth_is_refused(self, cam_config):
        cam_config(width=2, height=2)
        with pytest.raises(ValueError):
            _process_image(_ramp(4, 4), pixel_format="bayer12")


class TestPadLow:
    """
    the shift that fills the container, and what it costs

    Off by default: the samples then mean what the camera measured, and SignificantBits in
    the OME-XML describes them. On, every intensity is multiplied by 2**(container - camera),
    which is what makes the images look right in a viewer that ignores the metadata.
    """

    def test_off_leaves_the_samples_alone(self, cam_config):
        cam_config(width=2, height=2, pad_low=False)
        source = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]], dtype=np.uint16)
        out, _ = _process_image(source, pixel_format="mono12")
        assert out.tolist() == [[5, 6], [9, 10]]

    def test_on_shifts_twelve_bit_samples_up_by_four(self, cam_config):
        cam_config(width=2, height=2, pad_low=True)
        source = np.full((4, 4), 100, dtype=np.uint16)
        out, _ = _process_image(source, pixel_format="mono12")
        assert (out == 100 * 16).all()

    @pytest.mark.parametrize(("bits", "factor"), [(8, 256), (10, 64), (12, 16), (14, 4)])
    def test_the_factor_is_set_by_the_gap_to_the_container(self, cam_config, bits, factor):
        cam_config(width=2, height=2, pad_low=True)
        source = np.full((4, 4), 3, dtype=np.uint16)
        out, _ = _process_image(source, pixel_format=f"mono{bits}")
        assert (out == 3 * factor).all()

    def test_a_camera_already_filling_the_container_is_not_shifted(self, cam_config):
        """16 bits in 16 bits: there is no room to shift into, and doing so would be a loss"""
        cam_config(width=2, height=2, pad_low=True)
        source = np.full((4, 4), 40_000, dtype=np.uint16)
        out, _ = _process_image(source, pixel_format="mono16")
        assert (out == 40_000).all()

    def test_the_top_of_the_camera_range_reaches_the_top_of_the_container(self, cam_config):
        """what the shift is for: 4095 becomes 65520, not 4095 in a frame that allows 65535"""
        cam_config(width=2, height=2, pad_low=True)
        source = np.full((4, 4), 4095, dtype=np.uint16)
        out, _ = _process_image(source, pixel_format="mono12")
        assert (out == 65520).all()

    def test_the_shift_is_exactly_reversible(self, cam_config):
        """
        the one saving grace of padding

        The low bits are zeros, not lost data, so raw counts are recoverable by anyone who
        knows it happened - which is why the fact has to be recorded somewhere.
        """
        cam_config(width=4, height=4, pad_low=True)
        source = _ramp(4, 4) + 1000
        out, cambits = _process_image(source.copy(), pixel_format="mono12")
        assert (out >> (16 - cambits)).tolist() == source.tolist()


class TestCropAndFlipTogether:
    def test_the_crop_happens_before_the_flip(self, cam_config):
        """
        order matters, and an off-centre crop is where it shows

        Rows 1-3, cols 1-2 of a 5x5 ramp, then the columns reversed, is
        [[7,6],[12,11],[17,16]]. Flipping the whole 5x5 first and cropping after would take
        columns 2-3 of the reversed rows instead and give [[8,7],[13,12],[18,17]], so this
        distinguishes the two orders rather than just restating the crop.
        """
        cam_config(width=2, height=3, flip_h=True)
        out, _ = _process_image(_ramp(5, 5), pixel_format="mono16")
        # centre 3 rows x 2 cols of a 5x5 ramp is rows 1-3, cols 1-2, then columns reversed
        assert out.tolist() == [[7, 6], [12, 11], [17, 16]]
