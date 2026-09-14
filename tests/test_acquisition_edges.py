"""
The edges of an acquisition: empty, enormous, and the protocols that must be refused.

An acquisition is accepted or refused entirely up front - wells, sites, channels, z-planes,
timepoints, every site position against the forbidden areas, every channel against the
hardware limits - and only then handed to the worker for however many hours it takes. That
front door is the only place a bad protocol is cheap to reject; everything after it is a run
that has already been reported as started.

So these push on the counting and on the refusals: a protocol that would image nothing, one
that would image an implausible amount, a channel enabled with nothing selected to image it
on, and the boundaries of each multiplication in between.
"""

import asyncio

import pytest
import seaconfig as sc
from fastapi import HTTPException

from seafront.config.handles import CameraConfig
from seafront.config.registry import ConfigRegistry
from seafront.server import commands as cmd


def channel(handle: str = "bfledfull", **overrides) -> sc.AcquisitionChannelConfig:
    base = {
        "name": f"channel {handle}", "handle": handle, "illum_perc": 20.0,
        "exposure_time_ms": 5.0, "analog_gain": 0.0, "z_offset_um": 0.0,
        "num_z_planes": 1, "delta_z_um": 1.0, "filter_handle": None, "enabled": True,
    }
    return sc.AcquisitionChannelConfig(**{**base, **overrides})


def grid(num_x=1, num_y=1, num_t=1, selected_sites=None):
    """a site grid; `selected_sites` defaults to every site selected"""
    mask = [
        sc.acquisition.AcquisitionWellSiteConfigurationSiteSelectionItem(
            row=y, col=x, selected=True if selected_sites is None else (x, y) in selected_sites
        )
        for y in range(num_y)
        for x in range(num_x)
    ]
    return sc.AcquisitionWellSiteConfiguration(
        num_x=num_x, delta_x_mm=0.9, num_y=num_y, delta_y_mm=0.9, num_t=num_t,
        delta_t=sc.acquisition.AcquisitionWellSiteConfigurationDeltaTime(h=0, m=0, s=0),
        mask=mask,
    )


def protocol(
    *, wells=((1, 1),), channels=None, num_x=1, num_y=1, num_t=1,
    selected_sites=None, autofocus=False, project="proj", plate="plate",
) -> sc.AcquisitionConfig:
    return sc.AcquisitionConfig(
        project_name=project,
        plate_name=plate,
        cell_line="",
        grid=grid(num_x, num_y, num_t, selected_sites),
        wellplate_type=sc.Plates[0],
        plate_wells=[
            sc.acquisition.PlateWellConfig(row=r, col=c, selected=True) for r, c in wells
        ],
        channels=[channel()] if channels is None else channels,
        autofocus_enabled=autofocus,
    )


@pytest.fixture
def registry(mock_microscope_env):
    """the camera config the metrics read image size from"""
    return ConfigRegistry


class TestCountingImages:
    """
    every multiplication in the image count, one at a time

    `total = timepoints * wells * sites * sum(z_planes per enabled channel)`. Each factor has
    a zero case that must produce zero rather than one, and the z-plane term is a *sum* over
    channels rather than a product - a channel imaging ten planes and one imaging a single
    plane is eleven images per site, not ten.
    """

    def metrics(self, config):
        return cmd.calculate_acquisition_metrics(config)

    def test_the_simplest_protocol_is_one_image(self, registry):
        assert self.metrics(protocol()).total_num_images == 1

    def test_wells_multiply(self, registry):
        assert self.metrics(protocol(wells=((1, 1), (1, 2), (2, 1)))).total_num_images == 3

    def test_sites_multiply(self, registry):
        assert self.metrics(protocol(num_x=3, num_y=2)).total_num_images == 6

    def test_timepoints_multiply(self, registry):
        assert self.metrics(protocol(num_t=4)).total_num_images == 4

    def test_z_planes_add_across_channels_rather_than_multiplying(self, registry):
        config = protocol(channels=[channel("bfledfull", num_z_planes=10), channel("f488", num_z_planes=1)])
        assert self.metrics(config).total_num_images == 11

    def test_everything_at_once(self, registry):
        config = protocol(
            wells=((1, 1), (1, 2)),
            num_x=2, num_y=3, num_t=5,
            channels=[channel("bfledfull", num_z_planes=2), channel("f488", num_z_planes=3)],
        )
        # 5 timepoints * 2 wells * 6 sites * (2 + 3) planes
        assert self.metrics(config).total_num_images == 5 * 2 * 6 * 5

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"wells": ()},
            {"selected_sites": set()},
            {"channels": [channel(enabled=False)]},
            {"channels": []},
            {"num_t": 0},
        ],
    )
    def test_any_empty_factor_makes_it_zero(self, registry, kwargs):
        assert self.metrics(protocol(**kwargs)).total_num_images == 0

    def test_a_disabled_channel_contributes_nothing(self, registry):
        config = protocol(channels=[channel("bfledfull"), channel("f488", enabled=False, num_z_planes=99)])
        assert self.metrics(config).total_num_images == 1

    def test_deselected_wells_and_sites_are_not_counted(self, registry):
        config = protocol(num_x=3, num_y=1, selected_sites={(0, 0)})
        assert self.metrics(config).total_num_images == 1


class TestEstimatingStorage:
    def test_the_estimate_scales_with_the_image_count(self, registry):
        one = cmd.calculate_acquisition_metrics(protocol())
        ten = cmd.calculate_acquisition_metrics(protocol(wells=tuple((1, c) for c in range(1, 11))))
        assert ten.total_num_images == 10 * one.total_num_images
        assert ten.max_storage_size_GB == pytest.approx(
            10 * one.max_storage_size_GB, rel=1e-6
        )

    def test_an_empty_protocol_needs_no_storage(self, registry):
        assert cmd.calculate_acquisition_metrics(protocol(wells=())).max_storage_size_GB == 0

    @pytest.mark.parametrize(
        "pixel_format,bytes_per_pixel", [("mono8", 1), ("mono12", 2), ("mono16", 2)]
    )
    def test_the_pixel_format_decides_the_bytes_per_pixel(self, registry, pixel_format, bytes_per_pixel):
        ConfigRegistry.set_value(CameraConfig.MAIN_PIXEL_FORMAT, pixel_format)
        metrics = cmd.calculate_acquisition_metrics(protocol())
        width = ConfigRegistry.get(CameraConfig.MAIN_IMAGE_WIDTH_PX).intvalue
        height = ConfigRegistry.get(CameraConfig.MAIN_IMAGE_HEIGHT_PX).intvalue
        assert metrics.image_size_bytes == width * height * bytes_per_pixel

    def test_an_unknown_pixel_format_is_refused_rather_than_guessed(self, registry):
        """guessing the wrong width would under-report the storage a run needs by half"""
        ConfigRegistry.set_value(CameraConfig.MAIN_PIXEL_FORMAT, "mono24")
        with pytest.raises(HTTPException, match="unexpected pixel format"):
            cmd.calculate_acquisition_metrics(protocol())


class TestProtocolsThatMustBeRefused:
    """
    the front door

    Everything here is checked before the run is handed to the worker, because afterwards it
    is a job holding the microscope that has already been reported as started.
    """

    def start(self, core, config):
        return asyncio.run(core.start_acquisition(config_file=config))

    @pytest.mark.parametrize(
        "kwargs,what",
        [
            ({"wells": ()}, "no wells"),
            ({"selected_sites": set()}, "no sites"),
            ({"channels": [channel(enabled=False)]}, "no enabled channels"),
        ],
    )
    def test_a_protocol_that_would_image_nothing_is_refused(self, mock_core, kwargs, what):
        with pytest.raises(HTTPException) as caught:
            self.start(mock_core, protocol(**kwargs))
        assert "no images to acquire" in str(caught.value.detail), what

    def test_the_refusal_says_which_factor_was_empty(self, mock_core):
        """
        'no images to acquire' alone is not actionable

        The counts come back with it, so an operator can see whether they forgot the wells
        or forgot to tick a channel.
        """
        with pytest.raises(HTTPException) as caught:
            self.start(mock_core, protocol(wells=()))
        detail = str(caught.value.detail)
        assert "num_wells" in detail and "num_channels" in detail

    @pytest.mark.parametrize("name", [".", "..", "", "has space", "a/b"])
    def test_an_unusable_project_name_is_refused(self, mock_core, name):
        with pytest.raises(HTTPException) as caught:
            self.start(mock_core, protocol(project=name))
        assert "project name" in str(caught.value.detail)

    @pytest.mark.parametrize("name", [".", "..", "", "has space", "a/b"])
    def test_an_unusable_plate_name_is_refused(self, mock_core, name):
        with pytest.raises(HTTPException) as caught:
            self.start(mock_core, protocol(plate=name))
        assert "plate name" in str(caught.value.detail)

    def test_autofocus_without_calibration_is_refused(self, mock_core):
        """
        a run that would silently image out of focus

        The laser autofocus has to have been calibrated for this microscope; asking for it
        when it has not been is a protocol that cannot do what it says.
        """
        with pytest.raises(HTTPException) as caught:
            self.start(mock_core, protocol(autofocus=True))
        assert "autofocus" in str(caught.value.detail).lower()

    def test_a_site_in_a_forbidden_area_is_refused_before_the_run_starts(self, mock_core):
        """
        named, with the well it belongs to

        Checked for every site up front rather than discovered on the way to one, which is
        the difference between a refused protocol and a stage stopping mid-plate.
        """
        from seafront.config.handles import ProtocolConfig

        ConfigRegistry.set_value(
            ProtocolConfig.FORBIDDEN_AREAS,
            [{"name": "everything", "min_x_mm": -1e6, "max_x_mm": 1e6,
              "min_y_mm": -1e6, "max_y_mm": 1e6, "reason": "test"}],
        )
        with pytest.raises(HTTPException) as caught:
            self.start(mock_core, protocol())
        detail = str(caught.value.detail)
        assert "forbidden" in detail and "well" in detail.lower()

    def test_a_channel_outside_the_hardware_limits_is_refused(self, mock_core):
        with pytest.raises(HTTPException) as caught:
            self.start(mock_core, protocol(channels=[channel(illum_perc=1e6)]))
        assert "illum_perc" in str(caught.value.detail)


class TestTheAcquisitionMapAndIds:
    def test_every_run_gets_a_distinct_id(self):
        from seafront.server.protocol import make_unique_acquisition_id

        ids = {make_unique_acquisition_id() for _ in range(2000)}
        assert len(ids) == 2000

    def test_an_id_is_usable_as_a_path_component(self):
        """ids reach the filesystem in a timeseries run's directory name"""
        from seafront.server.core import name_check
        from seafront.server.protocol import make_unique_acquisition_id

        for _ in range(100):
            assert name_check(make_unique_acquisition_id()) is None

    def test_status_for_an_unknown_id_is_refused(self, mock_core):
        with pytest.raises(HTTPException) as caught:
            asyncio.run(mock_core.get_acquisition_status("never-existed"))
        assert "invalid" in str(caught.value.detail)
