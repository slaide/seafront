"""
Where an acquisition goes, and in what order.

`ProtocolGenerator` turns a protocol into a route: which wells, which sites inside each well,
and which z planes at each site. The route decides both what the images are of and how long
the run takes, and none of it had a test - the acquisition loop it lives in is the least
covered part of the server.

The two halves are worth separating. The *ordering* is an optimisation: a bad order costs
time and nothing else, because every site still gets imaged. The *coordinates* are not an
optimisation at all - a site placed wrongly images the wrong part of the well, and the
resulting file says it is the right part.
"""

import asyncio
import itertools

import pytest
import seaconfig as sc
from ome_types import OME

from seafront.server import commands as cmds
from seafront.server.protocol import ProtocolGenerator

PLATE_ID = "revvity-384-6057800"


def _plate() -> sc.Wellplate:
    return next(p for p in sc.plates.Plates if p.Model_id == PLATE_ID)


def _channel(
    handle: str = "fluo405",
    num_z_planes: int = 1,
    delta_z_um: float = 1.0,
    z_offset_um: float = 0.0,
) -> sc.AcquisitionChannelConfig:
    return sc.AcquisitionChannelConfig(
        name=handle,
        handle=handle,
        illum_perc=100.0,
        exposure_time_ms=5.0,
        analog_gain=0.0,
        z_offset_um=z_offset_um,
        num_z_planes=num_z_planes,
        delta_z_um=delta_z_um,
        enabled=True,
    )


@pytest.fixture
def make_protocol(mock_microscope_env):
    """
    a ProtocolGenerator over a real 384-well plate

    mock_microscope_env because model_post_init asks the config registry for the camera's
    frame size to work out how much storage the run will need, and asserts that the
    configured output directory exists before it plans anything.
    """
    (mock_microscope_env / "images").mkdir(exist_ok=True)

    def build(
        *,
        wells: list[tuple[int, int]] | None = None,
        sites: int = 1,
        delta_mm: float = 0.9,
        channels: list[sc.AcquisitionChannelConfig] | None = None,
        num_t: int = 1,
        imaging_order: str | None = None,
    ) -> ProtocolGenerator:
        plate = _plate()
        wells = wells if wells is not None else [(1, 1)]
        config = sc.AcquisitionConfig(
            project_name="p",
            plate_name="pl",
            cell_line="c",
            autofocus_enabled=False,
            grid=sc.AcquisitionWellSiteConfiguration(
                num_x=sites,
                delta_x_mm=delta_mm,
                num_y=sites,
                delta_y_mm=delta_mm,
                num_t=num_t,
                delta_t=sc.AcquisitionWellSiteConfigurationDeltaTime(h=0, m=0, s=1),
                mask=[
                    sc.AcquisitionWellSiteConfigurationSiteSelectionItem(row=r, col=c, selected=True)
                    for r in range(sites)
                    for c in range(sites)
                ],
            ),
            wellplate_type=plate,
            plate_wells=[
                sc.PlateWellConfig(col=c, row=r, selected=True) for r, c in wells
            ],
            channels=channels if channels is not None else [_channel()],
            machine_config=[],
            comment="",
            spec_version=sc.LATEST_SPEC_VERSION,
            timestamp=None,
        )
        status = cmds.AcquisitionStatus(
            acquisition_id="test",
            queue_in=asyncio.Queue(),
            queue_out=asyncio.Queue(),
            last_status=None,
            thread_is_running=False,
        )
        return ProtocolGenerator(
            config_file=config,
            handle_q_in=lambda: None,
            plate=plate,
            acquisition_status=status,
            ome_template=OME(),
        )

    return build


class TestHowManyImagesTheRunWillProduce:
    """
    wells x sites x channels x planes x timepoints

    The estimate is shown before a run starts and drives the storage-size warning, so an
    error here is an operator told a plate will take an hour when it will take four.
    """

    def test_a_single_site_single_channel_run_is_one_image(self, make_protocol):
        assert make_protocol().num_images_total == 1

    def test_sites_multiply(self, make_protocol):
        assert make_protocol(sites=3).num_images_total == 9  # a 3x3 grid

    def test_wells_multiply(self, make_protocol):
        assert make_protocol(wells=[(0, 0), (1, 1), (2, 2)]).num_images_total == 3

    def test_z_planes_multiply(self, make_protocol):
        assert make_protocol(channels=[_channel(num_z_planes=5)]).num_images_total == 5

    def test_channels_multiply(self, make_protocol):
        protocol = make_protocol(channels=[_channel("a"), _channel("b"), _channel("c")])
        assert protocol.num_images_total == 3

    def test_timepoints_multiply(self, make_protocol):
        assert make_protocol(num_t=4).num_images_total == 4

    def test_channels_with_different_stack_depths_are_summed_not_multiplied(self, make_protocol):
        """
        each channel brings its own number of planes

        A run with a 5-plane fluorescence channel and a 1-plane brightfield takes six images
        per site, not ten - the depths add up across channels rather than multiplying.
        """
        protocol = make_protocol(
            channels=[_channel("deep", num_z_planes=5), _channel("flat", num_z_planes=1)]
        )
        assert protocol.num_images_total == 6

    def test_everything_together(self, make_protocol):
        protocol = make_protocol(
            wells=[(0, 0), (1, 1)],
            sites=2,
            channels=[_channel("a", num_z_planes=3), _channel("b", num_z_planes=2)],
            num_t=2,
        )
        # 2 wells * 4 sites * (3 + 2) planes * 2 timepoints
        assert protocol.num_images_total == 80


class TestWhereASiteLands:
    """
    the coordinates, which are not an optimisation

    A site is the well's offset on the plate, plus the grid's own origin inside the well, plus
    this site's step. Getting it wrong images the wrong part of the well and records the
    position it meant to be at.
    """

    def test_a_single_site_sits_at_the_centre_of_the_well(self, make_protocol):
        protocol = make_protocol(sites=1)
        plate = _plate()
        well = protocol.config_file.plate_wells[0]
        site = protocol.config_file.grid.mask[0]
        x_mm, y_mm = protocol.get_physical_position_for_well_site(well, site)
        assert x_mm == pytest.approx(plate.get_well_offset_x(well.well_name) + plate.Well_size_x_mm / 2)
        assert y_mm == pytest.approx(plate.get_well_offset_y(well.well_name) + plate.Well_size_y_mm / 2)

    def test_a_grid_is_centred_on_the_well_too(self, make_protocol):
        """
        the sites straddle the centre rather than starting at it

        A 2x2 grid 0.9 mm apart puts its sites 0.45 mm either side of the middle, so the
        imaged area is the middle of the well and not a corner of it.
        """
        protocol = make_protocol(sites=2, delta_mm=0.9)
        plate = _plate()
        well = protocol.config_file.plate_wells[0]
        xs = [
            protocol.get_physical_position_for_well_site(well, site)[0]
            for site in protocol.config_file.grid.mask
        ]
        centre = plate.get_well_offset_x(well.well_name) + plate.Well_size_x_mm / 2
        assert min(xs) == pytest.approx(centre - 0.45)
        assert max(xs) == pytest.approx(centre + 0.45)

    def test_neighbouring_sites_are_one_step_apart(self, make_protocol):
        protocol = make_protocol(sites=3, delta_mm=1.25)
        well = protocol.config_file.plate_wells[0]
        by_col = {}
        for site in protocol.config_file.grid.mask:
            if site.row == 0:
                by_col[site.col] = protocol.get_physical_position_for_well_site(well, site)[0]
        assert by_col[1] - by_col[0] == pytest.approx(1.25)
        assert by_col[2] - by_col[1] == pytest.approx(1.25)

    def test_different_wells_are_a_well_pitch_apart(self, make_protocol):
        """the plate's own geometry, which the protocol must not add its own offset to"""
        protocol = make_protocol(wells=[(0, 0), (0, 1)])
        plate = _plate()
        site = protocol.config_file.grid.mask[0]
        first, second = protocol.config_file.plate_wells
        x1, _ = protocol.get_physical_position_for_well_site(first, site)
        x2, _ = protocol.get_physical_position_for_well_site(second, site)
        assert x2 - x1 == pytest.approx(plate.Well_distance_x_mm)

    def test_every_planned_position_is_on_the_plate(self, make_protocol):
        """a coordinate off the plate is a crash into the holder, not a bad image"""
        protocol = make_protocol(wells=[(0, 0), (7, 11), (15, 23)], sites=2)
        plate = _plate()
        for info in protocol.iter_positions():
            x_mm, y_mm = info.physical_position
            assert 0 < x_mm < plate.Length_mm
            assert 0 < y_mm < plate.Width_mm


class TestTheZStack:
    """
    the planes at one site, which are what a stack is

    n planes delta apart, centred on the site's reference z - so the reference is imaged, and
    the stack reaches equally far either side of it rather than only upward.
    """

    def test_a_single_plane_sits_exactly_at_the_reference(self, make_protocol):
        protocol = make_protocol(channels=[_channel(num_z_planes=1)])
        planes = list(protocol.iter_z_positions(reference_z_mm=5.0))
        assert len(planes) == 1
        assert planes[0].z_position_mm == pytest.approx(5.0)

    def test_the_stack_is_centred_on_the_reference(self, make_protocol):
        """three planes 2 um apart around z=5 mm reach 2 um below and 2 um above"""
        protocol = make_protocol(channels=[_channel(num_z_planes=3, delta_z_um=2.0)])
        zs = [p.z_position_mm for p in protocol.iter_z_positions(5.0)]
        assert zs == pytest.approx([5.0 - 2e-3, 5.0, 5.0 + 2e-3])

    def test_an_even_stack_straddles_the_reference(self, make_protocol):
        """with no middle plane, the reference falls between the two innermost"""
        protocol = make_protocol(channels=[_channel(num_z_planes=2, delta_z_um=4.0)])
        zs = [p.z_position_mm for p in protocol.iter_z_positions(5.0)]
        assert zs == pytest.approx([5.0 - 2e-3, 5.0 + 2e-3])

    def test_the_planes_are_evenly_spaced(self, make_protocol):
        protocol = make_protocol(channels=[_channel(num_z_planes=5, delta_z_um=1.5)])
        zs = [p.z_position_mm for p in protocol.iter_z_positions(0.0)]
        gaps = [b - a for a, b in itertools.pairwise(zs)]
        assert gaps == pytest.approx([1.5e-3] * 4)

    def test_a_channel_offset_moves_that_channel_only(self, make_protocol):
        """
        chromatic focus differs per channel, so each carries its own offset from the site's z

        The offset shifts one channel's whole stack; it must not move the other's.
        """
        protocol = make_protocol(
            channels=[
                _channel("plain", num_z_planes=1),
                _channel("offset", num_z_planes=1, z_offset_um=10.0),
            ]
        )
        by_handle = {p.channel.handle: p.z_position_mm for p in protocol.iter_z_positions(5.0)}
        assert by_handle["plain"] == pytest.approx(5.0)
        assert by_handle["offset"] == pytest.approx(5.0 + 10e-3)

    def test_the_plane_index_counts_from_the_bottom_of_the_stack(self, make_protocol):
        protocol = make_protocol(channels=[_channel(num_z_planes=3, delta_z_um=2.0)])
        planes = sorted(protocol.iter_z_positions(5.0), key=lambda p: p.z_position_mm)
        assert [p.plane_index for p in planes] == [0, 1, 2]


class TestTheRouteIsAnOptimisation:
    """
    the order, which costs time and nothing else

    Every selected well and site is imaged whichever order they come in, so these are about
    the stage travelling less - and about the route never dropping or repeating a position,
    which would stop being merely slow.
    """

    def test_every_well_is_visited_exactly_once(self, make_protocol):
        wells = [(0, 0), (3, 7), (1, 1), (15, 23), (8, 4)]
        protocol = make_protocol(wells=wells)
        visited = [(w.row, w.col) for w in protocol.get_greedy_well_order()]
        assert sorted(visited) == sorted(wells)

    def test_the_route_starts_at_the_top_left_most_well(self, make_protocol):
        """a fixed start, so the same protocol takes the same route twice"""
        protocol = make_protocol(wells=[(5, 5), (0, 2), (0, 1), (9, 9)])
        assert (protocol.get_greedy_well_order()[0].row, protocol.get_greedy_well_order()[0].col) == (0, 1)

    def test_it_takes_the_nearest_well_next(self, make_protocol):
        """
        the greedy step

        From (0,0) the well at (0,1) is nearer than the one at (9,9), so it comes first
        whatever order they were selected in.
        """
        protocol = make_protocol(wells=[(9, 9), (0, 0), (0, 1)])
        route = [(w.row, w.col) for w in protocol.get_greedy_well_order()]
        assert route == [(0, 0), (0, 1), (9, 9)]

    def test_a_single_well_needs_no_ordering(self, make_protocol):
        protocol = make_protocol(wells=[(4, 4)])
        assert [(w.row, w.col) for w in protocol.get_greedy_well_order()] == [(4, 4)]

    def test_every_site_is_visited_exactly_once(self, make_protocol):
        protocol = make_protocol(sites=3)
        well = protocol.config_file.plate_wells[0]
        order = protocol.get_optimized_site_order_for_well(well, (0.0, 0.0))
        assert sorted((s.row, s.col) for s in order) == sorted(
            (s.row, s.col) for s in protocol.config_file.grid.mask
        )

    def test_the_first_site_is_the_one_nearest_the_stage(self, make_protocol):
        """
        entering a well from wherever the last one left the stage

        Approaching from opposite corners has to pick opposite sites, or the ordering is not
        reading the position it was given.
        """
        protocol = make_protocol(sites=2, delta_mm=0.9)
        well = protocol.config_file.plate_wells[0]
        corners = [
            protocol.get_physical_position_for_well_site(well, s)
            for s in protocol.config_file.grid.mask
        ]
        near = protocol.get_optimized_site_order_for_well(well, min(corners))[0]
        far = protocol.get_optimized_site_order_for_well(well, max(corners))[0]
        assert (near.row, near.col) != (far.row, far.col)

    def test_the_position_iterator_covers_the_whole_plan(self, make_protocol):
        protocol = make_protocol(wells=[(0, 0), (2, 2)], sites=2)
        positions = list(protocol.iter_positions())
        assert len(positions) == 2 * 4
        assert len({(p.well.well_name, p.site.row, p.site.col) for p in positions}) == 8
