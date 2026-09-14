"""
Forbidden areas: the last thing between a protocol and a stage crash.

`ForbiddenArea` is an axis-aligned box in physical millimetres, and a position inside one is
refused - at protocol build time for every site in an acquisition, and per move on the way to
the hardware. Getting the boundary wrong by a hair in the permissive direction is a stage
driven into a clamp; getting it wrong in the safe direction makes wells unreachable.

The config side of this is already checked (`test_example_configs.py` holds the shape of what
is written down). What was not checked is what the geometry actually does, which is the half
that decides whether the stage moves.
"""

import pytest
from pydantic import ValidationError

from seafront.hardware.forbidden_areas import ForbiddenArea, ForbiddenAreaList


def area(name: str = "clamp", **bounds) -> ForbiddenArea:
    defaults = {"min_x_mm": 10.0, "max_x_mm": 20.0, "min_y_mm": 30.0, "max_y_mm": 40.0}
    return ForbiddenArea(name=name, reason="test", **{**defaults, **bounds})


class TestTheBox:
    def test_the_middle_is_inside(self):
        assert area().contains_point(15.0, 35.0) is True

    @pytest.mark.parametrize(
        "x,y",
        [(5.0, 35.0), (25.0, 35.0), (15.0, 25.0), (15.0, 45.0), (0.0, 0.0), (100.0, 100.0)],
    )
    def test_points_outside_are_outside(self, x, y):
        assert area().contains_point(x, y) is False

    @pytest.mark.parametrize(
        "x,y",
        [(10.0, 30.0), (20.0, 40.0), (10.0, 40.0), (20.0, 30.0), (10.0, 35.0), (15.0, 30.0)],
    )
    def test_the_edges_and_corners_count_as_inside(self, x, y):
        """
        the boundary is inclusive, and that is the safe direction

        A point exactly on the edge of a forbidden box is refused. Whichever way this went
        it should be written down, because "it was exactly on the line" is precisely the
        case nobody thinks about until a stage is against a clamp.
        """
        assert area().contains_point(x, y) is True

    def test_a_hair_outside_is_outside(self):
        assert area().contains_point(9.999, 35.0) is False
        assert area().contains_point(20.001, 35.0) is False

    def test_a_degenerate_box_is_still_a_point(self):
        """min == max is legal, and contains exactly that point"""
        line = area(min_x_mm=5.0, max_x_mm=5.0, min_y_mm=5.0, max_y_mm=5.0)
        assert line.contains_point(5.0, 5.0) is True
        assert line.contains_point(5.001, 5.0) is False


class TestBoundsAreValidated:
    def test_an_inverted_x_range_is_refused(self):
        with pytest.raises(ValidationError, match="max_x_mm"):
            area(min_x_mm=20.0, max_x_mm=10.0)

    def test_an_inverted_y_range_is_refused(self):
        with pytest.raises(ValidationError, match="max_y_mm"):
            area(min_y_mm=40.0, max_y_mm=30.0)

    def test_an_inverted_box_cannot_be_smuggled_in_through_a_list(self):
        """
        the path the server actually takes

        Forbidden areas arrive as config objects and are parsed with
        `ForbiddenAreaList.model_validate`. A box that silently validated with its corners
        swapped would contain nothing at all, and would refuse nothing.
        """
        with pytest.raises(ValidationError):
            ForbiddenAreaList.model_validate(
                {"areas": [{"name": "backwards", "min_x_mm": 20.0, "max_x_mm": 10.0,
                            "min_y_mm": 0.0, "max_y_mm": 5.0}]}
            )


class TestTheCircleCheck:
    """`intersects_circle` is the safety-margin version: a point plus an objective radius"""

    def test_a_circle_well_clear_does_not_intersect(self):
        assert area().intersects_circle(0.0, 0.0, 1.0) is False

    def test_a_circle_centred_inside_intersects(self):
        assert area().intersects_circle(15.0, 35.0, 0.1) is True

    def test_a_circle_reaching_over_an_edge_intersects(self):
        """centre 1mm to the left of the box, radius 2mm"""
        assert area().intersects_circle(9.0, 35.0, 2.0) is True

    def test_a_circle_stopping_short_of_an_edge_does_not(self):
        assert area().intersects_circle(9.0, 35.0, 0.5) is False

    def test_a_circle_exactly_touching_an_edge_intersects(self):
        """tangent counts, which is the safe direction again"""
        assert area().intersects_circle(9.0, 35.0, 1.0) is True

    def test_the_nearest_corner_is_what_a_diagonal_approach_hits(self):
        """
        distance to a box is distance to its closest point, not to its centre

        A circle approaching the corner at (10, 30) from (7, 27) is 3·sqrt(2) ~ 4.243mm
        away. A check that measured to the box centre would call this safe.
        """
        diagonal = (3**2 + 3**2) ** 0.5
        assert area().intersects_circle(7.0, 27.0, diagonal - 0.01) is False
        assert area().intersects_circle(7.0, 27.0, diagonal + 0.01) is True

    def test_a_zero_radius_circle_behaves_like_a_point(self):
        assert area().intersects_circle(15.0, 35.0, 0.0) is True
        assert area().intersects_circle(5.0, 35.0, 0.0) is False


class TestTheList:
    def test_an_empty_list_forbids_nothing(self):
        forbidden, which = ForbiddenAreaList().is_position_forbidden(15.0, 35.0)
        assert forbidden is False and which is None

    def test_it_names_the_area_that_refused(self):
        """the client is told which one, so a refusal is actionable rather than mysterious"""
        areas = ForbiddenAreaList(areas=[area("first"), area("clamp", min_x_mm=50.0, max_x_mm=60.0)])
        forbidden, which = areas.is_position_forbidden(15.0, 35.0)
        assert forbidden is True
        assert which is not None and which.name == "first"

    def test_a_position_clear_of_every_area_is_allowed(self):
        areas = ForbiddenAreaList(areas=[area("a"), area("b", min_x_mm=50.0, max_x_mm=60.0)])
        forbidden, which = areas.is_position_forbidden(35.0, 35.0)
        assert forbidden is False and which is None

    def test_overlapping_areas_report_the_first_that_matches(self):
        areas = ForbiddenAreaList(areas=[area("outer", min_x_mm=0.0, max_x_mm=100.0), area("inner")])
        _forbidden, which = areas.is_position_forbidden(15.0, 35.0)
        assert which is not None and which.name == "outer"

    def test_is_movement_safe_is_the_inverse(self):
        areas = ForbiddenAreaList(areas=[area()])
        safe, which = areas.is_movement_safe(15.0, 35.0)
        assert safe is False and which is not None
        safe, which = areas.is_movement_safe(0.0, 0.0)
        assert safe is True and which is None

    def test_conflicting_areas_with_no_margin_is_the_point_check(self):
        areas = ForbiddenAreaList(areas=[area("a"), area("b", min_x_mm=50.0, max_x_mm=60.0)])
        assert [a.name for a in areas.get_conflicting_areas(15.0, 35.0)] == ["a"]
        assert areas.get_conflicting_areas(35.0, 35.0) == []

    def test_a_margin_catches_areas_the_point_check_misses(self):
        """
        what a safety radius is for

        The stage centre is clear of the box, but an objective of that radius is not.
        """
        areas = ForbiddenAreaList(areas=[area("a")])
        assert areas.get_conflicting_areas(9.0, 35.0) == []
        assert [a.name for a in areas.get_conflicting_areas(9.0, 35.0, safety_radius_mm=2.0)] == ["a"]

    def test_a_margin_reports_every_area_it_reaches(self):
        areas = ForbiddenAreaList(
            areas=[area("left", min_x_mm=0.0, max_x_mm=10.0, min_y_mm=0.0, max_y_mm=10.0),
                   area("right", min_x_mm=12.0, max_x_mm=20.0, min_y_mm=0.0, max_y_mm=10.0)]
        )
        caught = {a.name for a in areas.get_conflicting_areas(11.0, 5.0, safety_radius_mm=1.5)}
        assert caught == {"left", "right"}


class TestTheMicroscopeRefusesToGoThere:
    """the geometry above, reached the way a client reaches it"""

    def test_a_move_into_a_forbidden_area_is_refused(self, mock_scope):
        import asyncio

        from fastapi import HTTPException

        from seafront.config.handles import ProtocolConfig
        from seafront.config.registry import ConfigRegistry
        from seafront.server import commands as cmd

        ConfigRegistry.set_value(
            ProtocolConfig.FORBIDDEN_AREAS,
            [{"name": "clamp", "min_x_mm": 10.0, "max_x_mm": 20.0,
              "min_y_mm": 30.0, "max_y_mm": 40.0, "reason": "test"}],
        )
        mock_scope.open_connections()
        try:
            forbidden, reason = mock_scope.is_position_forbidden(15.0, 35.0)
            assert forbidden is True and "clamp" in reason

            allowed, _ = mock_scope.is_position_forbidden(60.0, 60.0)
            assert allowed is False

            with pytest.raises(HTTPException):
                asyncio.run(mock_scope.execute(cmd.MoveTo(x_mm=15.0, y_mm=35.0)))
        finally:
            mock_scope.close()
