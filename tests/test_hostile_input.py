"""
Trying to break the microscope on purpose.

Everything here is written from the position of a client that is wrong - buggy, confused, or
malicious - rather than one that is merely unlucky. The question each test asks is not "does
this work" but "what reaches the stage if I send this", because the answer used to be
surprising in two places:

- NaN passes every safety check. `x_mm < 0` is False for NaN, and so is
  `min_x <= NaN <= max_x`, so a NaN coordinate is reported not-negative and not-forbidden by
  each check in turn and handed to the driver. Infinity goes the same way.
- A coordinate too large for the wire used to be truncated rather than refused. Asking for
  1e6 mm encoded as **-65353 mm**: a positive request, already past the "must not be
  negative" check and the forbidden-area check, arriving at the firmware with the opposite
  sign. Both checks had looked at the number the client sent, not the number that went out.

The layers are meant to be: the request model refuses what is not a number, the command
refuses what is out of bounds, the geometry refuses what is in a forbidden area, and the wire
encoder refuses what does not fit. These tests poke at each one and at what happens when one
of them is bypassed.
"""

import json
import math

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from seafront.hardware.firmware_config import get_firmware_config
from seafront.hardware.forbidden_areas import ForbiddenArea, ForbiddenAreaList
from seafront.hardware.microcontroller.teensy_microcontroller import (
    twos_complement,
    twos_complement_rev,
)
from seafront.server import commands as cmd
from seafront.server.api.requests import MoveByRequest, MoveToRequest

NOT_NUMBERS = [float("nan"), float("inf"), float("-inf")]


class TestNonFiniteCoordinatesAreRefused:
    """
    the first layer: a coordinate that is not a number never becomes a command

    This has to be caught here, because nothing downstream can catch it. Every check below
    this point is a comparison, and NaN compares False against everything - so each one in
    turn reports the value safe.
    """

    @pytest.mark.parametrize("value", NOT_NUMBERS)
    def test_move_to_refuses_it_on_every_axis(self, value):
        for axis in ("x_mm", "y_mm", "z_mm"):
            with pytest.raises(ValidationError, match="finite"):
                cmd.MoveTo(**{axis: value})

    @pytest.mark.parametrize("value", NOT_NUMBERS)
    def test_move_by_refuses_it(self, value):
        with pytest.raises(ValidationError, match="finite"):
            cmd.MoveBy(axis="x", distance_mm=value)

    @pytest.mark.parametrize("value", NOT_NUMBERS)
    def test_the_request_model_refuses_it_too(self, value):
        """the same rule at the edge of the server, where a client actually reaches it"""
        with pytest.raises(ValidationError, match="finite"):
            MoveToRequest(target_device="m", x_mm=value)
        with pytest.raises(ValidationError, match="finite"):
            MoveByRequest(target_device="m", axis="x", distance_mm=value)

    def test_real_coordinates_still_work(self):
        """the refusal has to be narrow, or it breaks every legitimate move"""
        assert cmd.MoveTo(x_mm=0.0, y_mm=60.0, z_mm=1.5).z_mm == 1.5
        assert cmd.MoveTo().x_mm is None  # omitting an axis still means "do not move it"
        assert cmd.MoveBy(axis="z", distance_mm=-0.001).distance_mm == -0.001

    def test_nan_would_have_passed_the_bounds_check(self):
        """
        why the model has to be the one to refuse it

        Not a test of our code - a test of the premise. If this ever stops being true,
        the layering argument above is wrong and should be revisited.
        """
        nan = float("nan")
        assert not (nan < 0), "NaN is not negative, so a sign check lets it through"
        assert not (nan > 0), "and it is not positive either"

    def test_nan_would_have_passed_the_forbidden_area_check(self):
        area = ForbiddenArea(
            name="clamp", min_x_mm=0.0, max_x_mm=1e9, min_y_mm=0.0, max_y_mm=1e9
        )
        nan = float("nan")
        assert area.contains_point(nan, nan) is False, (
            "a box covering the whole stage does not contain NaN, so geometry cannot be the gate"
        )


class TestTheWireRefusesWhatItCannotCarry:
    """
    the last layer: an encoder that truncates is worse than one that fails

    A truncated position is not a smaller move. It is a different move, often in the other
    direction, and every check upstream has already approved the number the client sent.
    """

    LIMIT = 2**31

    def test_it_accepts_the_whole_signed_range(self):
        assert twos_complement(0, 4) == 0
        assert twos_complement(self.LIMIT - 1, 4) == self.LIMIT - 1
        assert twos_complement(-self.LIMIT, 4) == 2**32 - self.LIMIT

    def test_a_round_trip_survives_the_extremes(self):
        for value in (0, 1, -1, self.LIMIT - 1, -self.LIMIT, 1209448, -1209448):
            assert twos_complement_rev(twos_complement(value, 4), 4) == value

    @pytest.mark.parametrize("value", [2**31, 2**31 + 1, 2**40, -(2**31) - 1, -(2**40)])
    def test_it_refuses_anything_outside_that_range(self, value):
        with pytest.raises(ValueError, match="refusing to truncate"):
            twos_complement(value, 4)

    def test_a_huge_move_is_refused_rather_than_wrapped(self):
        """
        the concrete hazard, in the units a user would type

        1e6 mm used to encode as roughly -65353 mm. Every layer above had approved it.
        """
        config = get_firmware_config()
        usteps = config.mm_to_ustep_x(1e6)
        assert usteps > 2**31, "pick a bigger number; this one now fits and proves nothing"
        with pytest.raises(ValueError, match="refusing to truncate"):
            twos_complement(usteps, 4)

    def test_a_reachable_move_still_encodes(self):
        config = get_firmware_config()
        for mm in (0.0, 1.0, 60.0, 120.0, -60.0):
            encoded = twos_complement(config.mm_to_ustep_x(mm), 4)
            back = twos_complement_rev(encoded, 4) * config.mm_per_ustep_x
            assert back == pytest.approx(mm, abs=1e-3)

    def test_the_error_says_what_was_asked_for(self):
        """a silent wrap is undebuggable; the refusal has to name the number"""
        with pytest.raises(ValueError, match="99999999999"):
            twos_complement(99999999999, 4)


class TestOutOfBoundsCoordinates:
    """
    values that are numbers, but not positions this stage has

    Where the check lives matters here, and right now it lives in one adapter and not the
    other. `SquidAdapter._execute_MoveTo` refuses a negative coordinate on any axis; the mock
    accepts it and simulates a move there. So code developed and tested against the mock can
    meet a refusal it has never seen the first time it runs on a microscope.

    Neither adapter bounds a coordinate from above at all. Until this round the only thing
    stopping an absurd one was the wire encoder truncating it into something plausible; now
    the encoder refuses, which turns a silent wrong move into a loud failure but is still the
    last layer rather than the first.
    """

    @pytest.fixture
    def scope(self, mock_scope):
        mock_scope.open_connections()
        yield mock_scope
        mock_scope.close()

    def _move(self, scope, **kwargs):
        import asyncio

        return asyncio.run(scope.execute(cmd.MoveTo(**kwargs)))

    def test_zero_is_a_legal_coordinate(self, scope):
        """the boundary of the negative check, which must not be off by one"""
        self._move(scope, x_mm=0.0, y_mm=0.0)

    def test_the_real_adapter_refuses_negative_coordinates(self):
        """
        read off SquidAdapter rather than run: it needs hardware to execute

        Held as source because the rule is worth pinning even where it cannot be exercised -
        if it is deleted, this fails and someone has to say why.
        """
        import inspect

        from seafront.hardware import squid

        body = inspect.getsource(squid.SquidAdapter._execute_MoveTo)
        for axis in ("x_mm", "y_mm", "z_mm"):
            assert f"command.{axis} < 0" in body, f"the negative check for {axis} is gone"
        assert "out of bounds" in body

    def test_the_mock_does_not_enforce_that_rule(self, scope):
        """
        the divergence, written down

        Not asserting this is correct - asserting that it is what happens, so that changing
        it is deliberate. A mock that is more permissive than the hardware is a mock that
        lets a bug through to the microscope.
        """
        self._move(scope, x_mm=-1.0)  # squid would refuse this

    def test_nothing_bounds_a_coordinate_from_above(self, scope):
        """the gap this round did not close: there is no declared stage travel to check against"""
        self._move(scope, x_mm=1e6, y_mm=1e6)


class TestForbiddenAreasUnderHostileInput:
    def test_a_denormal_coordinate_is_still_compared_sanely(self):
        area = ForbiddenArea(name="a", min_x_mm=10.0, max_x_mm=20.0, min_y_mm=10.0, max_y_mm=20.0)
        assert area.contains_point(5e-324, 5e-324) is False

    def test_a_step_too_small_to_represent_does_not_escape_the_box(self):
        """
        subtracting a denormal from 10.0 is still 10.0

        Worth pinning because it is the shape of a plausible bad idea: nudging a coordinate
        by an epsilon to get out of a forbidden area does nothing at all at this magnitude,
        and the position stays on the inclusive edge, which is inside.
        """
        area = ForbiddenArea(name="a", min_x_mm=10.0, max_x_mm=20.0, min_y_mm=10.0, max_y_mm=20.0)
        assert 10.0 - 5e-324 == 10.0
        assert area.contains_point(10.0 - 5e-324, 15.0) is True

    def test_a_huge_finite_coordinate_is_outside_a_small_box(self):
        area = ForbiddenArea(name="a", min_x_mm=10.0, max_x_mm=20.0, min_y_mm=10.0, max_y_mm=20.0)
        assert area.contains_point(1e300, 1e300) is False

    def test_a_box_that_covers_everything_covers_everything_finite(self):
        area = ForbiddenArea(
            name="all", min_x_mm=-1e308, max_x_mm=1e308, min_y_mm=-1e308, max_y_mm=1e308
        )
        assert area.contains_point(0.0, 0.0) is True
        assert area.contains_point(1e300, -1e300) is True

    def test_an_empty_area_list_forbids_nothing_rather_than_everything(self):
        """the failure mode that would make the microscope unusable rather than unsafe"""
        forbidden, which = ForbiddenAreaList(areas=[]).is_position_forbidden(15.0, 15.0)
        assert forbidden is False and which is None

    def test_a_negative_safety_radius_does_not_invert_the_check(self):
        areas = ForbiddenAreaList(
            areas=[ForbiddenArea(name="a", min_x_mm=10.0, max_x_mm=20.0, min_y_mm=10.0, max_y_mm=20.0)]
        )
        # a negative radius falls back to the plain point check rather than matching everything
        assert areas.get_conflicting_areas(0.0, 0.0, safety_radius_mm=-100.0) == []
        assert len(areas.get_conflicting_areas(15.0, 15.0, safety_radius_mm=-100.0)) == 1


class TestAbsurdImagingParameters:
    """
    what the hardware layer does with a channel nobody should have configured

    Illumination percentage is the one with teeth: a laser told to run at 1e6 percent, or for
    an hour, is a damaged sample at best.
    """

    @pytest.fixture
    def scope(self, mock_scope):
        mock_scope.open_connections()
        yield mock_scope
        mock_scope.close()

    def channel(self, **overrides):
        import seaconfig as sc

        base = {
            "name": "c", "handle": "bfledfull", "illum_perc": 20.0, "exposure_time_ms": 5.0,
            "analog_gain": 0.0, "z_offset_um": 0.0, "num_z_planes": 1, "delta_z_um": 1.0,
            "filter_handle": None, "enabled": True,
        }
        return sc.AcquisitionChannelConfig(**{**base, **overrides})

    def _snap(self, scope, channel):
        import asyncio

        return asyncio.run(scope.execute(cmd.ChannelSnapshot(channel=channel)))

    @pytest.mark.parametrize(
        "overrides",
        [
            {"illum_perc": 1e6},
            {"illum_perc": -50.0},
            {"exposure_time_ms": -5.0},
            {"exposure_time_ms": 1e9},
            {"analog_gain": -100.0},
            {"analog_gain": 1e6},
        ],
    )
    def test_the_hardware_layer_refuses_them(self, scope, overrides):
        """
        validation lives at the hardware boundary, not in the config model

        seaconfig's channel model takes any float - it is a serialisation format, shared with
        files on disk - so the check that matters is the one the adapter does before it turns
        the numbers into light.
        """
        with pytest.raises(HTTPException):
            self._snap(scope, self.channel(**overrides))

    def test_a_sane_channel_still_works(self, scope):
        result = self._snap(scope, self.channel())
        assert result is not None

    @pytest.mark.parametrize("value", NOT_NUMBERS)
    @pytest.mark.parametrize("param", ["illum_perc", "exposure_time_ms", "analog_gain"])
    def test_a_non_finite_imaging_parameter_is_refused(self, scope, param, value):
        """
        the same trap as the NaN coordinate, one layer over

        `_validate_parameter_range` asks "is it below the minimum" and "is it above the
        maximum". Both are False for NaN, so a value that is not a number was reported
        in-range and handed to the illumination controller.
        """
        with pytest.raises(HTTPException) as caught:
            self._snap(scope, self.channel(**{param: value}))
        assert "finite" in str(caught.value.detail)


class TestMalformedRequests:
    """
    what the HTTP edge does with a body that is not what it says it is

    These go through the request models rather than a live server: that is where the refusal
    is decided, and a model test costs a millisecond instead of a subprocess.
    """

    def test_a_missing_required_field_is_refused(self):
        with pytest.raises(ValidationError):
            MoveByRequest(target_device="m", axis="x")  # no distance_mm

    def test_a_field_of_the_wrong_type_is_refused(self):
        with pytest.raises(ValidationError):
            MoveToRequest(target_device="m", x_mm="over there")

    def test_an_unknown_axis_is_refused(self):
        with pytest.raises(ValidationError):
            MoveByRequest(target_device="m", axis="w", distance_mm=1.0)

    def test_an_unknown_field_is_refused(self):
        """the strictness added for framerate_hz, held to"""
        with pytest.raises(ValidationError):
            MoveToRequest(target_device="m", x_mm=1.0, velocity_mm_s=500.0)

    def test_null_where_a_number_is_required_is_refused(self):
        with pytest.raises(ValidationError):
            MoveByRequest(target_device="m", axis="x", distance_mm=None)

    def test_a_numeric_string_is_still_refused(self):
        """
        strings that look like numbers are not numbers

        Lax coercion here would mean a client's "1e400" arriving as infinity.
        """
        with pytest.raises(ValidationError):
            MoveToRequest(target_device="m", x_mm="1e400")

    def test_a_json_literal_nan_does_not_survive_the_model(self):
        """
        python's json module emits and accepts NaN, which is not valid JSON

        A client written in python that serialises a NaN produces a body this server's
        parser will happily read back, so the model is the only thing standing between that
        and the stage.
        """
        body = json.loads(json.dumps({"target_device": "m", "x_mm": float("nan")}))
        assert math.isnan(body["x_mm"]), "json round-tripped NaN, as expected"
        with pytest.raises(ValidationError, match="finite"):
            MoveToRequest(**body)

    def test_a_deeply_nested_body_is_refused_rather_than_explored(self):
        """a nested structure where a number belongs is a type error, not a traversal"""
        nested = {"a": 1}
        for _ in range(200):
            nested = {"a": nested}
        with pytest.raises(ValidationError):
            MoveToRequest(target_device="m", x_mm=nested)

    def test_an_enormous_string_where_a_device_name_belongs_is_accepted_but_mismatches(self):
        """
        routing compares it, it does not parse it

        Worth pinning: a huge target_device must be a cheap mismatch rather than anything
        that scales with its length.
        """
        request = MoveToRequest(target_device="x" * 1_000_000, x_mm=1.0)
        assert request.target_device != "mocroscope"


class TestTheErrorHandlerCannotCrashOnItsOwnInput:
    """
    A validation error quotes the offending input back, and the offending input is sometimes
    a value JSON has no syntax for.

    NaN and infinity are not JSON. Python's json module writes and reads them anyway, so any
    client can put one in any float field. The 422 that explains the problem then has to
    carry the problem inside it - and rendering it used to raise `ValueError: Out of range
    float values are not JSON compliant` *inside the response*, which starlette turns into a
    bare 500 with no body at all. A client asking "what did I do wrong" got nothing back.
    """

    def test_non_finite_floats_are_rendered_as_text(self):
        from seafront.server.app import json_safe

        assert json_safe(float("nan")) == "nan"
        assert json_safe(float("inf")) == "inf"
        assert json_safe(float("-inf")) == "-inf"

    def test_ordinary_values_are_left_alone(self):
        from seafront.server.app import json_safe

        for value in (1.0, 0.0, -2.5, 1e300, "text", None, True, 7):
            assert json_safe(value) == value

    def test_it_reaches_inside_the_shape_an_error_body_has(self):
        from seafront.server.app import json_safe

        errors = {"detail": [{"type": "finite_number", "loc": ["body", "x_mm"], "input": float("nan")}]}
        assert json_safe(errors)["detail"][0]["input"] == "nan"

    def test_lists_and_tuples_are_walked_too(self):
        from seafront.server.app import json_safe

        assert json_safe([1.0, float("inf"), {"a": [float("nan")]}]) == [1.0, "inf", {"a": ["nan"]}]

    @pytest.mark.parametrize("value", NOT_NUMBERS)
    def test_the_result_always_serialises(self, value):
        from seafront.server.app import json_safe

        body = {"detail": [{"loc": ["body", "x_mm"], "input": value, "ctx": {"limit": value}}]}
        json.dumps(json_safe(body), allow_nan=False)  # must not raise
