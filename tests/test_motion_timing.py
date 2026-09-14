"""
How long a move takes, according to the firmware model - and according to the mock.

`calculate_movement_time_s` is a trapezoidal motion profile: accelerate to a cruise speed,
hold it, decelerate. Below a certain distance there is no cruise phase at all and the profile
is a triangle. Every acquisition time estimate a user sees is built on it, and the mock
microscope's simulated movement is supposed to behave like it, so the two are checked against
each other as well as against the physics.

Closed-form answers are worth writing out because the implementation is not obviously right at
a glance and the two regimes meet at a point where an off-by-a-factor-of-two is invisible:

    triangle   (d <= v²/a)   t = 2·sqrt(d/a)
    trapezoid  (d >  v²/a)   t = v/a + d/v

They agree at d = v²/a, both giving 2v/a. A profile that is discontinuous there is wrong even
when both halves look plausible on their own.
"""

import math
import time

import pytest

from seafront.hardware.firmware_config import (
    calculate_movement_time_s,
    estimate_xy_movement_time_s,
    get_firmware_config,
)

# a round set to do arithmetic against: cruise at 10 mm/s, accelerate at 100 mm/s²
V = 10.0
A = 100.0
BOUNDARY_MM = V**2 / A  # 1.0 mm: below this there is no cruise phase


def t(distance_mm: float, v: float = V, a: float = A) -> float:
    return calculate_movement_time_s(distance_mm, v, a)


class TestTheTriangleRegime:
    """short moves never reach cruise speed"""

    @pytest.mark.parametrize("distance", [0.01, 0.1, 0.5, 0.9, BOUNDARY_MM])
    def test_it_matches_the_closed_form(self, distance):
        assert t(distance) == pytest.approx(2 * math.sqrt(distance / A), rel=1e-9)

    def test_it_never_claims_to_reach_cruise_speed(self):
        """average speed over a triangle profile is half the peak, which is below v"""
        for distance in (0.01, 0.1, 0.9):
            average = distance / t(distance)
            assert average < V, f"{distance}mm averaged {average:.2f}mm/s, above cruise"

    def test_halving_the_distance_does_not_halve_the_time(self):
        """accelerating the whole way means time goes as the square root of distance"""
        assert t(0.5) / t(0.25) == pytest.approx(math.sqrt(2), rel=1e-9)


class TestTheTrapezoidRegime:
    """long moves spend most of their time at cruise speed"""

    @pytest.mark.parametrize("distance", [1.5, 10.0, 100.0, 1000.0])
    def test_it_matches_the_closed_form(self, distance):
        assert t(distance) == pytest.approx(V / A + distance / V, rel=1e-9)

    def test_the_extra_time_over_cruising_is_constant(self):
        """
        the acceleration penalty does not grow with distance

        Whatever the move, the profile loses exactly v/a seconds against an imaginary one
        that was already at cruise speed when it started.
        """
        for distance in (2.0, 20.0, 200.0):
            penalty = t(distance) - distance / V
            assert penalty == pytest.approx(V / A, rel=1e-9)

    def test_long_moves_approach_cruise_speed(self):
        average = 1000.0 / t(1000.0)
        assert average == pytest.approx(V, rel=0.01)


class TestTheRegimesMeet:
    def test_the_profile_is_continuous_at_the_boundary(self):
        """
        both branches give 2v/a at d = v²/a

        A step here would mean a move one micron longer than the boundary is predicted to
        take meaningfully longer or shorter than one a micron below it.
        """
        just_under = t(BOUNDARY_MM - 1e-6)
        exactly = t(BOUNDARY_MM)
        just_over = t(BOUNDARY_MM + 1e-6)

        assert exactly == pytest.approx(2 * V / A, rel=1e-9)
        assert just_under == pytest.approx(exactly, abs=1e-5)
        assert just_over == pytest.approx(exactly, abs=1e-5)

    def test_time_never_decreases_as_distance_grows(self):
        distances = [0.002, 0.01, 0.1, 0.5, 0.99, 1.0, 1.01, 5.0, 50.0, 500.0]
        times = [t(d) for d in distances]
        assert times == sorted(times), list(zip(distances, times, strict=True))


class TestEdges:
    def test_direction_does_not_matter(self):
        for distance in (0.5, 5.0, 50.0):
            assert t(-distance) == pytest.approx(t(distance), rel=1e-12)

    def test_a_move_shorter_than_a_micron_is_free(self):
        """
        the deadband, written down

        Below 1 um the estimate is zero. Worth knowing when reading an estimate for a
        z-stack: `delta_z_um=1.0` is 0.001 mm, which is NOT inside the deadband, but half a
        micron is, and a stack of those is estimated as taking no time at all.
        """
        assert t(0.0) == 0.0
        assert t(0.0009) == 0.0
        assert t(0.0005) == 0.0
        assert t(0.001) > 0.0, "1um is the first distance that costs anything"

    def test_zero_distance_costs_nothing_at_any_speed(self):
        assert calculate_movement_time_s(0.0, 1.0, 1.0) == 0.0
        assert calculate_movement_time_s(0.0, 1000.0, 1000.0) == 0.0

    def test_a_slower_axis_takes_longer(self):
        assert t(100.0, v=5.0, a=A) > t(100.0, v=50.0, a=A)

    def test_weaker_acceleration_takes_longer(self):
        assert t(100.0, v=V, a=10.0) > t(100.0, v=V, a=1000.0)


class TestXyMovesTogether:
    """the axes move at the same time, so the pair costs whatever the slower one costs"""

    def test_the_pair_costs_the_slower_axis(self):
        config = get_firmware_config()
        x_only = calculate_movement_time_s(30.0, config.MAX_VELOCITY_X_mm, config.MAX_ACCELERATION_X_mm)
        y_only = calculate_movement_time_s(5.0, config.MAX_VELOCITY_Y_mm, config.MAX_ACCELERATION_Y_mm)
        assert estimate_xy_movement_time_s(30.0, 5.0) == pytest.approx(max(x_only, y_only))

    def test_moving_one_axis_costs_what_that_axis_costs(self):
        config = get_firmware_config()
        x_only = calculate_movement_time_s(30.0, config.MAX_VELOCITY_X_mm, config.MAX_ACCELERATION_X_mm)
        assert estimate_xy_movement_time_s(30.0, 0.0) == pytest.approx(x_only)

    def test_moving_neither_axis_is_free(self):
        assert estimate_xy_movement_time_s(0.0, 0.0) == 0.0

    def test_a_diagonal_is_not_the_sum_of_its_legs(self):
        """
        the property that makes it an estimate rather than a total

        Adding the axes would roughly double every site-to-site estimate in an acquisition.
        """
        diagonal = estimate_xy_movement_time_s(20.0, 20.0)
        one_leg = estimate_xy_movement_time_s(20.0, 0.0)
        assert diagonal == pytest.approx(one_leg)


class TestTheMockMovesAtAboutThatSpeed:
    """
    the simulation is checked against the model it claims to simulate

    Not to the millisecond - it is a mock, and it moves in discrete steps - but a mock that
    finished instantly, or took ten times as long, would make every timing test written
    against it meaningless.
    """

    def _move_takes(self, scope, x_mm: float, y_mm: float) -> float:
        import asyncio

        from seafront.server import commands as cmd

        started = time.monotonic()
        asyncio.run(scope.execute(cmd.MoveTo(x_mm=x_mm, y_mm=y_mm)))
        return time.monotonic() - started

    @pytest.fixture
    def scope(self, mock_scope, monkeypatch):
        """the mock with its simulated delays switched back on"""
        monkeypatch.delenv("MOCK_NO_DELAYS", raising=False)
        mock_scope.open_connections()
        yield mock_scope
        mock_scope.close()

    def test_a_long_move_takes_about_as_long_as_predicted(self, scope):
        self._move_takes(scope, 20.0, 20.0)  # get there first
        predicted = estimate_xy_movement_time_s(30.0, 0.0)
        measured = self._move_takes(scope, 50.0, 20.0)
        assert measured == pytest.approx(predicted, rel=0.6), (
            f"predicted {predicted:.2f}s, measured {measured:.2f}s"
        )

    def test_a_longer_move_takes_longer_than_a_shorter_one(self, scope):
        self._move_takes(scope, 10.0, 20.0)
        short = self._move_takes(scope, 15.0, 20.0)
        self._move_takes(scope, 10.0, 20.0)
        long = self._move_takes(scope, 60.0, 20.0)
        assert long > short

    def test_moving_nowhere_is_quick(self, scope):
        self._move_takes(scope, 30.0, 30.0)
        assert self._move_takes(scope, 30.0, 30.0) < 0.5
