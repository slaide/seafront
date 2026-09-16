"""
"About four hours remaining" - the number the operator decides their evening on.

A plate runs for hours, so the estimate is what tells somebody whether to wait for it or come
back tomorrow, and whether a run that is going slowly is going slowly or has stopped. It was
thirty-five lines of arithmetic inline in the acquisition generator with no test on any of it.

A time series is where it gets interesting, because imaging a timepoint and waiting for the
next one are separate costs. A run of twelve hourly timepoints that images in ten minutes
spends fifty minutes of each hour doing nothing, and an estimate that only counts imaging is
out by a factor of six.
"""

import pytest

from seafront.server.protocol import estimate_remaining_time

HOUR_S = 3600.0


def _estimate(
    *,
    acquired: int,
    total: int,
    elapsed_s: float,
    timepoints: int = 1,
    current_timepoint: int = 1,
    delta_t_s: float = 0.0,
    completed: list[float] | None = None,
):
    return estimate_remaining_time(
        num_images_acquired=acquired,
        num_images_total=total,
        time_since_start_s=elapsed_s,
        num_timepoints=timepoints,
        current_timepoint=current_timepoint,
        delta_t_s=delta_t_s,
        completed_timepoint_imaging_times=completed if completed is not None else [],
    )


class TestBeforeThereIsAnythingToGoOn:
    def test_no_images_yet_means_no_estimate(self):
        """
        None rather than zero or infinity

        Zero reads as "about to finish" and would show a full progress bar on a run that has
        not taken its first image.
        """
        assert _estimate(acquired=0, total=100, elapsed_s=5.0).remaining_s is None

    def test_a_single_timepoint_run_has_no_timepoint_duration(self):
        """the idea does not apply, so the field is absent rather than zero"""
        assert _estimate(acquired=1, total=10, elapsed_s=1.0).timepoint_duration_s is None


class TestASingleTimepoint:
    """
    the simple case: time per image so far, times the images left

    No waiting, so the whole estimate is imaging.
    """

    def test_halfway_through_means_about_as_long_again(self):
        est = _estimate(acquired=50, total=100, elapsed_s=600.0)
        assert est.remaining_s == pytest.approx(600.0)

    def test_the_last_image_leaves_nothing(self):
        assert _estimate(acquired=100, total=100, elapsed_s=600.0).remaining_s == pytest.approx(0.0)

    def test_the_first_image_extrapolates_the_whole_run(self):
        """one second for one of a hundred images, so ninety-nine seconds left"""
        est = _estimate(acquired=1, total=100, elapsed_s=1.0)
        assert est.remaining_s == pytest.approx(99.0)

    @pytest.mark.parametrize("acquired", [1, 10, 25, 50, 75, 99])
    def test_the_estimate_only_ever_shrinks_at_a_steady_rate(self, acquired):
        """
        a run going at a constant rate must show a countdown

        Each image takes a second here, so at image n the answer is 100-n. An estimate that
        wandered up and down at constant speed would be worse than none.
        """
        est = _estimate(acquired=acquired, total=100, elapsed_s=float(acquired))
        assert est.remaining_s == pytest.approx(100.0 - acquired)

    def test_a_slower_run_is_estimated_as_slower(self):
        fast = _estimate(acquired=10, total=100, elapsed_s=10.0).remaining_s
        slow = _estimate(acquired=10, total=100, elapsed_s=100.0).remaining_s
        assert slow == pytest.approx(fast * 10)


class TestATimeSeriesBeforeAnyTimepointFinishes:
    """
    the first timepoint, estimated from partial progress

    There is no completed timepoint to measure yet, so the imaging rate comes from the images
    taken so far and the waits are derived from it.
    """

    def test_the_waiting_is_counted_as_well_as_the_imaging(self):
        """
        four hourly timepoints, ten images each, one minute per image

        Ten minutes of imaging per timepoint and fifty minutes of waiting after each. From
        five images into the first: 35 images left at a minute each, plus three waits of
        fifty minutes.
        """
        est = _estimate(
            acquired=5, total=40, elapsed_s=5 * 60.0,
            timepoints=4, current_timepoint=1, delta_t_s=HOUR_S,
        )
        assert est.remaining_s == pytest.approx(35 * 60.0 + 3 * (HOUR_S - 10 * 60.0))

    def test_the_timepoint_duration_is_reported(self):
        est = _estimate(
            acquired=5, total=40, elapsed_s=5 * 60.0,
            timepoints=4, current_timepoint=1, delta_t_s=HOUR_S,
        )
        assert est.timepoint_duration_s == pytest.approx(10 * 60.0)

    def test_an_interval_shorter_than_the_imaging_adds_no_wait(self):
        """
        the microscope is the bottleneck, so it never idles

        Asking for a timepoint every minute when one takes ten means it runs back to back,
        and the estimate must not subtract time for waits that never happen.
        """
        est = _estimate(
            acquired=5, total=40, elapsed_s=5 * 60.0,
            timepoints=4, current_timepoint=1, delta_t_s=60.0,
        )
        assert est.remaining_s == pytest.approx(35 * 60.0)

    def test_a_longer_interval_means_a_longer_run(self):
        short = _estimate(acquired=5, total=40, elapsed_s=300.0, timepoints=4,
                          current_timepoint=1, delta_t_s=HOUR_S).remaining_s
        long = _estimate(acquired=5, total=40, elapsed_s=300.0, timepoints=4,
                         current_timepoint=1, delta_t_s=2 * HOUR_S).remaining_s
        assert long > short


class TestOnceATimepointHasFinished:
    """
    a measured timepoint beats an extrapolated one

    The first timepoint carries the cost of everything that happens once - homing, the first
    autofocus, warming up - so extrapolating the whole run from partway through it reads high.
    A completed timepoint is the real per-timepoint cost, and takes over as soon as there is
    one.
    """

    def test_the_measured_time_is_used_instead_of_the_current_rate(self):
        """
        the run so far looks slow, but the completed timepoint says otherwise

        Ten images per timepoint; one completed in 100 s. Thirty images left is three
        timepoints, so 300 s of imaging plus the waits - regardless of the elapsed time, which
        includes the wait already served.
        """
        est = _estimate(
            acquired=10, total=40, elapsed_s=5000.0,
            timepoints=4, current_timepoint=1, delta_t_s=0.0,
            completed=[100.0],
        )
        assert est.remaining_s == pytest.approx(300.0)

    def test_several_completed_timepoints_are_averaged(self):
        est = _estimate(
            acquired=20, total=40, elapsed_s=9999.0,
            timepoints=4, current_timepoint=2, delta_t_s=0.0,
            completed=[100.0, 200.0],
        )
        # average 150 s per timepoint, two timepoints' worth of images left
        assert est.remaining_s == pytest.approx(300.0)

    def test_the_reported_duration_is_the_measured_average(self):
        est = _estimate(
            acquired=20, total=40, elapsed_s=9999.0,
            timepoints=4, current_timepoint=2, delta_t_s=0.0,
            completed=[100.0, 200.0],
        )
        assert est.timepoint_duration_s == pytest.approx(150.0)

    def test_the_waits_still_count(self):
        est = _estimate(
            acquired=10, total=40, elapsed_s=5000.0,
            timepoints=4, current_timepoint=1, delta_t_s=HOUR_S,
            completed=[100.0],
        )
        assert est.remaining_s == pytest.approx(300.0 + 3 * (HOUR_S - 100.0))

    def test_later_timepoints_have_fewer_waits_left(self):
        """
        the wait count follows the current timepoint, not the images

        Three timepoints in, only one wait remains however many images are outstanding.
        """
        early = _estimate(acquired=10, total=40, elapsed_s=1.0, timepoints=4,
                          current_timepoint=1, delta_t_s=HOUR_S, completed=[100.0]).remaining_s
        late = _estimate(acquired=30, total=40, elapsed_s=1.0, timepoints=4,
                         current_timepoint=3, delta_t_s=HOUR_S, completed=[100.0]).remaining_s
        assert early > late


class TestItStaysSane:
    """
    the arithmetic has divisions in it, and a progress display must not crash the run
    """

    def test_the_final_image_of_a_time_series_leaves_only_waits(self):
        est = _estimate(
            acquired=40, total=40, elapsed_s=4000.0,
            timepoints=4, current_timepoint=4, delta_t_s=0.0, completed=[100.0],
        )
        assert est.remaining_s == pytest.approx(0.0)

    def test_a_one_image_run_is_finished_when_it_is_finished(self):
        assert _estimate(acquired=1, total=1, elapsed_s=3.0).remaining_s == pytest.approx(0.0)

    def test_the_estimate_is_never_negative(self):
        """an overrun reads as "any moment now", not as time owed back"""
        for acquired in range(1, 41):
            est = _estimate(
                acquired=acquired, total=40, elapsed_s=acquired * 10.0,
                timepoints=4, current_timepoint=4, delta_t_s=HOUR_S, completed=[100.0],
            )
            assert est.remaining_s >= 0

    def test_zero_timepoints_is_treated_as_a_plain_run(self):
        """a guard on the division by num_timepoints, which a malformed protocol could hit"""
        est = _estimate(acquired=1, total=2, elapsed_s=10.0, timepoints=0)
        assert est.remaining_s == pytest.approx(10.0)
