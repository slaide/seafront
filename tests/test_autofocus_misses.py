"""
A site the autofocus could not focus, and a cancel while it tries.

- the approach says how it ended: in focus, gave up, out of moves, or failed and why
- a site imaged without focus is counted in the acquisition status, with what happened and
  the z it was imaged at, as soon as it happens
- a failed approach is never counted as a success, however few moves it made
- a cancel is honoured between autofocus attempts, not only once all of them are done
"""

import asyncio
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_acquisition_edges import protocol
from test_autofocus_approach import _approach, _protocol, _put_z_mm, focusing  # noqa: F401 (fixture)

from seafront.server import commands as cmd
from seafront.server.commands import AcquisitionStatusStage, AutofocusOutcome

# --- the approach --------------------------------------------------------------------------


class TestTheApproachSaysHowItEnded:
    def test_arriving(self, focusing):  # noqa: F811
        adapter, device, _camera = focusing
        _put_z_mm(device, 20e-3)
        result = _approach(adapter)
        assert result.outcome == AutofocusOutcome.IN_FOCUS
        assert result.failure is None

    def test_an_offset_out_of_range_is_a_failure_with_the_numbers(self, focusing):  # noqa: F811
        from seafront.config.handles import LaserAutofocusConfig
        from seafront.config.registry import ConfigRegistry

        adapter, device, _camera = focusing
        # a spot 10 px off, read with a calibration that makes that 1 mm
        ConfigRegistry.set_value(LaserAutofocusConfig.CALIBRATION_UM_PER_PX, 100.0)
        _put_z_mm(device, 5e-3)
        result = _approach(adapter)
        assert result.outcome == AutofocusOutcome.FAILED
        assert result.failure is not None and "working range" in result.failure

    def test_a_camera_error_is_a_failure_naming_it(self, focusing):  # noqa: F811
        adapter, device, camera = focusing
        _put_z_mm(device, 5e-3)

        def explode(config):
            raise RuntimeError("camera died mid-approach")

        camera.snap = explode
        result = _approach(adapter)
        assert result.outcome == AutofocusOutcome.FAILED
        assert result.failure is not None and "camera died mid-approach" in result.failure

    def test_running_out_of_moves(self, focusing):  # noqa: F811
        "one move allowed, from 20 um off: made, and not checked"
        adapter, device, _camera = focusing
        _put_z_mm(device, 20e-3)
        result = _approach(adapter, max_num_reps=1)
        assert result.outcome == AutofocusOutcome.OUT_OF_MOVES
        assert result.num_compensating_moves == 1

    def test_giving_up(self, focusing):  # noqa: F811
        adapter, device, camera = focusing
        _put_z_mm(device, 20e-3)
        original = camera.snap

        def receding(config):
            camera.focus_z_mm -= 0.05
            return original(config)

        camera.snap = receding
        assert _approach(adapter).outcome == AutofocusOutcome.GAVE_UP

    def test_a_stage_that_cannot_be_returned_raises(self, focusing, monkeypatch):  # noqa: F811
        """
        the stage stays at whatever z the failure left it: that is not a result to hand back
        """
        adapter, device, camera = focusing
        _put_z_mm(device, 5e-3)

        def explode(config):
            raise RuntimeError("camera died mid-approach")

        def stuck(self, axis, mm):
            raise OSError("stage would not move")

        camera.snap = explode
        monkeypatch.setattr(type(adapter.microcontroller), "move_to_mm", stuck)
        with pytest.raises(OSError, match="stage would not move") as caught:
            _approach(adapter)
        assert any("camera died mid-approach" in note for note in caught.value.__notes__)


# --- the acquisition -----------------------------------------------------------------------


def result(outcome: AutofocusOutcome, offset_um: float = 0.0, failure: str | None = None):
    return cmd.AutofocusApproachTargetDisplacementResult(
        num_compensating_moves=0,
        uncompensated_offset_mm=offset_um * 1e-3,
        reached_threshold=outcome != AutofocusOutcome.OUT_OF_MOVES,
        outcome=outcome,
        failure=failure,
    )


@pytest.fixture
def autofocus_answers(mock_core, monkeypatch):
    """
    the mock microscope, with each autofocus approach answered by `answer(n)` for the n-th
    approach of the run; returns the list the approaches are recorded in
    """
    from seafront.config.handles import LaserAutofocusConfig
    from seafront.config.registry import ConfigRegistry

    ConfigRegistry.set_value(LaserAutofocusConfig.CALIBRATION_IS_CALIBRATED, "yes")
    scope_type = type(mock_core.microscope)
    real_execute = scope_type.execute
    calls: list[int] = []
    answer: list = [lambda n: result(AutofocusOutcome.IN_FOCUS)]

    async def execute(self, command):
        if isinstance(command, cmd.AutofocusApproachTargetDisplacement):
            calls.append(len(calls))
            return answer[0](calls[-1])
        return await real_execute(self, command)

    monkeypatch.setattr(scope_type, "execute", execute)

    def set_answer(fn):
        answer[0] = fn

    return calls, set_answer


def run(mock_core, config):
    started = asyncio.run(mock_core.start_acquisition(config_file=config))
    assert mock_core.acquisition_future is not None
    mock_core.acquisition_future.result(timeout=60)
    return asyncio.run(mock_core.get_acquisition_status(started.acquisition_id))


def test_sites_in_focus_are_not_misses(mock_core, autofocus_answers):
    status = run(mock_core, protocol(autofocus=True, num_x=2))
    assert status.acquisition_status == AcquisitionStatusStage.COMPLETED
    assert status.acquisition_progress.num_autofocus_misses == 0


def test_a_failed_autofocus_is_a_miss_with_its_reason(mock_core, autofocus_answers):
    calls, answer = autofocus_answers
    answer(lambda n: result(AutofocusOutcome.FAILED, failure="no signal found"))

    status = run(mock_core, protocol(autofocus=True, num_x=2))

    # the run goes on: a site that cannot be focused is not a broken microscope
    assert status.acquisition_status == AcquisitionStatusStage.COMPLETED
    progress = status.acquisition_progress
    assert progress.num_autofocus_misses == 2
    miss = progress.recent_autofocus_misses[0]
    assert miss.outcome == AutofocusOutcome.FAILED
    assert "no signal found" in miss.detail and "reference z" in miss.detail
    # a failure is not retried: the same z gives the same spot
    assert len(calls) == 2


def test_zero_moves_is_not_success_when_it_failed(mock_core, autofocus_answers):
    "what the protocol used to read as 'no compensation needed'"
    _calls, answer = autofocus_answers
    answer(lambda n: result(AutofocusOutcome.FAILED, failure="x"))
    status = run(mock_core, protocol(autofocus=True))
    assert status.acquisition_progress.num_autofocus_misses == 1


def test_an_offset_too_large_to_trust_is_a_miss(mock_core, autofocus_answers):
    _calls, answer = autofocus_answers
    answer(lambda n: result(AutofocusOutcome.GAVE_UP, offset_um=50.0))
    status = run(mock_core, protocol(autofocus=True))
    [miss] = status.acquisition_progress.recent_autofocus_misses
    assert "50.0 um left" in miss.detail


def test_close_but_never_in_focus_is_retried_then_a_miss(mock_core, autofocus_answers):
    calls, answer = autofocus_answers
    answer(lambda n: result(AutofocusOutcome.OUT_OF_MOVES, offset_um=2.0))
    status = run(mock_core, protocol(autofocus=True))
    assert len(calls) == 3
    [miss] = status.acquisition_progress.recent_autofocus_misses
    assert "3 attempts" in miss.detail
    assert math.isfinite(miss.imaged_at_z_mm)


def test_a_retry_that_arrives_is_not_a_miss(mock_core, autofocus_answers):
    calls, answer = autofocus_answers
    answer(
        lambda n: result(AutofocusOutcome.IN_FOCUS)
        if n > 0
        else result(AutofocusOutcome.OUT_OF_MOVES, offset_um=2.0)
    )
    status = run(mock_core, protocol(autofocus=True))
    assert len(calls) == 2
    assert status.acquisition_progress.num_autofocus_misses == 0


def test_the_status_keeps_the_recent_misses_and_counts_all(mock_core, autofocus_answers):
    from seafront.server.protocol import AcquisitionStatusReporter

    _calls, answer = autofocus_answers
    answer(lambda n: result(AutofocusOutcome.FAILED, failure=f"miss {n}"))
    keep = AcquisitionStatusReporter.KEEP_RECENT_AUTOFOCUS_MISSES
    status = run(mock_core, protocol(autofocus=True, num_x=3, num_y=4))

    progress = status.acquisition_progress
    assert progress.num_autofocus_misses == 12
    assert len(progress.recent_autofocus_misses) == keep
    assert "miss 11" in progress.recent_autofocus_misses[-1].detail


def test_a_miss_is_shown_before_the_next_image(mock_core, autofocus_answers, monkeypatch):
    """the operator should not have to wait for an exposure to learn the focus was lost"""
    _calls, answer = autofocus_answers
    answer(lambda n: result(AutofocusOutcome.FAILED, failure="no signal found"))
    scope_type = type(mock_core.microscope)
    answering_execute = scope_type.execute
    seen: list[int] = []

    async def execute(self, command):
        if isinstance(command, cmd.ChannelSnapshot) and not seen:
            status = next(iter(mock_core.acquisitions._statuses.values())).last_status
            seen.append(status.acquisition_progress.num_autofocus_misses)
        return await answering_execute(self, command)

    monkeypatch.setattr(scope_type, "execute", execute)
    run(mock_core, protocol(autofocus=True))
    assert seen == [1]


def test_a_cancel_is_honoured_between_autofocus_attempts(mock_core, autofocus_answers):
    calls, answer = autofocus_answers

    def cancel_during_first(n):
        # the protocol's cancel checkpoint reads the event the worker handed this run
        assert mock_core.microscope._current_cancel is not None
        mock_core.microscope._current_cancel.set()
        return result(AutofocusOutcome.OUT_OF_MOVES, offset_um=2.0)

    answer(cancel_during_first)
    status = run(mock_core, protocol(autofocus=True))

    assert status.acquisition_status == AcquisitionStatusStage.CANCELLED
    assert len(calls) == 1, "the retries ran after the cancel"


def test_the_number_of_frames_to_average_can_be_overridden(focusing):  # noqa: F811
    "override_num_images used to be read as `3 or override`, which is always 3"
    adapter, device, camera = focusing
    _put_z_mm(device, 0.0)
    before = camera.frames
    adapter._dispatch(
        adapter.microcontroller,
        cmd.AutofocusMeasureDisplacement(
            config_file=_protocol(), override_num_images=5
        ),
    )
    assert camera.frames - before == 5
