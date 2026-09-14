"""
`Locked[T]`: the one lock the concurrency refactor kept, and what it promises.

The nested `Microscope.lock()` hierarchy is gone; what survived is this - an RLock around a
value, with a `locked(blocking=...)` context manager. It guards the cameras, which are still
reached from off the worker thread (`get_hardware_limits`, `extend_machine_config`), and the
whole reason it survived is that those callers need to be able to *decline* to wait.

Three properties matter, and none of them had a test:

- `blocking=False` yields `None` instead of waiting. A caller that treats that `None` as a
  camera is reading from a device somebody else is using; a caller that waits instead is a
  status read that hangs.
- It is reentrant, because `_dispatch` re-enters itself for sub-commands (MoveToWell issues
  MoveTo, autofocus-approach issues autofocus-measure) while already holding the lock.
- It releases on the way out of the block however the block ends, exception included.
"""

import threading
import time

import pytest

from seafront.hardware.microscope import Locked

PROMPT_S = 2.0


class Thing:
    def __init__(self) -> None:
        self.touched = 0


@pytest.fixture
def held():
    """
    a Locked[Thing] held by another thread for the duration of the test

    Released in teardown, where a failed assertion cannot skip it: leaving it held would
    wedge whatever ran next inside the same process.
    """
    guarded = Locked(Thing())
    release = threading.Event()
    holding = threading.Event()

    def hold():
        with guarded.locked() as value:
            assert value is not None
            holding.set()
            release.wait(timeout=30)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert holding.wait(timeout=5), "the holding thread never took the lock"

    yield guarded

    release.set()
    holder.join(timeout=5)


class TestUncontended:
    def test_locked_yields_the_value(self):
        guarded = Locked(Thing())
        with guarded.locked() as value:
            assert isinstance(value, Thing)

    def test_non_blocking_yields_the_value_when_free(self):
        guarded = Locked(Thing())
        with guarded.locked(blocking=False) as value:
            assert isinstance(value, Thing)

    def test_value_reaches_the_thing_without_locking(self):
        """
        the documented escape hatch

        `value` is for reads that do not care, and it must stay usable while somebody else
        holds the lock - `Core.get_hardware_capabilities` reads camera identity through it.
        """
        guarded = Locked(Thing())
        assert isinstance(guarded.value, Thing)

    def test_the_lock_is_free_again_after_the_block(self):
        guarded = Locked(Thing())
        with guarded.locked():
            pass
        with guarded.locked(blocking=False) as value:
            assert value is not None


class TestContended:
    def test_non_blocking_yields_none_rather_than_waiting(self, held):
        started = time.monotonic()
        with held.locked(blocking=False) as value:
            assert value is None, "blocking=False handed out a value somebody else holds"
        took = time.monotonic() - started
        assert took < PROMPT_S, f"blocking=False waited {took:.1f}s"

    def test_value_still_reaches_the_thing_while_held(self, held):
        """the escape hatch is the escape hatch precisely when the lock is busy"""
        started = time.monotonic()
        assert isinstance(held.value, Thing)
        assert time.monotonic() - started < PROMPT_S

    def test_blocking_waits_for_the_holder_and_then_gets_it(self):
        """
        the other side of the same coin: blocking=True is a real wait, and it ends

        Holder managed here rather than by the fixture, so the release can happen mid-test
        and the waiter can be seen both blocked and then satisfied.
        """
        guarded = Locked(Thing())
        holding = threading.Event()
        release = threading.Event()
        got = threading.Event()

        def hold():
            with guarded.locked():
                holding.set()
                release.wait(timeout=30)

        def wait_for_it():
            with guarded.locked() as value:
                if value is not None:
                    got.set()

        holder = threading.Thread(target=hold, daemon=True)
        holder.start()
        assert holding.wait(timeout=5)

        waiter = threading.Thread(target=wait_for_it, daemon=True)
        waiter.start()
        try:
            assert not got.wait(timeout=0.3), "blocking=True did not wait for the holder"
            release.set()
            assert got.wait(timeout=5), "blocking=True never got the lock after it was freed"
        finally:
            release.set()
            holder.join(timeout=5)
            waiter.join(timeout=5)


class TestReentrancy:
    def test_the_same_thread_can_take_it_again(self):
        """
        `_dispatch` re-enters itself for sub-commands while holding the lock

        MoveToWell issues a MoveTo, autofocus-approach issues an autofocus-measure. A
        non-reentrant lock here would deadlock the microscope against itself on those paths.
        """
        guarded = Locked(Thing())
        with guarded.locked() as outer:
            assert outer is not None
            with guarded.locked() as inner:
                assert inner is outer

    def test_non_blocking_succeeds_for_the_thread_that_already_holds_it(self):
        guarded = Locked(Thing())
        with guarded.locked():
            with guarded.locked(blocking=False) as inner:
                assert inner is not None, "an RLock refused its own holder"

    def test_it_is_still_held_until_the_outermost_block_ends(self):
        guarded = Locked(Thing())
        taken_by_other = []

        def try_take():
            with guarded.locked(blocking=False) as value:
                taken_by_other.append(value)

        with guarded.locked():
            with guarded.locked():
                pass
            # the inner block ended, but this thread still holds it
            other = threading.Thread(target=try_take)
            other.start()
            other.join(timeout=5)
        assert taken_by_other == [None], "the lock was released by the inner block"


class TestRelease:
    def test_an_exception_inside_the_block_still_releases(self):
        guarded = Locked(Thing())
        with pytest.raises(RuntimeError, match="boom"):
            with guarded.locked():
                raise RuntimeError("boom")

        with guarded.locked(blocking=False) as value:
            assert value is not None, "the lock was left held by a raising block"

    def test_a_declined_non_blocking_attempt_does_not_release_the_holder(self, held):
        """
        the subtle one

        `locked(blocking=False)` that fails must not run the release path on the way out -
        releasing a lock it never acquired would hand the camera to a second thread while
        the first is still using it.
        """
        with held.locked(blocking=False) as value:
            assert value is None

        # somebody else still holds it, so a second attempt must still decline
        with held.locked(blocking=False) as value:
            assert value is None, "a declined attempt released the real holder's lock"


class TestUnderContention:
    THREADS = 8

    def test_only_one_thread_is_inside_at_a_time(self):
        guarded = Locked(Thing())
        inside = 0
        overlaps: list[int] = []
        bookkeeping = threading.Lock()
        start = threading.Barrier(self.THREADS)

        def worker():
            nonlocal inside
            start.wait()
            for _ in range(50):
                with guarded.locked() as value:
                    assert value is not None
                    with bookkeeping:
                        inside += 1
                        if inside > 1:
                            overlaps.append(inside)
                    time.sleep(0.0005)
                    with bookkeeping:
                        inside -= 1

        threads = [threading.Thread(target=worker) for _ in range(self.THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not overlaps, f"{len(overlaps)} overlapping entries, up to {max(overlaps)} at once"

    def test_decliners_never_see_a_value_somebody_else_has(self):
        """blocking=False under contention: every caller gets the value or None, never a lie"""
        guarded = Locked(Thing())
        inside = 0
        wrong: list[str] = []
        bookkeeping = threading.Lock()
        start = threading.Barrier(self.THREADS)

        def worker():
            nonlocal inside
            start.wait()
            for _ in range(50):
                with guarded.locked(blocking=False) as value:
                    if value is None:
                        continue
                    with bookkeeping:
                        inside += 1
                        if inside > 1:
                            wrong.append(f"{inside} holders at once")
                    time.sleep(0.0005)
                    with bookkeeping:
                        inside -= 1

        threads = [threading.Thread(target=worker) for _ in range(self.THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not wrong, wrong[:3]
