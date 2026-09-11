"""
Device lease ("lock"): one client at a time across requests, above the one-command-at-a-time
worker.

The rule these tests pin: driving the hardware always needs a lease. An unheld lease is a
refusal (`not_held`), not a free pass, so "nobody locked it" and "somebody else locked it"
are both denials and a client can tell them apart. A lease is renewed by use, lapses after
its TTL when the worker is idle, never lapses while the worker is busy, and can be broken
by anyone.

These are the unit tests of the lease itself. tests/test_lock_enforcement.py proves the
server actually applies it to every hardware route.
"""

import threading

import pytest
from pydantic import ValidationError

from seafront.server.lock import (
    DeviceLock,
    Lease,
    LockDecision,
    LockReason,
    LockState,
)


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def make(busy: bool = False, ttl: float = 60.0) -> tuple[DeviceLock, FakeClock, list[bool]]:
    clock = FakeClock()
    busy_flag = [busy]
    lock = DeviceLock(worker_is_busy=lambda: busy_flag[0], default_ttl_s=ttl, clock=clock)
    return lock, clock, busy_flag


def take(lock: DeviceLock, owner: str, **kwargs) -> Lease:
    """acquire, asserting it worked; `acquire` answers with a Lease or the refusing decision"""
    outcome = lock.acquire(owner=owner, **kwargs)
    assert isinstance(outcome, Lease), outcome
    return outcome


def refusal(outcome: Lease | LockDecision) -> LockDecision:
    """the other side of that: assert the acquire was refused and hand back the decision"""
    assert isinstance(outcome, LockDecision), outcome
    return outcome


class TestLeaseIsRequired:
    """The core rule: no lease, no hardware."""

    def test_unheld_lease_permits_nobody(self):
        lock, _, _ = make()
        assert lock.state().held is False
        assert lock.permits(None) is False
        assert lock.permits("anything") is False

    def test_unheld_lease_denies_with_not_held(self):
        lock, _, _ = make()
        decision = lock.check(None)
        assert decision.permitted is False
        assert decision.reason is LockReason.NOT_HELD
        assert decision.state.held is False
        assert decision.state.owner is None

    def test_holder_is_permitted_and_everyone_else_is_not(self):
        lock, _, _ = make()
        lease = take(lock, "alice")
        assert lock.permits(lease.token) is True
        assert lock.permits(None) is False
        assert lock.permits("wrong") is False
        state = lock.state()
        assert state.held and state.owner == "alice" and state.lock_id == lease.lock_id

    def test_foreign_denial_names_the_holder(self):
        lock, _, _ = make()
        take(lock, "alice")
        decision = lock.check("wrong")
        assert decision.permitted is False
        assert decision.reason is LockReason.HELD_BY_OTHER
        assert decision.state.held is True
        assert decision.state.owner == "alice"
        assert decision.state.expires_at is not None

    def test_permitted_decision_says_ok(self):
        lock, _, _ = make()
        lease = take(lock, "alice")
        decision = lock.check(lease.token)
        assert decision.permitted is True
        assert decision.reason is LockReason.OK
        assert decision.state.lock_id == lease.lock_id

    def test_decision_never_carries_the_token(self):
        lock, _, _ = make()
        lease = take(lock, "alice")
        assert lease.token not in str(lock.check(lease.token).model_dump())
        assert lease.token not in str(lock.check("wrong").model_dump())

    def test_released_lease_denies_the_old_token_as_not_held(self):
        lock, _, _ = make()
        lease = take(lock, "alice")
        assert lock.release(lease.token).reason is LockReason.OK
        decision = lock.check(lease.token)
        assert decision.permitted is False
        assert decision.reason is LockReason.NOT_HELD


class TestLockReason:
    """the reason enum is the closed set of answers; the decision is what carries them"""

    def test_every_reason_says_something_useful(self):
        lock, _, _ = make()
        assert "acquire" in lock.check(None).message
        lease = take(lock, "alice")
        assert "alice" in lock.check("wrong").message
        assert lock.check(lease.token).message  # non-empty for the permitted case too

    def test_reasons_travel_as_their_wire_strings(self):
        """a StrEnum member and the string a client sees are the same thing"""
        assert LockReason.NOT_HELD == "not_held"
        assert LockReason.HELD_BY_OTHER == "held_by_other"
        assert LockReason.OK == "ok"
        lock, _, _ = make()
        assert lock.check(None).model_dump()["reason"] == "not_held"

    def test_permitted_follows_the_reason(self):
        """there is no second field to disagree with the reason"""
        for reason in LockReason:
            decision = LockDecision(reason=reason, state=LockState())
            assert decision.permitted is (reason is LockReason.OK)


class TestDecisionsDoNotDrift:
    """
    A decision is a snapshot of the lease at the instant a request was judged.

    Everything that refuses a request hands back its own decision, and that decision keeps
    saying the same thing afterwards. Without this, a refusal that is turned into a 423 a
    moment later can name whoever holds the lease by then instead of whoever actually
    refused the request.
    """

    def test_message_keeps_naming_the_lease_that_refused(self):
        lock, _, _ = make()
        take(lock, "alice")
        denial = lock.check("wrong")
        assert "alice" in denial.message

        # alice goes away and bob takes over, after the request was already refused
        lock.break_lock()
        take(lock, "bob")

        assert "alice" in denial.message, "the refusal must still describe alice's lease"
        assert "bob" not in denial.message
        assert denial.state.owner == "alice"

    def test_a_decision_cannot_be_edited_after_the_fact(self):
        lock, _, _ = make()
        take(lock, "alice")
        denial = lock.check("wrong")
        with pytest.raises(ValidationError):
            denial.state = LockState(held=True, owner="bob")  # type: ignore[misc]
        with pytest.raises(ValidationError):
            denial.reason = LockReason.NOT_HELD  # type: ignore[misc]
        assert "alice" in denial.message

    def test_lock_state_is_a_snapshot_too(self):
        lock, _, _ = make()
        take(lock, "alice")
        state = lock.state()
        with pytest.raises(ValidationError):
            state.owner = "bob"  # type: ignore[misc]

    def test_refused_acquire_reports_the_lease_that_refused_it(self):
        lock, _, _ = make()
        take(lock, "alice")
        denied = refusal(lock.acquire(owner="bob"))
        lock.break_lock()
        take(lock, "carol")
        assert denied.state.owner == "alice" and "alice" in denied.message

    def test_refused_release_reports_the_lease_that_refused_it(self):
        lock, _, _ = make()
        take(lock, "alice")
        refused = lock.release("bob's token")
        lock.break_lock()
        take(lock, "carol")
        assert refused.reason is LockReason.HELD_BY_OTHER
        assert refused.state.owner == "alice" and "alice" in refused.message


class TestAcquireRelease:
    def test_second_client_is_refused(self):
        lock, _, _ = make()
        take(lock, "alice")
        denied = refusal(lock.acquire(owner="bob"))
        assert denied.reason is LockReason.HELD_BY_OTHER
        assert denied.state.owner == "alice"
        assert lock.state().owner == "alice"

    def test_holder_can_reacquire_to_renew(self):
        lock, clock, _ = make(ttl=60)
        lease = take(lock, "alice")
        clock.t += 50
        renewed = take(lock, "alice-renamed", ttl_s=120, token=lease.token)
        assert renewed.token == lease.token and renewed.lock_id == lease.lock_id
        clock.t += 100  # 150s after original acquire: would have lapsed under the old TTL
        assert lock.state().held and lock.state().owner == "alice-renamed"

    def test_release_is_idempotent_and_holder_only(self):
        lock, _, _ = make()
        assert lock.release(None).reason is LockReason.NOT_HELD  # nothing held
        lease = take(lock, "alice")
        refused = lock.release("wrong")
        assert refused.reason is LockReason.HELD_BY_OTHER
        assert lock.state().held is True
        released = lock.release(lease.token)
        assert released.reason is LockReason.OK
        assert released.state.owner == "alice", "the caller can still say whose lease it was"
        assert lock.state().held is False

    def test_lock_id_is_public_and_token_is_not_in_state(self):
        lock, _, _ = make()
        lease = take(lock, "alice")
        dumped = lock.state().model_dump()
        assert lease.token not in str(dumped)
        assert dumped["lock_id"] == lease.lock_id

    def test_a_new_lease_gets_a_new_token_and_id(self):
        lock, _, _ = make()
        first = take(lock, "alice")
        assert lock.release(first.token).reason is LockReason.OK
        second = take(lock, "bob")
        assert second.token != first.token and second.lock_id != first.lock_id
        assert lock.permits(first.token) is False

    def test_concurrent_acquires_produce_exactly_one_holder(self):
        lock, _, _ = make()
        leases = []
        barrier = threading.Barrier(8)

        def contend(n: int):
            barrier.wait()
            outcome = lock.acquire(owner=f"client-{n}")
            if isinstance(outcome, Lease):
                leases.append(outcome)

        threads = [threading.Thread(target=contend, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(leases) == 1
        assert lock.permits(leases[0].token) is True


class TestExpiry:
    def test_lease_lapses_after_ttl_when_worker_idle(self):
        lock, clock, _ = make(ttl=60)
        lease = take(lock, "alice")
        clock.t += 59
        assert lock.permits(lease.token) is True
        clock.t += 2
        assert lock.state().held is False
        assert lock.permits(lease.token) is False

    def test_a_lapsed_lease_leaves_the_hardware_locked_not_open(self):
        """the lapse frees the lease for the next client, it does not open the hardware up"""
        lock, clock, _ = make(ttl=60)
        take(lock, "alice")
        clock.t += 61
        decision = lock.check(None)
        assert decision.permitted is False
        assert decision.reason is LockReason.NOT_HELD
        take(lock, "bob")

    def test_lease_never_lapses_while_worker_busy(self):
        lock, clock, busy = make(ttl=60)
        lease = take(lock, "alice")
        busy[0] = True
        clock.t += 3600
        assert lock.state().held is True
        assert lock.permits(lease.token) is True
        assert lock.permits("wrong") is False
        busy[0] = False
        assert lock.state().held is False

    def test_check_renews_for_the_holder_only(self):
        lock, clock, _ = make(ttl=60)
        take(lock, "alice")
        clock.t += 50
        assert lock.check("wrong").permitted is False  # a stranger's request does not renew
        clock.t += 11  # 61s after acquire: would have lapsed if the stranger had renewed it
        assert lock.state().held is False

    def test_check_renews_the_expiry(self):
        lock, clock, _ = make(ttl=60)
        lease = take(lock, "alice")
        clock.t += 50
        assert lock.check(lease.token).permitted is True
        clock.t += 50  # 100s after acquire, 50s after the renewing request
        assert lock.state().held is True

    def test_permits_does_not_renew(self):
        """the read-only probe must not keep a lease alive on its own"""
        lock, clock, _ = make(ttl=60)
        lease = take(lock, "alice")
        clock.t += 50
        assert lock.permits(lease.token) is True
        clock.t += 11
        assert lock.state().held is False

    def test_ttl_has_a_floor(self):
        lock, clock, _ = make()
        take(lock, "alice", ttl_s=0.001)
        clock.t += 1
        assert lock.state().held is True  # MIN_TTL_S, not 1 ms


class TestBreak:
    def test_break_frees_a_foreign_lease_and_reports_it(self):
        lock, _, _ = make()
        lease = take(lock, "alice")
        previous = lock.break_lock()
        assert previous.held and previous.owner == "alice"
        assert lock.state().held is False
        # broken, not handed over: the hardware is closed again until someone acquires
        assert lock.permits(None) is False
        assert lock.check(lease.token).reason is LockReason.NOT_HELD
        fresh = take(lock, "bob")
        assert fresh.lock_id != lease.lock_id

    def test_break_when_unheld_is_a_noop(self):
        lock, _, _ = make()
        assert lock.break_lock().held is False

    def test_break_works_while_the_worker_is_busy(self):
        """the takeover path: a run is in flight, its owner is gone, someone else takes over"""
        lock, _, busy = make()
        lease = take(lock, "alice")
        busy[0] = True
        assert lock.break_lock().owner == "alice"
        assert lock.permits(lease.token) is False
        bob = take(lock, "bob")
        assert lock.permits(bob.token) is True
