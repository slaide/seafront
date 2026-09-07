"""
Device lease ("lock"): one client at a time across requests, above the one-command-at-a-time
worker. A lease is renewed by use, lapses after its TTL when the worker is idle, never lapses
while the worker is busy, and can be broken by anyone.
"""

from seafront.server.lock import DeviceLock


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


class TestAcquireRelease:
    def test_unheld_lock_allows_everyone(self):
        lock, _, _ = make()
        assert lock.state().held is False
        assert lock.allows(None) is True
        assert lock.allows("anything") is True

    def test_acquire_then_only_token_holder_is_allowed(self):
        lock, _, _ = make()
        lease = lock.acquire(owner="alice")
        assert lease is not None
        assert lock.allows(lease.token) is True
        assert lock.allows(None) is False
        assert lock.allows("wrong") is False
        state = lock.state()
        assert state.held and state.owner == "alice" and state.lock_id == lease.lock_id

    def test_second_client_is_refused(self):
        lock, _, _ = make()
        assert lock.acquire(owner="alice") is not None
        assert lock.acquire(owner="bob") is None
        assert lock.state().owner == "alice"

    def test_holder_can_reacquire_to_renew(self):
        lock, clock, _ = make(ttl=60)
        lease = lock.acquire(owner="alice")
        assert lease is not None
        clock.t += 50
        renewed = lock.acquire(owner="alice-renamed", ttl_s=120, token=lease.token)
        assert renewed is not None
        assert renewed.token == lease.token and renewed.lock_id == lease.lock_id
        clock.t += 100  # 150s after original acquire: would have lapsed under the old TTL
        assert lock.state().held and lock.state().owner == "alice-renamed"

    def test_release_is_idempotent_and_holder_only(self):
        lock, _, _ = make()
        assert lock.release(None) is True  # nothing held
        lease = lock.acquire(owner="alice")
        assert lease is not None
        assert lock.release("wrong") is False
        assert lock.state().held is True
        assert lock.release(lease.token) is True
        assert lock.state().held is False

    def test_lock_id_is_public_and_token_is_not_in_state(self):
        lock, _, _ = make()
        lease = lock.acquire(owner="alice")
        assert lease is not None
        dumped = lock.state().model_dump()
        assert lease.token not in str(dumped)
        assert dumped["lock_id"] == lease.lock_id


class TestExpiry:
    def test_lease_lapses_after_ttl_when_worker_idle(self):
        lock, clock, _ = make(ttl=60)
        lease = lock.acquire(owner="alice")
        assert lease is not None
        clock.t += 59
        assert lock.allows(None) is False
        clock.t += 2
        assert lock.state().held is False
        assert lock.allows(None) is True
        assert lock.matches(lease.token) is False

    def test_lease_never_lapses_while_worker_busy(self):
        lock, clock, busy = make(ttl=60)
        lease = lock.acquire(owner="alice")
        assert lease is not None
        busy[0] = True
        clock.t += 3600
        assert lock.state().held is True
        assert lock.allows(None) is False
        busy[0] = False
        assert lock.state().held is False

    def test_touch_renews_only_for_holder(self):
        lock, clock, _ = make(ttl=60)
        lease = lock.acquire(owner="alice")
        assert lease is not None
        clock.t += 50
        assert lock.touch("wrong") is False
        assert lock.touch(lease.token) is True
        clock.t += 50  # 100s after acquire, 50s after touch
        assert lock.state().held is True

    def test_ttl_has_a_floor(self):
        lock, clock, _ = make()
        assert lock.acquire(owner="alice", ttl_s=0.001) is not None
        clock.t += 1
        assert lock.state().held is True  # MIN_TTL_S, not 1 ms


class TestBreak:
    def test_break_frees_a_foreign_lease_and_reports_it(self):
        lock, _, _ = make()
        lease = lock.acquire(owner="alice")
        assert lease is not None
        previous = lock.break_lock()
        assert previous.held and previous.owner == "alice"
        assert lock.state().held is False
        assert lock.allows(None) is True
        # the old token is dead; a new acquire gets a new lock_id
        assert lock.matches(lease.token) is False
        fresh = lock.acquire(owner="bob")
        assert fresh is not None and fresh.lock_id != lease.lock_id

    def test_break_when_unheld_is_a_noop(self):
        lock, _, _ = make()
        assert lock.break_lock().held is False
