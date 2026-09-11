"""
Client-held device lease (the "lock"), layered above the MicroscopeWorker.

The worker guarantees one hardware command at a time. The lease guarantees one *client* at a
time across requests: every hardware-mutating route refuses (423) any request that does not
carry the lease token, including when nobody holds the lease at all. Driving the hardware is
therefore always a deliberate two-step: take the lease, then act. Status reads never need it.
Cancel is never gated either: it exists to prevent hardware damage, so anyone may stop a run.
Anyone may also break a lease, which is logged with owner and reason.

This is coordination, not authentication. Seafront has no auth; the token stops accidents
between browser tabs and scripts, not a hostile client. One lease per process, and a seafront
process is exactly one device (the gateway proxies one port per box), so `target_device` needs
no special handling here.
"""

from __future__ import annotations

import datetime as dt
import secrets
import threading
import time
import typing as tp
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, computed_field


class LockState(BaseModel):
    "public view of the device lease; safe to broadcast, carries no token"

    model_config = ConfigDict(frozen=True)

    held: bool = False
    owner: str | None = None
    "free-text label the holder gave on acquire (who to talk to)"
    lock_id: str | None = None
    "public id of the current lease; a client compares it with the id it got on acquire"
    expires_at: str | None = None
    "ISO-8601 UTC. Renewed by every request carrying the token; never lapses while the worker is busy"


class LockReason(StrEnum):
    """
    why a request may or may not touch the hardware

    A `StrEnum`, so it crosses the wire as the plain string a client sees in a 423 body while
    staying a closed set on this side: nothing constructs one of these from a literal, and a
    `match` over it is exhaustive.
    """

    OK = "ok"
    "the request carries the current lease's token"
    NOT_HELD = "not_held"
    "nobody holds the lease; the client must acquire one before driving the hardware"
    HELD_BY_OTHER = "held_by_other"
    "somebody else holds it; the lease state says who, and until when"


def _describe(reason: LockReason, state: LockState) -> str:
    """
    the one place the refusal wording lives

    Private, and it has to stay that way: a reason only means anything together with the lease
    state it was decided against. Describing `held_by_other` with a freshly read state would
    name whoever holds the lease *now*, which can be a different client than the one that
    actually refused the request. `LockDecision` pairs the two at decision time and is frozen,
    so there is no way to call this with a mismatched state.
    """
    match reason:
        case LockReason.OK:
            return "the request holds the device lease"
        case LockReason.NOT_HELD:
            return "hardware commands need the device lease; POST /api/lock/acquire first"
        case LockReason.HELD_BY_OTHER:
            return f"microscope is locked by '{state.owner}'"


class LockDecision(BaseModel):
    """
    outcome of checking one request's token against the lease

    Frozen, and always built inside the lock's mutex: reason, state and message are one
    snapshot of the lease at the instant the request was judged. Later acquires and releases
    do not change what this says, which is what makes it safe to carry into an error body or
    a log line well after the fact.
    """

    model_config = ConfigDict(frozen=True)

    reason: LockReason
    state: LockState
    "the lease as it was at decision time"

    @property
    def permitted(self) -> bool:
        "whether this request may run a hardware-mutating route"
        return self.reason is LockReason.OK

    @computed_field
    @property
    def message(self) -> str:
        "the refusal (or approval) in words, for an error body or a log line"
        return _describe(self.reason, self.state)


class Lease(BaseModel):
    "what the acquirer gets back; `token` is the secret, everything else is public"

    token: str
    lock_id: str
    owner: str
    expires_at: str


class DeviceLock:
    """
    Single lease with owner label, secret token and expiry.

    Expiry is evaluated lazily. A lease is live while `now < expires_at`, OR while the worker
    is busy: only the holder can submit while locked, so a running job is the holder's and an
    hour-long acquisition can't lose its lease. Every token-bearing request renews the expiry.
    """

    DEFAULT_TTL_S: float = 600.0
    MIN_TTL_S: float = 5.0

    def __init__(
        self,
        worker_is_busy: tp.Callable[[], bool],
        *,
        default_ttl_s: float = DEFAULT_TTL_S,
        clock: tp.Callable[[], float] = time.monotonic,
    ) -> None:
        self._worker_is_busy = worker_is_busy
        self._default_ttl_s = default_ttl_s
        self._clock = clock
        self._mutex = threading.Lock()
        self._token: str | None = None
        self._lock_id: str | None = None
        self._owner: str | None = None
        self._ttl_s: float = default_ttl_s
        self._expires_mono: float = 0.0
        self._expires_wall: dt.datetime | None = None

    # -- internals (call with self._mutex held) --------------------------------------------

    def _clear(self) -> None:
        self._token = None
        self._lock_id = None
        self._owner = None
        self._expires_mono = 0.0
        self._expires_wall = None

    def _purge_expired(self) -> None:
        if self._token is None:
            return
        if self._clock() >= self._expires_mono and not self._worker_is_busy():
            self._clear()

    def _renew(self) -> None:
        self._expires_mono = self._clock() + self._ttl_s
        self._expires_wall = dt.datetime.now(dt.UTC) + dt.timedelta(seconds=self._ttl_s)

    def _state(self) -> LockState:
        if self._token is None:
            return LockState()
        assert self._expires_wall is not None
        return LockState(
            held=True,
            owner=self._owner,
            lock_id=self._lock_id,
            expires_at=self._expires_wall.isoformat(),
        )

    def _matches(self, token: str | None) -> bool:
        return (
            self._token is not None
            and token is not None
            and secrets.compare_digest(self._token, token)
        )

    def _decision(self, reason: LockReason) -> LockDecision:
        "pair a reason with the lease as it is right now, so the two cannot drift apart"
        return LockDecision(reason=reason, state=self._state())

    # -- public -----------------------------------------------------------------------------

    def state(self) -> LockState:
        with self._mutex:
            self._purge_expired()
            return self._state()

    def check(self, token: str | None, *, renew: bool = True) -> LockDecision:
        """
        Decide, in one atomic step, whether a request carrying `token` may drive the hardware.

        A hardware-mutating request needs the lease: no lease at all is a refusal
        (`not_held`), not a free pass. `renew` pushes the expiry out when the token is the
        holder's, which is why every request goes through here and not only the gated ones:
        a client that keeps talking keeps its lease.
        """
        with self._mutex:
            self._purge_expired()
            if self._token is None:
                return self._decision(LockReason.NOT_HELD)
            if not self._matches(token):
                return self._decision(LockReason.HELD_BY_OTHER)
            if renew:
                self._renew()
            return self._decision(LockReason.OK)

    def permits(self, token: str | None) -> bool:
        "True if a hardware-mutating request carrying `token` may proceed (no renewal)"
        return self.check(token, renew=False).permitted

    def acquire(
        self, owner: str, ttl_s: float | None = None, token: str | None = None
    ) -> Lease | LockDecision:
        """
        Take the lease. Returns the `Lease` on success, or the `LockDecision` that refused it
        when somebody else holds it - the decision, not a bare failure, because the caller
        needs to say who refused and asking again afterwards could name a different client.
        If `token` is the current holder's, the lease is renewed (and re-labelled/re-timed)
        instead of refused.
        """
        ttl = self._default_ttl_s if ttl_s is None else max(float(ttl_s), self.MIN_TTL_S)
        with self._mutex:
            self._purge_expired()
            if self._token is not None and not self._matches(token):
                return self._decision(LockReason.HELD_BY_OTHER)
            if self._token is None:
                self._token = secrets.token_urlsafe(32)
                self._lock_id = secrets.token_urlsafe(8)
            self._owner = owner
            self._ttl_s = ttl
            self._renew()
            assert self._lock_id is not None and self._expires_wall is not None
            return Lease(
                token=self._token,
                lock_id=self._lock_id,
                owner=owner,
                expires_at=self._expires_wall.isoformat(),
            )

    def release(self, token: str | None) -> LockDecision:
        """
        Give the lease back. The decision says what happened, against the lease as it was at
        that instant:

        - `ok`            it was yours and is now given back; `state` is the lease you held,
                          so a caller can log who it belonged to
        - `not_held`      nothing was held; releasing again is a no-op
        - `held_by_other` somebody else holds it, so it is not yours to give back
        """
        with self._mutex:
            self._purge_expired()
            if self._token is None:
                return self._decision(LockReason.NOT_HELD)
            if not self._matches(token):
                return self._decision(LockReason.HELD_BY_OTHER)
            released = self._decision(LockReason.OK)
            self._clear()
            return released

    def break_lock(self) -> LockState:
        "force-free the lease regardless of holder; returns the state that was broken"
        with self._mutex:
            self._purge_expired()
            previous = self._state()
            self._clear()
            return previous


