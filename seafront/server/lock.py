"""
Client-held device lease (the "lock"), layered above the MicroscopeWorker.

The worker guarantees one hardware command at a time. The lease guarantees one *client* at a
time across requests: while it is held, hardware-mutating routes refuse (423) any request that
does not carry the lease token. Status reads never need it. Cancel is never gated either: it
exists to prevent hardware damage, so anyone may stop a run. Anyone may also break a lease,
which is logged with owner and reason.

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

from pydantic import BaseModel


class LockState(BaseModel):
    "public view of the device lease; safe to broadcast, carries no token"

    held: bool = False
    owner: str | None = None
    "free-text label the holder gave on acquire (who to talk to)"
    lock_id: str | None = None
    "public id of the current lease; a client compares it with the id it got on acquire"
    expires_at: str | None = None
    "ISO-8601 UTC. Renewed by every request carrying the token; never lapses while the worker is busy"


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

    # -- public -----------------------------------------------------------------------------

    def state(self) -> LockState:
        with self._mutex:
            self._purge_expired()
            return self._state()

    def matches(self, token: str | None) -> bool:
        "True if `token` is the current lease's token"
        with self._mutex:
            self._purge_expired()
            return self._matches(token)

    def allows(self, token: str | None) -> bool:
        "True if a hardware-mutating request carrying `token` may proceed"
        with self._mutex:
            self._purge_expired()
            return self._token is None or self._matches(token)

    def touch(self, token: str | None) -> bool:
        "renew the expiry if `token` is the current lease's; returns whether it was"
        with self._mutex:
            self._purge_expired()
            if not self._matches(token):
                return False
            self._renew()
            return True

    def acquire(
        self, owner: str, ttl_s: float | None = None, token: str | None = None
    ) -> Lease | None:
        """
        Take the lease. Returns None if someone else holds it. If `token` is the current
        holder's, the lease is renewed (and re-labelled/re-timed) instead of refused.
        """
        ttl = self._default_ttl_s if ttl_s is None else max(float(ttl_s), self.MIN_TTL_S)
        with self._mutex:
            self._purge_expired()
            if self._token is not None and not self._matches(token):
                return None
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

    def release(self, token: str | None) -> bool:
        """
        Give the lease back. True if released or not held (idempotent), False if held by
        someone else.
        """
        with self._mutex:
            self._purge_expired()
            if self._token is None:
                return True
            if not self._matches(token):
                return False
            self._clear()
            return True

    def break_lock(self) -> LockState:
        "force-free the lease regardless of holder; returns the state that was broken"
        with self._mutex:
            self._purge_expired()
            previous = self._state()
            self._clear()
            return previous


