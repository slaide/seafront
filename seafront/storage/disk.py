"""Local directory sink: one file per item under a base directory."""

from __future__ import annotations

import threading
from pathlib import Path

from seafront.storage.base import StoreReport


class DiskSink:
    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir)
        self._lock = threading.Lock()
        self._bytes = 0
        self._stored = 0

    def deliver(self, relpath: str, data: bytes) -> None:
        path = self.base_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        with self._lock:
            self._bytes += len(data)
            self._stored += 1

    def describe(self, relpath: str) -> str:
        return str(self.base_dir / relpath)

    @property
    def bytes_written(self) -> int:
        with self._lock:
            return self._bytes

    def close(self) -> StoreReport:
        with self._lock:
            return StoreReport(stored=self._stored, bytes_written=self._bytes)
