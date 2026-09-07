"""
S3 sink: upload straight from memory, spill to disk only when uploads fall behind.

`upload_threads` defaults to 1 like the disk writer (tune it against the link; a local mock S3 is
not network bound and would only flatter the number). At most `spill_threshold` encoded images wait
in memory for the uploader; anything beyond that is written to `spill_dir` and uploaded from there,
in order, so memory stays bounded and the disk is touched only under backpressure. An upload that
keeps failing leaves (or puts) its file in `spill_dir`, so a network problem never loses an image;
it is reported instead.

Credentials come from the usual AWS chain (environment, shared credentials file, instance role),
never from seafront's config.
"""

from __future__ import annotations

import queue
import threading
import typing as tp
from pathlib import Path

from seafront.storage.base import StoreReport

ClientFactory = tp.Callable[[], tp.Any]


def default_client_factory(endpoint_url: str | None, region: str | None) -> ClientFactory:
    def make() -> tp.Any:
        import boto3
        from botocore.config import Config

        return boto3.client(
            "s3",
            endpoint_url=endpoint_url or None,
            region_name=region or None,
            config=Config(retries={"max_attempts": 5, "mode": "standard"}),
        )

    return make


class S3Sink:
    def __init__(
        self,
        bucket: str,
        *,
        spill_dir: Path | str,
        prefix: str = "",
        endpoint_url: str | None = None,
        region: str | None = None,
        spill_threshold: int = 16,
        upload_threads: int = 1,
        client_factory: ClientFactory | None = None,
    ) -> None:
        if not bucket:
            raise ValueError("S3 storage needs a bucket name")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.spill_dir = Path(spill_dir)
        self.spill_threshold = max(0, spill_threshold)
        self._client_factory = client_factory or default_client_factory(endpoint_url, region)
        self._queue: queue.Queue[tuple[str, bytes | Path] | None] = queue.Queue()
        self._lock = threading.Lock()
        self._in_memory = 0
        self._bytes = 0
        self._stored = 0
        self._spilled = 0
        self._failed: list[str] = []
        self._threads = [
            threading.Thread(target=self._uploader, name=f"s3-upload-{i}", daemon=True)
            for i in range(max(1, upload_threads))
        ]
        for t in self._threads:
            t.start()

    # -- producer side (writer threads) -------------------------------------------------------

    def key(self, relpath: str) -> str:
        return f"{self.prefix}/{relpath}" if self.prefix else relpath

    def describe(self, relpath: str) -> str:
        return f"s3://{self.bucket}/{self.key(relpath)}"

    def deliver(self, relpath: str, data: bytes) -> None:
        with self._lock:
            spill = self._in_memory >= self.spill_threshold
            if not spill:
                self._in_memory += 1
        if spill:
            path = self._spill(relpath, data)
            with self._lock:
                self._spilled += 1
            self._queue.put((relpath, path))
        else:
            self._queue.put((relpath, data))

    def _spill(self, relpath: str, data: bytes) -> Path:
        path = self.spill_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    # -- consumer side (upload threads) -------------------------------------------------------

    def _uploader(self) -> None:
        client = self._client_factory()
        while True:
            item = self._queue.get()
            if item is None:
                return
            relpath, payload = item
            try:
                data = payload if isinstance(payload, bytes) else payload.read_bytes()
                client.put_object(Bucket=self.bucket, Key=self.key(relpath), Body=data)
                with self._lock:
                    self._bytes += len(data)
                    self._stored += 1
                if isinstance(payload, Path):
                    payload.unlink(missing_ok=True)
            except Exception as e:
                # never lose the image: keep (or create) its copy in the spill directory
                kept = payload if isinstance(payload, Path) else self._spill(relpath, payload)
                with self._lock:
                    self._failed.append(f"{relpath}: {e} (kept at {kept})")
            finally:
                if isinstance(payload, bytes):
                    with self._lock:
                        self._in_memory -= 1

    @property
    def bytes_written(self) -> int:
        with self._lock:
            return self._bytes

    @property
    def queued(self) -> int:
        "items waiting for an uploader (memory plus spill)"
        return self._queue.qsize()

    def close(self) -> StoreReport:
        for _ in self._threads:
            self._queue.put(None)
        for t in self._threads:
            t.join()
        with self._lock:
            return StoreReport(
                stored=self._stored,
                failed=list(self._failed),
                bytes_written=self._bytes,
                spilled_to_disk=self._spilled,
            )
