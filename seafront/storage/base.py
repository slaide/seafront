"""
Image storage for acquisitions: encode OME-TIFFs on a small thread pool and hand the bytes to a
sink (local disk or S3).

The acquiring thread never blocks on storage: `put()` only submits work. `writer_threads`
defaults to 1 on purpose: the reference deployment stores on an HDD server without RAID, where
parallel writers only contend for the same disk. Operators can raise it through machine config
(`storage.writer_threads`) or the API-driven script's `--store-workers` when the target is an SSD,
a RAID or a fast network backend.

Failures are collected, never swallowed: `close()` returns a report and the acquisition surfaces
it (an image that failed to store makes the run end with an error, not a silent gap).
"""

from __future__ import annotations

import concurrent.futures
import io
import threading
import typing as tp

import numpy as np
import tifffile
from pydantic import BaseModel, Field


class StorageFailedError(Exception):
    "images handed to the store were lost"

    def __init__(self, lost: list[str]):
        self.lost = lost
        shown = "; ".join(lost[:3])
        more = f"; and {len(lost) - 3} more" if len(lost) > 3 else ""
        super().__init__(f"{len(lost)} image(s) could not be stored: {shown}{more}")


class StoreReport(BaseModel):
    "what happened to everything handed to an ImageStore"

    stored: int = 0
    failed: list[str] = Field(default_factory=list)
    "one entry per lost or failed item: '<relpath>: <reason>'"
    bytes_written: int = 0
    spilled_to_disk: int = 0
    "S3 only: items that went through the local spill directory because uploads lagged"


class Sink(tp.Protocol):
    """where encoded images end up; deliver() is called from writer threads and may raise"""

    def deliver(self, relpath: str, data: bytes) -> None: ...

    def describe(self, relpath: str) -> str:
        "human-readable location of an item (path or s3:// URL)"
        ...

    def lost(self) -> list[str]:
        """
        items gone for good, found after deliver() returned (an upload that failed, and whose
        local copy could not be written either); '<relpath>: <reason>'
        """
        ...

    @property
    def bytes_written(self) -> int: ...

    def close(self) -> StoreReport:
        "wait for in-flight deliveries, release resources, report"
        ...


# Strip layout of every image seafront writes. TIFF strips are baseline and every reader we
# target iterates them (ImageJ's native reader, Bio-Formats, tifffile); tiles are not baseline and
# ImageJ's native reader cannot read them. 64-row strips let tifffile compress CODEC_THREADS strips
# at once, which is a ~3.5x faster encode for the same bytes on disk and the same file size. Codec
# threads compress in memory; the file is still written as one sequential stream, so this does not
# multiply concurrent disk writes the way storage.writer_threads does.
STRIP_ROWS = 64
CODEC_THREADS = 4


def encode_ome_tiff(
    image: np.ndarray,
    ome_xml: str,
    compression: str,
    strip_rows: int = STRIP_ROWS,
    codec_threads: int = CODEC_THREADS,
) -> bytes:
    """
    the exact bytes `tifffile.imwrite` puts on disk for this image and OME-XML description

    strip_rows <= 0 means tifffile's default strip size (256 KB strips).
    """
    buf = io.BytesIO()
    tifffile.imwrite(
        buf,
        image,
        compression=compression,
        compressionargs={},
        rowsperstrip=strip_rows if strip_rows > 0 else None,
        maxworkers=max(1, codec_threads),
        description=ome_xml,
        photometric="minisblack",
    )
    return buf.getvalue()


class ImageStore:
    def __init__(self, sink: Sink, writer_threads: int = 1) -> None:
        self.sink = sink
        self.writer_threads = max(1, writer_threads)
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.writer_threads, thread_name_prefix="image-writer"
        )
        self._lock = threading.Lock()
        self._pending = 0
        self._failures: list[str] = []
        self._report: StoreReport | None = None

    def put(self, relpath: str, encode: tp.Callable[[], bytes]) -> None:
        "encode on a writer thread, then deliver; returns immediately"
        self._submit(relpath, encode)

    def put_bytes(self, relpath: str, data: bytes) -> None:
        "deliver already-encoded bytes (protocol copies, companion files, ...)"
        self._submit(relpath, lambda: data)

    def _submit(self, relpath: str, encode: tp.Callable[[], bytes]) -> None:
        if self._report is not None:
            raise RuntimeError("image store is closed")
        with self._lock:
            self._pending += 1
        self._pool.submit(self._work, relpath, encode)

    def _work(self, relpath: str, encode: tp.Callable[[], bytes]) -> None:
        try:
            self.sink.deliver(relpath, encode())
        except Exception as e:
            with self._lock:
                self._failures.append(f"{relpath}: {e}")
        finally:
            with self._lock:
                self._pending -= 1

    def raise_if_failed(self) -> None:
        """
        Raise StorageFailedError if anything handed to this store so far is lost.

        For a caller still producing: close() reports the same failures, but only at the end,
        and a run that goes on after its storage has failed - a full disk stays full - spends
        hours taking images that go nowhere.
        """
        with self._lock:
            lost = list(self._failures)
        lost += self.sink.lost()
        if lost:
            raise StorageFailedError(lost)

    @property
    def pending(self) -> int:
        "items submitted but not yet delivered to the sink (for backpressure / status)"
        with self._lock:
            return self._pending

    @property
    def bytes_written(self) -> int:
        return self.sink.bytes_written

    def describe(self, relpath: str) -> str:
        return self.sink.describe(relpath)

    def close(self) -> StoreReport:
        "wait for everything, close the sink, report; safe to call more than once"
        if self._report is not None:
            return self._report
        self._pool.shutdown(wait=True)
        report = self.sink.close()
        with self._lock:
            report.failed = self._failures + report.failed
        self._report = report
        return report
