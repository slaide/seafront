"""Build the acquisition image store the machine config asks for."""

from __future__ import annotations

import shutil
import threading
from concurrent.futures import Future, wait
from pathlib import Path

from seafront.config.handles import StorageConfig
from seafront.config.registry import ConfigRegistry
from seafront.storage.base import ImageStore
from seafront.storage.disk import DiskSink
from seafront.storage.s3 import S3Sink


def image_store_from_config() -> ImageStore:
    """
    disk: files under `storage.base_image_output_dir`.
    s3:   direct uploads to `storage.s3.bucket`, with `storage.base_image_output_dir` as the spill
          directory for images the uploader could not take yet (or failed to upload).
    """
    base_dir = Path(ConfigRegistry.get(StorageConfig.BASE_IMAGE_OUTPUT_DIR).strvalue)
    writer_threads = ConfigRegistry.get(StorageConfig.WRITER_THREADS).intvalue
    backend = ConfigRegistry.get(StorageConfig.BACKEND).strvalue
    if backend == "disk":
        return ImageStore(DiskSink(base_dir), writer_threads=writer_threads)
    if backend == "s3":
        sink = S3Sink(
            ConfigRegistry.get(StorageConfig.S3_BUCKET).strvalue,
            spill_dir=base_dir,
            prefix=ConfigRegistry.get(StorageConfig.S3_PREFIX).strvalue,
            endpoint_url=ConfigRegistry.get(StorageConfig.S3_ENDPOINT_URL).strvalue or None,
            region=ConfigRegistry.get(StorageConfig.S3_REGION).strvalue or None,
            spill_threshold=ConfigRegistry.get(StorageConfig.S3_SPILL_THRESHOLD).intvalue,
            upload_threads=ConfigRegistry.get(StorageConfig.S3_UPLOAD_THREADS).intvalue,
        )
        return ImageStore(sink, writer_threads=writer_threads)
    raise ValueError(f"unknown storage backend {backend!r}")


STORAGE_RESERVE_BYTES = 1 << 30
"""
space left free on the disk a run writes to, beyond what the run needs: a disk filled to the
last byte fails whatever else writes to it too - the log, the config, the OS
"""


STORAGE_PROBE_TIMEOUT_S = 5.0
"""
how long the storage check waits for the output directory to answer. A hard-mounted network
share whose server is gone does not fail a stat, it blocks it for as long as the server stays
away; past this, the run is refused instead.
"""

_SHARE_HINT = (
    "If it is on a network share, check that the share is mounted and its server is reachable"
)


class InsufficientStorageError(Exception):
    "the configured storage cannot take the run being started"


_probes: dict[tuple[Path, bool], Future[tuple[bool, int]]] = {}
_probes_lock = threading.Lock()


def _probe(base_dir: Path, measure_free: bool) -> tuple[bool, int] | None:
    """
    (whether `base_dir` is a directory, its free bytes if `measure_free` and it is one, else 0),
    or None when it did not answer within STORAGE_PROBE_TIMEOUT_S.

    The stat runs on a thread of its own. One stuck on a hung share cannot be interrupted, so it
    is left behind and finishes whenever the share answers. Until then, later checks of the same
    directory wait on that probe instead of starting another, so retrying a start does not leave
    one more thread stuck in the kernel each time.
    """
    key = (base_dir, measure_free)
    with _probes_lock:
        probe = _probes.get(key)
        if probe is None or probe.done():
            probe = _probes[key] = Future()
            threading.Thread(
                target=_look,
                args=(probe, base_dir, measure_free),
                name=f"storage probe {base_dir}",
                daemon=True,
            ).start()
    done, _ = wait([probe], timeout=STORAGE_PROBE_TIMEOUT_S)
    return probe.result() if done else None


def _look(probe: Future[tuple[bool, int]], base_dir: Path, measure_free: bool) -> None:
    try:
        is_dir = base_dir.is_dir()
        probe.set_result((is_dir, shutil.disk_usage(base_dir).free if is_dir and measure_free else 0))
    except Exception as e:
        probe.set_exception(e)


def check_capacity(required_bytes: int) -> None:
    """
    Refuse, before it starts, a run the configured storage cannot take.

    `required_bytes` is the run's upper bound: uncompressed, every image. Compression only
    makes the real number smaller, so a run that passes this does not run out of space on
    account of its own images.

    disk: the output directory must exist and have `required_bytes` free beyond the reserve.
    s3:   the spill directory must exist. The bucket's capacity is not something S3 reports,
          and the spill directory holds only what the uploader has not caught up with, so
          neither is checked against the run's size; a failed upload stops the run instead
          (see ImageStore.raise_if_failed).

    A directory that cannot even be looked at is refused too. A network share that is linked
    but not mounted, or whose server is gone, fails stat with errors pathlib does not treat as
    "missing" (errno 521 from NFS, for one), or blocks it outright; neither may leave here as
    an OSError that names nothing but the path, or as a request that never returns.

    Blocks for up to STORAGE_PROBE_TIMEOUT_S, so async callers run it off the event loop.
    """
    base_dir = Path(ConfigRegistry.get(StorageConfig.BASE_IMAGE_OUTPUT_DIR).strvalue)
    backend = ConfigRegistry.get(StorageConfig.BACKEND).strvalue
    role = "image output directory" if backend == "disk" else "s3 spill directory"
    where = f"the {role} {base_dir} ({StorageConfig.BASE_IMAGE_OUTPUT_DIR.value})"
    try:
        answer = _probe(base_dir, measure_free=backend == "disk")
    except OSError as e:
        reason = f"{e.strerror} (errno {e.errno})" if e.strerror else str(e)
        raise InsufficientStorageError(f"{where} cannot be reached: {reason}. {_SHARE_HINT}") from e
    if answer is None:
        raise InsufficientStorageError(
            f"{where} did not answer within {STORAGE_PROBE_TIMEOUT_S:g} s. {_SHARE_HINT}"
        )
    is_dir, free = answer
    if not is_dir:
        raise InsufficientStorageError(f"{where} does not exist")
    if backend != "disk":
        return

    if free - STORAGE_RESERVE_BYTES < required_bytes:
        raise InsufficientStorageError(
            f"not enough space in {base_dir}: the run needs up to {_gb(required_bytes)} "
            f"and {_gb(free)} is free, of which {_gb(STORAGE_RESERVE_BYTES)} is kept in reserve"
        )


def _gb(num_bytes: int) -> str:
    return f"{num_bytes / 1024**3:.2f} GB"
