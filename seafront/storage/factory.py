"""Build the acquisition image store the machine config asks for."""

from __future__ import annotations

import shutil
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


class InsufficientStorageError(Exception):
    "the configured storage cannot take the run being started"


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
    """
    base_dir = Path(ConfigRegistry.get(StorageConfig.BASE_IMAGE_OUTPUT_DIR).strvalue)
    backend = ConfigRegistry.get(StorageConfig.BACKEND).strvalue
    role = "image output directory" if backend == "disk" else "s3 spill directory"
    if not base_dir.is_dir():
        raise InsufficientStorageError(
            f"the {role} {base_dir} ({StorageConfig.BASE_IMAGE_OUTPUT_DIR.value}) does not exist"
        )
    if backend != "disk":
        return

    free = shutil.disk_usage(base_dir).free
    if free - STORAGE_RESERVE_BYTES < required_bytes:
        raise InsufficientStorageError(
            f"not enough space in {base_dir}: the run needs up to {_gb(required_bytes)} "
            f"and {_gb(free)} is free, of which {_gb(STORAGE_RESERVE_BYTES)} is kept in reserve"
        )


def _gb(num_bytes: int) -> str:
    return f"{num_bytes / 1024**3:.2f} GB"
