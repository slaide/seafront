"""Build the acquisition image store the machine config asks for."""

from __future__ import annotations

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
