"""
Image storage backends for acquisitions (built-in protocol and the API-driven script alike).

    store = image_store_from_config(spill_dir)          # server: machine config decides
    store = ImageStore(DiskSink(base_dir), writer_threads=1)
    store = ImageStore(S3Sink("bucket", spill_dir=..., prefix="lab/scope1"), writer_threads=1)
    store.put("project/plate_2026/A01_s1_x0_y0_z0_DAPI.tiff", lambda: encode_ome_tiff(img, ome_xml, "LZW"))
    report = store.close()
"""

from seafront.storage.base import ImageStore, Sink, StoreReport, encode_ome_tiff
from seafront.storage.disk import DiskSink
from seafront.storage.factory import image_store_from_config
from seafront.storage.s3 import S3Sink

__all__ = [
    "DiskSink",
    "ImageStore",
    "S3Sink",
    "Sink",
    "StoreReport",
    "encode_ome_tiff",
    "image_store_from_config",
]
