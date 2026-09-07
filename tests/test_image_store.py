"""
seafront.storage: encode on writer threads, deliver to a sink. The acquiring thread never blocks,
failures are reported not swallowed, and the S3 sink spills to disk only when uploads lag and never
loses an image on a failed upload.
"""

import threading
import time

import numpy as np
import pytest
import tifffile

from seafront.storage import DiskSink, ImageStore, S3Sink, encode_ome_tiff


def _img(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 4000, size=(64, 64), dtype=np.uint16)


def _encode(seed: int = 0):
    return lambda: encode_ome_tiff(_img(seed), "<OME/>", "LZW")


class TestEncode:
    def test_roundtrip_keeps_pixels_and_description(self):
        data = encode_ome_tiff(_img(1), "<OME>x</OME>", "LZW")
        with tifffile.TiffFile(__import__("io").BytesIO(data)) as t:
            assert np.array_equal(t.asarray(), _img(1))
            assert t.pages[0].description == "<OME>x</OME>"

    def test_bytes_equal_imwrite_to_disk(self, tmp_path):
        data = encode_ome_tiff(_img(2), "<OME/>", "LZW")
        tifffile.imwrite(tmp_path / "a.tiff", _img(2), compression="LZW", compressionargs={}, maxworkers=1, description="<OME/>", photometric="minisblack")
        assert (tmp_path / "a.tiff").read_bytes() == data


class TestDiskStore:
    def test_files_land_under_base_with_report(self, tmp_path):
        store = ImageStore(DiskSink(tmp_path), writer_threads=1)
        for i in range(5):
            store.put(f"proj/plate_1/t0/img{i}.tiff", _encode(i))
        store.put_bytes("proj/plate_1/config.json", b"{}")
        report = store.close()
        files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
        assert files == ["proj/plate_1/config.json"] + [f"proj/plate_1/t0/img{i}.tiff" for i in range(5)]
        assert report.stored == 6 and report.failed == []
        assert report.bytes_written == sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file())
        assert store.describe("proj/x.tiff") == str(tmp_path / "proj/x.tiff")

    def test_encode_failure_is_reported_and_does_not_stop_others(self, tmp_path):
        store = ImageStore(DiskSink(tmp_path), writer_threads=2)

        def boom():
            raise ValueError("codec exploded")

        store.put("bad.tiff", boom)
        store.put("good.tiff", _encode())
        report = store.close()
        assert report.stored == 1
        assert report.failed == ["bad.tiff: codec exploded"]
        assert (tmp_path / "good.tiff").exists() and not (tmp_path / "bad.tiff").exists()

    def test_close_is_idempotent_and_put_after_close_is_an_error(self, tmp_path):
        store = ImageStore(DiskSink(tmp_path))
        store.put("a.tiff", _encode())
        assert store.close() is store.close()
        with pytest.raises(RuntimeError):
            store.put("b.tiff", _encode())

    def test_put_never_blocks_the_caller(self, tmp_path):
        gate = threading.Event()

        def slow():
            gate.wait(5)
            return b"x"

        store = ImageStore(DiskSink(tmp_path), writer_threads=1)
        t = time.perf_counter()
        for i in range(20):
            store.put(f"{i}.bin", slow)
        assert time.perf_counter() - t < 0.5
        assert store.pending == 20
        gate.set()
        assert store.close().stored == 20


class FakeS3:
    """duck-typed boto3 client: records puts, can hold uploads back or fail specific keys"""

    def __init__(self, fail_keys: set[str] | None = None):
        self.puts: list[tuple[str, int]] = []
        self.gate = threading.Event()
        self.gate.set()
        self.fail_keys = fail_keys or set()
        self.lock = threading.Lock()

    def put_object(self, Bucket: str, Key: str, Body: bytes) -> None:
        self.gate.wait(10)
        if Key in self.fail_keys:
            raise OSError(f"simulated upload failure for {Key}")
        with self.lock:
            self.puts.append((Key, len(Body)))


class TestS3Store:
    def test_uploads_in_order_with_prefix_and_describe(self, tmp_path):
        fake = FakeS3()
        sink = S3Sink("bucket", spill_dir=tmp_path, prefix="/lab/scope1/", client_factory=lambda: fake)
        store = ImageStore(sink, writer_threads=1)
        for i in range(4):
            store.put(f"proj/p_1/img{i}.tiff", _encode(i))
        report = store.close()
        assert [k for k, _ in fake.puts] == [f"lab/scope1/proj/p_1/img{i}.tiff" for i in range(4)]
        assert report.stored == 4 and report.failed == [] and report.spilled_to_disk == 0
        assert report.bytes_written == sum(n for _, n in fake.puts)
        assert store.describe("proj/p_1/img0.tiff") == "s3://bucket/lab/scope1/proj/p_1/img0.tiff"
        assert not any(tmp_path.rglob("*.tiff")), "nothing spilled when uploads keep up"

    def test_spills_to_disk_only_beyond_threshold_and_cleans_up(self, tmp_path):
        fake = FakeS3()
        fake.gate.clear()  # uploader stuck: simulate a slow link
        sink = S3Sink("bucket", spill_dir=tmp_path, spill_threshold=3, client_factory=lambda: fake)
        store = ImageStore(sink, writer_threads=1)
        for i in range(10):
            store.put(f"img{i}.tiff", _encode(i))
        deadline = time.time() + 5
        while store.pending and time.time() < deadline:
            time.sleep(0.01)
        spilled = sorted(p.name for p in tmp_path.glob("*.tiff"))
        assert spilled == [f"img{i}.tiff" for i in range(3, 10)], "the first 3 stay in memory, the rest spill"
        assert sink.queued in (9, 10), "everything waits; the uploader may hold one item in flight"
        fake.gate.set()
        report = store.close()
        assert [k for k, _ in fake.puts] == [f"img{i}.tiff" for i in range(10)], "order preserved across memory and spill"
        assert report.stored == 10 and report.spilled_to_disk == 7 and report.failed == []
        assert not list(tmp_path.glob("*.tiff")), "spill files removed after upload"

    def test_failed_upload_keeps_the_image_on_disk_and_is_reported(self, tmp_path):
        fake = FakeS3(fail_keys={"img1.tiff"})
        sink = S3Sink("bucket", spill_dir=tmp_path, client_factory=lambda: fake)
        store = ImageStore(sink, writer_threads=1)
        for i in range(3):
            store.put(f"img{i}.tiff", _encode(i))
        report = store.close()
        assert report.stored == 2
        assert len(report.failed) == 1 and report.failed[0].startswith("img1.tiff: simulated upload failure")
        assert (tmp_path / "img1.tiff").exists(), "the failed image is kept locally"
        assert str(tmp_path / "img1.tiff") in report.failed[0]

    def test_needs_a_bucket(self, tmp_path):
        with pytest.raises(ValueError):
            S3Sink("", spill_dir=tmp_path, client_factory=lambda: FakeS3())


class TestFactory:
    def test_disk_and_s3_from_registry(self, tmp_path, monkeypatch):
        from seafront.config.handles import StorageConfig
        from seafront.config.registry import ConfigRegistry
        from seafront.storage import image_store_from_config
        from seafront.storage.factory import S3Sink as FactoryS3Sink

        ConfigRegistry.reset()
        ConfigRegistry.init({})
        from seafront.config.core_config import register_core_config

        register_core_config(str(tmp_path), [], [])
        try:
            store = image_store_from_config()
            assert isinstance(store.sink, DiskSink) and store.writer_threads == 1
            store.close()
            ConfigRegistry.set_value(StorageConfig.WRITER_THREADS.value, 3)
            ConfigRegistry.set_value(StorageConfig.BACKEND.value, "s3")
            ConfigRegistry.set_value(StorageConfig.S3_BUCKET.value, "b")
            ConfigRegistry.set_value(StorageConfig.S3_PREFIX.value, "p")
            # don't create a real boto3 client in a unit test
            monkeypatch.setattr("seafront.storage.s3.default_client_factory", lambda *_: (lambda: FakeS3()))
            store = image_store_from_config()
            assert isinstance(store.sink, FactoryS3Sink) and store.writer_threads == 3
            assert store.describe("x") == "s3://b/p/x"
            store.close()
            ConfigRegistry.set_value(StorageConfig.S3_BUCKET.value, "")
            with pytest.raises(ValueError):
                image_store_from_config()
        finally:
            ConfigRegistry.reset()


class TestS3AgainstMinio:
    def test_real_client_uploads_and_reports(self, minio_s3, tmp_path):
        prefix = f"unit-{int(time.time())}"
        sink = S3Sink(minio_s3["bucket"], spill_dir=tmp_path, prefix=prefix, endpoint_url=minio_s3["endpoint_url"])
        store = ImageStore(sink, writer_threads=1)
        payloads = {f"p/plate_1/img{i}.tiff": encode_ome_tiff(_img(i), "<OME/>", "LZW") for i in range(4)}
        for rel, data in payloads.items():
            store.put_bytes(rel, data)
        store.put_bytes("p/plate_1/config.json", b"{}")
        report = store.close()
        assert report.failed == [] and report.stored == 5 and report.spilled_to_disk == 0
        listed = minio_s3["client"].list_objects_v2(Bucket=minio_s3["bucket"], Prefix=prefix + "/")
        objects = {o["Key"].removeprefix(prefix + "/"): o["Size"] for o in listed.get("Contents", [])}
        assert objects == {**{k: len(v) for k, v in payloads.items()}, "p/plate_1/config.json": 2}
        body = minio_s3["client"].get_object(Bucket=minio_s3["bucket"], Key=f"{prefix}/p/plate_1/img2.tiff")["Body"].read()
        assert body == payloads["p/plate_1/img2.tiff"]
        assert not list(tmp_path.rglob("*")), "nothing spilled"
