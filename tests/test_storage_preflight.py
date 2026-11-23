"""
Room for the images, before and during a run.

- a run the disk cannot take is refused at the front door, with the numbers
- a refused run leaves nothing on disk: the store is opened when the run starts
- a run whose storage fails partway stops there, instead of taking images that go nowhere
- an S3 upload that fails and cannot keep a local copy is reported as lost, and the uploader
  carries on with the rest of the queue
"""

import asyncio
import collections
import os
import sys
import time
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).parent))

from test_acquisition_edges import protocol

from seafront.server.commands import AcquisitionStatusStage
from seafront.storage import (
    DiskSink,
    ImageStore,
    InsufficientStorageError,
    S3Sink,
    StorageFailedError,
    check_capacity,
)
from seafront.storage import factory

DiskUsage = collections.namedtuple("DiskUsage", "total used free")


def with_free_space(monkeypatch, free_bytes: int) -> None:
    monkeypatch.setattr(
        factory.shutil, "disk_usage", lambda _path: DiskUsage(10 * free_bytes, 0, free_bytes)
    )


# --- the check -----------------------------------------------------------------------------


def test_enough_space_passes(mock_core, monkeypatch):
    with_free_space(monkeypatch, 100 * 1024**3)
    check_capacity(10 * 1024**3)


def test_too_little_space_is_refused_with_the_numbers(mock_core, monkeypatch):
    with_free_space(monkeypatch, 2 * 1024**3)
    with pytest.raises(InsufficientStorageError) as caught:
        check_capacity(5 * 1024**3)
    message = str(caught.value)
    assert "5.00 GB" in message and "2.00 GB" in message


def test_the_reserve_is_kept_free(mock_core, monkeypatch):
    "a run that would fill the disk to its last byte is refused too"
    with_free_space(monkeypatch, 5 * 1024**3)
    with pytest.raises(InsufficientStorageError):
        check_capacity(5 * 1024**3 - 1)


def test_a_missing_output_directory_is_refused(mock_core, mock_microscope_env):
    (mock_microscope_env / "images").rmdir()
    with pytest.raises(InsufficientStorageError, match="does not exist"):
        check_capacity(1)


def unreachable(monkeypatch, directory: Path) -> None:
    """
    `directory` answers like a network share that is linked but not mounted, or whose server is
    gone: stat fails with errno 521, which the kernel's NFS code lets out to userspace and
    which pathlib's is_dir() does not swallow (only ENOENT, ENOTDIR, EBADF and ELOOP).
    """
    real_stat, real_statvfs = os.stat, os.statvfs

    def check(path) -> None:
        if Path(path) == directory or directory in Path(path).parents:
            raise OSError(521, os.strerror(521), str(path))

    def stat(path, *args, **kwargs):
        check(path)
        return real_stat(path, *args, **kwargs)

    def statvfs(path):
        check(path)
        return real_statvfs(path)

    monkeypatch.setattr(os, "stat", stat)
    monkeypatch.setattr(os, "statvfs", statvfs)


def test_an_unreachable_output_directory_is_refused_with_where_and_why(
    mock_core, mock_microscope_env, monkeypatch
):
    images = mock_microscope_env / "images"
    unreachable(monkeypatch, images)
    with pytest.raises(InsufficientStorageError) as caught:
        check_capacity(1)
    message = str(caught.value)
    assert str(images) in message
    assert "storage.base_image_output_dir" in message
    assert "errno 521" in message
    assert "mounted" in message


def test_an_unreachable_output_directory_refuses_the_run_as_an_http_error(
    mock_core, mock_microscope_env, monkeypatch
):
    "an HTTPException is what the API turns into a JSON answer the dashboard can show"
    unreachable(monkeypatch, mock_microscope_env / "images")
    with pytest.raises(HTTPException) as caught:
        asyncio.run(mock_core.start_acquisition(config_file=protocol()))
    assert "cannot start the acquisition" in str(caught.value.detail)
    assert "errno 521" in str(caught.value.detail)
    assert mock_core.acquisitions._statuses == {}


# --- the front door ------------------------------------------------------------------------


def test_a_run_the_disk_cannot_take_is_refused_and_not_kept(mock_core, monkeypatch):
    with_free_space(monkeypatch, 1024**3)  # all of it reserve
    with pytest.raises(HTTPException) as caught:
        asyncio.run(mock_core.start_acquisition(config_file=protocol()))

    assert "not enough space" in str(caught.value.detail)
    assert mock_core.acquisitions._statuses == {}


def test_a_refused_run_writes_nothing(mock_core, mock_microscope_env):
    from seafront.config.handles import ProtocolConfig
    from seafront.config.registry import ConfigRegistry

    ConfigRegistry.set_value(
        ProtocolConfig.FORBIDDEN_AREAS,
        [{"name": "everything", "min_x_mm": -1e6, "max_x_mm": 1e6,
          "min_y_mm": -1e6, "max_y_mm": 1e6, "reason": "test"}],
    )
    with pytest.raises(HTTPException):
        asyncio.run(mock_core.start_acquisition(config_file=protocol()))

    assert list((mock_microscope_env / "images").iterdir()) == []


# --- during a run --------------------------------------------------------------------------


def test_a_run_whose_storage_fails_stops_early_and_says_why(mock_core, monkeypatch):
    def disk_full(self, relpath, data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(DiskSink, "deliver", disk_full)
    config = protocol(wells=tuple((1, c) for c in range(0, 12)))
    started = asyncio.run(mock_core.start_acquisition(config_file=config))
    assert mock_core.acquisition_future is not None
    mock_core.acquisition_future.result(timeout=60)

    status = asyncio.run(mock_core.get_acquisition_status(started.acquisition_id))
    assert status.acquisition_status == AcquisitionStatusStage.CRASHED
    assert "image storage failed" in status.message
    assert "No space left on device" in status.message
    progress = status.acquisition_progress
    assert progress.current_num_images < status.acquisition_meta_information.total_num_images


def test_raise_if_failed_reports_what_was_lost(tmp_path):
    class Failing(DiskSink):
        def deliver(self, relpath, data):
            raise OSError(28, "No space left on device")

    store = ImageStore(Failing(tmp_path))
    store.put_bytes("a.tiff", b"x")
    deadline = time.monotonic() + 5
    while store.pending and time.monotonic() < deadline:
        time.sleep(0.01)

    with pytest.raises(StorageFailedError, match="a.tiff: .*No space left"):
        store.raise_if_failed()
    store.close()


class FailingS3:
    def put_object(self, Bucket, Key, Body):
        raise OSError(f"simulated upload failure for {Key}")


def test_an_upload_that_cannot_keep_a_copy_is_lost_and_the_uploader_goes_on(tmp_path):
    # the spill directory is a file, so keeping a local copy fails too
    spill = tmp_path / "not-a-directory"
    spill.write_text("")
    sink = S3Sink("bucket", spill_dir=spill, client_factory=FailingS3)
    store = ImageStore(sink)
    store.put_bytes("one.tiff", b"1")
    store.put_bytes("two.tiff", b"2")
    report = store.close()

    # both reported: the first failure did not take the uploader down with it
    assert len(report.failed) == 2, report.failed
    assert all("keeping a local copy failed" in entry for entry in report.failed)
    assert [entry.split(":")[0] for entry in sink.lost()] == ["one.tiff", "two.tiff"]


def test_an_upload_kept_locally_is_not_lost(tmp_path):
    sink = S3Sink("bucket", spill_dir=tmp_path, client_factory=FailingS3)
    store = ImageStore(sink)
    store.put_bytes("one.tiff", b"1")
    report = store.close()

    assert sink.lost() == []
    assert len(report.failed) == 1 and "kept at" in report.failed[0]
