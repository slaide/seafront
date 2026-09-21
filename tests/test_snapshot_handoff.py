"""The boundary between a successful snapshot response and retrieval of its pixels.

`snap_channel` returns JSON while the image itself is fetched separately from the latest-image
WebSocket.  These tests characterize what is reliably bound together, what survives reconnects,
and where the public contract currently has no identity with which to name a particular frame.
"""

import json
import re
import typing as tp

import numpy as np
import pytest
from websockets.sync.client import connect as ws_connect

from tests.mock_server import MOCK, mock_server
from tests.test_api_operations import make_request

SERVER = ""


@pytest.fixture(scope="module", autouse=True)
def server(tmp_path_factory):
    global SERVER
    with mock_server(tmp_path_factory) as running:
        SERVER = running.url
        yield running


def post(path: str, data: dict[str, tp.Any] | None = None, token: str | None = None):
    return make_request(f"{SERVER}{path}", data=data or {}, token=token)


def ok(path: str, data: dict[str, tp.Any] | None = None, token: str | None = None):
    status, body = post(path, data, token)
    assert status == 200, (path, status, body)
    assert body is not None
    return body


@pytest.fixture(scope="module")
def channels(server) -> list[dict[str, tp.Any]]:
    protocol = ok("/api/acquisition/config_fetch", {"config_file": "default.json"})["file"]
    assert len(protocol["channels"]) >= 2
    return protocol["channels"][:2]


@pytest.fixture
def token() -> tp.Iterator[str]:
    lease = ok("/api/lock/acquire", {"owner": "snapshot handoff tests"})
    try:
        yield lease["token"]
    finally:
        post("/api/lock/release", {"lock_token": lease["token"]})


def snap(channel: dict[str, tp.Any], token: str) -> dict[str, tp.Any]:
    return ok("/api/action/snap_channel", {"channel": channel}, token)


def fetch(
    channel_handle: str,
    image_id: str,
    *,
    factor: int = 1,
    target_device: str = MOCK,
) -> tuple[dict[str, tp.Any], np.ndarray | None]:
    ws_url = re.sub(r"^http", "ws", SERVER) + "/ws/get_info/acquired_image"
    with ws_connect(ws_url, max_size=None, open_timeout=10) as ws:
        ws.send(json.dumps({"target_device": target_device}))
        ws.send(channel_handle)
        ws.send(image_id)
        ws.send(str(factor))
        metadata = json.loads(ws.recv(timeout=10))
        if not metadata or metadata.get("error"):
            return metadata, None
        raw = ws.recv(timeout=10)
        assert isinstance(raw, bytes)
        dtype = {8: np.uint8, 16: np.uint16}[metadata["bit_depth"]]
        image = np.frombuffer(raw, dtype=dtype).reshape(metadata["height"], metadata["width"])
        return metadata, image.copy()


class TestBasicPixelHandoff:
    def test_a_channel_with_no_image_says_so(self):
        metadata, image = fetch("channel-that-does-not-exist", "unknown-image")
        assert metadata == {}
        assert image is None

    def test_a_successful_snapshot_is_retrievable(self, channels, token):
        channel = channels[0]
        response = snap(channel, token)
        metadata, image = fetch(channel["handle"], response["image_id"])
        assert image is not None
        assert metadata["channel_handle"] == channel["handle"]
        assert image.shape == (metadata["height"], metadata["width"])
        assert image.dtype.itemsize * 8 == metadata["bit_depth"]

    def test_repeated_reads_return_the_same_cached_frame(self, channels, token):
        channel = channels[0]
        response = snap(channel, token)
        _, first = fetch(channel["handle"], response["image_id"])
        _, second = fetch(channel["handle"], response["image_id"])
        assert first is not None and second is not None
        assert np.array_equal(first, second)

    def test_downsampling_is_a_slice_of_the_same_frame(self, channels, token):
        channel = channels[0]
        response = snap(channel, token)
        _, full = fetch(channel["handle"], response["image_id"])
        metadata, reduced = fetch(channel["handle"], response["image_id"], factor=4)
        assert full is not None and reduced is not None
        assert reduced.shape == (metadata["height"], metadata["width"])
        assert np.array_equal(reduced, full[::4, ::4])

    def test_reconnecting_does_not_lose_the_latest_frame(self, channels, token):
        channel = channels[0]
        response = snap(channel, token)
        _, before = fetch(channel["handle"], response["image_id"])
        # fetch() uses a fresh WebSocket each time, so this is a close/reconnect boundary.
        _, after = fetch(channel["handle"], response["image_id"])
        assert before is not None and after is not None
        assert np.array_equal(before, after)


class TestLatestImageSlotSemantics:
    def test_a_second_snapshot_replaces_the_first_for_that_channel(self, channels, token):
        channel = channels[0]
        first_response = snap(channel, token)
        _, first = fetch(channel["handle"], first_response["image_id"])
        second_response = snap(channel, token)
        _, second = fetch(channel["handle"], second_response["image_id"])
        assert first is not None and second is not None
        assert not np.array_equal(first, second)

    def test_another_channel_does_not_replace_this_channels_image(self, channels, token):
        first_channel, second_channel = channels
        response = snap(first_channel, token)
        _, before = fetch(first_channel["handle"], response["image_id"])
        snap(second_channel, token)
        _, after = fetch(first_channel["handle"], response["image_id"])
        assert before is not None and after is not None
        assert np.array_equal(before, after)

    def test_a_failed_snapshot_leaves_the_previous_image_in_the_slot(self, channels, token):
        channel = channels[0]
        response = snap(channel, token)
        _, before = fetch(channel["handle"], response["image_id"])
        invalid = dict(channel, exposure_time_ms=-1.0)
        status, _body = post("/api/action/snap_channel", {"channel": invalid}, token)
        assert status >= 400
        _, after = fetch(channel["handle"], response["image_id"])
        assert before is not None and after is not None
        assert np.array_equal(before, after)

    def test_pixels_are_readable_without_the_device_lease(self, channels, token):
        channel = channels[0]
        response = snap(channel, token)
        metadata, image = fetch(channel["handle"], response["image_id"])
        assert metadata["channel_handle"] == channel["handle"]
        assert image is not None


class TestFrameIdentityContract:
    def test_snapshot_response_names_the_frame_it_created(self, channels, token):
        response = snap(channels[0], token)
        assert response.get("image_id"), response

    def test_pixel_metadata_names_the_frame_being_returned(self, channels, token):
        channel = channels[0]
        response = snap(channel, token)
        metadata, _image = fetch(channel["handle"], response["image_id"])
        assert metadata.get("image_id"), metadata

    def test_snapshot_response_and_pixels_carry_the_same_identity(self, channels, token):
        channel = channels[0]
        response = snap(channel, token)
        metadata, _image = fetch(channel["handle"], response["image_id"])
        assert response.get("image_id") == metadata.get("image_id")
        assert response.get("image_id") is not None

    def test_replaced_frame_is_an_explicit_error(self, channels, token):
        channel = channels[0]
        replaced = snap(channel, token)
        current = snap(channel, token)

        metadata, image = fetch(channel["handle"], replaced["image_id"])

        assert image is None
        assert metadata["error"] == "image_replaced"
        assert metadata["channel_handle"] == channel["handle"]
        assert current["image_id"] != replaced["image_id"]
