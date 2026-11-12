"""The boundary between a successful snapshot response and retrieval of its pixels.

`snap_channel` returns JSON while the image itself is fetched separately from the latest-image
WebSocket.  These tests characterize what is reliably bound together, what survives reconnects,
and where the public contract currently has no identity with which to name a particular frame.
"""

import io
import json
import re
import typing as tp

import numpy as np
import pytest
import tifffile
from ome_types import from_xml
from websockets.sync.client import connect as ws_connect

from tests.mock_server import MOCK, mock_server
from tests.test_api_operations import make_request
from seafront.storage import encode_ome_tiff

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


def snap(
    channel: dict[str, tp.Any],
    token: str,
    ome_context: dict[str, tp.Any] | None = None,
) -> dict[str, tp.Any]:
    body: dict[str, tp.Any] = {"channel": channel}
    if ome_context is not None:
        body["ome_context"] = ome_context
    return ok("/api/action/snap_channel", body, token)


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


class TestCaptureMetadataContract:
    def test_snapshot_and_pixel_handoff_report_the_same_capture_facts(self, channels, token):
        channel = channels[0]
        response = snap(channel, token)
        metadata, image = fetch(channel["handle"], response["image_id"])

        assert image is not None
        for field in (
            "image_id",
            "channel",
            "width_px",
            "height_px",
            "timestamp",
            "position",
            "pixel_format",
            "camera_bit_depth",
            "bit_depth",
        ):
            assert metadata["capture"][field] == response[field]

    def test_capture_metadata_describes_the_full_resolution_source(self, channels, token):
        channel = channels[0]
        response = snap(channel, token)
        metadata, reduced = fetch(channel["handle"], response["image_id"], factor=4)

        assert reduced is not None
        assert metadata["capture"]["width_px"] == response["width_px"]
        assert metadata["capture"]["height_px"] == response["height_px"]
        assert metadata["width"] == reduced.shape[1]
        assert metadata["height"] == reduced.shape[0]

    def test_capture_metadata_contains_calibrated_stage_coordinates(self, channels, token):
        channel = channels[0]
        response = snap(channel, token)

        assert set(response["position"]) == {"x_pos_mm", "y_pos_mm", "z_pos_mm"}
        assert all(np.isfinite(value) for value in response["position"].values())


class TestOmeXmlHandoff:
    def test_snapshot_without_optional_context_still_has_valid_ome_xml(self, channels, token):
        response = snap(channels[0], token)

        ome = from_xml(response["ome_xml"])
        assert ome.images[0].pixels.size_x == response["width_px"]
        assert ome.images[0].pixels.size_y == response["height_px"]

    def test_optional_context_enriches_the_server_generated_ome_xml(self, channels, token):
        context = {
            "project_name": "external-project",
            "plate_name": "plate-7",
            "cell_line": "HeLa",
            "well_name": "C04",
            "site_x": 2,
            "site_y": 3,
            "site_z": 4,
            "plane_index": 4,
            "autofocus_succeeded": True,
        }
        response = snap(channels[0], token, context)
        metadata, _image = fetch(channels[0]["handle"], response["image_id"])

        assert metadata["capture"]["ome_xml"] == response["ome_xml"]
        image = from_xml(response["ome_xml"]).images[0]
        assert image.stage_label is not None
        assert image.stage_label.name == "C04"
        assert "Project: external-project" in (image.description or "")
        assert "Site: (2, 3, 4)" in (image.description or "")
        assert "Autofocus: Success" in (image.description or "")

    @pytest.mark.parametrize("channel_index", [0, 1])
    def test_http_and_websocket_return_byte_identical_ome_xml(
        self, channels, token, channel_index
    ):
        channel = channels[channel_index]
        response = snap(channel, token)
        metadata, _image = fetch(channel["handle"], response["image_id"])

        assert metadata["capture"]["ome_xml"] == response["ome_xml"]

    def test_tiff_embedding_preserves_the_exact_transferred_ome_xml(self, channels, token):
        channel = channels[0]
        response = snap(
            channel,
            token,
            {"project_name": "exact-xml", "plate_name": "plate-3", "well_name": "B02"},
        )
        metadata, image = fetch(channel["handle"], response["image_id"])
        assert image is not None
        transferred_xml = metadata["capture"]["ome_xml"]

        encoded = encode_ome_tiff(image, transferred_xml, "LZW")
        with tifffile.TiffFile(io.BytesIO(encoded)) as tif:
            embedded_xml = tif.pages[0].description

        assert transferred_xml == response["ome_xml"]
        assert embedded_xml == response["ome_xml"]


class TestExplicitLatestFrame:
    def test_latest_returns_the_replacement_and_its_capture_facts(self, channels, token):
        channel = {**channels[0], "exposure_time_ms": 1.0}
        replaced = snap(channel, token)
        current = snap({**channel, "exposure_time_ms": 2.0}, token)
        metadata, image = fetch(channel["handle"], "latest")
        assert image is not None
        assert metadata["image_id"] == current["image_id"]
        assert metadata["capture"]["image_id"] == current["image_id"]
        assert metadata["capture"]["channel"]["exposure_time_ms"] == 2.0
        assert metadata["image_id"] != replaced["image_id"]
        _, pinned = fetch(channel["handle"], current["image_id"])
        assert np.array_equal(image, pinned)

    def test_latest_on_an_empty_channel_still_reports_no_image(self):
        metadata, image = fetch("never-acquired-channel", "latest")
        assert metadata == {}
        assert image is None
