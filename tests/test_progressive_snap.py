"""
How a progressive snap reports its end.

The snap sends a status as each channel starts and completes, and the stream ends with
exactly one of two statuses:

- "finished", when every channel was snapped
- "error", when it stopped early: naming the channel it stopped at and why, or saying it was
  cancelled

A snap that stopped early is a failed job, in the worker's log as in the call's result, and
nothing after the error says the channels are complete.
"""

import asyncio

import pytest
import seaconfig as sc

from seafront.logger import logger
from seafront.server import commands as cmd
from seafront.server.commands import ChannelSnapProgressiveStatus


def channel(handle: str) -> sc.AcquisitionChannelConfig:
    return sc.AcquisitionChannelConfig(
        name=f"channel {handle}", handle=handle, illum_perc=20.0, exposure_time_ms=5.0,
        analog_gain=0.0, z_offset_um=0.0, num_z_planes=1, delta_z_um=1.0,
        filter_handle=None, enabled=True,
    )


def config(*handles: str) -> sc.AcquisitionConfig:
    return sc.AcquisitionConfig(
        project_name="proj", plate_name="plate", cell_line="",
        grid=sc.AcquisitionWellSiteConfiguration(
            num_x=1, delta_x_mm=0.9, num_y=1, delta_y_mm=0.9, num_t=1,
            delta_t=sc.acquisition.AcquisitionWellSiteConfigurationDeltaTime(h=0, m=0, s=0),
            mask=[sc.acquisition.AcquisitionWellSiteConfigurationSiteSelectionItem(
                row=0, col=0, selected=True)],
        ),
        wellplate_type=sc.Plates[0],
        plate_wells=[sc.acquisition.PlateWellConfig(row=1, col=1, selected=True)],
        channels=[channel(h) for h in handles],
        autofocus_enabled=False,
    )


@pytest.fixture
def connected(mock_core):
    asyncio.run(mock_core.run(cmd.EstablishHardwareConnection()))
    return mock_core


@pytest.fixture
def log_lines():
    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(message.record["message"]), level="INFO")
    yield lines
    logger.remove(sink)


def snap(core, protocol) -> tuple[list[ChannelSnapProgressiveStatus], BaseException | None]:
    statuses: list[ChannelSnapProgressiveStatus] = []
    core._progressive_snap_callbacks["t"] = statuses.append
    try:
        asyncio.run(core.start_progressive_channel_snap(protocol, "t"))
        return statuses, None
    except BaseException as e:
        return statuses, e
    finally:
        core._progressive_snap_callbacks.pop("t", None)


def fail_channel(core, monkeypatch, handle: str):
    "snapping `handle` raises; every other channel snaps as usual"
    scope_type = type(core.microscope)
    real = scope_type.execute

    async def execute(self, command):
        if isinstance(command, cmd.ChannelSnapshot) and command.channel.handle == handle:
            raise RuntimeError("sensor fell over")
        return await real(self, command)

    monkeypatch.setattr(scope_type, "execute", execute)


def terminal(statuses):
    return [s for s in statuses if s.status in ("finished", "error")]


def test_a_snap_of_every_channel_ends_with_finished(connected):
    statuses, error = snap(connected, config("bfledfull", "fluo488"))
    assert error is None
    assert [s.status for s in terminal(statuses)] == ["finished"]
    assert statuses[-1].status == "finished"
    assert statuses[-1].completed_channels == 2


def test_a_failed_channel_ends_the_snap_with_one_error_naming_it(connected, monkeypatch):
    fail_channel(connected, monkeypatch, "fluo488")
    statuses, error = snap(connected, config("bfledfull", "fluo488", "fluo561"))

    ends = terminal(statuses)
    assert [s.status for s in ends] == ["error"], [s.message for s in statuses]
    assert ends[0].channel_handle == "fluo488"
    assert "sensor fell over" in (ends[0].error_detail or "")
    assert statuses[-1] is ends[0], "a status was sent after the error"
    assert "fluo561" not in {s.channel_handle for s in statuses}


def test_a_failed_snap_fails_the_call(connected, monkeypatch):
    fail_channel(connected, monkeypatch, "fluo488")
    _statuses, error = snap(connected, config("bfledfull", "fluo488"))
    assert isinstance(error, RuntimeError)
    assert "sensor fell over" in str(error)


def test_a_failed_snap_is_logged_as_a_failed_job(connected, monkeypatch, log_lines):
    fail_channel(connected, monkeypatch, "fluo488")
    snap(connected, config("bfledfull", "fluo488"))
    ends = [line for line in log_lines if line.startswith("command ") and "progressive_snap" in line
            and " - " in line and ("failed" in line or "finished" in line)]
    assert len(ends) == 1, ends
    assert "failed" in ends[0]
    assert "sensor fell over" in ends[0]


# --- over the socket ------------------------------------------------------------------------


def over_the_socket(core, protocol) -> list[dict]:
    "every message the socket sends for one snap, up to its close"
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    from seafront.server.app import make_app

    lease = core.lock.acquire("test")
    client = TestClient(make_app(core))
    received: list[dict] = []
    with client.websocket_connect("/ws/action/snap_selected_channels_progressive") as ws:
        ws.send_json({"target_device": core.microscope_name, "lock_token": lease.token})
        ws.send_json(protocol.model_dump(mode="json"))
        try:
            while True:
                received.append(ws.receive_json())
        except WebSocketDisconnect:
            pass
    return received


def test_the_socket_sends_the_finished_status_before_closing(connected):
    received = over_the_socket(connected, config("bfledfull", "fluo488"))
    assert [m["status"] for m in received][-1:] == ["finished"], received


def test_the_socket_sends_one_error_for_a_failed_snap(connected, monkeypatch):
    fail_channel(connected, monkeypatch, "fluo488")
    received = over_the_socket(connected, config("bfledfull", "fluo488"))
    ends = [m for m in received if m["status"] in ("finished", "error")]
    assert len(ends) == 1, received
    assert ends[0]["channel_handle"] == "fluo488"
    assert received[-1] == ends[0]
