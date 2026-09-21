"""Seafront microscope client."""

from __future__ import annotations

import atexit
import getpass
import json
import socket
import urllib.error
import urllib.request


class SeafrontClient:
    """
    Client for interacting with the Seafront microscope server.

    Hardware commands need the server's device lease (seafront/server/lock.py), so the client
    takes one the first time it drives anything and gives it back when the script exits. If
    somebody else holds the lease the server answers 423 and the request fails - that is the
    point of the lease, so it is not retried or broken automatically.
    """

    def __init__(self, base_url: str, microscope_name: str, owner: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.microscope_name = microscope_name
        self.owner = owner or f"laf scripts ({getpass.getuser()}@{socket.gethostname()})"
        self.lock_token: str | None = None

    def _post(self, endpoint: str, data: dict | None = None, take_lease: bool = True) -> dict:
        """Make a POST request to the server."""
        url = f"{self.base_url}{endpoint}"
        if data is None:
            data = {}
        data["target_device"] = self.microscope_name
        if self.lock_token is not None:
            data["lock_token"] = self.lock_token

        req = urllib.request.Request(url)
        req.method = "POST"
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(data).encode("utf-8")

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                body = response.read().decode()
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            if e.code != 423 or not take_lease:
                raise
            detail = json.loads(e.read().decode() or "{}").get("detail", {})
            if detail.get("reason") != "not_held":
                raise  # somebody else has the microscope: their call, not ours to take
            self.acquire_lock()
            return self._post(endpoint, data, take_lease=False)

    def acquire_lock(self, ttl_s: float = 600.0) -> None:
        """Take the device lease, and give it back when this process exits."""
        lease = self._post(
            "/api/lock/acquire", {"owner": self.owner, "ttl_s": ttl_s}, take_lease=False
        )
        self.lock_token = lease["token"]
        atexit.register(self.release_lock)

    def release_lock(self) -> None:
        """Give the device lease back; harmless when we do not hold one."""
        if self.lock_token is None:
            return
        try:
            self._post("/api/lock/release", take_lease=False)
        except urllib.error.HTTPError:
            pass  # already broken or expired
        self.lock_token = None

    def get_current_state(self) -> dict:
        return self._post("/api/get_info/current_state")

    def get_z_position_mm(self) -> float:
        state = self.get_current_state()
        return state["adapter_state"]["stage_position"]["z_pos_mm"]

    def get_position(self) -> tuple[float, float, float]:
        """Return current (x_mm, y_mm, z_mm) position."""
        state = self.get_current_state()
        pos = state["adapter_state"]["stage_position"]
        return pos["x_pos_mm"], pos["y_pos_mm"], pos["z_pos_mm"]

    def try_move_to(self, x_mm: float, y_mm: float, z_mm: float) -> bool:
        """Try to move to a position. Returns True if successful, False if forbidden/error."""
        try:
            self._post("/api/action/move_to", {"x_mm": x_mm, "y_mm": y_mm, "z_mm": z_mm})
            return True
        except urllib.error.HTTPError:
            # 500 = forbidden area or other internal error
            return False

    def move_by_z(self, distance_mm: float) -> None:
        self._post("/api/action/move_by", {"axis": "z", "distance_mm": distance_mm})

    def move_to_z(self, z_mm: float) -> None:
        self._post("/api/action/move_to", {"z_mm": z_mm})

    def move_to(self, x_mm: float, y_mm: float, z_mm: float) -> None:
        self._post("/api/action/move_to", {"x_mm": x_mm, "y_mm": y_mm, "z_mm": z_mm})

    def approach_target_offset(self, target_offset_um: float, config: dict) -> dict:
        return self._post(
            "/api/action/laser_autofocus_move_to_target_offset",
            {
                "target_offset_um": target_offset_um,
                "config_file": config,
                "max_num_reps": 3,
                "pre_approach_refz": True,
            },
        )

    def snap_bf_channel(self, channel_config: dict) -> str:
        return self._post("/api/action/snap_channel", {"channel": channel_config})["image_id"]

    def snap_laf(self, exposure_time_ms: float = 5, analog_gain: float = 10) -> str:
        return self._post(
            "/api/action/snap_reflection_autofocus",
            {
                "exposure_time_ms": exposure_time_ms,
                "analog_gain": analog_gain,
                "turn_laser_on": True,
                "turn_laser_off": True,
            },
        )["image_id"]

    def get_hardware_capabilities(self) -> dict:
        return self._post("/api/get_features/hardware_capabilities")

    def get_config(self, config_name: str) -> dict:
        return self._post("/api/acquisition/config_fetch", {"config_file": config_name})

    def get_config_list(self) -> list[dict]:
        result = self._post("/api/acquisition/config_list")
        return result.get("configs", [])

    def get_machine_defaults(self) -> list[dict]:
        """Get all machine config items from the server."""
        return self._post("/api/get_features/machine_defaults")  # type: ignore[return-value]
