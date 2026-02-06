"""Seafront microscope client."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class SeafrontClient:
    """Client for interacting with the Seafront microscope server."""

    def __init__(self, base_url: str, microscope_name: str):
        self.base_url = base_url.rstrip("/")
        self.microscope_name = microscope_name

    def _post(self, endpoint: str, data: dict | None = None) -> dict:
        """Make a POST request to the server."""
        url = f"{self.base_url}{endpoint}"
        if data is None:
            data = {}
        data["target_device"] = self.microscope_name

        req = urllib.request.Request(url)
        req.method = "POST"
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(data).encode("utf-8")

        with urllib.request.urlopen(req, timeout=60) as response:
            body = response.read().decode()
            return json.loads(body) if body else {}

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

    def snap_bf_channel(self, channel_config: dict) -> None:
        self._post("/api/action/snap_channel", {"channel": channel_config})

    def snap_laf(self, exposure_time_ms: float = 5, analog_gain: float = 10) -> None:
        self._post(
            "/api/action/snap_reflection_autofocus",
            {
                "exposure_time_ms": exposure_time_ms,
                "analog_gain": analog_gain,
                "turn_laser_on": True,
                "turn_laser_off": True,
            },
        )

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
