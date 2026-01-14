"""
API integration tests for microscope operations.

Tests actual API endpoints with the mock microscope running.
These tests verify that the server correctly handles:
- Movement commands (move_to, move_by, move_to_well)
- Position queries (current_state)
- Loading position operations
- Hardware capabilities and limits
- Channel streaming and snapshots
"""

import json
import os
import subprocess
import time
import urllib.error
import urllib.request

import pytest

# The mock microscope name used in tests
MOCK_MICROSCOPE_NAME = "mocroscope"


@pytest.fixture(scope="class")
def running_server(request):
    """Start server once for all tests in the class."""
    env = os.environ.copy()
    env["MOCK_NO_DELAYS"] = "1"

    proc = subprocess.Popen(
        ["uv", "run", "python", "-m", "seafront", "--microscope", MOCK_MICROSCOPE_NAME],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    # Wait for server to start
    server_ready = False
    start_time = time.time()
    timeout = 15

    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen("http://127.0.0.1:5000/", timeout=1) as response:
                if response.status == 200:
                    server_ready = True
                    break
        except (urllib.error.URLError, ConnectionRefusedError):
            time.sleep(0.5)

    if not server_ready:
        proc.terminate()
        proc.wait()
        pytest.fail("Server did not start within timeout")

    # Make server URL available to tests
    request.cls.server_url = "http://127.0.0.1:5000"

    yield proc

    # Cleanup
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def make_request(url: str, method: str = "POST", data: dict | None = None) -> tuple[int, dict | None]:
    """Helper to make HTTP requests and return status code and JSON response.

    All API endpoints except /api/health require:
    - POST method
    - target_device field in JSON body
    """
    req = urllib.request.Request(url)
    req.method = method

    # Ensure data dict exists and has target_device for POST requests
    if method == "POST":
        if data is None:
            data = {}
        if "target_device" not in data:
            data["target_device"] = MOCK_MICROSCOPE_NAME

    if data is not None:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(data).encode("utf-8")

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            body = response.read().decode()
            return response.status, json.loads(body) if body else None
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        return e.code, json.loads(body) if body else None


@pytest.mark.usefixtures("running_server")
class TestCurrentStateEndpoint:
    """Tests for the /api/get_info/current_state endpoint."""

    def test_current_state_returns_200(self):
        """Current state endpoint should return 200."""
        status, data = make_request(f"{self.server_url}/api/get_info/current_state", method="POST")
        assert status == 200, f"Expected 200, got {status}: {data}"

    def test_current_state_has_stage_position(self):
        """Current state should include stage position."""
        status, data = make_request(f"{self.server_url}/api/get_info/current_state", method="POST")
        assert status == 200, f"Failed with {status}: {data}"
        assert data is not None
        # Response structure: data["adapter_state"]["stage_position"]
        assert "adapter_state" in data, "Response missing adapter_state"
        assert "stage_position" in data["adapter_state"], "adapter_state missing stage_position"

    def test_stage_position_has_coordinates(self):
        """Stage position should have x, y, z coordinates."""
        status, data = make_request(f"{self.server_url}/api/get_info/current_state", method="POST")
        assert status == 200
        assert data is not None
        pos = data["adapter_state"]["stage_position"]
        assert "x_pos_mm" in pos, "Stage position missing x_pos_mm"
        assert "y_pos_mm" in pos, "Stage position missing y_pos_mm"
        assert "z_pos_mm" in pos, "Stage position missing z_pos_mm"

    def test_stage_position_values_are_numbers(self):
        """Stage position values should be numeric."""
        status, data = make_request(f"{self.server_url}/api/get_info/current_state", method="POST")
        assert status == 200
        assert data is not None
        pos = data["adapter_state"]["stage_position"]
        assert isinstance(pos["x_pos_mm"], (int, float))
        assert isinstance(pos["y_pos_mm"], (int, float))
        assert isinstance(pos["z_pos_mm"], (int, float))


@pytest.mark.usefixtures("running_server")
class TestHardwareCapabilitiesEndpoint:
    """Tests for the /api/get_features/hardware_capabilities endpoint."""

    def test_hardware_capabilities_returns_200(self):
        """Hardware capabilities endpoint should return 200."""
        status, data = make_request(f"{self.server_url}/api/get_features/hardware_capabilities", method="POST")
        assert status == 200, f"Expected 200, got {status}: {data}"

    def test_hardware_capabilities_has_limits(self):
        """Hardware capabilities should include imaging limits."""
        status, data = make_request(f"{self.server_url}/api/get_features/hardware_capabilities", method="POST")
        assert status == 200
        assert data is not None
        # Response structure: data["hardware_limits"]["imaging_exposure_time_ms"]
        assert "hardware_limits" in data, f"Missing hardware_limits in {list(data.keys())}"
        limits = data["hardware_limits"]
        # Check for common limit keys
        assert "imaging_exposure_time_ms" in limits, f"Missing imaging_exposure_time_ms in {list(limits.keys())}"
        assert "imaging_analog_gain_db" in limits

    def test_exposure_limits_have_min_max(self):
        """Exposure limits should have min, max, step values."""
        status, data = make_request(f"{self.server_url}/api/get_features/hardware_capabilities", method="POST")
        assert status == 200
        assert data is not None
        limits = data["hardware_limits"]
        exposure_limits = limits["imaging_exposure_time_ms"]
        assert "min" in exposure_limits
        assert "max" in exposure_limits
        assert "step" in exposure_limits
        # Validate values are sensible
        assert exposure_limits["min"] > 0
        assert exposure_limits["max"] > exposure_limits["min"]


@pytest.mark.usefixtures("running_server")
class TestMachineDefaultsEndpoint:
    """Tests for the /api/get_features/machine_defaults endpoint."""

    def test_machine_defaults_returns_200(self):
        """Machine defaults endpoint should return 200."""
        status, data = make_request(f"{self.server_url}/api/get_features/machine_defaults", method="POST")
        assert status == 200, f"Expected 200, got {status}: {data}"

    def test_machine_defaults_is_list(self):
        """Machine defaults should return a list of config items."""
        status, data = make_request(f"{self.server_url}/api/get_features/machine_defaults", method="POST")
        assert status == 200
        assert isinstance(data, list), "Machine defaults should be a list"
        assert len(data) > 0, "Machine defaults should not be empty"

    def test_machine_defaults_items_have_handle(self):
        """Each config item should have a handle."""
        status, data = make_request(f"{self.server_url}/api/get_features/machine_defaults", method="POST")
        assert status == 200
        assert data is not None
        for item in data:
            assert "handle" in item, f"Config item missing handle: {item}"


@pytest.mark.usefixtures("running_server")
class TestMovementEndpoints:
    """Tests for movement-related API endpoints."""

    def test_move_to_returns_200(self):
        """Move to endpoint should return 200 for valid coordinates."""
        # Move to a safe position in the center of the plate
        status, data = make_request(
            f"{self.server_url}/api/action/move_to",
            method="POST",
            data={"x_mm": 50.0, "y_mm": 40.0, "z_mm": 1.0},
        )
        assert status == 200, f"Expected 200, got {status}: {data}"

    def test_move_to_updates_position(self):
        """Move to should update the reported position."""
        # Move to specific coordinates
        target_x, target_y, target_z = 60.0, 45.0, 2.0
        status, _ = make_request(
            f"{self.server_url}/api/action/move_to",
            method="POST",
            data={"x_mm": target_x, "y_mm": target_y, "z_mm": target_z},
        )
        assert status == 200

        # Verify position updated
        status, data = make_request(f"{self.server_url}/api/get_info/current_state", method="POST")
        assert status == 200
        assert data is not None
        pos = data["adapter_state"]["stage_position"]
        # Allow small floating point tolerance
        assert abs(pos["x_pos_mm"] - target_x) < 0.01
        assert abs(pos["y_pos_mm"] - target_y) < 0.01
        assert abs(pos["z_pos_mm"] - target_z) < 0.01

    def test_move_by_returns_200(self):
        """Move by endpoint should return 200."""
        status, data = make_request(
            f"{self.server_url}/api/action/move_by",
            method="POST",
            data={"axis": "x", "distance_mm": 1.0},
        )
        assert status == 200, f"Expected 200, got {status}: {data}"

    def test_move_by_updates_position(self):
        """Move by should update position by the specified amount."""
        # First get current position
        status, data = make_request(f"{self.server_url}/api/get_info/current_state", method="POST")
        assert status == 200
        assert data is not None
        initial_x = data["adapter_state"]["stage_position"]["x_pos_mm"]

        # Move by a known amount
        move_distance = 5.0
        status, _ = make_request(
            f"{self.server_url}/api/action/move_by",
            method="POST",
            data={"axis": "x", "distance_mm": move_distance},
        )
        assert status == 200

        # Verify position changed
        status, data = make_request(f"{self.server_url}/api/get_info/current_state", method="POST")
        assert status == 200
        assert data is not None
        new_x = data["adapter_state"]["stage_position"]["x_pos_mm"]
        assert abs(new_x - (initial_x + move_distance)) < 0.01


@pytest.mark.usefixtures("running_server")
class TestLoadingPositionEndpoints:
    """Tests for loading position operations."""

    def test_enter_loading_position_returns_200(self):
        """Enter loading position should return 200."""
        status, data = make_request(
            f"{self.server_url}/api/action/enter_loading_position",
            method="POST",
        )
        assert status == 200, f"Expected 200, got {status}: {data}"

    def test_enter_loading_position_updates_state(self):
        """Entering loading position should update is_in_loading_position flag."""
        # Enter loading position
        status, data = make_request(
            f"{self.server_url}/api/action/enter_loading_position",
            method="POST",
        )
        assert status == 200, f"Expected 200, got {status}: {data}"

        # Check state
        status, data = make_request(f"{self.server_url}/api/get_info/current_state", method="POST")
        assert status == 200
        assert data is not None
        assert data["adapter_state"]["is_in_loading_position"] is True

    def test_leave_loading_position_returns_200(self):
        """Leave loading position should return 200."""
        # First enter loading position
        make_request(
            f"{self.server_url}/api/action/enter_loading_position",
            method="POST",
        )

        # Then leave
        status, data = make_request(
            f"{self.server_url}/api/action/leave_loading_position",
            method="POST",
        )
        assert status == 200, f"Expected 200, got {status}: {data}"

    def test_leave_loading_position_updates_state(self):
        """Leaving loading position should update is_in_loading_position flag."""
        # Enter then leave loading position
        make_request(
            f"{self.server_url}/api/action/enter_loading_position",
            method="POST",
        )
        make_request(
            f"{self.server_url}/api/action/leave_loading_position",
            method="POST",
        )

        # Check state
        status, data = make_request(f"{self.server_url}/api/get_info/current_state", method="POST")
        assert status == 200
        assert data is not None
        assert data["adapter_state"]["is_in_loading_position"] is False


@pytest.mark.usefixtures("running_server")
class TestIlluminationEndpoint:
    """Tests for illumination control."""

    def test_turn_off_all_illumination_returns_200(self):
        """Turn off all illumination should return 200."""
        status, data = make_request(
            f"{self.server_url}/api/action/turn_off_all_illumination",
            method="POST",
        )
        assert status == 200, f"Expected 200, got {status}: {data}"


@pytest.mark.usefixtures("running_server")
class TestAcquisitionConfigEndpoints:
    """Tests for acquisition configuration endpoints."""

    def test_config_list_returns_200(self):
        """Config list endpoint should return 200."""
        status, data = make_request(f"{self.server_url}/api/acquisition/config_list", method="POST")
        assert status == 200, f"Expected 200, got {status}: {data}"

    def test_config_list_returns_list(self):
        """Config list should return a dict with configs list."""
        status, data = make_request(f"{self.server_url}/api/acquisition/config_list", method="POST")
        assert status == 200
        assert "configs" in data, f"Response should have 'configs' key, got: {data}"
        assert isinstance(data["configs"], list)
