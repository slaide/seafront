"""
Unit tests for MockMicroscope implementation.

Tests verify that MockMicroscope:
- Properly implements the Microscope interface
- Correctly simulates movement operations
- Handles command execution
- Validates parameters against hardware limits
"""

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from seafront.config.basics import GlobalConfigHandler
from seafront.config.registry import ConfigRegistry


@pytest.fixture
def temp_config(tmp_path):
    """Create a temporary config file for testing."""
    config_file = tmp_path / "config.json"
    config_data = {
        "port": 5000,
        "microscopes": [
            {
                "system.microscope_name": "test-mock-microscope",
                "system.microscope_type": "mock",
                "camera.main.id": "mock-camera",
                "camera.main.driver": "galaxy",
                "microcontroller.id": "",
                "microcontroller.driver": "teensy",
                "storage.base_image_output_dir": str(tmp_path / "images"),
                "calibration.offset.x_mm": 0.0,
                "calibration.offset.y_mm": 0.0,
                "calibration.offset.z_mm": 0.0,
                "protocol.forbidden_areas": [],
                "laser.autofocus.available": "no",
                "filter.wheel.available": "no",
                "imaging.channels": [
                    {
                        "name": "Brightfield LED matrix full",
                        "handle": "bfledfull",
                        "source_slot": 0,
                    },
                    {
                        "name": "Fluorescence 488nm",
                        "handle": "f488",
                        "source_slot": 1,
                    },
                ],
                "filter.wheel.configuration": [],
            }
        ],
    }
    config_file.write_text(json.dumps(config_data, indent=2))
    return tmp_path, config_file


@pytest.fixture
def mock_microscope_env(temp_config):
    """Set up environment for MockMicroscope creation."""
    tmp_path, config_file = temp_config

    # Disable realistic delays for testing
    os.environ["MOCK_NO_DELAYS"] = "1"

    # Reset config registry and load test config
    ConfigRegistry.reset()

    with patch.object(GlobalConfigHandler, "home_config", return_value=config_file):
        GlobalConfigHandler.reset("test-mock-microscope")

        yield tmp_path

    # Cleanup
    ConfigRegistry.reset()
    os.environ.pop("MOCK_NO_DELAYS", None)


class TestMockMicroscopeCreation:
    """Tests for MockMicroscope instantiation."""

    def test_make_returns_mock_microscope(self, mock_microscope_env):
        """MockMicroscope.make() should return a MockMicroscope instance."""
        from seafront.hardware.mock_microscope import MockMicroscope

        mc = MockMicroscope.make()
        assert mc is not None
        assert isinstance(mc, MockMicroscope)

    def test_mock_microscope_is_microscope_subclass(self, mock_microscope_env):
        """MockMicroscope should be a subclass of Microscope."""
        from seafront.hardware.microscope import Microscope
        from seafront.hardware.mock_microscope import MockMicroscope

        assert issubclass(MockMicroscope, Microscope)

    def test_make_loads_channels(self, mock_microscope_env):
        """MockMicroscope should load channel configurations."""
        from seafront.hardware.mock_microscope import MockMicroscope

        mc = MockMicroscope.make()
        assert len(mc.channels) == 2
        handles = [ch.handle for ch in mc.channels]
        assert "bfledfull" in handles
        assert "f488" in handles


class TestMockMicroscopeConnection:
    """Tests for MockMicroscope connection management."""

    def test_open_connections_succeeds(self, mock_microscope_env):
        """open_connections should succeed without errors."""
        from seafront.hardware.mock_microscope import MockMicroscope

        mc = MockMicroscope.make()
        with mc.lock(reason="test"):
            mc.open_connections()
        assert mc.is_connected is True

    def test_close_succeeds(self, mock_microscope_env):
        """close should succeed without errors."""
        from seafront.hardware.mock_microscope import MockMicroscope

        mc = MockMicroscope.make()
        with mc.lock(reason="test"):
            mc.open_connections()
        mc.close()
        assert mc.is_connected is False


class TestMockMicroscopeState:
    """Tests for MockMicroscope state management."""

    def test_get_current_state_returns_adapter_state(self, mock_microscope_env):
        """get_current_state should return an AdapterState."""
        from seafront.hardware.adapter import AdapterState
        from seafront.hardware.mock_microscope import MockMicroscope

        mc = MockMicroscope.make()
        state = asyncio.run(mc.get_current_state())
        assert isinstance(state, AdapterState)

    def test_state_has_position(self, mock_microscope_env):
        """State should include stage position."""
        from seafront.hardware.mock_microscope import MockMicroscope

        mc = MockMicroscope.make()
        state = asyncio.run(mc.get_current_state())
        assert state.stage_position is not None
        assert hasattr(state.stage_position, "x_pos_mm")
        assert hasattr(state.stage_position, "y_pos_mm")
        assert hasattr(state.stage_position, "z_pos_mm")

    def test_initial_position_is_origin(self, mock_microscope_env):
        """Initial position should be at origin (0, 0, 0)."""
        from seafront.hardware.mock_microscope import MockMicroscope

        mc = MockMicroscope.make()
        state = asyncio.run(mc.get_current_state())
        # With zero calibration offsets, position should be (0, 0, 0)
        assert state.stage_position.x_pos_mm == 0.0
        assert state.stage_position.y_pos_mm == 0.0
        assert state.stage_position.z_pos_mm == 0.0


class TestMockMicroscopeHardwareLimits:
    """Tests for MockMicroscope hardware limits."""

    def test_get_hardware_limits_returns_limits(self, mock_microscope_env):
        """get_hardware_limits should return a HardwareLimits object."""
        from seafront.hardware.microscope import HardwareLimits
        from seafront.hardware.mock_microscope import MockMicroscope

        mc = MockMicroscope.make()
        limits = mc.get_hardware_limits()
        assert limits is not None

    def test_limits_have_exposure_time(self, mock_microscope_env):
        """Limits should include exposure time limits."""
        from seafront.hardware.mock_microscope import MockMicroscope

        mc = MockMicroscope.make()
        limits = mc.get_hardware_limits()
        assert "imaging_exposure_time_ms" in limits.to_dict()

    def test_exposure_limits_are_valid(self, mock_microscope_env):
        """Exposure time limits should have valid min/max/step."""
        from seafront.hardware.mock_microscope import MockMicroscope

        mc = MockMicroscope.make()
        limits = mc.get_hardware_limits()
        exposure = limits.imaging_exposure_time_ms
        assert exposure["min"] > 0
        assert exposure["max"] > exposure["min"]
        assert exposure["step"] > 0


class TestMockMicroscopeMovement:
    """Tests for MockMicroscope movement commands."""

    def test_move_to_command(self, mock_microscope_env):
        """MoveTo command should update position."""
        from seafront.hardware.mock_microscope import MockMicroscope
        from seafront.server import commands as cmd

        mc = MockMicroscope.make()

        # Execute move command
        move_cmd = cmd.MoveTo(x_mm=10.0, y_mm=20.0, z_mm=5.0)
        asyncio.run(mc.execute(move_cmd))

        # Check position updated
        state = asyncio.run(mc.get_current_state())
        assert abs(state.stage_position.x_pos_mm - 10.0) < 0.01
        assert abs(state.stage_position.y_pos_mm - 20.0) < 0.01
        assert abs(state.stage_position.z_pos_mm - 5.0) < 0.01

    def test_move_by_command(self, mock_microscope_env):
        """MoveBy command should update position by relative amount."""
        from seafront.hardware.mock_microscope import MockMicroscope
        from seafront.server import commands as cmd

        mc = MockMicroscope.make()

        # First move to known position
        asyncio.run(mc.execute(cmd.MoveTo(x_mm=10.0, y_mm=10.0, z_mm=1.0)))

        # Move by relative amount
        asyncio.run(mc.execute(cmd.MoveBy(axis="x", distance_mm=5.0)))

        # Check position
        state = asyncio.run(mc.get_current_state())
        assert abs(state.stage_position.x_pos_mm - 15.0) < 0.01


class TestMockMicroscopeLoadingPosition:
    """Tests for MockMicroscope loading position operations."""

    def test_enter_loading_position(self, mock_microscope_env):
        """LoadingPositionEnter should set is_in_loading_position."""
        from seafront.hardware.mock_microscope import MockMicroscope
        from seafront.server import commands as cmd

        mc = MockMicroscope.make()

        # Enter loading position
        asyncio.run(mc.execute(cmd.LoadingPositionEnter()))

        assert mc.is_in_loading_position is True

    def test_leave_loading_position(self, mock_microscope_env):
        """LoadingPositionLeave should clear is_in_loading_position."""
        from seafront.hardware.mock_microscope import MockMicroscope
        from seafront.server import commands as cmd

        mc = MockMicroscope.make()

        # Enter then leave
        asyncio.run(mc.execute(cmd.LoadingPositionEnter()))
        asyncio.run(mc.execute(cmd.LoadingPositionLeave()))

        assert mc.is_in_loading_position is False


class TestMockMicroscopeLocking:
    """Tests for MockMicroscope locking mechanism."""

    def test_lock_context_manager(self, mock_microscope_env):
        """lock() context manager should work."""
        from seafront.hardware.mock_microscope import MockMicroscope

        mc = MockMicroscope.make()

        with mc.lock(reason="test_operation") as locked_mc:
            assert locked_mc is mc

    def test_lock_tracks_reason(self, mock_microscope_env):
        """lock() should track the lock reason."""
        from seafront.hardware.mock_microscope import MockMicroscope

        mc = MockMicroscope.make()

        with mc.lock(reason="my_test_reason"):
            reasons = mc.get_lock_reasons()
            assert "my_test_reason" in reasons

    def test_lock_releases_on_exit(self, mock_microscope_env):
        """lock() should release when context exits."""
        from seafront.hardware.mock_microscope import MockMicroscope

        mc = MockMicroscope.make()

        with mc.lock(reason="temporary"):
            pass

        reasons = mc.get_lock_reasons()
        assert "temporary" not in reasons
