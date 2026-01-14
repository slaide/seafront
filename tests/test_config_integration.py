"""
Integration tests for the config system.

Tests the full flow of:
- Loading config from file
- GlobalConfigHandler.reset()
- Saving config back to file with object type conversion
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from seafront.config.registry import ConfigRegistry, config_item
from seafront.config.basics import GlobalConfigHandler, ServerConfig


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create a temporary config directory with a test config file."""
    config_file = tmp_path / "config.json"
    config_data = {
        "port": 5000,
        "microscopes": [
            {
                "system.microscope_name": "test-microscope",
                "system.microscope_type": "mock",
                "camera.main.id": "test-camera-id",
                "camera.main.driver": "galaxy",
                "microcontroller.id": "test-mc-id",
                "microcontroller.driver": "teensy",
                "storage.base_image_output_dir": str(tmp_path / "images"),
                "calibration.offset.x_mm": 1.0,
                "calibration.offset.y_mm": 2.0,
                "calibration.offset.z_mm": 3.0,
                "protocol.forbidden_areas": [
                    {"name": "Test Area", "min_x_mm": 0, "max_x_mm": 10, "min_y_mm": 0, "max_y_mm": 10, "reason": "Test"}
                ],
                "laser.autofocus.available": "yes",
                "laser.autofocus.camera.id": "test-laf-camera",
                "laser.autofocus.camera.driver": "galaxy",
                "filter.wheel.available": "no",
                "imaging.channels": [
                    {"name": "Test Channel", "handle": "test", "source_slot": 0}
                ],
                "filter.wheel.configuration": []
            },
            {
                "system.microscope_name": "second-microscope",
                "system.microscope_type": "mock",
                "camera.main.id": "second-camera",
                "camera.main.driver": "toupcam",
                "microcontroller.id": "",
                "microcontroller.driver": "teensy",
                "storage.base_image_output_dir": str(tmp_path / "images2"),
                "calibration.offset.x_mm": 0.0,
                "calibration.offset.y_mm": 0.0,
                "calibration.offset.z_mm": 0.0,
                "protocol.forbidden_areas": [],
                "laser.autofocus.available": "no",
                "filter.wheel.available": "no",
                "imaging.channels": [],
                "filter.wheel.configuration": []
            }
        ]
    }
    config_file.write_text(json.dumps(config_data, indent=2))
    return tmp_path, config_file


class TestGlobalConfigHandlerReset:
    """Tests for GlobalConfigHandler.reset() integration."""

    def setup_method(self):
        ConfigRegistry.reset()

    def teardown_method(self):
        ConfigRegistry.reset()

    def test_reset_loads_microscope_config(self, temp_config_dir):
        """reset() should load config for the specified microscope."""
        tmp_path, config_file = temp_config_dir

        with patch.object(GlobalConfigHandler, 'home_config', return_value=config_file):
            GlobalConfigHandler.reset("test-microscope")

        # Check that core config items are registered
        items = ConfigRegistry.get_all()
        assert len(items) > 0

        # Check microscope name is loaded
        name = ConfigRegistry.get_value("system.microscope_name")
        assert name == "test-microscope"

    def test_reset_loads_object_types_correctly(self, temp_config_dir):
        """reset() should handle object types from config file."""
        tmp_path, config_file = temp_config_dir

        with patch.object(GlobalConfigHandler, 'home_config', return_value=config_file):
            GlobalConfigHandler.reset("test-microscope")

        # imaging.channels should be a native list
        channels = ConfigRegistry.get_value("imaging.channels")
        assert isinstance(channels, list)
        assert len(channels) == 1
        assert channels[0]["name"] == "Test Channel"

    def test_reset_registers_laser_autofocus_when_available(self, temp_config_dir):
        """reset() should register LAF config when available."""
        tmp_path, config_file = temp_config_dir

        with patch.object(GlobalConfigHandler, 'home_config', return_value=config_file):
            GlobalConfigHandler.reset("test-microscope")

        # test-microscope has laser.autofocus.available = "yes"
        laf_available = ConfigRegistry.get_value("laser.autofocus.available")
        assert laf_available == "yes"

        # LAF camera config should be registered
        laf_camera_id = ConfigRegistry.get_value("laser.autofocus.camera.id")
        assert laf_camera_id == "test-laf-camera"

    def test_reset_skips_laser_autofocus_when_unavailable(self, temp_config_dir):
        """reset() should not register LAF config when unavailable."""
        tmp_path, config_file = temp_config_dir

        with patch.object(GlobalConfigHandler, 'home_config', return_value=config_file):
            GlobalConfigHandler.reset("second-microscope")

        # second-microscope has laser.autofocus.available = "no"
        laf_available = ConfigRegistry.get_value("laser.autofocus.available")
        assert laf_available == "no"

        # LAF camera config should NOT be registered
        with pytest.raises(KeyError):
            ConfigRegistry.get_value("laser.autofocus.camera.id")

    def test_reset_selects_correct_microscope(self, temp_config_dir):
        """reset() should select the correct microscope from multiple configs."""
        tmp_path, config_file = temp_config_dir

        with patch.object(GlobalConfigHandler, 'home_config', return_value=config_file):
            GlobalConfigHandler.reset("second-microscope")

        name = ConfigRegistry.get_value("system.microscope_name")
        assert name == "second-microscope"

        camera_driver = ConfigRegistry.get_value("camera.main.driver")
        assert camera_driver == "toupcam"

    def test_reset_raises_on_unknown_microscope(self, temp_config_dir):
        """reset() should raise ValueError for unknown microscope."""
        tmp_path, config_file = temp_config_dir

        with patch.object(GlobalConfigHandler, 'home_config', return_value=config_file):
            with pytest.raises(ValueError, match="not found"):
                GlobalConfigHandler.reset("nonexistent-microscope")


class TestConfigSaveRoundtrip:
    """Tests for saving config and round-trip integrity."""

    def setup_method(self):
        ConfigRegistry.reset()

    def teardown_method(self):
        ConfigRegistry.reset()

    def test_object_type_stored_and_saved_as_native(self):
        """Object types should be stored and saved as native objects."""
        # Initialize with native JSON (simulating a config file)
        file_values = {
            "system.microscope_name": "test-scope",
            "imaging.channels": [
                {"name": "Channel 1", "handle": "ch1", "source_slot": 1}
            ],
            "protocol.forbidden_areas": [
                {"name": "Area 1", "min_x_mm": 0, "max_x_mm": 10, "min_y_mm": 0, "max_y_mm": 10, "reason": "test"}
            ],
        }

        ConfigRegistry.init(file_values)

        # Register items with object type
        ConfigRegistry.register(
            config_item(
                handle="system.microscope_name",
                name="microscope name",
                value_kind="text",
                default="default",
                persistent=True,
            ),
            config_item(
                handle="imaging.channels",
                name="channels",
                value_kind="object",
                default=[],
                persistent=True,
            ),
            config_item(
                handle="protocol.forbidden_areas",
                name="forbidden areas",
                value_kind="object",
                default=[],
                persistent=True,
            ),
        )

        # Verify internal storage is native objects
        channels_value = ConfigRegistry.get_value("imaging.channels")
        assert isinstance(channels_value, list)
        assert channels_value[0]["name"] == "Channel 1"

        # Build save dict (simulating store() logic)
        save_dict = {}
        for handle in ConfigRegistry.get_persistent_handles():
            save_dict[handle] = ConfigRegistry.get_value(handle)

        # Verify channels is native list
        assert isinstance(save_dict["imaging.channels"], list)
        assert save_dict["imaging.channels"][0]["name"] == "Channel 1"

        # Verify forbidden_areas is native list
        assert isinstance(save_dict["protocol.forbidden_areas"], list)
        assert save_dict["protocol.forbidden_areas"][0]["name"] == "Area 1"

        # Verify text types are unchanged
        assert save_dict["system.microscope_name"] == "test-scope"

    def test_store_writes_native_json(self, temp_config_dir):
        """store() should write object types as native JSON."""
        tmp_path, config_file = temp_config_dir

        with patch.object(GlobalConfigHandler, 'home_config', return_value=config_file):
            GlobalConfigHandler.reset("test-microscope")

            # Modify a value
            ConfigRegistry.set_value("calibration.offset.x_mm", 99.0)

            # Save
            GlobalConfigHandler.store()

        # Read back the file
        saved_config = json.loads(config_file.read_text())
        microscope = saved_config["microscopes"][0]

        # Object types should be native JSON, not strings
        assert isinstance(microscope["imaging.channels"], list)
        assert isinstance(microscope["protocol.forbidden_areas"], list)
        assert microscope["imaging.channels"][0]["name"] == "Test Channel"

        # Modified value should be saved
        assert microscope["calibration.offset.x_mm"] == 99.0


class TestObjectTypeEnforcement:
    """Tests that object types strictly require dict/list, not strings."""

    def setup_method(self):
        ConfigRegistry.reset()

    def teardown_method(self):
        ConfigRegistry.reset()

    def test_object_type_rejects_json_string(self):
        """Object type must reject JSON strings - use native objects only."""
        json_string = '[{"name": "Bad", "handle": "bad", "source_slot": 0}]'

        file_values = {
            "imaging.channels": json_string,  # String - should be rejected!
        }

        ConfigRegistry.init(file_values)

        with pytest.raises(TypeError, match="got str"):
            ConfigRegistry.register(
                config_item(
                    handle="imaging.channels",
                    name="channels",
                    value_kind="object",
                    default=[],
                )
            )

    def test_object_type_rejects_string_in_mixed_config(self):
        """Even if some values are correct, string values should fail."""
        file_values = {
            "imaging.channels": [{"name": "Good", "handle": "good", "source_slot": 1}],
            "protocol.forbidden_areas": '[{"name": "Bad"}]',  # String - should fail!
        }

        ConfigRegistry.init(file_values)

        # First one succeeds
        ConfigRegistry.register(
            config_item(handle="imaging.channels", name="channels", value_kind="object", default=[]),
        )

        # Second one should fail
        with pytest.raises(TypeError, match="got str"):
            ConfigRegistry.register(
                config_item(handle="protocol.forbidden_areas", name="areas", value_kind="object", default=[]),
            )

    def test_legacy_config_file_fails_to_load(self, tmp_path):
        """Config file with legacy JSON-string format should fail to load."""
        config_file = tmp_path / "config.json"
        # Legacy format with escaped JSON strings - should be rejected
        config_data = {
            "port": 5000,
            "microscopes": [
                {
                    "system.microscope_name": "legacy-scope",
                    "system.microscope_type": "mock",
                    "camera.main.id": "cam",
                    "camera.main.driver": "galaxy",
                    "microcontroller.id": "",
                    "storage.base_image_output_dir": str(tmp_path),
                    "calibration.offset.x_mm": 0.0,
                    "calibration.offset.y_mm": 0.0,
                    "calibration.offset.z_mm": 0.0,
                    "protocol.forbidden_areas": '[{"name": "Legacy", "min_x_mm": 0, "max_x_mm": 1, "min_y_mm": 0, "max_y_mm": 1, "reason": "x"}]',
                    "laser.autofocus.available": "no",
                    "filter.wheel.available": "no",
                    "imaging.channels": '[{"name": "Legacy Ch", "handle": "leg", "source_slot": 0}]',
                    "filter.wheel.configuration": "[]"
                }
            ]
        }
        config_file.write_text(json.dumps(config_data))

        with patch.object(GlobalConfigHandler, 'home_config', return_value=config_file):
            with pytest.raises(TypeError, match="got str"):
                GlobalConfigHandler.reset("legacy-scope")

    def test_error_message_is_helpful(self):
        """Error message should tell user to use native JSON."""
        ConfigRegistry.init({"test.object": '["string value"]'})

        with pytest.raises(TypeError) as exc_info:
            ConfigRegistry.register(
                config_item(handle="test.object", name="test", value_kind="object", default=[])
            )

        error_msg = str(exc_info.value)
        assert "test.object" in error_msg
        assert "native JSON" in error_msg


class TestMicrocontrollerDriverConfig:
    """Tests for microcontroller driver configuration."""

    def setup_method(self):
        ConfigRegistry.reset()

    def teardown_method(self):
        ConfigRegistry.reset()

    def test_microcontroller_driver_is_loaded(self, temp_config_dir):
        """reset() should load microcontroller.driver from config."""
        tmp_path, config_file = temp_config_dir

        with patch.object(GlobalConfigHandler, 'home_config', return_value=config_file):
            GlobalConfigHandler.reset("test-microscope")

        driver = ConfigRegistry.get_value("microcontroller.driver")
        assert driver == "teensy"

    def test_microcontroller_driver_works_with_enum(self, temp_config_dir):
        """Microcontroller driver config should work with ConfigHandle enum."""
        from seafront.config.handles import MicrocontrollerConfig

        tmp_path, config_file = temp_config_dir

        with patch.object(GlobalConfigHandler, 'home_config', return_value=config_file):
            GlobalConfigHandler.reset("test-microscope")

        # Access via enum
        driver = ConfigRegistry.get(MicrocontrollerConfig.DRIVER).strvalue
        mc_id = ConfigRegistry.get(MicrocontrollerConfig.ID).strvalue

        assert driver == "teensy"
        assert mc_id == "test-mc-id"
