"""Shared pytest fixtures.

`temp_config` / `mock_microscope_env` mirror the local fixtures in
`test_mock_microscope.py` (which still shadow these for that file); exposing them
here lets other test modules build a `MockMicroscope` in-process too.
"""

import json
import os
from unittest.mock import patch

import pytest

from seafront.config.basics import GlobalConfigHandler
from seafront.config.registry import ConfigRegistry


@pytest.fixture
def temp_config(tmp_path):
    """Create a temporary config file for a mock microscope."""
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
    """Set up the config registry/environment for MockMicroscope creation."""
    tmp_path, config_file = temp_config

    os.environ["MOCK_NO_DELAYS"] = "1"
    ConfigRegistry.reset()

    with patch.object(GlobalConfigHandler, "home_config", return_value=config_file):
        GlobalConfigHandler.reset("test-mock-microscope")
        yield tmp_path

    ConfigRegistry.reset()
    os.environ.pop("MOCK_NO_DELAYS", None)


@pytest.fixture
def mock_scope(mock_microscope_env):
    """A ready-to-use in-process MockMicroscope."""
    from seafront.hardware.mock_microscope import MockMicroscope

    return MockMicroscope.make()
