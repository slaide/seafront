"""Tests to validate all example config files are up to standard."""

import json
from pathlib import Path

import pytest

# Fields that must be native JSON arrays/objects, not strings
OBJECT_TYPE_FIELDS = [
    "imaging.channels",
    "protocol.forbidden_areas",
    "filter.wheel.configuration",
]


def get_example_config_files() -> list[Path]:
    """Dynamically discover all example config files."""
    examples_dir = Path(__file__).parent.parent / "examples"
    return sorted(examples_dir.glob("*.json"))


def get_example_config_ids() -> list[str]:
    """Get test IDs for parametrization."""
    return [f.name for f in get_example_config_files()]


class TestExampleConfigs:
    """Validate all example config files."""

    @pytest.fixture(params=get_example_config_files(), ids=get_example_config_ids())
    def config_file(self, request: pytest.FixtureRequest) -> Path:
        """Parametrized fixture providing each example config file."""
        return request.param

    def test_is_valid_json(self, config_file: Path):
        """Example config should be valid JSON."""
        content = config_file.read_text()
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON in {config_file.name}: {e}")

    def test_has_required_structure(self, config_file: Path):
        """Example config should have required top-level structure."""
        data = json.loads(config_file.read_text())

        assert "microscopes" in data, f"{config_file.name} missing 'microscopes' key"
        assert isinstance(data["microscopes"], list), f"{config_file.name} 'microscopes' should be a list"
        assert len(data["microscopes"]) > 0, f"{config_file.name} 'microscopes' should not be empty"

    def test_microscope_has_required_fields(self, config_file: Path):
        """Each microscope entry should have required fields."""
        data = json.loads(config_file.read_text())
        required_fields = [
            "system.microscope_name",
            "system.microscope_type",
            "camera.main.id",
            "camera.main.driver",
        ]

        for i, microscope in enumerate(data["microscopes"]):
            for field in required_fields:
                assert field in microscope, (
                    f"{config_file.name} microscope[{i}] missing required field '{field}'"
                )

    def test_object_fields_are_native_json(self, config_file: Path):
        """Object-type fields must be native JSON arrays/dicts, not strings."""
        data = json.loads(config_file.read_text())

        for i, microscope in enumerate(data["microscopes"]):
            for field in OBJECT_TYPE_FIELDS:
                if field not in microscope:
                    continue

                value = microscope[field]

                # Must NOT be a string (which would indicate JSON-encoded data)
                assert not isinstance(value, str), (
                    f"{config_file.name} microscope[{i}] field '{field}' is a JSON string. "
                    f"Object-type fields must be native JSON arrays/dicts, not escaped strings. "
                    f"Got: {value[:100]}..." if len(str(value)) > 100 else f"Got: {value}"
                )

                # Must be a list or dict
                assert isinstance(value, (list, dict)), (
                    f"{config_file.name} microscope[{i}] field '{field}' should be list or dict, "
                    f"got {type(value).__name__}"
                )

    def test_channels_have_required_fields(self, config_file: Path):
        """Each channel in imaging.channels should have required fields."""
        data = json.loads(config_file.read_text())
        required_channel_fields = ["name", "handle", "source_slot"]

        for i, microscope in enumerate(data["microscopes"]):
            channels = microscope.get("imaging.channels", [])
            for j, channel in enumerate(channels):
                for field in required_channel_fields:
                    assert field in channel, (
                        f"{config_file.name} microscope[{i}] channel[{j}] missing '{field}'"
                    )

    def test_forbidden_areas_have_required_fields(self, config_file: Path):
        """Each forbidden area should have required fields."""
        data = json.loads(config_file.read_text())
        required_area_fields = ["name", "min_x_mm", "max_x_mm", "min_y_mm", "max_y_mm"]

        for i, microscope in enumerate(data["microscopes"]):
            areas = microscope.get("protocol.forbidden_areas", [])
            for j, area in enumerate(areas):
                for field in required_area_fields:
                    assert field in area, (
                        f"{config_file.name} microscope[{i}] forbidden_area[{j}] missing '{field}'"
                    )


class TestExampleConfigsDiscovery:
    """Test that we actually find example configs."""

    def test_example_configs_exist(self):
        """There should be at least one example config file."""
        configs = get_example_config_files()
        assert len(configs) > 0, "No example config files found in examples/ directory"

    def test_mock_config_exists(self):
        """mock_config.json should exist for development testing."""
        configs = get_example_config_files()
        names = [c.name for c in configs]
        assert "mock_config.json" in names, "mock_config.json should exist for development"
