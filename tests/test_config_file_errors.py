"""
A config file that cannot be used is refused at startup, with every problem in it named by key.

- JSON5 that does not parse says where
- a value of the wrong kind, an option that is not one of the choices, and a structured value
  (channels, filters, forbidden areas) that does not fit its shape are each a line naming
  the key; all of them are listed at once, not one per restart
- options the camera fills in when it connects (pixel formats) are not refused before it has
- the server prints that list and exits 1, with no traceback
"""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from seafront.config.basics import ConfigFileError, GlobalConfigHandler
from seafront.config.registry import ConfigRegistry


@pytest.fixture
def base_config(temp_config) -> dict:
    _tmp_path, config_file = temp_config
    return json.loads(config_file.read_text())


def problems_for(tmp_path: Path, config: dict | str) -> list[str]:
    "the problems reset() finds in `config` (a dict, or raw file text); [] if it accepts it"
    path = tmp_path / "config.json"
    path.write_text(config if isinstance(config, str) else json.dumps(config))
    ConfigRegistry.reset()
    try:
        with patch.object(GlobalConfigHandler, "home_config", return_value=path):
            GlobalConfigHandler.reset("test-mock-microscope")
    except ConfigFileError as e:
        assert e.path == path
        return e.problems
    finally:
        ConfigRegistry.reset()
    return []


def with_values(config: dict, **values) -> dict:
    changed = copy.deepcopy(config)
    changed["microscopes"][0].update(values)
    return changed


def test_the_test_config_is_accepted(tmp_path, base_config):
    assert problems_for(tmp_path, base_config) == []


def test_unparseable_json5_says_where(tmp_path):
    [problem] = problems_for(tmp_path, '{\n  "port": 5000,\n  "microscopes": [ }\n')
    assert problem.startswith("not valid JSON5: ")
    assert "3" in problem, "the line it failed on"


def test_a_malformed_top_level_names_the_field(tmp_path):
    assert problems_for(tmp_path, {"port": "five thousand", "microscopes": []}) == [
        "port: Input should be a valid integer, unable to parse string as an integer"
    ]


def test_a_wrong_kind_names_the_key(tmp_path, base_config):
    config = with_values(base_config, **{"camera.main.image.width_px": "wide"})
    assert problems_for(tmp_path, config) == ["camera.main.image.width_px: expected int, got str"]


def test_an_unknown_option_lists_the_choices(tmp_path, base_config):
    config = with_values(base_config, **{"storage.backend": "tape"})
    [problem] = problems_for(tmp_path, config)
    assert problem == "storage.backend: 'tape' is not one of ['disk', 's3']"


def test_an_object_written_as_a_string_says_how_to_write_it(tmp_path, base_config):
    config = with_values(base_config, **{"protocol.forbidden_areas": "[]"})
    [problem] = problems_for(tmp_path, config)
    assert problem.startswith("protocol.forbidden_areas: ") and "native JSON" in problem


def test_a_channel_missing_a_field_names_the_channel_and_field(tmp_path, base_config):
    channels = copy.deepcopy(base_config["microscopes"][0]["imaging.channels"])
    del channels[1]["source_slot"]
    config = with_values(base_config, **{"imaging.channels": channels})
    assert problems_for(tmp_path, config) == ["imaging.channels[1].source_slot: Field required"]


def test_a_channel_with_inconsistent_calibration_is_refused(tmp_path, base_config):
    channels = copy.deepcopy(base_config["microscopes"][0]["imaging.channels"])
    channels[0]["use_power_calibration"] = True
    config = with_values(base_config, **{"imaging.channels": channels})
    [problem] = problems_for(tmp_path, config)
    assert problem.startswith("imaging.channels[0]: ") and "power_calibration" in problem


def test_a_broken_forbidden_area_is_refused(tmp_path, base_config):
    areas = [{"name": "clamp", "min_x_mm": 10.0, "max_x_mm": 20.0, "min_y_mm": 30.0}]
    config = with_values(base_config, **{"protocol.forbidden_areas": areas})
    [problem] = problems_for(tmp_path, config)
    assert "protocol.forbidden_areas" in problem and "max_y_mm" in problem


def test_every_problem_is_listed_at_once(tmp_path, base_config):
    config = with_values(
        base_config,
        **{"camera.main.image.width_px": "wide", "storage.backend": "tape", "protocol.forbidden_areas": "[]"},
    )
    problems = problems_for(tmp_path, config)
    assert {p.split(":")[0] for p in problems} == {
        "camera.main.image.width_px", "storage.backend", "protocol.forbidden_areas",
    }


def test_a_pixel_format_is_left_for_the_camera_to_judge(tmp_path, base_config):
    "mono10 is not among the placeholder options, and a Galaxy camera may well support it"
    config = with_values(base_config, **{"camera.main.pixel_format": "mono10"})
    assert problems_for(tmp_path, config) == []


def test_a_runtime_request_is_still_checked_against_the_options(mock_microscope_env):
    "the file check leaves hardware options alone; a request is checked against what is there"
    assert ConfigRegistry.validate_value("storage.backend", "tape") is not None


def test_the_server_exits_1_with_the_list_and_no_traceback(tmp_path, base_config):
    home = tmp_path / "home"
    home.mkdir()
    config = with_values(base_config, **{"storage.backend": "tape"})
    (home / "config.json").write_text(json.dumps(config))
    proc = subprocess.run(
        [sys.executable, "-m", "seafront", "--microscope", "test-mock-microscope"],
        env={**os.environ, "SEAFRONT_HOME": str(home), "MOCK_NO_DELAYS": "1"},
        capture_output=True, text=True, timeout=60,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 1, output
    assert "cannot be used" in output and "storage.backend: 'tape' is not one of" in output
    assert "Traceback" not in output
