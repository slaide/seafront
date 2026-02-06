import json
from pathlib import Path

import pytest

from scripts.generate_default_protocol import generate_default_protocol


def _read_protocol(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def test_generate_default_protocol_uses_first_filter_from_direct_config(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    microscope_config = {
        "system.microscope_name": "test-scope",
        "filter.wheel.available": "yes",
        "imaging.channels": [
            {"name": "Fluorescence 405", "handle": "fluo405", "source_slot": 11},
            {"name": "BF Full", "handle": "bfledfull", "source_slot": 0},
        ],
        "filter.wheel.configuration": [
            {"name": "DAPI", "handle": "dapi", "slot": 1},
            {"name": "FITC", "handle": "fitc", "slot": 2},
        ],
    }

    output_file = generate_default_protocol(
        microscope_name="test-scope",
        microscope_config=microscope_config,
    )

    protocol = _read_protocol(output_file)
    channels = protocol["channels"]
    assert len(channels) == 2
    assert all(ch["filter_handle"] == "dapi" for ch in channels)


def test_generate_default_protocol_reads_custom_config_file(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    config_dir = tmp_path / "seafront"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"

    config_data = {
        "port": 5000,
        "microscopes": [
            {
                "system.microscope_name": "file-scope",
                "system.microscope_type": "mock",
                "camera.main.id": "cam",
                "camera.main.driver": "galaxy",
                "microcontroller.id": "mc",
                "microcontroller.driver": "teensy",
                "storage.base_image_output_dir": str(tmp_path / "images"),
                "calibration.offset.x_mm": 0.0,
                "calibration.offset.y_mm": 0.0,
                "calibration.offset.z_mm": 0.0,
                "laser.autofocus.available": "no",
                "filter.wheel.available": "yes",
                "filter.wheel.configuration": [
                    {"name": "Cy5", "handle": "cy5", "slot": 3}
                ],
                "imaging.channels": [
                    {"name": "Fluorescence 638", "handle": "fluo638", "source_slot": 13}
                ],
                "protocol.forbidden_areas": [],
            }
        ],
    }
    config_path.write_text(json.dumps(config_data, indent=2))

    output_file = generate_default_protocol(microscope_name="file-scope")
    protocol = _read_protocol(output_file)
    assert protocol["channels"][0]["filter_handle"] == "cy5"


@pytest.mark.parametrize(
    "example_file,expected_filter_handle",
    [
        ("mock_config.json", None),
        ("squid-v1.json", None),
        ("squid-v2.json", None),
        ("squid-v3.json", "em_multiband"),
        ("squid-v4.json", "em_multiband"),
    ],
)
def test_generate_default_protocol_matches_example_files(
    tmp_path,
    monkeypatch,
    example_file: str,
    expected_filter_handle: str | None,
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    repo_root = Path(__file__).resolve().parents[1]
    example_path = repo_root / "examples" / example_file
    example_data = json.loads(example_path.read_text())
    microscope_config = example_data["microscopes"][0]
    microscope_name = microscope_config["system.microscope_name"]

    output_file = generate_default_protocol(
        microscope_name=microscope_name,
        microscope_config=microscope_config,
    )
    protocol = _read_protocol(output_file)

    assert len(protocol["channels"]) > 0
    assert all(ch["filter_handle"] == expected_filter_handle for ch in protocol["channels"])
