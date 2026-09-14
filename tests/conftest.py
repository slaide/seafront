"""Shared pytest fixtures.

`temp_config` / `mock_microscope_env` mirror the local fixtures in
`test_mock_microscope.py` (which still shadow these for that file); exposing them
here lets other test modules build a `MockMicroscope` in-process too.
"""

import json
import os
from pathlib import Path
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


@pytest.fixture
def mock_core(temp_config, mock_microscope_env):
    """
    A real Core over a mock microscope, in this process.

    The whole server object graph minus HTTP: worker, device lease, image store and the frame
    sink wired to the adapter. Lets a test check what the pieces do to each other without
    paying ten seconds to boot a subprocess and talk to it over a socket.
    """
    from seafront.server.core import Core

    _tmp_path, config_file = temp_config
    scope_config = json.loads(config_file.read_text())["microscopes"][0]
    core = Core(scope_config)
    yield core
    core.close()


# --------------------------------------------------------------------------------------------
# mock S3 (MinIO in docker) for the S3 storage backend

# fixed test credentials for tests/mock-s3/docker-compose.yml; set at import time so any server
# subprocess a test starts inherits them
MINIO_ENDPOINT = "http://127.0.0.1:19000"
MINIO_BUCKET = "seafront-test"
os.environ.setdefault("AWS_ACCESS_KEY_ID", "seafront")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "seafront-test-secret")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture(scope="session")
def minio_s3():
    """endpoint URL and bucket of a MinIO container; skips when docker is not usable"""
    import shutil
    import subprocess
    import time

    compose = Path(__file__).parent / "mock-s3" / "docker-compose.yml"
    if shutil.which("docker") is None:
        pytest.skip("docker not installed")
    up = subprocess.run(["docker", "compose", "-f", str(compose), "up", "-d"], capture_output=True, text=True, timeout=300)
    if up.returncode != 0:
        pytest.skip(f"could not start mock S3: {up.stderr.strip()[-300:]}")
    import boto3
    from botocore.config import Config

    client = boto3.client("s3", endpoint_url=MINIO_ENDPOINT, config=Config(retries={"max_attempts": 1}, connect_timeout=2, read_timeout=5))
    deadline = time.time() + 60
    while True:
        try:
            client.list_buckets()
            break
        except Exception:
            if time.time() > deadline:
                subprocess.run(["docker", "compose", "-f", str(compose), "down", "-v"], capture_output=True, timeout=120)
                pytest.skip("mock S3 did not become ready")
            time.sleep(0.5)
    if MINIO_BUCKET not in {b["Name"] for b in client.list_buckets().get("Buckets", [])}:
        client.create_bucket(Bucket=MINIO_BUCKET)
    yield {"endpoint_url": MINIO_ENDPOINT, "bucket": MINIO_BUCKET, "client": client}
    subprocess.run(["docker", "compose", "-f", str(compose), "down", "-v"], capture_output=True, timeout=120)
