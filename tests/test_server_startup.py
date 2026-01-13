"""Tests for server startup with mock microscope."""

import json
import os
import subprocess
import time
import urllib.error
import urllib.request

import pytest


@pytest.fixture(scope="class")
def running_server(request):
    """Start server once for all tests in the class."""
    env = os.environ.copy()
    env["MOCK_NO_DELAYS"] = "1"

    proc = subprocess.Popen(
        ["uv", "run", "python", "-m", "seafront", "--microscope", "mocroscope"],
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


@pytest.mark.usefixtures("running_server")
class TestServerStartup:
    """Test that the server starts and responds correctly with mock microscope."""

    def test_root_returns_200(self):
        """Root endpoint should return 200."""
        with urllib.request.urlopen(f"{self.server_url}/", timeout=5) as response:
            assert response.status == 200

    def test_health_endpoint_returns_ok(self):
        """Health endpoint should return status ok."""
        with urllib.request.urlopen(f"{self.server_url}/api/health", timeout=5) as response:
            assert response.status == 200
            data = json.loads(response.read().decode())
            assert data.get("status") == "ok"

    def test_openapi_spec_is_available(self):
        """OpenAPI spec should be available."""
        with urllib.request.urlopen(f"{self.server_url}/openapi.json", timeout=5) as response:
            assert response.status == 200

    def test_openapi_spec_is_non_empty(self):
        """OpenAPI spec should not be empty."""
        with urllib.request.urlopen(f"{self.server_url}/openapi.json", timeout=5) as response:
            raw = response.read()
            assert len(raw) > 0, "OpenAPI spec is empty"

    def test_openapi_spec_is_valid_json(self):
        """OpenAPI spec should be valid JSON."""
        with urllib.request.urlopen(f"{self.server_url}/openapi.json", timeout=5) as response:
            raw = response.read().decode()
            data = json.loads(raw)  # Will raise if invalid
            assert isinstance(data, dict)

    def test_openapi_spec_has_version(self):
        """OpenAPI spec should have version field."""
        with urllib.request.urlopen(f"{self.server_url}/openapi.json", timeout=5) as response:
            data = json.loads(response.read().decode())
            assert "openapi" in data, "OpenAPI spec missing 'openapi' version field"
            assert data["openapi"].startswith("3."), f"Unexpected OpenAPI version: {data['openapi']}"

    def test_openapi_spec_has_paths(self):
        """OpenAPI spec should have paths defined."""
        with urllib.request.urlopen(f"{self.server_url}/openapi.json", timeout=5) as response:
            data = json.loads(response.read().decode())
            assert "paths" in data, "OpenAPI spec missing 'paths'"
            assert len(data["paths"]) > 0, "OpenAPI spec has no paths"

    def test_openapi_spec_includes_health_endpoint(self):
        """OpenAPI spec should include the health endpoint."""
        with urllib.request.urlopen(f"{self.server_url}/openapi.json", timeout=5) as response:
            data = json.loads(response.read().decode())
            assert "/api/health" in data["paths"], "OpenAPI spec missing /api/health endpoint"

    def test_docs_endpoint_available(self):
        """Swagger docs should be available."""
        with urllib.request.urlopen(f"{self.server_url}/docs", timeout=5) as response:
            assert response.status == 200
