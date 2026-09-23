"""Tests for server startup with mock microscope."""

import json
import time
import urllib.error
import urllib.request

import pytest

from tests.mock_server import MOCK, mock_server


@pytest.fixture(scope="class")
def running_server(request, tmp_path_factory):
    """Start server once for all tests in the class."""
    with mock_server(tmp_path_factory) as server:
        request.cls.server_url = server.url
        # tests that need to see what startup did read the server's log back
        request.cls.server_log = server.log_path
        yield server.proc


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


@pytest.mark.usefixtures("running_server")
class TestStartupConnectsTheHardware:
    """
    The server connects its own microscope at startup, without a device lease.

    This used to go over HTTP to itself, which stopped working the day hardware routes
    started requiring the lease: the request came back 423, so the hardware was never
    connected, the "another process owns this device" shutdown never fired (423 is not the
    503 it looks for), and the hardware-limits warm-up - nested inside the success branch -
    never ran either. None of it was visible except as a warning in the log.
    """

    def _log(self) -> str:
        # startup runs on its own thread; give it a moment to finish writing
        deadline = time.time() + 20
        while time.time() < deadline:
            text = self.server_log.read_text()
            if "hardware connection established" in text or "hardware connection failed" in text:
                return text
            time.sleep(0.25)
        return self.server_log.read_text()

    def test_the_hardware_is_connected_at_startup(self):
        assert "hardware connection established" in self._log()

    def test_the_startup_connect_is_not_refused(self):
        """the regression itself: a 423 on the way in, and nobody the wiser"""
        log = self._log()
        # "423" on its own also matches a millisecond in a timestamp
        assert "423 Locked" not in log, f"something was refused during startup:\n{log[-2000:]}"
        assert "hardware connection failed" not in log
        assert "failed to establish hardware connection" not in log

    def test_the_hardware_limits_cache_is_warmed(self):
        """it lives in the success branch, so it is also what proves that branch ran"""
        assert "hardware limits cache initialized" in self._log()

    def test_nobody_holds_a_lease_afterwards(self):
        """
        connecting needs no lease and must not leave one behind

        If startup took a lease to do its work, the first human to touch the microscope
        would be locked out by their own server.
        """
        self._log()
        body = json.dumps({"target_device": MOCK}).encode()
        req = urllib.request.Request(
            f"{self.server_url}/api/lock/status",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            assert json.loads(response.read().decode())["held"] is False
