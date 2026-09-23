"""
Every change to a config value after startup is in the log, with the old and the new value.

- a runtime write (set_value, a request's machine config, a privileged change) logs one line
  per changed handle; writing the value it already has logs nothing
- long values (channel lists) are cut short in the line
- a value the camera does not support is moved to one it does, and the log says which, why,
  and what was configured
- unknown handles in a request are a warning: a client sending them is out of date
"""

import pytest
from seaconfig import ConfigItem

from seafront.config.basics import GlobalConfigHandler
from seafront.config.handles import CameraConfig
from seafront.config.registry import ConfigRegistry, config_item, restrict_to_options
from seafront.logger import logger


@pytest.fixture
def log_lines():
    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(m.record["message"]), level="INFO")
    yield lines
    logger.remove(sink)


@pytest.fixture
def registry():
    ConfigRegistry.reset()
    ConfigRegistry.init({})
    ConfigRegistry.register(
        config_item(handle="test.exposure", name="exposure", value_kind="float", default=10.0),
        config_item(handle="test.blob", name="blob", value_kind="object", default=[]),
        config_item(
            handle="test.secret", name="secret", value_kind="text", default="a", privileged=True
        ),
    )
    yield
    ConfigRegistry.reset()


def config_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if line.startswith("config - ")]


def request_item(handle: str, value, kind="float") -> ConfigItem:
    return ConfigItem(handle=handle, name=handle, value_kind=kind, value=value)


def test_set_value_logs_old_and_new(registry, log_lines):
    ConfigRegistry.set_value("test.exposure", 20.0)
    assert config_lines(log_lines) == ["config - test.exposure: 10.0 -> 20.0"]


def test_writing_the_same_value_logs_nothing(registry, log_lines):
    ConfigRegistry.set_value("test.exposure", 10.0)
    ConfigRegistry.apply([request_item("test.exposure", 10.0)])
    assert config_lines(log_lines) == []


def test_a_request_logs_each_change(registry, log_lines):
    ConfigRegistry.apply([request_item("test.exposure", 5.0)])
    assert config_lines(log_lines) == ["config - test.exposure: 10.0 -> 5.0"]


def test_a_privileged_change_is_logged_with_its_values(registry, log_lines, monkeypatch):
    monkeypatch.setattr(GlobalConfigHandler, "store", staticmethod(lambda: None))
    GlobalConfigHandler.override([request_item("test.secret", "b", kind="text")], privileged=True)
    assert "config - test.secret: 'a' -> 'b'" in config_lines(log_lines)


def test_a_long_value_is_cut_short(registry, log_lines):
    ConfigRegistry.set_value("test.blob", [{"handle": f"channel{i}"} for i in range(100)])
    [line] = config_lines(log_lines)
    assert line.startswith("config - test.blob: [] -> [{'handle': 'channel0'}")
    assert len(line) < 300 and "chars)" in line


def test_unknown_handles_in_a_request_are_a_warning(registry):
    warnings: list[str] = []
    sink = logger.add(lambda m: warnings.append(m.record["message"]), level="WARNING")
    try:
        GlobalConfigHandler.override([request_item("test.gone", 1.0)])
    finally:
        logger.remove(sink)
    assert warnings == ["ignored unknown config handles: ['test.gone']"]


def test_an_unsupported_value_falls_back_and_says_so(log_lines):
    item = ConfigItem(
        handle=CameraConfig.MAIN_PIXEL_FORMAT.value, name="pixel format",
        value_kind="option", value="mono12",
    )
    restrict_to_options(item, ["mono8", "mono16"], "the main camera")
    assert item.value == "mono8"
    assert [o.handle for o in item.options or []] == ["mono16", "mono8"]
    assert (
        "config - camera.main.pixel_format: the main camera does not support 'mono12' "
        "(supports ['mono16', 'mono8']); using 'mono8'"
    ) in log_lines


def test_a_supported_value_is_kept_quietly(log_lines):
    item = ConfigItem(handle="x", name="x", value_kind="option", value="mono16")
    restrict_to_options(item, ["mono8", "mono16"], "the main camera")
    assert item.value == "mono16"
    assert config_lines(log_lines) == []
