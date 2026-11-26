"""
Privileged machine config.

Server-owned handles (calibration, device identity, ...) are marked `privileged`. A request
may only write them by explicitly asserting privilege, and such a write is persisted to the
config file at once. Unprivileged requests (snap, acquisition start, protocol store, a plain
flush) leave them untouched: the server value wins. Calibration results are written by the
server itself, so a client never needs privilege for the normal calibration flow.
"""

import asyncio
import json
from unittest.mock import patch

import json5
import pytest
from seaconfig import ConfigItem

from seafront.config.basics import GlobalConfigHandler
from seafront.config.handles import LaserAutofocusConfig
from seafront.config.registry import ConfigRegistry, bool_config_item, config_item
from seafront.server import commands as cmd


def _item(handle: str, value, value_kind: str = "float") -> ConfigItem:
    return ConfigItem(name="", handle=handle, value_kind=value_kind, value=value)  # type: ignore[arg-type]


@pytest.fixture
def registry():
    ConfigRegistry.reset()
    ConfigRegistry.init({})
    ConfigRegistry.register(
        config_item(handle="cal.um_per_px", name="", value_kind="float", default=1.0, privileged=True),
        config_item(handle="cam.exposure", name="", value_kind="float", default=5.0),
        config_item(handle="cam.gain", name="", value_kind="int", default=3),
        config_item(handle="sys.name", name="", value_kind="text", default="scope", privileged=True),
        bool_config_item(handle="cal.done", name="", default=False, privileged=True),
    )
    yield ConfigRegistry
    ConfigRegistry.reset()


class TestApply:
    def test_unprivileged_request_cannot_change_privileged_handle(self, registry):
        result = registry.apply([_item("cal.um_per_px", 2.5)])
        assert registry.get_value("cal.um_per_px") == 1.0
        assert result.skipped_privileged == ["cal.um_per_px"]
        assert result.changed == []

    def test_matching_privileged_value_is_not_reported(self, registry):
        # the frontend sends its whole list on every snap; identical values are the normal case
        result = registry.apply([_item("cal.um_per_px", 1.0)])
        assert result.skipped_privileged == []

    def test_unprivileged_request_writes_ordinary_handles(self, registry):
        result = registry.apply([_item("cam.exposure", 12.0), _item("cal.um_per_px", 9.0)])
        assert registry.get_value("cam.exposure") == 12.0
        assert result.changed == ["cam.exposure"]
        assert result.skipped_privileged == ["cal.um_per_px"]

    def test_privileged_request_writes_privileged_handles(self, registry):
        result = registry.apply(
            [_item("cal.um_per_px", 2.5), _item("cal.done", "yes", "option")], privileged=True
        )
        assert registry.get_value("cal.um_per_px") == 2.5
        assert registry.get_value("cal.done") == "yes"
        assert result.changed == ["cal.um_per_px", "cal.done"]
        assert result.changed_privileged == ["cal.um_per_px", "cal.done"]

    def test_unknown_handles_are_reported_not_written(self, registry):
        result = registry.apply([_item("does.not.exist", 1.0)])
        assert result.unknown_handles == ["does.not.exist"]
        assert "does.not.exist" not in registry.get_dict()

    def test_is_privileged(self, registry):
        assert registry.is_privileged("cal.um_per_px") is True
        assert registry.is_privileged("cam.exposure") is False


class TestValidation:
    def test_int_rejects_str_and_bool(self, registry):
        assert "cam.gain" in registry.apply([_item("cam.gain", "3", "int")]).invalid
        assert registry.get_value("cam.gain") == 3
        # pydantic already coerces a JSON `true` to 1 inside ConfigItem, so the bool guard is
        # only reachable for direct callers of validate_value
        assert registry.validate_value("cam.gain", True) is not None

    def test_float_accepts_int_but_not_bool_or_str(self, registry):
        assert registry.apply([_item("cam.exposure", 7)]).invalid == {}
        assert registry.get_value("cam.exposure") == 7
        assert "cam.exposure" in registry.apply([_item("cam.exposure", "7")]).invalid
        assert registry.validate_value("cam.exposure", True) is not None

    def test_option_must_be_one_of_the_options(self, registry):
        result = registry.apply([_item("cal.done", "maybe", "option")], privileged=True)
        assert "cal.done" in result.invalid
        assert registry.get_value("cal.done") == "no"

    def test_registered_kind_is_authoritative(self, registry):
        # the request claims "text" but the handle is a float: value is validated as float
        assert "cam.exposure" in registry.apply([_item("cam.exposure", "fast", "text")]).invalid

    def test_validate_value(self, registry):
        assert registry.validate_value("nope", 1) == "unknown handle"
        assert registry.validate_value("cam.exposure", 1.5) is None
        assert registry.validate_value("sys.name", 1.5) is not None


class TestMismatches:
    def test_lists_only_differing_privileged_handles(self, registry):
        items = [
            _item("cal.um_per_px", 1.0),  # privileged, same value
            _item("sys.name", "other", "text"),  # privileged, differs
            _item("cam.exposure", 99.0),  # not privileged
            _item("unknown", 1),
        ]
        assert registry.privileged_mismatches(items) == ["sys.name"]


class TestPersistence:
    def test_privileged_override_is_written_to_config_file_at_once(self, temp_config):
        _tmp_path, config_file = temp_config
        ConfigRegistry.reset()
        with patch.object(GlobalConfigHandler, "home_config", return_value=config_file):
            GlobalConfigHandler.reset("test-mock-microscope")
            result = GlobalConfigHandler.override(
                [_item("calibration.offset.x_mm", 0.25)], privileged=True
            )
            assert result.changed_privileged == ["calibration.offset.x_mm"]
            saved = json5.load(config_file.open())["microscopes"][0]
            assert saved["calibration.offset.x_mm"] == 0.25
        ConfigRegistry.reset()

    def test_unprivileged_override_leaves_privileged_handle_and_file_alone(self, temp_config):
        _tmp_path, config_file = temp_config
        before = config_file.read_text()
        ConfigRegistry.reset()
        with patch.object(GlobalConfigHandler, "home_config", return_value=config_file):
            GlobalConfigHandler.reset("test-mock-microscope")
            result = GlobalConfigHandler.override([_item("calibration.offset.x_mm", 0.25)])
            assert result.skipped_privileged == ["calibration.offset.x_mm"]
            assert ConfigRegistry.get_value("calibration.offset.x_mm") == 0.0
        assert config_file.read_text() == before
        ConfigRegistry.reset()

    def test_dict_form_still_accepted(self, temp_config):
        _tmp_path, config_file = temp_config
        ConfigRegistry.reset()
        with patch.object(GlobalConfigHandler, "home_config", return_value=config_file):
            GlobalConfigHandler.reset("test-mock-microscope")
            item = _item("calibration.offset.y_mm", -0.5)
            GlobalConfigHandler.override({item.handle: item}, privileged=True)
            assert ConfigRegistry.get_value("calibration.offset.y_mm") == -0.5
        ConfigRegistry.reset()


@pytest.fixture
def laser_af_mock_env(temp_config, monkeypatch):
    """Like mock_microscope_env, but with the laser autofocus subsystem enabled."""
    _tmp_path, config_file = temp_config
    data = json.loads(config_file.read_text())
    data["microscopes"][0]["laser.autofocus.available"] = "yes"
    config_file.write_text(json.dumps(data))
    monkeypatch.setenv("MOCK_NO_DELAYS", "1")
    ConfigRegistry.reset()
    with patch.object(GlobalConfigHandler, "home_config", return_value=config_file):
        GlobalConfigHandler.reset("test-mock-microscope")
        yield config_file
    ConfigRegistry.reset()


class TestLaserAutofocusCalibrationIsServerOwned:
    def test_calibration_handles_are_privileged(self, laser_af_mock_env):
        for handle in (
            LaserAutofocusConfig.CALIBRATION_IS_CALIBRATED,
            LaserAutofocusConfig.CALIBRATION_X_PEAK_POS,
            LaserAutofocusConfig.CALIBRATION_UM_PER_PX,
            LaserAutofocusConfig.CALIBRATION_REF_Z_MM,
            LaserAutofocusConfig.CALIBRATION_REF_X_MM,
            LaserAutofocusConfig.CALIBRATION_REF_Y_MM,
            LaserAutofocusConfig.CALIBRATION_REF_CHANNEL,
            LaserAutofocusConfig.OFFSET_UM,
        ):
            assert ConfigRegistry.is_privileged(handle), handle
            assert handle.value in ConfigRegistry.get_persistent_handles(), handle

    def test_calibrate_writes_registry_and_config_file(self, laser_af_mock_env):
        from seafront.hardware.mock_microscope import MockMicroscope

        config_file = laser_af_mock_env
        mc = MockMicroscope.make()
        assert ConfigRegistry.get(LaserAutofocusConfig.CALIBRATION_IS_CALIBRATED).boolvalue is False

        asyncio.run(mc.execute(cmd.MoveTo(x_mm=10.0, y_mm=10.0, z_mm=2.5)))
        response = asyncio.run(mc.execute(cmd.LaserAutofocusCalibrate()))
        data = response.calibration_data

        assert ConfigRegistry.get(LaserAutofocusConfig.CALIBRATION_IS_CALIBRATED).boolvalue is True
        assert ConfigRegistry.get(LaserAutofocusConfig.CALIBRATION_UM_PER_PX).floatvalue == data.um_per_px
        assert ConfigRegistry.get(LaserAutofocusConfig.CALIBRATION_X_PEAK_POS).floatvalue == data.x_reference
        assert ConfigRegistry.get(LaserAutofocusConfig.CALIBRATION_REF_Z_MM).floatvalue == pytest.approx(2.5)

        saved = json5.load(config_file.open())["microscopes"][0]
        assert saved[LaserAutofocusConfig.CALIBRATION_IS_CALIBRATED.value] == "yes"
        assert saved[LaserAutofocusConfig.CALIBRATION_REF_Z_MM.value] == pytest.approx(2.5)

    def test_calibrate_records_the_reference_channel_and_position(self, laser_af_mock_env):
        """
        a reloaded page shows the focus reference again from these; nothing else keeps which
        channel was in focus, or where, when the laser autofocus was calibrated
        """
        from seafront.hardware.mock_microscope import MockMicroscope

        config_file = laser_af_mock_env
        mc = MockMicroscope.make()
        asyncio.run(mc.execute(cmd.MoveTo(x_mm=10.0, y_mm=12.0, z_mm=2.5)))
        asyncio.run(mc.execute(cmd.LaserAutofocusCalibrate(reference_channel="fluo488")))

        assert ConfigRegistry.get(LaserAutofocusConfig.CALIBRATION_REF_CHANNEL).strvalue == "fluo488"
        assert ConfigRegistry.get(LaserAutofocusConfig.CALIBRATION_REF_X_MM).floatvalue == pytest.approx(10.0)
        assert ConfigRegistry.get(LaserAutofocusConfig.CALIBRATION_REF_Y_MM).floatvalue == pytest.approx(12.0)

        saved = json5.load(config_file.open())["microscopes"][0]
        assert saved[LaserAutofocusConfig.CALIBRATION_REF_CHANNEL.value] == "fluo488"
        assert saved[LaserAutofocusConfig.CALIBRATION_REF_X_MM.value] == pytest.approx(10.0)
        assert saved[LaserAutofocusConfig.CALIBRATION_REF_Y_MM.value] == pytest.approx(12.0)
