"""
Unit tests for the config registry system.

Tests cover:
- Basic ConfigRegistry operations
- Object type conversion (native JSON <-> JSON strings)
- Persistent and object handle tracking
- File value overrides
- Integration with GlobalConfigHandler
"""

import json
import pytest

from seafront.config.registry import (
    ConfigRegistry,
    ConfigItemSpec,
    config_item,
    bool_config_item,
)


class TestConfigRegistry:
    """Tests for ConfigRegistry class."""

    def setup_method(self):
        """Reset registry before each test."""
        ConfigRegistry.reset()

    def teardown_method(self):
        """Clean up after each test."""
        ConfigRegistry.reset()

    def test_init_stores_file_values(self):
        """init() should store file values for later use."""
        file_values = {"test.handle": "file_value"}
        ConfigRegistry.init(file_values)

        assert ConfigRegistry._initialized is True
        assert ConfigRegistry._file_values == file_values

    def test_register_creates_config_item(self):
        """register() should create a ConfigItem with correct properties."""
        ConfigRegistry.init({})

        ConfigRegistry.register(
            config_item(
                handle="test.item",
                name="Test Item",
                value_kind="text",
                default="default_value",
            )
        )

        item = ConfigRegistry.get("test.item")
        assert item.handle == "test.item"
        assert item.name == "Test Item"
        assert item.value == "default_value"
        assert item.value_kind == "text"

    def test_register_uses_file_value_over_default(self):
        """File values should override defaults."""
        ConfigRegistry.init({"test.item": "from_file"})

        ConfigRegistry.register(
            config_item(
                handle="test.item",
                name="Test Item",
                value_kind="text",
                default="default_value",
            )
        )

        assert ConfigRegistry.get_value("test.item") == "from_file"

    def test_register_tracks_persistent_handles(self):
        """Persistent handles should be tracked."""
        ConfigRegistry.init({})

        ConfigRegistry.register(
            config_item(
                handle="persistent.item",
                name="Persistent",
                value_kind="text",
                default="value",
                persistent=True,
            ),
            config_item(
                handle="non.persistent",
                name="Non-Persistent",
                value_kind="text",
                default="value",
                persistent=False,
            ),
        )

        persistent = ConfigRegistry.get_persistent_handles()
        assert "persistent.item" in persistent
        assert "non.persistent" not in persistent

    def test_get_value_returns_value(self):
        """get_value() should return just the value, not the ConfigItem."""
        ConfigRegistry.init({})
        ConfigRegistry.register(
            config_item(
                handle="test.item",
                name="Test",
                value_kind="int",
                default=42,
            )
        )

        assert ConfigRegistry.get_value("test.item") == 42

    def test_get_all_returns_list(self):
        """get_all() should return all registered items as a list."""
        ConfigRegistry.init({})
        ConfigRegistry.register(
            config_item(handle="item1", name="Item 1", value_kind="text", default="a"),
            config_item(handle="item2", name="Item 2", value_kind="text", default="b"),
        )

        items = ConfigRegistry.get_all()
        assert len(items) == 2
        handles = [item.handle for item in items]
        assert "item1" in handles
        assert "item2" in handles

    def test_get_dict_returns_dict(self):
        """get_dict() should return items as a dict keyed by handle."""
        ConfigRegistry.init({})
        ConfigRegistry.register(
            config_item(handle="item1", name="Item 1", value_kind="text", default="a"),
        )

        items = ConfigRegistry.get_dict()
        assert "item1" in items
        assert items["item1"].value == "a"

    def test_set_value_updates_existing(self):
        """set_value() should update an existing item's value."""
        ConfigRegistry.init({})
        ConfigRegistry.register(
            config_item(handle="test.item", name="Test", value_kind="text", default="old"),
        )

        ConfigRegistry.set_value("test.item", "new")
        assert ConfigRegistry.get_value("test.item") == "new"

    def test_reset_clears_everything(self):
        """reset() should clear all state."""
        ConfigRegistry.init({"key": "value"})
        ConfigRegistry.register(
            config_item(
                handle="test",
                name="Test",
                value_kind="object",
                default=[],
                persistent=True,
            )
        )

        ConfigRegistry.reset()

        assert ConfigRegistry._items == {}
        assert ConfigRegistry._persistent_handles == set()
        assert ConfigRegistry._file_values == {}
        assert ConfigRegistry._initialized is False


class TestObjectType:
    """Tests for 'object' value_kind handling."""

    def setup_method(self):
        ConfigRegistry.reset()

    def teardown_method(self):
        ConfigRegistry.reset()

    def test_object_type_stores_list_as_list(self):
        """Object type should store list as actual list."""
        ConfigRegistry.init({})
        ConfigRegistry.register(
            config_item(
                handle="test.list",
                name="Test List",
                value_kind="object",
                default=[{"key": "value"}],
            )
        )

        value = ConfigRegistry.get_value("test.list")
        assert isinstance(value, list)
        assert value == [{"key": "value"}]

    def test_object_type_stores_dict_as_dict(self):
        """Object type should store dict as actual dict."""
        ConfigRegistry.init({})
        ConfigRegistry.register(
            config_item(
                handle="test.dict",
                name="Test Dict",
                value_kind="object",
                default={"nested": {"key": "value"}},
            )
        )

        value = ConfigRegistry.get_value("test.dict")
        assert isinstance(value, dict)
        assert value == {"nested": {"key": "value"}}

    def test_object_type_from_file_stays_native(self):
        """Native JSON from file should stay as native object."""
        ConfigRegistry.init({
            "test.channels": [
                {"name": "Channel 1", "slot": 1},
                {"name": "Channel 2", "slot": 2},
            ]
        })
        ConfigRegistry.register(
            config_item(
                handle="test.channels",
                name="Channels",
                value_kind="object",
                default=[],
            )
        )

        value = ConfigRegistry.get_value("test.channels")
        assert isinstance(value, list)
        assert len(value) == 2
        assert value[0]["name"] == "Channel 1"

    def test_object_type_rejects_json_string(self):
        """Object type must reject JSON strings - native objects only."""
        json_string = '[{"name": "Bad"}]'
        ConfigRegistry.init({"test.bad": json_string})

        with pytest.raises(TypeError, match="got str"):
            ConfigRegistry.register(
                config_item(
                    handle="test.bad",
                    name="Bad",
                    value_kind="object",
                    default=[],
                )
            )

    def test_object_type_stored_natively_in_config_item(self):
        """Object type should be stored natively in ConfigItem (seaconfig supports objects)."""
        ConfigRegistry.init({})
        ConfigRegistry.register(
            config_item(
                handle="test.object",
                name="Object",
                value_kind="object",
                default=[{"data": 123}],
            )
        )

        # ConfigItem stores object natively with value_kind="object"
        item = ConfigRegistry.get("test.object")
        assert item.value_kind == "object"
        assert item.value == [{"data": 123}]

        # get_value also returns the actual object
        value = ConfigRegistry.get_value("test.object")
        assert value == [{"data": 123}]


class TestBoolConfigItem:
    """Tests for bool_config_item helper."""

    def setup_method(self):
        ConfigRegistry.reset()

    def teardown_method(self):
        ConfigRegistry.reset()

    def test_bool_true_becomes_yes(self):
        """True default should become 'yes'."""
        ConfigRegistry.init({})
        ConfigRegistry.register(
            bool_config_item(
                handle="test.bool",
                name="Bool",
                default=True,
            )
        )

        assert ConfigRegistry.get_value("test.bool") == "yes"

    def test_bool_false_becomes_no(self):
        """False default should become 'no'."""
        ConfigRegistry.init({})
        ConfigRegistry.register(
            bool_config_item(
                handle="test.bool",
                name="Bool",
                default=False,
            )
        )

        assert ConfigRegistry.get_value("test.bool") == "no"

    def test_bool_has_options(self):
        """Bool config should have yes/no options."""
        ConfigRegistry.init({})
        ConfigRegistry.register(
            bool_config_item(
                handle="test.bool",
                name="Bool",
                default=True,
            )
        )

        item = ConfigRegistry.get("test.bool")
        assert item.options is not None
        option_handles = [opt.handle for opt in item.options]
        assert "yes" in option_handles
        assert "no" in option_handles


class TestConfigItemSpec:
    """Tests for ConfigItemSpec creation."""

    def test_config_item_creates_spec(self):
        """config_item() should create a ConfigItemSpec."""
        spec = config_item(
            handle="test.handle",
            name="Test Name",
            value_kind="float",
            default=3.14,
            frozen=True,
            persistent=True,
        )

        assert isinstance(spec, ConfigItemSpec)
        assert spec.handle == "test.handle"
        assert spec.name == "Test Name"
        assert spec.value_kind == "float"
        assert spec.default == 3.14
        assert spec.frozen is True
        assert spec.persistent is True

    def test_config_item_defaults(self):
        """config_item() should have correct defaults."""
        spec = config_item(
            handle="test",
            name="Test",
            value_kind="text",
            default="",
        )

        assert spec.options is None
        assert spec.frozen is False
        assert spec.persistent is False


class TestConfigHandleEnumAccess:
    """Tests for accessing config via Enum handles."""

    def setup_method(self):
        ConfigRegistry.reset()

    def teardown_method(self):
        ConfigRegistry.reset()

    def test_get_accepts_string(self):
        """get() should accept string handles."""
        ConfigRegistry.init({})
        ConfigRegistry.register(
            config_item(handle="test.item", name="Test", value_kind="text", default="value"),
        )

        item = ConfigRegistry.get("test.item")
        assert item.value == "value"

    def test_get_accepts_enum(self):
        """get() should accept ConfigHandle enum values."""
        from enum import Enum

        class TestHandle(Enum):
            ITEM = "test.item"

        ConfigRegistry.init({})
        ConfigRegistry.register(
            config_item(handle="test.item", name="Test", value_kind="text", default="enum_value"),
        )

        item = ConfigRegistry.get(TestHandle.ITEM)
        assert item.value == "enum_value"

    def test_get_value_accepts_enum(self):
        """get_value() should also accept ConfigHandle enum values."""
        from enum import Enum

        class TestHandle(Enum):
            ITEM = "test.item"

        ConfigRegistry.init({})
        ConfigRegistry.register(
            config_item(handle="test.item", name="Test", value_kind="int", default=42),
        )

        value = ConfigRegistry.get_value(TestHandle.ITEM)
        assert value == 42

    def test_real_config_handle_enum_works(self):
        """Real ConfigHandle enums from handles.py should work."""
        from seafront.config.handles import MicrocontrollerConfig

        ConfigRegistry.init({})
        ConfigRegistry.register(
            config_item(
                handle=MicrocontrollerConfig.ID.value,
                name="Microcontroller ID",
                value_kind="text",
                default="test-mc-id",
            ),
            config_item(
                handle=MicrocontrollerConfig.DRIVER.value,
                name="Microcontroller Driver",
                value_kind="text",
                default="teensy",
            ),
        )

        # Access via enum
        mc_id = ConfigRegistry.get(MicrocontrollerConfig.ID).strvalue
        mc_driver = ConfigRegistry.get(MicrocontrollerConfig.DRIVER).strvalue

        assert mc_id == "test-mc-id"
        assert mc_driver == "teensy"
