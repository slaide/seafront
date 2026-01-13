"""
Decentralized config registration system.

Components define their own config items with defaults in their own files.
The startup config file can override any of these values.

Usage in a component file (e.g., kiri_camera.py):

    from seafront.config.registry import ConfigRegistry, config_item
    from seafront.config.handles import CameraConfig

    # Get a config item using the handle enum directly
    camera_id = ConfigRegistry.get(CameraConfig.MAIN_ID).strvalue

    # Register config items - values from config file automatically override defaults
    ConfigRegistry.register(
        config_item(
            handle="camera.kiri.exposure_compensation",
            name="Kiri exposure compensation",
            value_kind="float",
            default=0.0,
            persistent=True,  # will be saved to config file
        ),
    )
"""

from __future__ import annotations

import typing as tp
from enum import Enum

from seaconfig import ConfigItem, ConfigItemOption

# Value kinds supported by the config system
ValueKind = tp.Literal["text", "int", "float", "option", "action", "object"]


class ConfigRegistry:
    """
    Central registry for config items defined across the codebase.

    Components register their config items here. The registry handles:
    - Storing config items with their defaults
    - Overlaying values from the startup config file
    - Tracking which handles should be persisted
    """

    _items: dict[str, ConfigItem] = {}
    _persistent_handles: set[str] = set()
    _file_values: dict[str, tp.Any] = {}
    _initialized: bool = False

    @classmethod
    def init(cls, file_values: dict[str, tp.Any]) -> None:
        """
        Initialize the registry with values from the config file.
        Must be called before any components register their config items.
        """
        cls._file_values = file_values
        cls._initialized = True

    @classmethod
    def register(cls, *items: "ConfigItemSpec") -> None:
        """
        Register one or more config items.

        If the config file has a value for the handle, it overrides the default.
        """
        for spec in items:
            # Get value from file or use default
            value = cls._file_values.get(spec.handle, spec.default)

            # For "object" type, validate it's actually a dict or list
            if spec.value_kind == "object":
                if not isinstance(value, (dict, list)):
                    raise TypeError(
                        f"Config item '{spec.handle}' has value_kind='object' but got {type(value).__name__}. "
                        f"Object types require a dict or list, not a string. "
                        f"Update your config file to use native JSON instead of a JSON string."
                    )

            # Create the ConfigItem
            item = ConfigItem(
                handle=spec.handle,
                name=spec.name,
                value_kind=spec.value_kind,
                value=value,
                options=spec.options,
                frozen=spec.frozen,
            )

            cls._items[spec.handle] = item

            if spec.persistent:
                cls._persistent_handles.add(spec.handle)

    @classmethod
    def get(cls, handle: str | Enum) -> ConfigItem:
        """Get a config item by handle (string or ConfigHandle enum)."""
        key = handle.value if isinstance(handle, Enum) else handle
        return cls._items[key]

    @classmethod
    def get_value(cls, handle: str | Enum) -> tp.Any:
        """Get just the value of a config item."""
        key = handle.value if isinstance(handle, Enum) else handle
        return cls._items[key].value

    @classmethod
    def get_all(cls) -> list[ConfigItem]:
        """Get all registered config items."""
        return list(cls._items.values())

    @classmethod
    def get_dict(cls) -> dict[str, ConfigItem]:
        """Get all config items as a dict."""
        return dict(cls._items)

    @classmethod
    def get_persistent_handles(cls) -> set[str]:
        """Get handles that should be persisted to config file."""
        return set(cls._persistent_handles)

    @classmethod
    def set_value(cls, handle: str, value: tp.Any) -> None:
        """Update a config item's value at runtime."""
        if handle in cls._items:
            cls._items[handle].value = value

    @classmethod
    def reset(cls) -> None:
        """Clear all registered items. Used for testing."""
        cls._items.clear()
        cls._persistent_handles.clear()
        cls._file_values.clear()
        cls._initialized = False


class ConfigItemSpec(tp.NamedTuple):
    """Specification for a config item to be registered."""
    handle: str
    name: str
    value_kind: ValueKind
    default: tp.Any
    options: list[ConfigItemOption] | None = None
    frozen: bool = False
    persistent: bool = False


def config_item(
    handle: str,
    name: str,
    value_kind: ValueKind,
    default: tp.Any,
    options: list[ConfigItemOption] | None = None,
    frozen: bool = False,
    persistent: bool = False,
) -> ConfigItemSpec:
    """
    Create a config item specification for registration.

    Args:
        handle: Unique identifier (e.g., "camera.kiri.exposure_compensation")
        name: Human-readable name for UI
        value_kind: Type of value
        default: Default value if not in config file
        options: For "option" type, list of valid options
        frozen: If True, cannot be changed via UI
        persistent: If True, will be saved to config file on store()
    """
    return ConfigItemSpec(
        handle=handle,
        name=name,
        value_kind=value_kind,
        default=default,
        options=options,
        frozen=frozen,
        persistent=persistent,
    )


def bool_config_item(
    handle: str,
    name: str,
    default: bool,
    frozen: bool = False,
    persistent: bool = False,
) -> ConfigItemSpec:
    """Convenience function for boolean config items (yes/no options)."""
    return ConfigItemSpec(
        handle=handle,
        name=name,
        value_kind="option",
        default="yes" if default else "no",
        options=ConfigItemOption.get_bool_options(),
        frozen=frozen,
        persistent=persistent,
    )
