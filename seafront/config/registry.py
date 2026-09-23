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

import contextlib
import typing as tp
from enum import Enum

# loguru directly: seafront.logger imports the config module this one is part of
from loguru import logger
from pydantic import BaseModel, Field
from seaconfig import ConfigItem, ConfigItemOption

# Value kinds supported by the config system
ValueKind = tp.Literal["text", "int", "float", "option", "action", "object"]


class OverrideResult(BaseModel):
    """Outcome of writing request-provided values into the registry (see ConfigRegistry.apply)."""

    changed: list[str] = Field(default_factory=list)
    "handles whose value was written and actually differs from before"
    changed_privileged: list[str] = Field(default_factory=list)
    "subset of `changed` that is privileged (only non-empty for privileged requests)"
    skipped_privileged: list[str] = Field(default_factory=list)
    "privileged handles the request tried to change without privilege; left untouched"
    unknown_handles: list[str] = Field(default_factory=list)
    "handles the server does not know; ignored"
    invalid: dict[str, str] = Field(default_factory=dict)
    "handle -> reason, for values rejected by type/option validation; left untouched"


class ConfigValueError(TypeError):
    "config file values of the wrong kind, each as `handle: why`"

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


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
    _collecting: list[str] | None = None
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
    def register(cls, *items: ConfigItemSpec) -> None:
        """
        Register one or more config items.

        If the config file has a value for the handle, it overrides the default. A file value
        of the wrong kind is refused with ConfigValueError, once the whole group is registered
        and naming every bad value in it. Inside `collect_file_problems()` the problems go to
        its list instead, and registration carries on; the items are registered either way,
        with their defaults in place of the refused values.
        """
        problems: list[str] = []
        for spec in items:
            value = spec.default
            if spec.handle in cls._file_values:
                # options filled in from the hardware are checked when it connects
                options = None if spec.options_from_hardware else spec.options
                reason = _check_value(spec.value_kind, options, cls._file_values[spec.handle])
                if reason is None:
                    value = cls._file_values[spec.handle]
                else:
                    problems.append(f"{spec.handle}: {reason}")

            # Create the ConfigItem
            item = ConfigItem(
                handle=spec.handle,
                name=spec.name,
                value_kind=spec.value_kind,
                value=value,
                options=spec.options,
                privileged=spec.privileged,
            )

            cls._items[spec.handle] = item

            if spec.persistent:
                cls._persistent_handles.add(spec.handle)

        if problems:
            if cls._collecting is None:
                raise ConfigValueError(problems)
            cls._collecting.extend(problems)

    @classmethod
    @contextlib.contextmanager
    def collect_file_problems(cls) -> tp.Iterator[list[str]]:
        """
        register() adds bad file values to the yielded list instead of raising, so a file
        registered in many groups has all of its problems found in one pass
        """
        problems: list[str] = []
        cls._collecting = problems
        try:
            yield problems
        finally:
            cls._collecting = None

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
    def set_value(cls, handle: str | Enum, value: tp.Any) -> None:
        """
        Update a config item's value at runtime (no validation, no privilege check).

        Takes a string or a ConfigHandle the same way `get`/`get_value`/`is_privileged` do,
        and raises on a handle nobody registered. It used to take only `str` and silently do
        nothing otherwise, which meant passing one of the enums the rest of the codebase is
        written in - or mistyping a handle - wrote nothing and said nothing.
        """
        key = handle.value if isinstance(handle, Enum) else handle
        if key not in cls._items:
            raise KeyError(f"no config item registered under {key!r}")
        cls._write(cls._items[key], value)

    @staticmethod
    def _write(item: ConfigItem, value: tp.Any) -> None:
        """
        the one place a registered value changes after startup, so the log has every change

        A run that behaves differently from the one before it can then be traced to the
        setting that changed in between, and to what it was before.
        """
        if item.value != value:
            logger.info(f"config - {item.handle}: {_shown(item.value)} -> {_shown(value)}")
        item.value = value

    @classmethod
    def is_privileged(cls, handle: str | Enum) -> bool:
        key = handle.value if isinstance(handle, Enum) else handle
        return cls._items[key].privileged

    @classmethod
    def validate_value(cls, handle: str, value: tp.Any) -> str | None:
        """
        Check that `value` is acceptable for the registered item `handle`.

        Returns None when valid, otherwise a short reason. The registered item's
        value_kind is authoritative (the request's value_kind is ignored).
        """
        item = cls._items.get(handle)
        if item is None:
            return "unknown handle"
        return _check_value(item.value_kind, item.options, value)

    @classmethod
    def apply(cls, items: tp.Iterable[ConfigItem], *, privileged: bool = False) -> OverrideResult:
        """
        Write request-provided values into the registry.

        Privileged handles are only written when `privileged` is set; otherwise they
        are left untouched (the server value wins) and reported in `skipped_privileged`
        when the request actually tried to change them. Unknown handles and values that
        fail validation are never written.
        """
        result = OverrideResult()
        for item in items:
            current = cls._items.get(item.handle)
            if current is None:
                result.unknown_handles.append(item.handle)
                continue
            reason = cls.validate_value(item.handle, item.value)
            if reason is not None:
                result.invalid[item.handle] = reason
                continue
            differs = current.value != item.value
            if current.privileged and not privileged:
                if differs:
                    result.skipped_privileged.append(item.handle)
                continue
            if differs:
                result.changed.append(item.handle)
                if current.privileged:
                    result.changed_privileged.append(item.handle)
            cls._write(current, item.value)
        return result

    @classmethod
    def privileged_mismatches(cls, items: tp.Iterable[ConfigItem]) -> list[str]:
        """Privileged handles whose provided value differs from the registry value."""
        return [
            item.handle
            for item in items
            if (current := cls._items.get(item.handle)) is not None
            and current.privileged
            and current.value != item.value
        ]

    @classmethod
    def reset(cls) -> None:
        """Clear all registered items. Used for testing."""
        cls._items.clear()
        cls._persistent_handles.clear()
        cls._file_values.clear()
        cls._initialized = False


def _check_value(
    kind: str, options: list[ConfigItemOption] | None, value: tp.Any
) -> str | None:
    "None when `value` is acceptable for an item of this kind and these options, else why not"
    # bool is a subclass of int; never accept it for numeric kinds
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            return f"expected int, got {type(value).__name__}"
    elif kind == "float":
        if isinstance(value, bool) or not isinstance(value, int | float):
            return f"expected float, got {type(value).__name__}"
    elif kind == "text" or kind == "action":
        if not isinstance(value, str):
            return f"expected str, got {type(value).__name__}"
    elif kind == "option":
        if not isinstance(value, str):
            return f"expected option handle (str), got {type(value).__name__}"
        if options:
            allowed = [o.handle for o in options]
            if value not in allowed:
                return f"{value!r} is not one of {allowed}"
    elif kind == "object":
        if isinstance(value, str):
            return "expected a JSON object or list, got str: write it as native JSON, not as a JSON string"
        if not isinstance(value, dict | list):
            return f"expected dict or list, got {type(value).__name__}"
    return None


def restrict_to_options(item: ConfigItem, supported: list[str], device: str) -> None:
    """
    offer only the option handles `device` supports for `item`, and move a value it does not
    support onto the first one it does, saying so: the operator configured the other value, and
    otherwise nothing tells them it is not the one in use
    """
    item.options = [ConfigItemOption(name=fmt.capitalize(), handle=fmt) for fmt in sorted(supported)]
    if item.value not in supported:
        logger.warning(
            f"config - {item.handle}: {device} does not support {item.value!r} "
            f"(supports {sorted(supported)}); using {supported[0]!r}"
        )
        ConfigRegistry._write(item, supported[0])


def _shown(value: tp.Any, limit: int = 200) -> str:
    "a config value for a log line; object values (channel lists and such) can be long"
    text = repr(value)
    return text if len(text) <= limit else f"{text[:limit]}... ({len(text)} chars)"


class ConfigItemSpec(tp.NamedTuple):
    """Specification for a config item to be registered."""
    handle: str
    name: str
    value_kind: ValueKind
    default: tp.Any
    options: list[ConfigItemOption] | None = None
    privileged: bool = False
    persistent: bool = False
    options_from_hardware: bool = False
    "`options` is a placeholder until the device connects and says what it supports"


def config_item(
    handle: str,
    name: str,
    value_kind: ValueKind,
    default: tp.Any,
    options: list[ConfigItemOption] | None = None,
    privileged: bool = False,
    persistent: bool = False,
    options_from_hardware: bool = False,
) -> ConfigItemSpec:
    """
    Create a config item specification for registration.

    Args:
        handle: Unique identifier (e.g., "camera.kiri.exposure_compensation")
        name: Human-readable name for UI
        value_kind: Type of value
        default: Default value if not in config file
        options: For "option" type, list of valid options
        privileged: If True, the value is server-owned (calibration, device identity, ...):
            a request may only overwrite it by explicitly asserting privilege, and such
            a write is persisted at once. Changing one may require a server restart.
        persistent: If True, will be saved to config file on store()
        options_from_hardware: If True, `options` are replaced by what the device supports
            once it connects, so a file value outside them is not refused at startup
    """
    return ConfigItemSpec(
        handle=handle,
        name=name,
        value_kind=value_kind,
        default=default,
        options=options,
        privileged=privileged,
        persistent=persistent,
        options_from_hardware=options_from_hardware,
    )


def bool_config_item(
    handle: str,
    name: str,
    default: bool,
    privileged: bool = False,
    persistent: bool = False,
) -> ConfigItemSpec:
    """Convenience function for boolean config items (yes/no options)."""
    return ConfigItemSpec(
        handle=handle,
        name=name,
        value_kind="option",
        default="yes" if default else "no",
        options=ConfigItemOption.get_bool_options(),
        privileged=privileged,
        persistent=persistent,
    )
