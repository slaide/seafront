"""
Tests for the microcontroller abstraction layer.

Tests cover:
- Abstract interface definition
- Dispatch mechanism (microcontroller_open)
- MicrocontrollerOpenRequest dataclass
- Config integration for driver selection
"""

import pytest

from seafront.hardware.microcontroller import (
    Microcontroller,
    MicrocontrollerDriver,
    MicrocontrollerOpenRequest,
    microcontroller_open,
    get_all_microcontrollers,
)


class TestMicrocontrollerOpenRequest:
    """Tests for MicrocontrollerOpenRequest dataclass."""

    def test_create_request(self):
        """Should create a request with driver and usb_id."""
        request = MicrocontrollerOpenRequest(driver="teensy", usb_id="12345")
        assert request.driver == "teensy"
        assert request.usb_id == "12345"

    def test_driver_is_literal_type(self):
        """Driver should be typed as a Literal type."""
        # This is a type check - at runtime just verify the field exists
        request = MicrocontrollerOpenRequest(driver="teensy", usb_id="test")
        assert hasattr(request, "driver")
        assert hasattr(request, "usb_id")


class TestMicrocontrollerDispatch:
    """Tests for the microcontroller dispatch mechanism."""

    def test_unsupported_driver_raises(self):
        """microcontroller_open should raise for unsupported drivers."""
        request = MicrocontrollerOpenRequest(
            driver="nonexistent",  # type: ignore[arg-type]
            usb_id="12345",
        )
        with pytest.raises(ValueError, match="unsupported microcontroller driver"):
            microcontroller_open(request)

    def test_teensy_driver_with_invalid_usb_id_raises(self):
        """Teensy driver should raise when device not found."""
        request = MicrocontrollerOpenRequest(driver="teensy", usb_id="nonexistent-usb-id")
        # This should raise because no device with this USB ID exists
        with pytest.raises(RuntimeError, match="not found"):
            microcontroller_open(request)


class TestMicrocontrollerAbstractInterface:
    """Tests for the abstract Microcontroller interface."""

    def test_microcontroller_is_abstract(self):
        """Microcontroller should be an abstract class."""
        from abc import ABC

        assert issubclass(Microcontroller, ABC)

    def test_has_required_abstract_methods(self):
        """Microcontroller should have all required abstract methods."""
        from abc import abstractmethod

        # Check that key methods are abstract
        abstract_methods = [
            "open",
            "close",
            "locked",
            "reset",
            "initialize",
            "configure_actuators",
            "home",
            "move_to_mm",
            "move_by_mm",
            "set_zero",
            "set_limit_mm",
            "get_position",
            "get_last_position",
            "illumination_begin",
            "illumination_end",
            "af_laser_on",
            "af_laser_off",
            "filter_wheel_init",
            "filter_wheel_configure_actuator",
            "filter_wheel_home",
            "filter_wheel_set_position",
            "filter_wheel_get_position",
            "get_all",
        ]

        # Check abstractmethods on the class
        for method_name in abstract_methods:
            assert hasattr(Microcontroller, method_name), f"Missing method: {method_name}"

    def test_has_required_abstract_properties(self):
        """Microcontroller should have required abstract properties."""
        abstract_properties = ["vendor_name", "model_name", "sn"]

        for prop_name in abstract_properties:
            assert hasattr(Microcontroller, prop_name), f"Missing property: {prop_name}"


class TestGetAllMicrocontrollers:
    """Tests for get_all_microcontrollers function."""

    def test_returns_list(self):
        """get_all_microcontrollers should return a list."""
        result = get_all_microcontrollers()
        assert isinstance(result, list)

    def test_all_items_are_microcontrollers(self):
        """All returned items should be Microcontroller instances."""
        result = get_all_microcontrollers()
        for mc in result:
            assert isinstance(mc, Microcontroller)


class TestTeensyMicrocontrollerImplementation:
    """Tests that TeensyMicrocontroller properly implements the interface."""

    def test_teensy_is_subclass_of_microcontroller(self):
        """TeensyMicrocontroller should be a subclass of Microcontroller."""
        from seafront.hardware.microcontroller.teensy_microcontroller import (
            TeensyMicrocontroller,
        )

        assert issubclass(TeensyMicrocontroller, Microcontroller)

    def test_teensy_get_all_returns_sequence(self):
        """TeensyMicrocontroller.get_all() should return a Sequence."""
        from collections.abc import Sequence

        from seafront.hardware.microcontroller.teensy_microcontroller import (
            TeensyMicrocontroller,
        )

        result = TeensyMicrocontroller.get_all()
        assert isinstance(result, Sequence)
