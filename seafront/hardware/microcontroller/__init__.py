"""
Microcontroller hardware abstraction layer.

This package provides an abstract interface for microcontroller implementations,
allowing support for different microcontroller types/manufacturers.

Usage:
    from seafront.hardware.microcontroller import (
        Microcontroller,
        MicrocontrollerDriver,
        MicrocontrollerOpenRequest,
        microcontroller_open,
        get_all_microcontrollers,
    )

    # For Teensy-specific items, import directly from the driver module:
    from seafront.hardware.microcontroller.teensy_microcontroller import ILLUMINATION_CODE
"""

from .microcontroller import (
    Microcontroller,
    MicrocontrollerDriver,
    MicrocontrollerOpenRequest,
    microcontroller_open,
    get_all_microcontrollers,
)

__all__ = [
    "Microcontroller",
    "MicrocontrollerDriver",
    "MicrocontrollerOpenRequest",
    "microcontroller_open",
    "get_all_microcontrollers",
]
