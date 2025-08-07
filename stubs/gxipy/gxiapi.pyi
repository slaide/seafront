"""
Type stubs for gxipy.gxiapi module.

This provides type annotations for the Galaxy camera library which doesn't include them by default.
Based on runtime inspection of the actual gxiapi module.
"""

import ctypes
from collections.abc import Callable
from typing import Any

# Initialize library function
def gx_init_lib() -> None:
    """Initialize the Galaxy camera library"""
    ...

# Key constants
class GxFrameStatusList:
    SUCCESS: int
    INCOMPLETE: int
    INVALID_IMAGE_INFO: int

class GxPixelFormatEntry:
    MONO8: int
    MONO10: int
    MONO12: int
    MONO14: int
    MONO16: int
    BAYER_GR8: int
    BAYER_RG8: int
    BAYER_GB8: int
    BAYER_BG8: int
    # ... many more pixel formats

class GxStatusList:
    SUCCESS: int
    ERROR: int
    TIMEOUT: int
    OFFLINE: int
    # ... many more status codes

# Exception classes
class OffLine(Exception):
    """Camera offline exception"""
    ...

class InvalidParameter(Exception):
    """Invalid parameter exception"""
    ...

class Timeout(Exception):
    """Timeout exception"""
    ...

# Device info class
class GxDeviceBaseInfo:
    """Device information structure"""
    vendor_name: ctypes.Array[ctypes.c_char]
    model_name: ctypes.Array[ctypes.c_char]
    serial_number: ctypes.Array[ctypes.c_char]
    display_name: ctypes.Array[ctypes.c_char]
    device_id: ctypes.Array[ctypes.c_char]
    user_id: ctypes.Array[ctypes.c_char]
    access_status: int
    device_class: int
    reserved: ctypes.Array[ctypes.c_char]

# Raw image class
class RawImage:
    """Raw image data from camera"""
    def get_status(self) -> int:
        """Get image status (SUCCESS, INCOMPLETE, etc.)"""
        ...

    def get_numpy_array(self) -> Any | None:  # numpy.ndarray
        """Convert to numpy array, returns None if invalid"""
        ...

    def get_width(self) -> int:
        """Get image width in pixels"""
        ...

    def get_height(self) -> int:
        """Get image height in pixels"""
        ...

# Device Manager class
class DeviceManager:
    """Manages camera device enumeration and access"""

    def __init__(self) -> None: ...

    def update_device_list(self, timeout: int = 200) -> tuple[int, list[GxDeviceBaseInfo]]:
        """
        Update device list and return (device_count, device_info_list).
        Returns number of devices found and list of device info structures.
        """
        ...

    def update_all_device_list(self, timeout: int = 200) -> tuple[int, list[GxDeviceBaseInfo]]:
        """Update all device list including GigE cameras"""
        ...

    def get_device_number(self) -> int:
        """Get number of enumerated devices"""
        ...

    def open_device_by_index(self, index: int, access_mode: int = 3) -> Device | None:
        """Open device by index, returns Device object or None if failed"""
        ...

    def open_device_by_sn(self, sn: str, access_mode: int = 3) -> Device | None:
        """Open device by serial number"""
        ...

# Device class (camera)
class Device:
    """Galaxy camera device interface"""

    def close_device(self) -> None:
        """Close the camera device"""
        ...

    def stream_on(self) -> None:
        """Start data stream"""
        ...

    def stream_off(self) -> None:
        """Stop data stream"""
        ...

    def get_stream_channel_num(self) -> int:
        """Get number of stream channels"""
        ...

# Data Stream class
class DataStream:
    """Camera data stream interface"""

    def set_acquisition_buffer_number(self, buf_num: int) -> None:
        """Set number of acquisition buffers"""
        ...

    def register_capture_callback(self, callback_func: Callable[[RawImage], None]) -> None:
        """Register callback for captured images"""
        ...

    def unregister_capture_callback(self) -> None:
        """Unregister capture callback"""
        ...

    def flush_queue(self) -> None:
        """Flush the image queue"""
        ...

    def get_image(self, timeout: int = 1000) -> RawImage | None:
        """Get image with timeout, returns RawImage or None if timeout/error"""
        ...

# Feature classes for camera control
class IntFeature:
    """Integer feature control"""
    def get(self) -> int: ...
    def set(self, int_value: int) -> None: ...
    def get_range(self) -> tuple[int, int, int]: ...  # (min, max, increment)
    def is_readable(self) -> bool: ...
    def is_writable(self) -> bool: ...

class FloatFeature:
    """Float feature control"""
    def get(self) -> float: ...
    def set(self, float_value: float) -> None: ...
    def get_range(self) -> tuple[float, float, float]: ...  # (min, max, increment)
    def is_readable(self) -> bool: ...
    def is_writable(self) -> bool: ...

class EnumFeature:
    """Enumeration feature control"""
    def get(self) -> int: ...
    def set(self, enum_value: int) -> None: ...
    def get_range(self) -> list[int]: ...
    def is_readable(self) -> bool: ...
    def is_writable(self) -> bool: ...
