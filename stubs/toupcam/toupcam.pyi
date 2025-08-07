"""
Type stubs for toupcam library.

This provides type annotations for the toupcam library which doesn't include them by default.
Based on toupcam SDK source code analysis.
"""

from collections.abc import Callable
from typing import Any

# Constants from toupcam source
TOUPCAM_EVENT_IMAGE: int
TOUPCAM_OPTION_TRIGGER: int
TOUPCAM_OPTION_PIXEL_FORMAT: int
TOUPCAM_PIXELFORMAT_MONO8: int
TOUPCAM_PIXELFORMAT_MONO10: int
TOUPCAM_PIXELFORMAT_MONO12: int
TOUPCAM_PIXELFORMAT_MONO14: int
TOUPCAM_PIXELFORMAT_MONO16: int

class ToupcamModelV2:
    """Camera model information"""
    name: str
    flag: int
    maxspeed: int
    preview: int
    still: int
    maxfanspeed: int
    ioctrol: int
    xpixsz: float
    ypixsz: float

class ToupcamDeviceV2:
    """Represents a ToupCam device from EnumV2()"""
    displayname: str  # display name
    id: str          # unique and opaque id for Toupcam.Open()
    model: ToupcamModelV2  # ToupcamModelV2 object

class Toupcam:
    """ToupCam camera interface - based on actual source code"""

    @classmethod
    def EnumV2(cls) -> list[ToupcamDeviceV2]:
        """Enumerate all connected cameras, returns list of ToupcamDeviceV2"""
        ...

    @classmethod
    def Open(cls, camId: str | None) -> Toupcam | None:
        """
        Open camera by ID. Returns None if failed.
        camId can be None (first camera), device ID, or special formats like "sn:xxx"
        """
        ...

    @classmethod
    def OpenByIndex(cls, index: int) -> Toupcam | None:
        """Open camera by index (0=first, 1=second, etc). Returns None if failed."""
        ...

    def Close(self) -> None:
        """Close camera connection"""
        ...

    def Stop(self) -> None:
        """Stop current operation"""
        ...

    def get_Size(self) -> tuple[int, int]:
        """Get current resolution, returns (width, height)"""
        ...

    def StartPullModeWithCallback(
        self,
        fun: Callable[[int, Any], None],
        ctx: Any
    ) -> int:
        """Start pull mode with callback. Returns error code (0=success)"""
        ...

    def PullImageV3(
        self,
        pImageData: Any,
        bStill: int,
        bits: int,
        rowPitch: int,
        pInfo: Any
    ) -> int:
        """Pull image data. Returns error code (0=success)"""
        ...

    def put_Option(self, iOption: int, iValue: int) -> int:
        """Set camera option. Returns error code (0=success)"""
        ...

    def put_ExpoTime(self, Time: int) -> int:
        """Set exposure time in microseconds. Returns error code (0=success)"""
        ...

    def put_ExpoAGain(self, Gain: int) -> int:
        """Set analog gain. Returns error code (0=success)"""
        ...

    def Trigger(self, nNumber: int) -> int:
        """Trigger acquisition. Returns error code (0=success)"""
        ...
