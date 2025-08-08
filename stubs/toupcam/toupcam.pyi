"""
Type stubs for toupcam library.

This provides type annotations for the toupcam library which doesn't include them by default.
Based on toupcam SDK documentation v59.29030.20250722.
"""

from collections.abc import Callable
from typing import Any, Optional
import ctypes

# Event constants
TOUPCAM_EVENT_IMAGE: int
TOUPCAM_EVENT_STILLIMAGE: int
TOUPCAM_EVENT_EXPOSURE: int
TOUPCAM_EVENT_TEMPTINT: int
TOUPCAM_EVENT_WBGAIN: int
TOUPCAM_EVENT_ERROR: int
TOUPCAM_EVENT_DISCONNECTED: int
TOUPCAM_EVENT_NOFRAMETIMEOUT: int
TOUPCAM_EVENT_NOPACKETTIMEOUT: int
TOUPCAM_EVENT_TRIGGERFAIL: int
TOUPCAM_EVENT_BLACK: int
TOUPCAM_EVENT_FFC: int
TOUPCAM_EVENT_DFC: int
TOUPCAM_EVENT_FPNC: int
TOUPCAM_EVENT_ROI: int
TOUPCAM_EVENT_LEVELRANGE: int
TOUPCAM_EVENT_AUTOEXPO_CONV: int
TOUPCAM_EVENT_AUTOEXPO_CONVFAIL: int
TOUPCAM_EVENT_FACTORY: int
TOUPCAM_EVENT_EXPO_START: int
TOUPCAM_EVENT_EXPO_STOP: int
TOUPCAM_EVENT_TRIGGER_ALLOW: int
TOUPCAM_EVENT_TRIGGER_IN: int
TOUPCAM_EVENT_HEARTBEAT: int

# Option constants
TOUPCAM_OPTION_RAW: int
TOUPCAM_OPTION_ISP: int
TOUPCAM_OPTION_BITDEPTH: int
TOUPCAM_OPTION_TRIGGER: int
TOUPCAM_OPTION_RGB: int
TOUPCAM_OPTION_BYTEORDER: int
TOUPCAM_OPTION_UPSIDE_DOWN: int
TOUPCAM_OPTION_ZERO_PADDING: int
TOUPCAM_OPTION_FAN: int
TOUPCAM_OPTION_TEC: int
TOUPCAM_OPTION_TECTARGET: int
TOUPCAM_OPTION_TECTARGET_RANGE: int
TOUPCAM_OPTION_AUTOEXP_POLICY: int
TOUPCAM_OPTION_AUTOEXP_THRESHOLD: int
TOUPCAM_OPTION_FRAMERATE: int
TOUPCAM_OPTION_BLACKLEVEL: int
TOUPCAM_OPTION_MULTITHREAD: int
TOUPCAM_OPTION_BINNING: int
TOUPCAM_OPTION_ROTATE: int
TOUPCAM_OPTION_CG: int
TOUPCAM_OPTION_PIXEL_FORMAT: int
TOUPCAM_OPTION_DDR_DEPTH: int
TOUPCAM_OPTION_FFC: int
TOUPCAM_OPTION_DFC: int
TOUPCAM_OPTION_FPNC: int
TOUPCAM_OPTION_SHARPENING: int
TOUPCAM_OPTION_FACTORY: int
TOUPCAM_OPTION_TEC_VOLTAGE: int
TOUPCAM_OPTION_TEC_VOLTAGE_MAX: int
TOUPCAM_OPTION_TEC_VOLTAGE_MAX_RANGE: int
TOUPCAM_OPTION_POWER: int
TOUPCAM_OPTION_GLOBAL_RESET_MODE: int
TOUPCAM_OPTION_DEVICE_RESET: int
TOUPCAM_OPTION_FOCUSPOS: int
TOUPCAM_OPTION_AFMODE: int
TOUPCAM_OPTION_AFSTATUS: int
TOUPCAM_OPTION_TESTPATTERN: int
TOUPCAM_OPTION_NOFRAME_TIMEOUT: int
TOUPCAM_OPTION_NOPACKET_TIMEOUT: int
TOUPCAM_OPTION_BANDWIDTH: int
TOUPCAM_OPTION_MAX_PRECISE_FRAMERATE: int
TOUPCAM_OPTION_MIN_PRECISE_FRAMERATE: int
TOUPCAM_OPTION_PRECISE_FRAMERATE: int
TOUPCAM_OPTION_RELOAD: int
TOUPCAM_OPTION_CALLBACK_THREAD: int
TOUPCAM_OPTION_FRONTEND_DEQUE_LENGTH: int
TOUPCAM_OPTION_FRAME_DEQUE_LENGTH: int
TOUPCAM_OPTION_BACKEND_DEQUE_LENGTH: int
TOUPCAM_OPTION_SEQUENCER_ONOFF: int
TOUPCAM_OPTION_SEQUENCER_NUMBER: int
TOUPCAM_OPTION_SEQUENCER_EXPOTIME: int
TOUPCAM_OPTION_SEQUENCER_EXPOGAIN: int
TOUPCAM_OPTION_DENOISE: int
TOUPCAM_OPTION_HEAT_MAX: int
TOUPCAM_OPTION_HEAT: int
TOUPCAM_OPTION_LIGHTSOURCE_MAX: int
TOUPCAM_OPTION_LIGHTSOURCE: int
TOUPCAM_OPTION_HEARTBEAT: int
TOUPCAM_OPTION_EVENT_HARDWARE: int
TOUPCAM_OPTION_LOW_POWERCONSUMPTION: int
TOUPCAM_OPTION_LOW_POWER_EXPOTIME: int
TOUPCAM_OPTION_LOW_NOISE: int
TOUPCAM_OPTION_HIGH_FULLWELL: int
TOUPCAM_OPTION_AUTOEXPOSURE_PERCENT: int
TOUPCAM_OPTION_DEFECT_PIXEL: int
TOUPCAM_OPTION_HDR_KB: int
TOUPCAM_OPTION_HDR_THRESHOLD: int
TOUPCAM_OPTION_DYNAMIC_DEFECT: int
TOUPCAM_OPTION_ANTI_SHUTTER_EFFECT: int
TOUPCAM_OPTION_OVERCLOCK_MAX: int
TOUPCAM_OPTION_OVERCLOCK: int
TOUPCAM_OPTION_RESET_SENSOR: int
TOUPCAM_OPTION_RESET_SEQ_TIMESTAMP: int
TOUPCAM_OPTION_MODE_SEQ_TIMESTAMP: int
TOUPCAM_OPTION_CHAMBER_HT: int
TOUPCAM_OPTION_ENV_HT: int
TOUPCAM_OPTION_EXPOSURE_PRE_DELAY: int
TOUPCAM_OPTION_EXPOSURE_POST_DELAY: int
TOUPCAM_OPTION_LINE_PRE_DELAY: int
TOUPCAM_OPTION_LINE_POST_DELAY: int
TOUPCAM_OPTION_AUTOEXPO_CONV: int
TOUPCAM_OPTION_AUTOEXP_EXPOTIME_DAMP: int
TOUPCAM_OPTION_AUTOEXP_GAIN_STEP: int
TOUPCAM_OPTION_OVEREXP_POLICY: int
TOUPCAM_OPTION_AUTOEXPO_TRIGGER: int
TOUPCAM_OPTION_AWB_CONTINUOUS: int
TOUPCAM_OPTION_TIMED_TRIGGER_NUM: int
TOUPCAM_OPTION_TIMED_TRIGGER_LOW: int
TOUPCAM_OPTION_TIMED_TRIGGER_HIGH: int
TOUPCAM_OPTION_THREAD_PRIORITY: int
TOUPCAM_OPTION_LINEAR: int
TOUPCAM_OPTION_CURVE: int
TOUPCAM_OPTION_COLORMATIX: int
TOUPCAM_OPTION_WBGAIN: int
TOUPCAM_OPTION_DEMOSAIC: int
TOUPCAM_OPTION_DEMOSAIC_VIDEO: int
TOUPCAM_OPTION_DEMOSAIC_STILL: int
TOUPCAM_OPTION_OPEN_ERRORCODE: int
TOUPCAM_OPTION_FLUSH: int
TOUPCAM_OPTION_READOUT_MODE: int
TOUPCAM_OPTION_VOLTAGEBIAS: int
TOUPCAM_OPTION_VOLTAGEBIAS_RANGE: int
TOUPCAM_OPTION_TAILLIGHT: int
TOUPCAM_OPTION_PSEUDO_COLOR_START: int
TOUPCAM_OPTION_PSEUDO_COLOR_END: int
TOUPCAM_OPTION_PSEUDO_COLOR_ENABLE: int
TOUPCAM_OPTION_NUMBER_DROP_FRAME: int
TOUPCAM_OPTION_DUMP_CFG: int
TOUPCAM_OPTION_FRONTEND_DEQUE_CURRENT: int
TOUPCAM_OPTION_BACKEND_DEQUE_CURRENT: int
TOUPCAM_OPTION_FRONTEND_FULL: int
TOUPCAM_OPTION_BACKEND_FULL: int
TOUPCAM_OPTION_PACKET_NUMBER: int
TOUPCAM_OPTION_GVCP_TIMEOUT: int
TOUPCAM_OPTION_GVCP_RETRY: int
TOUPCAM_OPTION_GVSP_WAIT_PERCENT: int
TOUPCAM_OPTION_GIGETIMEOUT: int
TOUPCAM_OPTION_FILTERWHEEL_SLOT: int
TOUPCAM_OPTION_FILTERWHEEL_POSITION: int
TOUPCAM_OPTION_TRIGGER_CANCEL_MODE: int
TOUPCAM_OPTION_SCAN_DIRECTION: int
TOUPCAM_OPTION_BLACKLEVEL_AUTOADJUST: int
TOUPCAM_OPTION_MECHANICALSHUTTER: int
TOUPCAM_OPTION_CDS: int
TOUPCAM_OPTION_LINE_LENGTH: int
TOUPCAM_OPTION_LINE_TIME: int
TOUPCAM_OPTION_UPTIME: int
TOUPCAM_OPTION_GPS: int
TOUPCAM_OPTION_BITRANGE: int

# Pixel format constants
TOUPCAM_PIXELFORMAT_RAW8: int
TOUPCAM_PIXELFORMAT_RAW10: int
TOUPCAM_PIXELFORMAT_RAW10PACK: int
TOUPCAM_PIXELFORMAT_RAW11: int
TOUPCAM_PIXELFORMAT_RAW12: int
TOUPCAM_PIXELFORMAT_RAW12PACK: int
TOUPCAM_PIXELFORMAT_RAW14: int
TOUPCAM_PIXELFORMAT_RAW16: int
TOUPCAM_PIXELFORMAT_YUV411: int
TOUPCAM_PIXELFORMAT_VUYY: int
TOUPCAM_PIXELFORMAT_YUV422: int
TOUPCAM_PIXELFORMAT_YUV444: int
TOUPCAM_PIXELFORMAT_RGB888: int
TOUPCAM_PIXELFORMAT_GMCY8: int
TOUPCAM_PIXELFORMAT_GMCY12: int
TOUPCAM_PIXELFORMAT_UYVY: int
TOUPCAM_PIXELFORMAT_HDR8HL: int
TOUPCAM_PIXELFORMAT_HDR10HL: int
TOUPCAM_PIXELFORMAT_HDR11HL: int
TOUPCAM_PIXELFORMAT_HDR12HL: int
TOUPCAM_PIXELFORMAT_HDR14HL: int

# Flag constants
TOUPCAM_FLAG_CMOS: int
TOUPCAM_FLAG_CCD_PROGRESSIVE: int
TOUPCAM_FLAG_CCD_INTERLACED: int
TOUPCAM_FLAG_ROI_HARDWARE: int
TOUPCAM_FLAG_MONO: int
TOUPCAM_FLAG_BINSKIP_SUPPORTED: int
TOUPCAM_FLAG_USB32: int
TOUPCAM_FLAG_USB30: int
TOUPCAM_FLAG_USB32_OVER_USB30: int
TOUPCAM_FLAG_USB30_OVER_USB20: int
TOUPCAM_FLAG_ST4: int
TOUPCAM_FLAG_TEC: int
TOUPCAM_FLAG_GETTEMPERATURE: int
TOUPCAM_FLAG_HIGH_FULLWELL: int
TOUPCAM_FLAG_RAW10: int
TOUPCAM_FLAG_RAW10PACK: int
TOUPCAM_FLAG_RAW11: int
TOUPCAM_FLAG_RAW12: int
TOUPCAM_FLAG_RAW12PACK: int
TOUPCAM_FLAG_RAW14: int
TOUPCAM_FLAG_RAW16: int
TOUPCAM_FLAG_FAN: int
TOUPCAM_FLAG_TEC_ONOFF: int
TOUPCAM_FLAG_ISP: int
TOUPCAM_FLAG_TRIGGER_SOFTWARE: int
TOUPCAM_FLAG_TRIGGER_EXTERNAL: int
TOUPCAM_FLAG_TRIGGER_SINGLE: int
TOUPCAM_FLAG_BLACKLEVEL: int
TOUPCAM_FLAG_FOCUSMOTOR: int
TOUPCAM_FLAG_AUTO_FOCUS: int
TOUPCAM_FLAG_BUFFER: int
TOUPCAM_FLAG_DDR: int
TOUPCAM_FLAG_CG: int
TOUPCAM_FLAG_CGHDR: int
TOUPCAM_FLAG_EVENT_HARDWARE: int
TOUPCAM_FLAG_YUV411: int
TOUPCAM_FLAG_YUV422: int
TOUPCAM_FLAG_YUV444: int
TOUPCAM_FLAG_RGB888: int
TOUPCAM_FLAG_RAW8: int
TOUPCAM_FLAG_GMCY8: int
TOUPCAM_FLAG_GMCY12: int
TOUPCAM_FLAG_GLOBALSHUTTER: int
TOUPCAM_FLAG_PRECISE_FRAMERATE: int
TOUPCAM_FLAG_HEAT: int
TOUPCAM_FLAG_LOW_NOISE: int
TOUPCAM_FLAG_LEVELRANGE_HARDWARE: int
TOUPCAM_FLAG_GIGE: int
TOUPCAM_FLAG_10GIGE: int
TOUPCAM_FLAG_5GIGE: int
TOUPCAM_FLAG_25GIGE: int
TOUPCAM_FLAG_CAMERALINK: int
TOUPCAM_FLAG_CXP: int
TOUPCAM_FLAG_FILTERWHEEL: int
TOUPCAM_FLAG_AUTOFOCUSER: int
TOUPCAM_FLAG_LIGHTSOURCE: int
TOUPCAM_FLAG_LIGHT_SOURCE: int
TOUPCAM_FLAG_GHOPTO: int

# Level range constants
TOUPCAM_LEVELRANGE_MANUAL: int
TOUPCAM_LEVELRANGE_ONCE: int
TOUPCAM_LEVELRANGE_CONTINUE: int
TOUPCAM_LEVELRANGE_ROI: int

# Exception classes
class HRESULTException(OSError):
    """Exception raised when HRESULT indicates an error"""
    def __init__(self, hresult: int, message: str = "") -> None: ...

# Resolution structure
class ToupcamResolution:
    """Camera resolution information"""
    width: int
    height: int

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
    res: list[ToupcamResolution]

class ToupcamDeviceV2:
    """Represents a ToupCam device from EnumV2()"""
    displayname: str
    id: str
    model: ToupcamModelV2

class ToupcamFrameInfoV3:
    """Frame info structure V3"""
    width: int
    height: int
    flag: int
    seq: int
    timestamp: int
    shutterseq: int
    expotime: int
    expogain: int
    blacklevel: int

class ToupcamFrameInfoV4:
    """Frame info structure V4"""
    v3: ToupcamFrameInfoV3
    reserved: int
    uLum: int
    uFV: int
    timecount: int
    framecount: int
    tricount: int
    # gps: ToupcamGps  # GPS structure not fully defined

class Toupcam:
    """ToupCam camera interface"""

    @classmethod
    def EnumV2(cls) -> list[ToupcamDeviceV2]:
        """Enumerate all connected cameras, returns list of ToupcamDeviceV2"""
        ...

    @classmethod
    def EnumWithName(cls) -> list[ToupcamDeviceV2]:
        """Enumerate all connected cameras with names, returns list of ToupcamDeviceV2"""
        ...

    @classmethod
    def Open(cls, camId: Optional[str]) -> Optional["Toupcam"]:
        """
        Open camera by ID. Returns None if failed.
        camId can be None (first camera), device ID, or special formats like "sn:xxx", "ip:xxx", etc.
        """
        ...

    @classmethod
    def OpenByIndex(cls, index: int) -> Optional["Toupcam"]:
        """Open camera by index (0=first, 1=second, etc). Returns None if failed."""
        ...

    def Close(self) -> None:
        """Close camera connection"""
        ...

    def Stop(self) -> None:
        """Stop current operation"""
        ...

    def Pause(self, bPause: bool) -> int:
        """Pause/unpause video stream. Returns HRESULT"""
        ...

    # Size and resolution
    def get_Size(self) -> tuple[int, int]:
        """Get current resolution, returns (width, height)"""
        ...

    def put_Size(self, nWidth: int, nHeight: int) -> int:
        """Set resolution by width/height. Returns HRESULT"""
        ...

    def get_eSize(self) -> int:
        """Get current resolution index"""
        ...

    def put_eSize(self, nResolutionIndex: int) -> int:
        """Set resolution by index. Returns HRESULT"""
        ...

    def get_FinalSize(self) -> tuple[int, int]:
        """Get final size after ROI/binning/rotation. Returns (width, height)"""
        ...

    def get_ResolutionNumber(self) -> int:
        """Get number of supported resolutions"""
        ...

    def get_Resolution(self, nResolutionIndex: int) -> tuple[int, int]:
        """Get resolution by index. Returns (width, height)"""
        ...

    def get_ResolutionRatio(self, nResolutionIndex: int) -> tuple[int, int]:
        """Get resolution ratio (binning). Returns (num, den)"""
        ...

    def get_StillResolutionNumber(self) -> int:
        """Get number of supported still resolutions"""
        ...

    def get_StillResolution(self, nResolutionIndex: int) -> tuple[int, int]:
        """Get still resolution by index. Returns (width, height)"""
        ...

    # ROI (Region of Interest)
    def put_Roi(self, xOffset: int, yOffset: int, xWidth: int, yHeight: int) -> int:
        """Set ROI. Returns HRESULT"""
        ...

    def get_Roi(self) -> tuple[int, int, int, int]:
        """Get ROI. Returns (xOffset, yOffset, xWidth, yHeight)"""
        ...

    # Pull mode operations
    def StartPullModeWithCallback(
        self,
        fun: Callable[[int, Any], None],
        ctx: Any
    ) -> int:
        """Start pull mode with callback. Returns HRESULT"""
        ...

    def PullImageV4(
        self,
        pImageData: bytes,
        bStill: int,
        bits: int,
        rowPitch: int,
        pInfo: Optional[ToupcamFrameInfoV4] = None
    ) -> int:
        """Pull image data V4. Returns HRESULT"""
        ...

    def PullImageV3(
        self,
        pImageData: bytes,
        bStill: int,
        bits: int,
        rowPitch: int,
        pInfo: Optional[ToupcamFrameInfoV3] = None
    ) -> int:
        """Pull image data V3. Returns HRESULT"""
        ...

    def WaitImageV4(
        self,
        nWaitMS: int,
        pImageData: bytes,
        bStill: int,
        bits: int,
        rowPitch: int,
        pInfo: Optional[ToupcamFrameInfoV4] = None
    ) -> int:
        """Wait for and pull image data V4. Returns HRESULT"""
        ...

    def WaitImageV3(
        self,
        nWaitMS: int,
        pImageData: bytes,
        bStill: int,
        bits: int,
        rowPitch: int,
        pInfo: Optional[ToupcamFrameInfoV3] = None
    ) -> int:
        """Wait for and pull image data V3. Returns HRESULT"""
        ...

    # Push mode operations
    def StartPushModeV4(
        self,
        fun: Callable[[bytes, ToupcamFrameInfoV3, bool, Any], None],
        ctx: Any
    ) -> int:
        """Start push mode V4. Returns HRESULT"""
        ...

    def StartPushModeV3(
        self,
        fun: Callable[[bytes, Any, bool, Any], None],  # Simplified callback signature
        ctx: Any
    ) -> int:
        """Start push mode V3. Returns HRESULT"""
        ...

    # Still image capture
    def Snap(self, nResolutionIndex: int) -> int:
        """Snap still image. Returns HRESULT"""
        ...

    def SnapN(self, nResolutionIndex: int, nNumber: int) -> int:
        """Snap N still images. Returns HRESULT"""
        ...

    def SnapR(self, nResolutionIndex: int) -> int:
        """Snap RAW still image. Returns HRESULT"""
        ...

    # Trigger operations
    def Trigger(self, nNumber: int) -> int:
        """Trigger acquisition. Returns HRESULT"""
        ...

    def TriggerSyncV4(
        self,
        nWaitMS: int,
        pImageData: bytes,
        bits: int,
        rowPitch: int,
        pInfo: Optional[ToupcamFrameInfoV4] = None
    ) -> int:
        """Synchronous trigger V4. Returns HRESULT"""
        ...

    def TriggerSync(
        self,
        nWaitMS: int,
        pImageData: bytes,
        bits: int,
        rowPitch: int,
        pInfo: Optional[ToupcamFrameInfoV3] = None
    ) -> int:
        """Synchronous trigger. Returns HRESULT"""
        ...

    # Options
    def put_Option(self, iOption: int, iValue: int) -> int:
        """Set camera option. Returns HRESULT"""
        ...

    def get_Option(self, iOption: int) -> int:
        """Get camera option value"""
        ...

    # Exposure control
    def put_ExpoTime(self, Time: int) -> int:
        """Set exposure time in microseconds. Returns HRESULT"""
        ...

    def get_ExpoTime(self) -> int:
        """Get exposure time in microseconds"""
        ...

    def get_ExpTimeRange(self) -> tuple[int, int, int]:
        """Get exposure time range. Returns (min, max, default)"""
        ...

    def get_RealExpoTime(self) -> int:
        """Get actual exposure time in microseconds"""
        ...

    def put_ExpoAGain(self, AGain: int) -> int:
        """Set analog gain. Returns HRESULT"""
        ...

    def get_ExpoAGain(self) -> int:
        """Get analog gain"""
        ...

    def get_ExpoAGainRange(self) -> tuple[int, int, int]:
        """Get analog gain range. Returns (min, max, default)"""
        ...

    # Auto exposure
    def put_AutoExpoEnable(self, bAutoExposure: int) -> int:
        """Enable/disable auto exposure. Returns HRESULT"""
        ...

    def get_AutoExpoEnable(self) -> bool:
        """Get auto exposure state"""
        ...

    def put_AutoExpoTarget(self, Target: int) -> int:
        """Set auto exposure target. Returns HRESULT"""
        ...

    def get_AutoExpoTarget(self) -> int:
        """Get auto exposure target"""
        ...

    def put_AutoExpoRange(
        self,
        maxTime: int,
        minTime: int,
        maxAGain: int,
        minAGain: int
    ) -> int:
        """Set auto exposure range. Returns HRESULT"""
        ...

    def get_AutoExpoRange(self) -> tuple[int, int, int, int]:
        """Get auto exposure range. Returns (maxTime, minTime, maxAGain, minAGain)"""
        ...

    def put_MaxAutoExpoTimeAGain(self, maxTime: int, maxAGain: int) -> int:
        """Set max auto exposure time and gain. Returns HRESULT"""
        ...

    def get_MaxAutoExpoTimeAGain(self) -> tuple[int, int]:
        """Get max auto exposure time and gain. Returns (maxTime, maxAGain)"""
        ...

    def put_MinAutoExpoTimeAGain(self, minTime: int, minAGain: int) -> int:
        """Set min auto exposure time and gain. Returns HRESULT"""
        ...

    def get_MinAutoExpoTimeAGain(self) -> tuple[int, int]:
        """Get min auto exposure time and gain. Returns (minTime, minAGain)"""
        ...

    # Speed/framerate control
    def put_Speed(self, nSpeed: int) -> int:
        """Set speed level. Returns HRESULT"""
        ...

    def get_Speed(self) -> int:
        """Get current speed level"""
        ...

    def get_MaxSpeed(self) -> int:
        """Get maximum speed level"""
        ...

    def get_FrameRate(self) -> tuple[int, int, int]:
        """Get frame rate info. Returns (nFrame, nTime, nTotalFrame)"""
        ...

    # Real-time mode
    def put_RealTime(self, val: int) -> int:
        """Set real-time mode. Returns HRESULT"""
        ...

    def get_RealTime(self) -> int:
        """Get real-time mode"""
        ...

    # White balance
    def put_TempTint(self, nTemp: int, nTint: int) -> int:
        """Set temperature and tint. Returns HRESULT"""
        ...

    def get_TempTint(self) -> tuple[int, int]:
        """Get temperature and tint. Returns (temp, tint)"""
        ...

    def AwbOnce(
        self,
        funTT: Optional[Callable[[int, int, Any], None]] = None,
        ctxTT: Any = None
    ) -> int:
        """Perform auto white balance once. Returns HRESULT"""
        ...

    def put_WhiteBalanceGain(self, aGain: list[int]) -> int:
        """Set RGB white balance gains. Returns HRESULT"""
        ...

    def get_WhiteBalanceGain(self) -> list[int]:
        """Get RGB white balance gains. Returns [R, G, B]"""
        ...

    def AwbInit(
        self,
        funWB: Optional[Callable[[list[int], Any], None]] = None,
        ctxWB: Any = None
    ) -> int:
        """Initialize auto white balance. Returns HRESULT"""
        ...

    # Black balance
    def put_BlackBalance(self, aSub: list[int]) -> int:
        """Set RGB black balance offsets. Returns HRESULT"""
        ...

    def get_BlackBalance(self) -> list[int]:
        """Get RGB black balance offsets. Returns [R, G, B]"""
        ...

    def AbbOnce(
        self,
        funBB: Optional[Callable[[list[int], Any], None]] = None,
        ctxBB: Any = None
    ) -> int:
        """Perform auto black balance once. Returns HRESULT"""
        ...

    # Color adjustments
    def put_Hue(self, Hue: int) -> int:
        """Set hue. Returns HRESULT"""
        ...

    def get_Hue(self) -> int:
        """Get hue"""
        ...

    def put_Saturation(self, Saturation: int) -> int:
        """Set saturation. Returns HRESULT"""
        ...

    def get_Saturation(self) -> int:
        """Get saturation"""
        ...

    def put_Brightness(self, Brightness: int) -> int:
        """Set brightness. Returns HRESULT"""
        ...

    def get_Brightness(self) -> int:
        """Get brightness"""
        ...

    def put_Contrast(self, Contrast: int) -> int:
        """Set contrast. Returns HRESULT"""
        ...

    def get_Contrast(self) -> int:
        """Get contrast"""
        ...

    def put_Gamma(self, Gamma: int) -> int:
        """Set gamma. Returns HRESULT"""
        ...

    def get_Gamma(self) -> int:
        """Get gamma"""
        ...

    # Flip operations
    def put_VFlip(self, bVFlip: bool) -> int:
        """Set vertical flip. Returns HRESULT"""
        ...

    def get_VFlip(self) -> bool:
        """Get vertical flip state"""
        ...

    def put_HFlip(self, bHFlip: bool) -> int:
        """Set horizontal flip. Returns HRESULT"""
        ...

    def get_HFlip(self) -> bool:
        """Get horizontal flip state"""
        ...

    # Chrome/mono mode
    def put_Chrome(self, bChrome: bool) -> int:
        """Set chrome/mono mode. Returns HRESULT"""
        ...

    def get_Chrome(self) -> bool:
        """Get chrome/mono mode"""
        ...

    # Temperature control
    def put_Temperature(self, nTemperature: int) -> int:
        """Set target temperature. Returns HRESULT"""
        ...

    def get_Temperature(self) -> int:
        """Get sensor temperature"""
        ...

    # Power frequency
    def put_HZ(self, nHZ: int) -> int:
        """Set power frequency. Returns HRESULT"""
        ...

    def get_HZ(self) -> int:
        """Get power frequency"""
        ...

    # Bin/skip mode
    def put_Mode(self, bSkip: bool) -> int:
        """Set bin/skip mode. Returns HRESULT"""
        ...

    def get_Mode(self) -> bool:
        """Get bin/skip mode"""
        ...

    # Camera info
    def get_SerialNumber(self) -> str:
        """Get camera serial number"""
        ...

    def get_FwVersion(self) -> str:
        """Get firmware version"""
        ...

    def get_HwVersion(self) -> str:
        """Get hardware version"""
        ...

    def get_ProductionDate(self) -> str:
        """Get production date"""
        ...

    def get_Revision(self) -> int:
        """Get revision number"""
        ...

    def get_PixelSize(self, nResolutionIndex: int) -> tuple[float, float]:
        """Get pixel size. Returns (x, y) in micrometers"""
        ...

    def get_MonoMode(self) -> bool:
        """Check if camera is monochrome"""
        ...

    def get_MaxBitDepth(self) -> int:
        """Get maximum bit depth"""
        ...

    # Raw format info
    def get_RawFormat(self) -> tuple[int, int]:
        """Get raw format info. Returns (fourCC, bitsPerPixel)"""
        ...

    # Pixel format support
    def get_PixelFormatSupport(self, cmd: int) -> int:
        """Get pixel format support info"""
        ...

    # Level range
    def put_LevelRange(
        self,
        aLow: list[int],
        aHigh: list[int]
    ) -> int:
        """Set level range. Returns HRESULT"""
        ...

    def get_LevelRange(self) -> tuple[list[int], list[int]]:
        """Get level range. Returns (aLow, aHigh)"""
        ...

    def LevelRangeAuto(self) -> int:
        """Auto adjust level range. Returns HRESULT"""
        ...

    def put_LevelRangeV2(
        self,
        mode: int,
        pRoiRect: Optional[tuple[int, int, int, int]],
        aLow: list[int],
        aHigh: list[int]
    ) -> int:
        """Set level range V2. Returns HRESULT"""
        ...

    def get_LevelRangeV2(self) -> tuple[int, tuple[int, int, int, int], list[int], list[int]]:
        """Get level range V2. Returns (mode, roiRect, aLow, aHigh)"""
        ...

    # Auto exposure auxiliary rectangle
    def put_AEAuxRect(self, xOffset: int, yOffset: int, xWidth: int, yHeight: int) -> int:
        """Set auto exposure auxiliary rectangle. Returns HRESULT"""
        ...

    def get_AEAuxRect(self) -> tuple[int, int, int, int]:
        """Get auto exposure auxiliary rectangle. Returns (xOffset, yOffset, xWidth, yHeight)"""
        ...

    def put_AWBAuxRect(self, xOffset: int, yOffset: int, xWidth: int, yHeight: int) -> int:
        """Set auto white balance auxiliary rectangle. Returns HRESULT"""
        ...

    def get_AWBAuxRect(self) -> tuple[int, int, int, int]:
        """Get auto white balance auxiliary rectangle. Returns (xOffset, yOffset, xWidth, yHeight)"""
        ...

    def put_ABBAuxRect(self, xOffset: int, yOffset: int, xWidth: int, yHeight: int) -> int:
        """Set auto black balance auxiliary rectangle. Returns HRESULT"""
        ...

    def get_ABBAuxRect(self) -> tuple[int, int, int, int]:
        """Get auto black balance auxiliary rectangle. Returns (xOffset, yOffset, xWidth, yHeight)"""
        ...

    # LED control
    def put_LEDState(self, iLed: int, iState: int, iPeriod: int) -> int:
        """Set LED state. Returns HRESULT"""
        ...

    # EEPROM operations
    def read_EEPROM(self, addr: int, pBuffer: bytes, nBufferLen: int) -> int:
        """Read from EEPROM. Returns bytes read or HRESULT error"""
        ...

    def write_EEPROM(self, addr: int, pBuffer: bytes, nBufferLen: int) -> int:
        """Write to EEPROM. Returns bytes written or HRESULT error"""
        ...

    # UART operations
    def read_UART(self, pBuffer: bytes, nBufferLen: int) -> int:
        """Read from UART. Returns bytes read or HRESULT error"""
        ...

    def write_UART(self, pBuffer: bytes, nBufferLen: int) -> int:
        """Write to UART. Returns bytes written or HRESULT error"""
        ...

    # IO Control
    def IoControl(self, ioLine: int, nType: int, outVal: int) -> tuple[int, int]:
        """IO control operation. Returns (result, inVal)"""
        ...

    # Correction operations
    def FfcOnce(self) -> int:
        """Perform flat field correction once. Returns HRESULT"""
        ...

    def DfcOnce(self) -> int:
        """Perform dark field correction once. Returns HRESULT"""
        ...

    def FpncOnce(self) -> int:
        """Perform fixed pattern noise correction once. Returns HRESULT"""
        ...

    def FfcExport(self, filepath: str) -> int:
        """Export flat field correction data. Returns HRESULT"""
        ...

    def FfcImport(self, filepath: str) -> int:
        """Import flat field correction data. Returns HRESULT"""
        ...

    def DfcExport(self, filepath: str) -> int:
        """Export dark field correction data. Returns HRESULT"""
        ...

    def DfcImport(self, filepath: str) -> int:
        """Import dark field correction data. Returns HRESULT"""
        ...

    def FpncExport(self, filepath: str) -> int:
        """Export fixed pattern noise correction data. Returns HRESULT"""
        ...

    def FpncImport(self, filepath: str) -> int:
        """Import fixed pattern noise correction data. Returns HRESULT"""
        ...

    # Histogram
    def GetHistogram(
        self,
        funHistogram: Callable[[list[float], Any], None],
        ctxHistogram: Any
    ) -> int:
        """Get histogram data. Returns HRESULT"""
        ...

    # Firmware update
    @classmethod
    def Update(
        cls,
        camId: str,
        filePath: str,
        pFun: Optional[Callable[[int, Any], None]] = None,
        ctxProgress: Any = None
    ) -> int:
        """Update camera firmware. Returns HRESULT"""
        ...

    # Device replug simulation
    @classmethod
    def Replug(cls, camId: str) -> int:
        """Simulate device replug. Returns number of devices or error"""
        ...

    # Flush operations
    def Flush(self) -> int:
        """Flush camera buffers. Returns HRESULT"""
        ...

# Module-level functions
def GigeEnable(
    funHotPlug: Optional[Callable[[Any], None]] = None,
    ctxHotPlug: Any = None
) -> int:
    """Enable GigE camera support. Returns HRESULT"""
    ...

def HotPlug(
    funHotPlug: Callable[[Any], None],
    ctxHotPlug: Any
) -> None:
    """Register hot plug callback (Linux/macOS only)"""
    ...

# Version info
Version: str
