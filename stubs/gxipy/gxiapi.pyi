"""
Type stubs for gxipy.gxiapi module.

This provides comprehensive type annotations for the Galaxy camera library.
Based on runtime inspection of the actual gxiapi module and official documentation.
"""

import ctypes
from collections.abc import Callable
from typing import Any, Optional, Union, List, Tuple, Literal, TypedDict

# Initialize and cleanup functions
def gx_init_lib() -> None:
    """Initialize the Galaxy camera library"""
    ...

def gx_close_lib() -> None:
    """Close the Galaxy camera library and cleanup resources"""
    ...

# Access modes for device opening
class GxAccessMode:
    CONTROL: int
    EXCLUSIVE: int
    CONTROL_WITH_SWITCHOVER: int

# Frame status constants
class GxFrameStatusList:
    SUCCESS: int
    INCOMPLETE: int
    INVALID_IMAGE_INFO: int

# Pixel format constants
class GxPixelFormatEntry:
    # Mono formats
    MONO8: int
    MONO10: int
    MONO10_PACKED: int
    MONO12: int
    MONO12_PACKED: int
    MONO14: int
    MONO16: int
    
    # Bayer formats
    BAYER_GR8: int
    BAYER_RG8: int
    BAYER_GB8: int
    BAYER_BG8: int
    BAYER_GR10: int
    BAYER_RG10: int
    BAYER_GB10: int
    BAYER_BG10: int
    BAYER_GR12: int
    BAYER_RG12: int
    BAYER_GB12: int
    BAYER_BG12: int
    BAYER_GR16: int
    BAYER_RG16: int
    BAYER_GB16: int
    BAYER_BG16: int
    
    # RGB formats
    RGB8: int
    BGR8: int
    RGBA8: int
    BGRA8: int
    
    # YUV formats
    YUV422_8: int
    YUV422_PACKED: int

# Trigger source constants
class GxTriggerSourceEntry:
    SOFTWARE: int
    LINE0: int
    LINE1: int
    LINE2: int
    LINE3: int
    COUNTER0: int
    COUNTER1: int
    TIMER0: int
    TIMER1: int
    USER_OUTPUT0: int
    USER_OUTPUT1: int
    USER_OUTPUT2: int
    ACTION0: int
    ACTION1: int

# Trigger switch constants  
class GxTriggerSwitchEntry:
    OFF: int
    ON: int

# Trigger mode constants
class GxTriggerModeEntry:
    OFF: int
    ON: int

# Trigger activation constants
class GxTriggerActivationEntry:
    RISINGEDGE: int
    FALLINGEDGE: int
    ANYEDGE: int
    LEVELHEIGH: int
    LEVELLOW: int

# Acquisition mode constants
class GxAcquisitionModeEntry:
    SINGLE_FRAME: int
    MULTI_FRAME: int
    CONTINUOUS: int

# Exposure auto constants
class GxExposureAutoEntry:
    OFF: int
    ONCE: int
    CONTINUOUS: int

# Gain auto constants  
class GxGainAutoEntry:
    OFF: int
    ONCE: int
    CONTINUOUS: int

# Gain selector constants
class GxGainSelectorEntry:
    ALL: int
    RED: int
    GREEN: int
    BLUE: int
    DIGITAL_ALL: int
    DIGITAL_RED: int
    DIGITAL_GREEN: int
    DIGITAL_BLUE: int

# Balance white auto constants
class GxBalanceWhiteAutoEntry:
    OFF: int
    ONCE: int
    CONTINUOUS: int

# Balance ratio selector constants
class GxBalanceRatioSelectorEntry:
    RED: int
    GREEN: int
    BLUE: int

# Black level selector constants
class GxBlackLevelSelectorEntry:
    ALL: int
    RED: int
    GREEN: int
    BLUE: int
    DIGITAL_ALL: int
    DIGITAL_RED: int
    DIGITAL_GREEN: int
    DIGITAL_BLUE: int

# Black level auto constants
class GxBlackLevelAutoEntry:
    OFF: int
    ONCE: int
    CONTINUOUS: int

# Test pattern constants
class GxTestPatternEntry:
    OFF: int
    GRAY_HORIZONTAL_RAMP: int
    GRAY_VERTICAL_RAMP: int
    GRAY_HORIZONTAL_RAMP_MOVING: int
    GRAY_VERTICAL_RAMP_MOVING: int
    GRAY_DIAGONAL_SAWTOOTH: int
    COLOR_BAR: int
    COLOR_BAR_MOVING: int

# Line selector constants
class GxLineSelectorEntry:
    LINE0: int
    LINE1: int
    LINE2: int
    LINE3: int
    LINE4: int
    LINE5: int
    LINE6: int
    LINE7: int

# Line mode constants
class GxLineModeEntry:
    INPUT: int
    OUTPUT: int
    STROBE: int

# Line source constants
class GxLineSourceEntry:
    OFF: int
    STROBE: int
    USER_OUTPUT0: int
    USER_OUTPUT1: int
    USER_OUTPUT2: int
    TIMER0_ACTIVE: int
    TIMER1_ACTIVE: int
    EXPOSURE_ACTIVE: int
    FRAME_TRIGGER_WAIT: int

# User output selector constants
class GxUserOutputSelectorEntry:
    OUTPUT0: int
    OUTPUT1: int
    OUTPUT2: int
    OUTPUT3: int
    OUTPUT4: int
    OUTPUT5: int
    OUTPUT6: int
    OUTPUT7: int

# Event selector constants
class GxEventSelectorEntry:
    EXPOSUREEND: int
    BLOCK_DISCARD: int
    EVENT_OVERRUN: int
    FRAMESTART_OVERTRIGGER: int
    BLOCK_NOT_EMPTY: int
    INTERNAL_ERROR: int
    FRAMEBURSTSTART_OVERTRIGGER: int
    FRAMESTART_WAIT: int
    FRAMEBURSTSTART_WAIT: int

# Event notification constants
class GxEventNotificationEntry:
    OFF: int
    ON: int
    ONCE: int

# AWB lamp house constants
class GxAwbLampHouseEntry:
    ADAPTIVE: int
    D65: int
    FLUORESCENCE: int
    INCANDESCENT: int
    D50: int
    D75: int
    MANUAL: int

# Pixel color filter constants
class GxPixelColorFilterEntry:
    NONE: int
    BAYER_RG: int
    BAYER_GB: int
    BAYER_GR: int
    BAYER_BG: int

# Pixel size constants
class GxPixelSizeEntry:
    BPP8: int
    BPP10: int
    BPP12: int
    BPP14: int
    BPP16: int
    BPP24: int
    BPP32: int
    BPP48: int
    BPP64: int

# LUT selector constants
class GxLutSelectorEntry:
    LUMINANCE: int
    RED: int
    GREEN: int
    BLUE: int

# Device temperature selector constants  
class GxDeviceTemperatureSelectorEntry:
    SENSOR: int
    MAINBOARD: int
    TEC: int

# Color transformation mode constants
class GxColorTransformationModeEntry:
    RGB_TO_RGB: int
    USER: int

# Color transformation value selector constants
class GxColorTransformationValueSelectorEntry:
    GAIN00: int
    GAIN01: int
    GAIN02: int
    GAIN10: int
    GAIN11: int
    GAIN12: int
    GAIN20: int
    GAIN21: int
    GAIN22: int

# Gamma mode constants
class GxGammaModeEntry:
    sRGB: int
    USER: int

# Sharpness mode constants
class GxSharpnessModeEntry:
    OFF: int
    ON: int

# Noise reduction mode constants
class GxNoiseReductionModeEntry:
    OFF: int
    ON: int

# Dead pixel correct constants
class GxDeadPixelCorrectEntry:
    OFF: int
    ON: int

# Flat field correction constants
class GxFlatFieldCorrectionEntry:
    OFF: int
    ON: int

# Remove parameter limit constants
class GxRemoveParameterLimitEntry:
    OFF: int
    ON: int

# Acquisition frame rate mode constants
class GxAcquisitionFrameRateModeEntry:
    OFF: int
    ON: int

# Sensor shutter mode constants
class GxSensorShutterModeEntry:
    GLOBAL: int
    ROLLING: int
    GLOBAL_RESET_RELEASE: int

# Exposure mode constants
class GxExposureModeEntry:
    TIMED: int
    TRIGGER_WIDTH: int

# Transfer control mode constants
class GxTransferControlModeEntry:
    BASIC: int
    USER_CONTROLLED: int

# Transfer operation mode constants
class GxTransferOperationModeEntry:
    CONTINUOUS: int
    MULTI_BLOCK: int

# Binning horizontal mode constants
class GxBinningHorizontalModeEntry:
    SUM: int
    AVERAGE: int

# Binning vertical mode constants  
class GxBinningVerticalModeEntry:
    SUM: int
    AVERAGE: int

# Stream buffer handling mode constants
class GxStreamBufferHandlingModeEntry:
    OLDEST_FIRST: int
    OLDEST_FIRST_OVERWRITE: int
    NEWEST_ONLY: int

# Resend mode constants
class GxResendModeEntry:
    OFF: int
    ON: int

# IP configure mode constants
class GxIpConfigureModeEntry:
    STATIC: int
    DHCP: int
    LLA: int

# Device link throughput limit mode constants
class GxDeviceLinkThroughputLimitModeEntry:
    OFF: int
    ON: int

# Device link throughput limit mode (alternate naming)
class DeviceLinkThroughputLimitMode:
    OFF: int
    ON: int

# Switch entry (generic switch/toggle)
class GxSwitchEntry:
    OFF: int
    ON: int

# User set selector constants
class GxUserSetSelectorEntry:
    DEFAULT: int
    USERSET0: int
    USERSET1: int
    USERSET2: int

# User set default constants
class GxUserSetDefaultEntry:
    DEFAULT: int
    USERSET0: int
    USERSET1: int
    USERSET2: int

# Timer selector constants
class GxTimerSelectorEntry:
    TIMER0: int
    TIMER1: int
    TIMER2: int
    TIMER3: int

# Timer trigger source constants
class GxTimerTriggerSourceEntry:
    OFF: int
    EXPOSURE_START: int
    EXPOSURE_END: int
    FRAME_START: int
    FRAME_END: int
    LINE_RISING_EDGE: int
    LINE_FALLING_EDGE: int

# Counter selector constants
class GxCounterSelectorEntry:
    COUNTER0: int
    COUNTER1: int
    COUNTER2: int
    COUNTER3: int

# Counter event source constants
class GxCounterEventSourceEntry:
    OFF: int
    FRAME_START: int
    FRAME_END: int
    EXPOSURE_START: int
    EXPOSURE_END: int
    LINE_RISING_EDGE: int
    LINE_FALLING_EDGE: int

# Counter reset source constants
class GxCounterResetSourceEntry:
    OFF: int
    SOFTWARE: int
    LINE: int
    COUNTER: int
    TIMER: int

# Counter reset activation constants
class GxCounterResetActivationEntry:
    RISING_EDGE: int
    FALLING_EDGE: int
    ANY_EDGE: int
    LEVEL_HIGH: int
    LEVEL_LOW: int

# Counter trigger source constants
class GxCounterTriggerSourceEntry:
    OFF: int
    SOFTWARE: int
    LINE0: int
    LINE1: int
    LINE2: int
    LINE3: int
    COUNTER0: int
    COUNTER1: int
    TIMER0: int
    TIMER1: int

# Timer trigger activation constants
class GxTimerTriggerActivationEntry:
    RISING_EDGE: int
    FALLING_EDGE: int
    ANY_EDGE: int
    LEVEL_HIGH: int
    LEVEL_LOW: int

# Light source preset constants
class GxLightSourcePresetEntry:
    DAYLIGHT_5000K: int
    DAYLIGHT_6500K: int
    COOL_WHITE_FLUORESCENT: int
    INCANDESCENT_2800K: int
    OFF: int

# Acquisition status selector constants
class GxAcquisitionStatusSelectorEntry:
    FRAME_TRIGGER_WAIT: int
    ACQUISITION_TRIGGER_WAIT: int

# Saturation mode constants
class GxSaturationModeEntry:
    OFF: int
    ON: int

# Static defect correction constants
class GxStaticDefectCorrectionEntry:
    OFF: int
    ON: int

# 2D noise reduction mode constants
class Gx2dNoiseReductionModeEntry:
    OFF: int
    LOW: int
    MIDDLE: int
    HIGH: int

# 3D noise reduction mode constants  
class Gx3dNoiseReductionModeEntry:
    OFF: int
    LOW: int
    MIDDLE: int
    HIGH: int

# HDR mode constants
class GxHdrModeEntry:
    OFF: int
    CONTINUOUS: int

# MGC mode constants
class GxMgcModeEntry:
    OFF: int
    TWO_FRAME: int
    FOUR_FRAME: int

# Chunk selector constants
class GxChunkSelectorEntry:
    FRAME_ID: int
    TIMESTAMP: int
    EXPOSURE_TIME: int
    GAIN: int
    BLACK_LEVEL: int

# Strobe switch constants (USB2.0 specific)
class GxStrobeSwitchEntry:
    OFF: int
    ON: int

# User output mode constants (USB2.0 specific)
class GxUserOutputModeEntry:
    STROBE: int
    USER_DEFINED: int

# Exposure time mode constants
class GxExposureTimeModeEntry:
    STANDARD: int
    ULTRA_SHORT: int

# Status codes
class GxStatusList:
    SUCCESS: int
    ERROR: int
    NOT_FOUND_TL: int
    NOT_FOUND_DEVICE: int
    OFFLINE: int
    INVALID_PARAMETER: int
    INVALID_HANDLE: int
    INVALID_CALL: int
    INVALID_ACCESS: int
    NEED_MORE_BUFFER: int
    ERROR_TYPE: int
    OUT_OF_RANGE: int
    NOT_IMPLEMENTED: int
    NOT_INIT_API: int
    TIMEOUT: int

# Device class types
class GxDeviceClassList:
    UNKNOWN: int
    USB2: int
    USB3: int
    GEV: int
    U3V: int

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

class InvalidHandle(Exception):
    """Invalid handle exception"""
    ...

class InvalidCall(Exception):
    """Invalid call exception"""
    ...

class OutOfRange(Exception):
    """Out of range exception"""
    ...

class NotImplemented(Exception):
    """Not implemented exception"""
    ...

class NotInitApi(Exception):
    """API not initialized exception"""
    ...

# Device info structure
class GxDeviceBaseInfo(TypedDict):
    """Device information structure"""
    index:int
    vendor_name: str
    model_name: str
    sn: str
    display_name: str
    device_id: str
    user_id: str
    access_status: int
    device_class: int
    reserved: str

# Missing image processing parameter structures
class GxImageImprovementParam:
    """Image improvement parameters structure"""
    color_correction_param: int
    contrast_lut: Optional[bytes]
    gamma_lut: Optional[bytes]
    def __init__(self) -> None: ...

class GxWhiteBalanceParam:
    """White balance parameters structure"""
    ratio_red: float
    ratio_green: float
    ratio_blue: float
    def __init__(self) -> None: ...

class GxAutoWhiteBalanceROI:
    """Auto white balance region of interest"""
    x: int
    y: int
    width: int
    height: int
    def __init__(self) -> None: ...

# Raw image class
class RawImage:
    """Raw image data from camera"""
    
    def get_status(self) -> int:
        """Get image status (SUCCESS, INCOMPLETE, etc.)"""
        ...

    def get_numpy_array(self) -> Optional[Any]:  # numpy.ndarray or None
        """Convert to numpy array, returns None if invalid"""
        ...

    def get_width(self) -> int:
        """Get image width in pixels"""
        ...

    def get_height(self) -> int:
        """Get image height in pixels"""
        ...
    
    def get_pixel_format(self) -> int:
        """Get pixel format of the image"""
        ...
    
    def get_pixel_size(self) -> int:
        """Get pixel size in bits"""
        ...
    
    def get_image_size(self) -> int:
        """Get total image size in bytes"""
        ...
    
    def get_frame_id(self) -> int:
        """Get frame ID"""
        ...
    
    def get_timestamp(self) -> int:
        """Get frame timestamp"""
        ...
    
    def get_offset_x(self) -> int:
        """Get X offset of the image"""
        ...
    
    def get_offset_y(self) -> int:
        """Get Y offset of the image"""
        ...
    
    def get_padding_x(self) -> int:
        """Get X padding of the image"""
        ...
    
    def get_padding_y(self) -> int:
        """Get Y padding of the image"""
        ...
    
    def save_raw(self, file_path: str) -> None:
        """Save raw image data to file"""
        ...
    
    def save_ppm(self, file_path: str) -> None:
        """Save image as PPM file"""
        ...
    
    def save_bmp(self, file_path: str) -> None:
        """Save image as BMP file"""
        ...
    
    def save_png(self, file_path: str) -> None:
        """Save image as PNG file"""
        ...
    
    def save_jpg(self, file_path: str, quality: int = 100) -> None:
        """Save image as JPEG file"""
        ...
    
    def get_buffer(self) -> Optional[ctypes.c_void_p]:
        """Get raw buffer pointer"""
        ...

# Device Manager class
class DeviceManager:
    """Manages camera device enumeration and access"""

    def __init__(self) -> None: ...

    def update_device_list(self, timeout: int = 200) -> Tuple[int, List[GxDeviceBaseInfo]]:
        """
        Update device list and return (device_count, device_info_list).
        Returns number of devices found and list of device info structures.
        """
        ...

    def update_all_device_list(self, timeout: int = 200) -> Tuple[int, List[GxDeviceBaseInfo]]:
        """Update all device list including GigE cameras"""
        ...

    def get_device_number(self) -> int:
        """Get number of enumerated devices"""
        ...

    def open_device_by_index(self, index: int, access_mode: int = 3) -> Optional["Device"]:
        """Open device by index, returns Device object or None if failed"""
        ...

    def open_device_by_sn(self, sn: str, access_mode: int = 3) -> Optional["Device"]:
        """Open device by serial number"""
        ...
    
    def open_device_by_user_id(self, user_id: str, access_mode: int = 3) -> Optional["Device"]:
        """Open device by user ID"""
        ...
    
    def open_device_by_ip(self, ip: str, access_mode: int = 3) -> Optional["Device"]:
        """Open device by IP address (GigE cameras only)"""
        ...

# Device class (camera)
class Device:
    """Galaxy camera device interface"""

    # Device attribute properties (direct access to common features)
    # Acquisition Control
    AcquisitionMode: "EnumFeature"
    AcquisitionStart: Any  # Command feature
    AcquisitionStop: Any   # Command feature
    AcquisitionFrameCount: "IntFeature"
    AcquisitionFrameRate: "FloatFeature"
    AcquisitionFrameRateMode: "EnumFeature"
    TriggerMode: "EnumFeature"
    TriggerSource: "EnumFeature"
    TriggerActivation: "EnumFeature"
    TriggerSoftware: Any  # Command feature
    TriggerDelay: "FloatFeature"
    
    # Image Format Control
    SensorWidth: "IntFeature"
    SensorHeight: "IntFeature"
    WidthMax: "IntFeature"
    HeightMax: "IntFeature"
    OffsetX: "IntFeature"
    OffsetY: "IntFeature"
    Width: "IntFeature"
    Height: "IntFeature"
    PixelFormat: "EnumFeature"
    PixelSize: "EnumFeature"
    PixelColorFilter: "EnumFeature"
    ReverseX: "BoolFeature"
    ReverseY: "BoolFeature"
    TestPattern: "EnumFeature"
    
    # Analog Control
    Gain: "FloatFeature"
    GainAuto: "EnumFeature"
    GainRaw: "IntFeature"
    BlackLevel: "FloatFeature"
    BlackLevelRaw: "IntFeature"
    Gamma: "FloatFeature"
    ExposureTime: "FloatFeature"
    ExposureAuto: "EnumFeature"
    ExposureTimeRaw: "IntFeature"
    
    # Digital IO Control
    LineSelector: "EnumFeature"
    LineMode: "EnumFeature"
    LineInverter: "BoolFeature"
    LineStatus: "BoolFeature"
    LineStatusAll: "IntFeature"
    LineSource: "EnumFeature"
    LineDebouncerTime: "FloatFeature"
    UserOutputSelector: "EnumFeature"
    UserOutputValue: "BoolFeature"
    UserOutputValueAll: "IntFeature"
    
    # Device Information
    DeviceVendorName: "StringFeature"
    DeviceModelName: "StringFeature"
    DeviceSerialNumber: "StringFeature"
    DeviceVersion: "StringFeature"
    DeviceFirmwareVersion: "StringFeature"
    DeviceUserID: "StringFeature"
    DeviceTemperature: "FloatFeature"
    
    # Transport Layer Control
    PayloadSize: "IntFeature"
    
    # GigE Vision specific attributes
    GevCurrentIPAddress: "IntFeature"
    GevCurrentSubnetMask: "IntFeature"
    GevCurrentDefaultGateway: "IntFeature"
    GevDeviceModeIsBigEndian: "BoolFeature"
    GevSCPSPacketSize: "IntFeature"
    GevSCPD: "IntFeature"  # Packet delay
    GevSCFTD: "IntFeature" # Frame transmission delay
    GevSCBWR: "IntFeature" # Bandwidth reserve
    GevSCBWRA: "IntFeature" # Bandwidth reserve accumulation
    
    # USB3 Vision specific attributes
    U3vVersionMajor: "IntFeature"
    U3vVersionMinor: "IntFeature"
    U3vDeviceCapability: "IntFeature"
    U3vMaxRequestTransferLength: "IntFeature"
    U3vMaxAckTransferLength: "IntFeature"
    
    # Bandwidth Control
    DeviceLinkThroughputLimitMode: "EnumFeature"
    DeviceLinkThroughputLimit: "IntFeature"
    DeviceLinkCurrentThroughput: "IntFeature"
    
    # Auto Function Control
    BalanceWhiteAuto: "EnumFeature"
    BalanceRatioSelector: "EnumFeature"
    BalanceRatio: "FloatFeature"
    ColorTransformationSelector: "EnumFeature"
    ColorTransformationValue: "FloatFeature"
    ColorTransformationValueSelector: "EnumFeature"
    
    # Color Correction
    ColorCorrectionMatrix: "FloatFeature"
    ColorCorrectionMatrixSelector: "EnumFeature"
    
    # Counter and Timer Control
    CounterSelector: "EnumFeature"
    CounterEventSource: "EnumFeature"
    CounterValue: "IntFeature"
    CounterValueAtReset: "IntFeature"
    CounterDuration: "IntFeature"
    CounterDelay: "IntFeature"
    CounterTriggerSource: "EnumFeature"
    CounterTriggerActivation: "EnumFeature"
    CounterReset: Any  # Command feature
    
    # Timer Control
    TimerSelector: "EnumFeature"
    TimerDuration: "FloatFeature"
    TimerDelay: "FloatFeature"
    TimerTriggerSource: "EnumFeature"
    TimerTriggerActivation: "EnumFeature"
    
    # Event Control
    EventSelector: "EnumFeature"
    EventNotification: "EnumFeature"
    
    # LUT Control
    LUTSelector: "EnumFeature"
    LUTEnable: "BoolFeature"
    LUTIndex: "IntFeature"
    LUTValue: "IntFeature"
    
    # Sequencer Control (if available)
    SequencerMode: "EnumFeature"
    SequencerSetActive: "IntFeature"
    SequencerSetNext: "IntFeature"
    SequencerSetSelector: "IntFeature"
    SequencerSetStart: "IntFeature"
    SequencerSetLoad: Any  # Command feature
    SequencerSetSave: Any  # Command feature
    
    # Stream parameters
    StreamChannelCount: "IntFeature"
    
    # Data stream access (indexed like a list/dict)
    data_stream: "DataStreamCollection"
    
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
    
    def get_data_stream(self, stream_id: int = 0) -> "DataStream":
        """Get data stream by ID"""
        ...
    
    def send_command(self, feature_id: int) -> None:
        """Send command to camera"""
        ...
    
    def get_device_info(self) -> GxDeviceBaseInfo:
        """Get device information"""
        ...
    
    def is_open(self) -> bool:
        """Check if device is open"""
        ...
    
    def export_config_file(self, file_path: str) -> None:
        """Export camera configuration to file"""
        ...
    
    def import_config_file(self, file_path: str, verify: bool = False) -> None:
        """Import camera configuration from file"""
        ...

    # Feature access methods
    def get_int_feature(self, feature_id: Union[int, str]) -> "IntFeature":
        """Get integer feature control"""
        ...
    
    def get_float_feature(self, feature_id: Union[int, str]) -> "FloatFeature":
        """Get float feature control"""
        ...
    
    def get_enum_feature(self, feature_id: Union[int, str]) -> "EnumFeature":
        """Get enumeration feature control"""
        ...
    
    def get_bool_feature(self, feature_id: Union[int, str]) -> "BoolFeature":
        """Get boolean feature control"""
        ...
    
    def get_string_feature(self, feature_id: Union[int, str]) -> "StringFeature":
        """Get string feature control"""
        ...
    
    def get_buffer_feature(self, feature_id: Union[int, str]) -> "BufferFeature":
        """Get buffer feature control"""
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

    def get_image(self, timeout: int = 1000) -> Optional[RawImage]:
        """Get image with timeout, returns RawImage or None if timeout/error"""
        ...
    
    def start_acquisition(self, acquisition_flag: int, callback_func: Optional[Callable] = None) -> None:
        """Start acquisition with optional callback"""
        ...
    
    def stop_acquisition(self) -> None:
        """Stop acquisition"""
        ...
    
    def get_payload_size(self) -> int:
        """Get payload size"""
        ...
    
    def get_delivered_frame_count(self) -> int:
        """Get number of delivered frames"""
        ...
    
    def get_lost_frame_count(self) -> int:
        """Get number of lost frames"""
        ...
    
    def get_incomplete_frame_count(self) -> int:
        """Get number of incomplete frames"""
        ...
    
    def get_delivery_loop_count(self) -> int:
        """Get delivery loop count"""
        ...
    
    def reset_statistics(self) -> None:
        """Reset stream statistics"""
        ...

# Data Stream Collection class (for indexed access)
class DataStreamCollection:
    """Collection of data streams accessible by index"""
    
    def __getitem__(self, index: int) -> DataStream:
        """Get data stream by index"""
        ...
    
    def __len__(self) -> int:
        """Get number of data streams"""
        ...

class intfeaturerange:
    min:int
    max:int
    inc:int

# Feature control classes
class IntFeature:
    """Integer feature control"""
    
    def get(self) -> int:
        """Get current integer value"""
        ...
    
    def set(self, int_value: int) -> None:
        """Set integer value"""
        ...
    
    def get_range(self) -> intfeaturerange:
        """Get (min, max, increment) values"""
        ...
    
    def get_min(self) -> int:
        """Get minimum value"""
        ...
    
    def get_max(self) -> int:
        """Get maximum value"""
        ...
    
    def get_inc(self) -> int:
        """Get increment value"""
        ...
    
    def is_readable(self) -> bool:
        """Check if feature is readable"""
        ...
    
    def is_writable(self) -> bool:
        """Check if feature is writable"""
        ...
    
    def is_implemented(self) -> bool:
        """Check if feature is implemented"""
        ...

class floatfeaturerange(TypedDict):
    unit:Literal["us","ms"]

class FloatFeature:
    """Float feature control"""
    
    def get(self) -> float:
        """Get current float value"""
        ...
    
    def set(self, float_value: float) -> None:
        """Set float value"""
        ...
    
    def get_range(self) -> floatfeaturerange:
        """Get (min, max) values"""
        ...
    
    def get_min(self) -> float:
        """Get minimum value"""
        ...
    
    def get_max(self) -> float:
        """Get maximum value"""
        ...
    
    def is_readable(self) -> bool:
        """Check if feature is readable"""
        ...
    
    def is_writable(self) -> bool:
        """Check if feature is writable"""
        ...
    
    def is_implemented(self) -> bool:
        """Check if feature is implemented"""
        ...

class EnumFeature:
    """Enumeration feature control"""
    
    def get(self) -> int:
        """Get current enum value"""
        ...
    
    def set(self, enum_value: int) -> None:
        """Set enum value"""
        ...
    
    def get_range(self) -> dict[str,Any]:
        """Get list of valid enum values"""
        ...
    
    def get_symbol_dict(self) -> dict[int, str] | None:
        """Get dictionary mapping enum values to symbol names"""
        ...
    
    def is_readable(self) -> bool:
        """Check if feature is readable"""
        ...
    
    def is_writable(self) -> bool:
        """Check if feature is writable"""
        ...
    
    def is_implemented(self) -> bool:
        """Check if feature is implemented"""
        ...

class BoolFeature:
    """Boolean feature control"""
    
    def get(self) -> bool:
        """Get current boolean value"""
        ...
    
    def set(self, bool_value: bool) -> None:
        """Set boolean value"""
        ...
    
    def is_readable(self) -> bool:
        """Check if feature is readable"""
        ...
    
    def is_writable(self) -> bool:
        """Check if feature is writable"""
        ...
    
    def is_implemented(self) -> bool:
        """Check if feature is implemented"""
        ...

class StringFeature:
    """String feature control"""
    
    def get(self, max_size: int = 1024) -> str:
        """Get current string value"""
        ...
    
    def set(self, string_value: str) -> None:
        """Set string value"""
        ...
    
    def get_max_length(self) -> int:
        """Get maximum string length"""
        ...
    
    def is_readable(self) -> bool:
        """Check if feature is readable"""
        ...
    
    def is_writable(self) -> bool:
        """Check if feature is writable"""
        ...
    
    def is_implemented(self) -> bool:
        """Check if feature is implemented"""
        ...

class BufferFeature:
    """Buffer feature control"""
    
    def get(self, max_size: int = 1024) -> bytes:
        """Get current buffer value"""
        ...
    
    def set(self, buffer_value: bytes) -> None:
        """Set buffer value"""
        ...
    
    def get_length(self) -> int:
        """Get buffer length"""
        ...
    
    def get_max_length(self) -> int:
        """Get maximum buffer length"""
        ...
    
    def is_readable(self) -> bool:
        """Check if feature is readable"""
        ...
    
    def is_writable(self) -> bool:
        """Check if feature is writable"""
        ...
    
    def is_implemented(self) -> bool:
        """Check if feature is implemented"""
        ...

# Common feature IDs (based on GenICam standard)
class GxFeatureID:
    """Common feature ID constants"""
    
    # Device information
    DEVICE_VENDOR_NAME: int
    DEVICE_MODEL_NAME: int
    DEVICE_SERIAL_NUMBER: int
    DEVICE_VERSION: int
    DEVICE_FIRMWARE_VERSION: int
    DEVICE_USER_ID: int
    
    # Acquisition control
    ACQUISITION_MODE: int
    ACQUISITION_START: int
    ACQUISITION_STOP: int
    ACQUISITION_FRAME_COUNT: int
    ACQUISITION_FRAME_RATE: int
    TRIGGER_MODE: int
    TRIGGER_SOURCE: int
    TRIGGER_ACTIVATION: int
    TRIGGER_SOFTWARE: int
    
    # Image format control
    SENSOR_WIDTH: int
    SENSOR_HEIGHT: int
    WIDTH_MAX: int
    HEIGHT_MAX: int
    OFFSET_X: int
    OFFSET_Y: int
    WIDTH: int
    HEIGHT: int
    PIXEL_FORMAT: int
    PIXEL_SIZE: int
    PIXEL_COLOR_FILTER: int
    
    # Analog control
    GAIN: int
    GAIN_AUTO: int
    BLACKLEVEL: int
    EXPOSURE_TIME: int
    EXPOSURE_AUTO: int
    
    # Transport layer control
    PAYLOAD_SIZE: int
    
    # GigE specific
    GEV_CURRENT_IP_ADDRESS: int
    GEV_CURRENT_SUBNET_MASK: int
    GEV_CURRENT_DEFAULT_GATEWAY: int
    PACKET_SIZE: int
    PACKET_DELAY: int
    
    # Stream parameters
    BLOCK_TIMEOUT: int
    
# Utility functions
def gx_get_last_error() -> Tuple[int, str]:
    """Get last error code and description"""
    ...

def gx_get_lib_version() -> str:
    """Get library version string"""
    ...

def gx_enum_device_type_list() -> List[int]:
    """Enumerate available device types"""
    ...

# Image processing utilities
class ImageProcessor:
    """Image processing utilities"""
    
    @staticmethod
    def dx_raw8_to_rgb24(raw_image: RawImage, rgb_image: Any, pixel_format: int) -> int:
        """Convert 8-bit raw image to RGB24"""
        ...
    
    @staticmethod
    def dx_raw16_to_rgb48(raw_image: RawImage, rgb_image: Any, pixel_format: int) -> int:
        """Convert 16-bit raw image to RGB48"""
        ...
    
    @staticmethod
    def dx_mono8_to_rgb24(mono_image: RawImage, rgb_image: Any) -> int:
        """Convert mono8 image to RGB24"""
        ...
    
    @staticmethod
    def dx_get_color_filter(pixel_format: int) -> int:
        """Get color filter from pixel format"""
        ...
    
    @staticmethod
    def dx_image_improvement(raw_image: RawImage, rgb_image: Any, param: GxImageImprovementParam) -> int:
        """Apply image improvement algorithms"""
        ...
    
    @staticmethod
    def dx_calc_cc_param(raw_image: RawImage, param: GxImageImprovementParam) -> int:
        """Calculate color correction parameters"""
        ...
    
    @staticmethod
    def dx_auto_white_balance(raw_image: RawImage, param: GxWhiteBalanceParam, roi: Optional[GxAutoWhiteBalanceROI] = None) -> int:
        """Perform automatic white balance"""
        ...
    
    @staticmethod
    def dx_get_gamma_lut(gamma_value: float) -> List[int]:
        """Get gamma correction lookup table"""
        ...
    
    @staticmethod
    def dx_get_contrast_lut(contrast_value: int) -> List[int]:
        """Get contrast correction lookup table"""
        ...

# Network utilities (for GigE cameras)
class GigEUtils:
    """GigE Vision camera utilities"""
    
    @staticmethod
    def gx_enum_interface() -> Tuple[int, List[dict]]:
        """Enumerate network interfaces"""
        ...
    
    @staticmethod
    def gx_force_ip(mac_address: str, ip_address: str, subnet_mask: str, gateway: str) -> int:
        """Force IP configuration on GigE camera"""
        ...
    
    @staticmethod
    def gx_get_interface_by_ip(ip_address: str) -> Optional[dict]:
        """Get interface information by IP address"""
        ...

# Chunk data support
class GxChunkDataList:
    CHUNK_FRAME_ID: int
    CHUNK_TIMESTAMP: int
    CHUNK_EXPOSURE_TIME: int
    CHUNK_GAIN: int
    CHUNK_BLACK_LEVEL: int

class ChunkData:
    """Chunk data from image"""
    
    def get_chunk_data(self, chunk_id: int) -> Optional[Any]:
        """Get chunk data by ID"""
        ...
    
    def is_chunk_data_available(self, chunk_id: int) -> bool:
        """Check if chunk data is available"""
        ...

# Version info
class GxVersionInfo:
    """Library version information"""
    def __init__(self) -> None:
        self.major: int
        self.minor: int  
        self.build: int
        self.revision: int

# Device event handling
class GxEventData:
    """Device event data"""
    def __init__(self) -> None:
        self.event_id: int
        self.timestamp: int
        self.data: Any

EventCallbackFunction = Callable[[GxEventData], None]