import time
import typing as tp

import numpy as np
import toupcam.toupcam as tc
from seaconfig import AcquisitionChannelConfig

from seafront.config.basics import GlobalConfigHandler
from seafront.config.handles import CameraConfig, LaserAutofocusConfig
from seafront.hardware.camera import AcquisitionMode, Camera, HardwareLimitValue
from seafront.logger import logger
from pydantic import BaseModel, ConfigDict 

class toupcam_ctx():
    def __init__(self,msg:str="",ignore_error:bool=False):
        self.msg=msg
        self.ignore_error=ignore_error
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            if isinstance(exc_value.args[0],int):
                errval=exc_value.args[0]&0xffffffff
                logger.debug(f"Toupcam Error Code: {errval:0x}",self.msg)

        # raise exception ?
        return self.ignore_error
    
class measuretime():
    def __init__(self,msg:str,ignore_error:bool=False):
        self.msg=msg
        self.ignore_error=ignore_error
    
    def __enter__(self):
        self.time=time.time()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        logger.debug(f"{self.msg} took {time.time()-self.time}ms")

        # raise exception ?
        return self.ignore_error

class ImageContext(BaseModel):
    mode:tp.Literal["once","until_stop"]

    cam:"ToupCamCamera"

    image_ready:bool=False
    captured_image:np.ndarray|None=None
    stop_acquisition:bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    callback:tp.Callable[[np.ndarray],bool]|None=None

    bytes_per_pixel:int=8
    pullimage_pix_format:tp.Literal[24,32,48,8,16,64]=8
    dtype:type=np.uint8

def pullmode_callback(event_type: int, ctx: ImageContext) -> None:
    if event_type != tc.TOUPCAM_EVENT_IMAGE:
        return

    if not ctx.cam.handle:
        return
    
    if ctx.mode=="until_stop":
        if ctx.stop_acquisition:
            ctx.cam._acquisition_until_stop_cleanup()

            return
        
    try:
        buf = bytes(ctx.cam.width * ctx.cam.height * ctx.bytes_per_pixel)

        # Pull the image
        frame_info=tc.ToupcamFrameInfoV4()
        with toupcam_ctx("pullimagev4"):
            ctx.cam.handle.PullImageV4(buf, 0, ctx.pullimage_pix_format, 0, frame_info)
    
        ctx.captured_image = np.frombuffer(buf, dtype=ctx.dtype).reshape(
            (ctx.cam.height, ctx.cam.width)
        )

        ctx.captured_image = ctx.captured_image.copy()

        match ctx.mode:
            case "once":
                ctx.image_ready = True
            case "until_stop":
                # Call user callback with the image
                # Note: ToupCam doesn't have the same RawImage object as Galaxy
                # We'll pass the numpy array directly
                assert ctx.callback is not None, "no callback given for until_stop mode"
                ctx.stop_acquisition = ctx.callback(ctx.captured_image)

    except Exception as e:
        logger.error(f"Error capturing image: {e}")
        match ctx.mode:
            case "once":
                # Set to exit wait loop
                ctx.image_ready = True
            case "until_stop":
                ctx.stop_acquisition = True

class ToupCamCamera(Camera):
    """ToupCam Camera implementation using the toupcam SDK."""

    @staticmethod
    def get_all() -> list[Camera]:
        """Get all available ToupCam cameras."""
        devices = tc.Toupcam.EnumV2()

        ret: list[Camera] = []
        for device in devices:
            cam = ToupCamCamera(device)
            ret.append(cam)

        return ret

    def __init__(self, device_info:tc.ToupcamDeviceV2):
        super().__init__(device_info)

        self.vendor_name: str = "ToupTek"
        self.model_name: str = device_info.displayname
        self.sn: str = device_info.id  # ToupCam uses ID strings instead of serial numbers

        self.handle = None
        self.width = 0
        self.height = 0
        self.pixel_format = "mono8"
        self._original_device = device_info  # Store for hardware listing script

        self.ctx=ImageContext(mode="once",cam=self)
        self.pullmode_active=False

    def open(self, device_type: tp.Literal["main", "autofocus"]):
        """Open device for interaction."""

        for i in range(5):
            try:
                self.handle = tc.Toupcam.Open(self.device_info.id)
                if self.handle is None:
                    raise RuntimeError("Failed to open ToupCam camera")
                
            except Exception as e:
                logger.debug(f"toupcam - opening failed {i} times {e=}")

                try:
                    self.close()
                except Exception as e:
                    logger.warning(f"toupcam - opening failed with error {e=}. retrying.")

                time.sleep(2)
                logger.debug("toupcam - done sleeping to await camera response on open")
                continue

            logger.debug("opened toupcam camera")
            break

        if self.handle is None:
            logger.critical(f"failed to open toupcam camera {self.device_info}")
            raise RuntimeError(f"failed to open toupcam camera {self.device_info}")

        self.device_type = device_type
        self.acquisition_ongoing = False

        # Set default pixel format to RAW8 (monochrome 8-bit)
        self.pixel_format = "mono8"

        # set to largest size
        with toupcam_ctx("put esize"):
            self.handle.put_eSize(0)

        # Get camera resolution
        self.width, self.height = self.handle.get_Size()
        logger.debug(f"cam width {self.width} height {self.height}")

        # disable auto exposure
        with toupcam_ctx("put autoexpoenable"):
            self.handle.put_AutoExpoEnable(0)

        with toupcam_ctx("enable low noise mode",ignore_error=True):
            if (self._original_device.model.flag&tc.TOUPCAM_FLAG_LOW_NOISE)>0:
                self._set_toupcam_option(tc.TOUPCAM_OPTION_LOW_NOISE,1)
        with toupcam_ctx("disable low power consumption",ignore_error=True):
            self._set_toupcam_option(tc.TOUPCAM_OPTION_LOW_POWERCONSUMPTION,0)

        with toupcam_ctx("enable high zero padding"):
            # match galaxy camera behaviour (which pads high bits with zero)
            self._set_toupcam_option(tc.TOUPCAM_OPTION_ZERO_PADDING,0)

        with toupcam_ctx("get capabilities"):
            (valmin,valmax,valdef)=self.handle.get_ExpoAGainRange()
            logger.debug(f"analog gain range: {valmin}-{valmax}")

            (valmin,valmax,valdef)=self.handle.get_ExpTimeRange()
            logger.debug(f"exposure time range: {valmin*1e-3}ms-{valmax*1e-3}ms")

        with toupcam_ctx():
            self._set_toupcam_option(tc.TOUPCAM_OPTION_RAW,1)

        with toupcam_ctx("set single trigger mode"):
            supports_trigger_single=(self._original_device.model.flag&tc.TOUPCAM_FLAG_TRIGGER_SINGLE)!=0
            print(f"supports trigger single {supports_trigger_single}")
            # this is just a capability, not something you can enable..

        logger.debug(f"MaxBitDepth: {self.handle.MaxBitDepth()}")

        if False:
            # for debugging
            self.handle.put_Option(tc.TOUPCAM_OPTION_TESTPATTERN,3)

        # we use trigger for all modes
        with toupcam_ctx("put option"):
            self._set_toupcam_option(tc.TOUPCAM_OPTION_TRIGGER,1)

        self.acq_mode = None
        self.set_acquisition_mode_trigger()

        self._start_pullmode()

    def _stop_pullmode(self)->bool:
        """return true if stopped, otherwise false"""
        assert self.handle is not None
        if self.pullmode_active:
            self.handle.Stop()
            self.pullmode_active=False
            return True
        return False

    def _start_pullmode(self):
        """return true if started, otherwise false"""
        assert self.handle is not None
        if self.pullmode_active:
            return False
        
        self.handle.StartPullModeWithCallback(pullmode_callback,self.ctx)
        self.pullmode_active=True

        return True

    def set_acquisition_mode_trigger(self):
        self._set_acquisition_mode(AcquisitionMode.ON_TRIGGER)

    def stop_acquisition(self) -> None:
        """
        Force stop any ongoing acquisition.
        """
        if self.acquisition_ongoing:
            self.acquisition_ongoing = False
            logger.debug("toupcam camera - acquisition stopped")

    def _set_acquisition_mode(
        self,
        acq_mode: AcquisitionMode,
        with_cb: tp.Callable[[np.ndarray], bool] | None = None,
        continuous_target_fps: float | None = None,
    ):
        """Set acquisition mode."""
        if not self.handle:
            raise RuntimeError("Camera not opened")

        if self.acq_mode == acq_mode:
            return

        match acq_mode:
            case AcquisitionMode.ON_TRIGGER:
                if with_cb is not None:
                    raise RuntimeError("callback is not allowed for on trigger mode")
                
                self.ctx.mode="once"
                
                # ToupCam doesn't have explicit trigger modes like Galaxy
                # We'll use single-shot acquisition

            case AcquisitionMode.CONTINUOUS:
                if with_cb is None:
                    raise RuntimeError("callback must be set for continuous mode")

                self.ctx.mode="until_stop"
                self.ctx.callback=with_cb # type: ignore

        self.acq_mode = acq_mode

    def _set_toupcam_option(self,option,value)->bool:
        "return true if value has been changed, False if value was already set"

        assert self.handle is not None

        current_value=self.handle.get_Option(option)
        if current_value!=value:
            pullmodeisrunning=False
            if self._stop_pullmode():
                pullmodeisrunning=True

            self.handle.put_Option(option,value)

            if pullmodeisrunning:
                self._start_pullmode()

            return True
        return False

    def close(self):
        """Close device handle."""
        logger.debug("toupcam - closing")

        if self.handle is None:
            return

        try:
            self.handle.Close()
        except:
            pass

        self.handle = None
        logger.debug("closed toupcam camera")

    def get_exposure_time_limits(self) -> HardwareLimitValue:
        """Get camera's exposure time limits in milliseconds."""
        if not self.handle:
            raise RuntimeError("Camera not opened")
        
        valmin, valmax, valdef = self.handle.get_ExpTimeRange()
        # ToupCam returns exposure time in microseconds, convert to milliseconds
        return HardwareLimitValue(
            min=valmin * 1e-3,  # Convert µs to ms
            max=valmax * 1e-3,  # Convert µs to ms 
            step=0.1  # Step size in ms - reasonable default for ToupCam
        )

    def get_analog_gain_limits(self) -> HardwareLimitValue:
        """Get camera's analog gain limits in decibels.""" 
        if not self.handle:
            raise RuntimeError("Camera not opened")
            
        valmin, valmax, valdef = self.handle.get_ExpoAGainRange()
        logger.debug(f"ToupCam analog gain range: min={valmin}%, max={valmax}%, default={valdef}%")
        # ToupCam returns gain as percentage (100-10000%)
        # Convert to decibels: dB = 10 * log10(percentage / 100)
        
        # Handle edge cases that could cause NaN or invalid values
        # ToupCam typically returns 100-10000% range (0dB to 20dB)
        if valmin > 0:
            min_db = 10 * np.log10(valmin / 100)
        else:
            min_db = 0.0  # Default to 0dB if invalid min percentage
            
        if valmax > 0:
            max_db = 10 * np.log10(valmax / 100)
        else:
            # If max is invalid, we don't know the safe range - default both to 0
            min_db = 0.0
            max_db = 0.0
        
        # If calculated values are NaN (shouldn't happen with above logic, but safety check)
        if np.isnan(min_db):
            min_db = 0.0
        if np.isnan(max_db):
            # If max is NaN, we don't know the safe range - set both to 0  
            min_db = 0.0
            max_db = 0.0
        
        return HardwareLimitValue(
            min=min_db,
            max=max_db,
            step=0.1  # Step size in dB - reasonable default
        )

    def _exposure_time_ms_to_us(self, exposure_time_ms: float) -> int:
        """Convert exposure time from ms to microseconds (ToupCam native unit)."""
        return int(exposure_time_ms * 1000)

    def acquire_with_config(
        self,
        config: AcquisitionChannelConfig,
        mode: tp.Literal["once", "until_stop"] = "once",
        callback: tp.Callable[[np.ndarray], bool] | None = None,
    ) -> np.ndarray | None:
        """Acquire image with given configuration."""

        if not self.handle:
            raise RuntimeError("Camera not opened")

        if self.acquisition_ongoing:
            logger.warning(
                "toupcam - requested acquisition while one was already ongoing. returning None."
            )
            return None

        # Set pixel format based on configuration
        match self.device_type:
            case "main":
                pixel_format_item = CameraConfig.MAIN_PIXEL_FORMAT.value_item
            case "autofocus":
                pixel_format_item = LaserAutofocusConfig.CAMERA_PIXEL_FORMAT.value_item
            case _:
                raise RuntimeError(f"unsupported device type {self.device_type}")

        pixel_format = pixel_format_item.value

        # Map pixel format names to ToupCam constants
        toupcam_format = None
        if not isinstance(pixel_format, str):
            raise RuntimeError(f"unsupported pixel format {pixel_format}")
        
        pixel_format=pixel_format.lower()

        match pixel_format:
            case "mono8":
                # always supported
                toupcam_format = tc.TOUPCAM_PIXELFORMAT_RAW8

            case "mono10":
                toupcam_format = tc.TOUPCAM_PIXELFORMAT_RAW10
                if self._original_device.model.flag&tc.TOUPCAM_FLAG_RAW10 == 0:
                    raise RuntimeError(f"camera does not support 10bit depth")
            case "mono12":
                toupcam_format = tc.TOUPCAM_PIXELFORMAT_RAW12
                if self._original_device.model.flag&tc.TOUPCAM_FLAG_RAW12 == 0:
                    raise RuntimeError(f"camera does not support 12bit depth")
            case "mono14":
                toupcam_format = tc.TOUPCAM_PIXELFORMAT_RAW14
                if self._original_device.model.flag&tc.TOUPCAM_FLAG_RAW14 == 0:
                    raise RuntimeError(f"camera does not support 14bit depth")
            case "mono16":
                toupcam_format = tc.TOUPCAM_PIXELFORMAT_RAW16
                if self._original_device.model.flag&tc.TOUPCAM_FLAG_RAW16 == 0:
                    raise RuntimeError(f"camera does not support 16bit depth")
                
            case _:
                raise RuntimeError(f"unsupported pixel format {pixel_format}")

        self.pixel_format = pixel_format
        logger.debug(f"camera - pixel format is {pixel_format}")

        if True:
            if(self._original_device.model.flag&tc.TOUPCAM_FLAG_MONO):
                logger.debug("camera - supports mono mode")
            else:
                raise RuntimeError("camera does not support mono mode")

            if(self._original_device.model.flag&tc.TOUPCAM_FLAG_RAW10):
                logger.debug("camera - supports bitdepth 10")
            if(self._original_device.model.flag&tc.TOUPCAM_FLAG_RAW12):
                logger.debug("camera - supports bitdepth 12")
            if(self._original_device.model.flag&tc.TOUPCAM_FLAG_RAW14):
                logger.debug("camera - supports bitdepth 14")
            if(self._original_device.model.flag&tc.TOUPCAM_FLAG_RAW16):
                logger.debug("camera - supports bitdepth 16")

            if(self._original_device.model.flag&tc.TOUPCAM_FLAG_RAW10PACK):
                logger.debug("camera - supports bitdepth 10 (packed)")
            if(self._original_device.model.flag&tc.TOUPCAM_FLAG_RAW12PACK):
                logger.debug("camera - supports bitdepth 12 (packed)")

        self.acquisition_ongoing = True
        try:
            # Get image buffer size based on pixel format
            if toupcam_format == tc.TOUPCAM_PIXELFORMAT_RAW8:
                bytes_per_pixel = 1
                dtype=np.uint8
                # separate format!
                pullimage_pix_format=8

            elif toupcam_format == tc.TOUPCAM_PIXELFORMAT_RAW16:
                bytes_per_pixel = 2
                dtype=np.uint16
                # separate format!
                pullimage_pix_format=16

            else:
                raise ValueError(f"camera does not support format {toupcam_format}")
            

            with toupcam_ctx("put option TOUPCAM_OPTION_PIXEL_FORMAT"):
                self._set_toupcam_option(tc.TOUPCAM_OPTION_PIXEL_FORMAT,toupcam_format)

            logger.debug(f"camera - toupcam set pixel format to {toupcam_format} ({self.handle.get_Option(tc.TOUPCAM_OPTION_PIXEL_FORMAT)})")

            logger.debug(f"using toupcam_format {pullimage_pix_format} ({toupcam_format})")

            # TOUPCAM_OPTION_RGB for value explanations
            #with toupcam_ctx(f"set rgb to {bytes_per_pixel+1}"):
            #    self._set_toupcam_option(tc.TOUPCAM_OPTION_RGB, bytes_per_pixel+1)
            with toupcam_ctx(f"set bitdepth to {bytes_per_pixel-1}"):
                self._set_toupcam_option(tc.TOUPCAM_OPTION_BITDEPTH,bytes_per_pixel-1)

            # Set exposure time (ToupCam uses microseconds)
            exposure_us = self._exposure_time_ms_to_us(config.exposure_time_ms)
            context_msg=f"camera - toupcam exposure time set to {config.exposure_time_ms}ms (target: {exposure_us}us, real: {self.handle.get_RealExpoTime()}us)"
            with toupcam_ctx(context_msg):
                self.handle.put_ExpoTime(exposure_us)
            logger.debug(context_msg)

            # Set analog gain (ToupCam uses percentage, 100-10000%)
            # config.analog_gain is in dB, camera uses percentage
            # e.g. 0db->100%, 10db->1000%
            gain_factor=10**(config.analog_gain/10)
            gain_percent = int(gain_factor * 100)
            context_msg=f"camera - toupcam analog gain set to {config.analog_gain} ({gain_percent}%)"
            with toupcam_ctx(context_msg):
                self.handle.put_ExpoAGain(gain_percent)
            logger.debug(context_msg)

            self.ctx.bytes_per_pixel=bytes_per_pixel
            self.ctx.pullimage_pix_format=pullimage_pix_format
            self.ctx.dtype=dtype

            self.ctx.captured_image=None
            self.ctx.image_ready=False

            match mode:
                case "once":

                    try:

                        self._set_acquisition_mode(
                            acq_mode=AcquisitionMode.ON_TRIGGER
                        )

                        img_bytes=bytes(self.height*self.width*bytes_per_pixel)

                        # lower overhead than regular trigger for single acquisition
                        self.handle.TriggerSyncV4(
                            0,
                            img_bytes,
                            self.ctx.pullimage_pix_format,
                            0,
                            tc.ToupcamFrameInfoV4()
                        )

                        img_np=np.frombuffer(img_bytes, dtype=self.ctx.dtype).reshape(
                            (self.height, self.width)
                        )

                        return img_np

                    except Exception as e:
                        raise e
                    
                    finally:
                        self.acquisition_ongoing = False
                        self.handle.Trigger(0)

                case "until_stop":
                    if callback is None:
                        self.acquisition_ongoing = False
                        raise ValueError("callback must be provided for until_stop mode")

                    try:

                        self._set_acquisition_mode(
                            acq_mode=AcquisitionMode.CONTINUOUS,
                            with_cb=callback
                        )

                        # trigger until stop
                        self.handle.Trigger(0xffff)

                        # someone else calls _acquistion_until_stop_cleanup() later

                        return None

                    except Exception as e:
                        self._acquisition_until_stop_cleanup()
                        raise e

        except Exception as e:
            self.acquisition_ongoing = False
            raise e

    def _acquisition_until_stop_cleanup(self):
        assert self.handle is not None

        # Stop capture
        self.handle.Trigger(0)
        self.acquisition_ongoing = False
        self.ctx.stop_acquisition=False
