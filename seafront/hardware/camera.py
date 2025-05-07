import typing as tp
from enum import Enum
import time
import numpy as np

from seaconfig import AcquisitionChannelConfig
from gxipy import gxiapi

from ..config.basics import GlobalConfigHandler
from ..logger import logger

gxiapi.gx_init_lib()

class AcquisitionMode(str,Enum):
    """
        set acquisition mode of microscope

        modes:
            ON_TRIGGER - default mode, which is expected to be most reactive -> lowest latency, lowest throughput (40-50ms overhead)

            CONTINUOUS - highest latency, highest throughput (expect 500ms latency before first image, and 500ms after last image before return)
    """

    ON_TRIGGER="on_trigger"
    CONTINUOUS="continuous"

class Camera:
    device_manager=gxiapi.DeviceManager()

    @staticmethod
    def get_all()->tp.List["Camera"]:
        dev_num,dev_info_list=Camera.device_manager.update_all_device_list()

        ret=[]

        if dev_num > 0:
            assert dev_info_list is not None
            for dev_info in dev_info_list:
                cam=Camera(dev_info)
                ret.append(cam)

        return ret

    def __init__(self,device_info:tp.Dict[str,str]):
        self.device_info=device_info

        self.vendor_name:str=device_info["vendor_name"]
        self.model_name:str=device_info["model_name"]
        self.sn:str=device_info["sn"]
        self.device_type=None

        self.handle=None

        self.acquisition_ongoing=False

    def open(self,device_type:tp.Literal["main","autofocus"]):
        """
            open device for interaction
        """

        for i in range(5):
            try:
                Camera.device_manager.update_all_device_list()
                self.handle=Camera.device_manager.open_device_by_sn(self.sn)
            except Exception as e:
                logger.debug(f"camera - opening failed {i} times {e=}")

                try:
                    self.close()
                except Exception as e:
                    logger.warning(f"camera - opening failed with error {e=}. retrying. ")

                time.sleep(2)
                logger.debug("camera - done sleeping to await camera response on open")
                continue

            logger.debug("opened camera")
            break

        if self.handle is None:
            logger.critical(f"failed to open camera {self.device_info}")
            raise RuntimeError(f"failed to open camera {self.device_info}")
        
        self.device_type=device_type

        self.acquisition_ongoing=False

        # prints some camera hardware information
        if False:
            print(f" - pixel formats:")
            pixel_formats=self.handle.PixelFormat.get_range()
            for pf in pixel_formats:
                print(f"   - {pf}")

            widths=self.handle.Width.get_range()
            print( " - width:")
            print(f"   - min {widths['min']}")
            print(f"   - max {widths['max']}")
            print(f"   - inc {widths['inc']}")

            heights=self.handle.Height.get_range()
            print( " - height:")
            print(f"   - min {heights['min']}")
            print(f"   - max {heights['max']}")
            print(f"   - inc {heights['inc']}")

            if self.handle.ExposureTime.is_readable():
                for e in self.handle.ExposureTime.get_range().items():
                    print(f" - exposure time {e}")

            if self.handle.Gain.is_readable():
                for g in self.handle.Gain.get_range().items():
                    print(f" - gain {g}")
            
            if self.handle.BinningHorizontal.is_readable():
                for bh in self.handle.BinningHorizontal.get_range().items():
                    print(f" - binning horizontal {bh}")

            if self.handle.BinningVertical.is_readable():
                for bv in self.handle.BinningVertical.get_range().items():
                    print(f" - binning vertical {bv}")

            if self.handle.TriggerSource.is_readable():
                for ts in self.handle.TriggerSource.get_range().items():
                    print(f" - trigger source {ts}")

        # notable features:
        #  - AcquisitionMode: both cameras support continuous and single frame mode. both
        #                     modes are broadly similar (stream on, take image, stream off),
        #                     but single frame mode only allows taking a single image with a
        #                     running stream. continuous allows any number. there is no clear
        #                     downside to continuous mode. latency to stream on/off is the same.
        #  - TriggerMode: can be ON/OFF, i.e. trigger needs to be sent to trigger image acquisition
        #  - TriggerActivation: rising/falling edge - unused in this code

        # set software trigger
        #self.camera.TriggerMode.set(gx.GxSwitchEntry.ON)
        #self.camera.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)        

        # set hardware trigger
        #self.camera.TriggerMode.set(gx.GxSwitchEntry.ON)
        #self.camera.TriggerSource.set(gx.GxTriggerSourceEntry.LINE2)
        # self.camera.TriggerSource.set(gx.GxTriggerActivationEntry.RISING_EDGE) # this was commented out by hongquan, but why?

        # these features seem to have no effect, or I cannot find a way to use them properly
        #   - AcquisitionFrameRateMode
        #   - AcquisitionFrameRate

        # the following features only have one valid value for either camera:
        #   - ExposureMode = Timed
        #   - GainSelector = AnalogAll

        # the following features are not supported by either camera:
        #   - ExposureDelay
        #   - ExposureTimeMode

        # turn off device link throughput limit
        self.handle.DeviceLinkThroughputLimitMode.set(gxiapi.GxSwitchEntry.OFF)

        # set acquisition mode
        self.handle.AcquisitionMode.set(gxiapi.GxAcquisitionModeEntry.CONTINUOUS)

        # set pixel format
        self.handle.PixelFormat.set(gxiapi.GxPixelFormatEntry.MONO8)
        self.pixel_format=gxiapi.GxPixelFormatEntry.MONO8

        self.is_streaming=False
        self.acq_mode=None
        self._set_acquisition_mode(AcquisitionMode.ON_TRIGGER)

        # turning stream on takes 25ms for continuous mode, 20ms for single frame mode
        self.is_streaming=True
        self.handle.stream_on()

    def _set_acquisition_mode(
        self,
        acq_mode:AcquisitionMode,
        with_cb:tp.Optional[tp.Callable[[gxiapi.RawImage],None]]=None,
        continuous_target_fps:tp.Optional[float]=None
    ):
        """
            set acquisition mode
        """
        assert self.handle is not None

        if self.acq_mode==acq_mode:
            return
        
        # ensure stream is off, and unregister all registered callbacks
        was_streaming=self.is_streaming
        if was_streaming:
            self.handle.stream_off()
        self.handle.data_stream[0].unregister_capture_callback()
        
        match acq_mode:
            case AcquisitionMode.ON_TRIGGER:
                if with_cb is not None:
                    raise RuntimeError("callback is not allowed for on trigger mode")
                
                # enable trigger
                self.handle.TriggerMode.set(gxiapi.GxSwitchEntry.ON)
                self.handle.TriggerSource.set(gxiapi.GxTriggerSourceEntry.SOFTWARE)

                # disable framerate limit
                self.handle.AcquisitionFrameRateMode.set(gxiapi.GxSwitchEntry.OFF)

            case AcquisitionMode.CONTINUOUS:
                if with_cb is None:
                    raise RuntimeError("callback must be set for continuous mode")
                
                # disable trigger
                self.handle.TriggerMode.set(gxiapi.GxSwitchEntry.OFF)

                # limit framerate
                self.handle.AcquisitionFrameRateMode.set(gxiapi.GxSwitchEntry.ON)
                if continuous_target_fps is not None:
                    self.handle.AcquisitionFrameRate.set(continuous_target_fps)
                
                self.handle.data_stream[0].register_capture_callback(with_cb)

        # set acquisition mode
        self.handle.AcquisitionMode.set(gxiapi.GxAcquisitionModeEntry.CONTINUOUS)

        # turning stream on takes 25ms for continuous mode, 20ms for single frame mode
        if was_streaming:
            self.handle.stream_on()

        self.acq_mode=acq_mode
        
    def close(self):
        """
            close device handle
        """
        logger.debug("camera - closing")

        if self.handle is None:
            return

        # turning stream off takes 300ms (for continuous and single frame mode)
        self.is_streaming=False

        # this may throw if the device is already offline or just not streaming
        try:self.handle.stream_off()
        except:pass

        try:self.handle.close_device()
        except:pass

        self.handle=None

        logger.debug("closed camera")

    def _exposure_time_ms_to_native(self,exposure_time_ms:float):
        """
            convert exposure time from ms to the native unit of the camera
        """
        assert self.handle is not None

        exposure_time_native_unit=exposure_time_ms
        exposure_time_range=self.handle.ExposureTime.get_range()
        assert exposure_time_range is not None
        match exposure_time_range['unit']:
            case 'us':
                exposure_time_native_unit*=1e3
            case 'ms':
                pass
            case _:
                raise RuntimeError(f"unhandled unit {exposure_time_range['unit']}")
        
        return exposure_time_native_unit

    def acquire_with_config(
        self,
        config:AcquisitionChannelConfig,
        mode:tp.Literal["once","until_stop"]="once",
        callback:tp.Optional[tp.Callable[[gxiapi.RawImage],bool]]=None,
        target_framerate_hz:float=5.0
    )->tp.Optional[np.ndarray]:
        """
            acquire image with given configuration

            mode:
                - once: acquire a single image. see AcquisitionMode.ON_TRIGGER
                - until_stop: acquire images until callback returns True. see AcquisitionMode.CONTINUOUS

            returns:
                - np.ndarray of image data if mode is "once"
                - None if mode is "until_stop"
        """

        assert self.handle is not None

        if self.acquisition_ongoing:
            logger.warning("camera - requested acquisition while one was already ongoing. returning None.")
            return None

        # set pixel format
        g_config=GlobalConfigHandler.get_dict()
        match self.device_type:
            case "main":
                pixel_format_item=g_config["main_camera_pixel_format"]
            case "autofocus":
                pixel_format_item=g_config["laser_autofocus_pixel_format"]
            case _:
                raise RuntimeError(f"unsupported device type {self.device_type}")

        assert pixel_format_item is not None
        pixel_format=pixel_format_item.value
        format_is_supported=len([k for k in self.handle.PixelFormat.get_range().keys() if k.lower()==pixel_format.lower()])>0 # type: ignore
        if not format_is_supported:
            raise RuntimeError(f"unsupported pixel format {pixel_format}")

        # pixel format change is not possible while streaming
        # turning the stream off takes nearly half a second, so we cache the current pixel format
        # and only pause streaming to change it, if necessary
        match pixel_format:
            case "mono8":
                if self.pixel_format!=gxiapi.GxPixelFormatEntry.MONO8:
                    logger.debug("camera - perf warning - changing pixel format to mono8")
                    self.handle.stream_off()
                    self.pixel_format=gxiapi.GxPixelFormatEntry.MONO8
                    self.handle.PixelFormat.set(gxiapi.GxPixelFormatEntry.MONO8)
                    self.handle.stream_on()
            case "mono10":
                if self.pixel_format!=gxiapi.GxPixelFormatEntry.MONO10:
                    logger.debug("camera - perf warning - changing pixel format to mono10")
                    self.handle.stream_off()
                    self.pixel_format=gxiapi.GxPixelFormatEntry.MONO10
                    self.handle.PixelFormat.set(gxiapi.GxPixelFormatEntry.MONO10)
                    self.handle.stream_on()
            case "mono12":
                if self.pixel_format!=gxiapi.GxPixelFormatEntry.MONO12:
                    logger.debug("camera - perf warning - changing pixel format to mono12")
                    self.handle.stream_off()
                    self.pixel_format=gxiapi.GxPixelFormatEntry.MONO12
                    self.handle.PixelFormat.set(gxiapi.GxPixelFormatEntry.MONO12)
                    self.handle.stream_on()
            case _:
                raise RuntimeError(f"unsupported pixel format {pixel_format}")

        # takes ca 10ms
        exposure_time_native_unit=self._exposure_time_ms_to_native(config.exposure_time_ms)
        self.handle.ExposureTime.set(exposure_time_native_unit)

        # takes ca 8ms
        self.handle.Gain.set(config.analog_gain)

        self.acquisition_ongoing=True

        match mode:
            case "once":
                assert self.handle is not None
                # send command to trigger acquisition
                self.handle.TriggerSoftware.send_command()

                # wait for image to arrive
                img:gxiapi.RawImage|None=self.handle.data_stream[0].get_image()

                if img is None:
                    self.acquisition_ongoing=False
                    return None

                match img.get_status():
                    case gxiapi.GxFrameStatusList.INCOMPLETE:
                        self.acquisition_ongoing=False
                        raise RuntimeError("incomplete frame")
                    case gxiapi.GxFrameStatusList.SUCCESS:
                        pass

                np_img=img.get_numpy_array()
                assert np_img is not None
                np_img=np_img.copy()

                self.acquisition_ongoing=False

                return np_img
            
            case "until_stop":

                assert callback is not None
                stop_acquisition=False
                def run_callback(img:gxiapi.RawImage):
                    nonlocal stop_acquisition

                    if stop_acquisition or img is None: # type: ignore
                        if self.acquisition_ongoing:
                            self.acquisition_ongoing=False

                        return
                    
                    img_status=img.get_status()
                    match img_status:
                        case gxiapi.GxFrameStatusList.INCOMPLETE:
                            raise RuntimeError("incomplete frame")
                        case gxiapi.GxFrameStatusList.SUCCESS: 
                            pass

                    stop_acquisition=callback(img)
                    
                    if stop_acquisition:
                        if self.acquisition_ongoing:
                            self.acquisition_ongoing=False

                        return


                # adjust target framerate for acquisition overhead
                # ... this is not quite right, but yields better results than no adjustment
                s_per_frame=1.0/target_framerate_hz
                target_framerate_hz=1.0/(s_per_frame-max(20-config.exposure_time_ms,0)*1e-3)
                self._set_acquisition_mode(AcquisitionMode.CONTINUOUS,with_cb=run_callback,continuous_target_fps=target_framerate_hz)
