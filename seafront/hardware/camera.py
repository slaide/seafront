import typing as tp
from enum import Enum
import time
import numpy as np

from seaconfig import AcquisitionChannelConfig
from gxipy import gxiapi

from ..config.basics import GlobalConfigHandler

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
            for dev_info in dev_info_list:
                cam=Camera(dev_info)
                ret.append(cam)

        return ret

    def __init__(self,device_info:list):
        self.device_info=device_info

        self.vendor_name=device_info["vendor_name"]
        self.model_name=device_info["model_name"]
        self.sn=device_info["sn"]

        self.handle=None

    def open(self):
        """
            open device for interaction
        """

        self.handle=Camera.device_manager.open_device_by_sn(self.sn)
        if self.handle is None:
            raise RuntimeError(f"failed to open camera {self.device_info}")
        
        cam=self

        # prints some camera hardware information
        if False:
            print(f" - pixel formats:")
            pixel_formats=cam.handle.PixelFormat.get_range()
            for pf in pixel_formats:
                print(f"   - {pf}")

            widths=cam.handle.Width.get_range()
            print( " - width:")
            print(f"   - min {widths['min']}")
            print(f"   - max {widths['max']}")
            print(f"   - inc {widths['inc']}")

            heights=cam.handle.Height.get_range()
            print( " - height:")
            print(f"   - min {heights['min']}")
            print(f"   - max {heights['max']}")
            print(f"   - inc {heights['inc']}")

            if cam.handle.ExposureTime.is_readable():
                for e in cam.handle.ExposureTime.get_range().items():
                    print(f" - exposure time {e}")

            if cam.handle.Gain.is_readable():
                for g in cam.handle.Gain.get_range().items():
                    print(f" - gain {g}")
            
            if cam.handle.BinningHorizontal.is_readable():
                for bh in cam.handle.BinningHorizontal.get_range().items():
                    print(f" - binning horizontal {bh}")

            if cam.handle.BinningVertical.is_readable():
                for bv in cam.handle.BinningVertical.get_range().items():
                    print(f" - binning vertical {bv}")

            if cam.handle.TriggerSource.is_readable():
                for ts in cam.handle.TriggerSource.get_range().items():
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
        # self.camera.TriggerSource.set(gx.GxTriggerActivationEntry.RISING_EDGE) # this was commented out hongquan, but why?

        # these features seem to have no effect, or I cannot find a way to use them properly
        #   - AcquisitionFrameRateMode
        #   - AcquisitionFrameRate

        # the following features only have one valid value for either camera:
        #   - ExposureMode = Timed
        #   - GainSelector = AnalogAll

        # the following features are not supported by either camera:
        #   - ExposureDelay
        #   - ExposureTimeMode

        self.acq_mode=None
        self._set_acquisition_mode(AcquisitionMode.ON_TRIGGER)

        # set trigger source
        self.handle.TriggerMode.set(gxiapi.GxSwitchEntry.ON)
        self.handle.TriggerSource.set(gxiapi.GxTriggerSourceEntry.SOFTWARE)

        # set acquisition mode
        self.handle.AcquisitionMode.set(gxiapi.GxAcquisitionModeEntry.CONTINUOUS)

        # turning stream on takes 25ms for continuous mode, 20ms for single frame mode
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

        if self.acq_mode==acq_mode:
            return
        
        # ensure stream is off, and unregister all registered callbacks
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
        self.handle.stream_on()

        self.acq_mode=acq_mode
        
    def close(self):
        """
            close device handle
        """

        # turning stream off takes 300ms (for continuous and single frame mode)
        self.handle.stream_off()

        self.handle.close_device()
        self.handle=None

    def _exposure_time_ms_to_native(self,exposure_time_ms:float):
        """
            convert exposure time from ms to the native unit of the camera
        """

        exposure_time_native_unit=exposure_time_ms
        match self.handle.ExposureTime.get_range()['unit']:
            case 'us':
                exposure_time_native_unit*=1e3
            case 'ms':
                pass
            case _:
                raise RuntimeError(f"unhandled unit {self.handle.ExposureTime.get_range()['unit']}")
        
        return exposure_time_native_unit

    def acquire_with_config(
        self,
        config:AcquisitionChannelConfig,
        mode:tp.Literal["once","until_stop"]="once",
        callback:tp.Optional[tp.Callable[[gxiapi.RawImage],bool]]=None
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

        # takes ca 10ms
        exposure_time_native_unit=self._exposure_time_ms_to_native(config.exposure_time_ms)
        self.handle.ExposureTime.set(exposure_time_native_unit)

        # takes ca 8ms
        self.handle.Gain.set(config.analog_gain)

        match mode:
            case "once":
                # send command to trigger acquisition
                self.handle.TriggerSoftware.send_command()

                # wait for image to arrive
                img:gxiapi.RawImage=self.handle.data_stream[0].get_image()
                match img.get_status():
                    case gxiapi.GxFrameStatusList.INCOMPLETE:
                        raise RuntimeError("incomplete frame")
                    case gxiapi.GxFrameStatusList.SUCCESS: 
                        pass

                np_img=img.get_numpy_array()
                return np_img
            
            case "until_stop":

                stop_acquisition=False
                def run_callback(img:gxiapi.RawImage):
                    nonlocal stop_acquisition
                    if stop_acquisition:
                        return
                    stop_acquisition=callback(img)

                target_fps=5.0
                # adjust target framerate for acquisition overhead
                # ... this is not quite right, but yields better results than no adjustment
                s_per_frame=1.0/target_fps
                target_fps=1.0/(s_per_frame-max(20-config.exposure_time_ms,0)*1e-3)
                self._set_acquisition_mode(AcquisitionMode.CONTINUOUS,with_cb=run_callback,continuous_target_fps=target_fps)

                while not stop_acquisition:
                    # sleep for duration of exposure time, but no more than 100ms
                    time.sleep(min(0.1,config.exposure_time_ms*1e-3))

                self._set_acquisition_mode(AcquisitionMode.ON_TRIGGER)
        