#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "matplotlib",
# ]
# ///

import ctypes, time, json
import numpy as np
import matplotlib.pyplot as plt

import toupcam.toupcam as tc

from typing import TypedDict
from pydantic import BaseModel

us2ms=1e-3
ms2us=1e3

from enum import IntEnum
class HRESULT(IntEnum):
    S_OK=0x00000000
    " 	Success 		"
    S_FALSE=0x00000001
    " 	Yet another success 	Different from S_OK, such as internal values and user-set values have coincided, equivalent to noop 	"
    E_ACCESSDENIED=0x80070005
    " 	Permission denied 	The program on Linux does not have permission to open the USB device, please enable udev rules file or run as root 	"
    E_INVALIDARG=0x80070057
    " 	One or more arguments are not valid 		"
    E_NOTIMPL=0x80004001
    " 	Not supported or not implemented 	This feature is not supported on this model of camera 	"
    E_POINTER=0x80004003
    " 	Pointer that is not valid 	Pointer is NULL 	"
    E_UNEXPECTED=0x8000ffff
    " 	Catastrophic failure 	Generally indicates that the conditions are not met, such as calling put_Option setting some options that do not support modification when the camera is running, and so on 	"
    E_WRONG_THREAD=0x8001010e
    " 	Call function in the wrong thread 	See here, here, here 	"
    E_GEN_FAILURE=0x8007001f
    " 	Device not functioning 	It is generally caused by hardware errors, such as cable problems, USB port problems, poor contact, camera hardware damage, etc 	"
    E_BUSY=0x800700aa
    " 	The requested resource is in use 	The camera is already in use, such as duplicated opening/starting the camera, or being used by other application, etc 	"
    E_PENDING=0x8000000a
    " 	The data necessary to complete this operation is not yet available 	No data is available at this time 	"
    E_TIMEOUT=0x8001011f
    " 	This operation returned because the timeout period expired 		"
    E_FAIL=0x80004005
    " 	Unspecified failure 		"

class ToupcamCamera:
    @staticmethod
    def open(cam_handle:int):
        pass

all_cameras=tc.Toupcam.EnumV2()
if len(tc.Toupcam.EnumV2())==0:
    raise RuntimeError("no cameras found")

try:
    for cam_info in all_cameras:
        print(f"cam: {cam_info.id} {cam_info.displayname}")
        for res in cam_info.model.res:
            print(res.width,res.height)

        cam=tc.Toupcam.Open(cam_info.id)
        if cam is None:
            raise RuntimeError("could not open camera")
        
        cam.put_eSize(0)

        print("auto expo on?",cam.get_AutoExpoEnable())
        cam.put_AutoExpoEnable(0)
        print("auto expo on?",cam.get_AutoExpoEnable())

        exposure_time_ms=500.0
        analog_gain=20

        # this naming is stupid
        print(f"exposure time limits: {cam.get_ExpTimeRange()}")
        print("exposure time ms:",cam.get_ExpoTime()*us2ms)
        cam.put_ExpoTime(int(exposure_time_ms*ms2us))
        print("exposure time ms:",cam.get_ExpoTime()*us2ms)

        print("analog gain range:",cam.get_ExpoAGainRange())

        print("analog gain:",cam.get_ExpoAGain())
        cam.put_ExpoAGain(100+analog_gain)
        print("analog gain:",cam.get_ExpoAGain())

        cam.put_Option(tc.TOUPCAM_OPTION_RAW,1)

        try:
            print(f"TOUPCAM_OPTION_EXPOSURE_PRE_DELAY {cam.get_Option(tc.TOUPCAM_OPTION_EXPOSURE_PRE_DELAY)}")
        except:
            pass
        try:
            print(f"TOUPCAM_OPTION_EXPOSURE_PRE_DELAY {cam.get_Option(tc.TOUPCAM_OPTION_EXPOSURE_POST_DELAY)}")
        except:
            pass

        # put into trigger mode with
        cam.put_Option(tc.TOUPCAM_OPTION_TRIGGER,1)

        nimages=1
        nimages_pulled=0

        class Context(BaseModel):
            nimages:int
            nimages_pulled:int
            finished:bool

        ctx=Context(nimages=nimages,nimages_pulled=0,finished=False)

        def pullmodecallback(i:int,ctx:Context):
            if cam is None: raise RuntimeError("cam is none!")
            
            # no clue what should happen here
            print(f"pullmodecallback called {i}")

            if i==4:
                imagedata=bytes(cam_info.model.res[0].width*cam_info.model.res[0].height)
                
                frameinfo=tc.ToupcamFrameInfoV4()
                cam.PullImageV4(imagedata,0,8,0,frameinfo)

                ctx.nimages_pulled+=1
                if ctx.nimages_pulled==ctx.nimages:
                    cam.Trigger(0)
                    ctx.finished=True

        cam.StartPullModeWithCallback(pullmodecallback,ctx)

        # camera does not support push mode...

        """
        always startpullmodewithcallback

        then, either:
            trigger(n) -> calls callback when image done, pullimage inside callback
        or:
            triggersync() -> triggers immediate acquisition (no callback involved)
        """

        # else sync
        camera_trgger_async=True

        if camera_trgger_async:

            start=time.time()
            # we only receive n-3 images..??
            cam.Trigger(1)

            while True:
                time.sleep(0.05)
                if ctx.finished:
                    break

            print(f"received image with {exposure_time_ms}ms exposure time in {(time.time()-start)*1e3}ms")

        else:

            for k in range(nimages):
                imagedata=bytes(cam_info.model.res[0].width*cam_info.model.res[0].height)
                
                frameinfo=tc.ToupcamFrameInfoV4()

                start=time.time()
                cam.TriggerSyncV4(0,imagedata,8,0,frameinfo)
                print(f"received image with {exposure_time_ms}ms exposure time in {(time.time()-start)*1e3}ms")
                
                if False:
                    # Convert bytes to numpy array and reshape to image dimensions
                    width = cam_info.model.res[0].width
                    height = cam_info.model.res[0].height
                    image_array = np.frombuffer(imagedata, dtype=np.uint8).reshape((height, width))*16
                    
                    # Display the image
                    plt.figure(figsize=(10, 8))
                    plt.imshow(image_array, cmap='gray')
                    plt.title(f'Image {k+1} - {width}x{height}')
                    plt.colorbar()
                    plt.axis('off')
                    plt.show()

        cam.Close()

except Exception as e:
    try:
        error_code=HRESULT(e.args[0]&0xFFFFFFFF)
        print(f"{error_code.name}")
    except:
        pass
    raise e from None
