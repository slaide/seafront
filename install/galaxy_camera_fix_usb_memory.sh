# from their linux install documentation:

# 1.When using USB3.0 Vision Camera,If you need grab image from 4 or more U3V cameras, or you need increasing the package(URB) size or count, you will 
# likely run out of kernel space and see corresponding error messages on the console. Because of the default value of USB Kernel Space set by the kernel 
# is 16 MB. To set the value (in this example to 1000 MB) you can
#  execute as root:
echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb

# insufficient memory may also lead to an error at runtime, like
# Exception: Device.stream_on:{-1010}{CDSStartAcquisitionAgency:line[860]}{{-1010}{StartAcquisition:line[537]}{TL Error:Unable to start acquisition.}}
# which is fixed by this same command. i have not tested which number would be sufficient below 1000.