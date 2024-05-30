# seafront
microscope software stack

this software is mainly intended as an open-source interface for the Cephla SQUID microscope, specifically the HCS version.

The latest official interface is written in python with Qt GUI, which suffers from a few drawbacks (like no builtin scaling for different display resolutions, display code integrated into low level code, etc.).

This new interface has a backend in python, with a web interface on top. This allows microscope interaction over a network, and more fundamental separation of low level control from display functionality.

# notes

This software establishes a usb connection to two cameras simultaneously, which may require more usb stack memory than allowed by the operating system by default. on linux, this memory limit can be queried with ```bash cat /sys/module/usbcore/parameters/usbfs_memory_mb```. The camera manufacturers recommend setting this to 1000. This can be done, effective until next reboot, with (```bash echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb```), or effectively forever, by adding that line to ``` /etc/rc.local```. If this number is too low, connection to a camera my fail with an error like the following: ``` Exception: Device.stream_on:{-1010}{CDSStartAcquisitionAgency:line[860]}{{-1010}{StartAcquisition:line[537]}{TL Error:Unable to start acquisition.}}```, crashing the software.