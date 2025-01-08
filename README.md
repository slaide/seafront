# seafront
microscope software stack

this software is mainly intended as an open-source interface for the [Cephla SQUID](https://cephla.com/product/squid/) microscope, specifically the HCS version.

The latest official interface is written in python with Qt GUI, which suffers from a few drawbacks (like no builtin scaling 
for different display resolutions, display code integrated into low level code, etc.).

This new interface has a backend in python, with a web interface on top. This allows microscope interaction over a network, 
and more fundamental separation of low level control from display functionality.

note: issues encountered during operation or installation should be reported in this repository.

# install

lines with ` # instruction:` need to be followed manually.
lines with ` # note:` contain information that may be acted upon.

the computer needs to be connected to the internet during setup.

```sh
# instruction: unplug microscope

# download code
git clone https://github.com/slaide/seafront
cd seafront

# install udev rule to enable communication with microscope
cd install
bash teensyduino_udev_rules.sh
cd ..

# set up python environment
cd python_env 
bash install.sh
cd ..
source python_env/activate.sh

# install python code and dependencies
# (installs the camera api as dependency)
python3 -m pip install .

# instruction: plug microscope power in, wait for motors to engage (makes a 'clonk' sound)
# instruction: plug in microscope via usb (all cameras + microcontroller. i recommend using an external usb hub to simplify this.)

# run software, via:
python3 -m seafront

# note: config file is now in ~/seafront/config.json (change and reload software to apply)
```

# run

```sh
# enter seafront repo directory
cd seafront
# activate python environment
source python_env/activate.sh
# run software
python3 -m seafront
# note: this is optional! deactivate the environment with
source python_env/deactivate.sh
```

# notes

This software establishes a usb connection to two cameras simultaneously, which may require more usb stack memory than
allowed by the operating system by default. on linux, this memory limit can be queried with
```bash cat /sys/module/usbcore/parameters/usbfs_memory_mb```. The camera manufacturers recommend setting this to 1000. 
This can be done, effective until next reboot, with (```bash echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb```), 
or effectively forever, by adding that line to ``` /etc/rc.local```. If this number is too low, connection to a camera may 
fail with an error like the following (crashing the software):
``` 
Exception: Device.stream_on:{-1010}{CDSStartAcquisitionAgency:line[860]}{{-1010}{StartAcquisition:line[537]}{TL Error:
Unable to start acquisition.}}
```
