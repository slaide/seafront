# seafront
microscope software stack

this software is mainly intended as an open-source interface for the [Cephla SQUID](https://cephla.com/product/squid/) microscope, specifically the HCS version.

The latest official interface is written in python with Qt GUI, which suffers from a few drawbacks (like no builtin scaling 
for different display resolutions, display code integrated into low level code, etc.).

This new interface has a backend in python, with a web interface on top. This allows microscope interaction over a network, 
and more fundamental separation of low level control from display functionality.

note: issues encountered during operation or installation should be reported in this repository.

# install

lines with `# instruction:` need to be followed manually.
lines with `# note:` contain information that may be acted upon.

the computer needs to be connected to the internet during setup.

```sh
# instruction: unplug microscope

# download code
git clone https://github.com/slaide/seafront
cd seafront

# install udev rule to enable communication with microscope
cd install
sudo bash teensyduino_udev_rules.sh
sudo bash galaxy_camera_fix_usb_memory.sh # read this file for an additional comment
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

example config file:
```
$ cat seafront/config.json 
{
    "main_camera_model": "MER2-1220-32U3M",
    "laser_autofocus_camera_model": "MER2-630-60U3M",
    "microscope_name": "HCS SQUID - dev env",
    "base_image_output_dir": "/storageservermount/squidimages",
    "laser_autofocus_available": "yes",
    "calibration_offset_x_mm": 2.44,
    "calibration_offset_y_mm": 0.44,
    "calibration_offset_z_mm": 0.0,
    "forbidden_wells": "96:;384:A01,A24,P01,P24;1536:"
}
```
- the `"<x>_camera_model"` name strings are passed to the camera api to connect to the camera, so they need to be specific!
- the `"microscope_name"` can be any string, and will be passed as metadata.
- the `"base_image_output_dir"` is the parent container of the directories where the images are actually stored.
    the path of an image after acquisition is: 
    `<base_image_output_dir>/<project name>/<plate name>/<unique acquisition id>_<acqusition start timestamp>`.

    note: the `<unique acquisition id>` is currently hard-coded because i have not had a good idea in how to generate a unique id.
- the `"calibration_offset_<x|y|z>"` should be _mostly_ correct. there can be minor deviations between microscope restarts, so there is 
a setting in the seafront web interface to fine-tune the values used during acquisition, so the value in this config file does not 
need to be changed all the time.
- the `"forbidden_wells"` string has a specific format: because of hardware conflicts (the objective being able to crash into the
XY-stage), there is a list of wells that the microscope is not allowed to enter. this list will depend on the objective used
(an objective with a larger working distance from the bottom of the plate may not conflict with as many positions). the format is not
json itself because I did not think about that when i implemented it (pull request welcome), so it is a semicolon separated list of
`<num wells on plate>:<comma separated list of well names>`.

# calibration

the one major calibration step required to run this software properly is to calibrate the XY stage position. it requires a 384 well 
plate.
instruction:
1. with a config file present (offset values present, but may be zero)
2. put the top left corner (as seen in the interface) of well B02 (imagine square well) into the *center* of the field of view
3. with the top left corner in the center of the field of view, use the "calibrate top left of B2 here" button in the 
"low level config" menu in the interface to get the offset. (the calibration offset values are visible in the same "low level config"
interface, right underneath the "calibrate top left of B2 here" button). the values in this menu are ephemeral, so put the values
into the calibration file to store them permanently. when the software is restarted, the values in the interface are initialized
to the values in the calibration file.

while we are here, this "low level config" menu has several other entries. these values are for a microscope engineer to change, hence
the missing details on the exact functionality (there is no tooltip pop up on these items by design). to an engineer, they should be
self explanatory, and for an end user, these should remain unchanged.

some values not present in the Cephla SQUID software are those related to "streaming preview" and "full image display format".
sine this software uses a web frontend, the images may be sent over a network, so bandwidth and browser display performance can be
an issue. the "streaming preview resolution scaling" reduced the resolution on each axis of an image by that factor before sending
it to the browser for display during an ongoing acquisition. "full image display" describes the image format for cases where no new
images are currently being acquired, e.g. after using "snap", or when an acquisiton or streaming (formerly called "Live") has finished.

the grayed out values in this list are for informational purposes only, and are inherent to the hardware, so they cannot be changed
in this interface at all, like the microscope name (should be set during installation in the config file, then remain unchanged), 
the camera model names or the presence of a laser autofocus system.

the values in this menu are used for an aquisition, or any other interaction with the microscope made through this interface.
changes made in this interface remain active until the interface is reset, restarting the microscope will not change these!
(these values are stored in the browser, even across page reloads).

the helper function "microscopeConfigReset()" can be used in the browser terminal to reset the values to the defaults sent from
the microscope server.

# run

```sh
# enter seafront repo directory
cd seafront
# activate python environment
source python_env/activate.sh
# run software
python3 -m seafront
# note: this is optional! deactivate the python environment with
source python_env/deactivate.sh
```

# documentation

this server uses the fastapi framework, which is able to automatically generate openapi specs, which in turn can be used to generate client side code for a variety of languages (see [swagger editor](https://editor.swagger.io/)). This documentation (which is auto-generated upon server start) contains the doc strings from the python code, as well as type annotations, so it serves as automatic but also human-readable documentation. When the server is running, check `/docs` (uses the swagger UI, based on openapi.json), `/redoc` (uses redoc UI, based on openapi.json) and `/openapi.json` for the relevant pages.

# what even is a bug?

this software has a broad target audience: from student with no microscopy experience to microscope engineer, the software should support 
everyone in their work.

accordingly: there should be development and debugging features built into it, to assist an engineer in fast repairs, but the interface 
should also be clear enough to support someone new to microscopy in their microscopy endeavors.
this also means that broken functionality is a bug, but unclear or ambiguous functionality is also considered a bug (like missing 
documentation), since it can stop someone from effectively performing their work with this software.
layout improvements _may_ be considered a bug, including color choice, depending on their impact on the workflow.

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
