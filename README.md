# seafront
microscope software stack

this software is mainly intended as an open-source interface for the [Cephla SQUID](https://cephla.com/product/squid/) microscope, specifically the HCS version.

The latest official interface is written in python with Qt GUI, which suffers from a few drawbacks (like no builtin scaling 
for different display resolutions, display code integrated into low level code, etc.).

This new interface has a backend in python, with a web interface on top. This allows microscope interaction over a network, 
and more fundamental separation of low level control from display functionality.

note: issues encountered during operation or installation should be reported in this repository.

# setup workflow

1. **Install seafront** - Download, install dependencies, and set up environment ([details](#install))
2. **List hardware** - Identify connected cameras and devices
   ```bash
   uv run python scripts/list_squid_hardware.py
   ```
3. **Configure microscope** - Copy example config and adjust for your hardware ([details](#configuration))
   ```bash
   cp examples/squid_config.json ~/seafront/config.json
   # Edit ~/seafront/config.json with your camera models and settings
   ```
4. **Generate default protocol** - Create required startup configuration
   ```bash
   uv run python scripts/generate_default_protocol.py --microscope "your-microscope-name"
   ```
5. **Run seafront** - Start the server with your microscope configuration
   ```bash
   uv run python -m seafront --microscope "your-microscope-name"
   ```

# install

lines with `# instruction:` need to be followed manually.
lines with `# note:` contain information that may be acted upon.

the computer needs to be connected to the internet during setup.

```sh
# instruction: unplug microscope

# download seafront
cd ~
git clone https://github.com/slaide/seafront

# clone daheng imaging sdk repo (required for camera support)
cd ~
git clone https://github.com/slaide/daheng-imaging-gxipy

# install daheng imaging sdk (architecture will be detected automatically)
cd ~/daheng-imaging-gxipy/install_sdk

ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    cd linux-x86
    bash install.run
elif [ "$ARCH" = "aarch64" ]; then
    cd linux-arm64
    bash install.run
else
    echo "Unsupported architecture: $ARCH"
    echo "Please check the available install scripts in install_sdk/"
    exit 1
fi

cd ~

# set up python environment, install dependencies and seafront software
cd ~/seafront
bash install/all.sh

# instruction: plug microscope power in, wait for motors to engage (makes a 'clonk' sound)
# instruction: plug in microscope via usb (all cameras + microcontroller. i recommend using an external usb hub to simplify this.)

# installation complete - continue with hardware listing and configuration (see setup workflow above)
```

## configuration

**Note:** Follow the [setup workflow](#setup-workflow) for the complete configuration process including hardware detection.

See `examples/` directory for complete configuration examples:
- **`examples/squid_config.json`**: Standard SQUID microscope with Galaxy cameras and power calibration
- **`examples/squid+_config.json`**: SQUID+ microscope with ToupCam main camera and filter wheel
- **`examples/mock_config.json`**: Mock microscope for development and testing

Place your configuration file at `~/seafront/config.json` (JSON5 format supported).

### basic configuration structure
```json
{
    "port": 5002,
    "microscopes": [
        {
            "microscope_name": "unnamed HCS SQUID",
            "main_camera_model": "MER2-1220-32U3M", 
            "main_camera_driver": "galaxy",
            "base_image_output_dir": "/home/scientist/seafront/images",
            
            "laser_autofocus_available": "yes",
            "laser_autofocus_camera_model": "MER2-630-60U3M",
            "laser_autofocus_camera_driver": "galaxy",
            
            "filter_wheel_available": "no",
            "calibration_offset_x_mm": 0.0,
            "calibration_offset_y_mm": 0.0, 
            "calibration_offset_z_mm": 0.0,
            "forbidden_wells": "{\"384\":[\"A01\",\"A24\",\"P01\",\"P24\"]}",
            
            "channels": "[{\"name\": \"BF LED matrix full\", \"handle\": \"bfledfull\", \"source_slot\": 0}]",
            "filters": "[]"
        }
    ]
}
```
### configuration parameters

- **`"<x>_camera_model"`**: Camera model names passed to the camera API - must be specific! Use `scripts/list_squid_hardware.py` to find exact model names.
- **`"<x>_camera_driver"`**: Camera API to use (`"galaxy"` for Daheng cameras, `"toupcam"` for ToupTek cameras).
- **`"microscope_name"`**: Any string, used as metadata.
- **`"microscope_type"`**: Hardware implementation (`"squid"` for real hardware, `"mock"` for simulation). Defaults to `"squid"`.
- **`"base_image_output_dir"`**: Parent directory for image storage.
- **`"channels"`**: JSON string defining available imaging channels (see [Channel Configuration](#channel-configuration)).
- **`"filters"`**: JSON string defining filter wheel configuration (empty array `"[]"` if no filter wheel).

Image storage path format: `<base_image_output_dir>/<project name>/<plate name>/<unique acquisition id>_<acquisition start timestamp>`

### channel configuration

Channels define the available imaging modalities and their illumination sources. Each channel in the `"channels"` JSON string has these properties:

```json
{
    "name": "BF LED matrix full",              // Display name in interface
    "handle": "bfledfull",              // Internal identifier (must be unique)
    "source_slot": 0,                   // Illumination source slot (0-6: LED matrix, 11-15: lasers)
    "use_power_calibration": true,      // Enable power calibration
    "power_calibration": {              // Optional: calibration data
        "dac_percent": [0, 25, 50, 75, 100],
        "optical_power_mw": [0.0, 5.0, 20.0, 45.0, 80.0]
    }
}
```

**Hardware mapping**:
- **Slots 0-6**: LED matrix sources (brightfield)
- **Slots 11-15**: Laser sources (fluorescence)

The software automatically detects LED matrix vs laser sources and applies appropriate control methods. See [Power Calibration](#power-calibration-for-illumination-sources) for details on calibrated illumination control.

### mock microscope

Mock microscope implementation simulates hardware operations without physical devices.

Features:
- Simulates hardware operations without physical microscope
- Generates synthetic images with channel-specific patterns
- Supports all microscope commands
- Exposure time and analog gain scaling
- Streaming at 6 FPS
- Movement simulation at 4cm/s with position updates

#### enabling mock mode

Add `"microscope_type": "mock"` to your microscope configuration:

```json
{
    "port": 5002,
    "microscopes": [
        {
            "microscope_name": "Mock SQUID for Testing",
            "microscope_type": "mock",
            "base_image_output_dir": "/home/scientist/seafront/images",
            "calibration_offset_x_mm": 0.0,
            "calibration_offset_y_mm": 0.0,
            "calibration_offset_z_mm": 0.0,
            "channels": "[{\"name\": \"BF LED matrix full\", \"handle\": \"bfledfull\", \"source_slot\": 0}]",
            "filters": "[]"
        }
    ]
}
```

Camera and microcontroller fields are ignored in mock mode.

#### timing control

Environment variable `MOCK_NO_DELAYS`:
- `MOCK_NO_DELAYS=1`: Instant operations
- Default: Realistic delays (4cm/s movement, exposure time delays)

```bash
# Instant operations
MOCK_NO_DELAYS=1 uv run python -m seafront --microscope "Mock SQUID for Testing"

# Realistic timing (default)
uv run python -m seafront --microscope "Mock SQUID for Testing"
```

### additional parameters

- **`"calibration_offset_<x|y|z>"`**: Should be _mostly_ correct. Minor deviations between microscope restarts can be fine-tuned in the web interface during acquisition.
- **`"forbidden_wells"`**: Hardware conflict prevention. JSON string listing wells where the objective might crash into the XY-stage. Format: `{"<num_wells>": ["<well_name>", ...]}`. Wells not listed are allowed. This depends on the objective's working distance.

## power calibration for illumination sources

Seafront supports calibrated power control for illumination sources to compensate for non-linear response curves (especially important for LED sources). This ensures consistent optical power output regardless of the underlying hardware characteristics.

### configuration

Power calibration is configured per-channel in the `channels` configuration. Each channel can optionally include calibration data:

```json
{
  "name": "BF LED matrix full",
  "handle": "bfledfull", 
  "source_slot": 0,
  "use_power_calibration": true,
  "power_calibration": {
    "dac_percent": [0, 25, 50, 75, 100],
    "optical_power_mw": [0.0, 5.0, 20.0, 45.0, 80.0]
  }
}
```

**Parameters:**
- `use_power_calibration`: Set to `true` to enable calibrated power control, `false` for linear scaling
- `power_calibration`: Calibration data with matching arrays of DAC percentages and measured optical power
- `dac_percent`: DAC output percentages (0-100) used during calibration measurement
- `optical_power_mw`: Corresponding measured optical power in milliwatts

### how it works

1. **Without calibration**: 25% intensity request → 25% DAC output → unpredictable optical power
2. **With calibration**: 25% intensity request → lookup table interpolation → correct DAC output → 25% of maximum measured optical power

**Important**: Users still request intensity as percentages (0-100%), but the calibration ensures these percentages correspond to consistent optical power output.

For the example above, requesting 25% intensity would:
- Look up 25% of max measured power (80.0 mW) = 20.0 mW target  
- Find that 20.0 mW requires 50% DAC output (from calibration data)
- Send 50% to the hardware instead of 25%
- Achieve consistent 20.0 mW optical output (25% of max)

### hardware behavior

**LED Matrix Sources** (brightfield LEDs, slots 0-6):
- Power is controlled via RGB brightness values
- Calibration adjusts RGB intensity sent to hardware
- Uses `SET_ILLUMINATION_LED_MATRIX` microcontroller command

**Regular Sources** (lasers, slots 11-15): 
- Power is controlled via intensity percentage
- Calibration adjusts intensity percentage sent to hardware  
- Uses `SET_ILLUMINATION` microcontroller command

### creating calibration data

1. Set up your illumination source with a power meter
2. Measure optical power at different DAC settings (0%, 25%, 50%, 75%, 100%)
3. Record the DAC percentage and corresponding optical power in mW
4. Add the calibration data to your channel configuration
5. Set `use_power_calibration: true` for that channel
6. Restart seafront to load the new calibration

**Example measurement process:**
```
DAC 0%  → measure power → 0.0 mW
DAC 25% → measure power → 5.0 mW  
DAC 50% → measure power → 20.0 mW
DAC 75% → measure power → 45.0 mW
DAC 100% → measure power → 80.0 mW
```

## default protocol configuration

**Note:** This is step 4 in the [setup workflow](#setup-workflow).

Seafront requires a default protocol configuration file at startup. This file provides default acquisition settings and wellplate configuration that the interface loads when first started.

### generating default configuration

A default protocol must be present before starting the software. Protocols are stored in microscope-specific directories to prevent compatibility issues when multiple microscopes share the same server.

**Protocol Generation:**
```bash
# Generate with default 384-well plate for specific microscope
uv run python scripts/generate_default_protocol.py --microscope "your-microscope-name"

# List available wellplate types
uv run python scripts/generate_default_protocol.py --list

# Generate with specific wellplate and microscope
uv run python scripts/generate_default_protocol.py --plate revvity-384-6057800 --microscope "your-microscope-name"
```

**Note:** The `--microscope` argument is now required for protocol generation to ensure proper isolation between different microscope configurations.

The script automatically uses wellplate specifications from the seaconfig library, ensuring accurate dimensions and well layouts for supported plate types.

### what it contains

The default protocol provides initial settings for channel configurations including default power levels, exposure times, and channel-filter mappings. This serves as the starting point that users customize through the web interface for their specific experiments.

### propagating new default settings

When you update the microscope configuration with new channel names or other defaults (like the recent change from "BF LED Full" to "BF LED matrix full"), follow these steps to propagate the changes to existing installations:

1. **Update machine config**: Modify the default channel configurations in `seafront/config/basics.py` or update your local `~/seafront/config.json`

2. **Delete existing default protocol**: Remove the old protocol file to force regeneration with new settings
   ```bash
   rm ~/seafront/acquisition_configs/default.json
   # OR remove microscope-specific protocol
   rm ~/seafront/acquisition_configs/your-microscope-name/default.json
   ```

3. **Generate new default protocol**: Use the script to create a fresh protocol with updated channel names
   ```bash
   uv run python scripts/generate_default_protocol.py --microscope "your-microscope-name"
   ```

4. **Load protocol in GUI**: Start seafront and verify the new channel names appear correctly in the interface
   ```bash
   uv run python -m seafront --microscope "your-microscope-name"
   ```

5. **Reset machine config values**: In the web interface, go to "Machine Config" → "reset all values" to ensure all configuration values match the updated defaults

6. **Reload interface**: Refresh (not hard reload) the browser page to apply all changes

7. **Done**: The interface now uses the updated channel names (e.g., "BF LED matrix full" instead of "BF LED Full") and any other configuration changes

# run

**Note:** This is step 5 in the [setup workflow](#setup-workflow).

After completing installation, configuration, and protocol generation, start the seafront server:

```bash
# Start seafront with your microscope configuration
uv run python -m seafront --microscope "your-microscope-name"

# For mock microscope (development/testing)
uv run python -m seafront --microscope "Mock SQUID for Testing"

# With environment variable for instant mock operations
MOCK_NO_DELAYS=1 uv run python -m seafront --microscope "Mock SQUID for Testing"
```

The web interface will be available at `http://localhost:5002` (or the port specified in your config).

## browser configuration persistence

The web interface automatically saves your current configuration to browser localStorage for persistence across page reloads. This includes:

- Project information (name, plate name, cell line, comments)
- Well selections and grid settings  
- Channel configurations (exposure times, power levels, etc.)
- Theme preferences

**Important notes:**
- Browser storage is temporary and may be cleared by browser updates or cache clearing
- For permanent storage, save your configuration as a named protocol using the "Store Config" button
- When closing/refreshing the page, the browser will warn you if your current protocol has not been saved to the server (indicated by the unchecked "protocol saved?" status)

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

# development scripts

a few utility scripts are available in the `scripts/` directory for development and troubleshooting:

## scripts/check.py
runs type checking (pyright) and linting (ruff) on the codebase. use this before committing changes.
```bash
uv run python3 ./scripts/check.py
```

## scripts/list_squid_hardware.py  
comprehensive hardware listing script that shows all available SQUID microscope hardware including cameras, microcontrollers, and other peripherals. provides detailed information needed for configuration.
```bash
uv run python3 ./scripts/list_squid_hardware.py
```

**Important**: when configuring camera models in your `~/seafront/config.json`:
- **Galaxy cameras**: use the `Model` field from the hardware listing output
- **ToupCam cameras**: use the `Display Name` field from the hardware listing output

For example, if the hardware listing shows:
```
1. Driver: galaxy
   Vendor: Daheng Imaging
   Model: MER2-1220-32U3M
   Serial: ABC123

2. Driver: toupcam
   Vendor: ToupTek
   Display Name: ITR3CMOS26000KMA
   Serial: DEF456
```

Then your config should use:
- `"main_camera_model": "MER2-1220-32U3M"` for Galaxy cameras  
- `"main_camera_model": "ITR3CMOS26000KMA"` for ToupCam cameras

## scripts/test_filter_wheel_hardware.py
hardware test script specifically for filter wheel functionality. performs complete microcontroller initialization and cycles through all filter wheel positions (1-8) to verify proper operation.
```bash
uv run python3 ./scripts/test_filter_wheel_hardware.py
```
useful for:
- verifying filter wheel hardware after installation
- diagnosing filter wheel positioning issues
- testing microcontroller communication
- validating filter wheel initialization sequence

the script provides detailed logging of each step including movement calculations, command generation, and position verification.

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
