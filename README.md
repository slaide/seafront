# seafront

A headless-first microscope control framework for the [Cephla SQUID](https://cephla.com/product/squid/) microscope.

**Key features:**
- **Headless operation** - Full programmatic control via REST API, no GUI required
- **Optional web interface** - Browser-based GUI for interactive use
- **Network-accessible** - Control microscopes remotely over the network
- **Linux only** - Tested on Ubuntu 22.04+ (Windows support possible if needed, file an issue)

The architecture cleanly separates hardware control from display, enabling automation pipelines, remote operation, and custom integrations.

Report issues at: https://github.com/slaide/seafront/issues

# setup workflow

1. **Install seafront** - Download, install dependencies, and set up environment ([details](#install))
2. **List hardware** - Identify connected cameras and devices
   ```bash
   uv run python scripts/list_squid_hardware.py
   ```
3. **Configure microscope** - Copy example config and adjust for your hardware ([details](#configuration))
   ```bash
   cp examples/squid-v1.json ~/seafront/config.json
   # Edit ~/seafront/config.json with your camera models and settings
   ```
4. **Generate default protocol** - Create required startup configuration
   ```bash
   uv run python scripts/generate_default_protocol.py --microscope "squid"
   ```
5. **Run seafront** - Start the server with your microscope configuration
   ```bash
   uv run python -m seafront --microscope "squid"
   ```

# install

The installation process is automated using the `install/all.sh` script. This script handles downloading dependencies, installing camera drivers, and setting up the Python environment automatically.

**Prerequisites:**
- Computer with internet connection (required during setup)
- USB hub recommended (for connecting multiple microscope devices simultaneously)
- Administrator/sudo access (for driver and USB memory configuration)

**Installation Steps:**

```sh
# instruction: unplug microscope

# download seafront
cd ~
git clone https://github.com/slaide/seafront
cd ~/seafront

# instruction: plug microscope power in, wait for motors to engage (makes a 'clonk' sound)

# Install dependencies, drivers, and seafront software
bash install/all.sh

# instruction: plug in microscope via usb (all cameras + microcontroller. use an external usb hub if possible.)

# installation complete - continue with hardware listing and configuration (see setup workflow above)
```

The `install/all.sh` script automatically:
- Detects your system architecture
- Downloads and installs Galaxy camera SDK (if needed)
- Configures USB memory for camera connections
- Sets up the Python environment using uv package manager
- Installs all required dependencies

**Note:** The installation script uses `uv` to manage dependencies, which automatically handles Python version requirements. No manual Python installation is needed.

## configuration

**Note:** Follow the [setup workflow](#setup-workflow) for the complete configuration process including hardware detection.

### example configurations

See `examples/` directory for complete configuration examples:
- **`examples/squid-v1.json`** through **`examples/squid-v4.json`**: SQUID microscope configurations for different hardware revisions. Use the version matching your microscope.
- **`examples/mock_config.json`**: Mock microscope for development and testing without physical hardware. Use this for testing the software or learning the interface.

Copy the appropriate example to `~/seafront/config.json` and customize it for your hardware:
```bash
cp examples/squid-v1.json ~/seafront/config.json
# Edit ~/seafront/config.json with your specific settings
```

Place your configuration file at `~/seafront/config.json` (JSON5 format supported).

### basic configuration structure
```json
{
    "port": 5002,
    "microscopes": [
        {
            "system.microscope_name": "squid",
            "system.microscope_type": "squid",
            "camera.main.id": "ABC12345678",
            "camera.main.driver": "galaxy",
            "microcontroller.id": "DEF98765432",
            "storage.base_image_output_dir": "/home/scientist/seafront/images",

            "calibration.offset.x_mm": 0.0,
            "calibration.offset.y_mm": 0.0,
            "calibration.offset.z_mm": 0.0,

            "laser.autofocus.available": "yes",
            "laser.autofocus.camera.id": "GHI11223344",
            "laser.autofocus.camera.driver": "galaxy",

            "filter.wheel.available": "no",
            "filter.wheel.configuration": [],

            "imaging.channels": [
                {"name": "BF LED matrix full", "handle": "bfledfull", "source_slot": 0}
            ],

            "protocol.forbidden_areas": []
        }
    ]
}
```
### configuration parameters

Configuration keys use a dotted path syntax (e.g., `camera.main.id` instead of `main_camera_id`).

- **`"camera.main.id"`**: USB serial number of the main camera. Use `scripts/list_squid_hardware.py` to find this value (shown as `Serial` in the output).
- **`"camera.main.driver"`**: Camera API to use (`"galaxy"` for Daheng cameras, `"toupcam"` for ToupTek cameras).
- **`"microcontroller.id"`**: USB serial number of the microcontroller. Use `scripts/list_squid_hardware.py` to find this.
- **`"system.microscope_name"`**: Any string, used as metadata.
- **`"system.microscope_type"`**: Hardware implementation (`"squid"` for real hardware, `"mock"` for simulation). Defaults to `"squid"`.
- **`"storage.base_image_output_dir"`**: Parent directory for image storage.
- **`"imaging.channels"`**: Array of channel configurations (see [Channel Configuration](#channel-configuration)).
- **`"filter.wheel.configuration"`**: Array of filter configurations. Use empty array `[]` when `filter.wheel.available` is `"no"`.
- **`"protocol.forbidden_areas"`**: Array of forbidden area definitions for stage safety.

Image storage path format: `<base_image_output_dir>/<project name>/<plate name>/<unique acquisition id>_<acquisition start timestamp>`

### channel configuration

Channels define the available imaging modalities and their illumination sources. Each channel in the `"imaging.channels"` array has these properties:

```json
{
    "name": "BF LED matrix full",              // Display name in interface
    "handle": "bfledfull",              // Internal identifier (must be unique, alphanumeric + underscore)
    "source_slot": 0,                   // Illumination source slot (0-6: LED matrix, 11-15: lasers)
    "use_power_calibration": true,      // Enable power calibration
    "power_calibration": {              // Optional: calibration data
        "dac_percent": [0, 25, 50, 75, 100],
        "optical_power_mw": [0.0, 5.0, 20.0, 45.0, 80.0]
    }
}
```

**Channel Handle Requirements:**
- **Unique**: Each channel must have a distinct handle (no duplicates)
- **Identifier**: Used internally as the channel identifier throughout the system
- **Naming convention**: Use lowercase alphanumeric characters and underscores (e.g., `bfledfull`, `fluo405`, `fluo488`)

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

Set `"system.microscope_type": "mock"` in your microscope configuration:

```json
{
    "port": 5002,
    "microscopes": [
        {
            "system.microscope_name": "Mock SQUID for Testing",
            "system.microscope_type": "mock",
            "camera.main.id": "PLACEHOLDER",
            "camera.main.driver": "galaxy",
            "microcontroller.id": "PLACEHOLDER",
            "storage.base_image_output_dir": "/home/scientist/seafront/images",
            "calibration.offset.x_mm": 0.0,
            "calibration.offset.y_mm": 0.0,
            "calibration.offset.z_mm": 0.0,
            "laser.autofocus.available": "no",
            "filter.wheel.available": "no",
            "filter.wheel.configuration": [],
            "imaging.channels": [
                {"name": "BF LED matrix full", "handle": "bfledfull", "source_slot": 0}
            ],
            "protocol.forbidden_areas": []
        }
    ]
}
```

Camera and microcontroller ID fields can be placeholders in mock mode as they are not used.

#### timing control

The mock microscope can simulate realistic timing or run with instant operations:

**Environment variable `MOCK_NO_DELAYS`:**
- `MOCK_NO_DELAYS=1`: Instant operations (skip all delays)
- Default: Realistic delays (4cm/s movement, exposure time delays)

**Usage examples:**

```bash
# Instant operations - useful for rapid testing
MOCK_NO_DELAYS=1 uv run python -m seafront --microscope "mocroscope"

# Realistic timing (default) - simulates actual hardware behavior
uv run python -m seafront --microscope "mocroscope"
```

### additional parameters

- **`"calibration.offset.<x|y|z>_mm"`**: Should be _mostly_ correct. Minor deviations between microscope restarts can be fine-tuned in the web interface during acquisition.

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

A default protocol must be present before starting the software. Protocols are stored in microscope-specific directories (like `~/seafront/acquisition_configs/squid-5c27d4/`) to prevent compatibility issues when multiple microscopes share the same server.

**The `--microscope` argument is REQUIRED.** It determines which microscope configuration to use and where to store the protocol file.

**Protocol Generation:**
```bash
# Generate with default 384-well plate for your microscope (REQUIRED)
uv run python scripts/generate_default_protocol.py --microscope "your-microscope-name"

# List available microscopes from config
uv run python scripts/generate_default_protocol.py --list-microscopes

# List available wellplate types
uv run python scripts/generate_default_protocol.py --list-wellplates

# Generate with specific wellplate type and microscope
uv run python scripts/generate_default_protocol.py --plate revvity-384-6057800 --microscope "your-microscope-name"
```

The script automatically creates a microscope-specific subdirectory in `~/seafront/acquisition_configs/` based on your microscope name and stores the default protocol there. This approach ensures that if you switch microscopes or reconfigure your setup, each microscope maintains its own default protocol configuration.

The script automatically uses wellplate specifications from the seaconfig library, ensuring accurate dimensions and well layouts for supported plate types.

### what it contains

The default protocol provides initial settings for channel configurations including default power levels, exposure times, and channel-filter mappings. This serves as the starting point that users customize through the web interface for their specific experiments.

### propagating new default settings

When you update the microscope configuration with new channel names or other defaults, follow these steps to propagate the changes to existing installations:

1. **Update machine config**: Modify the configuration in your `~/seafront/config.json`

2. **Delete existing microscope-specific protocol**: Remove the old protocol file to force regeneration with new settings
   ```bash
   # Remove the microscope-specific protocol directory
   rm -rf ~/seafront/acquisition_configs/your-microscope-name/
   ```

3. **Generate new default protocol**: Use the script to create a fresh protocol with updated settings
   ```bash
   uv run python scripts/generate_default_protocol.py --microscope "your-microscope-name"
   ```

4. **Restart seafront**: Start the server which will load the new protocol
   ```bash
   uv run python -m seafront --microscope "your-microscope-name"
   ```

5. **Reset machine config values**: In the web interface, go to "Machine Config" → "reset all values" to ensure all configuration values match the updated defaults

6. **Reload interface**: Refresh (not hard reload) the browser page to apply all changes

7. **Done**: The interface now uses the updated configuration

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
since this software uses a web frontend, the images may be sent over a network, so bandwidth and browser display performance can be
an issue. the "streaming preview resolution scaling" reduces the resolution on each axis of an image by that factor before sending
it to the browser for display during an ongoing acquisition. "full image display" describes the image format for cases where no new
images are currently being acquired, e.g. after using "snap", or when an acquisition or streaming has finished.

the grayed out values in this list are for informational purposes only, and are inherent to the hardware, so they cannot be changed
in this interface at all, like the microscope name (should be set during installation in the config file, then remain unchanged), 
the camera model names or the presence of a laser autofocus system.

the values in this menu are used for an aquisition, or any other interaction with the microscope made through this interface.
changes made in this interface remain active until the interface is reset, restarting the microscope will not change these!
(these values are stored in the browser, even across page reloads).

the helper function "microscopeConfigReset()" can be used in the browser terminal to reset the values to the defaults sent from
the microscope server.


# documentation

This server uses the FastAPI framework, which automatically generates OpenAPI specifications that can be used to generate client-side code for a variety of languages (see [swagger editor](https://editor.swagger.io/)). The documentation is auto-generated upon server start and contains docstrings from the Python code as well as type annotations, providing both automatic and human-readable documentation.

**API Documentation Pages** (when the server is running):
- **`/docs`**: Interactive API documentation using Swagger UI (based on openapi.json)
- **`/redoc`**: Alternative API documentation using ReDoc UI (based on openapi.json)
- **`/openapi.json`**: Raw OpenAPI specification in JSON format

**WebSocket Endpoints**: Real-time data updates (image streaming, status updates, etc.) are provided via WebSocket endpoints. Check `/docs` for the complete list of available WebSocket endpoints and their functionality.

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

## running tests
runs the test suite including config registry tests and example config validation.
```bash
uv run pytest tests/ -v --tb=short
```

to measure test coverage:
```bash
uv run pytest tests/ --cov=seafront --cov-report=term-missing
```

## scripts/list_squid_hardware.py
comprehensive hardware listing script that shows all available SQUID microscope hardware including cameras, microcontrollers, and other peripherals. provides detailed information needed for configuration.
```bash
uv run python3 ./scripts/list_squid_hardware.py
```

**Important**: when configuring devices in your `~/seafront/config.json`, use the `USB Device ID` field from the hardware listing output for all device IDs.

Example output:
```
📷 SQUID Cameras:

  1. Driver: galaxy
     USB Manufacturer: Daheng Imaging
     USB Model: MER2-630-60U3M
     USB Device ID: FCS22111429
     USB VID:PID: 2ba2:4d55

  2. Driver: galaxy
     USB Manufacturer: Daheng Imaging
     USB Model: MER2-1220-32U3M
     USB Device ID: FCW23020121
     USB VID:PID: 2ba2:4d55

🔌 SQUID Microcontrollers:

  1. USB Serial
     Device: /dev/ttyACM0
     USB Manufacturer: Teensyduino
     USB Model: USB Serial
     USB Device ID: 12769440
     USB VID:PID: 16c0:0483

Driver status:
  📷 Galaxy: ✅ OK (2 cameras)
  📷 ToupCam: ✅ OK (0 cameras)
  🔌 Microcontroller: ✅ OK (1 device)
```

Then your config should use:
- `"camera.main.id": "FCW23020121"` (the USB Device ID for main camera)
- `"camera.main.driver": "galaxy"` (or `"toupcam"` depending on camera type)
- `"laser.autofocus.camera.id": "FCS22111429"` (the USB Device ID for autofocus camera)
- `"microcontroller.id": "12769440"`

# notes

## Hardware Requirements

**SQUID Microscope Hardware:**
- Microcontroller (Teensy) for hardware control - required for real hardware operation
- Connected via USB to the control computer
- Camera drivers: Galaxy SDK (for Daheng cameras) or ToupCam driver (for ToupTek cameras)

## USB Memory Configuration

This software establishes USB connections to cameras simultaneously, which may require more USB stack memory than allowed by the operating system by default.

**Automatic Configuration:**
The `install/all.sh` script automatically configures USB memory to 1000 MB, which is the recommended value by camera manufacturers.

**Manual Configuration** (if needed):
On Linux, check the current USB memory limit:
```bash
cat /sys/module/usbcore/parameters/usbfs_memory_mb
```

To adjust it temporarily (until next reboot):
```bash
echo 1000 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb
```

To adjust it permanently, add this line to `/etc/rc.local` (or directly change to the kernel parameters):
```bash
echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb
```

If this value is too low, camera connection may fail with an error like:
```
Exception: Device.stream_on:{-1010}{CDSStartAcquisitionAgency:line[860]}{{-1010}{StartAcquisition:line[537]}{TL Error:
Unable to start acquisition.}}
```

# troubleshooting

## Port Already in Use

**Error:** `Address already in use` or port binding error when starting seafront

**Solution:**
- Check which process is using the port:
  ```bash
  lsof -i :5002  # Check port 5002 (or your configured port)
  ```
- Kill the process:
  ```bash
  kill -9 <PID>
  ```
- Or change the port in your config file and restart

## Missing Default Protocol

**Error:** `FileNotFoundError: [Errno 2] No such file or directory: '.../acquisition_configs/...default.json'`

**Solution:**
Generate the default protocol for your microscope:
```bash
uv run python scripts/generate_default_protocol.py --microscope "your-microscope-name"
```

Make sure to use the exact microscope name from your configuration file.

## Camera Not Detected

**Error:** Camera connection fails or hardware listing shows no cameras

**Steps to diagnose:**
1. List all available hardware:
   ```bash
   uv run python scripts/list_squid_hardware.py
   ```
2. Check USB connections - ensure cameras and microcontroller are properly connected
3. Verify camera model names in your config match the exact model strings from the hardware listing
4. Check camera drivers are installed:
   - For Galaxy cameras: ensure Galaxy SDK is installed
   - For ToupCam cameras: ensure ToupCam driver is installed
5. Check user permissions for USB device access

## Microscope Name Mismatch

**Error:** Protocol loading fails or configuration doesn't match after changing microscope names

**Solution:**
If you change the `microscope_name` in your configuration:
1. Delete the old microscope-specific protocol directory:
   ```bash
   rm -rf ~/seafront/acquisition_configs/old-microscope-name/
   ```
2. Generate a new default protocol with the new name:
   ```bash
   uv run python scripts/generate_default_protocol.py --microscope "new-microscope-name"
   ```
3. Restart seafront:
   ```bash
   uv run python -m seafront --microscope "new-microscope-name"
   ```

## Web Interface Not Responsive

**Error:** Browser shows blank page or slow response

**Steps to check:**
1. Verify the server is running and check the terminal for errors
2. Check the server URL is correct (default: `http://localhost:5002`)
3. Try opening `/docs` or `/redoc` to verify API is responding
4. Clear browser cache and reload the page (F5)
5. Check browser console (F12) for JavaScript errors
6. Ensure WebSocket connections are not blocked by firewall/proxy

## Configuration Not Updating in Interface

**Issue:** Changes to config file don't appear in the web interface

**Solution:**
1. Reset the machine config in the interface if it was already loaded: go to "Machine Config" → "reset all values"
2. Refresh the browser page (F5, not Ctrl+Shift+R for hard reload)
3. [as a last resort, delete the local microscope cache for the current microscope via the "advanced" tab, then reload.]

**Note:** Browser configuration is stored in localStorage. Changes to the server-side config won't automatically sync with previously saved browser settings. Use the "reset all values" button to synchronize server-to-browser, or "flush to server" to sync browser-to-server.
