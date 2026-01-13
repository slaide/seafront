#!/usr/bin/env python3
"""
List all available SQUID microscope hardware.

This script discovers and lists cameras and microcontrollers from all supported SQUID 
microscope drivers that are currently connected to the system.

Use this to verify hardware detection before running the SQUID microscope software.
"""

import sys
from pathlib import Path
import subprocess
import re

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def scan_cameras_by_driver(driver_name: str) -> tuple[list, str | None]:
    """
    Scan for cameras from a specific driver.

    Returns:
        tuple: (list of cameras, error_message or None)
    """
    try:
        if driver_name.lower() == "galaxy":
            from seafront.hardware.camera.galaxy_camera import GalaxyCamera
            return GalaxyCamera.get_all(), None
        elif driver_name.lower() == "toupcam":
            from seafront.hardware.camera.toupcam_camera import ToupCamCamera
            return ToupCamCamera.get_all(), None
        else:
            return [], f"Unknown driver: {driver_name}"
    except ImportError as e:
        return [], f"Driver {driver_name} not available (import error: {e})"
    except Exception as e:
        return [], f"Driver {driver_name} failed to initialize: {e}"


def scan_microcontrollers() -> tuple[list, str | None]:
    """
    Scan for available microcontrollers.

    Returns:
        tuple: (list of microcontrollers, error_message or None)
    """
    try:
        from seafront.hardware.microcontroller import get_all_microcontrollers
        return get_all_microcontrollers(), None
    except ImportError as e:
        return [], f"Microcontroller driver not available (import error: {e})"
    except Exception as e:
        return [], f"Microcontroller driver failed to initialize: {e}"


def _build_usb_vidpid_lookup() -> dict[str, tuple[str, str]]:
    """
    Build a lookup table of USB device serial numbers to (VID, PID) hex strings.

    Uses lsusb -v to parse USB device information. Returns a dict mapping
    serial number strings to (vid, pid) tuples in hex format.

    Returns:
        dict: {serial_number: ('vid_hex', 'pid_hex')} or {} if lsusb fails
    """

    try:
        result = subprocess.run(['lsusb', '-v'], capture_output=True, text=True, timeout=15)
        lookup = {}
        current_vidpid = None

        for line in result.stdout.split('\n'):
            # Match VID:PID line: "Bus 001 Device 002: ID 2ba2:4d55 Daheng Imaging MER2-630-60U3M"
            bus_match = re.match(r'Bus \d+ Device \d+: ID ([0-9a-f]{4}):([0-9a-f]{4})', line)
            if bus_match:
                current_vidpid = (bus_match.group(1), bus_match.group(2))
            # Match iSerial line: "  iSerial                 3 FCS22070932"
            elif current_vidpid and 'iSerial' in line:
                serial_match = re.search(r'iSerial\s+\d+\s+(.+)', line)
                if serial_match:
                    serial = serial_match.group(1).strip()
                    lookup[serial] = current_vidpid

        return lookup
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        # Return empty dict if lsusb fails or is not available
        return {}


def main():
    """List all available SQUID microscope hardware from supported vendors."""
    print("🔍 Scanning for available SQUID microscope hardware...")
    print()

    all_cameras = []
    driver_status = {}
    all_microcontrollers = []
    microcontroller_error = None

    # Scan each driver independently to provide detailed status
    for driver_name in ["Galaxy", "ToupCam"]:
        cameras, error = scan_cameras_by_driver(driver_name)
        driver_status[driver_name] = {"cameras": cameras, "error": error}
        all_cameras.extend(cameras)

    # Scan for microcontrollers
    all_microcontrollers, microcontroller_error = scan_microcontrollers()

    # Report results
    if not all_cameras and not all_microcontrollers:
        print("❌ No SQUID hardware found")
        print()

        # Show detailed status for each driver
        print("Driver status:")
        for driver_name, status in driver_status.items():
            if status["error"]:
                print(f"  📷 {driver_name}: ❌ {status['error']}")
            else:
                print(f"  📷 {driver_name}: ✅ OK (0 cameras)")

        if microcontroller_error:
            print(f"  🔌 Microcontroller: ❌ {microcontroller_error}")
        else:
            print("  🔌 Microcontroller: ✅ OK (0 devices)")

        print()
        print("Common solutions:")
        print("- Check if SQUID hardware is physically connected via USB")
        print("- Verify power supply is connected")
        print("- Check USB permissions (try running as admin/sudo)")
        print("- Ensure driver software is properly installed")
        return

    # Display cameras if found
    if all_cameras:
        print("📷 SQUID Cameras:")
        print()

        # Build USB VID:PID lookup for all devices
        usb_vidpid_lookup = _build_usb_vidpid_lookup()

        # Map vendor names to driver names
        vendor_to_driver = {
            "Daheng Imaging": "galaxy",
            "ToupTek": "toupcam"
        }

        for i, camera in enumerate(all_cameras, 1):
            vendor = camera.vendor_name
            driver_name = vendor_to_driver.get(vendor, "unknown")

            # Handle different camera types with their specific information
            if vendor == "Daheng Imaging":
                print(f"  {i}. Driver: {driver_name}")
                print(f"     USB Manufacturer: {vendor}")
                print(f"     USB Model: {camera.model_name}")
                print(f"     USB Device ID: {camera.sn}")
                # Add VID:PID if available
                if camera.sn in usb_vidpid_lookup:
                    vid, pid = usb_vidpid_lookup[camera.sn]
                    print(f"     USB VID:PID: {vid}:{pid}")
            elif vendor == "ToupTek":
                # For ToupCam cameras, access the original device info
                if hasattr(camera, '_original_device') and camera._original_device:
                    device = camera._original_device
                    display_name = device.displayname if hasattr(device, 'displayname') else "Unknown Display Name"

                    print(f"  {i}. Driver: {driver_name}")
                    print(f"     USB Manufacturer: {vendor}")
                    print(f"     USB Model: {display_name}")
                    print(f"     USB Device ID: {camera.sn}")
                    # Add VID:PID if available
                    if camera.sn in usb_vidpid_lookup:
                        vid, pid = usb_vidpid_lookup[camera.sn]
                        print(f"     USB VID:PID: {vid}:{pid}")
                else:
                    # Fallback to basic info
                    print(f"  {i}. Driver: {driver_name}")
                    print(f"     USB Manufacturer: {vendor}")
                    print(f"     USB Model: {camera.model_name}")
                    print(f"     USB Device ID: {camera.sn}")
                    # Add VID:PID if available
                    if camera.sn in usb_vidpid_lookup:
                        vid, pid = usb_vidpid_lookup[camera.sn]
                        print(f"     USB VID:PID: {vid}:{pid}")
            else:
                # Generic fallback for other camera types
                print(f"  {i}. Driver: {driver_name}")
                print(f"     USB Manufacturer: {vendor}")
                print(f"     USB Model: {camera.model_name}")
                print(f"     USB Device ID: {camera.sn}")
                # Add VID:PID if available
                if camera.sn in usb_vidpid_lookup:
                    vid, pid = usb_vidpid_lookup[camera.sn]
                    print(f"     USB VID:PID: {vid}:{pid}")

            # Add empty line between camera entries
            if i < len(all_cameras):
                print()
        print()

    # Display microcontrollers if found
    if all_microcontrollers:
        print("🔌 SQUID Microcontrollers:")
        print()

        for i, mc in enumerate(all_microcontrollers, 1):
            print(f"  {i}. {mc.device_info.description}")
            print(f"     Device: {mc.device_info.device}")
            if hasattr(mc.device_info, 'manufacturer') and mc.device_info.manufacturer:
                print(f"     USB Manufacturer: {mc.device_info.manufacturer}")
            if hasattr(mc.device_info, 'product') and mc.device_info.product:
                print(f"     USB Model: {mc.device_info.product}")
            if hasattr(mc.device_info, 'serial_number') and mc.device_info.serial_number:
                print(f"     USB Device ID: {mc.device_info.serial_number}")
            if hasattr(mc.device_info, 'vid') and hasattr(mc.device_info, 'pid') and mc.device_info.vid and mc.device_info.pid:
                print(f"     USB VID:PID: {mc.device_info.vid:04x}:{mc.device_info.pid:04x}")

            # Add empty line between microcontroller entries
            if i < len(all_microcontrollers):
                print()
        print()

    # Show driver status for transparency
    print("Driver status:")
    for driver_name, status in driver_status.items():
        if status["error"]:
            print(f"  📷 {driver_name}: ❌ {status['error']}")
        else:
            camera_count = len(status["cameras"])
            print(f"  📷 {driver_name}: ✅ OK ({camera_count} camera{'s' if camera_count != 1 else ''})")

    if microcontroller_error:
        print(f"  🔌 Microcontroller: ❌ {microcontroller_error}")
    else:
        mc_count = len(all_microcontrollers)
        print(f"  🔌 Microcontroller: ✅ OK ({mc_count} device{'s' if mc_count != 1 else ''})")

    print()
    print("This hardware can be used in your SQUID microscope configuration.")


if __name__ == "__main__":
    main()
