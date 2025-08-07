#!/usr/bin/env python3
"""
List all available SQUID microscope cameras.

This script discovers and lists cameras from all supported SQUID microscope camera drivers
(Galaxy/Daheng and ToupCam) that are currently connected to the system.

Use this to verify camera detection before running the SQUID microscope software.
"""

import sys
from pathlib import Path

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
            from seafront.hardware.galaxy_camera import GalaxyCamera
            return GalaxyCamera.get_all(), None
        elif driver_name.lower() == "toupcam":
            from seafront.hardware.toupcam_camera import ToupCamCamera
            return ToupCamCamera.get_all(), None
        else:
            return [], f"Unknown driver: {driver_name}"
    except ImportError as e:
        return [], f"Driver {driver_name} not available (import error: {e})"
    except Exception as e:
        return [], f"Driver {driver_name} failed to initialize: {e}"


def main():
    """List all available SQUID microscope cameras from supported vendors."""
    print("🔍 Scanning for available SQUID microscope cameras...")
    print()

    all_cameras = []
    driver_status = {}

    # Scan each driver independently to provide detailed status
    for driver_name in ["Galaxy", "ToupCam"]:
        cameras, error = scan_cameras_by_driver(driver_name)
        driver_status[driver_name] = {"cameras": cameras, "error": error}
        all_cameras.extend(cameras)

    # Report results
    if not all_cameras:
        print("❌ No SQUID cameras found")
        print()

        # Show detailed status for each driver
        print("Driver status:")
        for driver_name, status in driver_status.items():
            if status["error"]:
                print(f"  📷 {driver_name}: ❌ {status['error']}")
            else:
                print(f"  📷 {driver_name}: ✅ OK (0 cameras)")

        print()
        print("Common solutions:")
        print("- Check if SQUID cameras are physically connected via USB")
        print("- Verify camera power supply is connected")
        print("- Check USB permissions (try running as admin/sudo)")
        print("- Ensure camera driver software is properly installed")
        return

    print(f"✅ Found {len(all_cameras)} SQUID camera(s):")
    print()

    # Group cameras by vendor for cleaner output
    by_vendor = {}
    vendor_to_driver = {}
    for camera in all_cameras:
        vendor = camera.vendor_name
        if vendor not in by_vendor:
            by_vendor[vendor] = []
        by_vendor[vendor].append(camera)
        
        # Map vendor names to driver names
        if vendor == "Daheng Imaging":
            vendor_to_driver[vendor] = "galaxy"
        elif vendor == "ToupTek":
            vendor_to_driver[vendor] = "toupcam"

    for vendor_name, vendor_cameras in by_vendor.items():
        driver_name = vendor_to_driver.get(vendor_name, "unknown")
        print(f"📷 {vendor_name} cameras (driver = \"{driver_name}\"):")
        for i, camera in enumerate(vendor_cameras, 1):
            # Add label for Galaxy/Daheng camera model names to clarify what they represent
            if hasattr(camera, 'index') and vendor_name == "Daheng Imaging":
                print(f"  {i}. {camera.model_name} (Product Model)")
                print(f"     Serial: {camera.sn}")
                print(f"     API Index: {camera.index}")
            elif hasattr(camera, 'index'):
                print(f"  {i}. {camera.model_name}")
                print(f"     Serial: {camera.sn}")
                print(f"     API Index: {camera.index}")
            else:
                # Fallback for cameras without index support
                print(f"  {i}. {camera.model_name}")
                print(f"     Serial: {camera.sn}")
        print()

    # Show driver status for transparency
    print("Driver status:")
    for driver_name, status in driver_status.items():
        if status["error"]:
            print(f"  📷 {driver_name}: ❌ {status['error']}")
        else:
            camera_count = len(status["cameras"])
            print(f"  📷 {driver_name}: ✅ OK ({camera_count} camera{'s' if camera_count != 1 else ''})")

    print()
    print("These cameras can be used in your SQUID microscope configuration.")


if __name__ == "__main__":
    main()
