#!/usr/bin/env python3
"""
Inspect and display OME metadata from OME-TIFF files.

This script reads OME-TIFF files and displays their metadata in a human-readable
format, helping validate that OME-TIFF generation is working correctly and that
the metadata can be read by standard OME tools.

Usage:
    uv run python scripts/inspect_ome_metadata.py image.tiff
    uv run python scripts/inspect_ome_metadata.py --xml image.tiff
    uv run python scripts/inspect_ome_metadata.py --json image.tiff
"""

import sys
import json
import argparse
from pathlib import Path
from xml.dom import minidom

import tifffile
from ome_types import from_xml


def extract_ome_xml(tiff_path: Path) -> str | None:
    """
    Extract OME-XML metadata from a TIFF file.

    Args:
        tiff_path: Path to the TIFF file

    Returns:
        OME-XML string if present, None otherwise
    """
    try:
        with tifffile.TiffFile(tiff_path) as tif:
            # Look for ImageDescription tag which contains OME-XML
            if tif.is_ome:
                return tif.ome_metadata
            else:
                # Try to extract from ImageDescription tag directly
                if tif.pages and tif.pages[0].tags and 'image_description' in tif.pages[0].tags:
                    description = tif.pages[0].tags['image_description'].value
                    if isinstance(description, bytes):
                        description = description.decode('utf-8')
                    # Check if it's OME-XML
                    if '<?xml' in description and 'OME' in description:
                        return description
    except Exception as e:
        print(f"Error reading TIFF file: {e}", file=sys.stderr)
        return None

    return None


def format_xml_pretty(xml_string: str) -> str:
    """Pretty print XML with indentation."""
    try:
        dom = minidom.parseString(xml_string)
        return dom.toprettyxml(indent="  ")
    except Exception:
        # If parsing fails, return original
        return xml_string


def display_summary(ome_obj) -> None:
    """Display a human-readable summary of OME metadata."""
    print("\n" + "="*60)
    print("OME METADATA SUMMARY")
    print("="*60)

    # Image information
    if ome_obj.images:
        for img_idx, image in enumerate(ome_obj.images):
            print(f"\n[Image {img_idx}]")
            print(f"  Name: {image.name}")
            if image.acquisition_date:
                print(f"  Acquisition Date: {image.acquisition_date.isoformat()}")

            # Pixels/Dimensions
            if image.pixels:
                pixels = image.pixels
                print(f"  Dimensions: {pixels.size_x}x{pixels.size_y}x{pixels.size_z} " +
                      f"(X×Y×Z), {pixels.size_c} channel(s)")
                print(f"  Type: {pixels.type}")
                if pixels.physical_size_x:
                    print(f"  Physical Size: {pixels.physical_size_x:.3f} µm/pixel (X)")
                if pixels.physical_size_y:
                    print(f"  Physical Size: {pixels.physical_size_y:.3f} µm/pixel (Y)")
                if pixels.physical_size_z:
                    print(f"  Physical Size: {pixels.physical_size_z:.3f} µm/pixel (Z)")

                # Channel information
                if pixels.channels:
                    print(f"\n  Channels ({len(pixels.channels)}):")
                    for ch_idx, channel in enumerate(pixels.channels):
                        print(f"    [{ch_idx}] {channel.name}")
                        if channel.light_source_settings:
                            lss = channel.light_source_settings
                            print(f"        Wavelength: {lss.wavelength} nm" if lss.wavelength else "")
                            if lss.attenuation is not None:
                                print(f"        Attenuation: {lss.attenuation*100:.1f}%")

                # Plane information
                if pixels.planes:
                    print(f"\n  Planes ({len(pixels.planes)}):")
                    for p_idx, plane in enumerate(pixels.planes[:3]):  # Show first 3
                        pos_str = f"({plane.position_x}, {plane.position_y}, {plane.position_z}) mm"
                        exp_str = f"{plane.exposure_time*1000:.2f} ms" if plane.exposure_time else "N/A"
                        print(f"    [{p_idx}] Z={plane.the_z} Exposure={exp_str} Pos={pos_str}")
                    if len(pixels.planes) > 3:
                        print(f"    ... ({len(pixels.planes) - 3} more planes)")

            # Stage Label
            if image.stage_label:
                stage = image.stage_label
                print(f"\n  Stage Label: {stage.name}")
                print(f"    Position: ({stage.x}, {stage.y}, {stage.z}) mm")

            # Description
            if image.description:
                print(f"\n  Description:")
                for line in image.description.split('\n'):
                    if line.strip():
                        print(f"    {line}")

    # Instrument information
    if ome_obj.instruments:
        for inst_idx, instrument in enumerate(ome_obj.instruments):
            print(f"\n[Instrument {inst_idx}]")

            # Microscope
            if instrument.microscope:
                microscope = instrument.microscope
                print(f"  Microscope: {microscope.manufacturer or '?'} {microscope.model or '?'}")
                if microscope.type:
                    print(f"    Type: {microscope.type}")

            # Detector
            if instrument.detectors:
                print(f"  Detectors ({len(instrument.detectors)}):")
                for det in instrument.detectors:
                    print(f"    - {det.manufacturer or '?'} {det.model or '?'} " +
                          f"(SN: {det.serial_number or 'N/A'})")
                    if det.type:
                        print(f"      Type: {det.type}")

            # Light sources
            total_sources = (len(instrument.lasers or []) +
                           len(instrument.light_emitting_diodes or []))
            if total_sources > 0:
                print(f"  Light Sources ({total_sources}):")
                if instrument.lasers:
                    for laser in instrument.lasers:
                        print(f"    - Laser {laser.id}: {laser.wavelength} nm" if laser.wavelength else f"    - Laser {laser.id}")
                if instrument.light_emitting_diodes:
                    for led in instrument.light_emitting_diodes:
                        print(f"    - LED {led.id}")

    print("\n" + "="*60 + "\n")


def display_full(ome_obj) -> None:
    """Display full structured OME object information."""
    print("\n" + "="*60)
    print("OME FULL STRUCTURE")
    print("="*60)
    print(ome_obj)
    print("="*60 + "\n")


def export_json(ome_obj) -> str:
    """Export OME object to JSON (simplified representation)."""
    def serialize_ome(obj):
        """Recursively convert OME objects to JSON-serializable dicts."""
        if isinstance(obj, list):
            return [serialize_ome(item) for item in obj]
        elif hasattr(obj, 'model_dump'):
            # Pydantic v2
            return serialize_ome(obj.model_dump())
        elif hasattr(obj, 'dict'):
            # Pydantic v1
            return serialize_ome(obj.dict())
        elif isinstance(obj, dict):
            return {k: serialize_ome(v) for k, v in obj.items() if v is not None}
        else:
            return str(obj) if obj is not None else None

    data = serialize_ome(ome_obj)
    return json.dumps(data, indent=2)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Inspect and display OME metadata from TIFF files"
    )
    parser.add_argument(
        "tiff_file",
        type=Path,
        help="Path to TIFF file to inspect"
    )
    parser.add_argument(
        "--xml",
        action="store_true",
        help="Display raw OME-XML (formatted)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Export metadata as JSON"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Display full structured OME object"
    )
    parser.add_argument(
        "--raw-xml",
        action="store_true",
        help="Display raw OME-XML (unformatted)"
    )

    args = parser.parse_args()

    # Validate file exists
    if not args.tiff_file.exists():
        print(f"Error: File not found: {args.tiff_file}", file=sys.stderr)
        sys.exit(1)

    if not args.tiff_file.suffix.lower() in ['.tiff', '.tif']:
        print(f"Warning: File does not have .tiff/.tif extension", file=sys.stderr)

    # Extract OME-XML
    print(f"Reading: {args.tiff_file}")
    ome_xml = extract_ome_xml(args.tiff_file)

    if ome_xml is None:
        print("Error: No OME metadata found in TIFF file", file=sys.stderr)
        print("\nThis file does not contain OME-XML metadata.", file=sys.stderr)
        sys.exit(1)

    # Parse OME-XML
    try:
        ome_obj = from_xml(ome_xml)
    except Exception as e:
        print(f"Error parsing OME-XML: {e}", file=sys.stderr)
        sys.exit(1)

    # Display based on arguments
    if args.raw_xml:
        print(ome_xml)
    elif args.xml:
        print(format_xml_pretty(ome_xml))
    elif args.json:
        print(export_json(ome_obj))
    elif args.full:
        display_full(ome_obj)
    else:
        # Default: summary
        display_summary(ome_obj)

    sys.exit(0)


if __name__ == "__main__":
    main()
