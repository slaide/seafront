#!/usr/bin/env python3
"""
Command line front end for seafront.config.default_protocol.

The generation itself lives in the package, because the server runs it on first start and an
installed seafront has no scripts/ directory beside it. This file is the argument parsing and
the listing options, nothing else.
"""

import argparse

import seaconfig as sc

from seafront.config.default_protocol import (
    generate_default_protocol,
    get_wellplate,
    load_server_config,
)

__all__ = ["generate_default_protocol", "get_wellplate", "load_server_config"]


def main():
    parser = argparse.ArgumentParser(description="Generate default acquisition protocol")
    parser.add_argument("--plate", "-p", type=str, help="Wellplate Model_id (default: revvity-384-6057800)")
    parser.add_argument("--list-wellplates", action="store_true", help="List available wellplates and exit")
    parser.add_argument("--list-microscopes", action="store_true", help="List available microscopes from config and exit")
    parser.add_argument("--microscope", "-m", type=str, help="Microscope name (must exist in config)")

    args = parser.parse_args()

    if args.list_wellplates:
        print("Available wellplates:")
        for plate in sc.plates.Plates:
            print(f"  {plate.Model_id:<25} - {plate.Manufacturer} {plate.Model_name} ({plate.Num_wells_x}x{plate.Num_wells_y} wells)")
        return

    if args.list_microscopes:
        try:
            server_config = load_server_config()
            print("Available microscopes (from ~/seafront/config.json):")
            for m in server_config.microscopes:
                name = m.get("system.microscope_name", "<unnamed>")
                print(f"  {name}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
        return

    if not args.microscope:
        raise ValueError("--microscope argument is required")

    output_file = generate_default_protocol(args.microscope, plate_id=args.plate)
    wellplate = get_wellplate(args.plate)

    print(f"Generated: {output_file}")
    print(f"Wellplate: {wellplate.Model_id} ({wellplate.Num_wells_x}x{wellplate.Num_wells_y})")


if __name__ == "__main__":
    main()
