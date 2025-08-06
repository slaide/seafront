#!/bin/bash
# run all other install scripts

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run all other install scripts with paths relative to this script's location
bash "$SCRIPT_DIR/galaxy_camera_fix_usb_memory.sh"
bash "$SCRIPT_DIR/teensyduino_udev_rules.sh"
bash "$SCRIPT_DIR/uv.sh"
