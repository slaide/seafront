#!/bin/bash

# Define the Python version
PYTHON_VERSION="3.10.9"

START_WD=$(pwd)

# Define the installation directory
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd $SCRIPT_DIR
INSTALL_DIR="$SCRIPT_DIR/python-$PYTHON_VERSION"

# Save current PATH
export OLD_PATH="$PATH"

# Modify PATH to use the installed Python version
export PATH="$INSTALL_DIR/bin:$PATH"

echo "Activated Python $PYTHON_VERSION environment."

cd $START_WD