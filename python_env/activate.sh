#!/bin/bash

# Define the Python version
PYTHON_VERSION="3.13.0"

START_WD=$(pwd)

# Define the installation directory
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd $SCRIPT_DIR
INSTALL_DIR="$SCRIPT_DIR/python-$PYTHON_VERSION"

# Save current PATH
export OLD_PATH="$PATH"

export PYTHONPATH=$INSTALL_DIR/lib/python3.13 
export PYTHONHOME=$INSTALL_DIR
export SSL_CERT_FILE=$(echo "import certifi;print(certifi.where())" | $INSTALL_DIR/bin/python3.13 -)

# Modify PATH to use the installed Python version
export PATH="$INSTALL_DIR/bin:$PATH"

echo "Activated Python $PYTHON_VERSION environment."

cd $START_WD
