#!/bin/bash

# Revert to the original PATH
export PATH="$OLD_PATH"
unset OLD_PATH

echo "Deactivated Python environment."
