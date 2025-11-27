#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"

# Activate the virtual environment
source .venv/bin/activate

# Set PYTHONPATH to include the project root so imports work
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Run pytest with all tests
pytest tests/ -v

# Capture the exit code
EXIT_CODE=$?

# Exit with the same code as pytest
exit $EXIT_CODE


