#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"

# Set PYTHONPATH to include the project root so imports work
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Run pytest via uv, which resolves the project environment itself.
# This respects UV_PROJECT_ENVIRONMENT, so the venv does not have to live
# at ./.venv — keeping it outside a cloud-synced folder avoids the file
# corruption and heavy I/O stalls that a sync client causes.
# Extra arguments are passed through, e.g. ./run_tests.sh -k centerline
uv run pytest tests/ -v "$@"

# Exit with the same code as pytest
exit $?


