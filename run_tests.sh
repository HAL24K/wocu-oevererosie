#!/bin/bash
# Run the test suite from the repo root; extra arguments go to pytest,
# e.g. ./run_tests.sh -k region_split
cd "$(dirname "${BASH_SOURCE[0]}")" && exec uv run pytest tests/ "$@"
