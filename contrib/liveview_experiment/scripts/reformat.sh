#!/bin/bash
# Reformat script for LiveView Experiment Monitor
# Formats with `ruff format`, the formatter tbp.monty itself uses.

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIVEVIEW_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TBP_MONTY_ROOT="$(cd "$LIVEVIEW_DIR/../.." && pwd)"

# Run from the repository root: `uv run` walks up from the working directory to
# find its project, and the contrib directory has a pyproject.toml of its own.
cd "$TBP_MONTY_ROOT"

# Check if ruff is available
if ! uv run ruff --version &> /dev/null 2>&1; then
    echo "ruff not found. Syncing the tbp.monty dev environment..." >&2
    uv sync --extra dev || {
        echo "Warning: Failed to sync the uv environment. Please run ./scripts/setup.sh first." >&2
        exit 1
    }
fi

# Format the contrib's code with ruff
CONFIG="contrib/liveview_experiment"
echo "Reformatting code with ruff..."
uv run ruff format "$CONFIG/src" "$CONFIG/tests" "$CONFIG/scripts"

echo "✅ Code reformatted successfully!"

