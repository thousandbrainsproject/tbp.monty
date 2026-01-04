#!/bin/bash
# Reformat script for LiveView Experiment Monitor
# Automatically detects and uses conda environment or .venv if available

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIVEVIEW_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$LIVEVIEW_DIR"

# Source common environment setup
source "$SCRIPT_DIR/common_env.sh"

# Set BLACK command based on environment
if [ -n "$RUN_CMD" ]; then
    BLACK="$RUN_CMD black"
elif [ -d ".venv" ]; then
    BLACK=".venv/bin/black"
else
    BLACK="$PYTHON -m black"
fi

# Check if black is available
if ! $BLACK --version &> /dev/null 2>&1; then
    echo "black not found. Installing dependencies..." >&2
    if [ -d ".venv" ]; then
        $PIP install -e ".[dev]"
    elif command -v uv &> /dev/null; then
        uv pip install -e ".[dev]"
    else
        $PIP install -e ".[dev]" || {
            echo "Warning: Failed to install dev dependencies. Please run ./scripts/setup.sh first or install dependencies manually." >&2
            exit 1
        }
    fi
fi

# Run black to format code
echo "Reformatting code with black..."
$BLACK src scripts

echo "✅ Code reformatted successfully!"

