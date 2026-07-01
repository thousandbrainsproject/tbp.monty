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
    # Check if we're in LiveView venv (Python 3.14+) or need to install individually
    if [ -d ".liveview_venv" ] && [ "$PYTHON" = ".liveview_venv/bin/python" ]; then
        # In LiveView venv - can install with extras
        $PIP install -e ".[liveview,dev]"
    elif [ -d ".venv" ]; then
        # In a venv - try with dev extras
        $PIP install -e ".[dev]" || {
            # Fallback to individual install if pyview-web not available
            $PIP install "black>=24.0.0" "ruff>=0.4.0" || {
                echo "Warning: Failed to install dev tools. Please run ./scripts/setup.sh first." >&2
                exit 1
            }
        }
    elif command -v uv &> /dev/null; then
        uv pip install -e ".[dev]" || {
            # Fallback to individual install
            uv pip install "black>=24.0.0" "ruff>=0.4.0" || {
                echo "Warning: Failed to install dev tools. Please run ./scripts/setup.sh first." >&2
                exit 1
            }
        }
    else
        # In Python 3.8 environment - install dev tools individually (no pyview-web)
        $PIP install "black>=24.0.0" "ruff>=0.4.0" || {
            echo "Warning: Failed to install dev tools. Please run ./scripts/setup.sh first or install dependencies manually." >&2
            exit 1
        }
    fi
fi

# Run black to format code
echo "Reformatting code with black..."
$BLACK src scripts

echo "✅ Code reformatted successfully!"

