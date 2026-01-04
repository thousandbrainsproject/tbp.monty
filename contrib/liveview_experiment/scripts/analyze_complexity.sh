#!/bin/bash
# Shell script wrapper for complexity analysis tool
#
# Usage:
#   ./scripts/analyze_complexity.sh [directory]
#
# If no directory is provided, defaults to src/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIVEVIEW_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ANALYZE_SCRIPT="${SCRIPT_DIR}/analyze_complexity.py"

cd "$LIVEVIEW_DIR"

# Check if conda is available
if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda not found. Please install conda." >&2
    exit 1
fi

# Check for tbp.monty environment
if ! conda env list | grep -q "^tbp.monty "; then
    echo "Error: tbp.monty conda environment not found." >&2
    exit 1
fi

# Activate conda environment if not already activated
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "tbp.monty" ]; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate tbp.monty || {
        echo "Error: Could not activate tbp.monty environment." >&2
        exit 1
    }
fi

# Check if radon is installed
if ! python -c "import radon" 2>/dev/null; then
    echo "Installing radon for complexity analysis..." >&2
    pip install radon >&2 || {
        echo "Warning: Could not install radon. Analysis will be limited." >&2
    }
fi

# Default to src/ if no argument provided
if [[ $# -eq 0 ]]; then
    TARGET_DIR="${LIVEVIEW_DIR}/src"
else
    TARGET_DIR="${1}"
    # Convert to absolute path if relative
    if [[ ! "$TARGET_DIR" = /* ]]; then
        TARGET_DIR="${LIVEVIEW_DIR}/$TARGET_DIR"
    fi
fi

# Check if Python script exists
if [[ ! -f "$ANALYZE_SCRIPT" ]]; then
    echo "Error: Analysis script not found at $ANALYZE_SCRIPT" >&2
    exit 1
fi

# Check if target directory exists
if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Error: Target directory not found: $TARGET_DIR" >&2
    exit 1
fi

# Run the analysis script
python "$ANALYZE_SCRIPT" "$TARGET_DIR"

