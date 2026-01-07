#!/bin/bash
# Shell script wrapper for complexity analysis tool.
#
# Usage:
#   ./scripts/analyze_complexity.sh [directory]
#
# If no directory is provided, analyzes both src/ and scripts/ and runs vulture.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIVEVIEW_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ANALYZE_SCRIPT="$SCRIPT_DIR/analyze_complexity.py"

# Prefer whatever `python` is on PATH (usually conda env when activated)
PYTHON_BIN="${PYTHON:-python}"

cd "$LIVEVIEW_DIR"

if [[ $# -eq 0 ]]; then
    # Analyze both source code and scripts
    echo "Analyzing source code and scripts..."
    "$PYTHON_BIN" -u "$ANALYZE_SCRIPT" "$LIVEVIEW_DIR/src"
    echo ""
    echo "=================================================================================="
    echo "SCRIPTS ANALYSIS"
    echo "=================================================================================="
    "$PYTHON_BIN" -u "$ANALYZE_SCRIPT" "$LIVEVIEW_DIR/scripts"
    echo ""
    echo "=================================================================================="
    echo "DEAD CODE DETECTION (vulture)"
    echo "=================================================================================="
    echo "Checking source code..."
    if "$PYTHON_BIN" -u -m vulture "$LIVEVIEW_DIR/src" --min-confidence 80; then
        echo "✓ No dead code found in source"
    fi
    echo ""
    echo "Checking scripts..."
    if "$PYTHON_BIN" -u -m vulture "$LIVEVIEW_DIR/scripts" --min-confidence 80; then
        echo "✓ No dead code found in scripts"
    fi
    exit 0
fi

TARGET_DIR="${1}"

# Convert to absolute path if relative
if [[ ! "$TARGET_DIR" = /* ]]; then
    TARGET_DIR="$LIVEVIEW_DIR/$TARGET_DIR"
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

# Run the analysis script directly
"$PYTHON_BIN" -u "$ANALYZE_SCRIPT" "$TARGET_DIR"
