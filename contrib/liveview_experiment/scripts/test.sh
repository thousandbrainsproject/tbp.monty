#!/bin/bash
# Test script for LiveView Experiment
# Runs all CI checks: formatting, linting, type checking, and tests
# Uses Python 3.14+ environment (LiveView venv) if available, otherwise tbp.monty (prefer latest)

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIVEVIEW_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "$LIVEVIEW_DIR"

# Check for LiveView venv (Python 3.11+) first
LIVEVIEW_VENV="${LIVEVIEW_DIR}/.liveview_venv"
if [ -d "$LIVEVIEW_VENV" ] && [ -f "$LIVEVIEW_VENV/bin/python" ]; then
    export PYTHON="$LIVEVIEW_VENV/bin/python"
    export PIP="$LIVEVIEW_VENV/bin/pip"
    export RUN_CMD=""
    echo "Using LiveView Python 3.14+ environment (prefer latest)" >&2
    
    # Define run_python_module function
    run_python_module() {
        $PYTHON -u -m "$@"
    }
    
    # Install dev dependencies if not present (check for black, which is needed first)
    # Note: In venv, we need both liveview and dev extras
    if ! $PYTHON -m black --version &> /dev/null 2>&1; then
        echo "Installing dev dependencies in LiveView environment..." >&2
        $PIP install -e ".[liveview,dev]" || {
            echo "Error: Failed to install dev dependencies. Please run ./scripts/setup.sh first or install dependencies manually." >&2
            exit 1
        }
    fi
else
    # Fall back to common environment setup (tbp.monty)
    source "$SCRIPT_DIR/common_env.sh"
    
    # Check if dependencies are installed (check for black, which is needed first)
    # Note: In Python 3.8 environment, we only install dev deps (no liveview extras)
    if ! run_python_module black --version &> /dev/null 2>&1; then
        echo "Dependencies not found. Installing dev tools (without LiveView dependencies)..." >&2
        # Install dev tools individually (pyview-web requires Python 3.11+)
        $PIP install "black>=24.0.0" "ruff>=0.4.0" "mypy>=1.0.0" "pytest>=8.0.0" "vulture>=2.0.0" || {
            echo "Error: Failed to install dev tools. Please run ./scripts/setup.sh first or install dependencies manually." >&2
            exit 1
        }
    fi
fi

# Function to run command with proper environment
run_check() {
    run_python_module "$@"
}

echo "=========================================="
echo "Formatting..."
echo "=========================================="
echo ""

# Check if reformat script exists, otherwise just check formatting
if [ -f "$SCRIPT_DIR/reformat.sh" ]; then
    "$SCRIPT_DIR/reformat.sh"
else
    echo "Note: reformat.sh not found, skipping auto-formatting" >&2
fi

# Run all CI checks
echo "=========================================="
echo "Running CI checks..."
echo "=========================================="
echo ""

# 1. Check code formatting with Black
echo "1. Checking code formatting with Black..."
run_python_module black --check src scripts || {
    echo "✗ Formatting check failed. Run: black src scripts" >&2
    exit 1
}
echo "✓ Formatting check passed"
echo ""

# 2. Lint with Ruff
echo "2. Linting with Ruff..."
run_python_module ruff check src scripts || {
    echo "✗ Linting failed" >&2
    exit 1
}
echo "✓ Linting passed"
echo ""

# 3. Type check with mypy
echo "3. Type checking with mypy..."
run_python_module mypy src || {
    echo "✗ Type checking failed" >&2
    exit 1
}
echo "✓ Type checking passed"
echo ""

# 4. Run tests (if tests directory exists)
if [ -d "$LIVEVIEW_DIR/tests" ]; then
    echo "4. Running tests..."
    run_python_module pytest -m "not integration" "$@" || {
        echo "✗ Tests failed" >&2
        exit 1
    }
    echo "✓ Tests passed"
    echo ""
else
    echo "4. No tests directory found, skipping tests"
    echo ""
fi

echo "=========================================="
echo "All checks passed! ✓"
echo "=========================================="

