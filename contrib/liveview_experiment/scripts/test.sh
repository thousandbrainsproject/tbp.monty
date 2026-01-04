#!/bin/bash
# Test script for LiveView Experiment
# Runs all CI checks: formatting, linting, type checking, and tests
# Automatically uses tbp.monty conda environment

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIVEVIEW_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "$LIVEVIEW_DIR"

# Source common environment setup
source "$SCRIPT_DIR/common_env.sh"

# Check if dependencies are installed
if ! run_python_module pytest --version &> /dev/null 2>&1; then
    echo "Dependencies not found. Installing..." >&2
    # Attempt to install dev dependencies using the detected pip
    $PIP install -e ".[dev]" || {
        echo "Error: Failed to install dev dependencies. Please run ./scripts/setup.sh first or install dependencies manually." >&2
        exit 1
    }
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

