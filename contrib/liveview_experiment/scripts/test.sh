#!/bin/bash
# Test script for LiveView Experiment
# Runs all CI checks: formatting, linting, type checking, and tests
# Formatting, linting and typing run in tbp.monty's uv environment; the tests
# prefer the dashboard venv, which is the only place pyview lives.

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIVEVIEW_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TBP_MONTY_ROOT="$(cd "${LIVEVIEW_DIR}/../.." && pwd)"
CONFIG="contrib/liveview_experiment"

# `uv run` finds its project by walking up from the working directory, and the
# contrib directory has a pyproject.toml of its own.
cd "$TBP_MONTY_ROOT"

echo "=========================================="
echo "Formatting..."
echo "=========================================="
echo ""

if [ -f "$SCRIPT_DIR/reformat.sh" ]; then
    "$SCRIPT_DIR/reformat.sh"
else
    echo "Note: reformat.sh not found, skipping auto-formatting" >&2
fi

echo "=========================================="
echo "Running CI checks..."
echo "=========================================="
echo ""

echo "1. Checking code formatting with ruff..."
uv run ruff format --check "$CONFIG/src" "$CONFIG/tests" "$CONFIG/scripts"
echo "✓ Formatting check passed"
echo ""

echo "2. Linting with ruff..."
# The tests insert the project root into sys.path before importing, so only the
# source and the tooling scripts are linted.
uv run ruff check "$CONFIG/src" "$CONFIG/scripts"
echo "✓ Linting passed"
echo ""

echo "3. Type checking with mypy..."
uv run mypy --config-file "$CONFIG/pyproject.toml" "$CONFIG/src"
echo "✓ Type checking passed"
echo ""

echo "4. Running tests..."
uv run python -m pytest -m "not integration" "$LIVEVIEW_DIR/tests" \
    --ignore="$LIVEVIEW_DIR/tests/test_template_parsing.py" "$@"
echo "✓ Tests passed"
echo ""

# The template tests import pyview, which only the dashboard venv has.
LIVEVIEW_PYTHON="${LIVEVIEW_DIR}/.liveview_venv/bin/python"
if [ -x "$LIVEVIEW_PYTHON" ]; then
    echo "5. Running the template tests in the dashboard venv..."
    "$LIVEVIEW_PYTHON" -m pytest "$LIVEVIEW_DIR/tests/test_template_parsing.py" "$@"
    echo "✓ Template tests passed"
    echo ""
else
    echo "5. Skipping the template tests (no .liveview_venv; run scripts/setup.sh)"
    echo ""
fi

echo "=========================================="
echo "All checks passed! ✓"
echo "=========================================="

