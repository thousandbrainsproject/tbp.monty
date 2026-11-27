#!/bin/bash
# Setup script for LiveView Experiment Monitor
# Syncs the tbp.monty uv environment (Python 3.13) and creates the venv the
# dashboard process runs in, with pyview-web and uvicorn.
# Installs pyzmq into the experiment environment for ZMQ communication

set -eo pipefail

# Find script and project directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIVEVIEW_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TBP_MONTY_ROOT="$(cd "${LIVEVIEW_DIR}/../.." && pwd)"

cd "$TBP_MONTY_ROOT"

echo "Setting up LiveView Experiment Monitor..." >&2
echo "" >&2

# tbp.monty is a uv project. Its `.python-version` (3.13) is the interpreter
# both MuJoCo and pyview-web support; `uv sync` puts the experiment's
# dependencies where `uv run` finds them.
if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv not found. Install it from https://docs.astral.sh/uv/." >&2
    echo "See docs/how-to-use-monty/getting-started.md for details." >&2
    exit 1
fi

echo "Syncing the tbp.monty environment..." >&2
uv sync --extra dev --extra simulator_mujoco >&2

PYTHON_VERSION=$(uv run python -c 'import platform; print(platform.python_version())')
echo "✓ tbp.monty environment ready (Python $PYTHON_VERSION)" >&2
echo "" >&2

# The environment is the uv one; there is nothing left to install by hand.
# pyzmq is the contrib's own dependency, so it is added beside Monty's.
if ! uv run python -c "import zmq" 2>/dev/null; then
    echo "Installing pyzmq into the tbp.monty environment..." >&2
    uv pip install --quiet "pyzmq>=25.0.0" >&2
    echo "✓ pyzmq installed" >&2
fi

# The dashboard process gets pyview-web and uvicorn in a venv of its own, so
# they never take part in the experiment's dependency resolution. It runs on
# the same interpreter as the experiment.
LIVEVIEW_VENV="${LIVEVIEW_DIR}/.liveview_venv"

venv_python_version() {
    "$1" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null
}

echo "Installing LiveView dependencies..." >&2
if [ -x "$LIVEVIEW_VENV/bin/python" ] &&
   [ "$(venv_python_version "$LIVEVIEW_VENV/bin/python")" != "${PYTHON_VERSION%.*}" ]; then
    echo "Recreating the LiveView venv for Python ${PYTHON_VERSION%.*}..." >&2
    rm -rf "$LIVEVIEW_VENV"
fi

if [ ! -x "$LIVEVIEW_VENV/bin/python" ]; then
    echo "Creating the LiveView venv with Python ${PYTHON_VERSION%.*}..." >&2
    uv venv --python "${PYTHON_VERSION%.*}" "$LIVEVIEW_VENV" >&2
fi

uv pip install --python "$LIVEVIEW_VENV/bin/python" -e "${LIVEVIEW_DIR}[liveview,dev]" >&2
echo "✓ LiveView dashboard environment ready" >&2
echo "" >&2

echo "✓ Setup complete!" >&2
echo "" >&2
echo "Run an experiment with the dashboard:" >&2
echo "  ./contrib/liveview_experiment/scripts/run.sh [experiment]" >&2
echo "  ./contrib/liveview_experiment/scripts/run.sh -l evidence   # which names fit a word" >&2
echo "" >&2
echo "Then open http://127.0.0.1:8000 in your browser to view the dashboard." >&2
echo "" >&2

