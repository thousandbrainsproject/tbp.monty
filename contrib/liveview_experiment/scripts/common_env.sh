#!/bin/bash
# Common environment setup for scripts
# Source this file to get PYTHON, PIP, and RUN_CMD variables set up correctly
#
# Usage:
#   source "$SCRIPT_DIR/common_env.sh"
#
# After sourcing, you'll have:
#   - PYTHON: Path to Python executable
#   - PIP: Path to pip executable
#   - RUN_CMD: Command prefix for running Python modules (empty or "uv run")
#   - run_python: Function to run Python scripts
#   - run_python_module: Function to run Python modules
#
# This script is idempotent - safe to source multiple times.

# Only set up if not already set (allows multiple sourcing - idempotent)
# Check if already loaded to prevent re-initialization
if [ -z "${COMMON_ENV_LOADED:-}" ]; then
    # Mark as loaded FIRST to prevent re-initialization even if detection fails
    export COMMON_ENV_LOADED=1
    
    # Find script and project directories
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    LIVEVIEW_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
    TBP_MONTY_ROOT="$(cd "${LIVEVIEW_DIR}/../.." && pwd)"
    
    # Detect conda environment for tbp.monty
    if command -v conda >/dev/null 2>&1 && conda env list | grep -q "^tbp.monty "; then
        # Activate conda environment if not already activated
        if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "tbp.monty" ]; then
            eval "$(conda shell.bash hook)" 2>/dev/null || true
            conda activate tbp.monty 2>/dev/null || true
        fi
        export PYTHON="python"
        export PIP="pip"
        export RUN_CMD=""
        echo "Using tbp.monty conda environment" >&2
    elif [ -d ".venv" ]; then
        export PYTHON=".venv/bin/python"
        export PIP=".venv/bin/pip"
        export RUN_CMD=""
        echo "Using existing .venv" >&2
    elif command -v uv &> /dev/null; then
        export PYTHON="python3"
        export PIP="uv pip"
        export RUN_CMD="uv run"
        echo "Using uv" >&2
    else
        export PYTHON="python3"
        export PIP="pip3"
        export RUN_CMD=""
        echo "Using system Python (ensure dependencies are installed)" >&2
    fi

    # Function to run Python command with or without uv
    # Only define if not already defined (allows function redefinition to be safe)
    if ! type run_python >/dev/null 2>&1; then
        run_python() {
            if [ -n "$RUN_CMD" ]; then
                # uv run handles buffering, but set PYTHONUNBUFFERED for consistency
                PYTHONUNBUFFERED=1 $RUN_CMD "$@"
            else
                # Always use -u flag for unbuffered output to show progress
                $PYTHON -u "$@"
            fi
        }
    fi

    # Function to run Python module with or without uv
    if ! type run_python_module >/dev/null 2>&1; then
        run_python_module() {
            if [ -n "$RUN_CMD" ]; then
                # uv run handles buffering, but set PYTHONUNBUFFERED for consistency
                PYTHONUNBUFFERED=1 $RUN_CMD "$@"
            else
                # Always use -u flag for unbuffered output to show progress
                $PYTHON -u -m "$@"
            fi
        }
    fi
fi
