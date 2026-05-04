#!/bin/bash
# Run script for LiveView Experiment Monitor (parallel mode)
# Runs a Monty parallel experiment with LiveView dashboard
# Works from any directory, automatically handles conda environment

set -e

# Find script and project directories (works from any location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIVEVIEW_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TBP_MONTY_ROOT="$(cd "${LIVEVIEW_DIR}/../.." && pwd)"

# Ensure we're in the tbp.monty root directory for running the experiment
cd "$TBP_MONTY_ROOT" || {
    echo "Error: Could not change to tbp.monty root: $TBP_MONTY_ROOT" >&2
    exit 1
}

# Usage helpers
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    cat >&2 <<'USAGE'
Usage: scripts/run_parallel.sh [EXPERIMENT_NAME] [NUM_PARALLEL] [EPISODES]

Arguments:
  EXPERIMENT_NAME  Hydra experiment name (yaml without .yaml).
                   Defaults to tutorial_surf_agent_2obj_with_liveview
  NUM_PARALLEL     Number of parallel processes to use. Defaults to 2.
  EPISODES         Optional episode selection string (e.g. "all", ":10", "5:",
                   "0,3,5:8"). See parse_episode_spec() in run_parallel.py.

Environment overrides:
  LIVEVIEW_HOST    Defaults to 127.0.0.1
  LIVEVIEW_PORT    Defaults to 8000
  ZMQ_PORT         Defaults to 5555 (publishers connect; server binds SUB)
USAGE
    exit 0
fi

# Defaults
EXPERIMENT_NAME="${1:-tutorial_surf_agent_2obj_with_liveview}"
NUM_PARALLEL="${2:-2}"
EPISODES_INPUT="${3:-}"  # empty -> all

echo "Starting LiveView Parallel Experiment Monitor..." >&2
echo "" >&2

# Check if conda is available
if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda not found. Please install conda." >&2
    exit 1
fi

# Check for tbp.monty environment
ENV_NAME=""
if conda env list | grep -q "^tbp.monty "; then
    ENV_NAME="tbp.monty"
else
    echo "Error: tbp.monty conda environment not found." >&2
    echo "" >&2
    echo "Please set up the tbp.monty environment first:" >&2
    echo "  conda env create" >&2
    echo "  conda activate tbp.monty" >&2
    echo "  pip install -e ." >&2
    echo "" >&2
    echo "Then run setup:" >&2
    echo "  ./contrib/liveview_experiment/scripts/setup.sh" >&2
    exit 1
fi

# Activate conda environment if not already activated
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    echo "Activating $ENV_NAME conda environment..." >&2
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "$ENV_NAME" || {
        echo "Error: Could not activate $ENV_NAME environment." >&2
        echo "Please run: conda activate $ENV_NAME" >&2
        exit 1
    }
fi

# Check for core tbp.monty dependencies
if ! python -c "import hydra" 2>/dev/null; then
    echo "Error: hydra-core not found in tbp.monty environment." >&2
    echo "" >&2
    echo "The conda environment exists but is missing dependencies." >&2
    echo "" >&2
    echo "Please install the package in editable mode:" >&2
    echo "  conda activate tbp.monty" >&2
    echo "  cd $TBP_MONTY_ROOT" >&2
    echo "  pip install -e ." >&2
    echo "" >&2
    echo "This will install all dependencies including hydra-core." >&2
    exit 1
fi

# Check if LiveView config exists in contrib directory first (preferred)
# Use absolute paths to ensure it works from any directory
CONFIG_PATH="${LIVEVIEW_DIR}/conf/experiment/${EXPERIMENT_NAME}.yaml"
IS_LIVEVIEW_CONFIG=false
if [ ! -f "$CONFIG_PATH" ]; then
    # Fall back to main experiment directory
    CONFIG_PATH="${TBP_MONTY_ROOT}/conf/experiment/${EXPERIMENT_NAME}.yaml"
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "Error: Experiment config not found:" >&2
        echo "  ${LIVEVIEW_DIR}/conf/experiment/${EXPERIMENT_NAME}.yaml" >&2
        echo "  ${CONFIG_PATH}" >&2
        echo "" >&2
        echo "Available experiments:" >&2
        if [ -d "${LIVEVIEW_DIR}/conf/experiment" ]; then
            echo "  In contrib:" >&2
            ls -1 "${LIVEVIEW_DIR}/conf/experiment"/*.yaml 2>/dev/null | sed 's|.*/|    |' | sed 's|\.yaml$||' || true
        fi
        if [ -d "${TBP_MONTY_ROOT}/conf/experiment" ]; then
            echo "  In main:" >&2
            ls -1 "${TBP_MONTY_ROOT}/conf/experiment"/*.yaml 2>/dev/null | sed 's|.*/|    |' | sed 's|\.yaml$||' || true
        fi
        exit 1
    fi
else
    # Config is in contrib directory, so it's a LiveView config with values already defined
    IS_LIVEVIEW_CONFIG=true
fi

echo "✓ Environment: $CONDA_DEFAULT_ENV" >&2
echo "✓ Experiment: $EXPERIMENT_NAME" >&2
echo "✓ Config: $CONFIG_PATH" >&2
echo "✓ Num parallel: $NUM_PARALLEL" >&2
if [ -n "$EPISODES_INPUT" ]; then
  echo "✓ Episodes: $EPISODES_INPUT" >&2
fi
echo "" >&2

# Check for pyzmq (required for ZMQ communication)
if ! python -c "import zmq" 2>/dev/null; then
    echo "Error: pyzmq not found. Please run setup.sh first." >&2
    exit 1
fi

# Start LiveView server early as a separate process (if Python 3.11+ available)
LIVEVIEW_SERVER_PID=""
LIVEVIEW_HOST="${LIVEVIEW_HOST:-127.0.0.1}"
LIVEVIEW_PORT="${LIVEVIEW_PORT:-8000}"
ZMQ_PORT="${ZMQ_PORT:-5555}"

# Check for Python 3.14+ for LiveView server (prefer latest, fallback to 3.11+)
if [ -d "${LIVEVIEW_DIR}/.liveview_venv" ]; then
    LIVEVIEW_PYTHON="${LIVEVIEW_DIR}/.liveview_venv/bin/python"
    if [ -f "$LIVEVIEW_PYTHON" ]; then
        echo "Starting LiveView server..." >&2
        # Start server in background, redirecting to a log file we can tail
        SERVER_LOG="/tmp/liveview_server_$$.log"
        "$LIVEVIEW_PYTHON" "${LIVEVIEW_DIR}/src/liveview_server_standalone.py" \
            --host "$LIVEVIEW_HOST" \
            --port "$LIVEVIEW_PORT" \
            --zmq-port "$ZMQ_PORT" \
            > "$SERVER_LOG" 2>&1 &
        LIVEVIEW_SERVER_PID=$!
        echo "✓ LiveView server started (PID: $LIVEVIEW_SERVER_PID)" >&2
        echo "  Dashboard: http://${LIVEVIEW_HOST}:${LIVEVIEW_PORT}" >&2
        echo "  ZMQ: tcp://${LIVEVIEW_HOST}:${ZMQ_PORT}" >&2
        echo "  Server log: $SERVER_LOG (tail -f to view)" >&2
        # Tail the log in background so we see server output
        tail -f "$SERVER_LOG" &
        TAIL_PID=$!
        # Give server a moment to start
        sleep 2
    fi
fi

echo "" >&2
echo "Starting parallel experiment..." >&2
if [ -n "$LIVEVIEW_SERVER_PID" ]; then
    echo "LiveView dashboard available at: http://${LIVEVIEW_HOST}:${LIVEVIEW_PORT}" >&2
else
    echo "Note: LiveView server not started (Python 3.11+ not available, prefer 3.14+)" >&2
    echo "      Experiment will still publish to ZMQ (if server started separately)" >&2
fi
echo "Press Ctrl+C to stop" >&2
echo "" >&2

# Cleanup function to kill LiveView server and tail process on exit
cleanup() {
    echo "" >&2
    echo "Cleaning up..." >&2
    
    # Kill tail process if running
    if [ -n "$TAIL_PID" ]; then
        kill "$TAIL_PID" 2>/dev/null || true
    fi
    
    # Kill LiveView server
    if [ -n "$LIVEVIEW_SERVER_PID" ]; then
        echo "Stopping LiveView server (PID: $LIVEVIEW_SERVER_PID)..." >&2
        kill "$LIVEVIEW_SERVER_PID" 2>/dev/null || true
        # Give it a moment to shut down gracefully
        sleep 1
        # Force kill if still running
        kill -9 "$LIVEVIEW_SERVER_PID" 2>/dev/null || true
        wait "$LIVEVIEW_SERVER_PID" 2>/dev/null || true
        echo "✓ LiveView server stopped" >&2
    fi
}

# Set up signal handlers - trap must be before running the experiment
trap cleanup INT TERM EXIT

# Build common Hydra searchpath to include both main and contrib directories
SEARCHPATH_OVERRIDE="hydra.searchpath=[${TBP_MONTY_ROOT}/conf,${LIVEVIEW_DIR}/conf]"

# Build optional overrides
EPISODES_OVERRIDE=()
if [ -n "$EPISODES_INPUT" ]; then
  EPISODES_OVERRIDE+=("episodes=${EPISODES_INPUT}")
fi

NUM_PARALLEL_OVERRIDE=("num_parallel=${NUM_PARALLEL}")

# Run experiment - if it hangs, Ctrl-C will trigger the trap
cd "$TBP_MONTY_ROOT"

if [ "$IS_LIVEVIEW_CONFIG" = "true" ]; then
    # LiveView config exists - use it as-is (values from yaml are defaults)
    # Only override ports/host if needed (but they should already be in the config)
    python run_parallel.py \
        "experiment=${EXPERIMENT_NAME}" \
        "$SEARCHPATH_OVERRIDE" \
        "${NUM_PARALLEL_OVERRIDE[@]}" \
        "${EPISODES_OVERRIDE[@]}" || {
        EXPERIMENT_EXIT_CODE=$?
        cleanup
        exit $EXPERIMENT_EXIT_CODE
    }
else
    # Base config - add LiveView support via command line options
    # Also disable wandb and excessive logging
    python run_parallel.py \
        "experiment=${EXPERIMENT_NAME}" \
        "$SEARCHPATH_OVERRIDE" \
        "${NUM_PARALLEL_OVERRIDE[@]}" \
        "${EPISODES_OVERRIDE[@]}" \
        "experiment._target_=contrib.liveview_experiment.src.monty_experiment_with_liveview.MontyExperimentWithLiveView" \
        "+experiment.config.liveview_host=${LIVEVIEW_HOST}" \
        "+experiment.config.liveview_port=${LIVEVIEW_PORT}" \
        "+experiment.config.zmq_port=${ZMQ_PORT}" \
        "+experiment.config.enable_liveview=true" \
        "+experiment.config.sensor_image_throttle_ms=100" \
        "experiment.config.logging.wandb_handlers=[]" \
        "experiment.config.logging.monty_handlers=[]" || {
        EXPERIMENT_EXIT_CODE=$?
        cleanup
        exit $EXPERIMENT_EXIT_CODE
    }
fi

EXPERIMENT_EXIT_CODE=$?

# Cleanup will be called by trap on exit, but explicit cleanup here too
cleanup
exit $EXPERIMENT_EXIT_CODE
