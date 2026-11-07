#!/bin/bash
# Run script for LiveView Experiment Monitor
# Runs a Monty experiment with LiveView dashboard
# Works from any directory, automatically handles the uv environment

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

# Default experiment name
EXPERIMENT_NAME="${1:-randrot_noise_77obj_surf_agent}"

# Find the config the name means. Monty's `conf/experiment` is a tree, so
# `evidence_mujoco` is a file under `test/evidence_lm/` and its Hydra name is the
# whole path; both are answered. The lookup is shared with `run_parallel.sh`.
LIVEVIEW_CONF="${LIVEVIEW_DIR}/conf/experiment"
MONTY_CONF="${TBP_MONTY_ROOT}/src/tbp/monty/conf/experiment"
# shellcheck source=experiment_lookup.sh
. "${SCRIPT_DIR}/experiment_lookup.sh"

# `-l PATTERN` lists the experiments whose name contains it, and runs nothing:
# with two hundred of them, finding the one you mean is most of the work.
if [ "${1:-}" = "-l" ] || [ "${1:-}" = "--list" ]; then
    experiment_configs | awk -v want="${2:-}" 'index($2, want) { print "    " $2 }'
    exit 0
fi

IS_LIVEVIEW_CONFIG=false
RESOLVED=""
if RESOLVED=$(resolve_experiment "$EXPERIMENT_NAME"); then
    SOURCE=$(printf '%s' "$RESOLVED" | awk '{ print $1 }')
    EXPERIMENT_NAME=$(printf '%s' "$RESOLVED" | awk '{ print $2 }')
    if [ "$SOURCE" = "contrib" ]; then
        CONFIG_PATH="${LIVEVIEW_CONF}/${EXPERIMENT_NAME}.yaml"
        # A contrib config already sets the LiveView target and its options.
        IS_LIVEVIEW_CONFIG=true
    else
        CONFIG_PATH="${MONTY_CONF}/${EXPERIMENT_NAME}.yaml"
    fi
else
    STATUS=$?
    echo "Error: Experiment config not found: ${EXPERIMENT_NAME}" >&2
    if [ "$STATUS" -eq 1 ]; then
        echo "" >&2
        echo "That name fits more than one experiment. Say which:" >&2
        printf '%s\n' "$RESOLVED" | awk '{ print "    " $2 }' >&2
    fi
    echo "" >&2
    echo "Available experiments (a path is the name Hydra knows):" >&2
    experiment_configs | awk '{ print "    " $2 }' >&2
    echo "" >&2
    echo "Or a part of a name, to see what fits:" >&2
    echo "    ./contrib/liveview_experiment/scripts/run.sh -l evidence" >&2
    exit 1
fi

echo "Starting LiveView Experiment Monitor..." >&2
echo "" >&2

# Check if uv is available (MuJoCo requires a native arm64 Python 3.11+ env,
# which this repo now manages via uv).
if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv not found. Please install uv (https://docs.astral.sh/uv/)." >&2
    exit 1
fi

# Check for core tbp.monty dependencies
if ! uv run python -c "import hydra" 2>/dev/null; then
    echo "Error: hydra-core not found in uv environment." >&2
    echo "" >&2
    echo "Please sync the uv environment first:" >&2
    echo "  cd $TBP_MONTY_ROOT" >&2
    echo "  uv sync --extra simulator_mujoco" >&2
    echo "" >&2
    echo "This will install all dependencies including hydra-core." >&2
    exit 1
fi

echo "✓ Environment: uv" >&2
echo "✓ Experiment: $EXPERIMENT_NAME" >&2
echo "✓ Config: $CONFIG_PATH" >&2
echo "" >&2

# Auto-install pyzmq if missing (needed for ZMQ communication)
if ! uv run python -c "import zmq" 2>/dev/null; then
    echo "Installing pyzmq..." >&2
    uv pip install --quiet "pyzmq>=25.0.0" >&2 || {
        echo "Error: Failed to install pyzmq" >&2
        exit 1
    }
    echo "✓ pyzmq installed" >&2
fi

# Start LiveView server early as a separate process (if Python 3.11+ available)
LIVEVIEW_SERVER_PID=""
TAIL_PID=""
LIVEVIEW_HOST="127.0.0.1"
LIVEVIEW_PORT="8000"
ZMQ_PORT="5555"

# Auto-create LiveView venv if missing (needs Python 3.11+ on host)
LIVEVIEW_VENV="${LIVEVIEW_DIR}/.liveview_venv"
if [ ! -d "$LIVEVIEW_VENV" ]; then
    # Find a Python 3.11+ on the host (prefer newest)
    LIVEVIEW_PYTHON_CMD=""
    for cmd in python3.14 python3.13 python3.12 python3.11 python3; do
        if command -v "$cmd" >/dev/null 2>&1; then
            ver=$($cmd --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
            maj=$(echo "$ver" | cut -d. -f1)
            min=$(echo "$ver" | cut -d. -f2)
            if [ "$maj" -ge 3 ] && [ "$min" -ge 11 ]; then
                LIVEVIEW_PYTHON_CMD="$cmd"
                echo "Creating LiveView venv with Python $ver..." >&2
                break
            fi
        fi
    done
    if [ -n "$LIVEVIEW_PYTHON_CMD" ]; then
        "$LIVEVIEW_PYTHON_CMD" -m venv "$LIVEVIEW_VENV" >&2 || true
        if [ -d "$LIVEVIEW_VENV" ]; then
            "$LIVEVIEW_VENV/bin/pip" install --upgrade --quiet pip >&2 || true
            "$LIVEVIEW_VENV/bin/pip" install --quiet -e "${LIVEVIEW_DIR}[liveview]" >&2 || true
            echo "✓ LiveView venv ready" >&2
        fi
    fi
fi

# Start LiveView server if venv exists
if [ -d "$LIVEVIEW_VENV" ]; then
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
echo "Starting experiment..." >&2
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

# Run the experiment with Hydra searchpath that includes both main and contrib directories
# This allows Hydra to find configs in both locations without copying files
# Using hydra.searchpath override to add both directories to search path
# Run in foreground so we see all output and signals work properly
cd "$TBP_MONTY_ROOT"

# Run experiment - if it hangs, Ctrl-C will trigger the trap
if [ "$IS_LIVEVIEW_CONFIG" = "true" ]; then
    # LiveView config exists - use it as-is (values from yaml are defaults)
    # Only override ports/host if needed (but they should already be in the config)
    uv run python run.py \
        "experiment=${EXPERIMENT_NAME}" \
        "hydra.searchpath=[${TBP_MONTY_ROOT}/src/tbp/monty/conf,${LIVEVIEW_DIR}/conf]" || {
        EXPERIMENT_EXIT_CODE=$?
        cleanup
        exit $EXPERIMENT_EXIT_CODE
    }
else
    # Monty's own config - add LiveView support via command line options, and keep
    # the experiment the config named: `MontyObjectRecognitionExperiment` and
    # `MontySupervisedObjectPretrainingExperiment` carry their own `pre_episode`
    # and `post_episode`, and pointing `_target_` straight at the LiveView class
    # would replace them with the generic one and run a different experiment.
    # The factory mixes the broadcasting into the class the config named, which
    # `--cfg job` reads out before the run starts.
    #
    # Also disable wandb and excessive logging.
    # Allow custom run name suffix via LIVEVIEW_RUN_NAME_SUFFIX env var.
    RUN_NAME_SUFFIX="${LIVEVIEW_RUN_NAME_SUFFIX:-_with_liveview}"

    BASE_EXPERIMENT_CLASS=$(uv run python run.py \
        "experiment=${EXPERIMENT_NAME}" \
        "hydra.searchpath=[${TBP_MONTY_ROOT}/src/tbp/monty/conf,${LIVEVIEW_DIR}/conf]" \
        --cfg job 2>/dev/null |
        uv run python -c 'import sys,yaml; print(yaml.safe_load(sys.stdin)["experiment"]["_target_"])' 2>/dev/null) || true
    if [ -z "$BASE_EXPERIMENT_CLASS" ]; then
        echo "Error: could not read the experiment class of ${EXPERIMENT_NAME}." >&2
        echo "       Its config has to compose on its own before LiveView can be added." >&2
        exit 1
    fi
    echo "✓ Experiment class: ${BASE_EXPERIMENT_CLASS}" >&2

    # The run's name is also a directory name, so a nested experiment's slashes
    # become dashes: `test/evidence_lm/evidence_mujoco` is one run, not a tree.
    RUN_NAME="${EXPERIMENT_NAME//\//_}"
    LITTERAL_RUN_NAME="${RUN_NAME}${RUN_NAME_SUFFIX}"

    uv run python run.py \
        "experiment=${EXPERIMENT_NAME}" \
        "hydra.searchpath=[${TBP_MONTY_ROOT}/src/tbp/monty/conf,${LIVEVIEW_DIR}/conf]" \
        "experiment._target_=contrib.liveview_experiment.src.experiment_factory.monty_experiment_with_liveview" \
        "+experiment.base_experiment_class=${BASE_EXPERIMENT_CLASS}" \
        "+experiment.config.liveview_host=${LIVEVIEW_HOST}" \
        "+experiment.config.liveview_port=${LIVEVIEW_PORT}" \
        "+experiment.config.zmq_port=${ZMQ_PORT}" \
        "+experiment.config.enable_liveview=true" \
        "+experiment.config.sensor_image_throttle_ms=100" \
        "experiment.config.monty_config.sensor_modules.sensor_module_0.save_raw_obs=true" \
        "logging=basic_warning_monty_runs" \
        "experiment.config.logging.run_name=${LITTERAL_RUN_NAME}" || {
        EXPERIMENT_EXIT_CODE=$?
        cleanup
        exit $EXPERIMENT_EXIT_CODE
    }
fi

EXPERIMENT_EXIT_CODE=$?

# Cleanup will be called by trap on exit, but explicit cleanup here too
cleanup
exit $EXPERIMENT_EXIT_CODE

