#!/bin/bash
# Setup script for LiveView Experiment Monitor
# Installs pyview-web and uvicorn in the existing tbp.monty conda environment (Python 3.8)

set -euo pipefail

# Find script and project directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIVEVIEW_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TBP_MONTY_ROOT="$(cd "${LIVEVIEW_DIR}/../.." && pwd)"

cd "$TBP_MONTY_ROOT"

echo "Setting up LiveView Experiment Monitor..." >&2
echo "" >&2

# Check if conda is available
if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda not found. Please install conda (Miniconda or Anaconda)." >&2
    echo "See: https://conda.io/projects/conda/en/latest/user-guide/install/index.html" >&2
    exit 1
fi

# Check if tbp.monty conda environment exists
if ! conda env list | grep -q "^tbp.monty "; then
    echo "Error: tbp.monty conda environment not found." >&2
    echo "" >&2
    echo "Please set up the tbp.monty environment first:" >&2
    echo "  1. cd $TBP_MONTY_ROOT" >&2
    echo "  2. conda env create" >&2
    echo "  3. conda activate tbp.monty" >&2
    echo "" >&2
    echo "See docs/how-to-use-monty/getting-started.md for details." >&2
    exit 1
fi

echo "✓ Found tbp.monty conda environment" >&2
echo "" >&2

# Check if environment is activated
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "tbp.monty" ]; then
    echo "Warning: tbp.monty conda environment is not activated." >&2
    echo "Activating now..." >&2
    echo "" >&2
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate tbp.monty || {
        echo "Error: Could not activate tbp.monty environment." >&2
        echo "Please run: conda activate tbp.monty" >&2
        exit 1
    }
fi

echo "✓ Using conda environment: $CONDA_DEFAULT_ENV" >&2
echo "" >&2

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION" >&2
echo "" >&2

# Verify core dependencies are installed
echo "Checking core dependencies..." >&2
MISSING_CORE=()

if ! python -c "import hydra" 2>/dev/null; then
    MISSING_CORE+=("hydra-core")
fi

if ! python -c "import torch" 2>/dev/null; then
    MISSING_CORE+=("torch")
fi

if [ ${#MISSING_CORE[@]} -gt 0 ]; then
    echo "Missing core dependencies: ${MISSING_CORE[*]}" >&2
    echo "Installing tbp.monty package..." >&2
    echo "" >&2
    
    # Workaround for antlr4-python3-runtime installation issue with Python 3.8 + newer setuptools
    echo "Applying workaround for antlr4-python3-runtime + setuptools compatibility..." >&2
    
    # Check setuptools version and downgrade if needed (required for antlr4 4.9.x)
    # antlr4 4.9.x's setup.py is incompatible with setuptools >= 70.0
    SETUPTOOLS_VERSION=$(pip show setuptools 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "")
    SETUPTOOLS_MAJOR=$(echo "$SETUPTOOLS_VERSION" | cut -d. -f1)
    NEED_DOWNGRADE=false
    
    if [ -n "$SETUPTOOLS_VERSION" ] && [ -n "$SETUPTOOLS_MAJOR" ] && [ "$SETUPTOOLS_MAJOR" -ge 70 ]; then
        NEED_DOWNGRADE=true
        echo "Detected setuptools $SETUPTOOLS_VERSION (>=70.0)" >&2
        echo "Downgrading to setuptools<70.0 for antlr4-python3-runtime 4.9.x compatibility..." >&2
        pip install "setuptools<70.0" >&2 2>&1 || {
            echo "Warning: Could not downgrade setuptools, trying alternative approach..." >&2
            NEED_DOWNGRADE=false
        }
    fi
    
    echo "" >&2
    echo "Installing tbp.monty package..." >&2
    
    # Now install the package (antlr4 4.9.x should install successfully with downgraded setuptools)
    if pip install -e . >&2 2>&1; then
        echo "✓ Successfully installed tbp.monty package" >&2
        echo "" >&2
    else
        echo "Error: Failed to install tbp.monty package" >&2
        echo "" >&2
        echo "This may be due to antlr4-python3-runtime compatibility issues." >&2
        echo "Try recreating the conda environment:" >&2
        echo "  conda env remove -n tbp.monty" >&2
        echo "  conda env create -f environment.yml" >&2
        echo "  conda activate tbp.monty" >&2
        echo "" >&2
        exit 1
    fi
    
    # Restore original setuptools if we downgraded (after successful install)
    if [ "$NEED_DOWNGRADE" = "true" ] && [ -n "$SETUPTOOLS_VERSION" ]; then
        echo "Restoring setuptools to $SETUPTOOLS_VERSION..." >&2
        pip install "setuptools==$SETUPTOOLS_VERSION" >&2 2>&1 || true
        echo "" >&2
    fi
    
    # Verify installation succeeded
    if ! python -c "import hydra" 2>/dev/null; then
        echo "Error: Installation completed but hydra-core is still not importable" >&2
        exit 1
    fi
fi

echo "✓ Core dependencies found" >&2
echo "" >&2

# Install LiveView dependencies
echo "Installing LiveView dependencies..." >&2

PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

# Check if Python version supports pyview-web
if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
    # Python 3.11+ - can install pyview-web directly
    if ! python -c "import pyview" 2>/dev/null; then
        echo "Installing pyview-web..." >&2
        pip install "pyview-web>=0.7.0" >&2 || {
            echo "Warning: Failed to install pyview-web" >&2
        }
        echo "✓ pyview-web installed" >&2
    else
        echo "✓ pyview-web already installed" >&2
    fi
    
    if ! python -c "import uvicorn" 2>/dev/null; then
        echo "Installing uvicorn..." >&2
        pip install "uvicorn[standard]>=0.32.0" >&2 || {
            echo "Warning: Failed to install uvicorn" >&2
        }
        echo "✓ uvicorn installed" >&2
    else
        echo "✓ uvicorn already installed" >&2
    fi
else
    # Python 3.8-3.10 - set up separate Python 3.11+ environment for LiveView server
    echo "Python $PYTHON_VERSION detected (pyview-web requires >= 3.11)" >&2
    echo "Setting up separate Python 3.11+ environment for LiveView server..." >&2
    echo "" >&2
    
    # Check if Python 3.11+ is available
    PYTHON311_CMD=""
    for cmd in python3.11 python3.12 python3.13 python3; do
        if command -v "$cmd" >/dev/null 2>&1; then
            PYTHON311_VERSION=$($cmd --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
            PYTHON311_MAJOR=$(echo "$PYTHON311_VERSION" | cut -d. -f1)
            PYTHON311_MINOR=$(echo "$PYTHON311_VERSION" | cut -d. -f2)
            if [ "$PYTHON311_MAJOR" -ge 3 ] && [ "$PYTHON311_MINOR" -ge 11 ]; then
                PYTHON311_CMD="$cmd"
                echo "✓ Found Python $PYTHON311_VERSION at: $PYTHON311_CMD" >&2
                break
            fi
        fi
    done
    
    if [ -z "$PYTHON311_CMD" ]; then
        echo "Warning: No Python 3.11+ found for LiveView server" >&2
        echo "  - Web dashboard will not be available" >&2
        echo "  - Pub/sub streaming system will still work" >&2
        echo "  - Install Python 3.11+ to enable web dashboard" >&2
        echo "" >&2
    else
        # Create a virtual environment for the LiveView server
        LIVEVIEW_VENV="${LIVEVIEW_DIR}/.liveview_venv"
        if [ ! -d "$LIVEVIEW_VENV" ]; then
            echo "Creating virtual environment for LiveView server..." >&2
            "$PYTHON311_CMD" -m venv "$LIVEVIEW_VENV" >&2 || {
                echo "Warning: Could not create virtual environment" >&2
                PYTHON311_CMD=""
            }
        fi
        
        if [ -n "$PYTHON311_CMD" ] && [ -d "$LIVEVIEW_VENV" ]; then
            echo "Installing pyview-web and pyzmq in LiveView virtual environment..." >&2
            "$LIVEVIEW_VENV/bin/pip" install --quiet "pyview-web>=0.7.0" "uvicorn[standard]>=0.32.0" "pyzmq>=25.0.0" >&2 || {
                echo "Warning: Could not install dependencies in virtual environment" >&2
            }
            echo "✓ LiveView server environment ready" >&2
            echo "  - Main experiment: Python $PYTHON_VERSION" >&2
            echo "  - LiveView server: Python $PYTHON311_VERSION (separate process)" >&2
            echo "" >&2
        fi
    fi
fi

echo "" >&2

# Verify config file exists
CONFIG_SOURCE="${LIVEVIEW_DIR}/conf/experiment/randrot_10distinctobj_surf_agent_with_liveview.yaml"

if [ ! -f "$CONFIG_SOURCE" ]; then
    echo "Warning: Config file not found at $CONFIG_SOURCE" >&2
    exit 1
fi

echo "✓ Experiment config found at:" >&2
echo "  $CONFIG_SOURCE" >&2
echo "" >&2

# Install pyzmq for ZMQ-based communication (required for cross-process pub/sub)
if ! python -c "import zmq" 2>/dev/null; then
    echo "Installing pyzmq for ZMQ communication..." >&2
    pip install pyzmq >&2 || {
        echo "Error: Could not install pyzmq. ZMQ communication will not work." >&2
        exit 1
    }
    echo "✓ pyzmq installed" >&2
else
    echo "✓ pyzmq already installed" >&2
fi

# Install radon for complexity analysis (optional but recommended)
if ! python -c "import radon" 2>/dev/null; then
    echo "Installing radon for complexity analysis..." >&2
    pip install radon >&2 || {
        echo "Warning: Could not install radon. Complexity analysis will be limited." >&2
    }
    echo "✓ radon installed (optional, for code complexity analysis)" >&2
else
    echo "✓ radon already installed" >&2
fi

# Install dev dependencies for code quality checks from pyproject.toml
echo "Installing dev dependencies for code quality checks..." >&2
cd "$LIVEVIEW_DIR"
if [ -f "pyproject.toml" ]; then
    if pip install -e ".[dev]" >&2; then
        echo "✓ Dev dependencies installed from pyproject.toml" >&2
    else
        echo "Warning: Could not install dev dependencies from pyproject.toml" >&2
        echo "Falling back to individual installations..." >&2
        pip install "black>=24.0.0" "ruff>=0.4.0" "mypy>=1.0.0" "pytest>=8.0.0" "vulture>=2.0.0" >&2 || {
            echo "Warning: Could not install dev dependencies. Code quality checks may not work." >&2
        }
    fi
else
    echo "Warning: pyproject.toml not found, installing dev dependencies individually..." >&2
    pip install "black>=24.0.0" "ruff>=0.4.0" "mypy>=1.0.0" "pytest>=8.0.0" "vulture>=2.0.0" >&2 || {
        echo "Warning: Could not install dev dependencies. Code quality checks may not work." >&2
    }
fi
cd "$TBP_MONTY_ROOT"

echo "" >&2
echo "✓ Setup complete!" >&2
echo "" >&2

# Check if web dashboard is available
if python -c "import pyview" 2>/dev/null; then
    echo "You can now run an experiment with LiveView web dashboard:" >&2
    echo "  ./contrib/liveview_experiment/scripts/run.sh" >&2
    echo "" >&2
    echo "Then open http://127.0.0.1:8000 in your browser to view the dashboard." >&2
else
    echo "You can now run an experiment (pub/sub streaming enabled):" >&2
    echo "  ./contrib/liveview_experiment/scripts/run.sh" >&2
    echo "" >&2
    echo "Note: Web dashboard not available on Python $PYTHON_VERSION" >&2
    echo "      Use experiment.broadcaster to stream data from parallel processes." >&2
fi
echo "" >&2
echo "Or manually:" >&2
echo "  conda activate tbp.monty" >&2
echo "  python run.py experiment=randrot_10distinctobj_surf_agent_with_liveview \\" >&2
echo "    hydra.searchpath=[contrib/liveview_experiment/conf]" >&2
echo "" >&2

