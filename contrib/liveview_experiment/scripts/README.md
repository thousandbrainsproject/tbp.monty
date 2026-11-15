# LiveView Experiment Scripts

One-shot setup and run scripts for LiveView experiment monitoring.

## Setup Script

### `setup.sh`

Sets up LiveView experiment monitoring with separate Python environments.

**Usage:**
```bash
./contrib/liveview_experiment/scripts/setup.sh
```

**Prerequisites:**
- [uv](https://docs.astral.sh/uv/) installed
- Python 3.13, which is what tbp.monty's `.python-version` pins (pyview-web requires 3.11–3.14, MuJoCo supports 3.11+)

**What it does:**
- Runs `uv sync --extra dev --extra simulator_mujoco` for the experiment
- Installs `pyzmq` into the tbp.monty environment (for ZMQ communication)
- Creates `.liveview_venv` on the same Python for the LiveView server, recreating it when the interpreter changes
- Installs `pyview-web>=0.8.1`, `uvicorn[standard]>=0.32.0` and the dev tools (ruff, mypy, pytest, radon, vulture) only in that venv

## Run Script

### `run.sh`

Runs a Monty experiment with LiveView monitoring.

**Usage:**
```bash
./contrib/liveview_experiment/scripts/run.sh [experiment_name]
```

**Default:** `randrot_noise_77obj_surf_agent`

**What it does:**
- Resolves the experiment name against `src/tbp/monty/conf/experiment` (a path is the name Hydra knows; `-l WORD` lists what fits)
- Starts the LiveView server from `.liveview_venv` (against `scripts/setup.sh` if it is missing)
- Runs the experiment with `uv run` and the LiveView dashboard at http://127.0.0.1:8000

### Running a bigger experiment

The default `randrot_noise_77obj_surf_agent` is a full benchmark run. For a
small, fast tutorial configuration, name one of the tutorial experiments. For
example, from the `tbp.monty` repo root:

```bash
./contrib/liveview_experiment/scripts/run.sh tutorial/surf_agent_2obj_train
```

This will:

- Start the standalone LiveView server on `http://127.0.0.1:8000`
- Run the named experiment in the `tbp.monty` uv environment
- Stream live evidence curves and sensor images into the dashboard

## Quick Start

```bash
# 1. Set up tbp.monty environment (if not already done)
uv sync --extra dev --extra simulator_mujoco

# 2. Setup LiveView (one-time)
./contrib/liveview_experiment/scripts/setup.sh

# 3. Run experiment
./contrib/liveview_experiment/scripts/run.sh

# 4. View dashboard
# Open http://127.0.0.1:8000 in your browser
```
