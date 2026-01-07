# LiveView Experiment Scripts

One-shot setup and run scripts for LiveView experiment monitoring.

## Setup Script

### `setup.sh`

Installs pyview-web and uvicorn in the existing tbp.monty conda environment (Python 3.8).

**Usage:**
```bash
./contrib/liveview_experiment/scripts/setup.sh
```

**Prerequisites:**
- tbp.monty conda environment must exist and be set up
- Core dependencies (hydra-core, torch) must be installed

**What it does:**
- Verifies tbp.monty environment exists
- Checks core dependencies are installed
- Installs pyview-web>=0.7.0 and uvicorn[standard]>=0.32.0 (same as my_mvg_departures)

## Run Script

### `run.sh`

Runs a Monty experiment with LiveView monitoring.

**Usage:**
```bash
./contrib/liveview_experiment/scripts/run.sh [experiment_name]
```

**Default:** `tutorial_surf_agent_2obj_with_liveview`

**What it does:**
- Automatically detects and activates `tbp.monty` environment
- Verifies pyview-web and uvicorn are installed
- Runs the experiment with LiveView dashboard at http://127.0.0.1:8000

### Running a bigger experiment

The default `tutorial_surf_agent_2obj_with_liveview` is a small, fast tutorial
configuration. To stress‑test the LiveView (more objects, more steps), you can
run one of the larger contrib experiments. For example, from the `tbp.monty`
repo root:

```bash
./contrib/liveview_experiment/scripts/run.sh randrot_10distinctobj_surf_agent_with_liveview
```

This will:

- Start the standalone LiveView server on `http://127.0.0.1:8000`
- Run the `randrot_10distinctobj_surf_agent_with_liveview` experiment in the
  `tbp.monty` conda environment
- Stream live evidence curves and sensor images into the dashboard

## Quick Start

```bash
# 1. Set up tbp.monty environment (if not already done)
conda env create
conda activate tbp.monty
pip install -e .

# 2. Setup LiveView (one-time)
./contrib/liveview_experiment/scripts/setup.sh

# 3. Run experiment
./contrib/liveview_experiment/scripts/run.sh

# 4. View dashboard
# Open http://127.0.0.1:8000 in your browser
```
