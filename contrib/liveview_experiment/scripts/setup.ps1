# Setup script for LiveView Experiment Monitor
# Syncs the tbp.monty uv environment (Python 3.13) and creates the venv the
# dashboard process runs in, with pyview-web and uvicorn.
# Installs pyzmq into the experiment environment for ZMQ communication

$ErrorActionPreference = "Stop"

# Get script and project directories
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LiveViewDir = Split-Path -Parent $ScriptDir
$TbpMontyRoot = Split-Path -Parent (Split-Path -Parent $LiveViewDir)

Set-Location $TbpMontyRoot

Write-Host "Setting up LiveView Experiment Monitor..." -ForegroundColor Cyan
Write-Host ""

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Error: uv not found. Install it from https://docs.astral.sh/uv/." -ForegroundColor Red
    Write-Host "See docs/how-to-use-monty/getting-started.md for details." -ForegroundColor Yellow
    exit 1
}

# tbp.monty's `.python-version` (3.13) is the interpreter both MuJoCo and
# pyview-web support; `uv sync` puts the experiment's dependencies in place.
Write-Host "Syncing the tbp.monty environment..." -ForegroundColor Cyan
uv sync --extra dev --extra simulator_mujoco

$pythonVersion = uv run python -c "import platform; print(platform.python_version())"
Write-Host "✓ tbp.monty environment ready (Python $pythonVersion)" -ForegroundColor Green
Write-Host ""

# The experiment's environment is the uv one; pyzmq is the contrib's own
# dependency, so it is added beside Monty's.
if (-not (uv run python -c "import zmq" 2>$null)) {
    Write-Host "Installing pyzmq into the tbp.monty environment..." -ForegroundColor Cyan
    uv pip install --quiet "pyzmq>=25.0.0"
}

# The dashboard process gets pyview-web and uvicorn in a venv of its own,
# running on the same interpreter as the experiment.
$liveViewVenv = Join-Path $LiveViewDir ".liveview_venv"
$venvPython = Join-Path (Join-Path $liveViewVenv "Scripts") "python.exe"
if (-not (Test-Path $venvPython)) {
    $venvPython = Join-Path (Join-Path $liveViewVenv "bin") "python"
}
$envVersion = $pythonVersion -replace '^(\d+\.\d+).*$', '$1'
$venvVersion = if (Test-Path $venvPython) {
    & $venvPython -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
} else {
    ""
}
if ($venvVersion -and $venvVersion -ne $envVersion) {
    Write-Host "Recreating the LiveView venv for Python $envVersion..." -ForegroundColor Cyan
    Remove-Item -Recurse -Force $liveViewVenv
}
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating the LiveView venv with Python $envVersion..." -ForegroundColor Cyan
    uv venv --python $envVersion $liveViewVenv
    $venvPython = Join-Path (Join-Path $liveViewVenv "Scripts") "python.exe"
    if (-not (Test-Path $venvPython)) {
        $venvPython = Join-Path (Join-Path $liveViewVenv "bin") "python"
    }
}

Write-Host "Installing liveview-experiment with the liveview and dev extras..." -ForegroundColor Cyan
uv pip install --python $venvPython -e "$LiveViewDir[liveview,dev]"
Write-Host "✓ LiveView dashboard environment ready" -ForegroundColor Green

Write-Host ""
Write-Host "Setup complete! ✓" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  Run an experiment: .\contrib\liveview_experiment\scripts\run.sh [experiment]" -ForegroundColor White
Write-Host "  View dashboard:    http://127.0.0.1:8000" -ForegroundColor White
Write-Host ""

