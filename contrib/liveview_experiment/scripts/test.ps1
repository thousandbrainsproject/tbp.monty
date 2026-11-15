# Test script for LiveView Experiment
# Runs all CI checks: formatting, linting, type checking, and tests
# Formatting, linting and typing run in tbp.monty's uv environment; the tests
# prefer the dashboard venv, which is the only place pyview lives.

$ErrorActionPreference = "Stop"

# Get the directory where this script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LiveViewDir = Split-Path -Parent $ScriptDir
$TbpMontyRoot = Split-Path -Parent (Split-Path -Parent $LiveViewDir)
$Config = "contrib/liveview_experiment"

# `uv run` finds its project by walking up from the working directory, and the
# contrib directory has a pyproject.toml of its own.
Set-Location $TbpMontyRoot

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Running CI checks..." -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "1. Checking code formatting with ruff..." -ForegroundColor Cyan
uv run ruff format --check "$Config/src" "$Config/tests" "$Config/scripts"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "✓ Formatting check passed" -ForegroundColor Green
Write-Host ""

Write-Host "2. Linting with ruff..." -ForegroundColor Cyan
# The tests insert the project root into sys.path before importing, so only the
# source and the tooling scripts are linted.
uv run ruff check "$Config/src" "$Config/scripts"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "✓ Linting passed" -ForegroundColor Green
Write-Host ""

Write-Host "3. Type checking with mypy..." -ForegroundColor Cyan
uv run mypy --config-file "$Config/pyproject.toml" "$Config/src"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "✓ Type checking passed" -ForegroundColor Green
Write-Host ""

Write-Host "4. Running tests..." -ForegroundColor Cyan
$TestsDir = Join-Path $LiveViewDir "tests"
$venvPython = Join-Path (Join-Path (Join-Path $LiveViewDir ".liveview_venv") "Scripts") "python.exe"
if (-not (Test-Path $venvPython)) {
    $venvPython = Join-Path (Join-Path (Join-Path $LiveViewDir ".liveview_venv") "bin") "python"
}
uv run python -m pytest -m "not integration" $TestsDir --ignore (Join-Path $TestsDir "test_template_parsing.py") @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "✓ Tests passed" -ForegroundColor Green
Write-Host ""

# The template tests import pyview, which only the dashboard venv has.
if (Test-Path $venvPython) {
    Write-Host "5. Running the template tests in the dashboard venv..." -ForegroundColor Cyan
    & $venvPython -m pytest (Join-Path $TestsDir "test_template_parsing.py") @args
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "✓ Template tests passed" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "5. Skipping the template tests (no .liveview_venv; run .\scripts\setup.ps1)" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "All checks passed! ✓" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan

