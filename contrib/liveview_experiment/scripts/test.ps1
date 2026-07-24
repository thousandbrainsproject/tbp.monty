# Test script for LiveView Experiment
# Runs all CI checks: formatting, linting, type checking, and tests
# Uses Python 3.14+ environment (LiveView venv) if available, otherwise tbp.monty (prefer latest)

$ErrorActionPreference = "Stop"

# Get the directory where this script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LiveViewDir = Split-Path -Parent $ScriptDir

Set-Location $LiveViewDir

# Check for LiveView venv (Python 3.11+) first
$liveViewVenv = Join-Path $LiveViewDir ".liveview_venv"

# Handle both Windows (Scripts) and Unix (bin) paths
if ($IsWindows -or $env:OS -eq "Windows_NT") {
    $venvPython = Join-Path $liveViewVenv "Scripts\python.exe"
    $venvPip = Join-Path $liveViewVenv "Scripts\pip.exe"
} else {
    $venvPython = Join-Path $liveViewVenv "bin\python"
    $venvPip = Join-Path $liveViewVenv "bin\pip"
}

if ((Test-Path $liveViewVenv) -and (Test-Path $venvPython)) {
    $env:PYTHON = $venvPython
    $env:PIP = $venvPip
    $pythonCmd = $venvPython
    $pipCmd = $venvPip
    
    Write-Host "Using LiveView Python 3.14+ environment (prefer latest)" -ForegroundColor Cyan
    
    # Install dev dependencies if not present
    # Note: In venv, we need both liveview and dev extras
    $blackCheck = & $pythonCmd -m black --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing dev dependencies in LiveView environment..." -ForegroundColor Cyan
        & $pipCmd install -e ".[liveview,dev]" 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error: Failed to install dev dependencies. Please run .\scripts\setup.ps1 first or install dependencies manually." -ForegroundColor Red
            exit 1
        }
    }
} else {
    # Fall back to conda environment (tbp.monty)
    if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
        Write-Host "Error: conda not found and LiveView venv not available." -ForegroundColor Red
        exit 1
    }
    
    # Activate conda environment
    conda activate tbp.monty
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Could not activate tbp.monty environment." -ForegroundColor Red
        exit 1
    }
    
    $pythonCmd = "python"
    $pipCmd = "pip"
    
    # Check if dependencies are installed
    # Note: In Python 3.8 environment, we only install dev tools (no liveview extras)
    $blackCheck = python -m black --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Dependencies not found. Installing dev tools (without LiveView dependencies)..." -ForegroundColor Cyan
        # Install dev tools individually (pyview-web requires Python 3.11+)
        pip install "black>=24.0.0" "ruff>=0.4.0" "mypy>=1.0.0" "pytest>=8.0.0" "vulture>=2.0.0" 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error: Failed to install dev tools. Please run .\scripts\setup.ps1 first or install dependencies manually." -ForegroundColor Red
            exit 1
        }
    }
}

# Function to run Python module
function Run-PythonModule {
    param([string[]]$Args)
    & $pythonCmd -u -m $Args
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Formatting..." -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if reformat script exists
$reformatScript = Join-Path $ScriptDir "reformat.ps1"
if (Test-Path $reformatScript) {
    & $reformatScript
} else {
    Write-Host "Note: reformat.ps1 not found, skipping auto-formatting" -ForegroundColor Yellow
}

# Run all CI checks
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Running CI checks..." -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check code formatting with Black
Write-Host "1. Checking code formatting with Black..." -ForegroundColor Cyan
Run-PythonModule @("black", "--check", "src", "scripts")
Write-Host "✓ Formatting check passed" -ForegroundColor Green
Write-Host ""

# 2. Lint with Ruff
Write-Host "2. Linting with Ruff..." -ForegroundColor Cyan
Run-PythonModule @("ruff", "check", "src", "scripts")
Write-Host "✓ Linting passed" -ForegroundColor Green
Write-Host ""

# 3. Type check with mypy
Write-Host "3. Type checking with mypy..." -ForegroundColor Cyan
Run-PythonModule @("mypy", "src")
Write-Host "✓ Type checking passed" -ForegroundColor Green
Write-Host ""

# 4. Run tests (if tests directory exists)
$testsDir = Join-Path $LiveViewDir "tests"
if (Test-Path $testsDir) {
    Write-Host "4. Running tests..." -ForegroundColor Cyan
    $testArgs = @("pytest", "-m", "not integration")
    if ($args.Count -gt 0) {
        $testArgs += $args
    }
    Run-PythonModule $testArgs
    Write-Host "✓ Tests passed" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "4. No tests directory found, skipping tests" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "All checks passed! ✓" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan

