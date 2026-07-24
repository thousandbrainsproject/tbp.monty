# Setup script for LiveView Experiment Monitor
# Sets up separate Python 3.14+ environment for LiveView server
# Installs pyview-web and uvicorn only in the venv (not in main Python 3.8 environment)
# Installs pyzmq in main environment for ZMQ communication

$ErrorActionPreference = "Stop"

# Get script and project directories
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LiveViewDir = Split-Path -Parent $ScriptDir
$TbpMontyRoot = Split-Path -Parent (Split-Path -Parent $LiveViewDir)

Set-Location $TbpMontyRoot

Write-Host "Setting up LiveView Experiment Monitor..." -ForegroundColor Cyan
Write-Host ""

# Check if conda is available
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "Error: conda not found. Please install conda (Miniconda or Anaconda)." -ForegroundColor Red
    Write-Host "See: https://conda.io/projects/conda/en/latest/user-guide/install/index.html" -ForegroundColor Yellow
    exit 1
}

# Check if tbp.monty conda environment exists
$envList = conda env list 2>&1
if ($envList -notmatch "tbp\.monty\s") {
    Write-Host "Error: tbp.monty conda environment not found." -ForegroundColor Red
    Write-Host ""
    Write-Host "Please set up the tbp.monty environment first:" -ForegroundColor Yellow
    Write-Host "  1. cd $TbpMontyRoot"
    Write-Host "  2. conda env create"
    Write-Host "  3. conda activate tbp.monty"
    Write-Host ""
    exit 1
}

Write-Host "✓ Found tbp.monty conda environment" -ForegroundColor Green
Write-Host ""

# Activate conda environment
conda activate tbp.monty
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Could not activate tbp.monty environment." -ForegroundColor Red
    Write-Host "Please run: conda activate tbp.monty" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ Using conda environment: $env:CONDA_DEFAULT_ENV" -ForegroundColor Green
Write-Host ""

# Check Python version
$pythonVersion = python --version 2>&1 | ForEach-Object { $_.ToString().Split(' ')[1] }
Write-Host "✓ Python version: $pythonVersion" -ForegroundColor Green
Write-Host ""

# Verify core dependencies are installed
Write-Host "Checking core dependencies..." -ForegroundColor Cyan
$missingCore = @()

if (-not (python -c "import hydra" 2>$null)) {
    $missingCore += "hydra-core"
}
if (-not (python -c "import torch" 2>$null)) {
    $missingCore += "torch"
}

if ($missingCore.Count -gt 0) {
    Write-Host "Error: Missing core dependencies: $($missingCore -join ', ')" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install them first:" -ForegroundColor Yellow
    Write-Host "  conda activate tbp.monty"
    Write-Host "  cd $TbpMontyRoot"
    Write-Host "  pip install -e ."
    Write-Host ""
    exit 1
}

Write-Host "✓ Core dependencies installed" -ForegroundColor Green
Write-Host ""

# Parse Python version
$versionParts = $pythonVersion -split '\.'
$pythonMajor = [int]$versionParts[0]
$pythonMinor = [int]$versionParts[1]

# Initialize variables
$liveViewVenv = $null
$python311Cmd = $null

# Check if Python version supports pyview-web
if ($pythonMajor -ge 3 -and $pythonMinor -ge 11) {
    # Python 3.11+ - can install pyview-web directly (prefer 3.14+ for latest features)
    Write-Host "Upgrading pip..." -ForegroundColor Cyan
    python -m pip install --upgrade --quiet pip 2>&1 | Out-Null
    
    # Install the package with LiveView dependencies (only needed in Python 3.11+)
    Set-Location $LiveViewDir
    Write-Host "Installing/upgrading liveview-experiment package with LiveView dependencies..." -ForegroundColor Cyan
    python -m pip install --upgrade --quiet -e ".[liveview]" 2>&1 | Out-Null
    
    # Install dev dependencies for testing
    Write-Host "Installing/upgrading dev dependencies (black, ruff, mypy, pytest)..." -ForegroundColor Cyan
    python -m pip install --upgrade --quiet -e ".[dev]" 2>&1 | Out-Null
    
    Write-Host "✓ LiveView dependencies installed/upgraded" -ForegroundColor Green
} else {
    # Python 3.8-3.10 - set up separate Python 3.14+ environment for LiveView server (prefer latest)
    Write-Host "Python $pythonVersion detected (pyview-web requires >= 3.11, prefer 3.14+)" -ForegroundColor Yellow
    Write-Host "Setting up separate Python 3.14+ environment for LiveView server..." -ForegroundColor Cyan
    Write-Host ""
    
    # Check if Python 3.14+ is available (prefer latest, fallback to 3.11+)
    # Variables already initialized above
    $python311Version = $null
    
    $pythonCommands = @("python3.14", "python3.13", "python3.12", "python3.11", "python3")
    foreach ($cmd in $pythonCommands) {
        if (Get-Command $cmd -ErrorAction SilentlyContinue) {
            $versionOutput = & $cmd --version 2>&1
            $versionStr = ($versionOutput -split ' ')[1]
            $versionParts = $versionStr -split '\.'
            $major = [int]$versionParts[0]
            $minor = [int]$versionParts[1]
            
            if ($major -ge 3 -and $minor -ge 11) {
                $python311Cmd = $cmd
                $python311Version = $versionStr
                Write-Host "✓ Found Python $python311Version at: $python311Cmd" -ForegroundColor Green
                if ($minor -ge 14) {
                    Write-Host "  Using Python 3.14+ for latest pyview-web features" -ForegroundColor Cyan
                }
                break
            }
        }
    }
    
    if (-not $python311Cmd) {
        Write-Host "Warning: No Python 3.11+ found for LiveView server" -ForegroundColor Yellow
        Write-Host "  - Web dashboard will not be available" -ForegroundColor Yellow
        Write-Host "  - Pub/sub streaming system will still work" -ForegroundColor Yellow
        Write-Host "  - Install Python 3.14+ (or 3.11+) to enable web dashboard" -ForegroundColor Yellow
        Write-Host ""
    } else {
        # Create or update virtual environment for the LiveView server
        $liveViewVenv = Join-Path $LiveViewDir ".liveview_venv"
        
        if (-not (Test-Path $liveViewVenv)) {
            Write-Host "Creating virtual environment for LiveView server..." -ForegroundColor Cyan
            & $python311Cmd -m venv $liveViewVenv 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) {
                Write-Host "Warning: Could not create virtual environment" -ForegroundColor Yellow
                $python311Cmd = $null
            }
        } else {
            Write-Host "Updating existing LiveView virtual environment..." -ForegroundColor Cyan
        }
        
        if ($python311Cmd -and (Test-Path $liveViewVenv)) {
            # Handle both Windows (Scripts) and Unix (bin) paths
            if ($IsWindows -or $env:OS -eq "Windows_NT") {
                $venvPython = Join-Path $liveViewVenv "Scripts\python.exe"
                $venvPip = Join-Path $liveViewVenv "Scripts\pip.exe"
            } else {
                $venvPython = Join-Path $liveViewVenv "bin\python"
                $venvPip = Join-Path $liveViewVenv "bin\pip"
            }
            
            # Upgrade pip first
            & $venvPip install --upgrade --quiet pip 2>&1 | Out-Null
            
            # Install the package with LiveView dependencies (only in the venv)
            Set-Location $LiveViewDir
            Write-Host "Installing/upgrading liveview-experiment package with LiveView dependencies..." -ForegroundColor Cyan
            & $venvPip install --upgrade --quiet -e ".[liveview]" 2>&1 | Out-Null
            
            # Install dev dependencies for testing
            Write-Host "Installing/upgrading dev dependencies (black, ruff, mypy, pytest)..." -ForegroundColor Cyan
            & $venvPip install --upgrade --quiet -e ".[dev]" 2>&1 | Out-Null
            
            Write-Host "✓ LiveView server environment ready" -ForegroundColor Green
            Write-Host "  - Main experiment: Python $pythonVersion" -ForegroundColor Cyan
            Write-Host "  - LiveView server: Python $python311Version (separate process)" -ForegroundColor Cyan
            Write-Host ""
        }
    }
}

Write-Host ""

# Verify config file exists
$configSource = Join-Path $LiveViewDir "conf\experiment\randrot_10distinctobj_surf_agent_with_liveview.yaml"

if (-not (Test-Path $configSource)) {
    Write-Host "Warning: Config file not found at $configSource" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ Experiment config found at:" -ForegroundColor Green
Write-Host "  $configSource" -ForegroundColor Cyan
Write-Host ""

# Install pyzmq in main environment (needed for ZMQ communication from experiment)
# LiveView dependencies (pyview-web, uvicorn) are only installed in the venv
if (-not (python -c "import zmq" 2>$null)) {
    Write-Host "Installing pyzmq for ZMQ communication..." -ForegroundColor Cyan
    python -m pip install --quiet "pyzmq>=25.0.0" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: Could not install pyzmq. ZMQ communication may not work." -ForegroundColor Yellow
    } else {
        Write-Host "✓ pyzmq installed" -ForegroundColor Green
    }
} else {
    Write-Host "✓ pyzmq available" -ForegroundColor Green
}

# Dev dependencies handling:
# - If Python 3.11+, dev dependencies were already installed above with the package
# - If Python < 3.11 and venv was created, dev dependencies are in the venv
# - test.ps1 will use the venv (Python 3.14+) for running tests, so dev deps are available there
# - We don't install the package in Python 3.8 environment (LiveView deps not needed there)
if ($pythonMajor -ge 3 -and $pythonMinor -ge 11) {
    # We're already in Python 3.11+, dev dependencies were installed above
    Write-Host "✓ Dev dependencies available (installed with package)" -ForegroundColor Green
} elseif ($python311Cmd -and $liveViewVenv -and (Test-Path $liveViewVenv)) {
    # Using separate venv - dev dependencies are already installed there
    Write-Host "✓ Dev dependencies available in LiveView venv (Python 3.14+)" -ForegroundColor Green
    Write-Host "  Note: Run tests with: .\scripts\test.ps1 (uses LiveView venv)" -ForegroundColor Cyan
    Write-Host "  Note: LiveView dependencies only installed in venv, not in main Python 3.8 environment" -ForegroundColor Cyan
} else {
    # No venv available and Python < 3.11 - skip dev deps (they require pyview-web which needs 3.11+)
    Write-Host "Note: Dev dependencies not installed in Python 3.8 environment" -ForegroundColor Yellow
    Write-Host "  They require pyview-web (needs Python 3.11+)" -ForegroundColor Yellow
    if (-not $python311Cmd) {
        Write-Host "  Install Python 3.14+ to enable dev dependencies and testing" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Setup complete! ✓" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Run an experiment: .\contrib\liveview_experiment\scripts\run.ps1" -ForegroundColor White
Write-Host "  2. View dashboard: http://127.0.0.1:8000" -ForegroundColor White
Write-Host ""

