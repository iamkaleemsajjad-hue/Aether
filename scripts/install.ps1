# =============================================================================
# Aether Runtime — One-Click Install Script (Windows PowerShell)
# =============================================================================
#
# Usage:
#   .\scripts\install.ps1                  # Install with auto-detected hardware
#   .\scripts\install.ps1 -Dev            # Install development dependencies too
#   .\scripts\install.ps1 -Cuda           # Force CUDA extras
#   .\scripts\install.ps1 -CpuOnly        # CPU-only (no GPU extras)
#   .\scripts\install.ps1 -NoVenv         # Skip virtual environment creation
#   .\scripts\install.ps1 -VenvDir myenv  # Custom venv directory
#
# Requirements:
#   - Python 3.10+ (from python.org or Windows Store)
#   - pip 23+
#   - PowerShell 5.1+ or PowerShell 7+
#   - (Optional) NVIDIA GPU + CUDA 11.8+ for GPU acceleration
#
# Run policy note:
#   If you get an "execution policy" error, run PowerShell as Administrator and:
#     Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
#
# =============================================================================

param(
    [switch]$Dev,
    [switch]$Cuda,
    [switch]$CpuOnly,
    [switch]$NoVenv,
    [string]$VenvDir = ".venv"
)

$ErrorActionPreference = "Stop"

# --------------------------------------------------------------------------- #
# Colour output helpers                                                         #
# --------------------------------------------------------------------------- #
function Write-Info    { param($msg) Write-Host "[INFO]  $msg" -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Host "[OK]    $msg" -ForegroundColor Green }
function Write-Warn    { param($msg) Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Write-Header  { param($msg) Write-Host "`n$msg" -ForegroundColor White }
function Write-Err     { param($msg) Write-Host "[ERR]   $msg" -ForegroundColor Red; exit 1 }

# --------------------------------------------------------------------------- #
# Header                                                                        #
# --------------------------------------------------------------------------- #
Write-Header "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Header "  Aether Runtime — Windows Installer"
Write-Header "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# --------------------------------------------------------------------------- #
# Check Python                                                                  #
# --------------------------------------------------------------------------- #
Write-Header "Checking prerequisites..."

$PythonCmd = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]; $minor = [int]$Matches[2]
            if ($major -ge 3 -and $minor -ge 10) {
                $PythonCmd = $cmd
                Write-Success "Python $major.$minor found ($cmd)"
                break
            }
        }
    } catch { }
}
if (-not $PythonCmd) {
    Write-Err "Python 3.10+ not found. Install from https://python.org and re-run."
}

# --------------------------------------------------------------------------- #
# Virtual environment                                                           #
# --------------------------------------------------------------------------- #
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot  = Split-Path -Parent $ScriptDir
$VenvPath  = Join-Path $RepoRoot $VenvDir
$PipCmd    = $PythonCmd

if (-not $NoVenv) {
    Write-Header "Setting up virtual environment: $VenvPath"
    if (-not (Test-Path $VenvPath)) {
        & $PythonCmd -m venv $VenvPath
        Write-Success "Virtual environment created"
    } else {
        Write-Info "Virtual environment already exists — reusing"
    }
    $PipCmd    = Join-Path $VenvPath "Scripts\pip.exe"
    $PythonCmd = Join-Path $VenvPath "Scripts\python.exe"
}

Write-Info "Upgrading pip..."
& $PythonCmd -m pip install --quiet --upgrade pip

# --------------------------------------------------------------------------- #
# Hardware detection                                                            #
# --------------------------------------------------------------------------- #
Write-Header "Detecting hardware..."
$HasCuda = $false
$HasROCm = $false

if (-not $CpuOnly) {
    # NVIDIA
    try {
        $smi = & nvidia-smi --query-gpu=name --format=csv,noheader 2>&1
        if ($LASTEXITCODE -eq 0 -and $smi) {
            $HasCuda = $true
            Write-Info "NVIDIA GPU detected: $($smi -join ', ')"
        }
    } catch { }

    # AMD ROCm (Windows preview)
    try {
        $rocm = & rocminfo 2>&1
        if ($LASTEXITCODE -eq 0) {
            $HasROCm = $true
            Write-Info "AMD ROCm detected"
        }
    } catch { }
}

# --------------------------------------------------------------------------- #
# Install PyTorch                                                               #
# --------------------------------------------------------------------------- #
Write-Header "Installing PyTorch..."

if ($CpuOnly) {
    & $PipCmd install --quiet torch --index-url https://download.pytorch.org/whl/cpu
    Write-Success "PyTorch installed (CPU-only)"
} elseif ($Cuda -or $HasCuda) {
    & $PipCmd install --quiet torch --index-url https://download.pytorch.org/whl/cu124
    Write-Success "PyTorch installed (CUDA 12.4)"
} elseif ($HasROCm) {
    & $PipCmd install --quiet torch --index-url https://download.pytorch.org/whl/rocm6.1
    Write-Success "PyTorch installed (ROCm 6.1)"
} else {
    & $PipCmd install --quiet torch
    Write-Success "PyTorch installed (CPU default)"
}

# --------------------------------------------------------------------------- #
# Install Aether                                                                #
# --------------------------------------------------------------------------- #
Write-Header "Installing Aether Runtime..."

$Extras = ""
if ($Cuda -or $HasCuda) { $Extras = "[cuda]" }
elseif ($HasROCm)        { $Extras = "[rocm]" }

Write-Info "Installing from: $RepoRoot"
& $PipCmd install --quiet -e "${RepoRoot}${Extras}"
Write-Success "Aether Runtime installed"

if ($Dev) {
    Write-Header "Installing development dependencies..."
    & $PipCmd install --quiet -e "${RepoRoot}[dev]"
    Write-Success "Development dependencies installed"
}

# --------------------------------------------------------------------------- #
# Post-install verification                                                     #
# --------------------------------------------------------------------------- #
Write-Header "Verifying installation..."

try {
    $ver = & $PythonCmd -c "import aether; print(aether.__version__)" 2>&1
    Write-Success "aether $ver importable"
} catch {
    Write-Warn "aether import failed — check above for errors"
}

try {
    & $PythonCmd -c "from aether import Runtime, Compiler; print('OK')" 2>&1 | Out-Null
    Write-Success "Core API (Runtime, Compiler) importable"
} catch {
    Write-Warn "Core API import failed"
}

$AetherCli = Join-Path $VenvPath "Scripts\aether.exe"
if (Test-Path $AetherCli) {
    Write-Success "aether CLI available at: $AetherCli"
} else {
    Write-Warn "aether CLI not found — ensure $VenvDir\Scripts is on PATH"
}

# --------------------------------------------------------------------------- #
# Environment check                                                             #
# --------------------------------------------------------------------------- #
Write-Header "Running environment check..."
try {
    $env:PYTHONIOENCODING = "utf-8"
    & $PythonCmd (Join-Path $RepoRoot "scripts\check_env.py")
} catch {
    Write-Info "Run 'python scripts\check_env.py' manually for details"
}

# --------------------------------------------------------------------------- #
# Done                                                                          #
# --------------------------------------------------------------------------- #
Write-Host ""
Write-Host "✓ Aether Runtime installed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "  $VenvDir\Scripts\Activate.ps1         # Activate environment"
Write-Host "  aether --version                        # Verify CLI"
Write-Host "  aether compile <model_path>             # Compile a model"
Write-Host "  aether serve <model.aeg>                # Serve a compiled model"
Write-Host "  python scripts\ci_smoke_test.py         # Run smoke tests"
Write-Host ""
