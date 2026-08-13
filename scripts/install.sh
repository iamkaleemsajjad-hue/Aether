#!/usr/bin/env bash
# =============================================================================
# Aether Runtime — One-Click Install Script (Linux / macOS)
# =============================================================================
#
# Usage:
#   ./scripts/install.sh               # Install with auto-detected hardware
#   ./scripts/install.sh --dev         # Install development dependencies too
#   ./scripts/install.sh --cuda        # Force CUDA extras
#   ./scripts/install.sh --cpu-only    # CPU-only (no GPU extras)
#   ./scripts/install.sh --no-venv     # Skip virtual environment creation
#
# Requirements:
#   - Python 3.10+
#   - pip 23+
#   - (Optional) NVIDIA GPU + CUDA 11.8+ for GPU acceleration
#   - (Optional) Apple Silicon for MLX support
#
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------- #
# Colour output helpers                                                         #
# --------------------------------------------------------------------------- #
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERR]${NC}   $*"; exit 1; }
header()  { echo -e "\n${BOLD}$*${NC}"; }

# --------------------------------------------------------------------------- #
# Argument parsing                                                              #
# --------------------------------------------------------------------------- #
DEV=false
FORCE_CUDA=false
CPU_ONLY=false
NO_VENV=false
VENV_DIR=".venv"

for arg in "$@"; do
  case "$arg" in
    --dev)       DEV=true ;;
    --cuda)      FORCE_CUDA=true ;;
    --cpu-only)  CPU_ONLY=true ;;
    --no-venv)   NO_VENV=true ;;
    --venv-dir=*) VENV_DIR="${arg#*=}" ;;
    -h|--help)
      echo "Usage: $0 [--dev] [--cuda] [--cpu-only] [--no-venv] [--venv-dir=PATH]"
      exit 0 ;;
    *) warn "Unknown argument: $arg" ;;
  esac
done

# --------------------------------------------------------------------------- #
# Prerequisites                                                                 #
# --------------------------------------------------------------------------- #
header "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
header "  Aether Runtime — Installer"
header "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check Python version
PYTHON=$(command -v python3 || command -v python || echo "")
[[ -z "$PYTHON" ]] && error "Python not found. Install Python 3.10+."

PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")
[[ "$PY_MAJOR" -lt 3 || ("$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10) ]] && \
  error "Python 3.10+ required (found $PY_VER)"

success "Python $PY_VER found at $PYTHON"

# --------------------------------------------------------------------------- #
# Virtual environment                                                           #
# --------------------------------------------------------------------------- #
if [[ "$NO_VENV" == false ]]; then
  header "Creating virtual environment: $VENV_DIR"
  if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON" -m venv "$VENV_DIR"
    success "Virtual environment created"
  else
    info "Virtual environment already exists — reusing"
  fi
  # Activate
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  PYTHON="$VENV_DIR/bin/python"
  PIP="$VENV_DIR/bin/pip"
else
  PIP=$(command -v pip3 || command -v pip)
fi

# Upgrade pip
info "Upgrading pip..."
"$PYTHON" -m pip install --quiet --upgrade pip

# --------------------------------------------------------------------------- #
# Hardware detection                                                            #
# --------------------------------------------------------------------------- #
header "Detecting hardware..."
HAS_CUDA=false
HAS_ROCM=false
HAS_MPS=false
TORCH_EXTRA=""

if [[ "$CPU_ONLY" == false ]]; then
  # CUDA
  if command -v nvidia-smi &>/dev/null; then
    CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "")
    if [[ -n "$CUDA_VER" ]]; then
      HAS_CUDA=true
      info "NVIDIA GPU detected (driver: $CUDA_VER)"
    fi
  fi

  # ROCm
  if command -v rocm-smi &>/dev/null; then
    HAS_ROCM=true
    info "AMD GPU detected (ROCm)"
  fi

  # Apple MPS
  if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    HAS_MPS=true
    info "Apple Silicon detected (MPS)"
  fi
fi

# --------------------------------------------------------------------------- #
# Install PyTorch                                                               #
# --------------------------------------------------------------------------- #
header "Installing PyTorch..."

if [[ "$CPU_ONLY" == true ]]; then
  "$PIP" install --quiet torch --index-url https://download.pytorch.org/whl/cpu
  success "PyTorch installed (CPU-only)"
elif [[ "$FORCE_CUDA" == true || "$HAS_CUDA" == true ]]; then
  "$PIP" install --quiet torch --index-url https://download.pytorch.org/whl/cu124
  success "PyTorch installed (CUDA 12.4)"
elif [[ "$HAS_ROCM" == true ]]; then
  "$PIP" install --quiet torch --index-url https://download.pytorch.org/whl/rocm6.1
  success "PyTorch installed (ROCm 6.1)"
else
  "$PIP" install --quiet torch
  success "PyTorch installed (default)"
fi

# --------------------------------------------------------------------------- #
# Install Aether                                                                #
# --------------------------------------------------------------------------- #
header "Installing Aether Runtime..."

# Determine extras
EXTRAS="[]"
if [[ "$HAS_CUDA" == true || "$FORCE_CUDA" == true ]]; then
  EXTRAS="[cuda]"
elif [[ "$HAS_ROCM" == true ]]; then
  EXTRAS="[rocm]"
elif [[ "$HAS_MPS" == true ]]; then
  EXTRAS="[apple]"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

info "Installing from: $REPO_ROOT"
"$PIP" install --quiet -e "${REPO_ROOT}${EXTRAS}"
success "Aether Runtime installed"

# --------------------------------------------------------------------------- #
# Dev dependencies                                                              #
# --------------------------------------------------------------------------- #
if [[ "$DEV" == true ]]; then
  header "Installing development dependencies..."
  "$PIP" install --quiet -e "${REPO_ROOT}[dev]"
  success "Development dependencies installed"
fi

# --------------------------------------------------------------------------- #
# Post-install verification                                                     #
# --------------------------------------------------------------------------- #
header "Verifying installation..."

"$PYTHON" -c "import aether; print(f'  aether version: {aether.__version__}')" 2>/dev/null && \
  success "aether importable" || warn "aether import failed — check above for errors"

"$PYTHON" -c "from aether import Runtime, Compiler; print('  Runtime and Compiler: OK')" 2>/dev/null && \
  success "Core API importable" || warn "Core API import failed"

command -v aether &>/dev/null && success "aether CLI available at: $(command -v aether)" || \
  warn "aether CLI not on PATH — run: source $VENV_DIR/bin/activate"

# --------------------------------------------------------------------------- #
# Environment check                                                             #
# --------------------------------------------------------------------------- #
header "Running environment check..."
"$PYTHON" scripts/check_env.py 2>/dev/null || info "Run 'python scripts/check_env.py' manually for details"

# --------------------------------------------------------------------------- #
# Done                                                                          #
# --------------------------------------------------------------------------- #
echo ""
echo -e "${GREEN}${BOLD}✓ Aether Runtime installed successfully!${NC}"
echo ""
echo "Next steps:"
echo "  source $VENV_DIR/bin/activate     # Activate environment"
echo "  aether --version                   # Verify CLI"
echo "  aether compile <model_path>        # Compile a model"
echo "  aether serve <model.aeg>           # Serve a compiled model"
echo "  python scripts/ci_smoke_test.py    # Run smoke tests"
echo ""
