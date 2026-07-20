#!/usr/bin/env bash
# Setup isolated conda env on Linux training server (RTX 3090).
# Safe: only creates/updates env name "nighttime-tsd" — never modifies other envs.
#
# Usage:
#   bash scripts/setup_conda_env.sh
#   bash scripts/setup_conda_env.sh --with-torch-cu124
#   bash scripts/setup_conda_env.sh --install-htop   # needs sudo for apt
set -euo pipefail

ENV_NAME="nighttime-tsd"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WITH_TORCH=0
INSTALL_HTOP=0
for arg in "$@"; do
  case "$arg" in
    --with-torch-cu124) WITH_TORCH=1 ;;
    --install-htop) INSTALL_HTOP=1 ;;
    -h|--help)
      echo "Usage: $0 [--with-torch-cu124] [--install-htop]"
      exit 0
      ;;
  esac
done

echo "=== 1) Locate conda ==="
if ! command -v conda >/dev/null 2>&1; then
  # common install locations
  for c in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh" \
    "$HOME/mambaforge/etc/profile.d/conda.sh" \
    "$HOME/miniforge3/etc/profile.d/conda.sh"
  do
    if [[ -f "$c" ]]; then
      # shellcheck source=/dev/null
      source "$c"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found."
  echo "Install Miniconda (user-local, no sudo):"
  echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh"
  echo "  bash /tmp/miniconda.sh -b -p \$HOME/miniconda3"
  echo "  source \$HOME/miniconda3/etc/profile.d/conda.sh"
  echo "  conda init bash"
  exit 1
fi

echo "conda: $(command -v conda)"
conda --version
echo
echo "=== Existing envs (will NOT modify other AI envs) ==="
conda env list
echo

echo "=== 2) Create/update env: ${ENV_NAME} ==="
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Env exists → updating from environment.yml"
  conda env update -n "$ENV_NAME" -f environment.yml --prune
else
  echo "Creating new env from environment.yml"
  conda env create -f environment.yml
fi

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo
echo "=== 3) nvitop (conda or pip) ==="
if ! command -v nvitop >/dev/null 2>&1; then
  pip install -U nvitop
fi
nvitop --version || true

if [[ "$WITH_TORCH" -eq 1 ]]; then
  echo
  echo "=== 4) PyTorch CUDA 12.4 wheels (3090-friendly) ==="
  pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu124
fi

echo
echo "=== 5) Project pip deps (idempotent) ==="
pip install -U -r requirements.txt || pip install -U ultralytics gdown

if [[ "$INSTALL_HTOP" -eq 1 ]]; then
  echo
  echo "=== 6) htop (system package — shared, not inside conda) ==="
  if command -v htop >/dev/null 2>&1; then
    echo "htop already installed: $(command -v htop)"
  elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y htop
  elif command -v yum >/dev/null 2>&1; then
    sudo yum install -y htop
  else
    echo "WARN: cannot install htop automatically. Try: sudo apt install htop"
  fi
fi

echo
echo "=== 7) Verify isolation + GPU ==="
python - <<'PY'
import sys
print("python :", sys.executable)
print("version:", sys.version.split()[0])
try:
    import torch
    print("torch  :", torch.__version__, "| cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu    :", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch  : not ready —", e)
try:
    import ultralytics
    print("ultralytics:", ultralytics.__version__)
except Exception as e:
    print("ultralytics:", e)
PY

echo
echo "=== DONE ==="
echo "Activate ONLY this env (other AI envs untouched):"
echo "  conda activate ${ENV_NAME}"
echo "Monitors:"
echo "  nvitop          # GPU processes (inside env)"
echo "  htop            # CPU/RAM (system)"
echo "Train:"
echo "  python scripts/load_model.py --model yolo11n.pt --info"
echo "  python scripts/train_baseline.py --model yolo11n.pt --batch 64 --epochs 100"
