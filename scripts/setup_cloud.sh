#!/bin/bash
# Setup script for Vast.ai training (PyTorch template)
# Usage: bash scripts/setup_cloud.sh
#
# This script is designed for Vast.ai's PyTorch template which provides:
# - /venv/main/ with PyTorch pre-installed
# - CUDA and cuDNN ready
# - uv package manager

set -e

echo "=== DeckGym Cloud Training Setup ==="

# 1. Install Rust
if ! command -v cargo &> /dev/null; then
    echo "[1/4] Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ~/.cargo/env
else
    echo "[1/4] Rust already installed"
    source ~/.cargo/env 2>/dev/null || true
fi

# 2. Use existing venv (Vast.ai template provides /venv/main/)
echo "[2/4] Using existing Python environment..."
if [ -d "/venv/main" ]; then
    source /venv/main/bin/activate
    echo "  Using Vast.ai venv: /venv/main/"
else
    # Fallback: create local venv if not on Vast.ai
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    echo "  Created local venv: .venv/"
fi

# 3. Install Python dependencies
echo "[3/4] Installing Python dependencies..."
pip install maturin
pip install -r requirements.txt

# 4. Build Rust library
echo "[4/4] Building Rust library..."
maturin develop --release

# Verify setup
echo ""
echo "=== Verifying Setup ==="
python -c "
import deckgym
import torch
print(f'✓ deckgym: observation_size = {deckgym.Game.observation_size()}')
print(f'✓ PyTorch: CUDA available = {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To start training:"
echo "  python python/scripts/train.py --steps 30000000 --n-envs 24"
echo ""
echo "TensorBoard is already running on the template (monitoring /workspace)"
