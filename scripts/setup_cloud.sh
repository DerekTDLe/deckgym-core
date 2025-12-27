#!/bin/bash
# Setup script for Vast.ai training
# Usage: bash scripts/setup_cloud.sh

set -e

echo "=== DeckGym Cloud Training Setup ==="

# 1. Install Rust
if ! command -v cargo &> /dev/null; then
    echo "[1/5] Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ~/.cargo/env
else
    echo "[1/5] Rust already installed"
fi

# 2. Create and activate venv
echo "[2/5] Setting up Python environment..."
python3 -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies
echo "[3/5] Installing Python dependencies..."
pip install --upgrade pip
pip install maturin
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 4. Build Rust library
echo "[4/5] Building Rust library..."
maturin develop --release

# 5. Verify setup
echo "[5/5] Verifying setup..."
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
echo "To start training (recommended for Vast.ai with 16+ cores):"
echo "  python python/scripts/train.py --steps 30000000 --n-envs 24"
echo ""
echo "To monitor with TensorBoard:"
echo "  tensorboard --logdir tensorboard_logs --bind_all"
