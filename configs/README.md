# Training Configurations

This directory contains training configuration files for different experimental setups.

## Usage

```bash
# Train with a specific config
python scripts/train.py --config configs/large_model.yaml

# Override specific parameters
python scripts/train.py --config configs/large_model.yaml --lr 1e-4 --steps 50000000
```

## Available Configs

- `baseline.yaml` - Original small model (256 dim, 4 heads, 2 layers) ~2-3M params
- `large_model.yaml` - Large model (512 dim, 8 heads, 3 layers) ~8-10M params
- `medium_model.yaml` - Medium model (384 dim, 8 heads, 3 layers) ~5-6M params

## Config Structure

```yaml
# Model architecture
model:
  attention_embed_dim: 512
  attention_num_heads: 8
  attention_num_layers: 3
  use_attention: true
  use_silu: true

# PPO hyperparameters
ppo:
  learning_rate: 5.0e-5
  min_learning_rate: 1.0e-5
  batch_size: 2048
  n_steps: 8192
  n_epochs: 8
  gamma: 0.98
  gae_lambda: 0.95
  ent_coef: 0.02
  vf_coef: 0.5
  clip_range: 0.2
  target_kl: 0.025

# Training setup
training:
  total_timesteps: 30000000
  checkpoint_freq: 100000
  n_envs: 32
  use_batched_env: true
  device: "auto"

# Paths
paths:
  simple_deck_path: "simple_deck.json"
  meta_deck_path: "meta_deck.json"
  save_path: "models/rl_bot"
  checkpoint_dir: "./checkpoints/"
  tensorboard_dir: "./logs/"
  resume_path: null  # Set to checkpoint path to resume
```
