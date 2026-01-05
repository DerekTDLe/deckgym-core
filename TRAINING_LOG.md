# Training Log - Pokemon TCG Pocket RL Agent

This document tracks the training progress, experiments, and key findings for the RL agent.

---

## Run #1: Initial Attention Model Training

**Date**: 2026-01-05  
**Model**: Attention-based policy (CardAttentionExtractor)  
**Total Steps**: 17,039,360 (~17M)  
**Status**: [PLATEAU] Plateaued at 47% win rate

### Configuration

```python
# Model Architecture
attention_embed_dim = 256
attention_num_heads = 4
attention_num_layers = 2
net_arch = dict(pi=[256, 128], vf=[256, 128])
# Estimated size: ~2-3M parameters

# PPO Hyperparameters
learning_rate = 5e-5 → 1e-5  # Linear decay
batch_size = 2048
n_steps = 8192
n_epochs = 8
gamma = 0.98
gae_lambda = 0.95
ent_coef = 0.02
vf_coef = 0.5
clip_range = 0.2
target_kl = 0.025

# Environment
n_envs = 32
use_batched_env = True  # Rust-side batching
opponent_type = "e2"  # Expectiminimax depth 2
```

### Training Metrics (at 17M steps)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Win Rate | 36.8% ± 14.3% | 55% | [BELOW] Below target |
| **Peak Win Rate** | **52.5%** (at 11.2M steps) | 55% | [PEAK] Regressed |
| Explained Variance | 0.57 ± 0.11 | > 0.8 | [LOW] Moderate (see notes) |
| Entropy Loss | -0.46 ± 0.14 | N/A | [OK] Healthy |
| Value Loss | 0.20 ± 0.05 | < 0.15 | [OK] Moderate |
| Approx KL | 0.009 ± 0.003 | < 0.025 | [OK] Good |

### Key Observations

1. **Progression Phase (0-4M steps)**
   - Win rate climbed from 0% to ~47%
   - Model learning basic game mechanics successfully

2. **Plateau Phase (4M-17M steps)**
   - Win rate oscillating around 47%
   - No significant improvement despite 13M additional steps

3. **Peak Performance (11.2M steps)**
   - Achieved 52.5% win rate (best performance)
   - Subsequently oscillated back to 47%
   - **Note**: Regression could be due to game variance, opponent deck diversity, or model capacity limits

### Root Cause Analysis

**Hypothesis**: Model capacity may be limiting performance

**Evidence**:
- [OK] Optimization is healthy (entropy, KL, value loss stable)
- [?] Explained variance at 0.57 - could be due to:
  - Intrinsic game randomness (deck draws, coin flips)
  - Model capacity limitations
  - Opponent diversity (different deck matchups)
- [?] Win rate oscillation 52.5% → 47% - variance or capacity?

**Open Questions**:
1. Is 0.57 explained variance acceptable given game randomness?
2. Is the win rate oscillation due to statistical variance or model forgetting?
3. Would a larger model actually help, or are we hitting a skill ceiling vs e2?

**Proposed Experiment**: Moderately increase model size to test capacity hypothesis

### Next Steps

**Plan**: Moderately increase model capacity to ~8-10M parameters
- `embed_dim`: 256 → 512 (conservative increase)
- `num_heads`: 4 → 8 (better attention granularity)
- `num_layers`: 2 → 3 (deeper reasoning)
- `net_arch`: pi=[512, 256, 128], vf=[512, 256, 128]

**Success Criteria**:
- Explained variance > 0.65 (accounting for game randomness)
- Win rate consistently > 52.5% (surpass previous peak)
- If no improvement after 5M steps → capacity is not the bottleneck

---

## Run #2: Large Model Training

**Date**: TBD  
**Model**: Moderately scaled attention model  
**Status**: [PLANNED] Ready to launch

### Configuration

```python
# Model Architecture (UPDATED)
attention_embed_dim = 512      # 256 → 512 (moderate increase)
attention_num_heads = 8        # 4 → 8
attention_num_layers = 3       # 2 → 3
net_arch = dict(pi=[512, 256, 128], vf=[512, 256, 128])
# Estimated size: ~8-10M parameters

# PPO Hyperparameters (UNCHANGED)
learning_rate = 5e-5 → 1e-5    # Keep same schedule
batch_size = 2048              # Keep same (2.7GB/4GB is fine)
n_steps = 8192                 # Keep same
n_epochs = 8
gamma = 0.98
ent_coef = 0.02
target_kl = 0.025

# Environment (unchanged)
n_envs = 32
opponent_type = "e2"
```

### Training Metrics

_To be filled during training..._

### Key Observations

_To be filled during training..._

---

## Curriculum Progress

| Stage | Opponent | Deck Source | Threshold | Status |
|-------|----------|-------------|-----------|--------|
| Warmup | e2 | Simple | 55% | � In Progress (47% WR) |
| Meta | e2 | Meta | 55% | [LOCKED] Locked |
| Advanced | e3 | Meta | 55% | [LOCKED] Locked |
| Mastery | Self-play | Meta | N/A | [LOCKED] Locked |

**Current Stage**: Warmup (vs e2, simple decks)  
**Bottleneck**: Testing if model capacity is limiting performance

---


## Notes & Insights

### 2026-01-05

#### Diagnostic Analysis (17M steps)
- Initial diagnostic completed on Run #1 (17M steps)
- Optimization metrics are healthy (entropy, KL, value loss stable)
- Win rate peaked at 52.5% (11.2M steps), currently oscillating around 47%
- Explained variance at 0.57 - unclear if due to game randomness or model limits
- **Hypothesis**: Model capacity may be limiting, but not certain
- **Decision**: Increase to 512 embed_dim (8-10M params) to test hypothesis
- **Rationale**: Go big to definitively test capacity bottleneck (goal is to beat e3)
- If no improvement after 5M steps → look elsewhere (reward shaping, curriculum, etc.)

#### Configuration System
- Created YAML-based config system for easy experiment management
- Two configs available:
  - `configs/baseline.yaml` - Run #1 settings (256 dim, 2-3M params)
  - `configs/large_model.yaml` - Run #2 settings (512 dim, 8-10M params)
- Added `--config` argument to `train.py` for loading YAML configs
- CLI arguments override YAML values for quick experiments
- Installed PyYAML dependency

---

## Useful Commands

```bash
# Start training with YAML config (RECOMMENDED)
cd python
python scripts/train.py --config ../configs/large_model.yaml

# Start training with baseline config
python scripts/train.py --config ../configs/baseline.yaml

# Override specific parameters
python scripts/train.py --config ../configs/large_model.yaml --lr 1e-4 --steps 50000000

# Legacy: Start training without config file
python scripts/train.py --attention-dim 512 --attention-heads 8 --attention-layers 3

# Monitor training
tensorboard --logdir ./logs

# Evaluate checkpoint
python scripts/evaluate.py eval --model ./checkpoints/rl_bot_17000000_steps.zip --games 100
```

---

## References

- Training script: `scripts/train.py`
- Config files: `configs/` (baseline.yaml, large_model.yaml)
- Curriculum manager: `python/deckgym/curriculum.py`
- Attention policy: `python/deckgym/attention_policy.py`
- TensorBoard logs: `./logs/`
- Checkpoints: `./checkpoints/`
