# Training Log - Pokemon TCG Pocket RL Agent

This document tracks the training progress, experiments, and key findings for the RL agent.

---

## Run #1: Initial Attention Model Training

**Date**: 2026-01-05  
**Model**: Attention-based policy (CardAttentionExtractor)  
**Total Steps**: 17,039,360 (~17M)  
**Training Mode**: Curriculum-based (vs e2 fixed opponent)  
**Status**: [PLATEAU] Plateaued at 47% win rate

### Configuration

```python
# Model Architecture
attention_embed_dim = 256
attention_num_heads = 4
attention_num_layers = 2
net_arch = dict(pi=[256, 128], vf=[256, 128])
# Architecture: POST-NORM (problematic)
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
opponent_type = "e2"  # Curriculum vs e2
observation_size = 2849  # FULL state (board + hand + deck + discard)
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

## Run #2: Large Attention Model Training

**Date**: 2026-01-05 to 2026-01-06  
**Model**: Attention-based policy (embed_dim=384)  
**Total Steps**: ~22,400,000 (~22M)  
**Status**: [COMPLETED] Identified critical gradient issues

### Configuration

```python
# Model Architecture
attention_embed_dim = 384      # Increased from 256
attention_num_heads = 8        # Increased from 4
attention_num_layers = 3       # Increased from 2
net_arch = dict(pi=[256, 128], vf=[256, 128])
# Architecture: POST-NORM (problematic!)
# Estimated size: ~6-7M parameters

# PPO Hyperparameters
learning_rate = 5e-5 → 1e-5    # Linear decay
batch_size = 2048
n_steps = 8192
n_epochs = 8
vf_coef = 0.8
max_grad_norm = 0.5  # Too low for attention
```

### Training Metrics (at 22M steps)

| Metric | Value | Status |
|--------|-------|--------|
| Win Rate (eval) | 56.6% | Below MLP baseline (72%) |
| Peak Elo | 1746 (16M) | Then declined to 1714 |
| Explained Variance | 0.63 | Healthy |
| Training Win Rate | 51.5% | Progressing |

### Critical Issues Found

**Diagnostic Analysis** (16M checkpoint):

1. **Gradient Imbalance**: Ratio 7 billion (card_embed: 11.4 vs attention_pool: 0.006)
2. **Vanishing Gradients**: K-projection biases all zero
3. **Post-Norm Architecture**: Root cause of gradient issues
4. **Dead Features**: 26.6% (normal for limited deck variety)

### Performance vs MLP (DIFFERENT TRAINING MODES)

**IMPORTANT**: These models used **different training modes**!

- **MLP (Run #1)**: 72% overall WR @ 22M steps
  - **Mode**: Pure self-play (vs self)
  - **Observation**: Board-only
  - **Evaluation**: Against **all baselines** (random, e2, e3, e4)
  - **Note**: 72% is overall WR, not per-opponent
  
- **Attention (Run #2)**: 56.6% overall WR @ 22M steps
  - **Mode**: Curriculum (vs e2 fixed opponent)
  - **Observation**: Full state (2849 dims)
  - **Evaluation**: Against **all baselines** (random, e2, e3, e4)
  - **Peak**: 1746 Elo (close to e3's Elo)

**Gap**: -15.4% overall WR

**Analysis**:
1. MLP (self-play) generalized well across all opponents
2. Attention (vs e2) struggled despite richer observation
3. Both models reached similar Elo range (~1700-1750)
4. **Gradient issues likely prevented attention from fully utilizing richer observation**

**Conclusion**: 
- Gradient issues are the primary bottleneck
- Attention has potential (reached ~1750 Elo despite issues)
- Fixing gradients should unlock significant improvement

### Fixes Applied → Run #4

- Switch to pre-norm architecture
- Increase gradient clipping (0.5 → 1.0)
- Add LR warmup (300 steps) + cosine annealing
- Increase base LR (5e-5 → 1e-4)

---

## Run #3: MLP Large Model Training

**Date**: 2026-01-06 (ongoing)  
**Model**: Pure MLP (no attention)  
**Config**: `configs/mlp_large.yaml`  
**Status**: [PLATEAU]

### Configuration

```python
# Model Architecture (Pure MLP)
use_attention = False
net_arch = dict(
    pi=[2048, 1024, 512, 256],  # 4-layer policy
    vf=[2048, 1024, 512]         # 3-layer value
)
# Estimated size: ~8.6M parameters

# PPO Hyperparameters
learning_rate = 1e-4 → 1e-5
n_epochs = 10
ent_coef = 0.03
vf_coef = 0.8
```

### Objectives

1. Test if pure MLP can match attention with same capacity
2. Evaluate full observation (2849 dims) benefit for MLP
3. Establish baseline for Run #4 (fixed attention model)

### Training Metrics (Final: 14.4M steps)

| Checkpoint | Elo | Δ Elo | Win Rate | Notes |
|------------|-----|-------|----------|-------|
| 1.6M | 1679 | - | 48.2% | Initial |
| 3.2M | 1697 | +18 | 53.4% | Learning |
| 4.8M | 1696 | -1 | 53.2% | Stable |
| **6.4M** | **1758** | **+62** | **57.4%** | **First peak** |
| 8.0M | 1734 | -24 | 57.0% | Dip |
| 9.6M | 1689 | -45 | 57.0% | Low point |
| 11.2M | 1756 | +67 | 57.6% | Recovery |
| **12.8M** | **1760** | **+4** | **57.6%** | **Best checkpoint** |
| 14.4M | 1758 | -2 | 56.0% | Stable |

**Best Performance**: 1760 Elo @ 12.8M steps

### Results Summary

**Performance**:
- ✅ Reached 1760 Elo
- ✅ Very stable (1758-1760 from 11M-14M)
- ⚠️ Plateaued after 6.4M steps
- ⚠️ No improvement 6M → 14M (8M wasted steps)
- ⚠️ Below e2 (1893 Elo) by 133 Elo
- ⚠️ Below e3 (1978 Elo) by 218 Elo

**Training Stability**:
- Explained variance: 0.569 (CV = 1.5%) ✅ Excellent
- Entropy: -0.48 (CV = 0.6%) ✅ Very stable
- Episode reward: -0.56 (CV = 25.7%) ⚠️ High variance (but misleading metric)

### Conclusions & Disappointments

**What Worked**:
- ✅ MLP architecture scales well (8.6M params)
- ✅ Training very stable (low variance on key metrics)
- ✅ Converged efficiently (peak @ 6.4M steps)

**What Disappointed**:
- ❌ **No improvement over Run #1** (1750 → 1760 Elo, only +10!)
- ❌ **Full observation didn't help** (2849 dims vs board-only)
- ❌ **More capacity didn't help** (8.6M vs 2-3M params)
- ❌ **Early plateau** (6M steps, then flat for 8M steps)

### Open Questions

**1. Is the observation space the problem?**
- **Historical success**: Previous runs with ~400 dim obs → **1935 Elo** 🎯
- **Current runs**: 2849 dim obs → **1760 Elo max** ❌
- **Hypothesis**: Full state (2849 dims) adds too much noise
- **Evidence**: Run #3 (full obs) barely beat Run #1 (also full obs)
- **Action**: Test with simplified observation (~400 dims like historical runs)

**2. Is pure self-play better than curriculum?**
- **Historical success**: Self-play + frozen opponent → **1935 Elo** 🎯
- **Run #1**: Curriculum vs e2 → 1750 Elo (estimated)
- **Run #3**: Curriculum vs e2 → 1760 Elo
- **Hypothesis**: Self-play provides more diverse training signal
- **Action**: Return to self-play methodology

**3. Have we hit a skill ceiling?**
- **Historical**: 1935 Elo achieved (between e2 and e3)
- **Current**: 1760 Elo max (below e2 by 133 Elo)
- **Difference**: -175 Elo regression!
- **Gap to e2**: 133 Elo (1893 - 1760)
- **Gap to e3**: 218 Elo (1978 - 1760)
- **Likely causes**: 
  - Observation space too complex (2849 vs 400)
  - Training methodology changed (curriculum vs self-play)
  - Architecture issues (post-norm attention)

**4. What made the historical runs successful?**
- ✅ MLP architecture (simpler)
- ✅ Self-play with frozen opponent
- ✅ Smaller observation space (~400 dims)
- ✅ Focused, clean signal
- **Conclusion**: Simpler is better!

### Critical Insight

**We regressed by 175 Elo** (1935 → 1760) by:
- ❌ Expanding observation space (400 → 2849 dims)
- ❌ Switching to curriculum training
- ❌ Using attention with post-norm

**Path forward**:
1. Return to ~400 dim observation space
2. Return to self-play + frozen opponent
3. Test if attention (with pre-norm fixes) can beat historical MLP

### Next Steps → Run #4

**Approach**: Test attention with gradient fixes
- Pre-norm architecture
- LR warmup + cosine annealing
- Conservative hyperparameters

**Success Criteria**:
- Elo > 1780 @ 15M steps (beat MLP baseline)
- If fails: Reconsider observation space and training methodology

**Status**: Ready to launch with `large_model_2.yaml`

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
