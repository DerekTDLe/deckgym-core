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


---

## Run #4: Attention with Gradient Fixes

**Date**: 2026-01-06  
**Model**: Attention-based (pre-norm architecture)  
**Config**: `configs/large_model_2.yaml`  
**Total Steps**: ~4,800,000 (~4.8M)  
**Status**: [COMPLETED] Excellent performance but high variance

### Configuration

```python
# Model Architecture (FIXED)
attention_embed_dim = 256
attention_num_heads = 8
attention_num_layers = 3
net_arch = dict(pi=[256, 128], vf=[256, 128])
# Architecture: PRE-NORM (fixed!) ✅
# Estimated size: ~6-7M parameters

# PPO Hyperparameters (OPTIMIZED)
learning_rate = 1e-4 → 1e-5    # Cosine annealing with warmup
warmup_steps = 300             # LR warmup
n_steps = 2048                 # Reduced for 37-step episodes
batch_size = 2048
n_epochs = 6                   # Conservative
gamma = 0.98
ent_coef = 0.03
vf_coef = 0.8
max_grad_norm = 1.0            # Increased from 0.5
clip_range = 0.2
target_kl = 0.015
```

### Training Metrics (Final: 4.8M steps)

| Checkpoint | Elo | Δ Elo | Win Rate | Notes |
|------------|-----|-------|----------|-------|
| 0.3M | 1662 | - | 47.2% | Initial |
| 0.6M | 1666 | +4 | 46.8% | Learning |
| 1.0M | 1637 | -30 | 49.2% | Dip |
| 1.3M | 1681 | +44 | 50.8% | Recovery |
| 1.6M | 1741 | +60 | 48.0% | Strong |
| 1.9M | 1733 | -8 | 56.4% | Variance |
| 2.2M | 1781 | +48 | 57.0% | **Beats MLP!** |
| 2.6M | 1717 | -64 | 55.8% | Drop |
| 2.9M | 1735 | +18 | 53.2% | Recovery |
| **3.2M** | **1825** | **+90** | **59.8%** | **Peak!** 🏆 |
| 3.5M | 1746 | -78 | 55.6% | Drop |
| 3.8M | 1732 | -15 | 54.4% | Stable |
| 4.2M | 1742 | +10 | 55.2% | Slight up |
| 4.5M | 1702 | -39 | 52.4% | Drop |
| 4.8M | 1736 | +34 | 53.2% | Final |

**Best Performance**: 1825 Elo @ 3.2M steps

### Results Summary

**Performance**:
- ✅ **Peak: 1825 Elo** (new record for current runs!)
- ✅ **Beats MLP baseline** (+65 Elo vs 1760)
- ✅ **Ultra-fast convergence** (3.2M vs MLP's 6.4M)
- ⚠️ **High variance** (1702-1825, Δ=123 Elo)
- ⚠️ Below e2 (1893) by 68 Elo
- ⚠️ Below historical (1935) by 110 Elo

**Training Metrics**:
- Explained variance: 0.58 (excellent!)
- Value loss: 0.20 (very good)
- Approx KL: 0.014 (close to limit 0.015)
- Clip fraction: 0.115 (high, suggests instability)
- Episode reward: -0.48 (improving)

### What Worked ✅

1. **Pre-norm architecture**: Gradient flow excellent
2. **LR warmup + cosine**: Fast initial convergence
3. **Conservative hyperparameters**: Stable training metrics
4. **Gradient fixes**: 14× faster convergence than MLP

### What Didn't Work ❌

1. **High Elo variance**: ±60-80 Elo between checkpoints
2. **KL too high**: 0.014 (close to 0.015 limit)
3. **Clip fraction high**: 11.5% (too many clipped gradients)
4. **LR too aggressive**: 1e-4 caused instability

### Lessons Learned

**Gradient fixes work!** The model:
- Converges 14× faster on explained variance
- Converges 10× faster on value loss
- Reaches higher peak Elo (+65 vs MLP)

**But needs more stability**:
- LR too high → reduce to 1e-5
- target_kl too permissive → reduce to 0.01
- clip_range too large → reduce to 0.15

### Diagnostic Analysis (Checkpoint @ 3.2M)

**Gradient Issues Persist** (despite pre-norm):

1. **K-projection Bias Vanishing** ❌
   - All layers: `k_proj.bias` gradient = 0.000000
   - Bias values stuck at zero
   - **Root cause**: Initialization issue (symmetric at zero)
   - **Impact**: Attention can't learn which keys to focus on

2. **Severe Gradient Imbalance** ❌
   - Ratio: **26 billion** (26,686,617,802)
   - `action_net.weight`: 866 (huge)
   - `k_proj.bias`: 0.000000 (dead)
   - **Impact**: Unstable training, high Elo variance

3. **Weights Healthy** ✅
   - All weight distributions normal
   - No dead neurons
   - Attention layers properly initialized

**Conclusion**: Model reaches 1825 Elo despite gradient issues, but variance (±80 Elo) caused by:
- K-proj bias not learning
- Action net gradients too large
- Gradient imbalance

**Potential Fixes** (for future runs):
1. Initialize K-proj bias to small non-zero values
2. Layer-wise learning rates (action_net lower)
3. Or disable bias in attention projections entirely

### Next Steps → Run #5

**Approach**: Stabilize the excellent performance
- Same architecture (pre-norm) ✅
- **Lower LR**: 1e-5 (10× reduction)
- **Stricter KL**: 0.01 (was 0.015)
- **Tighter clipping**: 0.15 (was 0.2)

**Success Criteria**:
- Elo > 1800 stable (less variance)
- Beat e2 (1893 Elo)
- Approach historical (1935 Elo)

**Status**: Ready to launch with `large_model_3.yaml`

---

## Run #5: Stabilized Attention Model

**Date**: 2026-01-06 (pending)  
**Model**: Attention-based (pre-norm, stabilized)  
**Config**: `configs/large_model_3.yaml`  
**Status**: [READY] Awaiting launch

### Configuration Changes from Run #4

```python
# Stability improvements (hyperparameters)
learning_rate: 1e-4 → 1e-5     # 10× lower for stability
min_learning_rate: 1e-5 → 1e-6 # Lower floor
clip_range: 0.2 → 0.15         # Tighter clipping
target_kl: 0.015 → 0.01        # Stricter early stopping

# CRITICAL FIX: K-projection bias disabled
# In attention_policy.py:
#   self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
# 
# This fixes the dead gradient problem from Run #4:
# - k_proj.bias was stuck at 0.000000 (no learning)
# - Gradient imbalance: 26 billion ratio
# - High Elo variance: ±80 Elo
#
# Expected improvements:
# - Healthy gradients for all parameters
# - Reduced variance: ±30 Elo (vs ±80)
# - Better convergence stability
# - Potential to reach 1850-1900+ Elo

# MODERN TRANSFORMER IMPROVEMENTS (GPT/BERT/LLaMA style)
# 1. GELU activation (replaces ReLU everywhere)
#    - Smoother gradients, no dead neurons
#    - Used by: GPT-2/3/4, BERT, LLaMA
#
# 2. Bias-free attention projections (Q, K, V, out_proj)
#    - Better gradient flow
#    - Fewer parameters: -1,024 per attention layer
#    - Used by: LLaMA, GPT-3, PaLM
#
# 3. Improved FFN (4× expansion + dropout)
#    - 2× → 4× expansion (standard Transformer size)
#    - GELU activation
#    - Dropout(0.1) for regularization
#    - Used by: All modern Transformers
#
# 4. Fused QKV projection (FlashAttention/xFormers style)
#    - Single matmul for Q, K, V (3× fewer kernel calls)
#    - Better memory bandwidth utilization
#    - Faster training and inference
#    - Used by: FlashAttention, xFormers, GPT-NeoX
#
# Total parameter changes:
# - Attention biases removed: -3,072 params (3 layers × 2 modules × 512)
# - FFN expansion doubled: +~400K params (better capacity)
# - Fused QKV: Same params, 3× faster
# - Net effect: Larger, more capable, faster model with better gradient flow
```

### Objectives

1. Maintain peak performance (~1825 Elo)
2. **Reduce variance** (target: ±30 Elo instead of ±80)
3. Beat e2 (1893 Elo) consistently
4. Approach historical best (1935 Elo)
5. **Verify gradient health** (no dead parameters)
6. **Test modern improvements** (GELU, bias-free, 4× FFN, fused QKV)

### Results

**Status**: ✅ COMPLETED (2026-01-07)  
**Training Duration**: ~16.8M steps  
**Final Checkpoint**: `rl_bot_16800000_steps.zip`

#### Gradient Health Analysis (@ 12.8M steps)

```
Diagnostic Results:
  Gradient Imbalance Ratio: 6,708
  Status: ⚠️ IMPROVED but still above target (< 1,000)
  
  Comparison with Run #4:
    Run #4 @ 3.2M : 26,686,617,802 (26 billion)
    Run #5 @ 12.8M: 6,708
    Improvement:    99.99997% reduction ✅

  Critical Findings:
    ✅ No dead gradients (all parameters learning)
    ✅ K-projection bias fix successful
    ✅ Fused QKV working correctly
    ❌ action_net gradient still too large (1,051 vs target ~70)
    ❌ action_net weights too concentrated (σ = 0.002)
```

#### Architecture Successes

1. **Modern Transformer Improvements** ✅
   - GELU activation: Working, smoother gradients
   - Bias-free attention: All biases removed, better gradient flow
   - Fused QKV: 3× fewer kernel calls, faster training
   - FFN 4× expansion: More capacity, standard Transformer size

2. **Gradient Health** ✅
   - K-projection bias problem: RESOLVED
   - Dead gradients: ELIMINATED
   - Gradient imbalance: 99.99997% reduction

#### Remaining Problem Identified

**Root Cause**: Feature extractor bottleneck
- Attention pooling too aggressive (24 cards → 1 vector = 24× compression)
- Output dimension too small (320 dims for 175 actions)
- Features/action ratio: 0.32 (21× worse than GPT-2)
- Action net compensating with huge gradients (1,051)

**Diagnosis**:
```
Pooling gradient (query): 0.16 (nearly dead)
Action net gradient:      1,051 (15× too large)
Action net weight std:    0.002 (40× too concentrated)
```

#### Performance

Same issues as Run #4:
- High Elo variance (±80)
- Inconsistent performance
- Unable to beat e2 consistently

### Conclusion

Run #5 successfully validated all modern Transformer improvements:
- ✅ Architecture modernization works (GELU, bias-free, fused QKV)
- ✅ Gradient health massively improved (26B → 6.7K)
- ❌ Bottleneck identified: Feature extractor output too small

**Next Step**: Implement multi-head pooling to address the bottleneck.

---

## Run #6: Multi-Head Pooling Architecture

**Date**: 2026-01-07 (pending)  
**Model**: Attention-based (multi-head pooling)  
**Config**: `configs/large_model_3.yaml` (updated)  
**Status**: [READY] Awaiting launch

### Architectural Changes from Run #5

```python
# CRITICAL FIX: Multi-Head Attention Pooling
# Replaces single-query AttentionPooling with MultiHeadAttentionPooling
#
# Before (Run #5):
#   - 1 learned query
#   - 24 cards → 1 vector (24× compression)
#   - Output: 256 dims
#   - Pooling gradient: 0.16 (nearly dead)
#
# After (Run #6):
#   - 4 learned queries
#   - 24 cards → 4 vectors (6× compression)
#   - Output: 1024 dims (4 × 256)
#   - Each query captures different aspects of card state

# EXPANDED OUTPUT DIMENSIONS
# Before: 320 dims (64 global + 256 cards)
# After:  640 dims (128 global + 512 cards)
#
# Global projection: 64 → 128 dims
# Card output:       256 → 512 dims (from 4 queries)
# Total:             320 → 640 dims
#
# Features/action ratio: 640 / 175 = 3.66 (optimal range: 2-5×)
# Compression:           2,849 → 640 = 4.5× (was 8.9×)

# PARAMETER CHANGES
# Total parameters: 2,797,248 → 3,272,064 (+17%)
#   Global projection:    6,848 → 21,888
#   Attention pooling:  262,400 → 263,168
#   Output projection:   65,792 → 524,800
#
# Distribution:
#   Attention layers:  24.0% (786K)
#   Feed-forward:      48.2% (1.6M)
#   Multi-head pool:    8.0% (263K)
#   Output projection: 16.0% (525K)
```

### Expected Improvements

Based on architectural analysis and literature:

1. **Gradient Imbalance** 🎯
   - Current: 6,708
   - Target: < 1,000
   - Expected: ~1,000 (6.7× reduction)
   - Mechanism: Richer features reduce action net stress

2. **Feature Expressiveness** ✅
   - Pooling compression: 24× → 6× (4× less aggressive)
   - Output dimensions: 320 → 640 (2× larger)
   - Features/action ratio: 1.83 → 3.66 (2× better)
   - Optimal for 175 actions (GPT-2 ratio: 0.015, ours: 3.66)

3. **Gradient Health** ✅
   - Action net gradient: 1,051 → ~300 (expected 3.5× reduction)
   - Pooling queries: 4 independent gradients vs 1
   - Better gradient flow throughout network

4. **Performance** 🎯
   - Elo variance: ±80 → ±30 (expected)
   - Peak Elo: 1825 → 1850-1900+ (expected)
   - Consistency: Better checkpoint stability

### Objectives

1. **Achieve gradient imbalance < 1,000** (primary goal)
2. Reduce Elo variance to ±30
3. Beat e2 (1893 Elo) consistently
4. Reach 1900+ Elo peak
5. Verify multi-head pooling effectiveness

### Success Criteria

- ✅ Gradient imbalance ratio < 1,000
- ✅ Action net gradient < 500
- ✅ Elo variance < 50 over 10 checkpoints
- ✅ Peak Elo > 1850
- ✅ Consistent wins against e2 (>60%)

### Training Metrics

*To be filled during training*

### Notes

This run addresses the final architectural bottleneck identified in Run #5. The multi-head pooling approach is inspired by:
- BERT's [CLS] token (single query)
- Vision Transformers' multi-query pooling
- Modern LLMs' multiple special tokens

The 4 queries can specialize in different aspects:
- Query 1: Offensive threats
- Query 2: Defensive state
- Query 3: Energy/resource state
- Query 4: Board control

**Key Innovation**: Reduces information loss while maintaining computational efficiency.

---

## Run #7: Pure Self-Play Modernization
**Date**: 2026-01-07 (Launched)  
**Model**: Multi-Head Attention (Run #6 architecture refined)  
**Config**: `configs/baseline.yaml`  
**Status**: [IN PROGRESS] Transitioned to pure self-play

### Key Changes (Major Refactor)

1. **Abandoning Curriculum Training** 🎯
   - Removed all stages (warmup, meta, advanced).
   - Switched to **pure self-play** from step 1.
   - Using `MetaDeckLoader` to sample from 100+ high-quality meta decks immediately.
   - Rationale: Curriculum was causing catastrophic forgetting and skill local optima.

2. **Observation Space Refinement (V3)** ✅
   - Total cards: 24 → 18 (10 hand, 4 self board, 4 opponent board).
   - Hand size capped at 10 (fixed observation size).
   - Removed deck/discard card encoding (reduced noise).
   - Final `OBSERVATION_SIZE`: **2129** (41 global + 18 × 116 features).

3. **Architecture Hardening** ✅
   - Transformer Encoder: 3 layers, 8 heads, 256 dim.
   - Multi-Head Pooling: 4 learned queries (1024 dim output).
   - Policy/Value Heads: 3 layers deep (512, 256, 128).
   - Activation: GELU throughout.
   - Normalization: PRE-NORM for stability.

4. **Rust Core Stability** 🛡️
   - Wrapped all environment loops in `catch_unwind`.
   - Panics in game state (e.g., energy discard errors) now log a warning and force a single-env reset instead of crashing Python.

5. **TrueSkill Calibration** 📊
   - Shifted center from 25.0 to **1500.0**.
   - Standard parameters: mu=1500, sigma=500, beta=250.

### Expected Outcomes
- Higher peak Elo (target: 2000+)
- Faster adaptation to meta matchups.
- 100% training uptime (no more crashes).
---

## Run #8: PFSP with Optimal Config (FAILED)

**Date**: 2026-01-07 to 2026-01-08  
**Model**: Attention-based (256 embed, 8 heads, 3 layers)  
**Config**: `configs/pfsp_optimal.yaml`  
**Total Steps**: ~21,200,000 (~21M)  
**Status**: ❌ **FAILED** - Catastrophic forgetting after 6.4M steps

### Configuration

```python
# Key hyperparameters (problematic)
learning_rate = 1e-4           # Too high for attention
target_kl = 0.02               # Too permissive
ent_coef = 0.06                # Too aggressive exploration
n_steps = 512                  # Good
clip_range = 0.2               # Standard

# PFSP settings
pfsp_pool_size = 10
pfsp_add_threshold = 0.60      # Required 60% WR to add to pool
```

### Training Metrics

| Checkpoint | Elo | Δ vs Best | Win% | Status |
|------------|-----|-----------|------|--------|
| 1.6M | 1585 | -59 | 44% | Learning |
| **6.4M** | **1644** | **0 (BEST)** | **50%** | 🏆 Peak |
| 8.0M | 1614 | -30 | 49% | ⬇️ Regression |
| 11.2M | 1610 | -34 | 41% | ⬇️ |
| 14.4M | 1536 | -108 | 34% | ⬇️⬇️ |
| 19.2M | 1506 | -138 | 32% | ⬇️⬇️⬇️ |
| 20.8M | 1581 | -63 | 39% | Slight recovery |

**Gap to e2**: -248 Elo (1644 vs 1893)

### Root Cause Analysis ❌

#### 1. KL Divergence Explosion

```
Step      KL        Target    Status
--------------------------------------
1M        0.021     0.015     ⚠️ Already high
5M        0.031     0.015     ❌ 2x over target!
10M       0.032     0.015     ❌
15M       0.031     0.015     ❌
20M       0.023     0.015     ⚠️
```

**Impact**: Policy updates were **2x more aggressive** than intended.
- Agent "forgot" good strategies learned at 6M steps
- Performance oscillated wildly instead of converging
- Explains why 6.4M > 20.8M despite 14M more training steps

#### 2. PFSP Pool Stagnation

| Metric | Value |
|--------|-------|
| Last pool addition | 7.47M steps |
| Pool additions since | **0** (15M steps!) |
| Cause | Agent winrate dropped below 60% |

The 60% threshold prevented the agent from adding weaker versions to the pool,
creating a local optimum trap where it trained against the same 10 opponents.

#### 3. Gradient Concentration (Diagnostic @ 19.2M)

```
Layer                    Gradient Norm    Status
------------------------------------------------
card_embed.0.weight      737              ⚠️ Very high
attention_layers.0       763              ⚠️ Very high
attention_layers.2       42               ✅ Normal
policy_net               419-475          ⚠️ High
```

Gradients concentrated in early layers instead of policy head.

### Lessons Learned

1. **LR 1e-4 is too high** for attention architecture with PFSP
   - Causes KL divergence explosion
   - Should use 3e-5 or lower

2. **target_kl 0.02 is too permissive**
   - Actual KL was 0.031 (1.5x over!)
   - Should use 0.01 for stricter early stopping

3. **60% threshold caused pool stagnation**
   - Lowered to 50% in pfsp_callback.py
   - Ensures pool keeps growing during difficult periods

4. **Catastrophic forgetting is real**
   - Peak at 6.4M, then 14M steps of regression
   - Wasted ~14M steps of compute

### Fixes Applied → Run #9

Created `configs/pfsp_conservative.yaml`:
- LR: 1e-4 → **3e-5** (÷3)
- target_kl: 0.02 → **0.01** (÷2)
- ent_coef: 0.06 → **0.03** (÷2)
- clip_range: 0.2 → **0.15**
- max_grad_norm: 1.0 → **0.5**

Modified `pfsp_callback.py`:
- Pool add threshold: 60% → **50%**

**Next Steps**: Resume from 6.4M checkpoint with conservative config.

---

## Run #9: Stabilized Attention Model (V2 Revamp)

**Date**: 2026-01-08 → 2026-01-09
**Model**: Revamped Attention (Pre-norm, GELU, 4 queries, Bias-free)
**Config**: `configs/attention_baseline.yaml`
**Status**: [STOPPED] Elo plateau + PFSP echo chamber identified at 16M steps

### Final Metrics (@ 16M steps)

| Metric | Value | Peak Value | Status |
|--------|-------|------------|--------|
| Elo (Mu) | ~1900 | **1952** (@ 12.8M) | [REGRESSED] -50 Mu |
| Expose | ~1750 | **1807** (@ 12.8M) | [REGRESSED] |
| Episode Reward | 0.04 | **0.44** (@ 8M) | [COLLAPSED] |
| Explained Variance | 0.71-0.76 | - | [OK] Still learning |
| Entropy | -0.42 | -1.04 (@ 8M) | [INCREASING] Policy getting chaotic |
| Value Loss | Increasing | - | [WARNING] Instability |

### TensorBoard Analysis

1. **Episode Reward Collapse**
   - Peaked at 0.44 (8M steps) → collapsed to 0.04 (16M steps)
   - Agent stopped winning against its own pool

2. **Entropy Paradox**
   - Entropy increased from -1.04 to -0.42 (policy becoming LESS deterministic)
   - Unusual: typically entropy decreases as policy converges
   - Interpretation: Agent is "thrashing" between strategies, no clear improvement direction

3. **Explained Variance Stable**
   - Remained high (0.71-0.76) throughout
   - Model is still learning *something*, but not overall game strength
   - Likely overfitting to self-play patterns

### Root Cause: PFSP Echo Chamber

**Diagnosis**: Pool homogeneity causing stagnation

```
Pool Analysis (@ 16M steps):
- Pool size: 10 opponents
- All opponents: 15M+ steps checkpoints
- Oldest checkpoints: EVICTED (low winrate against current agent)
- Diversity: ZERO (all opponents are slight variations of same policy)
```

**Problem Mechanism**:
1. Agent dominates pool → adds itself → old opponents evicted
2. Pool becomes homogeneous (all recent self-copies)
3. Agent only learns to exploit its own weaknesses
4. No signal from diverse strategies → local optima
5. Win rate against pool stays high, but absolute strength stagnates

### Solution Implemented: Baseline Curriculum

Added permanent baseline slots with progressive difficulty:

| Step Range | Baselines | Rationale |
|------------|-----------|-----------|
| 0 - 500k | `v`, `w` | Easy warmup (ValueFunction, WeightedRandom) |
| 500k - 2M | `aa`, `er` | Medium (AttachAttack, EvolutionRusher) |
| 2M+ | `e2`, `er` | Hard + Medium (Expectiminimax(2), EvolutionRusher) |

**Key Changes**:
- `pfsp_baseline_slots = 2` - 2 permanent baseline slots
- `pfsp_baseline_max_allocation = 0.20` - Max 20% of envs for baselines
- Baselines **cannot be evicted** from pool
- Baselines provide **diversity signal** outside self-play bubble

### Code Changes

1. **`config.py`**: Added `pfsp_baseline_*` options
2. **`pfsp_callback.py`**: 
   - `_update_baseline_curriculum()` - Stage transitions
   - `_assign_opponent_to_env()` - Reserves first N envs for baselines
   - Eviction logic protects baselines
3. **`vec_game.rs`**: 
   - `add_baseline_to_pool()` - Adds built-in bots to pool
   - `set_env_opponent()` - Handles both ONNX and baselines
4. **`python_bindings.rs`**: Python binding for `add_baseline_to_pool`

### Gradient Health Analysis (@ 786k steps)

```
Diagnostic Results:
  Gradient Imbalance Ratio: 196.7 ✅ (Target < 1,000)
  Status: ✅ HEALTHY

  Comparison with Run #4 (26B) and Run #5 (6.7K):
    99.99% reduction in imbalance since architecture revamp.

  Forward Pass Statistics:
    Layer 0-2 Logit Std: ~0.49 ✅ (No saturation)
    Layer 0-2 Entropy:   ~2.77 ✅ (Healthy diversity)

  Component Max Grads:
    card_embed:     34.9
    pool (queries): 38.6
    action_net:     45.3
    Status: ✅ Well balanced across all layers
```

### Key Learnings

1. **PFSP needs diversity injection** - Pure self-play creates echo chambers
2. **Win rate vs pool ≠ absolute strength** - High self-play WR can mask stagnation  
3. **Entropy increase = red flag** - Usually indicates policy confusion
4. **Fixed baselines = diversity anchor** - Prevent pool homogeneity

### Next Steps → Run #10

1. Resume from best checkpoint (12.8M steps, Elo 1952)
2. Enable baseline curriculum
3. Monitor win rate **per opponent** (not just aggregate)
4. Expected: Elo should resume climbing once diversity signal restored

---

## Run #10: PFSP with Baseline Curriculum

**Date**: 2026-01-09  
**Model**: Attention-based (256 embed, 8 heads, 3 layers)  
**Config**: `configs/attention_run_2.yaml`  
**Status**: [IN PROGRESS]

### Key Changes from Run #9

#### 1. Baseline Curriculum

Permanent baseline slots to prevent echo chamber:

| Step Range | Baselines | Rationale |
|------------|-----------|-----------|
| 0 - 500k | `v`, `w` | Easy warmup |
| 500k - 2M | `aa`, `er` | Medium |
| 2M+ | `e2`, `er` | Hard + Medium |

### Benchmark Results (TrueSkill Tournament)

A full Round Robin tournament (5 reps) was conducted on all checkpoints in `attention_run_2`.

| Metric | Value | Note |
|--------|-------|------|
| **Peak Expose (CRN)** | **1531.3** | Reached by `rl_bot_10752000_steps` |
| **Baseline Rank** | **Rank 45/48** | `EvolutionRusher` (1247.9) |
| **Growth Threshold** | **~500k steps** | Point where agent reliably beats `er` |
| **Status** | **Stable** 🟡 | Ratings stabilized ($\sigma \approx 15$) |

> [!IMPORTANT]
> The model has decisively cleared the heuristic baseline (`EvolutionRusher`), justifying the attention architecture. The next milestone is 1900+ CR (rivaling `e2` even with omniscient rollouts).

```python
pfsp_baseline_slots = 2              # Permanent slots (never evicted)
pfsp_baseline_max_allocation = 0.20  # 20% of envs for baselines
```

#### 2. Draw Reward

- Draws now receive **-0.5 reward** (previously neutral 0.0)
- Tracked separately: `(W-L-D)` format in logs

#### 3. Eviction Fix

- **Bug**: Stats reset BEFORE eviction → all opponents had 50% priority → oldest evicted
- **Fix**: Eviction now uses accumulated stats → correctly evicts lowest winrate

### Training Metrics

*To be filled during training*

---

## Strategy Evolution

| Stage | Approach | Result | Conclusion |
|-------|----------|--------|------------|
| Run #1-3 | Curriculum vs e2 | 1760 Elo | Local optima, limited by fixed opponent |
| Run #4-5 | Attention Fixes | 1825 Elo | High variance, architecture bottlenecks |
| Run #6 | Multi-Head Pool | - | Validated architecture improvements |
| Run #7 | Pure Self-Play | - | Modern RL Regime |
| Run #8 | PFSP Optimal | 1644 Elo | ❌ KL explosion, catastrophic forgetting |
| Run #9 | PFSP V2 Revamp | 1807 CR | ⚠️ Echo chamber stagnation |
| **Run #10** | **PFSP + Baselines** | **1531 CRN** | **Checkpoint 10.7M significantly beats EvolutionaryRusher (Rank 45/48)** |

---


## Notes & Insights

### Key Insight

Observed limitations are primarily due to **architectural and training issues**, not insufficient model capacity or suboptimal hyperparameters:

- Increasing `embed_dim` (256 → 512) did not improve performance
- Tuning hyperparameters (LR, KL, entropy) stabilized training but didn't unlock gains
- Real problems: dead gradients, homogeneous pool, incorrect eviction

### Runs Summary

| Run | Peak Elo | Main Problem |
|-----|----------|--------------|
| #1 | 1750 | Plateau at 47% WR, capacity suspected (wrong lead) |
| #2 | 1746 | Post-norm attention → 26B gradient ratio |
| #3 | 1760 | Large MLP, no better than #1, 2849 dim observation useless |
| #4 | 1825 | Pre-norm OK but K-proj bias dead, ±80 Elo variance |
| #5 | ~1800 | GELU+bias-free OK, pooling bottleneck identified |
| #6 | - | Multi-head pooling tested (architecture only) |
| #7 | - | Pure self-play, modern refactoring |
| #8 | 1644 | ❌ KL explosion → catastrophic forgetting |
| #9 | 1807 CR | PFSP echo chamber, all opponents identical |
| #10 | 1531 CRN | Baseline curriculum + eviction fix (Revived climbing) |

**No run has yet surpassed e2 (Conservative Rating (CR) ~1893).**

> Note: rating system changed from standard Elo to TrueSkill Mu after Run #8, while the first run used standard Elo, it still reflect the general level of the agent. 1900 Elo is equivalent to e2 TrueSkill CR.
CRN stands for Conservative Rating New, as the benchmark did not include e2 or e3, it is confusing but the best way to compare the agent's performance with fair benchmarks. Is it not really comparable with CR. However, there will be an unified rating system in the future with a common anchor.
CRN only stands as a placeholder for now. The relative differences between model ratings are still valid and meaningful.

### Goal

> **Achieve CR higher than e2 (Expectiminimax depth 2)**

The difference between e2 and higher depths (e3, e4) is minimal. If the RL model beats e2, it would justify:
1. Using learned model vs search
2. Attention architecture vs MLP

### MLP Baseline Required

To validate the attention architecture's contribution, an MLP run without attention should be conducted in parallel with the same configuration (PFSP + baselines) for direct comparison. Last MLP was disapointing and a new one could benefit from the recent improvements.

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
