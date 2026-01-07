# RL Architecture: Attention-Based Card Observation

This document describes the attention-based reinforcement learning architecture for the Pokemon TCG Pocket agent.

## Overview

The agent uses a custom feature extractor (`CardAttentionExtractor`) that processes the game state as a **set of cards** rather than a fixed feature vector. This enables:

1. **Permutation invariance** - Order of cards doesn't affect the representation
2. **Card-level reasoning** - Agent can attend to specific cards and their relationships
3. **Generalization** - Same model works for any card combination

## Observation Structure

### Global Features (25 dims)

| Feature | Dimensions | Description |
|---------|------------|-------------|
| Turn count | 1 | Normalized by max turns |
| Points | 2 | Self and opponent (normalized) |
| Deck sizes | 2 | Self and opponent (normalized) |
| Hand sizes | 2 | Self and opponent (normalized) |
| Discard sizes | 2 | Self and opponent (normalized) |
| Deck energy types | 32 | 8 energy types × 2 slots (dual type) × 2 players (multi-hot) |

### Card Features (117 dims per card × 24 cards = 2808 dims)

Each card is encoded with intrinsic properties and position information:

#### Intrinsic Features (108 dims)
- `is_pokemon` (1) - Pokemon vs Trainer
- `stage` (1) - Basic/Stage1/Stage2 normalized
- `energy_type` (10) - One-hot energy type
- `hp_current`, `hp_max` (2) - Normalized HP
- `weakness` (10) - One-hot weakness type
- `ko_points` (1) - Points awarded on KO
- `energy_attached` (10) - Per-type energy count
- `attack_1` (4 + 24) - Damage, cost, effect flag, effect categories
- `attack_2` (4 + 24) - Same as attack_1
- `status` (4) - Poison, sleep, paralysis, confusion
- `ability_categories` (8) - Ability effect encoding
- `supporter_categories` (7) - Supporter effect encoding
- `retreat_cost` (1) - Normalized retreat cost
- `attached_tool` (8) - One-hot tool type

#### Position Features (9 dims)
- `zone` (4) - One-hot: board/hand/discard/deck
- `owner` (1) - 0=self, 1=opponent
- `slot` (4) - One-hot board position (active + 3 bench)

> Note: `visibility` was removed since all 24 encoded cards are now visible.

### Total Observation Size: 2849 dims (41 + 24×117)

## Card Distribution (Optimized)

| Player | Zone | Cards | Visibility |
|--------|------|-------|------------|
| Self | Board | 0-4 | Full |
| Self | Hand | 0-10 | Full |
| Self | Discard | 0-20 | Full |
| Self | Deck | 0-20 | Full |
| Opponent | Board | 0-4 | Full |

**Note:** Opponent hand/deck/discard are NOT encoded as cards (only their sizes in global features). This reduces observation from 40 to 24 cards for faster training.

## Neural Network Architecture (Modern Transformer - Run #6)

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Observation (2849)                        │
└─────────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────┐             ┌─────────────────────────┐
│ Global Features │             │   Card Features         │
│     (41)        │             │   (24 × 117)            │
└─────────────────┘             └─────────────────────────┘
          │                               │
          ▼                               ▼
┌─────────────────┐             ┌─────────────────────────┐
│   Global MLP    │             │     Card Embedding      │
│ 41 → 128 → 128  │             │   117 → 256 → 256       │
│   (GELU)        │             │   (GELU)                │
└─────────────────┘             └─────────────────────────┘
          │                               │
          │                               ▼
          │               ┌─────────────────────────────────┐
          │               │ Transformer Encoder (3 layers)  │
          │               │ - 256 dim, 8 heads              │
          │               │ - Fused QKV projection          │
          │               │ - Pre-norm architecture         │
          │               │ - FFN 4× expansion (256→1024)   │
          │               │ - GELU activation               │
          │               │ - Bias-free attention           │
          │               │ - Mask empty card slots         │
          │               └─────────────────────────────────┘
          │                               │
          │                               ▼
          │               ┌─────────────────────────────────┐
          │               │ Multi-Head Attention Pooling    │
          │               │ - 4 learned queries             │
          │               │ - 24×256 → 4×256 = 1024         │
          │               │ - Compression: 6× (was 24×)     │
          │               └─────────────────────────────────┘
          │                               │
          │                               ▼
          │               ┌─────────────────────────────────┐
          │               │     Output Projection           │
          │               │     1024 → 512 (GELU)           │
          │               └─────────────────────────────────┘
          │                               │
          └───────────────┬───────────────┘
                          ▼
              ┌───────────────────────┐
              │      Concatenate      │
              │   128 + 512 = 640     │
              └───────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────┐             ┌─────────────────┐
│   Policy Head   │             │   Value Head    │
│ 640 → 256 → 128 │             │ 640 → 256 → 128 │
│     → actions   │             │     → value     │
└─────────────────┘             └─────────────────┘
```

### Modern Transformer Improvements (Run #6)

**1. GELU Activation** (GPT/BERT/LLaMA style)
- Replaces ReLU throughout the network
- Smoother gradients, no dead neurons
- Better empirical performance

**2. Bias-Free Attention Projections**
- All Q, K, V, out_proj without biases
- Better gradient flow
- Fewer parameters: -3,072 total
- Used by: LLaMA, GPT-3, PaLM

**3. Fused QKV Projection** (FlashAttention style)
- Single matmul for Q, K, V (3× fewer kernel calls)
- Better memory bandwidth utilization
- Faster training and inference

**4. Multi-Head Attention Pooling**
- 4 learned queries instead of 1
- Reduces compression: 24× → 6×
- Output: 1024 dims (4 × 256)
- Captures multiple aspects of card state

**5. Expanded Output Dimensions**
- Global: 64 → 128 dims
- Cards: 256 → 512 dims
- Total: 320 → 640 dims
- Features/action ratio: 3.66 (optimal for 175 actions)

**6. FFN 4× Expansion**
- Standard Transformer size (256 → 1024 → 256)
- GELU activation
- Dropout(0.1) for regularization

### Parameter Count

| Component | Parameters | % of Total |
|-----------|------------|------------|
| Global projection | 21,888 | 0.7% |
| Card embedding | 96,000 | 2.9% |
| Attention layers | 786,432 | 24.0% |
| Feed-forward layers | 1,576,704 | 48.2% |
| Layer norms | 3,072 | 0.1% |
| Multi-head pooling | 263,168 | 8.0% |
| Output projection | 524,800 | 16.0% |
| **Total** | **3,272,064** | **100%** |

**Compression**: 2,849 input dims → 640 output dims (4.5× compression)

## Curriculum Training

Training uses a **progressive curriculum** that starts with easier opponents:

| Stage | Opponent | Decks | Win Rate Threshold |
|-------|----------|-------|-------------------|
| warmup | e2 (Expectiminimax depth 2) | simple | 55% |
| meta | e2 | meta | 55% |
| advanced | e3 (Expectiminimax depth 3) | meta | 50% |
| mastery | self-play | meta | - |

**Key benefits:**
- **Faster training**: BatchedDeckGymEnv runs at ~1078 it/s vs ~720 with DummyVecEnv
- **Score-based rewards**: +1.5 for 3-0 victory, -1.5 for 0-3 loss (encourages dominant play)
- **Stable learning**: Gradual difficulty increase prevents catastrophic forgetting
- **Automatic progression**: Transitions based on win rate, not fixed steps

## Key Implementation Files

| File | Purpose |
|------|---------|
| `src/rl/observation.rs` | Rust observation encoding (2849 dims) |
| `src/vec_game.rs` | Rust-side batched VecEnv for fast training |
| `python/deckgym/batched_env.py` | SB3-compatible BatchedDeckGymEnv wrapper |
| `python/deckgym/attention_policy.py` | CardAttentionExtractor & policy |
| `python/deckgym/curriculum.py` | CurriculumManager for stage transitions |
| `python/scripts/train.py` | Training with curriculum & attention |
| `python/scripts/evaluate.py` | Evaluation with quick_eval_vs_bot() |

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `attention_embed_dim` | 256 | Embedding dimension after card projection |
| `attention_num_heads` | 8 | Multi-head attention heads |
| `attention_num_layers` | 3 | Transformer encoder layers |
| `learning_rate` | 5e-5 → 1e-5 | Linear decay from base to min LR |
| `ent_coef` | 0.02 | Higher entropy for exploration |
| `batch_size` | 2048 | Larger batch for better GPU utilization |
| `n_steps` | 8192 | Experience per update |
| `n_epochs` | 8 | PPO epochs per update |
| `gamma` | 0.98 | Reduced for shorter episodes |
| `target_kl` | 0.015 | Early stop if KL divergence too high |
| `frozen_opponent_update_freq` | 75,000 | Steps between opponent updates |

## Training Speed (RTX 3050, 8-32 envs)

| Configuration | Speed | Notes |
|---------------|-------|-------|
| BatchedDeckGymEnv + e2 | ~1078 it/s | **Default** (Rust-side batching) |
| BatchedDeckGymEnv (no bot) | ~7000 it/s | Pure env stepping |
| DummyVecEnv + e2 | ~720 it/s | Legacy (use --no-batched-env) |
| DummyVecEnv + self-play | ~490 it/s | Final mastery stage |

*Note: Scaling the number of environments yields diminishing returns; 4× envs provides only ~1.3× speedup.*

## Training Command

```bash
python python/scripts/train.py --steps 30000000
```

Training now uses curriculum by default (starts with e2, progresses automatically).

## Known Issues

- `torch.compile()` breaks checkpoint loading - do not use
- SB3 cannot properly reload compiled models for frozen opponent updates