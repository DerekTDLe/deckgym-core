# RL Architecture: Attention-Based Card Observation

This document describes the attention-based reinforcement learning architecture for the Pokemon TCG Pocket agent.

## Overview

The agent uses a custom feature extractor (`CardAttentionExtractor`) that processes the game state as a **set of cards** rather than a fixed feature vector. This enables:

1. **Permutation invariance** - Order of cards doesn't affect the representation
2. **Card-level reasoning** - Agent can attend to specific cards and their relationships
3. **Generalization** - Same model works for any card combination

## Observation Structure

### Global Features (41 dims)

| Feature | Dimensions | Description |
|---------|------------|-------------|
| Turn count | 1 | Normalized by max turns |
| Points | 2 | Self and opponent (normalized) |
| Deck sizes | 2 | Self and opponent (normalized) |
| Hand sizes | 2 | Self and opponent (normalized) |
| Discard sizes | 2 | Self and opponent (normalized) |
| Deck energy types | 32 | 8 energy types × 2 slots (dual type) × 2 players (multi-hot) |

### Card Features (116 dims per card × 18 cards = 2088 dims)

Each card is encoded with intrinsic properties and position information:

#### Intrinsic Features (108 dims)
- `is_pokemon` (1) - Pokemon vs Trainer
- `stage` (1) - Basic/Stage1/Stage2 normalized
- `energy_type` (10) - One-hot energy type
- `hp_current`, `hp_max` (2) - Normalized HP
- `weakness` (10) - One-hot weakness type
- `ko_points` (1) - Points awarded on KO
- `energy_attached` (10) - Per-type energy count
- `attack_1` (4 + 12) - Damage, cost, effect flag, effect categories
- `attack_2` (4 + 12) - Same as attack_1
- `status` (4) - Poison, sleep, paralysis, confusion
- `ability_categories` (16) - Ability effect encoding (enhanced)
- `supporter_categories` (12) - Supporter effect encoding (enhanced)
- `retreat_cost` (1) - Normalized retreat cost
- `attached_tool` (8) - One-hot tool type

#### Position Features (8 dims)
- `zone` (2) - One-hot: hand/board
- `owner` (1) - 0=self, 1=opponent
- `is_active` (1) - Active pokemon flag
- `slot` (4) - One-hot board position (active + 3 bench)

### Total Observation Size: 2129 dims (41 + 18×116)

## Card Distribution (Capped V3)

| Player | Zone | Cards | Visibility |
|--------|------|-------|------------|
| Self | Board | 0-4 | Full |
| Self | Hand | 0-10 | Full |
| Opponent | Board | 0-4 | Full |

**Note:** Capped hand size at 10 and removed deck/discard cards from the explicit card encoding to reduce noise and speed up training (24 -> 18 cards total).

## Neural Network Architecture (Multi-Head Pooling - Run #7)

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Observation (2129)                        │
└─────────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────┐             ┌─────────────────────────┐
│ Global Features │             │   Card Features         │
│     (41)        │             │   (18 × 116)            │
└─────────────────┘             └─────────────────────────┘
          │                               │
          ▼                               ▼
┌─────────────────┐             ┌─────────────────────────┐
│   Global MLP    │             │     Card Embedding      │
│ 41 → 128 → 128  │             │   116 → 256 → 256       │
│   (GELU)        │             │   (GELU)                │
└─────────────────┘             └─────────────────────────┘
          │                               │
          │                               ▼
          │               ┌─────────────────────────────────┐
          │               │ Transformer Encoder (3 layers)  │
          │               │ - 256 dim, 8 heads              │
          │               │ - Fused QKV projection          │
          │               │ - PRE-NORM architecture         │
          │               │ - FFN 4× expansion (256→1024)   │
          │               │ - GELU activation               │
          │               │ - Bias-free attention           │
          │               │ - Mask empty card slots         │
          │               └─────────────────────────────────┘
          │                               │
          │                               ▼
          │               ┌─────────────────────────────────┐
          │               │ Multi-Head Attention Pooling    │
          │               │ - 4 independent queries         │
          │               │ - 18×256 → 4×256 = 1024         │
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
│ 640 → 512 → 256 │             │ 640 → 512 → 256 │
│  → 128 → actions│             │  → 128 → value  │
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

**6. Larger FFN and Deep Heads**
- Standard Transformer size (256 → 1024 → 256)
- Policy/Value heads increased to 3 layers (512, 256, 128)
- GELU activation and Dropout(0.15) for regularization
- Pre-norm architecture for stable training in Run #7

### Parameter Count (Run #7)

| Component | Parameters | % of Total |
|-----------|------------|------------|
| Global projection | 21,888 | 0.6% |
| Card embedding | 30,000 | 0.8% |
| Attention layers | 786,432 | 22.0% |
| Feed-forward layers | 1,576,704 | 44.2% |
| Multi-head pooling | 263,168 | 7.4% |
| Output projection | 524,800 | 14.7% |
| Policy & Value Heads| 360,000 | 10.1% |
| **Total** | **~3,570,000** | **100%** |

## Pure Self-Play Training

Starting from **Run #7**, curriculum training was abandoned in favor of **pure self-play** from step 1:

1. **Meta Decks From Start**: No warmup with simple decks. Agent faces 100+ meta decks (`meta_deck.json`) immediately.
2. **Rollout-Based Updates**: Frozen opponent (`FrozenOpponentCallback`) is replaced with the current model every 8 rollouts.
3. **TrueSkill Scale**: Leaderboard shifted to 1500 base (sigma 500) to match external standards.
4. **Rust Stability**: Core environment loops are wrapped in `catch_unwind` to gracefully handle game-state panics without crashing training.

### Key Benefits
- **No biased learning**: Agent doesn't learn "bad habits" from simple decks.
- **Faster Mastery**: High-quality diverse decks provide a much denser training signal.
- **Robustness**: Rust hardening ensures 24/7 training stability.


## Key Implementation Files

| File | Purpose |
|------|---------|
| `src/rl/observation.rs` | Rust observation encoding (2129 dims) |
| `src/vec_game.rs` | Rust-side batched VecEnv with `catch_unwind` safety |
| `python/deckgym/batched_env.py` | SB3-compatible BatchedDeckGymEnv wrapper |
| `python/deckgym/attention_policy.py` | CardAttentionExtractor & policy (GELU, MultiHead) |
| `python/scripts/train.py` | Pure self-play training with MetaDeckLoader |
| `python/scripts/evaluate.py` | TrueSkill leaderboard and calibration |

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `attention_embed_dim` | 256 | Embedding dimension after card projection |
| `attention_num_heads` | 8 | Multi-head attention heads |
| `attention_num_layers` | 3 | Transformer encoder layers |
| `learning_rate` | 3e-4 → 1e-4 | Linear warmup + Cosine decay |
| `ent_coef` | 0.05 | High entropy for better exploratory self-play |
| `batch_size` | 1024 | Balanced batch size for attention stability |
| `n_steps` | 256 | Steps per environment per PPO update |
| `n_epochs` | 8 | PPO epochs per update |
| `gamma` | 0.98 | Standard discount factor |
| `target_kl` | 0.015 | Conservative early stop for divergence protection |
| `frozen_opponent_rollouts` | 8 | Rollouts before updating self-play opponent |

## Training Speed (RTX 3050, 8-32 envs)

| Configuration | Speed | Notes |
|---------------|-------|-------|
| BatchedDeckGymEnv + e2 | ~1078 it/s | **Default** (Rust-side batching) |
| BatchedDeckGymEnv (no bot) | ~7000 it/s | Pure env stepping |
| DummyVecEnv + e2 | ~720 it/s | Legacy (use --no-batched-env) |
| DummyVecEnv + self-play | ~490 it/s | Final mastery stage |

*Note: Scaling the number of environments yields diminishing returns; 4× envs provides only ~1.3× speedup.*

Training now uses pure self-play by default. You can use YAML configs for easy experimentation:

```bash
python python/scripts/train.py --config configs/baseline.yaml
```

## Known Issues

- `torch.compile()` breaks checkpoint loading - do not use
- SB3 cannot properly reload compiled models for frozen opponent updates