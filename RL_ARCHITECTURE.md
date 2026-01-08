# RL Architecture: Attention-Based Card Observation

This document describes the attention-based reinforcement learning architecture for the Pokemon TCG Pocket agent.

## Overview

The agent uses a custom feature extractor (`CardAttentionExtractor`) that processes the game state as a **set of cards** rather than a fixed feature vector. This enables:

1. **Permutation invariance** - Order of cards doesn't affect the representation
2. **Card-level reasoning** - Agent can attend to specific cards and their relationships
3. **Generalization** - Same model works for any card combination

### Observation Structure

### Global Features (131 dims)

| Feature | Dimensions | Description |
|---------|------------|-------------|
| Turn count | 1 | Normalized by max turns |
| Points | 2 | Self and opponent (normalized) |
| Deck sizes | 2 | Self and opponent (normalized) |
| Hand sizes | 2 | Self and opponent (normalized) |
| Discard sizes | 2 | Self and opponent (normalized) |
| Active Energy | 9 | One-hot active energy (8 types + none) |
| Deck energy types | 16 | 8 energy types × 2 slots (dual type) (multi-hot) |
| Opponent Deck energy types | 16 | 8 energy types × 2 slots (dual type) (multi-hot) |
| Opponent Active Energy | 9 | One-hot active energy (8 types + none) |
| Support used | 2 | Flags for self/opponent support used this turn |
| Supporter features | 70 | 7 types × 5 features × 2 players |

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

### Total Observation Size: 2219 dims (131 + 18×116)

## Neural Network Architecture (Dynamic Attention - Run #8)

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Observation (2219)                        │
└─────────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────┐             ┌─────────────────────────┐
│ Global Features │             │   Card Features         │
│     (131)       │             │   (18 × 116)            │
└─────────────────┘             └─────────────────────────┘
          │                               │
          ▼                               ▼
┌─────────────────┐             ┌─────────────────────────┐
│   Global MLP    │             │     Card Embedding      │
│ 131 → 128 → 128 │             │   116 → 256 → 256       │
│   (GELU)        │             │   (GELU)                │
└─────────────────┘             └─────────────────────────┘
          │                               │
          │                               ▼
          │               ┌─────────────────────────────────┐
          │               │ Transformer Encoder (3 layers)  │
          │               │ - Config: embed_dim, heads      │
          │               │ - Fused QKV projection          │
          │               │ - PRE-NORM architecture         │
          │               │ - FFN 4× expansion (dynamic)    │
          │               │ - GELU activation               │
          │               │ - Mask empty card slots         │
          │               └─────────────────────────────────┐
          │                               │
          │                               ▼
          │               ┌─────────────────────────────────┐
          │               │ Multi-Head Attention Pooling    │
          │               │ - 4 learned queries (config)    │
          │               │ - dynamic attended features     │
          │               └─────────────────────────────────┐
          │                               │
          │                               ▼
          │               ┌─────────────────────────────────┐
          │               │     Output Projection           │
          │               │     1024 → 512 (dynamic factor) │
          │               └─────────────────────────────────┐
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

### Dynamic Architecture & Configuration (Run #8)

**1. Centralized Configuration** (`deckgym/config.py`)
- All hyperparameters managed in a single `TrainingConfig` dataclass.
- Support for YAML generation and overrides (`--generate-template`).
- Unified observation (2219) and action (175) space constants.

**2. Dynamic Attention Settings**
- `attention_global_proj_dim`: Configurable global hidden size (default: 128).
- `attention_ff_expansion_factor`: FF expansion ratio (default: 4).
- `attention_output_proj_factor`: Final pooling projection factor (default: 2).
- `attention_mask_threshold`: Configurable precision for padding masks (default: 1e-6).

**3. Multi-Head Attention Pooling**
- Number of learned queries is now configurable via `attention_pool_queries`.
- Reduces information loss and captures specialized card states (offensive, defensive, etc.).

**4. Advanced Hardware Optimization**
- Integrated support for device selection, worker counts, and batch environment optimization in the central config.

## Key Implementation Files

| File | Purpose |
|------|---------|
| `python/deckgym/config.py` | **Central source of truth** for all training & model parameters |
| `src/rl/observation.rs` | Rust observation encoding (2219 dims) |
| `src/vec_game.rs` | Rust-side batched VecEnv with `catch_unwind` safety |
| `python/deckgym/attention_policy.py` | CardAttentionExtractor (Fully dynamic architecture) |
| `python/scripts/train.py` | Training loop (loads from `TrainingConfig`) |
| `python/scripts/evaluate.py` | Leaderboard & TrueSkill (uses centralized constants) |

## Hyperparameters (Managed in `config.py`)

The default configuration uses the following values:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `attention_embed_dim` | 256 | Transformer embedding dimension |
| `attention_num_heads` | 8 | Multi-head attention heads |
| `attention_num_layers` | 3 | Transformer encoder layers |
| `base_learning_rate` | 1e-5 | Peak LR after warmup |
| `ent_coef` | 0.02 | Entropy coefficient for exploration |
| `batch_size` | 1024 | Minibatch size for PPO updates |
| `n_steps` | 256 | Steps per environment per PPO update |
| `gamma` | 0.98 | PPO discount factor |
| `total_timesteps` | 1.5M | Target training duration |
| `use_pfsp` | True | Enable Prioritized Fictitious Self-Play |

## PFSP (Prioritized Fictitious Self-Play)

The system manages a pool of frozen opponents to prevent catastrophic forgetting and strategy cycling:
- **Pool Management**: Automatically adds current model to pool based on winrate thresholds.
- **Priority Selection**: Opponents are selected using TrueSkill-based weights.
- **Adaptive Scheduling**: Update frequency scales based on agent performance.

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