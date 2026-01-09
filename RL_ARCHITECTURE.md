# RL Architecture: Attention-Based Card Observation

This document describes the attention-based reinforcement learning architecture for the Pokemon TCG Pocket agent.

## Overview

The agent uses a custom feature extractor (`CardAttentionExtractor`) that processes the game state as a **set of cards** rather than a fixed feature vector. This enables:

1. **Permutation invariance** - Order of cards doesn't affect the representation
2. **Card-level reasoning** - Agent can attend to specific cards and their relationships
3. **Generalization** - Same model works for any card combination

### Observation Structure

### Global Features (41 dims)

| Feature | Dimensions | Description |
|---------|------------|-------------|
| Turn count | 1 | Normalized by max turns |
| Points | 2 | Self and opponent (normalized) |
| Deck sizes | 2 | Self and opponent (normalized) |
| Hand sizes | 2 | Self and opponent (normalized) |
| Discard sizes | 2 | Self and opponent (normalized) |
| Deck energy types | 32 | 8 energy types × 2 slots × 2 players (multi-hot) |

### Card Features (121 dims per card × 18 cards = 2178 dims)

Each card is encoded with intrinsic properties and position information:

#### Intrinsic Features (121 dims)
- `is_pokemon` (1) - Pokemon vs Trainer
- `stage` (1) - Basic/Stage1/Stage2 normalized
- `energy_type` (10) - One-hot energy type
- `hp_current`, `hp_max` (2) - Normalized HP
- `weakness` (10) - One-hot weakness type
- `ko_points` (1) - Points awarded on KO
- `energy_attached` (10) - Per-type energy count
- `attack_1` (4 + 13) - Damage, cost, effect flag, has_attack, effect categories (V3: 13 cats)
- `attack_2` (4 + 13) - Same as attack_1
- `status` (4) - Poison, sleep, paralysis, confusion
- `ability_categories` (17) - Ability effect encoding (enhanced)
- `supporter_categories` (14) - Supporter effect encoding (enhanced)
- `retreat_cost` (1) - Normalized retreat cost
- `attached_tool` (8) - One-hot tool type

#### Position Features (8 dims)
- `zone` (2) - One-hot: hand/board
- `owner` (1) - 0=self, 1=opponent
- `is_active` (1) - Active pokemon flag
- `slot` (4) - One-hot board position (active + 3 bench)

### Total Observation Size: 2219 dims (41 + 18×121)

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
│     (131)       │             │   (18 × 121)            │
└─────────────────┘             └─────────────────────────┘
          │                               │
           ▼                               ▼
┌─────────────────┐             ┌─────────────────────────┐
│     Global MLP  │             │     Card Embedding      │
│  41 → 128 → 128 │             │   121 → 256 → 256       │
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
          │               │ - Bias-free projections         │
          │               │ - Mask empty card slots         │
          │               └─────────────────────────────────┐
          │                               │
          │                               ▼
          │               ┌─────────────────────────────────┐
          │               │ Multi-Head Attention Pooling    │
          │               │ - 4 learned queries (config)    │
          │               │ - Final LayerNorm before Pool   │
          │               │ - Output: 4 × 256 = 1024 dims   │
          │               └─────────────────────────────────┐
          │                               │
          │                               ▼
          │               ┌─────────────────────────────────┐
          │               │     Output Projection           │
          │               │   1024 → 512 (LayerNorm, GELU)  │
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
│ 640 → 256 → 128 │             │ 640 → 256 → 128 │
│  → actions      │             │  → value        │
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
| `python/scripts/evaluate.py` | Leaderboard & TrueSkill evaluation (revamped from Elo) |

## Hyperparameters (Managed in `config.py`)

The default configuration uses the following values:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `attention_embed_dim` | 256 | Transformer embedding dimension |
| `attention_num_heads` | 8 | Multi-head attention heads |
| `attention_num_layers` | 3 | Transformer encoder layers |
| `base_learning_rate` | 1e-5 | Peak LR after warmup (stabilized) |
| `ent_coef` | 0.02 | Entropy coefficient for exploration |
| `batch_size` | 1024 | Minibatch size for PPO updates |
| `n_steps` | 256 | Steps per environment per PPO update |
| `gamma` | 0.98 | PPO discount factor |
| `total_timesteps` | 30M | Target training duration (long-term) |
| `use_pfsp` | True | Enable Prioritized Fictitious Self-Play |

*Note : current run uses n_steps = 64 to compensate for n_envs = 128, making full use of the GPU.*

## Evaluation (TrueSkill)

The evaluation system in `evaluate.py` has been revamped to use **TrueSkill** instead of traditional Elo. This provides:
- **Faster Convergence**: Ratings stabilize with fewer matches.
- **Uncertainty Tracking**: Models have a `mu` (mean) and `sigma` (uncertainty).
- **Matchmaking**: More accurate assessment of model performance against baselines and other pool models.

## Training Performance

With the current attention-based architecture and `configs/attention_baseline.yaml`:
- **Training Speed**: ~440 it/s (on RTX 3050).
- **Adaptability**: Training now uses pure self-play by default, sampling from 1k+ meta decks via `MetaDeckLoader`.

```bash
python python/scripts/train.py --config configs/baseline.yaml
```