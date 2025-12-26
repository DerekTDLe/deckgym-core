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
| Deck energy types | 16 | 8 energy types × 2 players (multi-hot) |

### Card Features (118 dims per card × 40 cards = 4720 dims)

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

#### Position Features (10 dims)
- `zone` (4) - One-hot: board/hand/discard/deck
- `owner` (1) - 0=self, 1=opponent
- `slot` (4) - One-hot board position (active + 3 bench)
- `visibility` (1) - 1.0=visible, 0.0=hidden

### Total Observation Size: 4745 dims (25 + 40×118)

## Card Distribution

| Player | Zone | Cards | Visibility |
|--------|------|-------|------------|
| Self | Board | 0-4 | Full |
| Self | Hand | 0-10 | Full |
| Self | Discard | 0-20 | Full |
| Self | Deck | 0-20 | Full |
| Opponent | Board | 0-4 | Full |
| Opponent | Discard | 0-20 | Full |
| Opponent | Hand | 0-10 | **Hidden** (visibility=0) |
| Opponent | Deck | 0-20 | **Hidden** (visibility=0) |

Each player always has exactly 20 cards distributed across zones.

## Neural Network Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Observation (4745)                        │
└─────────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────┐             ┌─────────────────────────┐
│ Global Features │             │   Card Features         │
│     (25)        │             │   (40 × 118)            │
└─────────────────┘             └─────────────────────────┘
          │                               │
          ▼                               ▼
┌─────────────────┐             ┌─────────────────────────┐
│   Global MLP    │             │     Card Embedding      │
│   25 → 64 → 64  │             │     118 → 128 → 128     │
└─────────────────┘             └─────────────────────────┘
          │                               │
          │                               ▼
          │               ┌─────────────────────────────────┐
          │               │   Transformer Encoder (2 layers)│
          │               │   - 128 dim, 4 heads            │
          │               │   - Mask empty slots            │
          │               └─────────────────────────────────┘
          │                               │
          │                               ▼
          │               ┌─────────────────────────────────┐
          │               │     Mean Pooling (masked)       │
          │               │     40×128 → 128                │
          │               └─────────────────────────────────┘
          │                               │
          └───────────────┬───────────────┘
                          ▼
              ┌───────────────────────┐
              │      Concatenate      │
              │      64 + 128 = 192   │
              └───────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────┐             ┌─────────────────┐
│   Policy Head   │             │   Value Head    │
│ 192 → 256 → 128 │             │ 192 → 256 → 128 │
│     → actions   │             │     → value     │
└─────────────────┘             └─────────────────┘
```

## Key Implementation Files

| File | Purpose |
|------|---------|
| `src/rl/observation.rs` | Rust observation encoding (4745 dims) |
| `python/deckgym/attention_policy.py` | CardAttentionExtractor & policy |
| `python/scripts/train.py` | Training with attention architecture |

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `attention_embed_dim` | 128 | Embedding dimension after card projection |
| `attention_num_heads` | 4 | Multi-head attention heads |
| `attention_num_layers` | 2 | Transformer encoder layers |
| `learning_rate` | 5e-5 | Lower LR for larger model |
| `ent_coef` | 0.02 | Higher entropy for exploration |
| `batch_size` | 512 | Minibatch size |
| `n_steps` | 8192 | Experience per update |

## Training Notes

### Speed vs Old MLP
- MLP (445 dims): ~1000 it/s
- Attention (4745 dims): ~350 it/s
Note : This is for a RTX3050 with 12 envs and only indicative of relative speed.

### Known Issues
- `torch.compile()` breaks checkpoint loading
- SB3 cannot properly reload compiled models for the frozen opponent updates