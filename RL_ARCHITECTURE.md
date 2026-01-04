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

## Neural Network Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Observation (2849)                        │
└─────────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────┐             ┌─────────────────────────┐
│ Global Features │             │   Card Features         │
│     (25)        │             │   (24 × 118)            │
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
          │               │     24×128 → 128                │
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

## Curriculum Training

Training uses a **progressive curriculum** that starts with easier opponents:

| Stage | Opponent | Decks | Win Rate Threshold |
|-------|----------|-------|-------------------|
| warmup | e2 (Expectiminimax depth 2) | simple | 65% |
| meta | e2 | meta | 65% |
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
| `src/rl/observation.rs` | Rust observation encoding (2857 dims) |
| `src/vec_game.rs` | Rust-side batched VecEnv for fast training |
| `python/deckgym/batched_env.py` | SB3-compatible BatchedDeckGymEnv wrapper |
| `python/deckgym/attention_policy.py` | CardAttentionExtractor & policy |
| `python/deckgym/curriculum.py` | CurriculumManager for stage transitions |
| `python/scripts/train.py` | Training with curriculum & attention |
| `python/scripts/evaluate.py` | Evaluation with quick_eval_vs_bot() |

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

## Training Speed (RTX 3050, 8 envs)

| Configuration | Speed | Notes |
|---------------|-------|-------|
| BatchedDeckGymEnv + e2 | ~1078 it/s | **Default** (Rust-side batching) |
| BatchedDeckGymEnv (no bot) | ~7000 it/s | Pure env stepping |
| DummyVecEnv + e2 | ~720 it/s | Legacy (use --no-batched-env) |
| DummyVecEnv + self-play | ~490 it/s | Final mastery stage |

## Training Command

```bash
python python/scripts/train.py --steps 30000000
```

Training now uses curriculum by default (starts with e2, progresses automatically).

## Known Issues

- `torch.compile()` breaks checkpoint loading - do not use
- SB3 cannot properly reload compiled models for frozen opponent updates