# Deep Learning & Self-Play Architecture

This document outlines the architecture for the high-performance Reinforcement Learning (RL) bot for `deckgym-core`.

## Goal
Create a **Self-Play RL Loop** where a model learns from scratch by playing against itself, bypassing the need for heuristic bots. The final model is exported to ONNX for fast inference in Rust.

---

## Architecture Overview

### 1. Neural Network Inputs (Observation Space)

We flatten the game state into a structured tensor of **755 floats**.

| Category | Features | Description |
|----------|----------|-------------|
| **Global** | 17 | Turn count, Points, Deck/Discard sizes, Energy composition |
| **Board State** | 8 slots × 92 features | Per-Pokemon features for Active + 3 Bench (both players) |
| **Hand Features** | 2 players × 15 features | Pokemon counts, trainer categories, can-evolve flags |

#### Board Slot Features (92 per slot)
- **Stage**: 0=Basic, 0.5=Stage1, 1.0=Stage2
- **HP**: Current ratio, max normalized
- **Card Energy Type**: One-hot (10 floats)
- **Weakness Type**: One-hot (10 floats)
- **Knockout Points**: 0.33=normal, 0.67=ex, 1.0=mega
- **Energy Attached**: Count per type (10 floats)
- **Energy Delta**: Missing energy for strongest attack
- **Attack 1**: Damage, cost, has_effect + **12 effect categories**
- **Attack 2**: Damage, cost, has_effect + **12 effect categories**
- **Status**: Poison, Sleep, Paralyze, Confusion
- **Ability**: **9 ability effect categories**
- **Retreat Cost**: Normalized

#### Effect Category Encoding
Attacks are encoded by semantic categories (multi-hot), enabling generalization:
- `Heal`, `StatusInflict`, `Variance`, `EnergyManip`, `CardAdvantage`
- `ConditionalDamage`, `SpreadDamage`, `SelfDamage`
- `Protection`, `Disruption`, `Movement`, `BoardDevelopment`

### 2. Neural Network Outputs (Action Space)

- **Size**: 130 discrete actions
- **Algorithm**: `MaskablePPO` (Proximal Policy Optimization with Action Masking)
- **Action Masking**: Simulator provides boolean mask of valid actions. Invalid actions are zeroed before sampling.
- **Canonical Sorting**: Strict action ordering for stability (Index 5 = "Attack 1", etc.)

### 3. Training Strategy

#### Self-Play with Frozen Opponent
- Agent (Player 0) trains against a **frozen copy** of itself (Player 1)
- Frozen opponent updated every 50,000 steps
- Prevents self-play collapse while maintaining meaningful training signal

#### Curriculum Learning
Progressive deck difficulty based on training progress:
- **Phase 0-100k**: 100% simple decks (28 expert solo battle decks)
- **Phase 100k+**: Linear transition to meta decks
- **Full meta**: 389 archetypes, ~5,500 decks from competitive play

#### Deck Sampling Strategy
Two sampling modes for maximum diversity:

| Mode | Description | Use Case |
|------|-------------|----------|
| **Hierarchical** | Random archetype → random deck | Archetype diversity |
| **Stratified** | 10% weak / 35% mid / 55% strong | Strength diversity |

Strength tiers:
- **Weak**: strength < 0.45
- **Mid**: 0.45 ≤ strength < 0.6
- **Strong**: strength ≥ 0.6

### 4. Safety Mechanisms

To prevent training crashes from simulator edge cases:
- **MAX_TURNS = 99**: Official game limit
- **MAX_ACTIONS_PER_TURN = 50**: Prevents single-turn infinite loops
- **MAX_TOTAL_ACTIONS = 500**: Prevents runaway episodes
- **Exception handling**: Catches Rust panics, ends episode with neutral reward

### 5. Technical Stack

| Component | Technology |
|-----------|------------|
| **Simulator** | Rust (core game logic, observation tensor, action mask) |
| **RL Framework** | Python (`stable-baselines3`, `sb3_contrib`) |
| **Environment** | `gymnasium` with custom `SelfPlayEnv` wrapper |
| **Neural Network** | PyTorch MLP (512-512-256 policy, 512-512-256 value) |
| **Logging** | TensorBoard |

---

## Implementation Status

### ✅ Phase 0: Rust Core Extensions
| Task | Status | Description |
|------|--------|-------------|
| 0.1 | ✅ Done | Canonical action sorting (`Ord` trait) |
| 0.2 | ✅ Done | `get_observation_tensor()` → 755 floats |
| 0.3 | ✅ Done | `get_action_mask()` → 130 bools |
| 0.4 | ✅ Done | Python bindings (`get_obs()`, `action_masks()`, `step_action()`) |

### ✅ Phase 1: Python Gym Environment
| Task | Status | Description |
|------|--------|-------------|
| 1.1 | ✅ Done | `DeckGymEnv` with `reset()`, `step()`, `action_masks()` |
| 1.2 | ✅ Done | `MetaDeckLoader` with hierarchical/stratified sampling |
| 1.3 | ✅ Done | Stress tests (100+ games without errors) |

### ✅ Phase 2: Self-Play Training Loop
| Task | Status | Description |
|------|--------|-------------|
| 2.1 | ✅ Done | `train.py` with `MaskablePPO` |
| 2.2 | ✅ Done | Self-play with frozen opponent |
| 2.3 | ✅ Done | Curriculum learning (simple → meta decks) |
| 2.4 | ✅ Done | `TrainingConfig` dataclass for hyperparameters |
| 2.5 | ✅ Done | Safety limits (max turns, max actions, exception handling) |

### 🔄 Phase 3: Evaluation & Export
| Task | Status | Description |
|------|--------|-------------|
| 3.1 | ✅ Done | `evaluate.py` with multiple bot benchmarks |
| 3.2 | 📋 TODO | ONNX export for fast inference |
| 3.3 | 📋 TODO | `RLPlayer` in Rust with ONNX inference |

### 📋 Phase 4: Performance Optimization (TODO)
| Task | Description |
|------|-------------|
| 4.1 | **Vectorized environments** (`SubprocVecEnv`) for +2x speedup |
| 4.2 | Hyperparameter tuning (entropy, learning rate schedules) |
| 4.3 | Population-based training for diverse policies |

---

## Data Sources

Meta deck data sourced from:
[pokemon-tcg-pocket-tier-list](https://github.com/chase-manning/pokemon-tcg-pocket-tier-list)

---

## Usage

### Training
```bash
python python/scripts/train.py --steps 5000000
```

### Evaluation
```bash
python python/scripts/evaluate.py --model models/rl_bot.zip --benchmark all
```

### Key CLI Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--steps` | 1,000,000 | Total training timesteps |
| `--lr` | 3e-4 | Learning rate |
| `--ent-coef` | 0.01 | Entropy coefficient (exploration) |
| `--batch-size` | 512 | Minibatch size |
| `--opponent-update-freq` | 50,000 | Frozen opponent update frequency |
