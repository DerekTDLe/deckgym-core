# Deep Learning & Self-Play Architecture

This document outlines the architecture for the high-performance Reinforcement Learning (RL) bot for `deckgym-core`.

## Goal
Create a **Self-Play RL Loop** where a model learns from scratch by playing against itself, bypassing the need for heuristic bots. The final model is exported to ONNX for fast inference in Rust.

## Architecture Overview

### 1. Neural Network Inputs (Observation Space)
We flatten the game state into a structured Tensor (~600 floats).
*   **Global Info**: Turn count, Points, Deck/Discard/Hand sizes, Energy Composition.
*   **Board State** (Active + Bench slots):
    *   **HP**: Current/Max (Normalized).
    *   **Energy**: Count per type & **Energy Delta** (Attached - Cost of Strongest Attack).
    *   **Status**: Poison, Sleep, Paralyze, Confusion.
    *   **Card ID**: **Canonical Embeddings** (Trainable vector of size 32).
*   **Hand & Discard**: Multi-hot encoding of current hand and opponent's discard pile.

> [!NOTE]
> **Why Embeddings?**
> We map ~2000 printing IDs to ~500 "Canonical" IDs. A trainable vector allows the network to learn semantic clusters (e.g., "Fire Type", "Stage 2") automatically.

### 2. Neural Network Outputs (Action Space)
*   **Algorithm**: `MaskablePPO` (Proximal Policy Optimization with Action Masking).
*   **Action Masking**: The network outputs probabilities for all possible actions (~200). The simulator provides a boolean mask of *valid* actions. `MaskablePPO` zeroes out invalid actions before sampling.
*   **Canonical Sorting**: We enforce a strict ordering of actions in the simulator (e.g., Index 5 always means "Attack 1") to ensure stability.

### 3. Training Strategy
We use **Asymmetric Training** to ensure robustness:
1.  **Agent Deck**: Randomly assigned from the entire pool (Weak & Strong).
2.  **Opponent Deck**: Follows a **Curriculum**:
    *   *Phase 1*: Random/Weak decks.
    *   *Phase 2*: Popular Meta decks (Score metric).
    *   *Phase 3*: "Boss" Decks (Strength > 0.6).

### 4. Technical Stack
*   **Rust**: Core simulator, Move Generation ($O(1)$), Observation Tensor construction.
*   **Python**: `stable-baselines3`, Gym Environment, Training Loop.
*   **Parallelism**: `SubprocVecEnv` (16-32 parallel environments).

---

## Implementation Phases

### Phase 0: Rust Core Extensions (Foundation)
**Goal**: Expose all necessary data from Rust to Python for RL.
| Task | File | Description |
|------|------|-------------|
| 0.1 | `src/actions/types.rs` | Implement `Ord` trait on `SimpleAction` for canonical sorting. |
| 0.2 | `src/rl/mod.rs` (**NEW**) | Create `get_observation_tensor(&State) -> Vec<f32>` function. |
| 0.3 | `src/rl/mod.rs` | Create `get_action_mask(&State) -> Vec<bool>` function. |
| 0.4 | `src/python_bindings.rs` | Expose new RL functions to Python (`PyGame.get_obs()`, `PyGame.get_action_mask()`, `PyGame.step_action(idx)`). |

**✅ Success Criteria**: `cargo test` passes. Python can call `game.get_obs()` and receive a `numpy` array.

---

### Phase 1: Python Gym Environment
**Goal**: Create a standard Gym-compatible environment.
| Task | File | Description |
|------|------|-------------|
| 1.1 | `python/deckgym/env.py` (**NEW**) | Implement `DeckGymEnv(gymnasium.Env)` with `reset()`, `step()`, `action_masks()`. |
| 1.2 | `python/deckgym/deck_loader.py` (**NEW**) | Implement `MetaDeckLoader` to parse `best-decks.json` and filter unimplemented cards. |
| 1.3 | `python/tests/test_env.py` (**NEW**) | Add tests using `stable_baselines3.common.env_checker.check_env`. |

**✅ Success Criteria**: `check_env(DeckGymEnv())` passes without errors. A random agent can play 1000 steps.

---

### Phase 2: Self-Play Training Loop
**Goal**: Train an agent using MaskablePPO.
| Task | File | Description |
|------|------|-------------|
| 2.1 | `python/scripts/train.py` (**NEW**) | Main training script using `sb3_contrib.MaskablePPO`. |
| 2.2 | `python/scripts/train.py` | Implement Self-Play logic (Agent plays both sides, or plays against a frozen copy). |
| 2.3 | `python/scripts/train.py` | Implement Curriculum (Phase 1-3 deck sampling). |

**✅ Success Criteria**: Training runs for 100k steps without crashing. TensorBoard shows increasing reward.

---

### Phase 3: Evaluation & ONNX Export
**Goal**: Verify the bot's strength and export for Rust inference.
| Task | File | Description |
|------|------|-------------|
| 3.1 | `python/scripts/evaluate.py` (**NEW**) | Pit trained model against `RandomPlayer` and `AttachAttackPlayer`. Target: >80% win rate. |
| 3.2 | `python/scripts/export_onnx.py` (**NEW**) | Export trained PyTorch model to ONNX format. |
| 3.3 | `src/players/rl_player.rs` (**NEW**) | Implement `RLPlayer` that loads ONNX and runs inference via `ort` crate. |

**✅ Success Criteria**: `RLPlayer` beats `AttachAttackPlayer` with >70% win rate using ONNX inference.

---

### Phase 4 (Optional): MCTS Enhancement
**Goal**: Combine NN policy with lookahead search for stronger play.
| Task | Description |
|------|-------------|
| 4.1 | Implement basic MCTS using NN policy as prior. |
| 4.2 | Benchmark search depth vs inference speed. |

