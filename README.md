<img src="./images/logo.svg" alt="Logo" width="100" height="100">

# deckgym-core: Pokémon TCG Pocket Simulator

![Card Implemented](https://img.shields.io/badge/Cards_Implemented-1709_%2F_2408_%2871.0%25%29-yellow)

**deckgym-core** is a high-performance Rust library designed for simulating Pokémon TCG Pocket games. It features a command-line interface (CLI) capable of running 10,000 simulations in approximately 3 seconds. This is the library that powers https://www.deckgym.com.

Its mission is to elevate the competitive TCG Pocket scene by helping players optimize their decks through large-scale simulations.

Join our Discord: https://discord.gg/ymxXHrzhak!

## System Requirements

While the core simulation runs on CPU, using ONNX neural networks with GPU acceleration requires the following:

### GPU Acceleration (Optional)

If you plan to use the `onnx` feature with CUDA or TensorRT:
- **NVIDIA GPU** (Compute Capability 7.0+)
- **NVIDIA Driver**
- **CUDA Toolkit** (11.8+)
- **cuDNN** (8.x or 9.x)
- **TensorRT** (Optional, for TensorRT provider)

> [!NOTE]
> **Automatic Discovery**: The build system automatically discovers these libraries on your system. You **do not** need to manually set `LD_LIBRARY_PATH`.

## Usage

We already provide several example decks in the repo you can use to get started. For example, to face off a VenusaurEx-ExeggutorEx deck with a Weezing-Arbok deck 1,000 times, run:

```bash
cargo run simulate example_decks/venusaur-exeggutor.txt example_decks/weezing-arbok.txt --num 1000 -v
```

You can also simulate one deck against multiple decks in a folder. The total games will be distributed evenly across all decks:

```bash
# Simulate your deck against all decks in example_decks folder (1000 games total)
cargo run simulate my_deck.txt example_decks/ --num 1000 -v
```

## Terminal User Interface (TUI)

The TUI provides an interactive way to view and replay games with a visual representation of the game state.

To use the TUI, you need to enable the `tui` feature:

```bash
cargo run --bin tui --features tui -- example_decks/venusaur-exeggutor.txt example_decks/weezing-arbok.txt --players e,e
```

### Controls

- **↑/↓/Space**: Navigate between game states (forward/backward)
- **PageUp/PageDown**: Scroll through battle log
- **A/D**: Scroll through your hand cards
- **Shift+A/Shift+D**: Scroll through opponent's hand cards
- **Q/Esc**: Quit

## Contributing

New to Open Source? See [CONTRIBUTING.md](./CONTRIBUTING.md).

The main contribution is to implement more cards, basically their attack and abilities logic. This makes the cards eligible for simulation and thus available for use in https://www.deckgym.com.

See the Claude [SKILL.md](./.claude/skills/implement-cards/SKILL.md) describing how to implement cards.
It's good documentation for humans and AIs alike.


## Appendix: Useful Commands

Once you have Rust installed (see https://www.rust-lang.org/tools/install) you should be able to use the following commands from the root of the repo:

**Running Automated Test Suite**

```bash
cargo test
```

**Running Benchmarks**

```bash
cargo bench
```

**Running Main Script**

```bash
# Simulate between two specific decks
cargo run simulate example_decks/venusaur-exeggutor.txt example_decks/weezing-arbok.txt --num 1000 --players r,r
cargo run simulate example_decks/venusaur-exeggutor.txt example_decks/weezing-arbok.txt --num 1 --players r,r -vv
cargo run simulate example_decks/venusaur-exeggutor.txt example_decks/weezing-arbok.txt --num 1 --players r,r -vvvv

# Simulate one deck against all decks in a folder (games distributed evenly)
cargo run simulate example_decks/venusaur-exeggutor.txt example_decks/ --num 1000 --players r,r -v

# Optimize incomplete decks
cargo run optimize example_decks/incomplete-chari.txt A2147,A2148 example_decks/ --num 10 --players e,e -v
cargo run optimize example_decks/incomplete-chari.txt A2147,A2147,A2148,A2148 example_decks/ --num 1000 --players r,r -v --parallel
```

**Card Search Tool**

The repository includes a search utility that's particularly useful for agentic AI applications, as reading the complete `database.json` file (which contains all card data) often exceeds context limits.

```bash
# Search for cards by name
cargo run --bin search "Charizard"

# Search for cards with specific attacks
cargo run --bin search "Venusaur" --attack "Giant Bloom"
```

**Card Implementation Status Tool**

Check which cards are fully implemented versus which are missing attack effects, abilities, or trainer logic. This tool helps contributors identify cards that need implementation work.

```bash
# Show all cards with their implementation status
cargo run --bin card_status

# Show only incomplete cards
cargo run --bin card_status -- --incomplete-only

# Get the first incomplete card (useful for automation)
cargo run --bin card_status -- --first-incomplete
```

The tool displays a summary showing total cards, completion percentage, and a breakdown of missing implementations by type (attacks, abilities, tools, trainer logic).

Compare decision-making speed of different player strategies (running 1000 games in parallel):

```bash
python python/scripts/benchmark_players.py
```

| Player | Code | Games/s | Notes |
|--------|------|---------|-------|
| AttachAttack | `aa` | ~30300 | Simple heuristic, highly parallelizable |
| EvolutionRusher | `er` | ~20000 | Prioritizes evolutions |
| WeightedRandom | `w` | ~15000 | Slight heuristics overhead |
| Random | `r` | ~10600 | Baseline random policy |
| EndTurn | `et` | ~4200 | Always ends turn immediately |
| ValueFunction | `v` | ~900 | Evaluates board state heuristically |
| **ONNX o1 (GPU)** | `o1g` | **~210** | **Flagship Model** (Attention-based) |
| ExpectiMiniMax (d=2) | `e2` | ~150 | Search tree with depth 2 |
| ExpectiMiniMax (d=3) | `e3` | ~40 | Search tree with depth 3 |
| **ONNX o1 (CPU)** | `o1c` | **~35** | Attention model (15MB) |
| **MCTSPlayer** | `m[n]` | Varies | Omniscient search (lookahead) |

> [!NOTE]
> **Omniscience & "Cheating"**: Currently, `ExpectiMiniMaxPlayer` and `MCTSPlayer` have access to the full `State` object during their search, including hidden information like the opponent's hand and deck order. This makes them extremely difficult to beat and explains why a ~30% winrate against `e2` is considered an excellent performance for a non-cheating agent.
> 
> **Heuristics**: `EvolutionRusher` (`er`) is currently the strongest non-cheating heuristic baseline.

> **Note**: ONNX performance is limited by single-sample inference overhead. In batch mode (training with `VecGame`), throughput is much higher. The `o[n]` player loads the *n* newest `.onnx` model from `models/` directory.

**Temporary Deck Generator**

Generate a valid temporary test deck for a specific card id (it considers the evolution chain of a card and the required energy types if its a pokemon card).

```bash
cargo run --bin temp_deck_generator -- "A1 035"
```

**Setting Up Git Hooks (Optional)**

The repository includes a pre-commit hook that ensures code quality by automatically fixing issues and running tests before each commit. To enable it:

```bash
git config core.hooksPath .githooks
```

The pre-commit hook runs:
1. `cargo clippy --fix --allow-dirty --features tui -- -D warnings` - Auto-fixes linting issues
2. `cargo fmt` - Auto-formats code
3. `git add -u` - Adds clippy and formatting fixes to the commit
4. `cargo test --features tui` - Runs the full test suite (fails commit if tests fail)

This helps maintain code quality and prevents broken commits, but it's optional and each developer can choose whether to enable it.

**Generating database.rs**

Ensure database.json is up-to-date with latest data. Mock the `get_card_by_enum` in `database.rs` with a `_ => panic` so that
it compiles mid-way through the generation.

```bash
cargo run --bin card_enum_generator > tmp.rs && mv tmp.rs src/card_ids.rs && cargo fmt
```

Then temporarily edit `database.rs` for `_` to match Bulbasaur (this is so that the next code can compile-run).

```bash
cargo run --bin card_enum_generator -- --database > tmp.rs && mv tmp.rs src/database.rs && cargo fmt
```

To generate attacks do:
```bash
cargo run --bin card_enum_generator -- --attack-map > tmp.rs && mv tmp.rs src/actions/effect_mechanic_map.rs && cargo fmt
```

**Profiling Main Script**

```bash
cargo install flamegraph
sudo cargo flamegraph --root --dev -- simulate example_decks/venusaur-exeggutor.txt example_decks/weezing-arbok.txt --num 1000 && open flamegraph.svg
```

## Reinforcement Learning Agent

The repository includes a deep reinforcement learning agent trained using PPO (Proximal Policy Optimization) with curriculum learning and Prioritized Fictitious Self-Play.

### Architecture: Attention-Based Observation

The agent uses a **card-level attention mechanism** that processes each card individually:

| Feature | Value |
|---------|-------|
| Observation size | 2219 dims (41 global + 18 cards × 121 features) |
| Architecture | Modern Transformer (3 layers, 8 heads, GELU) |
| Card encoding | Intrinsic properties + Position (Hand + Board) |
| Training Mode | **Pure Self-Play** (PFSP Enabled) |
| Configuration | **Centralized** via `python/deckgym/config.py` |

See [RL_ARCHITECTURE.md](RL_ARCHITECTURE.md) for detailed architecture documentation.

### Pure Self-Play Training (Default)

The agent learns through continuous self-play against its own previous versions using **Prioritized Fictitious Self-Play (PFSP)**. It faces 1k+ meta decks from step 1, ensuring high-quality adversarial gradients.

| Feature | Description |
|---------|-------------|
| **Meta-Sampling** | Randomly samples from meta_deck.json every episode |
| **PFSP Pool** | Manages a prioritized pool of historical versions based on winrate |
| **Stability** | Rust-side `catch_unwind` prevents game-state crashes |
| **YAML Config** | Fully configurable via `configs/template.yaml` |

| Tier | Participant | Win% (vs e2) | Evaluation Mode |
|:----:|-------------|-----:|:--- |
| **S+** | **Expectiminimax(2)** 🎯 | 50% | Reference (Cheating) |
| **S** | **ONNX o1 (best)** 🚀 | **45.02%** | **Extensive Chaos** |

> [!TIP]
> **Context on o1 Performance**: While `o1` consistently maintained a >50% winrate during training (stochastic sampling), its definitive **45.02%** winrate on the **Extensive Chaos** benchmark reflects a more rigorous, deterministic evaluation across 396 unique archetypes. This makes `o1` our first non-cheating agent to break the 40% barrier against a cheating minimax reference.

### Usefull Commands

```bash
# Generate a training configuration template
python python/deckgym/config.py --generate-template my_config.yaml

# Train with pure self-play using YAML config
python python/scripts/train.py --config my_config.yaml

# Audit model or directory of models compared to TrueSkill baselines leaderboard
python python/scripts/evaluate.py  audit checkpoints/rl_bot_xxx_steps.zip

python python/scripts/evaluate.py  audit_dir checkpoints/

# Diagnose model gradients, weights, biais
python python/scripts/diagnose.py --model checkpoints/rl_bot_xxx_steps.zip

# Export model .zip to .onnx
python python/scripts/onnx_export.py checkpoints/rl_bot_xxx_steps.zip
```
*Note : Almost all of the .py scripts should have helpers, use -h --help to see them*

### Hyperparameters

See [RL_ARCHITECTURE.md](RL_ARCHITECTURE.md) for detailed hyperparameters information.

