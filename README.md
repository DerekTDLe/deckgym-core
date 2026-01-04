<img src="./images/logo.svg" alt="Logo" width="100" height="100">

# deckgym-core: Pokémon TCG Pocket Simulator

![Card Implemented](https://img.shields.io/badge/Cards_Implemented-1709_%2F_2408_%2871.0%25%29-yellow)

**deckgym-core** is a high-performance Rust library designed for simulating Pokémon TCG Pocket games. It features a command-line interface (CLI) capable of running 10,000 simulations in approximately 3 seconds. This is the library that powers https://www.deckgym.com.

Its mission is to elevate the competitive TCG Pocket scene by helping players optimize their decks through large-scale simulations.

Join our Discord: https://discord.gg/ymxXHrzhak!

## Usage

The CLI runs simulations between two decks in DeckGym Format. To create these files, build your decks in https://www.deckgym.com/builder, select **Share** > **Copy as Text**, and save the content as a text file.

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

The repository includes a deep reinforcement learning agent trained using PPO (Proximal Policy Optimization) with curriculum learning.

### Architecture: Attention-Based Observation

The agent uses a **card-level attention mechanism** that processes each card individually:

| Feature | Value |
|---------|-------|
| Observation size | 2849 dims (41 global + 24 cards × 117 features) |
| Architecture | Transformer Encoder (2 layers, 4 heads) |
| Card encoding | Intrinsic properties + position (no hidden cards) |
| Permutation invariant | ✅ Order of cards doesn't matter |

See [RL_ARCHITECTURE.md](RL_ARCHITECTURE.md) for detailed architecture documentation.

### Curriculum Training (Default)

Training uses a **progressive curriculum** that starts with easier opponents:

| Stage | Opponent | Win Rate Threshold |
|-------|----------|-------------------|
| warmup | e2 (Expectiminimax depth 2) | 55% |
| meta | e2 with meta decks | 55% |
| advanced | e3 (Expectiminimax depth 3) | 50% |
| mastery | Self-play | - |

### Elo Leaderboard (300 games/baseline)

*Note : Newest and current model architecture did not finish training yet, this serves as a temporary demonstration of the agent level of play*

| Rank | Player | Elo | Win Rate |
|------|--------|-----|----------|
| 1 | Expectiminimax(5) | 2020 | 84.0% |
| 2 | Expectiminimax(4) | 2000 | 80.0% |
| 3 | Expectiminimax(3) | 1978 | 82.0% |
| 4 | **RL Agent (42M)** | **1935** | **69.4%** |
| 5 | Expectiminimax(2) | 1893 | 76.2% |
| 6 | EvolutionRusher | 1549 | 44.2% |
| 7 | ValueFunction | 1451 | 31.3% |
| 8 | WeightedRandom | 1340 | 38.2% |
| 9 | AttachAttack | 1273 | 29.6% |
| 10 | EndTurn | 1200 | 0.0% |
| 11 | Random | 1081 | 19.8% |

### Training Commands

```bash
# Train with curriculum (default - uses Rust-side batching)
python python/scripts/train.py --steps 30000000

# Train with legacy DummyVecEnv (required for self-play stage)
python python/scripts/train.py --no-batched-env --steps 30000000

# Evaluate model
python python/scripts/evaluate.py eval checkpoints/rl_bot_xxx_steps.zip --games 300
```

### Hyperparameters

See [RL_ARCHITECTURE.md](RL_ARCHITECTURE.md) for detailed hyperparameters information.

