#!/usr/bin/env python3
"""
Evaluate a trained Pokemon TCG Pocket RL agent against various bot opponents.

Benchmarks:
- Mirror: Same deck for both players (tests pure skill)
- Easy: RL uses high-WR decks vs low-WR decks
- Hard: RL uses low-WR decks vs high-WR decks
- Random: Random meta decks for both players
"""

import argparse
import json
import os
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from rich.console import Console
from rich.table import Table

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from deckgym.deck_loader import MetaDeckLoader


# =============================================================================
# Configuration
# =============================================================================

OPPONENT_TYPES = ["r", "aa", "et", "w", "v", "e2", "er"]
OPPONENT_NAMES = {
    "r": "Random",
    "aa": "AttachAttack",
    "et": "EndTurn",
    "w": "WeightedRandom",
    "v": "ValueFunction",
    "e2": "Expectiminimax(2)",
    "er": "EvolutionRusher",
}

console = Console()


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class BenchmarkResult:
    """Aggregated results for a benchmark scenario."""
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_steps: int = 0

    @property
    def total(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total > 0 else 0.0


# =============================================================================
# Model Interface
# =============================================================================

def load_model(model_path: str):
    """Load a trained MaskablePPO model."""
    from sb3_contrib import MaskablePPO
    return MaskablePPO.load(model_path)


def get_rl_action(model, obs: np.ndarray, action_mask: np.ndarray) -> int:
    """Get deterministic action from RL model."""
    action, _ = model.predict(obs, deterministic=True, action_masks=action_mask)
    return int(action)


# =============================================================================
# Game Execution
# =============================================================================

def play_game(
    deck_a: str,
    deck_b: str,
    model,
    rl_is_player_a: bool,
    opponent_type: str,
    seed: Optional[int] = None,
) -> tuple[int, int]:
    """
    Play a single game between RL agent and bot opponent.

    Args:
        deck_a: Deck string for player A
        deck_b: Deck string for player B
        model: Trained RL model
        rl_is_player_a: Whether RL controls player A
        opponent_type: Bot type code (e.g., "aa", "v")
        seed: Random seed for reproducibility

    Returns:
        (result, steps): result is +1 win, -1 loss, 0 draw
    """
    import deckgym

    # Write decks to temp files (required by game constructor)
    deck_a_path, deck_b_path = _write_temp_decks(deck_a, deck_b)

    try:
        players = ["r", opponent_type] if rl_is_player_a else [opponent_type, "r"]
        game = deckgym.Game(deck_a_path, deck_b_path, players=players, seed=seed)
        rl_player_idx = 0 if rl_is_player_a else 1

        result, steps = _play_game_loop(game, model, rl_player_idx)

    except BaseException:
        result, steps = 0, 0  # Count errors as draws

    finally:
        os.unlink(deck_a_path)
        os.unlink(deck_b_path)

    return result, steps


def _write_temp_decks(deck_a: str, deck_b: str) -> tuple[str, str]:
    """Write deck strings to temporary files."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(deck_a)
        path_a = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(deck_b)
        path_b = f.name

    return path_a, path_b


def _play_game_loop(game, model, rl_player_idx: int) -> tuple[int, int]:
    """Execute the main game loop."""
    max_steps = 500
    steps = 0

    try:
        while not game.is_game_over() and steps < max_steps:
            current_player = game.current_player()

            if current_player == rl_player_idx:
                _execute_rl_turn(game, model)
            else:
                game.play_tick()

            steps += 1
    except BaseException:
        # Rust panic or other error - count as draw
        return 0, steps

    return _determine_result(game, rl_player_idx), steps


def _execute_rl_turn(game, model):
    """Execute a single RL agent turn."""
    obs = np.array(game.get_obs(), dtype=np.float32)
    mask = np.array(game.get_action_mask(), dtype=bool)

    if not mask.any():
        mask[0] = True  # Fallback: force end turn

    action = get_rl_action(model, obs, mask)

    try:
        game.step_action(action)
    except ValueError:
        # Invalid action - try first valid one
        valid = [i for i, v in enumerate(mask) if v]
        game.step_action(valid[0] if valid else 0)


def _determine_result(game, rl_player_idx: int) -> int:
    """Determine game result from RL agent's perspective."""
    outcome = game.get_state().winner

    if outcome is None:
        return 0  # Draw/timeout

    outcome_str = str(outcome)
    match = re.search(r"Win\((\d+)\)", outcome_str)

    if match:
        winner_idx = int(match.group(1))
        return 1 if winner_idx == rl_player_idx else -1

    return 0  # Tie or unknown


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_benchmark(
    name: str,
    rl_decks: list[str],
    opponent_decks: list[str],
    model,
    total_games_per_bot: int,
    mirror: bool = False,
) -> dict[str, BenchmarkResult]:
    """Run a benchmark scenario against all opponent types.
    
    Args:
        total_games_per_bot: Target total games per opponent bot type.
    """
    results = {opp: BenchmarkResult() for opp in OPPONENT_TYPES}

    # Calculate games per matchup to reach target total
    num_matchups = len(rl_decks) * (1 if mirror else len(opponent_decks))
    games_per_matchup = max(1, total_games_per_bot // num_matchups)
    total_games = num_matchups * games_per_matchup * len(OPPONENT_TYPES)
    game_count = 0

    with console.status(f"[bold green]Running {name}...") as status:
        for rl_deck in rl_decks:
            opp_deck_list = [rl_deck] if mirror else opponent_decks

            for opp_deck in opp_deck_list:
                for opp_type in OPPONENT_TYPES:
                    for game_idx in range(games_per_matchup):
                        rl_is_first = game_idx % 2 == 0
                        seed = random.randint(0, 2**32 - 1)

                        try:
                            result, steps = play_game(
                                rl_deck, opp_deck, model,
                                rl_is_player_a=rl_is_first,
                                opponent_type=opp_type,
                                seed=seed,
                            )
                            _record_result(results[opp_type], result, steps)
                        except Exception:
                            results[opp_type].draws += 1

                        game_count += 1
                        if game_count % 100 == 0:
                            pct = game_count / total_games * 100
                            status.update(f"[bold green]{name}: {game_count}/{total_games} ({pct:.0f}%)")

    return results


def _record_result(result: BenchmarkResult, outcome: int, steps: int):
    """Record a single game result."""
    if outcome == 1:
        result.wins += 1
    elif outcome == -1:
        result.losses += 1
    else:
        result.draws += 1
    result.total_steps += steps


# =============================================================================
# Deck Selection
# =============================================================================

def sample_decks_by_winrate(loader: MetaDeckLoader, n: int, high_wr: bool) -> list[str]:
    """Sample decks from high or low win rate archetypes."""
    if high_wr:
        decks_info = loader.filter_by_win_rate(min_rate=55.0, max_rate=100.0)
    else:
        decks_info = loader.filter_by_win_rate(min_rate=0.0, max_rate=45.0)

    if not decks_info:
        decks_info = loader.decks  # Fallback to all

    # Sample unique archetypes
    result = []
    used_archetypes = set()
    shuffled = list(decks_info)
    random.shuffle(shuffled)

    for deck_info in shuffled:
        if len(result) >= n:
            break
        if deck_info.archetype in used_archetypes:
            continue
        if deck_info.deck_string:
            result.append(deck_info.deck_string)
            used_archetypes.add(deck_info.archetype)

    return result


def sample_diverse_meta_decks(loader: MetaDeckLoader, n: int) -> list[str]:
    """Sample meta decks from diverse archetypes."""
    decks = []
    used = set()

    for _ in range(n * 4):  # Over-sample to ensure diversity
        if len(decks) >= n:
            break
        info = loader.sample_deck_info()
        if info.archetype not in used:
            decks.append(info.deck_string)
            used.add(info.archetype)

    return decks


# =============================================================================
# Output Formatting
# =============================================================================

def print_results_table(name: str, results: dict[str, BenchmarkResult]):
    """Display results as a formatted table."""
    table = Table(title=f"{name} Results")
    table.add_column("Opponent", style="cyan")
    table.add_column("Wins", justify="right", style="green")
    table.add_column("Losses", justify="right", style="red")
    table.add_column("Draws", justify="right", style="yellow")
    table.add_column("Win Rate", justify="right", style="bold")

    for opp in OPPONENT_TYPES:
        r = results[opp]
        wr_style = "green" if r.win_rate >= 0.5 else "red"
        table.add_row(
            OPPONENT_NAMES.get(opp, opp),
            str(r.wins),
            str(r.losses),
            str(r.draws),
            f"[{wr_style}]{r.win_rate:.1%}[/{wr_style}]",
        )

    console.print(table)


def print_summary(all_results: dict):
    """Print overall summary across all benchmarks."""
    console.print("\n[bold green]═══ Summary ═══[/bold green]")
    for scenario, results in all_results.items():
        total_w = sum(r["wins"] for r in results.values())
        total_g = sum(r["wins"] + r["losses"] + r["draws"] for r in results.values())
        wr = total_w / total_g if total_g else 0
        console.print(f"  {scenario.capitalize():8s}: {total_w:4d}/{total_g:4d} ({wr:.1%})")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    console.print("[bold blue]═══ Pokemon TCG Pocket RL Evaluation ═══[/bold blue]")
    console.print(f"Model: {args.model}")
    console.print(f"Games per bot: {args.games}")

    # Load resources
    console.print("\n[1/2] Loading model...")
    model = load_model(args.model)

    console.print("[2/2] Loading decks...")
    simple_loader = MetaDeckLoader(args.simple_decks)
    meta_loader = MetaDeckLoader(args.meta_decks)

    all_results = {}

    # Run selected benchmarks
    if args.benchmark in ["all", "mirror"]:
        all_results["mirror"] = _run_mirror_benchmark(
            simple_loader, meta_loader, model, args.games
        )

    if args.benchmark in ["all", "easy"]:
        all_results["easy"] = _run_winrate_benchmark(
            "Easy", meta_loader, model, args.games,
            rl_high_wr=True, opp_high_wr=False
        )

    if args.benchmark in ["all", "hard"]:
        all_results["hard"] = _run_winrate_benchmark(
            "Hard", meta_loader, model, args.games,
            rl_high_wr=False, opp_high_wr=True
        )

    if args.benchmark in ["all", "random"]:
        all_results["random"] = _run_random_benchmark(
            meta_loader, model, args.games
        )

    # Output
    print_summary(all_results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        console.print(f"\n[green]Results saved to {args.output}[/green]")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate RL agent")
    parser.add_argument("--model", type=str, required=True, help="Path to model (.zip)")
    parser.add_argument("--simple-decks", type=str, default="simple_deck.json")
    parser.add_argument("--meta-decks", type=str, default="meta_deck.json")
    parser.add_argument("--games", type=int, default=1000, help="Total games per bot type")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["all", "mirror", "easy", "hard", "random"],
        default="all",
        help="Which benchmark to run",
    )
    return parser.parse_args()


# =============================================================================
# Benchmark Implementations
# =============================================================================

def _run_mirror_benchmark(simple_loader, meta_loader, model, total_games) -> dict:
    """Run mirror matchup benchmark."""
    console.print("\n[bold cyan]═══ Mirror Matchup ═══[/bold cyan]")
    console.print("Same deck for both players (3 simple + 7 meta archetypes)")

    simple_decks = [simple_loader.sample_deck() for _ in range(3)]
    meta_decks = sample_diverse_meta_decks(meta_loader, 7)

    results = run_benchmark(
        "Mirror", simple_decks + meta_decks, [], model, total_games, mirror=True
    )
    print_results_table("Mirror Matchup", results)
    return {k: vars(v) for k, v in results.items()}


def _run_winrate_benchmark(
    name: str, loader, model, total_games, rl_high_wr: bool, opp_high_wr: bool
) -> dict:
    """Run win-rate based benchmark (easy or hard)."""
    console.print(f"\n[bold cyan]═══ {name} Matchup ═══[/bold cyan]")
    rl_desc = "high WR" if rl_high_wr else "low WR"
    opp_desc = "high WR" if opp_high_wr else "low WR"
    console.print(f"RL uses {rl_desc} decks vs opponent {opp_desc} decks")

    # Use fewer decks to keep total games manageable (5x5 = 25 matchups)
    rl_decks = sample_decks_by_winrate(loader, 5, high_wr=rl_high_wr)
    opp_decks = sample_decks_by_winrate(loader, 5, high_wr=opp_high_wr)
    console.print(f"  RL decks: {len(rl_decks)}, Opponent decks: {len(opp_decks)}")

    results = run_benchmark(name, rl_decks, opp_decks, model, total_games)
    print_results_table(f"{name} Matchup", results)
    return {k: vars(v) for k, v in results.items()}


def _run_random_benchmark(loader, model, total_games) -> dict:
    """Run random matchup benchmark."""
    console.print("\n[bold cyan]═══ Random Matchup ═══[/bold cyan]")
    console.print("Random meta decks for both players")

    # Use fewer decks (5x5 = 25 matchups) 
    rl_decks = [loader.sample_deck() for _ in range(5)]
    opp_decks = [loader.sample_deck() for _ in range(5)]

    results = run_benchmark("Random", rl_decks, opp_decks, model, total_games)
    print_results_table("Random Matchup", results)
    return {k: vars(v) for k, v in results.items()}


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
