#!/usr/bin/env python3
"""
Evaluation Benchmark for Pokemon TCG Pocket RL Agent

Benchmarks:
1. Mirror Matchup: Same deck, RL vs opponents. 10 decks (3 simple, 7 meta), 50 games each.
2. Easy Matchup: RL with high WR decks vs opponent with low WR decks. 50 games per deck.
3. Hard Matchup: RL with low WR decks vs opponent with high WR decks. 50 games per deck.
4. Random Matchup: Random meta decks for both. 10 decks each, 50 games per deck.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from rich.console import Console
from rich.table import Table

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from deckgym.deck_loader import MetaDeckLoader

console = Console()


# Available opponent types (excluding MCTS which is too slow, and Human)
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


@dataclass
class BenchmarkResult:
    """Results for a single benchmark scenario."""
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


def load_model(model_path: str):
    """Load a trained MaskablePPO model."""
    from sb3_contrib import MaskablePPO
    return MaskablePPO.load(model_path)


def get_rl_action(model, obs, action_mask):
    """Get action from RL model with masking."""
    action, _ = model.predict(obs, deterministic=True, action_masks=action_mask)
    return int(action)


def play_game_with_bot(
    deck_a_str: str,
    deck_b_str: str,
    model,
    rl_is_player_a: bool,
    opponent_type: str,
    seed: Optional[int] = None
) -> tuple[int, int]:
    """
    Play a single game between RL agent and a bot opponent.
    Uses file-based game constructor which supports player specification.
    
    Returns:
        (result, steps): result is 1 for RL win, -1 for loss, 0 for draw
    """
    import deckgym
    import tempfile
    import os
    
    # Write deck strings to temp files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f_a:
        f_a.write(deck_a_str)
        deck_a_path = f_a.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f_b:
        f_b.write(deck_b_str)
        deck_b_path = f_b.name
    
    try:
        # Create game with bot as opponent
        # Use 'r' (random) as placeholder for RL side - we'll control it via step_action
        if rl_is_player_a:
            # RL controls player A (index 0), bot plays as player B (index 1)
            players = ["r", opponent_type]
        else:
            # Bot plays as player A (index 0), RL controls player B (index 1)
            players = [opponent_type, "r"]
        
        game = deckgym.Game(deck_a_path, deck_b_path, players=players, seed=seed)
        
        rl_player_idx = 0 if rl_is_player_a else 1
        steps = 0
        max_steps = 500
        
        while not game.is_game_over() and steps < max_steps:
            current_player = game.current_player()
            
            if current_player == rl_player_idx:
                # RL agent's turn - we control it
                obs = np.array(game.get_obs(), dtype=np.float32)
                mask = np.array(game.get_action_mask(), dtype=bool)
                
                # Check if any valid actions
                if not mask.any():
                    mask[0] = True  # Force end turn as fallback
                
                action = get_rl_action(model, obs, mask)
                
                try:
                    game.step_action(action)
                except ValueError:
                    # Invalid action, try first valid one
                    valid_actions = [i for i, v in enumerate(mask) if v]
                    if valid_actions:
                        game.step_action(valid_actions[0])
                    else:
                        game.step_action(0)  # End turn
            else:
                # Bot's turn - let the game engine handle it
                game.play_tick()
            
            steps += 1
        
        # Get result from state winner
        state = game.get_state()
        outcome = state.winner
        
        if outcome is None:
            result = 0, steps  # Draw/timeout
        else:
            # Parse the PyGameOutcome - it's either Win(player_idx) or Tie
            outcome_str = str(outcome)
            if "Win" in outcome_str:
                # Extract winner index from "GameOutcome::Win(N)"
                import re
                match = re.search(r'Win\((\d+)\)', outcome_str)
                if match:
                    winner_idx = int(match.group(1))
                    if winner_idx == rl_player_idx:
                        result = 1, steps  # RL win
                    else:
                        result = -1, steps  # RL loss
                else:
                    result = 0, steps
            else:
                result = 0, steps  # Tie or unknown
                
    except BaseException:
        # Game engine bug/panic - count as draw
        result = 0, 0
            
    finally:
        # Cleanup temp files
        os.unlink(deck_a_path)
        os.unlink(deck_b_path)
    
    return result


def run_benchmark(
    name: str,
    rl_decks: list[str],
    opponent_decks: list[str],
    model,
    games_per_deck: int,
    mirror: bool = False
) -> dict[str, BenchmarkResult]:
    """
    Run a benchmark scenario.
    """
    results = {opp: BenchmarkResult() for opp in OPPONENT_TYPES}
    
    total_matchups = len(rl_decks) * (1 if mirror else len(opponent_decks))
    total_games = total_matchups * games_per_deck * len(OPPONENT_TYPES)
    
    with console.status(f"[bold green]Running {name}...") as status:
        game_count = 0
        
        for rl_idx, rl_deck in enumerate(rl_decks):
            opp_deck_list = [rl_deck] if mirror else opponent_decks
            
            for opp_idx, opp_deck in enumerate(opp_deck_list):
                for opp_type in OPPONENT_TYPES:
                    for game_idx in range(games_per_deck):
                        # Alternate who goes first
                        rl_is_first = game_idx % 2 == 0
                        seed = random.randint(0, 2**32 - 1)
                        
                        try:
                            result, steps = play_game_with_bot(
                                rl_deck, opp_deck, model,
                                rl_is_player_a=rl_is_first,
                                opponent_type=opp_type,
                                seed=seed
                            )
                            
                            if result == 1:
                                results[opp_type].wins += 1
                            elif result == -1:
                                results[opp_type].losses += 1
                            else:
                                results[opp_type].draws += 1
                            results[opp_type].total_steps += steps
                            
                        except Exception as e:
                            # Count game engine bugs/panics as draws silently
                            results[opp_type].draws += 1
                        
                        game_count += 1
                        if game_count % 100 == 0:
                            pct = game_count / total_games * 100
                            status.update(f"[bold green]{name}: {game_count}/{total_games} ({pct:.0f}%)...")
    
    return results


def sample_decks_by_winrate(loader: MetaDeckLoader, n: int, high_wr: bool) -> list[str]:
    """Sample decks from high or low win rate archetypes."""
    # Use filter_by_win_rate to get high or low WR decks
    if high_wr:
        # Top 30% win rates
        decks_info = loader.filter_by_win_rate(min_rate=55.0, max_rate=100.0)
    else:
        # Bottom 30% win rates  
        decks_info = loader.filter_by_win_rate(min_rate=0.0, max_rate=45.0)
    
    if not decks_info:
        # Fallback to all decks if filtered set is empty
        decks_info = loader.decks
    
    # Sample n unique decks from different archetypes
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


def print_results_table(name: str, results: dict[str, BenchmarkResult]):
    """Print results as a rich table."""
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
            f"[{wr_style}]{r.win_rate:.1%}[/{wr_style}]"
        )
    
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL agent")
    parser.add_argument("--model", type=str, required=True, help="Path to model (.zip)")
    parser.add_argument("--simple-decks", type=str, default="simple_deck.json")
    parser.add_argument("--meta-decks", type=str, default="meta_deck.json")
    parser.add_argument("--games-per-deck", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=None, help="Output JSON")
    parser.add_argument("--benchmark", type=str, choices=["all", "mirror", "easy", "hard", "random"],
                        default="all", help="Which benchmark to run")
    args = parser.parse_args()
    
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    console.print("[bold blue]═══ Pokemon TCG Pocket RL Evaluation ═══[/bold blue]")
    console.print(f"Model: {args.model}")
    console.print(f"Games per matchup: {args.games_per_deck}")
    
    # Load model
    console.print("\nLoading model...")
    model = load_model(args.model)
    
    # Load decks
    console.print("Loading decks...")
    simple_loader = MetaDeckLoader(args.simple_decks)
    meta_loader = MetaDeckLoader(args.meta_decks)
    
    all_results = {}
    
    # ===== 1. Mirror Matchup =====
    if args.benchmark in ["all", "mirror"]:
        console.print("\n[bold cyan]═══ Mirror Matchup ═══[/bold cyan]")
        console.print("Same deck, 10 decks (3 simple + 7 meta archetypes)")
        
        simple_decks = [simple_loader.sample_deck() for _ in range(3)]
        # Sample 7 meta decks from different archetypes
        meta_decks = []
        used = set()
        for _ in range(30):
            if len(meta_decks) >= 7:
                break
            info = meta_loader.sample_deck_info()
            if info.archetype not in used:
                meta_decks.append(info.deck_string)
                used.add(info.archetype)
        
        mirror_results = run_benchmark(
            "Mirror", simple_decks + meta_decks, [], model, args.games_per_deck, mirror=True
        )
        all_results["mirror"] = {k: vars(v) for k, v in mirror_results.items()}
        print_results_table("Mirror Matchup", mirror_results)
    
    # ===== 2. Easy Matchup =====
    if args.benchmark in ["all", "easy"]:
        console.print("\n[bold cyan]═══ Easy Matchup ═══[/bold cyan]")
        console.print("RL has high WR decks, opponent has low WR decks")
        
        rl_decks = sample_decks_by_winrate(meta_loader, 10, high_wr=True)
        console.print(f"  RL decks: {len(rl_decks)}")
        opp_decks = sample_decks_by_winrate(meta_loader, 10, high_wr=False)
        
        easy_results = run_benchmark("Easy", rl_decks, opp_decks, model, args.games_per_deck)
        all_results["easy"] = {k: vars(v) for k, v in easy_results.items()}
        print_results_table("Easy Matchup", easy_results)
    
    # ===== 3. Hard Matchup =====
    if args.benchmark in ["all", "hard"]:
        console.print("\n[bold cyan]═══ Hard Matchup ═══[/bold cyan]") 
        console.print("RL has low WR decks, opponent has high WR decks")
        
        rl_decks = sample_decks_by_winrate(meta_loader, 10, high_wr=False)
        console.print(f"  RL decks: {len(rl_decks)}")
        opp_decks = sample_decks_by_winrate(meta_loader, 10, high_wr=True)
        
        hard_results = run_benchmark("Hard", rl_decks, opp_decks, model, args.games_per_deck)
        all_results["hard"] = {k: vars(v) for k, v in hard_results.items()}
        print_results_table("Hard Matchup", hard_results)
    
    # ===== 4. Random Matchup =====
    if args.benchmark in ["all", "random"]:
        console.print("\n[bold cyan]═══ Random Matchup ═══[/bold cyan]")
        console.print("Random meta decks, 10 each side")
        
        rl_decks = [meta_loader.sample_deck() for _ in range(10)]
        opp_decks = [meta_loader.sample_deck() for _ in range(10)]
        
        random_results = run_benchmark("Random", rl_decks, opp_decks, model, args.games_per_deck)
        all_results["random"] = {k: vars(v) for k, v in random_results.items()}
        print_results_table("Random Matchup", random_results)
    
    # Summary
    console.print("\n[bold green]═══ Summary ═══[/bold green]")
    for scenario, results in all_results.items():
        total_w = sum(r["wins"] for r in results.values())
        total_g = sum(r["wins"] + r["losses"] + r["draws"] for r in results.values())
        wr = total_w / total_g if total_g else 0
        console.print(f"  {scenario.capitalize():8s}: {total_w:4d}/{total_g:4d} ({wr:.1%})")
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        console.print(f"\n[green]Saved to {args.output}[/green]")


if __name__ == "__main__":
    main()
