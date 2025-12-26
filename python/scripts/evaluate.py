#!/usr/bin/env python3
"""
Incremental Elo Evaluation System for Pokemon TCG Pocket RL.

Features:
- Initialize baseline bot ratings with round-robin tournament
- Evaluate a new agent against baselines and compute its Elo
- Store Elo in model metadata for RL checkpoints
"""

import argparse
import itertools
import json
import os
import random
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from deckgym.deck_loader import MetaDeckLoader


# =============================================================================
# Configuration
# =============================================================================

INITIAL_ELO = 1500.0
K_FACTOR_NEW = 40      # K for players with < 30 games
K_FACTOR_MID = 24      # K for players with 30-100 games
K_FACTOR_STABLE = 16   # K for players with > 100 games

# Baseline bots for evaluation
BASELINE_BOTS = [
    ("Random", "r"),
    ("AttachAttack", "aa"),
    ("EndTurn", "et"),
    ("WeightedRandom", "w"),
    ("ValueFunction", "v"),
    ("EvolutionRusher", "er"),
    ("Expectiminimax(2)", "e2"),
    ("Expectiminimax(3)", "e3"),
    ("Expectiminimax(4)", "e4"),
    ("Expectiminimax(5)", "e5"),
]

ELO_CACHE_PATH = Path("elo_baselines.json")

console = Console()


# =============================================================================
# Elo System
# =============================================================================

@dataclass
class EloPlayer:
    """A player in the Elo rating system."""
    name: str
    bot_code: str  # Code for game creation (e.g., "r", "e2")
    elo: float = INITIAL_ELO
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    is_rl: bool = False  # True for RL agents
    
    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses + self.draws
        return self.wins / total if total > 0 else 0.0
    
    def get_k_factor(self) -> float:
        """Adaptive K-factor: higher for new players, lower for established."""
        if self.games < 30:
            return K_FACTOR_NEW
        elif self.games < 100:
            return K_FACTOR_MID
        return K_FACTOR_STABLE
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "EloPlayer":
        return cls(**d)


def expected_score(elo_a: float, elo_b: float) -> float:
    """Calculate expected score (probability of winning) for player A."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def update_elo(player_a: EloPlayer, player_b: EloPlayer, winner: Optional[EloPlayer]):
    """Update Elo ratings after a match. Winner is None for draw."""
    expected_a = expected_score(player_a.elo, player_b.elo)
    expected_b = expected_score(player_b.elo, player_a.elo)
    
    k_a = player_a.get_k_factor()
    k_b = player_b.get_k_factor()
    
    if winner is None:
        # Draw
        player_a.elo += k_a * (0.5 - expected_a)
        player_b.elo += k_b * (0.5 - expected_b)
        player_a.draws += 1
        player_b.draws += 1
    elif winner.name == player_a.name:
        player_a.elo += k_a * (1.0 - expected_a)
        player_b.elo += k_b * (0.0 - expected_b)
        player_a.wins += 1
        player_b.losses += 1
    else:
        player_a.elo += k_a * (0.0 - expected_a)
        player_b.elo += k_b * (1.0 - expected_b)
        player_a.losses += 1
        player_b.wins += 1
    
    player_a.games += 1
    player_b.games += 1


# =============================================================================
# Game Execution
# =============================================================================

def play_match(
    player_a: EloPlayer,
    player_b: EloPlayer,
    deck_a: str,
    deck_b: str,
    rl_model=None,
    seed: Optional[int] = None,
) -> Optional[EloPlayer]:
    """
    Play a single match between two players.
    
    Returns:
        Winner EloPlayer, or None for draw
    """
    import deckgym
    
    deck_a_path, deck_b_path = _write_temp_decks(deck_a, deck_b)
    
    try:
        # If player is RL, use "r" as placeholder (we'll control manually)
        bot_a = "r" if player_a.is_rl else player_a.bot_code
        bot_b = "r" if player_b.is_rl else player_b.bot_code
        
        game = deckgym.Game(deck_a_path, deck_b_path, players=[bot_a, bot_b], seed=seed)
        
        max_steps = 500
        steps = 0
        
        while not game.is_game_over() and steps < max_steps:
            current = game.current_player()
            current_player = player_a if current == 0 else player_b
            
            if current_player.is_rl:
                # RL agent plays
                _execute_rl_turn(game, rl_model)
            else:
                # Bot plays
                game.play_tick()
            
            steps += 1
        
        # Determine winner
        if steps >= max_steps and not game.is_game_over():
            return None  # Draw (timeout)
        
        outcome = game.get_state().winner
        if outcome is None:
            return None  # Draw
        
        outcome_str = str(outcome)
        match = re.search(r"Win\((\d+)\)", outcome_str)
        if match:
            winner_idx = int(match.group(1))
            return player_a if winner_idx == 0 else player_b
        
        return None
        
    except BaseException:
        return None  # Draw (exception)
    finally:
        os.unlink(deck_a_path)
        os.unlink(deck_b_path)


def _execute_rl_turn(game, model):
    """Execute a single RL agent turn."""
    obs = np.array(game.get_obs(), dtype=np.float32)
    mask = np.array(game.get_action_mask(), dtype=bool)
    
    if not mask.any():
        mask[0] = True
    
    if model is None:
        valid = np.where(mask)[0]
        action = int(np.random.choice(valid)) if len(valid) > 0 else 0
    else:
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        action = int(action)
    
    try:
        game.step_action(action)
    except (ValueError, Exception):
        valid = [i for i, v in enumerate(mask) if v]
        game.step_action(valid[0] if valid else 0)


def _write_temp_decks(deck_a: str, deck_b: str) -> tuple[str, str]:
    """Write deck strings to temporary files."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(deck_a)
        path_a = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(deck_b)
        path_b = f.name
    return path_a, path_b


# =============================================================================
# Quick Evaluation (for training curriculum)
# =============================================================================

def quick_eval_vs_bot(
    model,
    bot_code: str,
    deck_loader: MetaDeckLoader,
    n_games: int = 50,
    verbose: bool = False,
) -> float:
    """
    Quick win rate evaluation against a specific bot.
    
    Designed for curriculum training - fast evaluation without Elo calculation.
    Uses the existing play_match infrastructure.
    
    Args:
        model: Trained RL model (MaskablePPO)
        bot_code: Bot code to play against ("e2", "e3", "v", etc.)
        deck_loader: Deck loader for sampling games
        n_games: Number of games to play (default: 50)
        verbose: Print progress
        
    Returns:
        Win rate as float (0.0 - 1.0)
    """
    wins = 0
    losses = 0
    draws = 0
    
    # Create player objects
    rl_player = EloPlayer(name="agent", bot_code="rl", is_rl=True)
    bot_player = EloPlayer(name=bot_code, bot_code=bot_code)
    
    for i in range(n_games):
        deck_a = deck_loader.sample_deck()
        deck_b = deck_loader.sample_deck()
        seed = random.randint(0, 2**32 - 1)
        
        # Alternate starting player
        if i % 2 == 0:
            winner = play_match(rl_player, bot_player, deck_a, deck_b, 
                               rl_model=model, seed=seed)
            if winner is None:
                draws += 1
            elif winner.name == "agent":
                wins += 1
            else:
                losses += 1
        else:
            winner = play_match(bot_player, rl_player, deck_b, deck_a,
                               rl_model=model, seed=seed)
            if winner is None:
                draws += 1
            elif winner.name == "agent":
                wins += 1
            else:
                losses += 1
        
        if verbose and (i + 1) % 10 == 0:
            wr = wins / (i + 1)
            console.print(f"  [{i+1}/{n_games}] vs {bot_code}: {wins}W-{losses}L-{draws}D ({wr:.1%})")
    
    total = wins + losses + draws
    win_rate = wins / total if total > 0 else 0.0
    
    if verbose:
        console.print(f"  Final: {wins}W-{losses}L-{draws}D ({win_rate:.1%} WR)")
    
    return win_rate


# =============================================================================
# Baseline Initialization
# =============================================================================

def _play_baseline_game(args: tuple) -> tuple:
    """
    Worker function for parallel baseline game execution.
    
    Args:
        args: (bot_code_a, bot_code_b, deck_a, deck_b, seed, swap_order)
    
    Returns:
        (bot_code_a, bot_code_b, winner_code_or_none)
    """
    import deckgym
    
    bot_code_a, bot_code_b, deck_a, deck_b, seed, swap_order = args
    
    # Swap order if needed
    if swap_order:
        bot_code_a, bot_code_b = bot_code_b, bot_code_a
        deck_a, deck_b = deck_b, deck_a
    
    deck_a_path, deck_b_path = _write_temp_decks(deck_a, deck_b)
    
    try:
        game = deckgym.Game(deck_a_path, deck_b_path, players=[bot_code_a, bot_code_b], seed=seed)
        
        max_steps = 500
        steps = 0
        
        while not game.is_game_over() and steps < max_steps:
            game.play_tick()
            steps += 1
        
        if steps >= max_steps and not game.is_game_over():
            winner_code = None  # Draw
        else:
            outcome = game.get_state().winner
            if outcome is None:
                winner_code = None
            else:
                outcome_str = str(outcome)
                match = re.search(r"Win\((\d+)\)", outcome_str)
                if match:
                    winner_idx = int(match.group(1))
                    winner_code = bot_code_a if winner_idx == 0 else bot_code_b
                else:
                    winner_code = None
        
        # Return original order (before swap)
        if swap_order:
            return (bot_code_b, bot_code_a, winner_code)
        return (bot_code_a, bot_code_b, winner_code)
        
    except BaseException:
        if swap_order:
            return (bot_code_b, bot_code_a, None)
        return (bot_code_a, bot_code_b, None)
    finally:
        os.unlink(deck_a_path)
        os.unlink(deck_b_path)


def initialize_baselines(
    deck_loader: MetaDeckLoader,
    games_per_matchup: int = 10,
    save_path: Path = ELO_CACHE_PATH,
    n_workers: int = 8,
) -> Dict[str, EloPlayer]:
    """
    Initialize baseline bot ratings via round-robin tournament.
    
    Each baseline plays against every other baseline.
    Games are run in parallel for speedup.
    Results are saved to a cache file for reuse.
    """
    console.print("[bold cyan]═══ Initializing Baseline Elo Ratings ═══[/bold cyan]")
    
    # Create baseline players
    players = {
        name: EloPlayer(name=name, bot_code=code)
        for name, code in BASELINE_BOTS
    }
    code_to_name = {code: name for name, code in BASELINE_BOTS}
    
    matchups = list(itertools.combinations(BASELINE_BOTS, 2))
    total_games = len(matchups) * games_per_matchup
    
    console.print(f"Running {total_games} games ({len(BASELINE_BOTS)} bots, {games_per_matchup} games/matchup, {n_workers} workers)")
    
    # Prepare all game arguments
    game_args = []
    for (name_a, code_a), (name_b, code_b) in matchups:
        for game_idx in range(games_per_matchup):
            deck_a = deck_loader.sample_deck()
            deck_b = deck_loader.sample_deck()
            seed = random.randint(0, 2**32 - 1)
            swap_order = game_idx % 2 == 1  # Alternate who goes first
            game_args.append((code_a, code_b, deck_a, deck_b, seed, swap_order))
    
    # Run games in parallel
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Baseline tournament", total=total_games)
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_play_baseline_game, args) for args in game_args]
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                progress.update(task, completed=len(results))
    
    # Update Elo ratings from results (sequential to maintain consistency)
    for code_a, code_b, winner_code in results:
        name_a = code_to_name[code_a]
        name_b = code_to_name[code_b]
        player_a = players[name_a]
        player_b = players[name_b]
        
        if winner_code is None:
            winner = None
        elif winner_code == code_a:
            winner = player_a
        else:
            winner = player_b
        
        update_elo(player_a, player_b, winner)
    
    # Save to cache
    cache_data = {name: player.to_dict() for name, player in players.items()}
    with open(save_path, "w") as f:
        json.dump(cache_data, f, indent=2)
    console.print(f"[green]Saved baseline ratings to {save_path}[/green]")
    
    # Print leaderboard
    print_leaderboard(list(players.values()))
    
    return players


def load_baselines(cache_path: Path = ELO_CACHE_PATH) -> Optional[Dict[str, EloPlayer]]:
    """Load baseline ratings from cache if available."""
    if not cache_path.exists():
        return None
    
    with open(cache_path) as f:
        cache_data = json.load(f)
    
    return {name: EloPlayer.from_dict(data) for name, data in cache_data.items()}


# =============================================================================
# Agent Evaluation
# =============================================================================

def evaluate_agent(
    model_path: str,
    baselines: Dict[str, EloPlayer],
    deck_loader: MetaDeckLoader,
    games_per_baseline: int = 20,
    save_elo_to_model: bool = True,
) -> EloPlayer:
    """
    Evaluate an RL agent against all baselines and compute its Elo.
    
    Args:
        model_path: Path to the RL model checkpoint
        baselines: Dictionary of baseline EloPlayers
        deck_loader: Deck loader for sampling game decks
        games_per_baseline: Games to play against each baseline
        save_elo_to_model: If True, append Elo to model filename
    
    Returns:
        EloPlayer for the evaluated agent
    """
    from sb3_contrib import MaskablePPO
    
    console.print(f"\n[bold cyan]═══ Evaluating Agent ═══[/bold cyan]")
    console.print(f"Model: {model_path}")
    console.print(f"Games per baseline: {games_per_baseline}")
    
    # Load model
    model = MaskablePPO.load(model_path)
    
    # Create RL player
    agent_name = Path(model_path).stem
    rl_player = EloPlayer(name=agent_name, bot_code="rl", is_rl=True)
    
    total_games = len(baselines) * games_per_baseline
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating agent", total=total_games)
        
        game_count = 0
        for baseline_name, baseline in baselines.items():
            # Create a copy of baseline for this evaluation (don't modify cached baselines)
            baseline_copy = EloPlayer.from_dict(baseline.to_dict())
            
            for game_idx in range(games_per_baseline):
                deck_a = deck_loader.sample_deck()
                deck_b = deck_loader.sample_deck()
                seed = random.randint(0, 2**32 - 1)
                
                # Alternate who goes first
                if game_idx % 2 == 0:
                    winner = play_match(rl_player, baseline_copy, deck_a, deck_b, 
                                       rl_model=model, seed=seed)
                else:
                    winner = play_match(baseline_copy, rl_player, deck_b, deck_a,
                                       rl_model=model, seed=seed)
                
                update_elo(rl_player, baseline_copy, winner)
                
                game_count += 1
                progress.update(task, completed=game_count,
                               description=f"vs {baseline_name}: {game_count}/{total_games}")
    
    # Print results
    console.print(f"\n[bold green]Agent Elo: {rl_player.elo:.0f}[/bold green]")
    console.print(f"Record: {rl_player.wins}W - {rl_player.losses}L - {rl_player.draws}D")
    console.print(f"Win Rate: {rl_player.win_rate:.1%}")
    
    # Save Elo to model by renaming/copying
    if save_elo_to_model:
        _save_elo_to_model(model_path, rl_player.elo)
    
    # Print comparison leaderboard
    all_players = list(baselines.values()) + [rl_player]
    all_players.sort(key=lambda p: p.elo, reverse=True)
    print_leaderboard(all_players, highlight=agent_name)
    
    return rl_player


def _save_elo_to_model(model_path: str, elo: float):
    """Save Elo rating alongside the model."""
    elo_path = Path(model_path).with_suffix(".elo")
    with open(elo_path, "w") as f:
        json.dump({"elo": round(elo, 1), "model": model_path}, f)
    console.print(f"[green]Saved Elo to {elo_path}[/green]")


def load_model_elo(model_path: str) -> Optional[float]:
    """Load Elo rating for a model if available."""
    elo_path = Path(model_path).with_suffix(".elo")
    if elo_path.exists():
        with open(elo_path) as f:
            return json.load(f).get("elo")
    return None


# =============================================================================
# Output
# =============================================================================

def print_leaderboard(players: List[EloPlayer], highlight: str = None):
    """Display Elo leaderboard."""
    sorted_players = sorted(players, key=lambda p: p.elo, reverse=True)
    
    table = Table(title="Elo Leaderboard")
    table.add_column("Rank", justify="right", style="bold")
    table.add_column("Player", style="cyan")
    table.add_column("Elo", justify="right", style="bold yellow")
    table.add_column("Games", justify="right")
    table.add_column("W-L-D", justify="right")
    table.add_column("Win Rate", justify="right")
    
    for rank, player in enumerate(sorted_players, 1):
        elo_str = f"{player.elo:.0f}"
        wld = f"{player.wins}-{player.losses}-{player.draws}"
        wr = f"{player.win_rate:.1%}"
        wr_style = "green" if player.win_rate >= 0.5 else "red"
        
        # Highlight the evaluated agent
        name_style = "bold magenta" if player.name == highlight else "cyan"
        
        table.add_row(
            str(rank),
            f"[{name_style}]{player.name}[/{name_style}]",
            elo_str,
            str(player.games),
            wld,
            f"[{wr_style}]{wr}[/{wr_style}]",
        )
    
    console.print(table)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Elo Evaluation System")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # init-baselines command
    init_parser = subparsers.add_parser("init", help="Initialize baseline bot ratings")
    init_parser.add_argument("--meta-decks", default="meta_deck.json", help="Meta deck file")
    init_parser.add_argument("--games", type=int, default=20, help="Games per matchup")
    init_parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    init_parser.add_argument("--output", default="elo_baselines.json", help="Output cache file")
    
    # evaluate command
    eval_parser = subparsers.add_parser("eval", help="Evaluate an RL agent")
    eval_parser.add_argument("model", help="Path to model checkpoint (.zip)")
    eval_parser.add_argument("--meta-decks", default="meta_deck.json", help="Meta deck file")
    eval_parser.add_argument("--games", type=int, default=20, help="Games per baseline")
    eval_parser.add_argument("--baselines", default="elo_baselines.json", help="Baseline cache file")
    eval_parser.add_argument("--no-save-elo", action="store_true", help="Don't save Elo to model")
    
    args = parser.parse_args()
    
    if args.command == "init":
        deck_loader = MetaDeckLoader(args.meta_decks)
        initialize_baselines(deck_loader, args.games, Path(args.output), args.workers)
        
    elif args.command == "eval":
        deck_loader = MetaDeckLoader(args.meta_decks)
        
        # Load or initialize baselines
        baselines = load_baselines(Path(args.baselines))
        if baselines is None:
            console.print("[yellow]No baseline cache found. Initializing...[/yellow]")
            baselines = initialize_baselines(deck_loader, games_per_matchup=10)
        
        evaluate_agent(
            args.model,
            baselines,
            deck_loader,
            games_per_baseline=args.games,
            save_elo_to_model=not args.no_save_elo,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
