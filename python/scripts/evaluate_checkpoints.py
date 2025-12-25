#!/usr/bin/env python3
"""
Batch Checkpoint Evaluation Script.

Evaluates all checkpoints in a directory and logs Elo progression.
Outputs a summary table and CSV for visualization.
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()


@dataclass
class CheckpointResult:
    path: str
    steps: int
    elo: float
    win_rate: float
    wins: int
    losses: int
    draws: int


def extract_steps(filename: str) -> Optional[int]:
    """Extract step count from checkpoint filename."""
    match = re.search(r"(\d+)_steps", filename)
    return int(match.group(1)) if match else None


def evaluate_all_checkpoints(
    checkpoint_dir: str,
    games_per_baseline: int = 50,
    output_csv: str = "checkpoint_elo.csv",
    filter_pattern: str = None,
) -> list[CheckpointResult]:
    """Evaluate all checkpoints and return results sorted by steps."""
    
    # Import here to avoid slow startup
    from evaluate import (
        EloPlayer,
        load_baselines,
        initialize_baselines,
        play_match,
        update_elo,
        BASELINE_BOTS,
    )
    from deckgym.deck_loader import MetaDeckLoader
    from sb3_contrib import MaskablePPO
    import random
    import numpy as np
    
    # Find all checkpoints
    checkpoint_path = Path(checkpoint_dir)
    checkpoints = list(checkpoint_path.glob("*.zip"))
    
    if filter_pattern:
        checkpoints = [c for c in checkpoints if filter_pattern in c.name]
    
    # Sort by steps
    checkpoints_with_steps = []
    for cp in checkpoints:
        steps = extract_steps(cp.name)
        if steps is not None:
            checkpoints_with_steps.append((cp, steps))
    
    checkpoints_with_steps.sort(key=lambda x: x[1])
    
    if not checkpoints_with_steps:
        console.print("[red]No valid checkpoints found![/red]")
        return []
    
    console.print(f"[cyan]Found {len(checkpoints_with_steps)} checkpoints[/cyan]")
    
    # Load baselines
    deck_loader = MetaDeckLoader("meta_deck.json")
    baselines = load_baselines()
    if baselines is None:
        console.print("[yellow]No baseline cache. Initializing...[/yellow]")
        baselines = initialize_baselines(deck_loader, games_per_matchup=20)
    
    code_to_name = {code: name for name, code in BASELINE_BOTS}
    
    results = []
    
    for idx, (checkpoint_path, steps) in enumerate(checkpoints_with_steps):
        console.print(f"\n[bold cyan]({idx+1}/{len(checkpoints_with_steps)}) Evaluating {checkpoint_path.name}[/bold cyan]")
        
        # Check if already evaluated
        elo_file = checkpoint_path.with_suffix(".elo")
        if elo_file.exists():
            with open(elo_file) as f:
                data = json.load(f)
                console.print(f"  [green]Already evaluated: {data['elo']:.0f} Elo[/green]")
                results.append(CheckpointResult(
                    path=str(checkpoint_path),
                    steps=steps,
                    elo=data["elo"],
                    win_rate=data.get("win_rate", 0),
                    wins=data.get("wins", 0),
                    losses=data.get("losses", 0),
                    draws=data.get("draws", 0),
                ))
                continue
        
        # Load model
        try:
            model = MaskablePPO.load(str(checkpoint_path))
        except Exception as e:
            console.print(f"  [red]Failed to load: {e}[/red]")
            continue
        
        # Evaluate against baselines
        rl_player = EloPlayer(name=checkpoint_path.stem, bot_code="rl", is_rl=True)
        
        for baseline_name, baseline in baselines.items():
            baseline_copy = EloPlayer.from_dict(baseline.to_dict())
            
            for game_idx in range(games_per_baseline):
                deck_a = deck_loader.sample_deck()
                deck_b = deck_loader.sample_deck()
                seed = random.randint(0, 2**32 - 1)
                
                if game_idx % 2 == 0:
                    winner = play_match(rl_player, baseline_copy, deck_a, deck_b, 
                                       rl_model=model, seed=seed)
                else:
                    winner = play_match(baseline_copy, rl_player, deck_b, deck_a,
                                       rl_model=model, seed=seed)
                
                update_elo(rl_player, baseline_copy, winner)
        
        # Save result
        result = CheckpointResult(
            path=str(checkpoint_path),
            steps=steps,
            elo=rl_player.elo,
            win_rate=rl_player.win_rate,
            wins=rl_player.wins,
            losses=rl_player.losses,
            draws=rl_player.draws,
        )
        results.append(result)
        
        # Save .elo file
        with open(elo_file, "w") as f:
            json.dump({
                "elo": round(rl_player.elo, 1),
                "win_rate": round(rl_player.win_rate, 3),
                "wins": rl_player.wins,
                "losses": rl_player.losses,
                "draws": rl_player.draws,
                "model": str(checkpoint_path),
            }, f)
        
        console.print(f"  [green]Elo: {rl_player.elo:.0f} | WR: {rl_player.win_rate:.1%}[/green]")
    
    # Save CSV
    with open(output_csv, "w") as f:
        f.write("steps,elo,win_rate,wins,losses,draws,path\n")
        for r in results:
            f.write(f"{r.steps},{r.elo:.1f},{r.win_rate:.3f},{r.wins},{r.losses},{r.draws},{r.path}\n")
    
    console.print(f"\n[green]Saved results to {output_csv}[/green]")
    
    return results


def print_summary(results: list[CheckpointResult]):
    """Print summary table of all results."""
    if not results:
        return
    
    table = Table(title="Checkpoint Elo Progression")
    table.add_column("Steps", justify="right", style="cyan")
    table.add_column("Elo", justify="right", style="bold yellow")
    table.add_column("Δ Elo", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("Record", justify="right")
    
    prev_elo = None
    best_elo = max(r.elo for r in results)
    
    for r in results:
        delta = ""
        delta_style = ""
        if prev_elo is not None:
            d = r.elo - prev_elo
            delta = f"{d:+.0f}"
            delta_style = "green" if d > 0 else "red" if d < 0 else ""
        prev_elo = r.elo
        
        steps_str = f"{r.steps/1e6:.1f}M"
        elo_str = f"{r.elo:.0f}"
        wr_str = f"{r.win_rate:.1%}"
        record = f"{r.wins}-{r.losses}-{r.draws}"
        
        # Highlight best
        elo_style = "bold green" if r.elo == best_elo else "yellow"
        
        table.add_row(
            steps_str,
            f"[{elo_style}]{elo_str}[/{elo_style}]",
            f"[{delta_style}]{delta}[/{delta_style}]" if delta else "",
            wr_str,
            record,
        )
    
    console.print(table)
    
    # Find best
    best = max(results, key=lambda r: r.elo)
    console.print(f"\n[bold green]Best checkpoint: {Path(best.path).name}[/bold green]")
    console.print(f"  Steps: {best.steps/1e6:.1f}M | Elo: {best.elo:.0f} | WR: {best.win_rate:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints")
    parser.add_argument("--dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--games", type=int, default=50, help="Games per baseline")
    parser.add_argument("--output", default="checkpoint_elo.csv", help="Output CSV")
    parser.add_argument("--filter", default=None, help="Filter pattern for checkpoint names")
    
    args = parser.parse_args()
    
    results = evaluate_all_checkpoints(
        args.dir,
        games_per_baseline=args.games,
        output_csv=args.output,
        filter_pattern=args.filter,
    )
    
    print_summary(results)


if __name__ == "__main__":
    main()
