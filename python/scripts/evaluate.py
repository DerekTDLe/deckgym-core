#!/usr/bin/env python3
"""
Unified Evaluation Script for DeckGym using TrueSkill and Rust-Native Execution.

Modes:
- calibrate: Run baselines against each other to establish ground truth ratings.
- audit: Evaluate a single model (converts to ONNX + runs in Rust).
- bench: Evaluate a directory of checkpoints.
"""

import argparse
import json
import math
import os
import shutil
import tempfile
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import trueskill
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import deckgym
from deckgym.deck_loader import MetaDeckLoader

# Try importing ONNX export tools (required for audit/bench)
try:
    from deckgym.onnx_export import export_policy_to_onnx
    from sb3_contrib import MaskablePPO
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


console = Console()

def print_system_info():
    """Print hardware and software information related to acceleration."""
    console.print(f"[bold]System Info:[/bold]")
    try:
        import torch
        console.print(f"  PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    except ImportError:
        console.print("  PyTorch: Not installed")
        
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        console.print(f"  ONNX Runtime: {ort.__version__}")
        console.print(f"  Available Providers: [cyan]{', '.join(providers)}[/cyan]")
        if "TensorrtExecutionProvider" in providers:
            console.print("  [green]✓ TensorRT Execution Provider is available![/green]")
        else:
            console.print("  [yellow]⚠ TensorRT Execution Provider not found.[/yellow]")
    except ImportError:
        console.print("  ONNX Runtime: Not installed")
    console.print("─" * 40)

# --- Configuration ---
TRUESKILL_ENV = trueskill.TrueSkill(
    mu=1500,              
    sigma=500,            # Initial uncertainty (large for new agents)
    beta=250,             # Performance variance (sigma / 2)
    tau=5,                # Dynamic factor
    draw_probability=0.02 # Probability of draw
)
BASELINES_FILE = Path("trueskill_baselines.json")
DEFAULT_BASELINES_TO_CALIBRATE = [
    "r", "aa", "et", "w", "v", "er", "e2", "e3", "e4", "e5"
]
BOT_NAMES = {
    "r": "Random",
    "aa": "AttachAttack",
    "et": "EndTurn",
    "w": "WeightedRandom",
    "v": "ValueFunction",
    "er": "EvolutionRusher",
    "e2": "Expectiminimax(2)",
    "e3": "Expectiminimax(3)",
    "e4": "Expectiminimax(4)",
    "e5": "Expectiminimax(5)",
}

@dataclass
class Rating:
    mu: float
    sigma: float
    
    @property
    def expose(self) -> float:
        return self.mu - 3 * self.sigma

    def to_trueskill(self) -> trueskill.Rating:
        return trueskill.Rating(mu=self.mu, sigma=self.sigma)
    
    @staticmethod
    def from_trueskill(r: trueskill.Rating) -> 'Rating':
        return Rating(mu=r.mu, sigma=r.sigma)

import threading

class TrueSkillTracker:
    def __init__(self, storage_file: Path = BASELINES_FILE):
        self.storage_file = storage_file
        self.ratings: Dict[str, Rating] = {}
        self.lock = threading.Lock()
        self.load()

    def load(self):
        if self.storage_file.exists():
            with open(self.storage_file) as f:
                data = json.load(f)
                for name, r in data.items():
                    self.ratings[name] = Rating(mu=r["mu"], sigma=r["sigma"])
        
        # Initialize missing defaults
        for code in BOT_NAMES:
            if code not in self.ratings:
                self.ratings[code] = Rating.from_trueskill(TRUESKILL_ENV.create_rating())

    def save(self):
        data = {name: {"mu": r.mu, "sigma": r.sigma} for name, r in self.ratings.items()}
        with open(self.storage_file, "w") as f:
            json.dump(data, f, indent=2)

    def update(self, winner_code: str, loser_code: str, draw: bool = False):
        with self.lock:
            r_winner = self.get(winner_code).to_trueskill()
            r_loser = self.get(loser_code).to_trueskill()
            
            if draw:
                new_w, new_l = TRUESKILL_ENV.rate_1vs1(r_winner, r_loser, drawn=True)
            else:
                new_w, new_l = TRUESKILL_ENV.rate_1vs1(r_winner, r_loser)
            
            self.ratings[winner_code] = Rating.from_trueskill(new_w)
            self.ratings[loser_code] = Rating.from_trueskill(new_l)

    def get(self, code: str) -> Rating:
        return self.ratings.get(code, Rating.from_trueskill(TRUESKILL_ENV.create_rating()))


def run_rust_match(
    deck_a_json: str, 
    deck_b_json: str, 
    player_a_code: str, 
    player_b_code: str, 
    seed: int = None
) -> int:
    """
    Run a single match in Rust. 
    Requires writing decks to temp files because py_simulate takes paths.
    Returns: 1 (A wins), -1 (B wins), 0 (Draw)
    """
    if seed is None:
        import random
        seed = random.randint(0, 2**64 - 1)
        
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=True) as f_a, \
         tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=True) as f_b:
        
        f_a.write(deck_a_json)
        f_a.flush()
        f_b.write(deck_b_json)
        f_b.flush()
        
        try:
            # players argument expects [p1, p2]
            res = deckgym.simulate(
                f_a.name, 
                f_b.name, 
                players=[player_a_code, player_b_code],
                num_simulations=1,
                seed=seed
            )
            
            if res.player_a_wins > 0:
                return 1
            elif res.player_b_wins > 0:
                return -1
            else:
                return 0
        except Exception as e:
            console.print(f"[red]Error in simulation: {e}[/red]")
            return 0 # Treat error as draw/void

def calibrate(n_games_per_pair: int = 50):
    """Run round-robin tournament between baselines."""
    tracker = TrueSkillTracker()
    loader = MetaDeckLoader("meta_deck.json")
    
    bots = DEFAULT_BASELINES_TO_CALIBRATE
    pairs = []
    for i in range(len(bots)):
        for j in range(i + 1, len(bots)):
            pairs.append((bots[i], bots[j]))
    
    total_games = len(pairs) * n_games_per_pair
    console.print(f"[bold]Starting Calibration: {len(pairs)} pairs, {n_games_per_pair} games each ({total_games} total).[/bold]")
    
    from rich.progress import TimeRemainingColumn, MofNCompleteColumn
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        overall_task = progress.add_task("[bold cyan]Total Progress[/bold cyan]", total=total_games)
        
        # Track active sub-tasks for each pair to show what's happening
        pair_tasks = {}
        for p1, p2 in pairs:
            name = f"{BOT_NAMES.get(p1, p1)} vs {BOT_NAMES.get(p2, p2)}"
            pair_tasks[(p1, p2)] = progress.add_task(f"  {name}", total=n_games_per_pair, visible=False)

        def play_pair_game(p1, p2):
            d1 = loader.sample_deck()
            d2 = loader.sample_deck()
            d1_s = d1 if isinstance(d1, str) else d1.deck_string
            d2_s = d2 if isinstance(d2, str) else d2.deck_string
            
            res = run_rust_match(d1_s, d2_s, p1, p2)
            
            if res == 1: tracker.update(p1, p2, draw=False)
            elif res == -1: tracker.update(p2, p1, draw=False)
            else: tracker.update(p1, p2, draw=True)
            
            progress.update(overall_task, advance=1)
            progress.update(pair_tasks[(p1, p2)], advance=1, visible=True)
            
            # Hide task when finished to avoid clutter
            if progress.tasks[pair_tasks[(p1, p2)]].finished:
                progress.update(pair_tasks[(p1, p2)], visible=False)

        # Use ThreadPoolExecutor for parallel simulations
        # Since Rust simulation releases GIL, this is very effective.
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for p1, p2 in pairs:
                for _ in range(n_games_per_pair):
                    futures.append(executor.submit(play_pair_game, p1, p2))
            
            # Wait for all to finish
            for future in futures:
                future.result()
                
    tracker.save()
    print_ratings(tracker)

def evaluate_single_model(model_path: str, n_games: int = 100):
    """Evaluate a single model against baselines."""
    if not ONNX_AVAILABLE:
        console.print("[red]ONNX tools not installed. Cannot evaluate model.[/red]")
        return

    tracker = TrueSkillTracker()
    loader = MetaDeckLoader("meta_deck.json")
    
    model_name = Path(model_path).stem
    console.print(f"Evaluating [bold cyan]{model_name}[/bold cyan] via ONNX/Rust...")
    
    # 1. Convert to ONNX
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "model.onnx")
        try:
            model = MaskablePPO.load(model_path, device="cpu")
            obs_size = model.observation_space.shape[0]
            export_policy_to_onnx(model, onnx_path, observation_size=obs_size, validate=False)
        except Exception as e:
            console.print(f"[red]Failed to export ONNX: {e}[/red]")
            return

        onnx_code = f"onnx:{onnx_path}"
        
        # 2. Run against baselines
        baselines = DEFAULT_BASELINES_TO_CALIBRATE
        
        # Add model to tracker temporarily (or permanently?)
        # For single audit, we can just print stats. Or update a separate file?
        # Let's use a temporary Rating object for the model
        model_rating = TRUESKILL_ENV.create_rating()
        
        for bl in baselines:
            console.print(f"  vs {BOT_NAMES[bl]}...", end=" ")
            
            wins, losses, draws = 0, 0, 0
            bl_rating = tracker.get(bl).to_trueskill()
            
            # Simple loop for now (could parallelize)
            for _ in range(n_games):
                d1 = loader.sample_deck()
                d2 = loader.sample_deck()
                
                # Ensure JSON strings
                d1_s = d1.to_string() if hasattr(d1, "to_string") else str(d1)
                d2_s = d2.to_string() if hasattr(d2, "to_string") else str(d2)
                
                # Agent (A) vs Baseline (B)
                # Note: Rust load ONNX is somewhat expensive per match if reload?
                # Does `OnnxPlayer::new` reload session every time? Yes.
                # Is it fast enough? Hopefully. Loading 5MB ONNX is ~10ms.
                
                # To optimize: use `py_simulate` with num_simulations=10 and same decks?
                # No, TrueSkill needs diversity. 
                # We accept the overhead for accuracy.
                
                # Agent (A) vs Baseline (B)
                res = run_rust_match(d1_s, d2_s, onnx_code, bl)
                if res == 1:
                    model_rating, bl_rating = TRUESKILL_ENV.rate_1vs1(model_rating, bl_rating)
                    wins += 1
                elif res == -1:
                    bl_rating, model_rating = TRUESKILL_ENV.rate_1vs1(bl_rating, model_rating)
                    losses += 1
                else:
                    model_rating, bl_rating = TRUESKILL_ENV.rate_1vs1(model_rating, bl_rating, drawn=True)
                    draws += 1
            
            console.print(f"[green]{wins}W {losses}L {draws}D[/green]")
            
        console.print(f"\nFinal Rating: [bold green]mu={model_rating.mu:.2f} sigma={model_rating.sigma:.2f}[/bold green] (Expose: {model_rating.mu - 3*model_rating.sigma:.2f})")

def print_ratings(tracker: TrueSkillTracker):
    table = Table(title="TrueSkill Ratings")
    table.add_column("Bot", style="cyan")
    table.add_column("Mu", style="green")
    table.add_column("Sigma", style="yellow")
    table.add_column("Expose (Mu-3Sig)", style="bold white")
    
    sorted_ratings = sorted(tracker.ratings.items(), key=lambda x: x[1].expose, reverse=True)
    
    for code, r in sorted_ratings:
        name = BOT_NAMES.get(code, code)
        table.add_row(name, f"{r.mu:.2f}", f"{r.sigma:.2f}", f"{r.expose:.2f}")
    
    console.print(table)

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Calibrate
    calib_p = subparsers.add_parser("calibrate", help="Calibrate baselines")
    calib_p.add_argument("--games", type=int, default=50, help="Games per pair")
    
    # Audit
    audit_p = subparsers.add_parser("audit", help="Audit model")
    audit_p.add_argument("model", help="Path to .zip model")
    audit_p.add_argument("--games", type=int, default=30)

    # Bench (Directory) - To Implement
    bench_p = subparsers.add_parser("bench", help="Benchmark directory")
    bench_p.add_argument("dir", help="Directory")
    
    args = parser.parse_args()
    
    print_system_info()
    
    if args.command == "calibrate":
        calibrate(args.games)
    elif args.command == "audit":
        evaluate_single_model(args.model, args.games)
    elif args.command == "bench":
        console.print("[yellow]Bench mode not fully implemented yet, use audit loop manually.[/yellow]")

if __name__ == "__main__":
    main()
