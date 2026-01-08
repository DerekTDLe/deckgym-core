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
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import trueskill
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import deckgym
from deckgym.deck_loader import MetaDeckLoader
from deckgym.config import (
    TRUESKILL_MU,
    TRUESKILL_SIGMA,
    TRUESKILL_BETA,
    TRUESKILL_TAU,
    TRUESKILL_DRAW_PROBABILITY,
    EVAL_GAMES_PER_PAIR_CALIBRATE,
    EVAL_GAMES_PER_PAIR_AUDIT,
    EVAL_GAMES_PER_PAIR_BENCH,
    EVAL_REPORTS_DIR,
    EVAL_DEFAULT_DEVICE,
)

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

        console.print(
            f"  PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})"
        )
    except ImportError:
        console.print("  PyTorch: Not installed")

    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        console.print(f"  ONNX Runtime: {ort.__version__}")
        console.print(f"  Available Providers: [cyan]{', '.join(providers)}[/cyan]")
        if "TensorrtExecutionProvider" in providers:
            console.print(
                "  [green]✓ TensorRT Execution Provider is available![/green]"
            )
        else:
            console.print("  [yellow]⚠ TensorRT Execution Provider not found.[/yellow]")
    except ImportError:
        console.print("  ONNX Runtime: Not installed")
    console.print("─" * 40)


# --- Configuration ---
TRUESKILL_ENV = trueskill.TrueSkill(
    mu=TRUESKILL_MU,
    sigma=TRUESKILL_SIGMA,
    beta=TRUESKILL_BETA,
    tau=TRUESKILL_TAU,
    draw_probability=TRUESKILL_DRAW_PROBABILITY,
)
BASELINES_FILE = Path("trueskill_baselines.json")
DEFAULT_BASELINES_TO_CALIBRATE = [
    "r",
    "aa",
    "et",
    "w",
    "v",
    "er",
    "e2",
    "e3",
    "e4",
    "e5",
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
    def from_trueskill(r: trueskill.Rating) -> "Rating":
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
                self.ratings[code] = Rating.from_trueskill(
                    TRUESKILL_ENV.create_rating()
                )

    def save(self):
        data = {
            name: {"mu": r.mu, "sigma": r.sigma} for name, r in self.ratings.items()
        }
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
        return self.ratings.get(
            code, Rating.from_trueskill(TRUESKILL_ENV.create_rating())
        )


def save_report(data: dict, mode: str, name: Optional[str] = None):
    """Save evaluation results to a JSON file."""
    os.makedirs(EVAL_REPORTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{mode}_"
    if name:
        # Sanitize name for filename
        safe_name = "".join(c for c in name if c.isalnum() or c in ("-", "_")).rstrip()
        filename += f"{safe_name}_"
    filename += f"{timestamp}.json"

    report_path = os.path.join(EVAL_REPORTS_DIR, filename)

    report_data = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        **data,
    }

    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)

    console.print(f"\n[bold green]Report saved to:[/bold green] {report_path}")


def run_rust_match(
    deck_a_json: str,
    deck_b_json: str,
    player_a_code: str,
    player_b_code: str,
    seed: int = None,
) -> int:
    """
    Run a single match in Rust.
    Requires writing decks to temp files because py_simulate takes paths.
    Returns: 1 (A wins), -1 (B wins), 0 (Draw)
    """
    if seed is None:
        import random

        seed = random.randint(0, 2**64 - 1)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=True
    ) as f_a, tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=True) as f_b:

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
                seed=seed,
            )

            if res.player_a_wins > 0:
                return 1
            elif res.player_b_wins > 0:
                return -1
            else:
                return 0
        except Exception as e:
            console.print(f"[red]Error in simulation: {e}[/red]")
            return 0  # Treat error as draw/void


def calibrate(n_games_per_pair: int = EVAL_GAMES_PER_PAIR_CALIBRATE):
    """Run round-robin tournament between baselines."""
    tracker = TrueSkillTracker()
    loader = MetaDeckLoader("meta_deck.json")

    bots = DEFAULT_BASELINES_TO_CALIBRATE
    pairs = []
    for i in range(len(bots)):
        for j in range(i + 1, len(bots)):
            pairs.append((bots[i], bots[j]))

    total_games = len(pairs) * n_games_per_pair
    console.print(
        f"[bold]Starting Calibration: {len(pairs)} pairs, {n_games_per_pair} games each ({total_games} total).[/bold]"
    )

    from rich.progress import TimeRemainingColumn, MofNCompleteColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        overall_task = progress.add_task(
            "[bold cyan]Total Progress[/bold cyan]", total=total_games
        )

        # Track active sub-tasks for each pair to show what's happening
        pair_tasks = {}
        for p1, p2 in pairs:
            name = f"{BOT_NAMES.get(p1, p1)} vs {BOT_NAMES.get(p2, p2)}"
            pair_tasks[(p1, p2)] = progress.add_task(
                f"  {name}", total=n_games_per_pair, visible=False
            )

        def play_pair_game(p1, p2):
            d1 = loader.sample_deck()
            d2 = loader.sample_deck()
            d1_s = d1 if isinstance(d1, str) else d1.deck_string
            d2_s = d2 if isinstance(d2, str) else d2.deck_string

            res = run_rust_match(d1_s, d2_s, p1, p2)

            if res == 1:
                tracker.update(p1, p2, draw=False)
            elif res == -1:
                tracker.update(p2, p1, draw=False)
            else:
                tracker.update(p1, p2, draw=True)

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


def evaluate_single_model(
    model_path: str,
    n_games: int = EVAL_GAMES_PER_PAIR_AUDIT,
    tracker: Optional[TrueSkillTracker] = None,
    show_table: bool = True,
    device: str = EVAL_DEFAULT_DEVICE,
) -> Optional[dict]:
    """Evaluate a single model against baselines."""
    if not ONNX_AVAILABLE:
        console.print("[red]ONNX tools not installed. Cannot evaluate model.[/red]")
        return

    if tracker is None:
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
            export_policy_to_onnx(
                model, onnx_path, observation_size=obs_size, validate=False
            )
        except Exception as e:
            console.print(f"[red]Failed to export ONNX: {e}[/red]")
            return

        onnx_code = f"onnx:{onnx_path}:{device}"

        # 2. Run against baselines
        baselines = DEFAULT_BASELINES_TO_CALIBRATE
        model_rating = TRUESKILL_ENV.create_rating()

        for bl in baselines:
            console.print(f"  vs {BOT_NAMES[bl]}...", end=" ")

            wins, losses, draws = 0, 0, 0
            bl_rating = tracker.get(bl).to_trueskill()

            for _ in range(n_games):
                d1 = loader.sample_deck()
                d2 = loader.sample_deck()

                d1_s = d1.to_string() if hasattr(d1, "to_string") else str(d1)
                d2_s = d2.to_string() if hasattr(d2, "to_string") else str(d2)

                res = run_rust_match(d1_s, d2_s, onnx_code, bl)
                if res == 1:
                    model_rating, bl_rating = TRUESKILL_ENV.rate_1vs1(
                        model_rating, bl_rating
                    )
                    wins += 1
                elif res == -1:
                    bl_rating, model_rating = TRUESKILL_ENV.rate_1vs1(
                        bl_rating, model_rating
                    )
                    losses += 1
                else:
                    model_rating, bl_rating = TRUESKILL_ENV.rate_1vs1(
                        model_rating, bl_rating, drawn=True
                    )
                    draws += 1

            console.print(f"[green]{wins}W {losses}L {draws}D[/green]")

        # Add to tracker for table display
        tracker.ratings[model_name] = Rating.from_trueskill(model_rating)

        if show_table:
            print_ratings(tracker)
        else:
            console.print(
                f"Done. Rating: [bold green]mu={model_rating.mu:.2f} sigma={model_rating.sigma:.2f}[/bold green] (Expose: {expose:.2f})\n"
            )

        # Build results for storage
        results = {
            "meta": {
                "model_name": model_name,
                "model_path": str(model_path),
                "games_per_match": n_games,
            },
            "ratings": {
                model_name: {
                    "mu": model_rating.mu,
                    "sigma": model_rating.sigma,
                    "expose": model_rating.mu - 3 * model_rating.sigma,
                }
            },
            "stats": {},
        }

        # Add baseline ratings to report
        for bl in baselines:
            bl_r = tracker.get(bl)
            results["ratings"][BOT_NAMES[bl]] = {
                "mu": bl_r.mu,
                "sigma": bl_r.sigma,
                "expose": bl_r.expose,
            }

        if tracker is None or show_table:  # Only save if it's a direct audit
            save_report(results, "audit", name=model_name)

        return results


def audit_directory(
    directory: str,
    n_games: int = EVAL_GAMES_PER_PAIR_AUDIT,
    device: str = EVAL_DEFAULT_DEVICE,
):
    """Evaluate all models in a directory against baselines."""
    model_dir = Path(directory)
    if not model_dir.exists():
        console.print(f"[red]Directory not found: {directory}[/red]")
        return

    model_files = sorted(model_dir.glob("*.zip"))
    if not model_files:
        console.print(f"[red]No .zip files found in {directory}[/red]")
        return

    console.print(f"[bold]Auditing {len(model_files)} models in {directory}[/bold]\n")

    tracker = TrueSkillTracker()
    all_results = []
    for mf in model_files:
        res = evaluate_single_model(
            str(mf), n_games=n_games, tracker=tracker, show_table=False, device=device
        )
        if res:
            all_results.append(res)

    console.print("\n[bold cyan]Final Audit Results:[/bold cyan]")
    print_ratings(tracker)

    # Save summary report for the whole directory
    summary = {
        "directory": str(directory),
        "model_count": len(all_results),
        "results": all_results,
    }
    save_report(summary, "audit_dir", name=model_dir.name)


def benchmark_directory(
    directory: str,
    n_games_per_pair: int = EVAL_GAMES_PER_PAIR_BENCH,
    include_baselines: bool = True,
    device: str = EVAL_DEFAULT_DEVICE,
):
    """
    Benchmark all models in a directory against each other and baselines.

    This runs a full round-robin tournament:
    1. Model vs Model (all pairs)
    2. Model vs Baselines (all combinations)

    Results are ranked by TrueSkill expose rating.
    """
    if not ONNX_AVAILABLE:
        console.print("[red]ONNX tools not installed. Cannot benchmark models.[/red]")
        return

    model_dir = Path(directory)
    if not model_dir.exists():
        console.print(f"[red]Directory not found: {directory}[/red]")
        return

    # Find all .zip model files
    model_files = sorted(model_dir.glob("*.zip"))
    if not model_files:
        console.print(f"[red]No .zip files found in {directory}[/red]")
        return

    console.print(f"[bold]Found {len(model_files)} models in {directory}[/bold]")
    for mf in model_files:
        console.print(f"  • {mf.name}")

    # Create temp directory for ONNX files
    tmpdir = tempfile.mkdtemp(prefix="bench_onnx_")
    onnx_paths: Dict[str, str] = {}  # model_name -> onnx_path

    try:
        # Convert all models to ONNX
        console.print("\n[bold cyan]Converting models to ONNX...[/bold cyan]")
        for model_file in model_files:
            model_name = model_file.stem
            onnx_path = os.path.join(tmpdir, f"{model_name}.onnx")

            try:
                model = MaskablePPO.load(str(model_file), device="cpu")
                obs_size = model.observation_space.shape[0]
                export_policy_to_onnx(
                    model, onnx_path, observation_size=obs_size, validate=False
                )
                onnx_paths[model_name] = onnx_path
                console.print(f"  [green]✓[/green] {model_name}")
                del model
            except Exception as e:
                console.print(f"  [red]✗[/red] {model_name}: {e}")

        if not onnx_paths:
            console.print("[red]No models could be converted to ONNX![/red]")
            return

        # Build participants list
        participants: Dict[str, str] = {}  # name -> code (for rust)

        # Add models
        for model_name, onnx_path in onnx_paths.items():
            participants[model_name] = f"onnx:{onnx_path}:{device}"

        # Add baselines
        if include_baselines:
            baselines_to_test = ["e2", "e3", "e4"]  # Main baselines
            for bl in baselines_to_test:
                participants[BOT_NAMES[bl]] = bl

        console.print(f"\n[bold]Participants: {len(participants)}[/bold]")

        # Build all matchup pairs
        names = list(participants.keys())
        pairs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pairs.append((names[i], names[j]))

        total_games = len(pairs) * n_games_per_pair
        console.print(
            f"[bold]Running {len(pairs)} matchups × {n_games_per_pair} games = {total_games} total games[/bold]\n"
        )

        # Initialize ratings
        ratings: Dict[str, trueskill.Rating] = {}
        for name in participants:
            ratings[name] = TRUESKILL_ENV.create_rating()

        # Load calibrated baselines for initial ratings
        if include_baselines and BASELINES_FILE.exists():
            tracker = TrueSkillTracker()
            for bl_code in ["e2", "e3", "e4"]:
                bl_name = BOT_NAMES[bl_code]
                if bl_name in participants:
                    rating = tracker.get(bl_code)
                    ratings[bl_name] = trueskill.Rating(
                        mu=rating.mu, sigma=rating.sigma
                    )

        # Statistics tracking
        wins: Dict[str, int] = {name: 0 for name in participants}
        losses: Dict[str, int] = {name: 0 for name in participants}
        draws: Dict[str, int] = {name: 0 for name in participants}

        loader = MetaDeckLoader("meta_deck.json")
        ratings_lock = threading.Lock()

        from rich.progress import TimeRemainingColumn, MofNCompleteColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            overall_task = progress.add_task(
                "[bold cyan]Tournament Progress[/bold cyan]", total=total_games
            )

            def play_game(name_a: str, name_b: str):
                code_a = participants[name_a]
                code_b = participants[name_b]

                d1 = loader.sample_deck()
                d2 = loader.sample_deck()
                d1_s = d1.to_string() if hasattr(d1, "to_string") else str(d1)
                d2_s = d2.to_string() if hasattr(d2, "to_string") else str(d2)

                res = run_rust_match(d1_s, d2_s, code_a, code_b)

                with ratings_lock:
                    r_a = ratings[name_a]
                    r_b = ratings[name_b]

                    if res == 1:  # A wins
                        new_a, new_b = TRUESKILL_ENV.rate_1vs1(r_a, r_b)
                        wins[name_a] += 1
                        losses[name_b] += 1
                    elif res == -1:  # B wins
                        new_b, new_a = TRUESKILL_ENV.rate_1vs1(r_b, r_a)
                        wins[name_b] += 1
                        losses[name_a] += 1
                    else:  # Draw
                        new_a, new_b = TRUESKILL_ENV.rate_1vs1(r_a, r_b, drawn=True)
                        draws[name_a] += 1
                        draws[name_b] += 1

                    ratings[name_a] = new_a
                    ratings[name_b] = new_b

                progress.update(overall_task, advance=1)

            # Run games in parallel
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = []
                for name_a, name_b in pairs:
                    for _ in range(n_games_per_pair):
                        futures.append(executor.submit(play_game, name_a, name_b))

                for future in futures:
                    future.result()

        # Print results
        print_bench_results(ratings, wins, losses, draws, participants)

        # Save benchmark report
        bench_results = {
            "meta": {
                "directory": str(directory),
                "games_per_match": n_games_per_pair,
                "participants": list(participants.keys()),
            },
            "ratings": {
                name: {
                    "mu": r.mu,
                    "sigma": r.sigma,
                    "expose": r.mu - 3 * r.sigma,
                }
                for name, r in ratings.items()
            },
            "stats": {
                name: {
                    "wins": wins[name],
                    "losses": losses[name],
                    "draws": draws[name],
                }
                for name in participants
            },
        }
        save_report(bench_results, "bench", name=model_dir.name)

    finally:
        # Cleanup temp ONNX files
        shutil.rmtree(tmpdir, ignore_errors=True)


def print_bench_results(
    ratings: Dict[str, trueskill.Rating],
    wins: Dict[str, int],
    losses: Dict[str, int],
    draws: Dict[str, int],
    participants: Dict[str, str],
):
    """Print benchmark results as a rich table."""
    console.print("\n")

    table = Table(title="[bold]Benchmark Results[/bold]", show_lines=True)
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Name", style="cyan")
    table.add_column("Mu", style="green", justify="right")
    table.add_column("Sigma", style="yellow", justify="right")
    table.add_column("Expose", style="bold white", justify="right")
    table.add_column("W-L-D", justify="center")
    table.add_column("Win%", style="magenta", justify="right")

    # Sort by expose rating
    sorted_participants = sorted(
        ratings.items(), key=lambda x: x[1].mu - 3 * x[1].sigma, reverse=True
    )

    for rank, (name, rating) in enumerate(sorted_participants, 1):
        expose = rating.mu - 3 * rating.sigma
        w, l, d = wins[name], losses[name], draws[name]
        total = w + l + d
        win_pct = (w / total * 100) if total > 0 else 0

        # Highlight baselines differently
        is_baseline = name in BOT_NAMES.values()
        name_style = "[dim italic]" if is_baseline else ""
        name_end = "[/dim italic]" if is_baseline else ""

        table.add_row(
            str(rank),
            f"{name_style}{name}{name_end}",
            f"{rating.mu:.1f}",
            f"{rating.sigma:.1f}",
            f"{expose:.1f}",
            f"{w}-{l}-{d}",
            f"{win_pct:.1f}%",
        )

    console.print(table)

    # Summary
    console.print(
        "\n[dim]Legend: Mu=TrueSkill mean, Sigma=uncertainty, Expose=Mu-3σ (conservative rating)[/dim]"
    )


def print_ratings(tracker: TrueSkillTracker):
    table = Table(title="TrueSkill Ratings")
    table.add_column("Bot", style="cyan")
    table.add_column("Mu", style="green")
    table.add_column("Sigma", style="yellow")
    table.add_column("Expose (Mu-3Sig)", style="bold white")

    sorted_ratings = sorted(
        tracker.ratings.items(), key=lambda x: x[1].expose, reverse=True
    )

    for code, r in sorted_ratings:
        name = BOT_NAMES.get(code, code)
        table.add_row(name, f"{r.mu:.2f}", f"{r.sigma:.2f}", f"{r.expose:.2f}")

    console.print(table)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Calibrate
    calib_p = subparsers.add_parser("calibrate", help="Calibrate baselines")
    calib_p.add_argument(
        "--games",
        type=int,
        default=EVAL_GAMES_PER_PAIR_CALIBRATE,
        help="Games per pair",
    )

    # Audit
    audit_p = subparsers.add_parser("audit", help="Audit model")
    audit_p.add_argument("model", help="Path to .zip model")
    audit_p.add_argument(
        "--games",
        type=int,
        default=EVAL_GAMES_PER_PAIR_AUDIT,
        help=f"Games per baseline (default: {EVAL_GAMES_PER_PAIR_AUDIT})",
    )
    audit_p.add_argument(
        "--device",
        default=EVAL_DEFAULT_DEVICE,
        help=f"Device for ONNX (default: {EVAL_DEFAULT_DEVICE})",
    )

    # Bench (Directory)
    bench_p = subparsers.add_parser("bench", help="Benchmark directory")
    bench_p.add_argument("dir", help="Directory containing .zip models")
    bench_p.add_argument(
        "--games",
        type=int,
        default=EVAL_GAMES_PER_PAIR_BENCH,
        help=f"Games per matchup (default: {EVAL_GAMES_PER_PAIR_BENCH})",
    )
    bench_p.add_argument(
        "--no-baselines", action="store_true", help="Exclude baselines from tournament"
    )
    bench_p.add_argument(
        "--device",
        default=EVAL_DEFAULT_DEVICE,
        help=f"Device for ONNX (default: {EVAL_DEFAULT_DEVICE})",
    )

    args = parser.parse_args()

    print_system_info()

    if args.command == "calibrate":
        calibrate(args.games)
    elif args.command == "audit":
        path = Path(args.model)
        if path.is_dir():
            audit_directory(args.model, args.games, args.device)
        else:
            evaluate_single_model(args.model, args.games, device=args.device)
    elif args.command == "bench":
        benchmark_directory(
            args.dir,
            args.games,
            include_baselines=not args.no_baselines,
            device=args.device,
        )


if __name__ == "__main__":
    main()
