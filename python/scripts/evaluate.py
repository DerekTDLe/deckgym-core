#!/usr/bin/env python3
"""
Unified Evaluation Script for DeckGym using TrueSkill and Rust-Native Execution.

Modes:
- bench: Evaluate a directory of checkpoints in a tournament.
"""

import argparse
import json
import math
import os
import shutil
import tempfile
import sys
import threading
import random
from itertools import combinations
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
    N_ROBIN_ROUNDS,
    EVAL_REPORTS_DIR,
    EVAL_DEFAULT_DEVICE,
)

# Try importing ONNX export tools (required for audit/bench)
try:
    from deckgym.onnx_export import export_policy_to_onnx
    from sb3_contrib import MaskablePPO
    from deckgym.diagnostic_logger import get_logger

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

    def get_logger():
        class DummyLogger:
            def log_error(self, *args, **kwargs):
                pass

        return DummyLogger()


# PanicException is often not importable directly but can be caught by name
PanicException = None
try:
    from pyo3_runtime import PanicException
except ImportError:
    pass


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
BOT_NAMES = {
    "r": "Random",
    "aa": "AttachAttack",
    "et": "EndTurn",
    "w": "WeightedRandom",
    "v": "ValueFunction",
    "er": "EvolutionRusher",
    "e2": "Expectiminimax(2)",
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


def assign_tier(percentile: float) -> Tuple[str, str]:
    """Assign a tier and color based on percentile."""
    if percentile >= 95:
        return "S+", "bold magenta"
    if percentile >= 80:
        return "S", "magenta"
    if percentile >= 60:
        return "A", "green"
    if percentile >= 40:
        return "B", "yellow"
    if percentile >= 20:
        return "C", "orange"
    return "D", "red"


def get_reliability(sigma: float) -> str:
    """Return a descriptive reliability string based on sigma."""
    if sigma < 40:
        return "[green]Highly Reliable 🟢[/green]"
    if sigma < 80:
        return "[yellow]Stable 🟡[/yellow]"
    if sigma < 150:
        return "[orange3]Estimating 🟠[/orange3]"
    return "[red]Initial 🔴[/red]"


def save_report(data: dict, mode: str, name: Optional[str] = None):
    """Save evaluation results to a JSON file."""
    os.makedirs(EVAL_REPORTS_DIR, exist_ok=True)

    filename = f"{mode}"
    if name:
        # Sanitize name for filename
        safe_name = "".join(c for c in name if c.isalnum() or c in ("-", "_")).rstrip()
        filename += f"_{safe_name}"
    filename += ".json"

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
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise e

            is_panic = (PanicException and isinstance(e, PanicException)) or type(
                e
            ).__name__ == "PanicException"
            reason = "evaluation_panic" if is_panic else "evaluation_error"

            if is_panic:
                console.print(
                    f"[bold red]CRITICAL: Rust Panic during simulation![/bold red]"
                )
            else:
                console.print(f"[red]Error in simulation: {e}[/red]")

            console.print(f"[red]{e}[/red]")

            try:
                # Log detailed info for reproduction
                get_logger().log_error(
                    reason,
                    0,
                    None,  # State not available during simulate() crash
                    {
                        "p1": player_a_code,
                        "p2": player_b_code,
                        "deck_a": deck_a_json,
                        "deck_b": deck_b_json,
                        "seed": seed,
                        "error": str(e),
                    },
                )
            except Exception as log_err:
                console.print(f"[dim red]Failed to log diagnostic: {log_err}[/dim red]")

            return 0  # Treat as draw/void


def benchmark_directory(
    directory: str,
    n_robin_rounds: int = N_ROBIN_ROUNDS,
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

    # Use TemporaryDirectory for automatic cleanup of ONNX files
    with tempfile.TemporaryDirectory(prefix="bench_onnx_") as tmpdir:
        onnx_paths: Dict[str, str] = {}  # model_name -> onnx_path

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
            baselines_to_test = ["e2", "er", "aa", "w"]  # e2 is the anchor
            for bl in baselines_to_test:
                participants[BOT_NAMES[bl]] = bl

        console.print(f"\n[bold]Participants: {len(participants)}[/bold]")

        # Build all matchup pairs (All-play-all)
        names = list(participants.keys())
        base_pairs = list(combinations(names, 2))

        # Multiply by repetitions and shuffle GLOBALLY
        # This improves TrueSkill convergence by interleaving participants
        matchups = base_pairs * n_robin_rounds
        random.shuffle(matchups)

        total_games = len(matchups)
        console.print(
            f"[bold]Running {len(base_pairs)} matchups × {n_robin_rounds} reps = {total_games} total games[/bold]\n"
        )

        # Initialize ratings
        ratings: Dict[str, trueskill.Rating] = {}
        for name in participants:
            ratings[name] = TRUESKILL_ENV.create_rating()

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
                futures = [executor.submit(play_game, p[0], p[1]) for p in matchups]

                # Wait for all to finish
                for future in futures:
                    future.result()

        # Print results
        print_bench_results(ratings, wins, losses, draws, participants)

        # Save benchmark report
        bench_results = {
            "meta": {
                "directory": str(directory),
                "games_per_match": n_robin_rounds,
                "model_count": len(onnx_paths),
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
                for name in ratings.keys()
            },
        }
        save_report(bench_results, "bench", name=model_dir.name)


def print_leaderboard(
    ratings: Dict[str, trueskill.Rating],
    wins: Dict[str, int],
    losses: Dict[str, int],
    draws: Dict[str, int],
    participants: Dict[str, str],
    title: str = "Tournament Standings",
):
    """Print a production-grade leaderboard using rich."""
    from rich.panel import Panel

    # Sort by expose rating (mu - 3*sigma)
    sorted_items = sorted(
        ratings.items(), key=lambda x: x[1].mu - 3 * x[1].sigma, reverse=True
    )

    # Calculate percentiles for tiers
    n = len(sorted_items)

    # Find anchor for deltas (default: Expectiminimax(2))
    anchor_name = BOT_NAMES.get("e2", "Expectiminimax(2)")
    anchor_rating = ratings.get(anchor_name)
    if not anchor_rating:
        # Fallback to strongest baseline if e2 not present
        baselines = [
            name
            for name, code in participants.items()
            if code in BOT_NAMES or code in BOT_NAMES.values()
        ]
        if baselines:
            baselines.sort(key=lambda x: ratings[x].mu, reverse=True)
            anchor_name = baselines[0]
            anchor_rating = ratings[anchor_name]

    table = Table(show_lines=True, header_style="bold cyan")
    table.add_column("Rank", style="dim", width=4, justify="center")
    table.add_column("Tier", justify="center")
    table.add_column("Participant", min_width=20)
    table.add_column("Rating (Mu/σ)", justify="right")
    table.add_column("Δ Anchor", justify="right")
    table.add_column("Expose", style="bold white", justify="right")
    table.add_column("W-L-D", justify="center")
    table.add_column("Win%", justify="right")
    table.add_column("Reliability", justify="left")

    for rank, (name, r) in enumerate(sorted_items, 1):
        # Percentile: (n - rank) / (n - 1) * 100
        percentile = ((n - rank) / (n - 1) * 100) if n > 1 else 100
        tier_str, tier_style = assign_tier(percentile)

        expose = r.mu - 3 * r.sigma
        w, l, d = wins[name], losses[name], draws[name]
        total = w + l + d
        win_pct = (w / total * 100) if total > 0 else 0

        # Style baselines differently
        is_baseline = (
            participants[name] in BOT_NAMES or participants[name] in BOT_NAMES.values()
        )
        display_name = (
            f"[italic dim]{name}[/italic dim]"
            if is_baseline
            else f"[bold]{name}[/bold]"
        )

        # Calculate delta vs anchor
        delta_str = ""
        if anchor_rating and name != anchor_name:
            delta = r.mu - anchor_rating.mu
            color = "green" if delta >= 0 else "red"
            delta_str = f"[{color}]{delta:+3.0f}[/{color}]"
        elif name == anchor_name:
            delta_str = "[bold cyan]ANCHOR[/bold cyan]"

        table.add_row(
            str(rank),
            f"[{tier_style}]{tier_str}[/{tier_style}]",
            display_name,
            f"{r.mu:.0f} [dim]±{r.sigma:.0f}[/dim]",
            delta_str,
            f"{expose:.1f}",
            f"{w}-{l}-{d}",
            f"{win_pct:.1f}%",
            get_reliability(r.sigma),
        )

    console.print("\n")
    console.print(Panel(table, title=f"[bold white]{title}[/bold white]", expand=False))
    console.print("[dim]Expose = Mu - 3σ (Conservative Skill Estimate)[/dim]\n")


def print_bench_results(
    ratings: Dict[str, trueskill.Rating],
    wins: Dict[str, int],
    losses: Dict[str, int],
    draws: Dict[str, int],
    participants: Dict[str, str],
):
    print_leaderboard(
        ratings, wins, losses, draws, participants, "Benchmark Tournament Results"
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Bench (Directory)
    bench_p = subparsers.add_parser(
        "bench", help="Benchmark directory (Tournament mode)"
    )
    bench_p.add_argument("dir", help="Directory containing .zip models")
    bench_p.add_argument(
        "--games",
        type=int,
        default=N_ROBIN_ROUNDS,
        help=f"Number of repetitions for the Round Robin tournament (default: {N_ROBIN_ROUNDS})",
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

    if args.command == "bench":
        benchmark_directory(
            args.dir,
            args.games,
            include_baselines=not args.no_baselines,
            device=args.device,
        )


if __name__ == "__main__":
    main()
