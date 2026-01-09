#!/usr/bin/env python3
"""
Benchmark script to compare steps per second of various player baselines.
Usage: python python/scripts/benchmark_players.py
"""

import subprocess
import re
import sys

# Players to benchmark (excluding mcts and human)
PLAYERS = [
    ("aa", "AttachAttack"),
    ("et", "EndTurn"),
    ("r", "Random"),
    ("w", "WeightedRandom"),
    ("v", "ValueFunction"),
    ("er", "EvolutionRusher"),
    ("e2", "ExpectiMiniMax (depth 2)"),
    ("e3", "ExpectiMiniMax (depth 3)"),
    ("o1c", "ONNX CPU"),
    ("o1g", "ONNX GPU"),
    ("o1t", "ONNX TensorRT"),
]

NUM_GAMES = 200
DECK_A = "example_decks/venusaur-exeggutor.txt"
DECK_B = "example_decks/altaria.txt"
# Should not matter


def run_benchmark(player_code: str) -> tuple[float, float, float]:
    """Run benchmark for a player and return (steps/sec, avg_plys, total_time_sec)."""
    cmd = ["cargo", "run", "--release"]

    # Add --features onnx for ONNX-based players
    if player_code.startswith("o"):
        cmd.extend(["--features", "onnx"])

    cmd.extend(
        [
            "simulate",
            DECK_A,
            DECK_B,
            "--num",
            str(NUM_GAMES),
            "--players",
            f"{player_code},{player_code}",
        ]
    )

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    # Parse average plys per game
    plys_match = re.search(r"Average number of plys per game: ([\d.]+)", output)
    avg_plys = float(plys_match.group(1)) if plys_match else 0.0

    # Parse total time - format: "Ran X simulations in Xm Xs Xms..." or "Xs Xms..."
    total_time_sec = 0.0

    # Try to parse "Xm Xs Xms" format
    time_match = re.search(r"in (\d+)m (\d+)s (\d+)ms", output)
    if time_match:
        total_time_sec = (
            int(time_match.group(1)) * 60
            + int(time_match.group(2))
            + int(time_match.group(3)) / 1000.0
        )
    else:
        # Try "Xs Xms" format
        time_match = re.search(r"in (\d+)s (\d+)ms", output)
        if time_match:
            total_time_sec = (
                int(time_match.group(1)) + int(time_match.group(2)) / 1000.0
            )
        else:
            # Try "Xms" only format
            time_match = re.search(r"in (\d+)ms", output)
            if time_match:
                total_time_sec = int(time_match.group(1)) / 1000.0

    # Calculate steps per second
    total_steps = avg_plys * NUM_GAMES
    steps_per_sec = total_steps / total_time_sec if total_time_sec > 0 else 0.0

    return steps_per_sec, avg_plys, total_time_sec


def main():
    print("=" * 72)
    print("Player Baseline Benchmark (Steps per Second)")
    print(f"Running {NUM_GAMES} games per player (both sides same player)")
    print("=" * 72)
    print()

    results = []

    for code, name in PLAYERS:
        print(f"Benchmarking {name}...", end=" ", flush=True)
        try:
            steps_per_sec, avg_plys, total_time = run_benchmark(code)
            results.append((name, code, steps_per_sec, avg_plys, total_time))
            print(
                f"{steps_per_sec:.0f} steps/s (avg {avg_plys:.1f} plys/game, {total_time:.2f}s total)"
            )
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((name, code, 0.0, 0.0, 0.0))

    print()
    print("=" * 72)
    print("Results (sorted by steps/second)")
    print("=" * 72)
    print(
        f"{'Player':<30} {'Code':<6} {'Steps/s':>12} {'Plys/game':>10} {'Time (s)':>10}"
    )
    print("-" * 72)

    for name, code, sps, plys, time in sorted(results, key=lambda x: -x[2]):
        print(f"{name:<30} {code:<6} {sps:>12,.0f} {plys:>10.1f} {time:>10.2f}")


if __name__ == "__main__":
    main()
