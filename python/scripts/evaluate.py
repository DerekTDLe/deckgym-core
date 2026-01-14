#!/usr/bin/env python3
"""
Scientific Evaluation Script for DeckGym (Extensive Chaos Mode).
Performs a full matrix cross-evaluation (396x396) to eliminate deck bias.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
import sys
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deckgym.deck_loader import MetaDeckLoader, DeckInfo

console = Console()

class ScientificEvaluator:
    def __init__(self, meta_json: str, model_code: str = "o1"):
        self.meta_json = meta_json
        self.model_code = model_code
        self.num_games = 1
        self.loader = MetaDeckLoader(meta_json)
        self.stats = {}
        
    def analyze_data(self):
        """Minimal data analysis for Chaos mode."""
        strengths = [d.strength for d in self.loader.decks]
        self.stats = {
            "total_decks": len(self.loader.decks),
            "archetypes_count": len(self.loader.archetypes),
            "mean_strength": np.mean(strengths)
        }
        
        console.print(Panel(
            f"Total Decks in Meta-DB: [bold cyan]{self.stats['total_decks']}[/bold cyan]\n"
            f"Archetypes Identified: [bold magenta]{self.stats['archetypes_count']}[/bold magenta]\n"
            f"Global Mean Strength: [bold white]{self.stats['mean_strength']:.3f}[/bold white]",
            title="Dataset Summary", 
            border_style="blue"
        ))


    def run_simulation(self, player_folder: str, opponent_folder: str, players: str, num_total_games: int, progress=None, task_id=None) -> List[Dict[str, Any]]:
        """Run the optimized matrix simulation and stream results/progress."""
        binary = "./target/release/deckgym"
        if not os.path.exists(binary):
            cmd_prefix = ["cargo", "run", "--release", "--features", "onnx", "--"]
        else:
            # Check if binary is executable
            if not os.access(binary, os.X_OK):
                os.chmod(binary, 0o755)
            cmd_prefix = [binary]

        cmd = cmd_prefix + [
            "simulate",
            player_folder,
            opponent_folder,
            "--num", str(num_total_games),
            "--players", players,
            "--parallel",
            "-vv"
        ]
        
        # Stream stderr in real-time
        process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, bufsize=1)
        
        matchups = []
        # Progress bar parsing (e.g., [00:00:05] [#####] 2500/5000)
        # Result parsing (e.g., RESULT: ...)
        
        if process.stderr:
            for line in process.stderr:
                line = line.strip()
                if not line: continue

                # 1. Parse progress for the UI
                if "/" in line and "[" in line and "]" in line:
                    try:
                        progress_part = line.split("]")[-1].strip().split()[0]
                        current, total = map(int, progress_part.split("/"))
                        if progress and task_id:
                            progress.update(task_id, completed=current)
                    except:
                        pass

                # 2. Parse Result lines
                if "RESULT:" in line:
                    try:
                        parts = line.split("|")
                        matchup_part = parts[0].replace("RESULT:", "").strip()
                        da_db = matchup_part.split(" VS ")
                        da_path = da_db[0].strip()
                        db_path = da_db[1].strip()
                        
                        stats_part = parts[1].strip()
                        stats_parts = stats_part.split()
                        wins = int(stats_parts[1])
                        losses = int(stats_parts[3])
                        ties = int(stats_parts[5])
                        
                        matchups.append({
                            "p1_deck": da_path,
                            "p2_deck": db_path,
                            "p1_wins": wins,
                            "p2_wins": losses,
                            "ties": ties,
                            "total": wins + losses + ties
                        })
                    except Exception:
                        pass
                
                # 3. Echo ONNX/INFO/Error logs
                if "[ONNX]" in line or "[INFO]" in line or "[PROGRESS]" in line or "Error" in line or "panic" in line:
                    console.print(f"[dim blue]{line}[/dim blue]")

        process.wait()
        
        if process.returncode != 0:
            console.print(f"[bold red]Simulation failed with exit code {process.returncode}[/bold red]")
            return []

        return matchups

    def run_paradigm(self, name: str, player_pool: List[DeckInfo], opponent_pool: List[DeckInfo], silent: bool = False):
        """Execute a full paradigm evaluation using optimized matrix batching."""
        if not silent:
            console.print(f"\n[bold green]>>> Starting Paradigm: {name}[/bold green] ({len(player_pool)}x{len(opponent_pool)} matchups)")
        
        p1_results = []
        
        with tempfile.TemporaryDirectory(prefix="p1_") as tmp_p1, tempfile.TemporaryDirectory(prefix="p2_") as tmp_p2:
            # Write player decks to p1 folder
            for i, d in enumerate(player_pool):
                fname = f"p1_{i:03d}.json"
                with open(os.path.join(tmp_p1, fname), "w") as f:
                    f.write(d.deck_string)

            # Write opponent decks to p2 folder
            for i, d in enumerate(opponent_pool):
                fname = f"p2_{i:03d}.json"
                with open(os.path.join(tmp_p2, fname), "w") as f:
                    f.write(d.deck_string)

            # Configure players
            opponent_code = "e2" if self.model_code != "e2" else "r"
            players = f"{self.model_code},{opponent_code}" 
            total_games = len(player_pool) * len(opponent_pool) * self.num_games
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
                disable=silent
            ) as progress:
                task = progress.add_task(f"Simulating {name}...", total=total_games)
                
                # Execute single matrix run
                matchups = self.run_simulation(tmp_p1, tmp_p2, players, total_games, progress=progress, task_id=task)
                
                if not matchups and not silent:
                    console.print("[red]No matchups were processed successfully.[/red]")
                
                if not silent:
                    progress.update(task, completed=total_games)

            # Re-map results back to p1_results using index-based filenames
            # Regex to extract the index from filenames like 'p1_000.json'
            import re
            for m in matchups:
                p1_fname = os.path.basename(m["p1_deck"])
                p2_fname = os.path.basename(m["p2_deck"])
                
                p1_match = re.search(r"p1_(\d+)", p1_fname)
                p2_match = re.search(r"p2_(\d+)", p2_fname)
                
                if p1_match and p2_match:
                    p1_idx = int(p1_match.group(1))
                    p2_idx = int(p2_match.group(1))
                    
                    if p1_idx < len(player_pool) and p2_idx < len(opponent_pool):
                        p1_deck = player_pool[p1_idx]
                        p2_deck = opponent_pool[p2_idx]
                        
                        if m["total"] == 0:
                            continue
                            
                        wr = m["p1_wins"] / m["total"]
                        p1_results.append({
                            "p1_strength": p1_deck.strength,
                            "p2_strength": p2_deck.strength,
                            "wr": wr,
                            "diff": p1_deck.strength - p2_deck.strength,
                            "p1_arch": p1_deck.archetype,
                            "p2_arch": p2_deck.archetype,
                            "wins": m["p1_wins"],
                            "losses": m["p2_wins"],
                            "ties": m["ties"],
                            "total": m["total"]
                        })
        
        if not silent:
            self._report_paradigm(name, p1_results)
        return p1_results

    def _report_paradigm(self, name: str, results: List[Dict]):
        if not results: return
        
        wrs = np.array([r["wr"] for r in results])
        total_wins = sum(r.get("wins", 0) for r in results)
        total_games = sum(r.get("total", 0) for r in results)
        global_wr = total_wins / total_games if total_games > 0 else np.mean(wrs)
        
        ci = 1.96 * (np.std(wrs) / math.sqrt(len(wrs)))
 
        panel_content = f"Global Winrate: [bold green]{global_wr*100:.2f}%[/bold green]\n"
        panel_content += f"Confidence Interval (95% CI): ±{ci*100:.3f}%\n"
        panel_content += f"Sample Size: {len(results)} matchups"
        
        console.print(Panel(panel_content, title=f"Paradigm: {name}", border_style="green"))

    def run_all(self):
        """Execute the Extensive Chaos Tie-Breaker (Full Matrix Aller-Retour)."""
        self.analyze_data()
        
        # 1. Sample 1 deck per archetype (randomly)
        chaos_decks = []
        for name, arch in self.loader.archetypes.items():
            if arch.decks:
                chaos_decks.append(random.choice(arch.decks))
            
        n = len(chaos_decks)
        total_matchups = n * n # Full matrix
        
        console.print(Panel(
            f"Archetypes Sampled: [bold cyan]{n}[/bold cyan]\n"
            f"Total Matrix Matchups: [bold magenta]{total_matchups}[/bold magenta]\n"
            f"Format: [bold white]Aller-Retour (Fixed 1 game/side)[/bold white]",
            title="[red]Extensive Chaos Mode[/red]",
            border_style="red"
        ))
        
        # 2. Run the matrix
        # Force 1 game per direction for Chaos balance
        self.num_games = 1
        
        results = self.run_paradigm("Extensive Chaos", chaos_decks, chaos_decks)
        
        # 3. Report and Save
        all_results = {"Extensive Chaos": results}
        self._print_final_summary(all_results)
        self.save_results(all_results)

    def save_results(self, all_results: Dict):
        """Save results to eval_reports directory."""
        report_dir = Path("eval_reports")
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"scientific_eval_{self.model_code}_{timestamp}"
        
        # JSON Report
        json_path = report_dir / f"{filename_base}.json"
        
        # We need to make results JSON serializable (DeckInfo -> dict)
        serializable_results = {}
        for paradigm, matchups in all_results.items():
            serializable_results[paradigm] = matchups

        report_data = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_code,
            "global_stats": {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in self.stats.items()},
            "paradigms": serializable_results
        }
        
        with open(json_path, "w") as f:
            json.dump(report_data, f, indent=2)
            
        # Markdown summary report
        md_path = report_dir / f"{filename_base}.md"
        with open(md_path, "w") as f:
            f.write(f"# Scientific Evaluation Report: {self.model_code}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Global Dataset Statistics\n")
            f.write("| Metric | Value |\n| :--- | :--- |\n")
            for k, v in self.stats.items():
                val = f"{v:.4f}" if isinstance(v, float) else str(v)
                f.write(f"| {k} | {val} |\n")
            f.write("\n")
            
            f.write("## Paradigm Summary\n")
            f.write("| Paradigm | Global Winrate | 95% CI | Matchups |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            
            for name, results in all_results.items():
                if not results: continue
                wrs = [r["wr"] for r in results]
                total_wins = sum(r.get("wins", 0) for r in results)
                total_games = sum(r.get("total", 0) for r in results)
                global_wr = total_wins / total_games if total_games > 0 else np.mean(wrs)
                
                ci = 1.96 * np.std(wrs) / math.sqrt(len(wrs))
                f.write(f"| {name} | {global_wr*100:.2f}% | ±{ci*100:.2f}% | {len(results)} |\n")
                
        console.print(f"\n[bold green]Report saved to:[/bold green] {json_path}")
        console.print(f"[bold green]Markdown summary saved to:[/bold green] {md_path}")

    def _print_final_summary(self, all_results: Dict):
        summary_table = Table(title="[bold yellow]Scientific Evaluation Summary[/bold yellow]", show_lines=True)
        summary_table.add_column("Paradigm", style="cyan")
        summary_table.add_column("Global Winrate", justify="right", style="green")
        summary_table.add_column("Precision (95% CI)", justify="right")
        summary_table.add_column("Matchups", justify="right")
        
        for name, results in all_results.items():
            if not results: continue
            wrs = [r["wr"] for r in results]
            ci = 1.96 * np.std(wrs) / math.sqrt(len(wrs))
            
            total_wins = sum(r.get("wins", 0) for r in results)
            total_games = sum(r.get("total", 0) for r in results)
            global_wr = total_wins / total_games if total_games > 0 else np.mean(wrs)
            
            summary_table.add_row(
                name,
                f"{global_wr*100:.2f}%",
                f"±{ci*100:.3f}%",
                str(len(results))
            )
            
        console.print("\n")
        console.print(summary_table)

def main():
    parser = argparse.ArgumentParser(description="DeckGym Scientific Evaluation")
    parser.add_argument("--meta", default="meta_deck.json", help="Path to meta_deck.json")
    parser.add_argument("--model", default="o1", help="Model code (e.g. o1, o1c, aa, r)")
    
    args = parser.parse_args()
    
    evaluator = ScientificEvaluator(args.meta, args.model)
    evaluator.run_all()

if __name__ == "__main__":
    main()
