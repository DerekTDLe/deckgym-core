#!/usr/bin/env python3
"""
League Logger for training progress tracking.

Handles winrate calculations (fair vs unfair) and TensorBoard integration.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from stable_baselines3.common.logger import Logger

from deckgym.league.pool import OpponentPool


class LeagueLogger:
    """
    Handles logging of league metrics to TensorBoard and console.
    
    Responsibilities:
    - Calculating winrates against different pools (Self-Play, Baselines).
    - Isolating milestone winrates (e.g., against e2).
    - Recording metrics via SB3 Logger.
    """

    def __init__(self, pool: OpponentPool, logger: Optional[Logger] = None, verbose: int = 1):
        self.pool = pool
        self.logger = logger
        self.verbose = verbose

    def _is_fair_baseline(self, baseline_code: str) -> bool:
        """Omniscient bots (e/m) are excluded from fair winrate metrics."""
        lower = baseline_code.lower()
        if lower in ["er", "et"]:
            return True
        if lower.startswith("e") or lower.startswith("m"):
            return False
        return True

    def calculate_winrate(self, names: List[str]) -> float:
        """Calculate average agent winrate against a specific list of opponent names."""
        if not names:
            return 0.5
            
        winrates = []
        for name in names:
            data = self.pool.get_data(name)
            if not data:
                continue
                
            total = data["wins"] + data["losses"] + data["draws"]
            if total > 0:
                # Agent winrate = Opponent losses / Total
                winrates.append(data["losses"] / total)
            else:
                winrates.append(0.5)
                
        return float(np.mean(winrates)) if winrates else 0.5

    def get_self_play_winrate(self) -> float:
        """Average agent winrate against non-baseline opponents."""
        names = [n for n, d in self.pool.opponents.items() if not d.get("is_baseline")]
        return self.calculate_winrate(names)

    def get_baseline_winrate(self, only_fair: bool = True) -> Optional[float]:
        """Average agent winrate against baseline opponents."""
        names = [
            n for n, d in self.pool.opponents.items() 
            if d.get("is_baseline") and (not only_fair or self._is_fair_baseline(d.get("baseline_code", "")))
        ]
        if not names:
            return None
        return self.calculate_winrate(names)

    def log_metrics(self, rollout_count: int):
        """Record league metrics to TensorBoard/Console."""
        sp_wr = self.get_self_play_winrate()
        bl_wr_fair = self.get_baseline_winrate(only_fair=True)
        
        # Specific milestones
        e2_wr = self.calculate_winrate(["baseline_e2"]) if "baseline_e2" in self.pool.opponents else None

        if self.logger:
            self.logger.record("pfsp/winrate_self_play", sp_wr)
            if bl_wr_fair is not None:
                self.logger.record("pfsp/winrate_baselines_fair", bl_wr_fair)
            
            if e2_wr is not None:
                self.logger.record("pfsp/winrate_e2", e2_wr)

            # Global "Fair" Winrate
            fair_metrics = [sp_wr]
            if bl_wr_fair is not None:
                fair_metrics.append(bl_wr_fair)
            self.logger.record("pfsp/winrate_global_fair", np.mean(fair_metrics))

        if self.verbose > 0:
            bl_str = f", Baselines: {bl_wr_fair:.1%}" if bl_wr_fair is not None else ""
            print(f"[League Summary] Rollout {rollout_count} | Self-Play WR: {sp_wr:.1%}{bl_str}")

    def log_detailed_info(self, rollout_results: Dict[str, Dict[str, int]]):
        """
        Log detailed performance against each opponent.
        Format: opponent: Rollout WR% (Ag Wins - Ag Losses - Draws) Total WR% (Ag Wins - Ag Losses - Draws)
        """
        if self.verbose <= 0:
            return

        print("\n" + "="*100)
        print(f"{'Opponent Name':25} | {'Rollout (Agent Wins-Losses-Draws)':^35} | {'Total (Agent Wins-Losses-Draws)':^32}")
        print("-" * 100)
        
        # Sort opponents: baselines first, then models by age
        sorted_names = sorted(
            self.pool.opponents.keys(), 
            key=lambda n: (not self.pool.opponents[n].get("is_baseline"), self.pool.opponents[n].get("added_at_step", 0))
        )

        for name in sorted_names:
            data = self.pool.opponents[name]
            
            # Total stats (Lifetime)
            tw, tl, td = data.get("total_wins", 0), data.get("total_losses", 0), data.get("total_draws", 0)
            total = tw + tl + td
            t_wr = (tl / total * 100) if total > 0 else 50.0
            
            # Rollout stats (Current rollout)
            r_stats = rollout_results.get(name, {"wins": 0, "losses": 0, "draws": 0})
            rw, rl, rd = r_stats["wins"], r_stats["losses"], r_stats["draws"]
            r_total = rw + rl + rd
            r_wr = (rl / r_total * 100) if r_total > 0 else 50.0
            
            # Agent Perspective: WR% is based on opponent's losses (which are agent's wins)
            print(f"{name:25} |  {r_wr:5.1f}% ({rl:3}-{rw:3}-{rd:3}) |  {t_wr:5.1f}% ({tl:5}-{tw:5}-{td:5})")
            
        print("="*100 + "\n")
