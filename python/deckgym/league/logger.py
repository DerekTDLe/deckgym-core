#!/usr/bin/env python3
"""
League Logger for training progress tracking.

Simplified metrics:
- WR Global: Agent winrate against all opponents EXCEPT omniscient bots (e2, e3, etc.)
- WR vs e2: Agent winrate specifically against e2 (expectiminimax depth 2)

Baseline Codes:
- v, w: Simple heuristic bots (ValueFunction, WeightedRandom)
- aa, er: Attach-Attack, Evolution Rusher
- e2, e3, e4: Expectiminimax (depth 2/3/4) - omniscient, excluded from global WR
- o[n][device]: ONNX model (n=index from newest, device=c/g/t for cpu/cuda/trt)
  Examples: o1t = newest ONNX on TensorRT, o2c = 2nd newest on CPU
"""

from typing import Dict, List, Optional
import numpy as np
from stable_baselines3.common.logger import Logger

from deckgym.league.pool import OpponentPool


class LeagueLogger:
    """
    Handles logging of league metrics to TensorBoard and console.

    Simplified to two key metrics:
    - WR Global (e2 excluded): Overall performance against fair opponents
    - WR vs e2: Benchmark against optimal play
    """

    def __init__(
        self, pool: OpponentPool, logger: Optional[Logger] = None, verbose: int = 1
    ):
        self.pool = pool
        self.logger = logger
        self.verbose = verbose

    def _is_omniscient(self, baseline_code: str) -> bool:
        """Check if baseline is omniscient (e[n] except er/et)."""
        lower = baseline_code.lower()
        if lower in ["er", "et"]:
            return False
        if lower.startswith("e") and len(lower) >= 2 and lower[1:].isdigit():
            return True
        if lower.startswith("m"):
            return True
        return False

    def _is_onnx_model(self, baseline_code: str) -> bool:
        """Check if baseline is an ONNX model (o[n][device])."""
        return baseline_code.lower().startswith("o")

    def _get_winrate(self, name: str) -> float:
        """Get agent winrate against a specific opponent."""
        data = self.pool.get_data(name)
        if not data:
            return 0.5
        total = data["wins"] + data["losses"] + data["draws"]
        if total > 0:
            return data["losses"] / total  # Agent wins = opponent losses
        return 0.5

    def get_global_winrate(self) -> float:
        """
        Agent winrate against ALL opponents EXCEPT omniscient bots.
        Includes: self-play, fair baselines, ONNX models.
        """
        names = []
        for name, data in self.pool.opponents.items():
            if data.get("is_baseline"):
                code = data.get("baseline_code", "")
                if self._is_omniscient(code):
                    continue
            names.append(name)

        if not names:
            return 0.5

        winrates = [self._get_winrate(n) for n in names]
        return float(np.mean(winrates))

    def get_e2_winrate(self) -> Optional[float]:
        """Agent winrate specifically against e2."""
        if "baseline_e2" not in self.pool.opponents:
            return None
        return self._get_winrate("baseline_e2")

    def get_self_play_winrate(self) -> float:
        """Backward compatibility: average agent winrate against non-baseline opponents."""
        names = [n for n, d in self.pool.opponents.items() if not d.get("is_baseline")]
        if not names:
            return 0.5
        winrates = [self._get_winrate(n) for n in names]
        return float(np.mean(winrates))

    def log_metrics(self, rollout_count: int):
        """Record league metrics to TensorBoard/Console."""
        global_wr = self.get_global_winrate()
        e2_wr = self.get_e2_winrate()

        if self.logger:
            self.logger.record("pfsp/winrate_global", global_wr)
            if e2_wr is not None:
                self.logger.record("pfsp/winrate_e2", e2_wr)

        if self.verbose > 0:
            self._print_summary(rollout_count, global_wr, e2_wr)

    def _print_summary(
        self, rollout_count: int, global_wr: float, e2_wr: Optional[float]
    ):
        """Print beautiful CLI summary."""
        e2_str = f"{e2_wr:.1%}" if e2_wr is not None else "—"

        print(f"\n┌{'─'*50}┐")
        print(f"│{'LEAGUE STATUS':^50}│")
        print(f"├{'─'*50}┤")
        print(
            f"│  Rollout: {rollout_count:<10} Pool: {self.pool.total_count} agents{' '*(18-len(str(self.pool.total_count)))}│"
        )
        print(f"├{'─'*50}┤")
        print(f"│  📊 WR Global (e2 excl.):  {global_wr:>6.1%}{' '*17}│")
        print(f"│  🎯 WR vs e2:              {e2_str:>6}{' '*17}│")
        print(f"└{'─'*50}┘")

    def log_detailed_info(self, rollout_results: Dict[str, Dict[str, int]]):
        """Log detailed performance against each opponent."""
        if self.verbose <= 0:
            return

        # Sort: baselines first (omniscient last), then models by age
        def sort_key(name):
            data = self.pool.opponents.get(name, {})
            is_baseline = data.get("is_baseline", False)
            code = data.get("baseline_code", "")
            is_omni = self._is_omniscient(code) if is_baseline else False
            return (not is_baseline, is_omni, data.get("added_at_step", 0))

        sorted_names = sorted(self.pool.opponents.keys(), key=sort_key)

        print(f"\n┌{'─'*72}┐")
        print(f"│{'DETAILED OPPONENT BREAKDOWN':^72}│")
        print(f"├{'─'*25}┬{'─'*22}┬{'─'*22}┤")
        print(f"│{'Opponent':^25}│{'This Rollout':^22}│{'All-Time':^22}│")
        print(f"├{'─'*25}┼{'─'*22}┼{'─'*22}┤")

        for name in sorted_names:
            data = self.pool.opponents[name]

            # Display name with type indicator
            if data.get("is_baseline"):
                code = data.get("baseline_code", "?")
                if self._is_omniscient(code):
                    display = f"⚠️  {name}"
                elif self._is_onnx_model(code):
                    display = f"🤖 {name}"
                else:
                    display = f"🎮 {name}"
            else:
                display = f"📸 {name}"

            # Truncate if too long
            display = display[:24] if len(display) > 24 else display

            # Total stats
            tw = data.get("total_wins", 0)
            tl = data.get("total_losses", 0)
            td = data.get("total_draws", 0)
            total = tw + tl + td
            t_wr = (tl / total * 100) if total > 0 else 50.0

            # Rollout stats
            r = rollout_results.get(name, {"wins": 0, "losses": 0, "draws": 0})
            rw, rl, rd = r["wins"], r["losses"], r["draws"]
            r_total = rw + rl + rd
            r_wr = (rl / r_total * 100) if r_total > 0 else 50.0

            # Format: WR% W-L-D
            r_str = (
                f"{r_wr:5.1f}% {rl:2}-{rw:2}-{rd:2}" if r_total > 0 else "     —      "
            )
            t_str = f"{t_wr:5.1f}% {tl:3}-{tw:3}-{td:3}"

            print(f"│{display:^25}│{r_str:^22}│{t_str:^22}│")

        print(f"└{'─'*25}┴{'─'*22}┴{'─'*22}┘")
        print()
