#!/usr/bin/env python3
"""
Prioritized Fictitious Self-Play (PFSP) Callback.

Implements AlphaStar-style league training where the agent trains against
a pool of historical checkpoints, prioritizing harder opponents.

Reference: https://www.nature.com/articles/s41586-019-1724-z
"""

import os
import gc
import tempfile
from typing import Dict, Optional, Tuple
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO

from deckgym.batched_env import BatchedDeckGymEnv


class PFSPCallback(BaseCallback):
    """
    Prioritized Fictitious Self-Play callback.

    Maintains a pool of historical checkpoints and selects opponents based on
    their win rate against the current agent (harder opponents = higher priority).

    Algorithm:
    1. Every N rollouts: save current agent to pool
    2. Track win rate for each opponent in pool
    3. Select next opponent using priority function: f(x) = (1 - winrate)^p
    4. Export selected opponent to ONNX and set as frozen opponent

    This ensures the agent focuses on its weaknesses and prevents exploitation
    of a single frozen opponent's flaws.
    """

    def __init__(
        self,
        env,
        n_envs: int = 32,
        pool_size: int = 10,
        add_to_pool_every_n_rollouts: int = 8,
        select_opponent_every_n_rollouts: int = 2,
        priority_exponent: float = 2.0,
        winrate_window: int = 200,
        checkpoint_dir: str = "./checkpoints/pfsp_pool/",
        verbose: int = 1,
    ):
        """
        Initialize PFSP callback.

        Args:
            env: Training environment (BatchedDeckGymEnv or DummyVecEnv)
            n_envs: Number of parallel environments
            pool_size: Maximum number of opponents to keep in pool
            add_to_pool_every_n_rollouts: Add current agent to pool every N rollouts
            select_opponent_every_n_rollouts: Select new opponent every N rollouts
            priority_exponent: Exponent for PFSP priority (f(x) = (1-x)^p)
                Higher = focus more on hard opponents (typical: 1.0-3.0)
            winrate_window: Number of recent episodes to track per opponent
            checkpoint_dir: Directory to save opponent checkpoints
            verbose: Verbosity level (0=silent, 1=normal, 2=debug)
        """
        super().__init__(verbose)
        self.env = env
        self.n_envs = n_envs
        self.pool_size = pool_size
        self.add_to_pool_every_n_rollouts = add_to_pool_every_n_rollouts
        self.select_opponent_every_n_rollouts = select_opponent_every_n_rollouts
        self.priority_exponent = priority_exponent
        self.winrate_window = winrate_window
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # State tracking
        self.rollout_count = 0
        self.current_opponent_name = None
        self._frozen_model = None

        # Opponent pool: {name: {"path": str, "wins": int, "losses": int, "added_at_step": int}}
        self.opponent_pool: Dict[str, Dict] = {}

        # Episode tracking: [(opponent_name, won: bool), ...]
        self.episode_results = []

        if self.verbose > 0:
            print(f"[PFSP] Initialized with pool_size={pool_size}, priority_exp={priority_exponent}")
            print(f"[PFSP] Adding to pool every {add_to_pool_every_n_rollouts} rollouts")
            print(f"[PFSP] Selecting opponent every {select_opponent_every_n_rollouts} rollouts")

    def _on_training_start(self) -> None:
        """Called when training starts - initialize pool with current model."""
        if self.verbose > 0:
            print("[PFSP] Training started - adding initial opponent to pool")
        
        # Add current (untrained) model to pool as initial opponent
        self._add_to_pool()
        
        # Select it as the first opponent
        if self.opponent_pool:
            first_opponent = list(self.opponent_pool.keys())[0]
            self._set_opponent(first_opponent, self.opponent_pool[first_opponent]["path"])
            if self.verbose > 0:
                print(f"[PFSP] Initial opponent set: {first_opponent}")

    def _on_rollout_end(self) -> None:
        """Called at end of each rollout collection."""
        self.rollout_count += 1

        # Update win rates from recent episodes
        if len(self.episode_results) >= 10:  # Update every 10 episodes minimum
            if self.verbose > 0:
                # Count wins/losses per opponent
                from collections import Counter
                wins = Counter(name for name, won in self.episode_results if won)
                losses = Counter(name for name, won in self.episode_results if not won)
                print(f"[PFSP Debug] Rollout {self.rollout_count}: {len(self.episode_results)} episodes tracked")
                for name in set(wins.keys()) | set(losses.keys()):
                    print(f"  {name}: agent won {wins[name]}, agent lost {losses[name]}")
            self._update_winrates()
            self.episode_results.clear()
        elif self.verbose > 0 and self.rollout_count % 5 == 0:
            print(f"[PFSP Debug] Rollout {self.rollout_count}: only {len(self.episode_results)} episodes (need 10+)")

        # Add current agent to pool periodically
        if self.rollout_count % self.add_to_pool_every_n_rollouts == 0:
            self._add_to_pool()

        # Select new opponent periodically (can be different from add frequency)
        if self.rollout_count % self.select_opponent_every_n_rollouts == 0:
            self._select_opponent_pfsp()

    def _on_step(self) -> bool:
        """Called at each env step - track episode outcomes."""
        # Collect episode results from infos
        # Note: self.locals is a dict, so we use 'in' not hasattr
        infos = self.locals.get("infos", [])
        if not infos:
            # Fallback for non-vectorized envs
            info = self.locals.get("info", {})
            infos = [info] if info else []

        for info in infos:
            if info and "episode" in info and self.current_opponent_name:
                # Episode ended: check if agent won
                episode_reward = info["episode"].get("r", 0)
                agent_won = episode_reward > 0
                self.episode_results.append((self.current_opponent_name, agent_won))
                if self.verbose > 1:
                    print(f"[PFSP] Episode tracked: vs {self.current_opponent_name}, reward={episode_reward:.2f}, agent_won={agent_won}")

        return True

    def _update_winrates(self):
        """Update opponent win rates based on recent episode results."""
        # First, accumulate all results
        for opp_name, agent_won in self.episode_results:
            if opp_name in self.opponent_pool:
                if agent_won:
                    self.opponent_pool[opp_name]["losses"] += 1  # Opponent lost
                else:
                    self.opponent_pool[opp_name]["wins"] += 1   # Opponent won
        
        # Then, apply decay ONCE per opponent (not per episode!)
        for opp_name in self.opponent_pool:
            total = self.opponent_pool[opp_name]["wins"] + self.opponent_pool[opp_name]["losses"]
            if total > self.winrate_window:
                # Scale down to keep within window
                scale = self.winrate_window / total
                # Use round() instead of int() to avoid systematic bias
                self.opponent_pool[opp_name]["wins"] = max(1, round(self.opponent_pool[opp_name]["wins"] * scale))
                self.opponent_pool[opp_name]["losses"] = max(1, round(self.opponent_pool[opp_name]["losses"] * scale))

    def _add_to_pool(self):
        """Save current model to opponent pool."""
        checkpoint_name = f"pfsp_{self.num_timesteps // 1000}k"
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.zip"

        # Save current model
        self.model.save(str(checkpoint_path))

        # Add to pool
        self.opponent_pool[checkpoint_name] = {
            "path": str(checkpoint_path),
            "wins": 0,
            "losses": 0,
            "added_at_step": self.num_timesteps,
        }

        # Remove oldest if pool full (FIFO), but NEVER remove current opponent
        if len(self.opponent_pool) > self.pool_size:
            # Find oldest that is NOT the current opponent
            candidates = [
                (name, data["added_at_step"])
                for name, data in self.opponent_pool.items()
                if name != self.current_opponent_name
            ]
            if candidates:
                oldest_name = min(candidates, key=lambda x: x[1])[0]
                oldest_path = Path(self.opponent_pool[oldest_name]["path"])
                if oldest_path.exists():
                    oldest_path.unlink()  # Delete checkpoint file
                del self.opponent_pool[oldest_name]
                gc.collect()

                if self.verbose > 0:
                    print(f"[PFSP] Removed oldest opponent: {oldest_name}")

        if self.verbose > 0:
            print(f"[PFSP Rollout {self.rollout_count}] Added {checkpoint_name} to pool (size: {len(self.opponent_pool)})")

    def _select_opponent_pfsp(self):
        """Select opponent using PFSP priority function."""
        if not self.opponent_pool:
            if self.verbose > 0:
                print("[PFSP] Pool empty, skipping opponent selection")
            return

        # Calculate win rates for each opponent
        winrates = {}
        for name, data in self.opponent_pool.items():
            total = data["wins"] + data["losses"]
            if total > 0:
                # Opponent's win rate against agent
                winrates[name] = data["wins"] / total
            else:
                # New opponent: assume 50% win rate
                winrates[name] = 0.5

        # PFSP priority: higher priority for opponents that beat the agent
        # f(x) = (1 - agent_winrate)^p = opponent_winrate^p
        priorities = {
            name: (wr ** self.priority_exponent)
            for name, wr in winrates.items()
        }

        # Normalize to probabilities
        total_priority = sum(priorities.values())
        if total_priority == 0:
            # All priorities zero (shouldn't happen): uniform distribution
            probs = np.ones(len(self.opponent_pool)) / len(self.opponent_pool)
            selected_name = np.random.choice(list(self.opponent_pool.keys()))
        else:
            probs = np.array([priorities[name] / total_priority
                            for name in self.opponent_pool.keys()])
            selected_name = np.random.choice(list(self.opponent_pool.keys()), p=probs)

        # Load and set opponent
        selected_path = self.opponent_pool[selected_name]["path"]
        self._set_opponent(selected_name, selected_path)

        if self.verbose > 0:
            wr = winrates[selected_name]
            print(f"[PFSP Rollout {self.rollout_count}] Selected opponent: {selected_name} "
                  f"(opp WR: {wr:.1%}, priority: {priorities[selected_name]:.3f})")

        if self.verbose > 1:
            # Debug: show all opponents and their priorities
            print(f"[PFSP Debug] Pool state:")
            for name in sorted(self.opponent_pool.keys(),
                             key=lambda k: priorities[k], reverse=True):
                wr = winrates[name]
                prio = priorities[name]
                prob = prio / total_priority if total_priority > 0 else 0
                w = self.opponent_pool[name]["wins"]
                l = self.opponent_pool[name]["losses"]
                print(f"  {name}: WR={wr:.1%} ({w}W-{l}L), prio={prio:.3f}, p(select)={prob:.1%}")

    def _set_opponent(self, opponent_name: str, checkpoint_path: str):
        """Load checkpoint and set as current opponent."""
        # Clean up old frozen model
        if self._frozen_model is not None:
            del self._frozen_model
            gc.collect()

        self.current_opponent_name = opponent_name

        if isinstance(self.env, BatchedDeckGymEnv):
            # BatchedDeckGymEnv: export to ONNX for Rust-side inference
            try:
                from deckgym.onnx_export import export_policy_to_onnx

                # Load checkpoint temporarily
                loaded_model = MaskablePPO.load(checkpoint_path, device="cpu")

                # Export to ONNX
                onnx_path = os.path.join(tempfile.gettempdir(), f"pfsp_{opponent_name}.onnx")
                export_policy_to_onnx(loaded_model, onnx_path, validate=False)

                # Set in VecGame
                self.env.vec_game.set_onnx_opponent(onnx_path, deterministic=False)

                # Clean up loaded model
                del loaded_model
                gc.collect()

            except AttributeError as e:
                print(f"[PFSP WARNING] ONNX opponent not available: {e}")
            except Exception as e:
                print(f"[PFSP WARNING] Failed to set ONNX opponent: {e}")
        else:
            # Non-batched envs: use CPU model
            self._frozen_model = MaskablePPO.load(checkpoint_path, device="cpu")

            if self.n_envs == 1:
                # Single environment
                inner_env = self.env.env.env if hasattr(self.env.env, 'env') else self.env.env
                inner_env.set_opponent_model(self._frozen_model)
            else:
                # Multiple environments (DummyVecEnv)
                for i in range(self.n_envs):
                    self.env.envs[i].env.env.set_opponent_model(self._frozen_model)

    def get_pool_stats(self) -> Dict:
        """Return statistics about the opponent pool (for logging/monitoring)."""
        if not self.opponent_pool:
            return {
                "pool_size": 0,
                "current_opponent": None,
            }

        winrates = {}
        for name, data in self.opponent_pool.items():
            total = data["wins"] + data["losses"]
            if total > 0:
                winrates[name] = data["wins"] / total
            else:
                winrates[name] = 0.5

        return {
            "pool_size": len(self.opponent_pool),
            "current_opponent": self.current_opponent_name,
            "pool_winrates": winrates,
            "avg_opponent_winrate": np.mean(list(winrates.values())) if winrates else 0.5,
        }
