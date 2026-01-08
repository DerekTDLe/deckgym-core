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

        # Per-env opponent tracking (for pool mode)
        self.env_opponent_names = [
            None
        ] * n_envs  # Track which opponent each env is using
        self.pool_mode = False  # True if using pool with per-env opponents

        # Opponent pool: {name: {"path": str, "onnx_path": str, "wins": int, "losses": int, "added_at_step": int}}
        self.opponent_pool: Dict[str, Dict] = {}

        # Episode tracking: [(opponent_name, won: bool), ...]
        self.episode_results = []

        if self.verbose > 0:
            print(
                f"[PFSP] Initialized with pool_size={pool_size}, priority_exp={priority_exponent}"
            )
            print(
                f"[PFSP] Adding to pool every {add_to_pool_every_n_rollouts} rollouts"
            )
            print(
                f"[PFSP] Selecting opponent every {select_opponent_every_n_rollouts} rollouts"
            )
            print(f"[PFSP] Per-episode opponent selection ENABLED")

    def _on_training_start(self) -> None:
        """Called when training starts - initialize pool with current model."""
        if self.verbose > 0:
            print("[PFSP] Training started - adding initial opponent to pool")

        # Add current (untrained) model to pool as initial opponent
        self._add_to_pool()

        # Check if we can use pool mode (BatchedDeckGymEnv with ONNX support)
        if isinstance(self.env, BatchedDeckGymEnv):
            try:
                # Try to enable pool mode
                self._enable_pool_mode()
                if self.verbose > 0:
                    print(
                        f"[PFSP] Pool mode ENABLED - per-episode opponent selection active"
                    )
            except Exception as e:
                print(
                    f"[PFSP WARNING] Pool mode not available, falling back to single opponent: {e}"
                )
                self.pool_mode = False

        # If not pool mode, select first opponent as usual
        if not self.pool_mode and self.opponent_pool:
            first_opponent = list(self.opponent_pool.keys())[0]
            self._set_opponent(
                first_opponent, self.opponent_pool[first_opponent]["path"]
            )
            if self.verbose > 0:
                print(f"[PFSP] Initial opponent set (legacy mode): {first_opponent}")

    def _enable_pool_mode(self):
        """Enable per-episode opponent selection by preloading all pool opponents."""
        from deckgym.onnx_export import export_policy_to_onnx
        import tempfile

        # Clear any existing pool in Rust
        if hasattr(self.env.vec_game, "clear_onnx_pool"):
            self.env.vec_game.clear_onnx_pool()

        # Load all opponents into pool
        for opp_name, opp_data in self.opponent_pool.items():
            # Export to ONNX if not already done
            if "onnx_path" not in opp_data or not Path(opp_data["onnx_path"]).exists():
                loaded_model = MaskablePPO.load(opp_data["path"], device="cpu")
                onnx_path = os.path.join(
                    tempfile.gettempdir(), f"pfsp_pool_{opp_name}.onnx"
                )
                export_policy_to_onnx(loaded_model, onnx_path, validate=False)
                opp_data["onnx_path"] = onnx_path
                del loaded_model

            # Add to Rust pool
            self.env.vec_game.add_onnx_to_pool(opp_name, opp_data["onnx_path"], False)
            if self.verbose > 1:
                print(f"[PFSP] Added {opp_name} to pool")

        # Assign initial opponents to each env
        for env_idx in range(self.n_envs):
            self._assign_opponent_to_env(env_idx)

        self.pool_mode = True

    def _assign_opponent_to_env(self, env_idx: int):
        """Assign an opponent to a specific env using PFSP priority."""
        if not self.opponent_pool:
            return

        # Use PFSP priority to select opponent
        selected_name = self._select_opponent_pfsp_for_env()

        if selected_name:
            try:
                self.env.vec_game.set_env_opponent(env_idx, selected_name)
                self.env_opponent_names[env_idx] = selected_name
            except Exception as e:
                if self.verbose > 0:
                    print(
                        f"[PFSP WARNING] Failed to set opponent for env {env_idx}: {e}"
                    )

    def _select_opponent_pfsp_for_env(self) -> Optional[str]:
        """Select opponent using PFSP priority (same as _select_opponent_pfsp but returns name only)."""
        if len(self.opponent_pool) == 0:
            return None

        if len(self.opponent_pool) == 1:
            return list(self.opponent_pool.keys())[0]

        # Calculate PFSP priorities
        priorities = {}
        for name, data in self.opponent_pool.items():
            total = data["wins"] + data["losses"]
            if total == 0:
                opp_winrate = 0.5
            else:
                opp_winrate = data["wins"] / total
            priorities[name] = opp_winrate**self.priority_exponent

        # Sample
        total_priority = sum(priorities.values())
        if total_priority <= 0:
            return np.random.choice(list(self.opponent_pool.keys()))

        probs = [priorities[name] / total_priority for name in self.opponent_pool]
        return np.random.choice(list(self.opponent_pool.keys()), p=probs)

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
                print(
                    f"[PFSP Debug] Rollout {self.rollout_count}: {len(self.episode_results)} episodes tracked"
                )
                for name in set(wins.keys()) | set(losses.keys()):
                    print(
                        f"  {name}: agent won {wins[name]}, agent lost {losses[name]}"
                    )
            self._update_winrates()
            self.episode_results.clear()
        elif self.verbose > 0 and self.rollout_count % 5 == 0:
            print(
                f"[PFSP Debug] Rollout {self.rollout_count}: only {len(self.episode_results)} episodes (need 10+)"
            )

        # Calculate average agent winrate against pool for adaptive decisions
        avg_agent_winrate = self._get_avg_agent_winrate()

        # Adaptive add frequency: if agent is dominating (>65%), add more frequently
        effective_add_freq = (
            4 if avg_agent_winrate > 0.65 else self.add_to_pool_every_n_rollouts
        )

        # Add current agent to pool periodically, but ONLY if performing well (>50%)
        # NOTE: Lowered from 60% to 50% to prevent pool stagnation (Run #8 analysis)
        if self.rollout_count % effective_add_freq == 0:
            if avg_agent_winrate >= 0.50 or len(self.opponent_pool) < 2:
                # Add if winning enough OR pool needs more opponents
                self._add_to_pool()
                if self.verbose > 0 and avg_agent_winrate >= 0.50:
                    print(
                        f"[PFSP] Agent winrate {avg_agent_winrate:.1%} >= 50%, adding to pool"
                    )
            elif self.verbose > 0:
                print(
                    f"[PFSP] Agent winrate {avg_agent_winrate:.1%} < 50%, skipping pool add"
                )

        # Select new opponent periodically (can be different from add frequency)
        if self.rollout_count % self.select_opponent_every_n_rollouts == 0:
            self._select_opponent_pfsp()

    def _get_avg_agent_winrate(self) -> float:
        """Calculate average agent winrate across all opponents in pool."""
        if not self.opponent_pool:
            return 0.5

        total_wins = 0
        total_games = 0
        for data in self.opponent_pool.values():
            games = data["wins"] + data["losses"]
            if games > 0:
                # Agent winrate = opponent losses / total
                total_wins += data["losses"]
                total_games += games

        return total_wins / total_games if total_games > 0 else 0.5

    def _on_step(self) -> bool:
        """Called at each env step - track episode outcomes and reassign opponents."""
        # Collect episode results from infos
        # Note: self.locals is a dict, so we use 'in' not hasattr
        infos = self.locals.get("infos", [])
        if not infos:
            # Fallback for non-vectorized envs
            info = self.locals.get("info", {})
            infos = [info] if info else []

        for env_idx, info in enumerate(infos):
            if info and "episode" in info:
                # Episode ended: check if agent won
                episode_reward = info["episode"].get("r", 0)
                agent_won = episode_reward > 0

                # Get opponent name for this env (pool mode or single opponent)
                if self.pool_mode and env_idx < len(self.env_opponent_names):
                    opp_name = self.env_opponent_names[env_idx]
                else:
                    opp_name = self.current_opponent_name

                if opp_name:
                    self.episode_results.append((opp_name, agent_won))
                    if self.verbose > 1:
                        print(
                            f"[PFSP] Episode tracked: env{env_idx} vs {opp_name}, reward={episode_reward:.2f}, agent_won={agent_won}"
                        )

                # In pool mode, reassign opponent for this env on episode end
                if self.pool_mode and env_idx < self.n_envs:
                    self._assign_opponent_to_env(env_idx)

        return True

    def _update_winrates(self):
        """Update opponent win rates based on recent episode results."""
        # First, accumulate all results
        for opp_name, agent_won in self.episode_results:
            if opp_name in self.opponent_pool:
                if agent_won:
                    self.opponent_pool[opp_name]["losses"] += 1  # Opponent lost
                else:
                    self.opponent_pool[opp_name]["wins"] += 1  # Opponent won

        # Then, apply decay ONCE per opponent (not per episode!)
        for opp_name in self.opponent_pool:
            total = (
                self.opponent_pool[opp_name]["wins"]
                + self.opponent_pool[opp_name]["losses"]
            )
            if total > self.winrate_window:
                # Scale down to keep within window
                scale = self.winrate_window / total
                # Use round() instead of int() to avoid systematic bias
                self.opponent_pool[opp_name]["wins"] = max(
                    1, round(self.opponent_pool[opp_name]["wins"] * scale)
                )
                self.opponent_pool[opp_name]["losses"] = max(
                    1, round(self.opponent_pool[opp_name]["losses"] * scale)
                )

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

        # In pool mode, also add to Rust pool
        if self.pool_mode and isinstance(self.env, BatchedDeckGymEnv):
            try:
                from deckgym.onnx_export import export_policy_to_onnx
                import tempfile

                # Export to ONNX
                onnx_path = os.path.join(
                    tempfile.gettempdir(), f"pfsp_pool_{checkpoint_name}.onnx"
                )
                export_policy_to_onnx(self.model, onnx_path, validate=False)
                self.opponent_pool[checkpoint_name]["onnx_path"] = onnx_path

                # Add to Rust pool
                self.env.vec_game.add_onnx_to_pool(checkpoint_name, onnx_path, False)
                if self.verbose > 0:
                    print(f"[PFSP] Added {checkpoint_name} to Rust pool")
            except Exception as e:
                if self.verbose > 0:
                    print(f"[PFSP WARNING] Failed to add to Rust pool: {e}")

        # Remove WEAKEST opponent if pool full (keep hardest opponents!)
        # But NEVER remove current opponent
        if len(self.opponent_pool) > self.pool_size:
            # Find opponent with LOWEST winrate (easiest for agent)
            candidates = []
            for name, data in self.opponent_pool.items():
                # In pool mode, don't remove if any env is using this opponent
                if self.pool_mode and name in self.env_opponent_names:
                    continue
                if name == self.current_opponent_name:
                    continue  # Never remove current opponent
                total = data["wins"] + data["losses"]
                if total > 0:
                    opp_wr = data["wins"] / total
                else:
                    opp_wr = 0.5  # Untested opponents get neutral priority
                candidates.append((name, opp_wr))

            if candidates:
                # Remove the opponent with LOWEST winrate (easiest to beat)
                weakest_name = min(candidates, key=lambda x: x[1])[0]
                weakest_path = Path(self.opponent_pool[weakest_name]["path"])
                weakest_wr = min(candidates, key=lambda x: x[1])[1]
                if weakest_path.exists():
                    weakest_path.unlink()  # Delete checkpoint file

                # Also remove from Rust pool if in pool mode
                # Note: We'd need to implement remove_from_pool in Rust, for now just clear and reload

                del self.opponent_pool[weakest_name]
                gc.collect()

                if self.verbose > 0:
                    print(
                        f"[PFSP] Evicted weakest opponent: {weakest_name} (WR: {weakest_wr:.1%})"
                    )

        if self.verbose > 0:
            print(
                f"[PFSP →Pool] Saved agent as {checkpoint_name} (pool size: {len(self.opponent_pool)})"
            )

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
            name: (wr**self.priority_exponent) for name, wr in winrates.items()
        }

        # Normalize to probabilities
        total_priority = sum(priorities.values())
        if total_priority == 0:
            # All priorities zero (shouldn't happen): uniform distribution
            probs = np.ones(len(self.opponent_pool)) / len(self.opponent_pool)
            selected_name = np.random.choice(list(self.opponent_pool.keys()))
        else:
            probs = np.array(
                [
                    priorities[name] / total_priority
                    for name in self.opponent_pool.keys()
                ]
            )
            selected_name = np.random.choice(list(self.opponent_pool.keys()), p=probs)

        # Load and set opponent
        selected_path = self.opponent_pool[selected_name]["path"]
        self._set_opponent(selected_name, selected_path)

        if self.verbose > 0:
            wr = winrates[selected_name]
            print(
                f"[PFSP Pool→] Loading opponent: {selected_name} "
                f"(opp WR: {wr:.1%}, priority: {priorities[selected_name]:.3f})"
            )

        if self.verbose > 1:
            # Debug: show all opponents and their priorities
            print(f"[PFSP Debug] Pool state:")
            for name in sorted(
                self.opponent_pool.keys(), key=lambda k: priorities[k], reverse=True
            ):
                wr = winrates[name]
                prio = priorities[name]
                prob = prio / total_priority if total_priority > 0 else 0
                w = self.opponent_pool[name]["wins"]
                l = self.opponent_pool[name]["losses"]
                print(
                    f"  {name}: WR={wr:.1%} ({w}W-{l}L), prio={prio:.3f}, p(select)={prob:.1%}"
                )

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
                onnx_path = os.path.join(
                    tempfile.gettempdir(), f"pfsp_{opponent_name}.onnx"
                )
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
                inner_env = (
                    self.env.env.env if hasattr(self.env.env, "env") else self.env.env
                )
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
            "avg_opponent_winrate": (
                np.mean(list(winrates.values())) if winrates else 0.5
            ),
        }
