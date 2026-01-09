#!/usr/bin/env python3
"""
Prioritized Fictitious Self-Play (PFSP) Callback.

Implements AlphaStar-style league training where the agent trains against
a pool of historical checkpoints, prioritizing harder opponents.

Reference: https://www.nature.com/articles/s41586-019-1724-z
"""

from collections import Counter
import os
import gc
import tempfile
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO

from deckgym.batched_env import BatchedDeckGymEnv
from deckgym.onnx_export import export_policy_to_onnx
from deckgym.config import (
    DEFAULT_CONFIG,
    PFSP_NEUTRAL_WINRATE,
    PFSP_MIN_EPISODES_FOR_UPDATE,
    PFSP_ADAPTIVE_DOMINATION_THRESHOLD,
)


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
        n_envs: int = DEFAULT_CONFIG.n_envs,
        pool_size: int = DEFAULT_CONFIG.pfsp_pool_size,
        add_to_pool_every_n_rollouts: int = DEFAULT_CONFIG.pfsp_add_every_n_rollouts,
        priority_exponent: float = DEFAULT_CONFIG.pfsp_priority_exponent,
        winrate_window: int = DEFAULT_CONFIG.pfsp_winrate_window,
        checkpoint_dir: str = DEFAULT_CONFIG.pfsp_checkpoint_dir,
        # Baseline curriculum parameters
        baseline_slots: int = DEFAULT_CONFIG.pfsp_baseline_slots,
        baseline_max_allocation: float = DEFAULT_CONFIG.pfsp_baseline_max_allocation,
        baseline_curriculum: list = None,
        verbose: int = 1,
        **kwargs, # Accept legacy args but ignore them
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
            baseline_slots: Number of permanent baseline slots (never evicted)
            baseline_max_allocation: Max fraction of envs that can play vs baselines
            baseline_curriculum: List of (step_threshold, [baseline_codes]) tuples
            verbose: Verbosity level (0=silent, 1=normal, 2=debug)
        """
        super().__init__(verbose)
        self.env = env
        self.n_envs = n_envs
        self.pool_size = pool_size
        self.add_to_pool_every_n_rollouts = add_to_pool_every_n_rollouts
        self.priority_exponent = priority_exponent
        self.winrate_window = winrate_window
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Baseline curriculum settings
        self.baseline_slots = baseline_slots
        self.baseline_max_allocation = baseline_max_allocation
        self.baseline_curriculum = baseline_curriculum or DEFAULT_CONFIG.pfsp_baseline_curriculum
        self.current_baselines: list = []  # Currently active baseline codes
        self.baseline_envs_count = max(1, int(n_envs * baseline_max_allocation))  # How many envs for baselines

        # State tracking
        self.rollout_count = 0
        self.current_opponent_name = None
        self._frozen_model = None

        # Per-env opponent tracking (for pool mode)
        self.env_opponent_names = [
            None
        ] * n_envs  # Track which opponent each env is using
        self.pool_mode = False  # True if using pool with per-env opponents
        
        # New: Tracking for fair statistics (skip episodes that were mid-game when opponent changed)
        self.skip_next_episode = [False] * n_envs

        # Opponent pool: {name: {"path": str, "onnx_path": str, "wins": int, "losses": int, "added_at_step": int, "is_baseline": bool}}
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
            print(f"[PFSP] Per-episode opponent selection active")
            print(f"[PFSP] Baseline curriculum: {baseline_slots} slots, {baseline_max_allocation:.0%} max allocation")
            print(f"[PFSP] Baseline stages: {len(self.baseline_curriculum)}")

    def _on_training_start(self) -> None:
        """Called when training starts - initialize pool with current model."""
        if self.verbose > 0:
            print("[PFSP] Training started - adding initial opponent to pool")

        # Initialize baseline curriculum (add initial baselines to pool)
        self._update_baseline_curriculum()

        # Ensure we are in pool mode (BatchedDeckGymEnv with ONNX support)
        if isinstance(self.env, BatchedDeckGymEnv):
            try:
                self._enable_pool_mode()
            except Exception as e:
                raise RuntimeError(f"[PFSP CRITICAL] Failed to initialize pool mode: {e}")
        else:
            raise TypeError("[PFSP CRITICAL] Only BatchedDeckGymEnv is supported (legacy mode removed)")

    def _enable_pool_mode(self):
        """Enable per-episode opponent selection by preloading all pool opponents."""
        from deckgym.onnx_export import export_policy_to_onnx
        import tempfile

        # Clear any existing pool in Rust
        if hasattr(self.env.vec_game, "clear_onnx_pool"):
            self.env.vec_game.clear_onnx_pool()

        # Load all opponents into pool
        for opp_name, opp_data in self.opponent_pool.items():
            # Skip baselines - they use built-in bots, not ONNX
            if opp_data.get("is_baseline", False):
                # Add baseline to Rust pool using built-in bot code
                baseline_code = opp_data.get("baseline_code", opp_name.replace("baseline_", ""))
                try:
                    self.env.vec_game.add_baseline_to_pool(opp_name, baseline_code)
                    if self.verbose > 1:
                        print(f"[PFSP] Added baseline {opp_name} ({baseline_code}) to pool")
                except Exception as e:
                    if self.verbose > 0:
                        print(f"[PFSP WARNING] Failed to add baseline {opp_name}: {e}")
                continue
            
            # Export ONNX opponents if not already done
            if "onnx_path" not in opp_data or not Path(opp_data["onnx_path"]).exists():
                loaded_model = MaskablePPO.load(opp_data["path"], device="cpu")
                onnx_path = os.path.join(
                    tempfile.gettempdir(), f"pfsp_pool_{opp_name}.onnx"
                )
                export_policy_to_onnx(loaded_model, onnx_path, validate=False)
                opp_data["onnx_path"] = onnx_path
                del loaded_model

            # Add to Rust pool
            device = self.env.config.pfsp_opponent_device if hasattr(self.env, "config") else "trt"
            self.env.vec_game.add_onnx_to_pool(
                opp_name, opp_data["onnx_path"], False, device
            )
            if self.verbose > 1:
                print(f"[PFSP] Added {opp_name} to pool")

        # Assign initial opponents to each env
        for env_idx in range(self.n_envs):
            self._assign_opponent_to_env(env_idx)

        self.pool_mode = True

    def _update_baseline_curriculum(self):
        """
        Update baseline opponents based on current training step.
        
        Checks the curriculum stages and swaps baselines when thresholds are crossed.
        Baselines are added to the pool with is_baseline=True (never evicted).
        """
        current_step = self.num_timesteps
        
        # Find the appropriate curriculum stage
        new_baselines = []
        for step_threshold, baselines in sorted(self.baseline_curriculum, key=lambda x: x[0], reverse=True):
            if current_step >= step_threshold:
                new_baselines = baselines
                break
        
        # Check if baselines changed
        if set(new_baselines) == set(self.current_baselines):
            return  # No change needed
        
        # Remove old baselines from pool
        for old_bl in self.current_baselines:
            bl_name = f"baseline_{old_bl}"
            if bl_name in self.opponent_pool:
                del self.opponent_pool[bl_name]
                if self.pool_mode and isinstance(self.env, BatchedDeckGymEnv):
                    try:
                        self.env.vec_game.remove_onnx_from_pool(bl_name)
                    except Exception:
                        pass
        
        # Add new baselines to pool
        for bl_code in new_baselines:
            bl_name = f"baseline_{bl_code}"
            self.opponent_pool[bl_name] = {
                "path": None,  # No path for baselines (they're built-in)
                "baseline_code": bl_code,  # Store the actual code (e.g., "e2", "v")
                "wins": 0,
                "losses": 0,
                "added_at_step": current_step,
                "is_baseline": True,  # Mark as baseline (never evicted)
            }
            
            # Add to Rust pool if in pool mode
            if self.pool_mode and isinstance(self.env, BatchedDeckGymEnv):
                try:
                    # Use the built-in baseline code directly
                    self.env.vec_game.add_baseline_to_pool(bl_name, bl_code)
                    if self.verbose > 0:
                        print(f"[PFSP Baseline] Added {bl_name} ({bl_code}) to Rust pool")
                except Exception as e:
                    if self.verbose > 0:
                        print(f"[PFSP WARNING] Failed to add baseline {bl_name}: {e}")
        
        if self.verbose > 0 and new_baselines != self.current_baselines:
            old_str = ", ".join(self.current_baselines) if self.current_baselines else "none"
            new_str = ", ".join(new_baselines)
            print(f"[PFSP Baseline] Curriculum update at step {current_step}: [{old_str}] -> [{new_str}]")
        
        self.current_baselines = new_baselines

    def _assign_opponent_to_env(self, env_idx: int):
        """
        Assign an opponent to a specific env.
        
        First `baseline_envs_count` envs are reserved for baselines (uniform selection).
        Remaining envs use PFSP priority for opponent selection.
        """
        if not self.opponent_pool:
            return

        # Determine if this env should use baselines or PFSP pool
        baseline_names = [f"baseline_{bl}" for bl in self.current_baselines]
        
        if env_idx < self.baseline_envs_count and baseline_names:
            # This env is reserved for baselines - uniform selection among baselines
            selected_name = np.random.choice(baseline_names)
        else:
            # Use PFSP priority (excluding baselines for fairness)
            selected_name = self._select_opponent_pfsp_for_env(exclude_baselines=True)
            
            # Fallback to any opponent if pool is empty
            if selected_name is None:
                selected_name = self._select_opponent_pfsp_for_env(exclude_baselines=False)

        if selected_name:
            try:
                self.env.vec_game.set_env_opponent(env_idx, selected_name)
                self.env_opponent_names[env_idx] = selected_name
            except Exception as e:
                if self.verbose > 0:
                    print(
                        f"[PFSP WARNING] Failed to set opponent for env {env_idx}: {e}"
                    )

    def _flag_reassigned_envs(self, opponent_name: str):
        """Flag all environments currently playing against a specific opponent to skip their next result."""
        for env_idx, name in enumerate(self.env_opponent_names):
            if name == opponent_name:
                self.skip_next_episode[env_idx] = True
                if self.verbose > 1:
                    print(f"[PFSP Refresh] Flagged env {env_idx} to skip next result (reassigned mid-game)")

    def _get_pool_priorities(self, pool: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate PFSP priorities for a given set of opponents."""
        priorities = {}
        for name, data in pool.items():
            total = data["wins"] + data["losses"]
            opp_winrate = data["wins"] / total if total > 0 else PFSP_NEUTRAL_WINRATE
            priorities[name] = float(opp_winrate**self.priority_exponent)
        return priorities

    def _select_opponent_pfsp_for_env(self, exclude_baselines: bool = False) -> Optional[str]:
        """
        Select opponent using PFSP priority.
        
        Args:
            exclude_baselines: If True, only select from self-play opponents (not baselines)
        """
        # Filter pool based on exclude_baselines
        pool = {
            name: data for name, data in self.opponent_pool.items()
            if not (exclude_baselines and data.get("is_baseline", False))
        }
        
        if not pool:
            return None

        if len(pool) == 1:
            return next(iter(pool.keys()))

        # Calculate PFSP priorities
        priorities = self._get_pool_priorities(pool)
        total_priority = sum(priorities.values())

        if total_priority <= 0:
            return np.random.choice(list(pool.keys()))

        # Select using normalized probabilities
        names = list(pool.keys())
        probs = [priorities[name] / total_priority for name in names]
        return np.random.choice(names, p=probs)

    def _on_rollout_end(self) -> None:
        """Called at end of each rollout collection."""
        self.rollout_count += 1

        # Check for curriculum stage transitions (every 10 rollouts to reduce overhead)
        if self.rollout_count % 10 == 0:
            self._update_baseline_curriculum()

        # Update win rates from recent episodes
        if (
            len(self.episode_results) >= PFSP_MIN_EPISODES_FOR_UPDATE
        ):  # Update every N episodes minimum
            if self.verbose > 0:
                # Count wins/losses per opponent
                wins = Counter(name for name, won in self.episode_results if won)
                losses = Counter(name for name, won in self.episode_results if not won)
                
                print(f"[PFSP Debug] Rollout {self.rollout_count}: {len(self.episode_results)} episodes tracked")
                
                for name in sorted(set(wins.keys()) | set(losses.keys())):
                    w, l = wins[name], losses[name]
                    wr = l / (w + l) if (w + l) > 0 else 0
                    print(f"  {name:15}: agent WR {wr:5.1%}, wins {l:2}, losses {w:2}")
            
            self._update_winrates()
            self.episode_results.clear()
        elif self.verbose > 0 and self.rollout_count % 5 == 0:
            print(
                f"[PFSP Debug] Rollout {self.rollout_count}: only {len(self.episode_results)} episodes (need 10+)"
            )

        # Calculate average agent winrate against pool for adaptive decisions
        avg_agent_winrate = self._get_avg_agent_winrate()

        # Adaptive add frequency: if agent is dominating, add more frequently (Refresh period decreased by 2)
        effective_add_freq = (
            max(1, self.add_to_pool_every_n_rollouts // 2)
            if avg_agent_winrate > PFSP_ADAPTIVE_DOMINATION_THRESHOLD
            else self.add_to_pool_every_n_rollouts
        )

        # Add current agent to pool periodically, but ONLY if performing well
        min_wr_to_add = getattr(self.env.config, "pfsp_min_winrate_to_add", 0.50) if hasattr(self.env, "config") else 0.50

        if self.rollout_count % effective_add_freq == 0:
            if avg_agent_winrate >= min_wr_to_add or len(self.opponent_pool) < 2:
                self._add_to_pool()
                if self.verbose > 0 and avg_agent_winrate >= min_wr_to_add:
                    print(f"[PFSP] Agent winrate {avg_agent_winrate:.1%} >= {min_wr_to_add:.0%}, adding to pool")
            elif self.verbose > 0:
                print(f"[PFSP] Agent winrate {avg_agent_winrate:.1%} < {min_wr_to_add:.0%}, skipping pool add")

    def _get_avg_agent_winrate(self) -> float:
        """Calculate average agent winrate across all opponents in pool."""
        if not self.opponent_pool:
            return PFSP_NEUTRAL_WINRATE

        total_wins = 0
        total_games = 0
        for data in self.opponent_pool.values():
            games = data["wins"] + data["losses"]
            if games > 0:
                # Agent winrate = opponent losses / total
                total_wins += data["losses"]
                total_games += games

        return total_wins / total_games if total_games > 0 else PFSP_NEUTRAL_WINRATE

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
            if not info or "episode" not in info:
                continue

            # Episode ended: check if agent won
            agent_won = info["episode"].get("r", 0) > 0
            opp_name = self.env_opponent_names[env_idx]

            if opp_name:
                # Only record result if the episode was "full" (not reassigned mid-game)
                if self.skip_next_episode[env_idx]:
                    self.skip_next_episode[env_idx] = False
                    if self.verbose > 1:
                        print(f"[PFSP Refresh] Skipping result for env {env_idx} (reassigned mid-game)")
                else:
                    self.episode_results.append((opp_name, agent_won))

            # Reassign opponent for this env on episode end
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
                # Use round() to avoid systematic bias
                self.opponent_pool[opp_name]["wins"] = round(
                    self.opponent_pool[opp_name]["wins"] * scale
                )
                self.opponent_pool[opp_name]["losses"] = round(
                    self.opponent_pool[opp_name]["losses"] * scale
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
                device = (
                    getattr(self.env.config, "pfsp_opponent_device", "trt")
                    if hasattr(self.env, "config")
                    else "trt"
                )
                self.env.vec_game.add_onnx_to_pool(
                    checkpoint_name, onnx_path, False, device
                )
                if self.verbose > 0:
                    print(f"[PFSP] Added {checkpoint_name} to Rust pool")
            except Exception as e:
                if self.verbose > 0:
                    print(f"[PFSP WARNING] Failed to add to Rust pool: {e}")

        # Remove WEAKEST opponent if pool full (keep hardest opponents!)
        # But NEVER remove: baselines or the one we just added
        non_baseline_pool = {n: d for n, d in self.opponent_pool.items() if not d.get("is_baseline", False)}
        
        while len(non_baseline_pool) > self.pool_size:
            # Rank candidates by winrate
            priorities = self._get_pool_priorities(non_baseline_pool)
            
            # Filter candidates for eviction (exclude the one we just added)
            candidates = [
                (name, priorities[name], int(data.get("added_at_step", 0)))
                for name, data in non_baseline_pool.items()
                if name != checkpoint_name
            ]

            if candidates:
                # Remove candidate with LOWEST priority (easiest to beat)
                # If tied, remove oldest one
                candidates.sort(key=lambda x: (x[1], x[2]))
                weakest_name = candidates[0][0]
                weakest_wr = candidates[0][1] ** (1/self.priority_exponent) if self.priority_exponent != 0 else 0
                
                # 1. Flag environments to skip next result and reassign
                self._flag_reassigned_envs(weakest_name)
                
                # 2. Cleanup files and pool
                weakest_data = self.opponent_pool.pop(weakest_name)
                non_baseline_pool.pop(weakest_name)

                if weakest_data.get("path"):
                    p = Path(weakest_data["path"])
                    if p.exists(): p.unlink()

                try:
                    self.env.vec_game.remove_onnx_from_pool(weakest_name)
                except Exception as e:
                    if self.verbose > 0: print(f"[PFSP WARNING] Failed to remove {weakest_name} from Rust: {e}")

                # 3. Force reassign envs
                for env_idx, name in enumerate(self.env_opponent_names):
                    if name == weakest_name:
                        self._assign_opponent_to_env(env_idx)

                if self.verbose > 0:
                    print(f"[PFSP] Evicted weakest: {weakest_name} (WR: {weakest_wr:.1%})")
                
                gc.collect()
            else:
                break

        # Refresh Logic: Reset WR stats for all models in the pool at each addition
        for data in self.opponent_pool.values():
            data["wins"] = 0
            data["losses"] = 0
        
        if self.verbose > 0:
            print(f"[PFSP Refresh] Pool size: {len(self.opponent_pool)}, statistics reset")
            print(f"[PFSP →Pool] Saved agent as {checkpoint_name}")

    def get_pool_stats(self) -> Dict:
        """Return statistics about the opponent pool."""
        if not self.opponent_pool:
            return {"pool_size": 0}

        priorities = self._get_pool_priorities(self.opponent_pool)
        total_prio = sum(priorities.values())

        winrates = {}
        for name, data in self.opponent_pool.items():
            total = data["wins"] + data["losses"]
            winrates[name] = data["wins"] / total if total > 0 else PFSP_NEUTRAL_WINRATE

        return {
            "pool_size": len(self.opponent_pool),
            "pool_winrates": winrates,
            "avg_opponent_winrate": np.mean(list(winrates.values())) if winrates else 0.5,
            "selection_probs": {n: p / total_prio for n, p in priorities.items()} if total_prio > 0 else {}
        }
