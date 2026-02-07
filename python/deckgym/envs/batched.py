#!/usr/bin/env python3
"""
BatchedDeckGymEnv - High-performance vectorized environment using Rust-side batching.

This provides ~3-5x speedup over DummyVecEnv by reducing Python/Rust FFI overhead
from O(n_envs) calls per step to O(1) calls per batch step.

Compatible with Stable-Baselines3's VecEnv interface.
"""

import numpy as np
import traceback
from typing import List, Tuple, Optional, Any
import gymnasium as gym

# SB3 VecEnv imports
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvObs,
    VecEnvStepReturn,
)

from deckgym.config import DEFAULT_CONFIG


class BatchedDeckGymEnv(VecEnv):
    """
    High-performance vectorized environment using Rust-side batching.

    This is a drop-in replacement for DummyVecEnv that provides much faster
    training by executing all environment steps in a single FFI call.

    Compatible with SB3's MaskablePPO.
    """

    def __init__(
        self,
        n_envs: int,
        deck_loader,
        opponent_type: Optional[str] = "e2",
        seed: Optional[int] = None,
        config=None,
    ):
        """
        Initialize the batched environment.

        Args:
            n_envs: Number of parallel environments
            deck_loader: Object with sample_pair() method returning (deck_a_str, deck_b_str)
            opponent_type: Bot opponent type ("e2", "e3", etc.) or None for self-play
            seed: Optional random seed
            config: Optional TrainingConfig object
        """
        import deckgym

        self.n_envs = n_envs
        self.deck_loader = deck_loader
        self.opponent_type = opponent_type

        # Sample initial deck pairs
        deck_pairs = [self._sample_deck_pair() for _ in range(n_envs)]

        # Create the Rust-side vectorized game
        if config is not None:
            self.config = config
        else:
            self.config = getattr(deck_loader, "config", DEFAULT_CONFIG)

        self.vec_game = deckgym.VecGame(
            deck_pairs,
            base_seed=seed,
            opponent_type=opponent_type,
        )

        # Get dimensions from Rust
        obs_size = deckgym.VecGame.observation_size()
        action_size = deckgym.VecGame.action_space_size()

        # Define spaces
        observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        action_space = gym.spaces.Discrete(action_size)

        # Initialize parent VecEnv
        super().__init__(n_envs, observation_space, action_space)

        # Cached action masks (updated after each step)
        self._action_masks: Optional[np.ndarray] = None

        # Async step state
        self._pending_actions: Optional[np.ndarray] = None

        # Episode tracking for SB3 logging (ep_rew_mean, ep_len_mean)
        self._episode_rewards = np.zeros(n_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(n_envs, dtype=np.int32)
        self._episode_actions_since_end_turn = np.zeros(n_envs, dtype=np.int32)
        self._episode_turn_count = np.zeros(n_envs, dtype=np.int32)
        self._episode_total_actions_per_turn = np.zeros(n_envs, dtype=np.float32)

        from deckgym.diagnostic_logger import get_logger

        self.diagnostic_logger = get_logger()

    def _sample_deck_pair(self) -> Tuple[str, str]:
        """Sample a deck pair from the loader.

        Handles both MetaDeckLoader (sample_deck) and CurriculumDeckLoader (sample_pair).
        """
        if hasattr(self.deck_loader, "sample_pair"):
            return self.deck_loader.sample_pair()
        else:
            # MetaDeckLoader only has sample_deck - use same deck for both sides
            deck = self.deck_loader.sample_deck()
            return (deck, deck)

    def reset(self) -> VecEnvObs:
        """Reset all environments and return initial observations."""
        # Sample new deck pairs for all envs
        deck_pairs = [self._sample_deck_pair() for _ in range(self.n_envs)]
        self.vec_game.set_deck_pairs(deck_pairs)

        # Reset and get observations
        obs_flat = self.vec_game.reset_all()
        obs = np.array(obs_flat, dtype=np.float32).reshape(self.n_envs, -1)

        # Update cached action masks
        self._update_action_masks()

        # Reset episode tracking
        self._episode_rewards[:] = 0
        self._episode_lengths[:] = 0
        self._episode_actions_since_end_turn[:] = 0
        self._episode_turn_count[:] = 0
        self._episode_total_actions_per_turn[:] = 0

        return obs

    def step_async(self, actions: np.ndarray) -> None:
        """Store actions for later execution in step_wait()."""
        self._pending_actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        """Execute pending actions and return results."""
        if self._pending_actions is None:
            raise RuntimeError("step_wait called without step_async")

        actions = self._pending_actions.tolist()
        self._pending_actions = None

        # Track actions per turn (action 0 = EndTurn)
        for i, action in enumerate(actions):
            self._episode_actions_since_end_turn[i] += 1
            if action == 0:  # EndTurn action
                self._episode_turn_count[i] += 1
                self._episode_total_actions_per_turn[
                    i
                ] += self._episode_actions_since_end_turn[i]
                self._episode_actions_since_end_turn[i] = 0
            self.diagnostic_logger.record_action(i, action)

        # Single FFI call steps all environments with panic protection
        try:
            obs_flat, rewards, dones, masks_flat, terminal_obs = (
                self.vec_game.step_batch(actions)
            )
        except BaseException as e:
            print(f"\n{'!' * 60}")
            print(f"CRITICAL: Panic in vec_game.step_batch: {e}")
            traceback.print_exc()
            print(f"{'!' * 60}\n")

            # Try to log states for all envs to help debug the panic
            for i in range(self.n_envs):
                try:
                    state = self.vec_game.get_state(i)
                    self.diagnostic_logger.log_error(
                        "batch_panic", i, state, {"error": str(e), "actions": actions}
                    )
                except Exception as dump_err:
                    # If get_state also fails, log with None state
                    self.diagnostic_logger.log_error(
                        "batch_panic_corrupted",
                        i,
                        None,
                        {"error": str(e), "dump_error": str(dump_err)},
                    )
            raise e

        # Reshape outputs
        obs = np.array(obs_flat, dtype=np.float32).reshape(self.n_envs, -1)
        rewards_arr = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=bool)

        # Update episode tracking
        self._episode_rewards += rewards_arr
        self._episode_lengths += 1

        # Cache action masks
        self._action_masks = np.array(masks_flat, dtype=bool).reshape(self.n_envs, -1)

        # Build infos with terminal observations and episode stats for SB3
        infos: List[dict] = [{} for _ in range(self.n_envs)]
        for env_idx, term_obs in terminal_obs:
            infos[env_idx]["terminal_observation"] = np.array(
                term_obs, dtype=np.float32
            )

        # Add episode info for finished episodes (SB3 looks for this)
        for i, done in enumerate(dones):
            if done:
                # Calculate actions per turn
                avg_actions_per_turn = self._episode_total_actions_per_turn[i] / max(
                    1, self._episode_turn_count[i]
                )

                infos[i]["episode"] = {
                    "r": self._episode_rewards[i],
                    "l": self._episode_lengths[i],
                    "actions_per_turn": avg_actions_per_turn,
                }

                # Reset episode tracking for this env
                self._episode_rewards[i] = 0
                self._episode_lengths[i] = 0
                self._episode_actions_since_end_turn[i] = 0
                self._episode_turn_count[i] = 0
                self._episode_total_actions_per_turn[i] = 0
                # Note: Rust step_batch already auto-resets on done, no need for reset_single

        return obs, rewards_arr, dones, infos

    def close(self) -> None:
        """Clean up resources."""
        pass  # Rust handles cleanup via Drop

    def get_attr(self, attr_name: str, indices=None) -> List[Any]:
        """Get attribute from environments.

        Handles common attributes required by SB3.
        """
        if indices is None:
            indices = range(self.n_envs)

        # Handle common attributes that SB3 expects
        if attr_name == "render_mode":
            return [None] * len(list(indices))
        elif attr_name == "spec":
            return [None] * len(list(indices))
        else:
            # Return None for unknown attributes instead of raising
            return [None] * len(list(indices))

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        """Set attribute on environments (not supported)."""
        raise NotImplementedError("set_attr is not supported for BatchedDeckGymEnv")

    def env_method(
        self, method_name: str, *method_args, indices=None, **method_kwargs
    ) -> List[Any]:
        """Call method on environments.

        Supports action_masks which is required by MaskablePPO.
        """
        if indices is None:
            indices = range(self.n_envs)

        if method_name == "action_masks":
            # Return action masks for each env as a list
            all_masks = self.action_masks()
            return [all_masks[i] for i in indices]
        else:
            # Return None for unsupported methods
            return [None] * len(list(indices))

    def env_is_wrapped(self, wrapper_class, indices=None) -> List[bool]:
        """Check if environments are wrapped."""
        return [False] * self.n_envs

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        """Set random seeds (handled at creation time)."""
        return [None] * self.n_envs

    # -------------------------------------------------------------------------
    # MaskablePPO Support
    # -------------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Return action masks for MaskablePPO.

        Returns:
            Boolean array of shape (n_envs, action_size) where True = valid action
        """
        if self._action_masks is None:
            self._update_action_masks()
        return self._action_masks

    def _update_action_masks(self) -> None:
        """Update cached action masks from Rust."""
        masks_flat = self.vec_game.get_action_masks()
        action_size = self.action_space.n
        self._action_masks = np.array(masks_flat, dtype=bool).reshape(
            self.n_envs, action_size
        )

    # -------------------------------------------------------------------------
    # Curriculum Support
    # -------------------------------------------------------------------------

    def set_deck_loader(self, deck_loader) -> None:
        """Update the deck loader (e.g., for curriculum transitions)."""
        self.deck_loader = deck_loader

    def set_opponent_type(self, opponent_type: Optional[str]) -> None:
        """Update opponent type.

        Note: This requires creating a new VecGame instance.
        Automatically calls reset() after changing opponent type.
        """
        if opponent_type != self.opponent_type:
            self.opponent_type = opponent_type
            # Recreate VecGame instance with new opponent type
            import deckgym

            deck_pairs = [self._sample_deck_pair() for _ in range(self.n_envs)]
            self.vec_game = deckgym.VecGame(deck_pairs, opponent_type=opponent_type)
            # Automatically reset since the internal state in Rust is now fresh
            self.reset()

    def get_turn_counts(self) -> List[int]:
        """Get current turn counts for all environments (for debugging)."""
        return self.vec_game.get_turn_counts()
