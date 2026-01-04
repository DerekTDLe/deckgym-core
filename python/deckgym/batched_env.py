#!/usr/bin/env python3
"""
BatchedDeckGymEnv - High-performance vectorized environment using Rust-side batching.

This provides ~3-5x speedup over DummyVecEnv by reducing Python/Rust FFI overhead
from O(n_envs) calls per step to O(1) calls per batch step.

Compatible with Stable-Baselines3's VecEnv interface.
"""

import numpy as np
from typing import List, Tuple, Optional, Any
import gymnasium as gym

# SB3 VecEnv imports
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn


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
    ):
        """
        Initialize the batched environment.
        
        Args:
            n_envs: Number of parallel environments
            deck_loader: Object with sample_pair() method returning (deck_a_str, deck_b_str)
            opponent_type: Bot opponent type ("e2", "e3", etc.) or None for self-play
            seed: Optional random seed
        """
        import deckgym
        
        self.n_envs = n_envs
        self.deck_loader = deck_loader
        self.opponent_type = opponent_type
        
        # Sample initial deck pairs
        deck_pairs = [self._sample_deck_pair() for _ in range(n_envs)]
        
        # Create the Rust-side vectorized game
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
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )
        action_space = gym.spaces.Discrete(action_size)
        
        # Initialize parent VecEnv
        super().__init__(n_envs, observation_space, action_space)
        
        # Cached action masks (updated after each step)
        self._action_masks: Optional[np.ndarray] = None
        
        # Async step state
        self._pending_actions: Optional[np.ndarray] = None
    
    def _sample_deck_pair(self) -> Tuple[str, str]:
        """Sample a deck pair from the loader.
        
        Handles both MetaDeckLoader (sample_deck) and CurriculumDeckLoader (sample_pair).
        """
        if hasattr(self.deck_loader, 'sample_pair'):
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
        
        # Single FFI call steps all environments!
        obs_flat, rewards, dones, masks_flat, terminal_obs = self.vec_game.step_batch(actions)
        
        # Reshape outputs
        obs = np.array(obs_flat, dtype=np.float32).reshape(self.n_envs, -1)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=bool)
        
        # Cache action masks
        self._action_masks = np.array(masks_flat, dtype=bool).reshape(self.n_envs, -1)
        
        # Build infos with terminal observations for SB3
        infos: List[dict] = [{} for _ in range(self.n_envs)]
        for env_idx, term_obs in terminal_obs:
            infos[env_idx]["terminal_observation"] = np.array(term_obs, dtype=np.float32)
        
        # Sample new deck pairs for environments that finished
        for i, done in enumerate(dones):
            if done:
                deck_a, deck_b = self._sample_deck_pair()
                self.vec_game.reset_single(i, deck_a, deck_b)
        
        return obs, rewards, dones, infos
    
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
    
    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> List[Any]:
        """Call method on environments (not supported)."""
        raise NotImplementedError("env_method is not supported for BatchedDeckGymEnv")
    
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
        self._action_masks = np.array(masks_flat, dtype=bool).reshape(self.n_envs, action_size)
    
    # -------------------------------------------------------------------------
    # Curriculum Support
    # -------------------------------------------------------------------------
    
    def set_deck_loader(self, deck_loader) -> None:
        """Update the deck loader (e.g., for curriculum transitions)."""
        self.deck_loader = deck_loader
    
    def set_opponent_type(self, opponent_type: Optional[str]) -> None:
        """Update opponent type.
        
        Note: This requires creating a new VecGame instance.
        Call reset() after changing opponent type.
        """
        if opponent_type != self.opponent_type:
            self.opponent_type = opponent_type
            # Recreate with new opponent type
            import deckgym
            deck_pairs = [self._sample_deck_pair() for _ in range(self.n_envs)]
            self.vec_game = deckgym.VecGame(
                deck_pairs,
                opponent_type=opponent_type,
            )
    
    def get_turn_counts(self) -> List[int]:
        """Get current turn counts for all environments (for debugging)."""
        return self.vec_game.get_turn_counts()
