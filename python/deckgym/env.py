#!/usr/bin/env python3
"""
DeckGymEnv - Gymnasium environment for Pokemon TCG Pocket

This environment allows an RL agent to play Pokemon TCG Pocket
via the deckgym-core Rust simulator.
"""

import gymnasium as gym
import numpy as np


class DeckGymEnv(gym.Env):
    """
    Gymnasium environment to play Pokémon TCG Pocket.

    Observation space:
        Box(2219,) - Flattened game state tensor including:
        - Global info (41): turn, points, deck/hand/discard sizes, energy types
        - Card features (18 cards × 121 features = 2178): HP, energy, attacks, status, abilities, position

    Action space:
        Discrete(175) - Canonical action indices covering all SimpleAction variants:
        ... (rest of action space doc)

    Reward:
        Score-based reward: 1.0 + (point_diff / 6.0) for wins, -1.0 - (point_diff / 6.0) for losses. Multiplied by a speed factor.
    """

    def __init__(self, deck_a_str: str, deck_b_str: str, seed: int = None):
        """
        Initialize the environment.

        Args:
            deck_a_str: Deck string for player A (deck file format)
            deck_b_str: Deck string for player B
            seed: Optional random seed for reproducibility
        """
        super().__init__()
        import deckgym

        self.deck_a_str = deck_a_str
        self.deck_b_str = deck_b_str
        self.seed = seed

        # Gym spaces
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(deckgym.Game.observation_size(),),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(deckgym.Game.action_space_size())

        self.game = None

    def reset(self, seed=None, options=None):
        """
        Reset the environment and start a new game.

        Returns:
            observation: Initial game state as numpy array
            info: Empty dict (required by Gymnasium API)
        """
        super().reset(seed=seed)
        import deckgym

        game_seed = seed if seed is not None else self.seed
        self.game = deckgym.Game.from_deck_strings(
            self.deck_a_str, self.deck_b_str, seed=game_seed
        )

        obs = self._sanitize_obs(np.array(self.game.get_obs(), dtype=np.float32))
        return obs, {}

    def step(self, action: int):
        """
        Execute an action in the environment.

        Args:
            action: Action index to execute

        Returns:
            observation: New game state
            reward: +1 win, -1 loss, 0 otherwise
            terminated: True if game is over
            truncated: Always False (no time limit)
            info: Dict with action description
        """
        try:
            reward, done, info_str = self.game.step_action(action)
        except BaseException as e:
            # Handle Rust panics and invalid actions gracefully
            # End episode with neutral reward to avoid crashing training
            print(f"WARNING: Panic in step_action: {e}")
            try:
                obs = self._sanitize_obs(
                    np.array(self.game.get_obs(), dtype=np.float32)
                )
            except BaseException:
                obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, True, False, {"error": str(e)}

        obs = self._sanitize_obs(np.array(self.game.get_obs(), dtype=np.float32))
        return obs, reward, done, False, {"action": info_str}

    def _sanitize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Check and fix corrupted observations containing NaN or inf values."""
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            bad_indices = np.where(np.isnan(obs) | np.isinf(obs))[0]
            print(
                f"WARNING: Corrupted observation at indices {bad_indices[:10]}... Sanitizing."
            )
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    def action_masks(self) -> np.ndarray:
        """
        Return valid action mask for MaskablePPO.

        Returns:
            Boolean array where True = valid action
        """
        try:
            mask = np.array(self.game.get_action_mask(), dtype=np.bool_)
        except BaseException as e:
            # Rust panic - game in corrupted state
            # Reset and return EndTurn-only mask to recover
            print(f"WARNING: Panic in action_masks, resetting game: {e}")
            self.reset()
            mask = np.zeros(self.action_space.n, dtype=np.bool_)
            mask[0] = True  # Only EndTurn valid
            return mask

        # Fallback: if no actions valid, force EndTurn (action 0) to continue
        # This handles edge cases where the game is stuck
        if not mask.any():
            mask[0] = True  # EndTurn is always index 0

        return mask
