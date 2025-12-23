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
        Box(755,) - Flattened game state tensor including:
        - Global info (17): turn, points, deck/hand/discard sizes, energy types
        - Board state (704): 8 slots × 88 features (HP, energy, attacks, status, abilities)
        - Hand features (34): Pokemon counts, trainer categories, can-evolve flags

    Action space:
        Discrete(130) - Canonical action indices covering all SimpleAction variants:
        - 0: EndTurn
        - 1-2: Attack (index 0, 1)
        - 3-5: Retreat to bench position (1, 2, 3)
        - 6-9: Attach energy to position (0, 1, 2, 3)
        - 10-19: Place Pokémon from hand
        - 20-29: Evolve Pokémon from hand
        - 30-39: Play trainer from hand
        - 40-43: Use ability (position 0-3)
        - 44: DrawCard
        - 45: Noop (decline optional action)
        - 46-49: Activate bench Pokémon
        - 50-53: AttachTool
        - 54-57: Heal
        - 58-61: DiscardFossil
        - 62-65: AttachFromDiscard
        - 66-75: CommunicatePokemon
        - 76-85: ShufflePokemonIntoDeck
        - 86-95: ShuffleOpponentSupporter
        - 96-105: DiscardOpponentSupporter
        - 106-115: DiscardOwnCard
        - 116-129: Special actions (EeveeBag, MoveEnergy, etc.)

    Reward:
        +1.0 for winning, -1.0 for losing, 0.0 otherwise.
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
            dtype=np.float32
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
            self.deck_a_str,
            self.deck_b_str,
            seed=game_seed
        )

        obs = np.array(self.game.get_obs(), dtype=np.float32)
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
        except Exception as e:
            # Handle Rust panics and invalid actions gracefully
            # End episode with neutral reward to avoid crashing training
            try:
                obs = np.array(self.game.get_obs(), dtype=np.float32)
            except Exception:
                obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, True, False, {"error": str(e)}
        
        obs = np.array(self.game.get_obs(), dtype=np.float32)
        return obs, reward, done, False, {"action": info_str}

    def action_masks(self) -> np.ndarray:
        """
        Return valid action mask for MaskablePPO.

        Returns:
            Boolean array where True = valid action
        """
        mask = np.array(self.game.get_action_mask(), dtype=np.bool_)
        
        # Fallback: if no actions valid, force EndTurn (action 0) to continue
        # This handles edge cases where the game is stuck
        if not mask.any():
            mask[0] = True  # EndTurn is always index 0
        
        return mask