#!/usr/bin/env python3
"""
train.py - MaskablePPO training script for Pokemon TCG Pocket

This script trains an RL agent using self-play and curriculum learning.
The agent learns to play Pokemon TCG Pocket against copies of itself,
starting with simple decks and progressively moving to meta decks.

LEARNING EXERCISE: Fill in the TODO sections below!
"""

import os
import random
from typing import Optional

import numpy as np
import gymnasium as gym
import torch

# RL modules
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

# Custom modules
from deckgym.env import DeckGymEnv
from deckgym.deck_loader import CurriculumDeckLoader

# PART 1: Self-Play Environment Wrapper with Frozen Opponent

class SelfPlayEnv(gym.Env):
    """
    Wrapper that lets player 0 (training agent) play against player 1 (frozen opponent).
    
    The frozen opponent uses a separate model that is updated periodically.
    This prevents self-play collapse where both players converge to degenerate strategies.
    """
    
    MAX_TURNS = 99  # Official Pokemon TCG Pocket rule
    MAX_ACTIONS_PER_TURN = 50  # Safety limit for infinite loop detection
    
    def __init__(self, deck_loader, opponent_model=None):
        """
        Args:
            deck_loader: CurriculumDeckLoader to sample decks
            opponent_model: Optional frozen model for opponent (player 1)
        """
        self.deck_loader = deck_loader
        self.opponent_model = opponent_model
        
        # Create a dummy game to get observation/action space sizes
        self._env = DeckGymEnv(*deck_loader.sample_pair())
        
        # Copy observation_space and action_space from the inner env
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
    
    def set_opponent_model(self, model):
        """Update the frozen opponent model."""
        self.opponent_model = model
    
    def reset(self, seed=None, options=None):
        """
        Reset for a new game with random decks.
        
        Returns:
            observation: Initial state of the game
            info: Empty dict
        """
        deck_a, deck_b = self.deck_loader.sample_pair()
        self._env = DeckGymEnv(deck_a, deck_b, seed=seed)      
        obs, info = self._env.reset(seed=seed)
        
        # If it's opponent's turn at start, play their moves
        obs, info, _, _ = self._play_opponent_turns(obs, info)
        
        return obs, info
    
    def _play_opponent_turns(self, obs, info):
        """
        Play all opponent (player 1) turns until it's player 0's turn or game over.
        Returns final obs, info, and any reward/done from opponent's actions.
        """
        final_reward = 0.0
        done = False
        actions = 0  # Count actions in this turn sequence
        
        while self._env.game.current_player() == 1 and not self._env.game.is_game_over():
            actions += 1
            if actions > self.MAX_ACTIONS_PER_TURN:
                print(f"WARNING: Opponent stuck in single turn after {actions} actions, forcing game end")
                return obs, info, 0.0, True  # Force game end with neutral reward
            if self.opponent_model is not None:
                # Use frozen model to select action
                action_mask = self._env.action_masks()
                action, _ = self.opponent_model.predict(obs, action_masks=action_mask, deterministic=False)
            else:
                # Fallback: random valid action
                action_mask = self._env.action_masks()
                valid_actions = np.where(action_mask)[0]
                action = np.random.choice(valid_actions)
            
            obs, reward, done, truncated, info = self._env.step(action)
            # Invert reward since it's from opponent's perspective
            final_reward = -reward
            if done or truncated:
                break
        
        return obs, info, final_reward, done
    
    def step(self, action):
        """
        Execute player 0's action, then play opponent's turn(s).
        
        Returns:
            obs: New state (from player 0's perspective)
            reward: +1 win, -1 loss, 0 otherwise
            terminated: True if game over
            truncated: True if max turns exceeded
            info: Dict with action info
        """
        # Check max turns limit (official rule)
        turn_count = getattr(self._env.game, 'turn_count', lambda: 0)
        if callable(turn_count):
            turn_count = turn_count()
        if turn_count > self.MAX_TURNS:
            obs = np.array(self._env.game.get_obs(), dtype=np.float32)
            return obs, 0.0, False, True, {"info": "max_turns_99_exceeded"}
        
        # Execute training agent's action (player 0)
        obs, reward, done, truncated, info = self._env.step(action)
        
        if not done and not truncated:
            # Play opponent's (player 1) turn(s)
            obs, info, opp_reward, opp_done = self._play_opponent_turns(obs, info)
            
            # If game ended during opponent's turn, use opponent's reward (inverted)
            if opp_done:
                done = True
                reward = opp_reward
        
        return obs, reward, done, truncated, info
    
    def action_masks(self):
        """Return action mask for player 0."""
        return self._env.action_masks()


# PART 2: Curriculum Callback

class CurriculumCallback(BaseCallback):
    """
    Callback who increases difficulty progressively.
    
    SB3 calls this callback at each step of training.
    We use it to increase the ratio of meta decks over time.
    
    CONCEPT: Curriculum Learning
    - At the start: 100% simple decks (difficulty = 0)
    - At the end: 100% meta decks (difficulty = 1)
    - The transition is progressive
    """
    
    def __init__(self, deck_loader, total_timesteps: int, warmup_steps: int = 100_000, verbose: int = 0):
        """
        Args:
            deck_loader: CurriculumDeckLoader
            total_timesteps: Total number of training steps
            warmup_steps: Steps before starting to increase difficulty
        """
        super().__init__(verbose)
        self.deck_loader = deck_loader
        self.total_timesteps = total_timesteps
        self.warmup_steps = warmup_steps
    
    def _on_step(self) -> bool:
        """
        Called at each training step (SB3 requires _on_step, not on_step)
        
        Returns:
            True to continue training
        """
        # Avoid division by zero if warmup_steps >= total_timesteps
        if self.warmup_steps >= self.total_timesteps:
            difficulty = 1.0  # Skip warmup, go straight to full difficulty
        else:
            progress = (self.num_timesteps - self.warmup_steps) / (self.total_timesteps - self.warmup_steps)
            difficulty = max(0.0, min(1.0, progress))
        
        self.deck_loader.set_difficulty(difficulty)
        
        return True


class FrozenOpponentCallback(BaseCallback):
    """
    Callback that updates the frozen opponent model periodically.
    
    This prevents self-play collapse by ensuring the opponent doesn't
    change too quickly, providing a more stable training target.
    """
    
    def __init__(self, env, update_freq: int = 50_000, verbose: int = 0):
        """
        Args:
            env: SelfPlayEnv instance (will be unwrapped from ActionMasker)
            update_freq: Steps between frozen model updates
        """
        super().__init__(verbose)
        self.env = env
        self.update_freq = update_freq
        self.last_update = 0
    
    def _on_step(self) -> bool:
        """Update frozen opponent if enough steps have passed."""
        if self.num_timesteps - self.last_update >= self.update_freq:
            # Save current model weights and load into a new model
            import tempfile
            import os
            
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "frozen_model")
                self.model.save(path)
                frozen_model = MaskablePPO.load(path)
            
            # Get the underlying SelfPlayEnv from ActionMasker wrapper
            if hasattr(self.env, 'env'):
                self.env.env.set_opponent_model(frozen_model)
            else:
                self.env.set_opponent_model(frozen_model)
            
            self.last_update = self.num_timesteps
            if self.verbose > 0:
                print(f"Updated frozen opponent at step {self.num_timesteps}")
        
        return True

# PART 3: Action Masker Function

def mask_fn(env):
    """
    Function called by MaskablePPO to get the action mask.
    
    This function is required by sb3_contrib.common.wrappers.ActionMasker.
    
    Args:
        env: The environment (SelfPlayEnv)
        
    Returns:
        numpy array of bools indicating valid actions
    """
    return env.action_masks()


# PART 4: Main Training Function

def train(
    simple_deck_path: str = "simple_deck.json",
    meta_deck_path: str = "meta_deck.json",
    total_timesteps: int = 1_000_000,
    save_path: str = "models/rl_bot",
    checkpoint_freq: int = 50_000,
    device: str = "auto"  # "cpu", "cuda", or "auto"
):
    """
    Train the agent with MaskablePPO
    
    Args:
        simple_deck_path: Path to simple_deck.json
        meta_deck_path: Path to meta_deck.json  
        total_timesteps: Total number of steps to train
        save_path: Path to save the final model
        checkpoint_freq: Frequency of checkpoints (in steps)
        device: "cpu", "cuda", or "auto"
    """
    print("=" * 60)
    print("Pokemon TCG Pocket RL Training")
    print("=" * 60)
    
    # Load the deck loaders
    print("\nLoading decks...")
    deck_loader = CurriculumDeckLoader(simple_deck_path, meta_deck_path)
    
    # Create the self-play environment
    print("\nCreating environment...")
    env = SelfPlayEnv(deck_loader)
    
    # Wrap the environment with ActionMasker
    # ActionMasker tells MaskablePPO how to get the action mask
    # from sb3_contrib.common.wrappers import ActionMasker
    env = ActionMasker(env, mask_fn)
    
    # Create the MaskablePPO model
    # Architecture sized for ~500+ dim observation space
    print("\nCreating model...")
    policy_kwargs = dict(
        net_arch=dict(
            pi=[512, 512, 256],  # Policy network (3 layers)
            vf=[512, 512, 256],  # Value network (separate, 3 layers)
        ),
        activation_fn=torch.nn.ReLU,
    )
    
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=4096,          # Experience per update
        batch_size=512,        # Larger batch for stability
        n_epochs=10,           # More optimization per batch
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,         # Encourage exploration
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        tensorboard_log="./logs/"
    )
    
    # Create callbacks
    curriculum_callback = CurriculumCallback(
        deck_loader,
        total_timesteps
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path="./checkpoints/",
        name_prefix="rl_bot"
    )
    frozen_opponent_callback = FrozenOpponentCallback(
        env,
        update_freq=50_000,
        verbose=1
    )
    
    # Initialize frozen opponent with initial model
    import tempfile
    import os as os_module
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os_module.path.join(tmpdir, "initial_frozen")
        model.save(path)
        initial_frozen = MaskablePPO.load(path)
    env.env.set_opponent_model(initial_frozen)
    
    # Train the model!
    print("\nStarting training...")
    print(f"   Total timesteps: {total_timesteps:,}")
    print(f"   Device: {device}")
    print(f"   Checkpoints every {checkpoint_freq:,} steps")
    print(f"   Frozen opponent updates every 50,000 steps")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[curriculum_callback, checkpoint_callback, frozen_opponent_callback],
        progress_bar=True
    )
    
    # Save the final model
    print(f"\nSaving model to {save_path}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    
    print("\nTraining complete!")
    

#  PART 5: Entry Point

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Pokemon TCG Pocket RL bot")
    parser.add_argument("--simple", default="simple_deck.json", help="Path to simple decks")
    parser.add_argument("--meta", default="meta_deck.json", help="Path to meta decks")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total timesteps")
    parser.add_argument("--save", default="models/rl_bot", help="Save path")
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    
    args = parser.parse_args()
    
    train(
        simple_deck_path=args.simple,
        meta_deck_path=args.meta,
        total_timesteps=args.steps,
        save_path=args.save,
        device=args.device
    )
