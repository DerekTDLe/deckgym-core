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

# PART 1: Self-Play Environment Wrapper

class SelfPlayEnv(gym.Env):
    """
    Wrapper that let a player play against itself
    
    At each reset(), we:
    1. Sample two decks (one for each player)
    2. Create a new game
    3. The agent plays for both players (self-play)
    
    The goal: the agent learns the good strategies by playing against
    a copy of itself that also improves.
    """
    
    def __init__(self, deck_loader):
        """
        Args:
            deck_loader: CurriculumDeckLoader to sample the
        """
        self.deck_loader = deck_loader
        
        # Create a dummy game to get observation/action space sizes
        self._env = DeckGymEnv(*deck_loader.sample_pair()) # Besoin d'une seed?
        
        # Copy observation_space and action_space from the inner env
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
    
    def reset(self, seed=None, options=None):
        """
        Reset for a new game with random decks.
        
        Returns:
            observation: Initial state of the game (numpy array of 163 floats)
            info: Empty dict (required by Gymnasium)
        """
        # Sample two decks from the curriculum loader
        deck_a, deck_b = self.deck_loader.sample_pair()
        # Create a new DeckGymEnv with these decks
        self._env = DeckGymEnv(deck_a, deck_b, seed=seed)      
        # Return the initial observation
        return self._env.reset(seed=seed)
    
    def step(self, action):
        """
        Execute a action in the game.
        
        Args:
            action: Index of the action (0-49)
            
        Returns:
            obs: New state of the game
            reward: +1 victory, -1 defeat, 0 otherwise
            terminated: True if the game is finished
            truncated: False (no time limit)
            info: Dict with additional information
        """
        # Forward the action to the inner environment
        return self._env.step(action)
    
    def action_masks(self):
        """
        Return the action mask from inner env.
        
        Important for MaskablePPO: we can only choose actions
        where the mask is True.
        
        Returns:
            numpy array of bools, shape (50,)
        """
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
    
    # Train the model!
    print("\nStarting training...")
    print(f"   Total timesteps: {total_timesteps:,}")
    print(f"   Device: {device}")
    print(f"   Checkpoints every {checkpoint_freq:,} steps")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[curriculum_callback, checkpoint_callback],
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
