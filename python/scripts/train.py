#!/usr/bin/env python3
"""
Train a Pokemon TCG Pocket RL agent using MaskablePPO with self-play.

The agent learns by playing against frozen copies of itself, with curriculum
learning that progressively increases deck difficulty from simple to meta.

Starting player alternates randomly each game (controlled by game seed).
"""

import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
import gymnasium as gym
import torch

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from deckgym.env import DeckGymEnv
from deckgym.deck_loader import CurriculumDeckLoader


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """All training hyperparameters in one place."""
    
    # Paths
    simple_deck_path: str = "simple_deck.json"
    meta_deck_path: str = "meta_deck.json"
    save_path: str = "models/rl_bot"
    checkpoint_dir: str = "./checkpoints/"
    tensorboard_dir: str = "./logs/"
    
    # Training duration
    total_timesteps: int = 1_000_000
    checkpoint_freq: int = 50_000
    
    # PPO hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 4096              # Experience collected per update
    batch_size: int = 512            # Minibatch size for optimization
    n_epochs: int = 10               # Optimization epochs per update
    gamma: float = 0.99              # Discount factor
    gae_lambda: float = 0.95         # GAE lambda
    ent_coef: float = 0.01           # Entropy bonus (exploration)
    clip_range: float = 0.2          # PPO clipping range
    
    # Network architecture
    policy_layers: tuple = (512, 512, 256)
    value_layers: tuple = (512, 512, 256)
    
    # Self-play
    frozen_opponent_update_freq: int = 50_000
    curriculum_warmup_steps: int = 100_000
    
    # Game limits
    max_turns: int = 99              # Official Pokemon TCG Pocket limit
    max_actions_per_turn: int = 50   # Prevents single-turn infinite loops
    max_total_actions: int = 500     # Prevents runaway episodes
    
    # Device
    device: str = "auto"


# Default configuration
DEFAULT_CONFIG = TrainingConfig()


# =============================================================================
# Self-Play Environment
# =============================================================================

class SelfPlayEnv(gym.Env):
    """
    Gymnasium wrapper for self-play training.
    
    Player 0 (agent) trains against Player 1 (frozen opponent).
    The frozen opponent is periodically updated with the agent's weights
    to prevent self-play collapse.
    
    Note: Starting player is randomized by the game seed, so the agent
    naturally learns to play from both first and second positions.
    """
    
    def __init__(
        self,
        deck_loader: CurriculumDeckLoader,
        opponent_model=None,
        config: TrainingConfig = DEFAULT_CONFIG,
    ):
        """
        Initialize the self-play environment.
        
        Args:
            deck_loader: Samples deck pairs with curriculum difficulty
            opponent_model: Frozen model for opponent (updated periodically)
            config: Training configuration with game limits
        """
        self.deck_loader = deck_loader
        self.opponent_model = opponent_model
        self.config = config
        self._episode_actions = 0
        
        # Initialize with a dummy game to get space dimensions
        self._env = DeckGymEnv(*deck_loader.sample_pair())
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
    
    def set_opponent_model(self, model):
        """Update the frozen opponent model."""
        self.opponent_model = model
    
    def reset(self, seed=None, options=None):
        """
        Reset environment with new random decks.
        
        The game seed determines starting player (random ~50/50).
        """
        deck_a, deck_b = self.deck_loader.sample_pair()
        self._env = DeckGymEnv(deck_a, deck_b, seed=seed)
        obs, info = self._env.reset(seed=seed)
        
        # Play opponent's turns if they go first
        obs, info, _, _ = self._play_opponent_turns(obs, info)
        self._episode_actions = 0
        
        return obs, info
    
    def step(self, action):
        """
        Execute agent's action, then play opponent's turn(s).
        
        Returns:
            obs: Game state from agent's perspective
            reward: +1 win, -1 loss, 0 otherwise
            terminated: True if game ended normally
            truncated: True if game ended due to limits
            info: Additional information
        """
        # Check turn limit
        turn_count = self._get_turn_count()
        if turn_count > self.config.max_turns:
            return self._end_episode_truncated("max_turns_exceeded")
        
        # Check action limit
        self._episode_actions += 1
        if self._episode_actions > self.config.max_total_actions:
            return self._end_episode_truncated("max_actions_exceeded")
        
        # Execute agent's action
        try:
            obs, reward, done, truncated, info = self._env.step(action)
        except Exception as e:
            print(f"WARNING: Game error during agent turn: {e}")
            return self._end_episode_error(str(e))
        
        # Play opponent's response
        if not done and not truncated:
            obs, info, opp_reward, opp_done = self._play_opponent_turns(obs, info)
            if opp_done:
                done = True
                reward = opp_reward
        
        return obs, reward, done, truncated, info
    
    def action_masks(self) -> np.ndarray:
        """Return valid action mask for the agent."""
        return self._env.action_masks()
    
    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------
    
    def _play_opponent_turns(self, obs, info):
        """Play all opponent turns until agent's turn or game over."""
        final_reward = 0.0
        done = False
        turn_actions = 0
        
        while self._env.game.current_player() == 1 and not self._env.game.is_game_over():
            turn_actions += 1
            self._episode_actions += 1
            
            # Safety limits
            if turn_actions > self.config.max_actions_per_turn:
                self._log_freeze_warning("opponent_turn_limit", turn_actions)
                return obs, info, 0.0, True
            
            if self._episode_actions > self.config.max_total_actions:
                self._log_freeze_warning("episode_action_limit", self._episode_actions)
                return obs, info, 0.0, True
            
            # Select opponent action
            action = self._select_opponent_action(obs)
            
            # Execute action
            try:
                obs, reward, done, truncated, info = self._env.step(action)
            except Exception as e:
                print(f"WARNING: Game error during opponent turn: {e}")
                return obs, info, 0.0, True
            
            final_reward = -reward  # Invert reward (opponent's loss = agent's gain)
            if done or truncated:
                break
        
        return obs, info, final_reward, done
    
    def _select_opponent_action(self, obs) -> int:
        """Select action for the opponent using frozen model or random."""
        action_mask = self._env.action_masks()
        
        if self.opponent_model is not None:
            action, _ = self.opponent_model.predict(obs, action_masks=action_mask, deterministic=False)
        else:
            valid_actions = np.where(action_mask)[0]
            action = np.random.choice(valid_actions)
        
        return action
    
    def _get_turn_count(self) -> int:
        """Get current turn count from game state."""
        return self._env.game.get_state().turn_count
    
    def _end_episode_truncated(self, reason: str):
        """End episode due to limit exceeded."""
        obs = np.array(self._env.game.get_obs(), dtype=np.float32)
        return obs, 0.0, False, True, {"truncated_reason": reason}
    
    def _end_episode_error(self, error: str):
        """End episode due to game error."""
        obs = np.array(self._env.game.get_obs(), dtype=np.float32)
        return obs, 0.0, True, False, {"error": error}
    
    def _log_freeze_warning(self, reason: str, count: int):
        """Log warning when safety limit is hit."""
        turn = self._get_turn_count()
        player = self._env.game.current_player()
        print(f"WARNING: {reason} triggered (count={count}, turn={turn}, player={player})")


# =============================================================================
# Training Callbacks
# =============================================================================

class CurriculumCallback(BaseCallback):
    """
    Progressively increases deck difficulty during training.
    
    Difficulty ramps from 0 (simple decks only) to 1 (meta decks only)
    after a warmup period.
    """
    
    def __init__(
        self,
        deck_loader: CurriculumDeckLoader,
        total_timesteps: int,
        warmup_steps: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.deck_loader = deck_loader
        self.total_timesteps = total_timesteps
        self.warmup_steps = warmup_steps
    
    def _on_step(self) -> bool:
        if self.warmup_steps >= self.total_timesteps:
            difficulty = 1.0
        else:
            progress = (self.num_timesteps - self.warmup_steps) / (self.total_timesteps - self.warmup_steps)
            difficulty = max(0.0, min(1.0, progress))
        
        self.deck_loader.set_difficulty(difficulty)
        return True


class FrozenOpponentCallback(BaseCallback):
    """
    Periodically updates the frozen opponent with current agent weights.
    
    This creates a slowly-moving target that prevents self-play collapse
    while still providing meaningful training signal.
    """
    
    def __init__(self, env, update_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.env = env
        self.update_freq = update_freq
        self.last_update = 0
    
    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_update >= self.update_freq:
            self._update_frozen_opponent()
            self.last_update = self.num_timesteps
            
            if self.verbose > 0:
                print(f"[Step {self.num_timesteps:,}] Updated frozen opponent")
        
        return True
    
    def _update_frozen_opponent(self):
        """Save current model and load as frozen opponent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "frozen_model")
            self.model.save(path)
            frozen_model = MaskablePPO.load(path)
        
        # Unwrap ActionMasker to get SelfPlayEnv
        inner_env = self.env.env if hasattr(self.env, 'env') else self.env
        inner_env.set_opponent_model(frozen_model)


# =============================================================================
# Action Masking
# =============================================================================

def mask_fn(env: SelfPlayEnv) -> np.ndarray:
    """Provide action mask to MaskablePPO."""
    return env.action_masks()


# =============================================================================
# Training Loop
# =============================================================================

def train(config: TrainingConfig = DEFAULT_CONFIG):
    """
    Train the RL agent with MaskablePPO.
    
    Args:
        config: Training configuration object
    """
    print("=" * 60)
    print("Pokemon TCG Pocket RL Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Total timesteps:    {config.total_timesteps:,}")
    print(f"  Learning rate:      {config.learning_rate}")
    print(f"  Batch size:         {config.batch_size}")
    print(f"  Entropy coef:       {config.ent_coef}")
    print(f"  Policy network:     {config.policy_layers}")
    print(f"  Device:             {config.device}")
    
    # Setup environment
    print("\n[1/4] Loading decks...")
    deck_loader = CurriculumDeckLoader(config.simple_deck_path, config.meta_deck_path)
    
    print("[2/4] Creating environment...")
    env = SelfPlayEnv(deck_loader, config=config)
    env = ActionMasker(env, mask_fn)
    
    # Setup model
    print("[3/4] Initializing model...")
    policy_kwargs = dict(
        net_arch=dict(
            pi=list(config.policy_layers),
            vf=list(config.value_layers),
        ),
        activation_fn=torch.nn.ReLU,
    )
    
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        ent_coef=config.ent_coef,
        clip_range=config.clip_range,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=config.device,
        tensorboard_log=config.tensorboard_dir,
    )
    
    # Setup callbacks
    callbacks = [
        CurriculumCallback(
            deck_loader,
            config.total_timesteps,
            config.curriculum_warmup_steps,
        ),
        CheckpointCallback(
            save_freq=config.checkpoint_freq,
            save_path=config.checkpoint_dir,
            name_prefix="rl_bot",
        ),
        FrozenOpponentCallback(
            env,
            update_freq=config.frozen_opponent_update_freq,
            verbose=1,
        ),
    ]
    
    # Initialize frozen opponent
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "initial_frozen")
        model.save(path)
        initial_frozen = MaskablePPO.load(path)
    env.env.set_opponent_model(initial_frozen)
    
    # Train
    print("[4/4] Starting training...")
    print(f"      Checkpoints: every {config.checkpoint_freq:,} steps")
    print(f"      Opponent updates: every {config.frozen_opponent_update_freq:,} steps")
    
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    # Save final model
    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)
    model.save(config.save_path)
    print(f"\nModel saved to {config.save_path}")
    print("Training complete!")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train Pokemon TCG Pocket RL bot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Paths
    parser.add_argument("--simple", default="simple_deck.json", help="Simple deck definitions")
    parser.add_argument("--meta", default="meta_deck.json", help="Meta deck definitions")
    parser.add_argument("--save", default="models/rl_bot", help="Model save path")
    
    # Training
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total training steps")
    parser.add_argument("--checkpoint-freq", type=int, default=50_000, help="Checkpoint frequency")
    
    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO epochs per update")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    
    # Self-play
    parser.add_argument("--opponent-update-freq", type=int, default=50_000, help="Frozen opponent update frequency")
    parser.add_argument("--warmup-steps", type=int, default=100_000, help="Curriculum warmup steps")
    
    # Device
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    
    args = parser.parse_args()
    
    # Build config from args
    config = TrainingConfig(
        simple_deck_path=args.simple,
        meta_deck_path=args.meta,
        save_path=args.save,
        total_timesteps=args.steps,
        checkpoint_freq=args.checkpoint_freq,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        ent_coef=args.ent_coef,
        frozen_opponent_update_freq=args.opponent_update_freq,
        curriculum_warmup_steps=args.warmup_steps,
        device=args.device,
    )
    
    train(config)
