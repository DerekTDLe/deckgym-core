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
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

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
    resume_path: str = None  # Path to checkpoint to resume from (None = start fresh)
    
    # Training duration
    total_timesteps: int = 30_000_000
    checkpoint_freq: int = 100_000
    
    # PPO hyperparameters
    base_learning_rate: float = 1e-4      # Base LR for cyclical schedule
    min_learning_rate: float = 3e-5       # Floor LR
    n_steps: int = 8192                   # More experience per update (was 4096)
    batch_size: int = 512                 # Minibatch size for optimization
    n_epochs: int = 8                     # Reduced to compensate for larger n_steps
    gamma: float = 0.98                   # Reduced for shorter episodes (was 0.99)
    gae_lambda: float = 0.95              # GAE lambda
    ent_coef: float = 0.01                # Entropy bonus (exploration)
    vf_coef: float = 0.5                  # Value function coefficient (default)
    clip_range: float = 0.2               # PPO clipping range
    target_kl: float = 0.015              # Stop epoch early if KL > target (stabilizes training)
    
    # Network architecture
    policy_layers: tuple = (512, 512, 256, 128)   # Deeper policy network
    value_layers: tuple = (512, 512, 256)         # Match policy depth for better value estimation
    use_silu: bool = True                         # Use SiLU activation (better than ReLU)
    
    # Self-play
    frozen_opponent_update_freq: int = 75_000     # Slower updates for stability (was 50k)
    adaptive_frozen_opponent: bool = True         # Enable adaptive opponent updates
    target_win_rate: float = 0.55                 # Target win rate for adaptive updates
    
    # Curriculum learning
    curriculum_warmup_ratio: float = 0.10         # Phase 1 = 10% of total steps (simple decks)
    
    # Game limits
    max_turns: int = 99              # Official Pokemon TCG Pocket limit
    max_actions_per_turn: int = 50   # Prevents single-turn infinite loops
    max_total_actions: int = 500     # Prevents runaway episodes
    
    # Parallel environments (DummyVecEnv - no IPC overhead)
    n_envs: int = 8                  # 8 parallel environments for diversity
    
    # Device
    device: str = "auto"
    
    @property
    def curriculum_warmup_steps(self) -> int:
        """Compute warmup steps from ratio."""
        return int(self.total_timesteps * self.curriculum_warmup_ratio)
    
    def get_learning_rate_schedule(self):
        """
        Create cyclical learning rate schedule for self-play.
        
        LR boosts slightly after each frozen opponent update to help 
        the agent adapt to the new opponent, then decays during the cycle.
        Global decay also applied over training progress.
        """
        base_lr = self.base_learning_rate
        min_lr = self.min_learning_rate
        frozen_freq = self.frozen_opponent_update_freq
        total_steps = self.total_timesteps
        
        def lr_schedule(progress_remaining: float) -> float:
            # progress_remaining goes from 1.0 to 0.0
            progress = 1.0 - progress_remaining
            current_step = progress * total_steps
            
            # Global decay: 100% -> 50% over training
            global_decay = 0.5 + 0.5 * progress_remaining
            
            # Cycle progress within frozen opponent window
            cycle_progress = (current_step % frozen_freq) / frozen_freq
            
            # Boost at start of cycle (after frozen update), decay during cycle
            # Boost: +30% at cycle start, decays to 0% at cycle end
            cycle_boost = 1.0 + 0.3 * (1.0 - cycle_progress)
            
            lr = base_lr * global_decay * cycle_boost
            return max(min_lr, lr)
        
        return lr_schedule


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
        except BaseException as e:
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
        try:
            return self._env.action_masks()
        except BaseException as e:
            # Panic recovery - reset and return EndTurn-only
            print(f"WARNING: Panic in SelfPlayEnv.action_masks: {e}")
            self._env.reset()
            mask = np.zeros(self._env.action_space.n, dtype=np.bool_)
            mask[0] = True
            return mask
    
    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------
    
    def _play_opponent_turns(self, obs, info):
        """Play all opponent turns until agent's turn or game over."""
        final_reward = 0.0
        done = False
        turn_actions = 0
        
        try:
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
                obs, reward, done, truncated, info = self._env.step(action)
                final_reward = -reward  # Invert reward (opponent's loss = agent's gain)
                if done or truncated:
                    break
        except BaseException as e:
            print(f"WARNING: Game error during opponent turn: {e}")
            return obs, info, 0.0, True
        
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
        """End episode due to game error - reset internal game completely."""
        # Reset the internal game to get a fresh state
        self._env.reset()
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
    
    If adaptive=True, adjusts update frequency based on win rate:
    - Win rate too high → update more often (harder opponent)
    - Win rate too low → update less often (easier opponent)
    
    Supports both single env and DummyVecEnv.
    """
    
    def __init__(
        self,
        env,
        n_envs: int = 1,
        update_freq: int = 50_000,
        adaptive: bool = True,
        target_win_rate: float = 0.55,
        min_freq: int = 25_000,
        max_freq: int = 100_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.env = env
        self.n_envs = n_envs
        self.base_freq = update_freq
        self.current_freq = update_freq
        self.adaptive = adaptive
        self.target_win_rate = target_win_rate
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.last_update = 0
        self.win_buffer = []  # Track recent episode outcomes
    
    def _on_step(self) -> bool:
        # Track wins/losses from episode info
        if self.adaptive:
            infos = self.locals.get('infos', [])
            for info in infos:
                if 'episode' in info:
                    # Episode ended: reward > 0 means win
                    ep_reward = info['episode'].get('r', 0)
                    self.win_buffer.append(1 if ep_reward > 0 else 0)
                    if len(self.win_buffer) > 100:
                        self.win_buffer.pop(0)
            
            # Adjust frequency based on win rate
            if len(self.win_buffer) >= 20:
                win_rate = sum(self.win_buffer) / len(self.win_buffer)
                if win_rate > self.target_win_rate + 0.10:
                    # Too easy → update more often
                    self.current_freq = self.min_freq
                elif win_rate < self.target_win_rate - 0.10:
                    # Too hard → update less often
                    self.current_freq = self.max_freq
                else:
                    # In target range → use base frequency
                    self.current_freq = self.base_freq
        
        if self.num_timesteps - self.last_update >= self.current_freq:
            self._update_frozen_opponent()
            self.last_update = self.num_timesteps
            
            if self.verbose > 0:
                wr = sum(self.win_buffer) / len(self.win_buffer) if self.win_buffer else 0.5
                print(f"[Step {self.num_timesteps:,}] Updated frozen opponent (WR: {wr:.1%}, freq: {self.current_freq:,})")
        
        return True
    
    def _update_frozen_opponent(self):
        """Save current model and load as frozen opponent on CPU."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "frozen_model")
            self.model.save(path)
            # Load on CPU to avoid GPU contention with policy training
            frozen_model = MaskablePPO.load(path, device="cpu")
        
        if self.n_envs == 1:
            # Single env: unwrap Monitor -> ActionMasker to get SelfPlayEnv
            inner_env = self.env.env.env if hasattr(self.env.env, 'env') else self.env.env
            inner_env.set_opponent_model(frozen_model)
        else:
            # DummyVecEnv: update all environments (Monitor -> ActionMasker -> SelfPlayEnv)
            for i in range(self.n_envs):
                self.env.envs[i].env.env.set_opponent_model(frozen_model)


# =============================================================================
# Action Masking & Environment Factory
# =============================================================================

def mask_fn(env: SelfPlayEnv) -> np.ndarray:
    """Provide action mask to MaskablePPO."""
    return env.action_masks()


def make_env(
    deck_loader: CurriculumDeckLoader,
    config: TrainingConfig,
) -> callable:
    """
    Factory function to create self-play environments.
    
    For DummyVecEnv, all envs share the same deck_loader (efficient).
    Monitor wrapper enables episode stats for TensorBoard.
    """
    def _init() -> gym.Env:
        env = SelfPlayEnv(deck_loader, config=config)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)  # Track ep_len, ep_rew for TensorBoard
        return env
    return _init


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
    print(f"  Learning rate:      {config.base_learning_rate} (cyclical, min={config.min_learning_rate})")
    print(f"  Target KL:          {config.target_kl}")
    print(f"  Batch size:         {config.batch_size}")
    print(f"  Entropy coef:       {config.ent_coef}")
    print(f"  Policy network:     {config.policy_layers}")
    print(f"  N environments:     {config.n_envs}")
    print(f"  Device:             {config.device}")
    if config.resume_path:
        print(f"  Resume from:        {config.resume_path}")
    
    # Setup environment(s)
    print("\n[1/4] Loading decks...")
    deck_loader = CurriculumDeckLoader(config.simple_deck_path, config.meta_deck_path)
    
    print(f"[2/4] Creating {config.n_envs} environment(s)...")
    if config.n_envs == 1:
        # Single environment (simpler)
        single_env = SelfPlayEnv(deck_loader, config=config)
        env = ActionMasker(single_env, mask_fn)
        env = Monitor(env)  # Track ep_len, ep_rew for TensorBoard
    else:
        # Multiple environments with DummyVecEnv (no IPC overhead)
        env = DummyVecEnv([make_env(deck_loader, config) for _ in range(config.n_envs)])
        single_env = None  # Will be set after model init
    
    # Setup model
    print("[3/4] Initializing model...")
    activation_fn = torch.nn.SiLU if config.use_silu else torch.nn.ReLU
    policy_kwargs = dict(
        net_arch=dict(
            pi=list(config.policy_layers),
            vf=list(config.value_layers),
        ),
        activation_fn=activation_fn,
    )
    
    if config.resume_path:
        # Resume from existing checkpoint
        print(f"      Resuming from: {config.resume_path}")
        model = MaskablePPO.load(
            config.resume_path,
            env=env,
            device=config.device,
            tensorboard_log=config.tensorboard_dir,
        )
        # Update hyperparameters for fine-tuning
        model.learning_rate = config.get_learning_rate_schedule()
        model.n_steps = config.n_steps
        model.batch_size = config.batch_size
        model.n_epochs = config.n_epochs
        model.gamma = config.gamma
        model.gae_lambda = config.gae_lambda
        model.ent_coef = config.ent_coef
        model.vf_coef = config.vf_coef
        model.clip_range = config.clip_range
        model.target_kl = config.target_kl
        print(f"      Updated hyperparameters (LR={config.base_learning_rate}, target_kl={config.target_kl})")
    else:
        # Create new model from scratch
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=config.get_learning_rate_schedule(),  # Cyclical LR
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            clip_range=config.clip_range,
            target_kl=config.target_kl,  # Stop epoch early if KL too high
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
            n_envs=config.n_envs,
            update_freq=config.frozen_opponent_update_freq,
            adaptive=config.adaptive_frozen_opponent,
            target_win_rate=config.target_win_rate,
            verbose=1,
        ),
    ]
    
    # Initialize frozen opponent on CPU (to avoid GPU contention)
    print("      Loading frozen opponent on CPU...")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "initial_frozen")
        model.save(path)
        initial_frozen = MaskablePPO.load(path, device="cpu")
    
    if config.n_envs == 1:
        # Single env: unwrap Monitor -> ActionMasker to get SelfPlayEnv 
        env.env.env.set_opponent_model(initial_frozen)
    else:
        # Set opponent on all envs in DummyVecEnv (Monitor -> ActionMasker -> SelfPlayEnv)
        for i in range(config.n_envs):
            env.envs[i].env.env.set_opponent_model(initial_frozen)
    
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
    parser.add_argument("--steps", type=int, default=30_000_000, help="Total training steps")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000, help="Checkpoint frequency")
    
    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate (cyclical)")
    parser.add_argument("--min-lr", type=float, default=3e-5, help="Minimum learning rate floor")
    parser.add_argument("--target-kl", type=float, default=0.015, help="Target KL for early stopping")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--n-epochs", type=int, default=8, help="PPO epochs per update")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    
    # Self-play
    parser.add_argument("--opponent-update-freq", type=int, default=75_000, help="Frozen opponent update frequency")
    parser.add_argument("--warmup-ratio", type=float, default=0.10, help="Curriculum warmup ratio (0-1)")
    
    # Parallelization
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments (DummyVecEnv)")
    
    # Resume
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume training from")
    
    # Device
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    
    args = parser.parse_args()
    
    # Build config from args
    config = TrainingConfig(
        simple_deck_path=args.simple,
        meta_deck_path=args.meta,
        save_path=args.save,
        resume_path=args.resume,
        total_timesteps=args.steps,
        checkpoint_freq=args.checkpoint_freq,
        base_learning_rate=args.lr,
        min_learning_rate=args.min_lr,
        target_kl=args.target_kl,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        ent_coef=args.ent_coef,
        frozen_opponent_update_freq=args.opponent_update_freq,
        curriculum_warmup_ratio=args.warmup_ratio,
        n_envs=args.n_envs,
        device=args.device,
    )
    
    train(config)

