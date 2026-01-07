#!/usr/bin/env python3
"""
Train a Pokemon TCG Pocket RL agent using MaskablePPO with self-play.

The agent learns by playing against frozen copies of itself, with curriculum
learning that progressively increases deck difficulty from simple to meta.

Starting player alternates randomly each game (controlled by game seed).
"""

# Suppress warnings BEFORE importing anything else
import warnings
warnings.filterwarnings("ignore", message=".*nested tensors.*")
warnings.filterwarnings("ignore", message=".*np.object.*", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import gymnasium as gym
import torch
import yaml

# Enable TensorFloat32 for faster matmul on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from deckgym.env import DeckGymEnv
from deckgym.batched_env import BatchedDeckGymEnv
from deckgym.deck_loader import CurriculumDeckLoader, MetaDeckLoader
from deckgym.attention_policy import CardAttentionExtractor, create_attention_policy_kwargs
from deckgym.curriculum import CurriculumManager

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
    checkpoint_freq: int = 100_000        # Per env (actual = this × n_envs)
    
    # PPO hyperparameters
    base_learning_rate: float = 1e-4      # Higher LR for attention (was 5e-5)
    min_learning_rate: float = 1e-5       # Lower floor LR
    warmup_steps: int = 300              # LR warmup steps (critical for attention stability)
    n_steps: int = 8192                   # More experience per update
    batch_size: int = 2048                # Larger batch for better GPU utilization
    n_epochs: int = 8                     # Reduced to compensate for larger n_steps
    gamma: float = 0.98                   # Reduced for shorter episodes
    gae_lambda: float = 0.95              # GAE lambda
    ent_coef: float = 0.02                # Higher entropy for larger model
    vf_coef: float = 0.5                  # Value function coefficient
    clip_range: float = 0.2               # PPO clipping range
    max_grad_norm: float = 1.0            # Gradient clipping (was 0.5, higher for attention)
    target_kl: float = 0.015              # Stop epoch early if KL > target
    
    # Network architecture
    policy_layers: tuple = (512, 512, 256, 128)   # Deeper policy network (MLP only)
    value_layers: tuple = (512, 512, 256)         # Match policy depth for better value estimation (MLP only)
    use_attention: bool = True                    # Default to attention-based policy
    attention_embed_dim: int = 256                # Larger embedding for better card representations
    attention_num_heads: int = 4                  # More heads for diverse attention patterns
    attention_num_layers: int = 2                 # Deeper transformer for complex reasoning
    use_silu: bool = True                         # Use SiLU activation (better than ReLU)
    
    # Self-play
    frozen_opponent_update_freq: int = 75_000     # Slower updates for stability (was 50k)
    adaptive_frozen_opponent: bool = True         # Enable adaptive opponent updates
    target_win_rate: float = 0.55                 # Target win rate for adaptive updates
    
    # Game limits
    max_turns: int = 99              # Official Pokemon TCG Pocket limit
    max_actions_per_turn: int = 50   # Prevents single-turn infinite loops
    max_total_actions: int = 500     # Prevents runaway episodes
    
    # Parallel environments
    n_envs: int = 32                  # 32 parallel environments for faster training
    use_batched_env: bool = True     # Use Rust-side batched env (faster, ~7000 steps/sec)
    
    # Device
    device: str = "auto"
    
    def get_learning_rate_schedule(self, total_timesteps: int = None):
        """
        Learning rate schedule with warmup + cosine annealing.
        
        Standard for modern transformers (GPT, BERT, etc.):
        - Linear warmup: 0 → base_lr
        - Cosine decay: base_lr → min_lr (smooth curve)
        """
        import math
        
        base_lr = self.base_learning_rate
        min_lr = self.min_learning_rate
        warmup_steps = self.warmup_steps
        
        if warmup_steps > 0 and total_timesteps:
            # Calculate total updates
            total_updates = total_timesteps // self.n_steps
            
            def lr_schedule_cosine(progress_remaining: float) -> float:
                # progress_remaining goes from 1.0 → 0.0
                current_update = int((1.0 - progress_remaining) * total_updates)
                
                if current_update < warmup_steps:
                    # Warmup phase: linear 0 → base_lr
                    return (current_update / warmup_steps) * base_lr
                else:
                    # Cosine annealing: base_lr → min_lr
                    progress = (current_update - warmup_steps) / (total_updates - warmup_steps)
                    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                    return min_lr + (base_lr - min_lr) * cosine_decay
            
            return lr_schedule_cosine
        else:
            # No warmup: simple cosine decay
            import math
            
            def lr_schedule_simple(progress_remaining: float) -> float:
                # Cosine annealing from base_lr to min_lr
                progress = 1.0 - progress_remaining
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return min_lr + (base_lr - min_lr) * cosine_decay
            
            return lr_schedule_simple


# Default configuration
DEFAULT_CONFIG = TrainingConfig()


def load_config_from_yaml(yaml_path: str) -> TrainingConfig:
    """
    Load training configuration from YAML file.
    
    Args:
        yaml_path: Path to YAML config file
    
    Returns:
        TrainingConfig instance with values from YAML
    """
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Flatten nested structure
    flat_config = {}
    
    # Model params
    if 'model' in config_dict:
        flat_config['attention_embed_dim'] = config_dict['model'].get('attention_embed_dim', 256)
        flat_config['attention_num_heads'] = config_dict['model'].get('attention_num_heads', 4)
        flat_config['attention_num_layers'] = config_dict['model'].get('attention_num_layers', 2)
        flat_config['use_attention'] = config_dict['model'].get('use_attention', True)
        flat_config['use_silu'] = config_dict['model'].get('use_silu', True)
        
        # Load net_arch for MLP models
        if 'net_arch' in config_dict['model']:
            net_arch = config_dict['model']['net_arch']
            flat_config['policy_layers'] = tuple(net_arch.get('pi', [512, 512, 256, 128]))
            flat_config['value_layers'] = tuple(net_arch.get('vf', [512, 512, 256]))
    
    # PPO params
    if 'ppo' in config_dict:
        flat_config['base_learning_rate'] = config_dict['ppo'].get('learning_rate', 5e-5)
        flat_config['min_learning_rate'] = config_dict['ppo'].get('min_learning_rate', 1e-5)
        flat_config['batch_size'] = config_dict['ppo'].get('batch_size', 2048)
        flat_config['n_steps'] = config_dict['ppo'].get('n_steps', 8192)
        flat_config['n_epochs'] = config_dict['ppo'].get('n_epochs', 8)
        flat_config['gamma'] = config_dict['ppo'].get('gamma', 0.98)
        flat_config['gae_lambda'] = config_dict['ppo'].get('gae_lambda', 0.95)
        flat_config['ent_coef'] = config_dict['ppo'].get('ent_coef', 0.02)
        flat_config['vf_coef'] = config_dict['ppo'].get('vf_coef', 0.5)
        flat_config['clip_range'] = config_dict['ppo'].get('clip_range', 0.2)
        flat_config['target_kl'] = config_dict['ppo'].get('target_kl', 0.025)
    
    # Training params
    if 'training' in config_dict:
        flat_config['total_timesteps'] = config_dict['training'].get('total_timesteps', 30_000_000)
        flat_config['checkpoint_freq'] = config_dict['training'].get('checkpoint_freq', 100_000)
        flat_config['n_envs'] = config_dict['training'].get('n_envs', 32)
        flat_config['use_batched_env'] = config_dict['training'].get('use_batched_env', True)
        flat_config['device'] = config_dict['training'].get('device', 'auto')
        flat_config['frozen_opponent_update_freq'] = config_dict['training'].get('frozen_opponent_update_freq', 75_000)
        flat_config['adaptive_frozen_opponent'] = config_dict['training'].get('adaptive_frozen_opponent', True)
        flat_config['target_win_rate'] = config_dict['training'].get('target_win_rate', 0.55)
    
    # Environment params
    if 'environment' in config_dict:
        flat_config['max_turns'] = config_dict['environment'].get('max_turns', 99)
        flat_config['max_actions_per_turn'] = config_dict['environment'].get('max_actions_per_turn', 50)
        flat_config['max_total_actions'] = config_dict['environment'].get('max_total_actions', 500)
    
    # Paths
    if 'paths' in config_dict:
        flat_config['simple_deck_path'] = config_dict['paths'].get('simple_deck_path', 'simple_deck.json')
        flat_config['meta_deck_path'] = config_dict['paths'].get('meta_deck_path', 'meta_deck.json')
        flat_config['save_path'] = config_dict['paths'].get('save_path', 'models/rl_bot')
        flat_config['checkpoint_dir'] = config_dict['paths'].get('checkpoint_dir', './checkpoints/')
        flat_config['tensorboard_dir'] = config_dict['paths'].get('tensorboard_dir', './logs/')
        flat_config['resume_path'] = config_dict['paths'].get('resume_path', None)
    
    return TrainingConfig(**flat_config)


# =============================================================================
# Self-Play Environment
# =============================================================================

class SelfPlayEnv(gym.Env):
    """
    Gymnasium wrapper for training against various opponents.
    
    Supports:
    - "self": Frozen copy of agent (self-play)
    - "e2", "e3", etc.: Rust expectiminimax bots
    - "random": Random baseline
    
    Player 0 (agent) trains against Player 1 (opponent).
    Starting player is randomized by game seed.
    """
    
    def __init__(
        self,
        deck_loader,
        opponent_model=None,
        opponent_type: str = "self",
        config: TrainingConfig = DEFAULT_CONFIG,
    ):
        """
        Initialize the training environment.
        
        Args:
            deck_loader: Samples deck pairs
            opponent_model: Frozen model for self-play (used when opponent_type="self")
            opponent_type: "self", "e2", "e3", "v", "random"
            config: Training configuration
        """
        self.deck_loader = deck_loader
        self.opponent_model = opponent_model
        self.opponent_type = opponent_type
        self.config = config
        self._episode_actions = 0
        self._current_decks = None  # (deck_a, deck_b) strings
        
        # Initialize with a dummy game to get space dimensions
        self._env = DeckGymEnv(*deck_loader.sample_pair())
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
    
    def set_opponent_model(self, model):
        """Update the frozen opponent model."""
        self.opponent_model = model
    
    def set_opponent_type(self, opponent_type: str):
        """Change opponent type (e.g., 'self', 'e2', 'e3')."""
        self.opponent_type = opponent_type
        if opponent_type != "self":
            self.opponent_model = None  # Clear frozen model when using bots
    
    def reset(self, seed=None, options=None):
        """
        Reset environment with new random decks.
        
        The game seed determines starting player (random ~50/50).
        """
        deck_a, deck_b = self.deck_loader.sample_pair()
        self._current_decks = (deck_a, deck_b)
        
        # Create the base environment
        self._env = DeckGymEnv(deck_a, deck_b, seed=seed)
        obs, info = self._env.reset(seed=seed)
        
        # For bot opponents, replace the game with one that has the bot
        if self.opponent_type != "self":
            self._replace_game_with_bot(deck_a, deck_b, seed)
            # Re-get observation from the new game
            obs = np.array(self._env.game.get_obs(), dtype=np.float32)
        
        # Play opponent's turns if they go first
        obs, info, _, _ = self._play_opponent_turns(obs, info)
        self._episode_actions = 0
        
        return obs, info
    
    def _replace_game_with_bot(self, deck_a: str, deck_b: str, seed=None):
        """Replace _env.game with a PyGame that has bot opponent."""
        import deckgym
        
        # Write decks to temp files (PyGame requires file paths for bot support)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(deck_a)
            path_a = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(deck_b)
            path_b = f.name
        
        try:
            # Create game with: player 0 = random (agent controls), player 1 = bot
            bot_game = deckgym.Game(path_a, path_b, players=['r', self.opponent_type], seed=seed)
            # Replace the environment's game
            self._env.game = bot_game
        finally:
            os.unlink(path_a)
            os.unlink(path_b)
    
    def step(self, action):
        """
        Execute agent's action, then play opponent's turn(s).
        
        Returns:
            obs: Game state from agent's perspective
            reward: +1 win, -1 loss, 0 otherwise, plus bonus or penalty based on game score
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
                
                if self.opponent_type == "self":
                    # Self-play: select action with frozen model, execute in _env
                    action = self._select_opponent_action(obs)
                    obs, reward, done, truncated, info = self._env.step(action)
                    final_reward = -reward  # Invert reward (opponent's loss = agent's gain)
                else:
                    # Bot opponent: use play_tick() on the underlying game
                    # play_tick() both selects AND executes the bot's action
                    self._env.game.play_tick()
                    # Get updated observation after bot's action
                    obs = np.array(self._env.game.get_obs(), dtype=np.float32)
                    done = self._env.game.is_game_over()
                    if done:
                        # Determine turn-based reward (matches vec_game.rs calculate_reward)
                        state = self._env.game.get_state()
                        if state.winner and not state.winner.is_tie:
                            winner = state.winner.winner
                            my_points = float(state.points[0])
                            opp_points = float(state.points[1])
                            point_diff = my_points - opp_points
                            turn_count = float(state.turn_count)
                            
                            # Base reward from point difference
                            base = 1.0 + (point_diff / 6.0)
                            
                            # Speed factor: 1 + (13 - turns) / 13
                            speed_factor = 1.0 + ((13.0 - turn_count) / 13.0)
                            
                            if winner == 0:
                                # Agent won: base * max(1.0, speed_factor)
                                final_reward = base * max(1.0, speed_factor)
                            else:
                                # Opponent won: -base * speed_factor
                                final_reward = -base * speed_factor
                        else:
                            final_reward = 0.0  # Tie
                
                if done:
                    break
        except BaseException as e:
            print(f"WARNING: Game error during opponent turn: {e}")
            return obs, info, 0.0, True
        
        return obs, info, final_reward, done
    
    def _select_opponent_action(self, obs) -> int:
        """Select action for the opponent (self-play only).
        
        Note: Bot opponents are handled directly via play_tick() in _play_opponent_turns(),
        so this method is only used for self-play with frozen model.
        """
        if self.opponent_type != "self":
            # Bot opponent: action is handled by play_tick() in _play_opponent_turns()
            return -1  # Sentinel: not used for bots
        
        # Self-play: use frozen model
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

# NOTE: CurriculumCallback (deck difficulty) was removed.
# Deck sources are now controlled by CurriculumManager stage transitions.


class EpisodeMetricsCallback(BaseCallback):
    """
    Logs custom episode metrics to TensorBoard.
    
    Tracks metrics from BatchedDeckGymEnv's episode info dict:
    - actions_per_turn: Average actions taken between EndTurn calls
    
    Also shows training phase indicators (Collecting vs Optimizing).
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._actions_per_turn_buffer = []
    
    def _on_rollout_start(self) -> None:
        """Called when rollout collection starts."""
        print("\r[Phase: Collecting rollouts...]", end="", flush=True)
    
    def _on_rollout_end(self) -> None:
        """Called when rollout collection ends, optimization starts."""
        print("\r[Phase: PPO Optimization...]   ", end="", flush=True)
        
        # Log buffered metrics
        if self._actions_per_turn_buffer and self.logger:
            mean_apt = np.mean(self._actions_per_turn_buffer)
            self.logger.record("rollout/actions_per_turn_mean", mean_apt)
            self._actions_per_turn_buffer.clear()
    
    def _on_step(self) -> bool:
        # Check for finished episodes in infos
        for info in self.locals.get("infos", []):
            if "episode" in info:
                episode_info = info["episode"]
                if "actions_per_turn" in episode_info:
                    self._actions_per_turn_buffer.append(episode_info["actions_per_turn"])
        
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
        import gc
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "frozen_model")
            self.model.save(path)
            # Load on CPU to avoid GPU contention with policy training
            new_frozen = MaskablePPO.load(path, device="cpu")
        
        # Clean up old frozen model to prevent memory leak
        if hasattr(self, '_frozen_model') and self._frozen_model is not None:
            del self._frozen_model
            gc.collect()
        
        self._frozen_model = new_frozen
        
        if self.n_envs == 1:
            # Single env: unwrap Monitor -> ActionMasker to get SelfPlayEnv
            inner_env = self.env.env.env if hasattr(self.env.env, 'env') else self.env.env
            inner_env.set_opponent_model(self._frozen_model)
        elif isinstance(self.env, BatchedDeckGymEnv):
            # BatchedDeckGymEnv: export to ONNX and set as opponent
            try:
                from deckgym.onnx_export import export_policy_to_onnx
                
                # Export current policy to ONNX file
                onnx_path = os.path.join(tempfile.gettempdir(), "frozen_opponent.onnx")
                export_policy_to_onnx(self.model, onnx_path, validate=False)
                
                # Set the ONNX model as opponent in VecGame
                self.env.vec_game.set_onnx_opponent(onnx_path, deterministic=False)
                
                if self.verbose > 0:
                    print(f"[ONNX] Set frozen opponent from {onnx_path}")
            except AttributeError as e:
                # set_onnx_opponent may not exist if compiled without onnx feature
                print(f"[WARNING] ONNX opponent not available (compile with 'onnx' feature): {e}")
            except Exception as e:
                print(f"[WARNING] Failed to set ONNX opponent: {e}")
        else:
            # DummyVecEnv: update all environments (Monitor -> ActionMasker -> SelfPlayEnv)
            for i in range(self.n_envs):
                self.env.envs[i].env.env.set_opponent_model(self._frozen_model)


class CurriculumCallback(BaseCallback):
    """
    Manages curriculum stage transitions using continuous win rate tracking.
    
    Tracks wins/losses from training episodes (no separate evaluation runs).
    Checks for stage advancement at each rollout end.
    """
    
    def __init__(
        self,
        env,
        curriculum,  # CurriculumManager instance
        n_envs: int = 1,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.env = env
        self.curriculum = curriculum
        self.n_envs = n_envs
        self._last_log_step = 0
    
    def _on_step(self) -> bool:
        """Track episode outcomes from training."""
        for info in self.locals.get("infos", []):
            if "episode" in info:
                reward = info["episode"].get("r", 0)
                self.curriculum.record_episode(won=(reward > 0))
        return True
    
    def _on_rollout_end(self) -> None:
        """Check for stage advancement at end of each rollout."""
        stage = self.curriculum.current_stage
        
        # Log progress periodically (every ~50k steps)
        if self.verbose > 0 and self.num_timesteps - self._last_log_step >= 50_000:
            self._last_log_step = self.num_timesteps
            wr = self.curriculum.win_rate
            count = self.curriculum.episode_count
            threshold = stage.win_rate_threshold
            threshold_str = f"{threshold:.0%}" if threshold else "∞"
            print(f"[Step {self.num_timesteps:,}] Stage: {stage.name}, WR: {wr:.1%} ({count} eps), threshold: {threshold_str}")
            
            # Log to TensorBoard if available
            if self.logger:
                self.logger.record("curriculum/win_rate", wr)
                self.logger.record("curriculum/stage_idx", self.curriculum.current_stage_idx)
                self.logger.record("curriculum/episode_count", count)
        
        # Check for stage advancement
        if self.curriculum.check_advancement():
            new_stage = self.curriculum.current_stage
            self._update_opponent_type(new_stage.opponent)
    
    def _update_opponent_type(self, opponent_type: str):
        """Update all environments to use the new opponent type."""
        if self.n_envs == 1:
            inner_env = self.env.env.env if hasattr(self.env.env, 'env') else self.env.env
            inner_env.set_opponent_type(opponent_type)
        elif isinstance(self.env, BatchedDeckGymEnv):
            # BatchedDeckGymEnv handles this directly
            self.env.set_opponent_type(opponent_type)
        else:
            # DummyVecEnv with SelfPlayEnv wrappers
            for i in range(self.n_envs):
                self.env.envs[i].env.env.set_opponent_type(opponent_type)
        
        if self.verbose > 0:
            print(f"[CURRICULUM] Switched all envs to opponent_type='{opponent_type}'")


# =============================================================================
# Action Masking & Environment Factory
# =============================================================================

def mask_fn(env: SelfPlayEnv) -> np.ndarray:
    """Provide action mask to MaskablePPO."""
    return env.action_masks()


def make_env(
    deck_loader: CurriculumDeckLoader,
    config: TrainingConfig,
    opponent_type: str = "e2",
) -> callable:
    """
    Factory function to create training environments.
    
    Args:
        deck_loader: Deck loader for sampling
        config: Training config
        opponent_type: "e2", "e3", or "self"
    
    For DummyVecEnv, all envs share the same deck_loader (efficient).
    Monitor wrapper enables episode stats for TensorBoard.
    """
    def _init() -> gym.Env:
        env = SelfPlayEnv(deck_loader, opponent_type=opponent_type, config=config)
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
    
    # Build config dict for logging/saving
    config_dict = {
        # Paths
        "simple_deck_path": config.simple_deck_path,
        "meta_deck_path": config.meta_deck_path,
        "save_path": config.save_path,
        "checkpoint_dir": config.checkpoint_dir,
        "tensorboard_dir": config.tensorboard_dir,
        "resume_path": config.resume_path,
        # Training duration
        "total_timesteps": config.total_timesteps,
        "checkpoint_freq": config.checkpoint_freq,
        # PPO hyperparameters
        "base_learning_rate": config.base_learning_rate,
        "min_learning_rate": config.min_learning_rate,
        "n_steps": config.n_steps,
        "batch_size": config.batch_size,
        "n_epochs": config.n_epochs,
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "ent_coef": config.ent_coef,
        "vf_coef": config.vf_coef,
        "clip_range": config.clip_range,
        "target_kl": config.target_kl,
        # Network architecture
        "use_attention": config.use_attention,
        "attention_embed_dim": config.attention_embed_dim,
        "attention_num_heads": config.attention_num_heads,
        "attention_num_layers": config.attention_num_layers,
        "policy_layers": list(config.policy_layers),
        "value_layers": list(config.value_layers),
        "use_silu": config.use_silu,
        # Self-play
        "frozen_opponent_update_freq": config.frozen_opponent_update_freq,
        "adaptive_frozen_opponent": config.adaptive_frozen_opponent,
        "target_win_rate": config.target_win_rate,
        # Game limits
        "max_turns": config.max_turns,
        "max_actions_per_turn": config.max_actions_per_turn,
        "max_total_actions": config.max_total_actions,
        # Environment
        "n_envs": config.n_envs,
        "use_batched_env": config.use_batched_env,
        "device": config.device,
    }
    
    print(f"\n{'─' * 60}")
    print("  TRAINING CONFIGURATION")
    print(f"{'─' * 60}")
    
    print("\n  [▸] Paths:")
    print(f"     Save path:         {config.save_path}")
    print(f"     Checkpoint dir:    {config.checkpoint_dir}")
    print(f"     TensorBoard dir:   {config.tensorboard_dir}")
    if config.resume_path:
        print(f"     Resume from:       {config.resume_path}")
    
    print("\n  [◷] Training Duration:")
    print(f"     Total timesteps:   {config.total_timesteps:,}")
    print(f"     Checkpoint freq:   {config.checkpoint_freq:,} (per env)")
    
    print("\n  [⬢] PPO Hyperparameters:")
    print(f"     Learning rate:     {config.base_learning_rate} → {config.min_learning_rate} (linear decay)")
    print(f"     Batch size:        {config.batch_size:,}")
    print(f"     N steps:           {config.n_steps:,} (buffer = {config.n_steps * config.n_envs:,})")
    print(f"     N epochs:          {config.n_epochs}")
    print(f"     Gamma:             {config.gamma}")
    print(f"     GAE lambda:        {config.gae_lambda}")
    print(f"     Entropy coef:      {config.ent_coef}")
    print(f"     Value coef:        {config.vf_coef}")
    print(f"     Clip range:        {config.clip_range}")
    print(f"     Target KL:         {config.target_kl}")
    
    print("\n  [△] Network Architecture:")
    if config.use_attention:
        print(f"     Type:              Attention-based (CardAttentionExtractor)")
        print(f"     Embed dim:         {config.attention_embed_dim}")
        print(f"     Num heads:         {config.attention_num_heads}")
        print(f"     Num layers:        {config.attention_num_layers}")
        print(f"     Policy head:       [256, 128]")
        print(f"     Value head:        [256, 128]")
    else:
        print(f"     Type:              MLP")
        print(f"     Policy layers:     {config.policy_layers}")
        print(f"     Value layers:      {config.value_layers}")
    print(f"     Activation:        {'SiLU' if config.use_silu else 'ReLU'}")
    
    print("\n  [↻] Self-Play:")
    print(f"     Opponent update:   every {config.frozen_opponent_update_freq:,} steps")
    print(f"     Adaptive:          {config.adaptive_frozen_opponent}")
    print(f"     Target win rate:   {config.target_win_rate:.0%}")
    
    print("\n  [⬟] Environment:")
    print(f"     N envs:            {config.n_envs}")
    print(f"     Batched (Rust):    {config.use_batched_env}")
    print(f"     Device:            {config.device}")
    print(f"     Max turns:         {config.max_turns}")
    
    print(f"{'─' * 60}\n")
    
    # Setup environment(s)
    print("\n[1/4] Loading decks...")
    deck_loader = CurriculumDeckLoader(config.simple_deck_path, config.meta_deck_path)
    meta_loader = MetaDeckLoader(config.meta_deck_path)  # For evaluation
    
    # Create curriculum manager for automatic stage transitions
    curriculum = CurriculumManager(
        simple_loader=MetaDeckLoader(config.simple_deck_path),
        meta_loader=meta_loader,
    )
    initial_opponent = curriculum.current_stage.opponent
    print(f"      Curriculum stage: {curriculum.current_stage.name} (vs {initial_opponent})")
    
    print(f"[2/4] Creating {config.n_envs} environment(s) with opponent_type='{initial_opponent}'...")
    if config.n_envs == 1:
        # Single environment (simpler)
        single_env = SelfPlayEnv(deck_loader, opponent_type=initial_opponent, config=config)
        env = ActionMasker(single_env, mask_fn)
        env = Monitor(env)  # Track ep_len, ep_rew for TensorBoard
    elif config.use_batched_env:
        # High-performance Rust-side batched environment
        print(f"      Using BatchedDeckGymEnv (Rust-side batching)")
        env = BatchedDeckGymEnv(
            n_envs=config.n_envs,
            deck_loader=deck_loader,
            opponent_type=initial_opponent,
        )
        single_env = None
    else:
        # Legacy: Multiple environments with DummyVecEnv
        env = DummyVecEnv([make_env(deck_loader, config, initial_opponent) for _ in range(config.n_envs)])
        single_env = None  # Will be set after model init
    
    # Verify observation space
    expected_obs_size = 2873  # 41 global + 24 cards × 118 features
    if env.observation_space.shape[0] != expected_obs_size:
        raise ValueError(
            f"Observation size mismatch! Expected {expected_obs_size}, "
            f"but environment returned {env.observation_space.shape[0]}. "
            "Check observation.rs and attention_policy.py constants."
        )
    print(f"      Observation space verified: {env.observation_space.shape}")
    
    # Setup model
    print("[3/4] Initializing model...")
    activation_fn = torch.nn.SiLU if config.use_silu else torch.nn.ReLU
    
    if config.use_attention:
        # Use attention-based policy with CardAttentionExtractor
        print(f"      Using attention policy (embed_dim={config.attention_embed_dim}, heads={config.attention_num_heads}, layers={config.attention_num_layers})")
        policy_kwargs = create_attention_policy_kwargs(
            embed_dim=config.attention_embed_dim,
            num_heads=config.attention_num_heads,
            num_layers=config.attention_num_layers,
            net_arch=dict(pi=[256, 128], vf=[256, 128]),  # Smaller heads after attention
        )
        policy_kwargs["activation_fn"] = activation_fn
    else:
        # Use standard MLP policy
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
        model.learning_rate = config.get_learning_rate_schedule(config.total_timesteps)
        model.n_steps = config.n_steps
        model.batch_size = config.batch_size
        model.n_epochs = config.n_epochs
        model.gamma = config.gamma
        model.gae_lambda = config.gae_lambda
        model.ent_coef = config.ent_coef
        model.vf_coef = config.vf_coef
        # clip_range must be a callable (schedule) for SB3
        model.clip_range = lambda _: config.clip_range
        model.target_kl = config.target_kl
        print(f"      Updated hyperparameters (LR={config.base_learning_rate}, target_kl={config.target_kl})")
    else:
        # Create new model from scratch
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=config.get_learning_rate_schedule(config.total_timesteps),
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,  # Gradient clipping
            clip_range=config.clip_range,
            clip_range_vf=config.clip_range,  # Clip value function updates to reduce spikes
            target_kl=config.target_kl,  # Stop epoch early if KL too high
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=config.device,
            tensorboard_log=config.tensorboard_dir,
        )
    
    # NOTE: torch.compile() is intentionally NOT used because it breaks model saving/loading.
    # The compiled model saves weights with "_orig_mod." prefix which SB3 cannot properly reload.
    # This results in models loading with random weights instead of trained weights.
    
    # Save training configuration metadata
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    config_path = os.path.join(config.checkpoint_dir, "training_config.json")
    with open(config_path, "w") as f:
        import json
        json.dump(config_dict, f, indent=2)
    print(f"      Config saved to {config_path}")
    
    # Also save to the TensorBoard run folder (MaskablePPO_*)
    if hasattr(model, 'logger') and model.logger is not None:
        tb_log_dir = model.logger.dir
        if tb_log_dir:
            config_tb_path = os.path.join(tb_log_dir, "training_config.json")
            with open(config_tb_path, "w") as f:
                json.dump(config_dict, f, indent=2)
            print(f"      Config saved to {config_tb_path}")
    
    # Setup callbacks
    callbacks = [
        CheckpointCallback(
            save_freq=config.checkpoint_freq,
            save_path=config.checkpoint_dir,
            name_prefix="rl_bot",
        ),
        CurriculumCallback(
            env,
            curriculum=curriculum,
            n_envs=config.n_envs,
            verbose=1,
        ),
        EpisodeMetricsCallback(verbose=0),  # Log actions_per_turn_mean
    ]
    
    # FrozenOpponentCallback only needed for self-play stage
    if initial_opponent == "self":
        callbacks.append(
            FrozenOpponentCallback(
                env,
                n_envs=config.n_envs,
                update_freq=config.frozen_opponent_update_freq,
                adaptive=config.adaptive_frozen_opponent,
                target_win_rate=config.target_win_rate,
                verbose=1,
            )
        )
    
    # Initialize frozen opponent only for self-play stage
    if initial_opponent == "self":
        print("      Loading frozen opponent on CPU...")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "initial_frozen")
            model.save(path)
            initial_frozen = MaskablePPO.load(path, device="cpu")
        
        if config.n_envs == 1:
            env.env.env.set_opponent_model(initial_frozen)
        elif isinstance(env, BatchedDeckGymEnv):
            # BatchedDeckGymEnv doesn't support self-play yet
            print("      [WARNING] Self-play not supported with BatchedDeckGymEnv")
            print("                Falling back to bot opponents. Set use_batched_env=False for self-play.")
        else:
            for i in range(config.n_envs):
                env.envs[i].env.env.set_opponent_model(initial_frozen)
    
    # Train
    print("[4/4] Starting training...")
    print(f"      Checkpoints: every {config.checkpoint_freq:,} steps")
    if initial_opponent == "self":
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
    
    # Create default config for default values
    _defaults = TrainingConfig()
    
    # Config file (YAML) - load entire config from file
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file (e.g., configs/large_model.yaml)")
    
    # Paths
    parser.add_argument("--simple", default=_defaults.simple_deck_path, help="Simple deck definitions")
    parser.add_argument("--meta", default=_defaults.meta_deck_path, help="Meta deck definitions")
    parser.add_argument("--save", default=_defaults.save_path, help="Model save path")
    
    # Training
    parser.add_argument("--steps", type=int, default=_defaults.total_timesteps, help="Total training steps")
    parser.add_argument("--checkpoint-freq", type=int, default=_defaults.checkpoint_freq, help="Checkpoint frequency")
    
    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=_defaults.base_learning_rate, help="Base learning rate (linear decay)")
    parser.add_argument("--min-lr", type=float, default=_defaults.min_learning_rate, help="Minimum learning rate floor")
    parser.add_argument("--target-kl", type=float, default=_defaults.target_kl, help="Target KL for early stopping")
    parser.add_argument("--batch-size", type=int, default=_defaults.batch_size, help="Batch size")
    parser.add_argument("--n-epochs", type=int, default=_defaults.n_epochs, help="PPO epochs per update")
    parser.add_argument("--ent-coef", type=float, default=_defaults.ent_coef, help="Entropy coefficient")
    
    # Self-play
    parser.add_argument("--opponent-update-freq", type=int, default=_defaults.frozen_opponent_update_freq, help="Frozen opponent update frequency")
    
    # Parallelization
    parser.add_argument("--n-envs", type=int, default=_defaults.n_envs, help="Number of parallel environments")
    parser.add_argument("--no-batched-env", action="store_true", help="Disable Rust-side batching (use DummyVecEnv instead) [Not recommanded]")
    
    # Resume
    parser.add_argument("--resume", default=_defaults.resume_path, help="Path to checkpoint to resume training from")
    
    # Attention policy (enabled by default now)
    parser.add_argument("--no-attention", action="store_true", help="Disable attention, use MLP instead")
    parser.add_argument("--attention-dim", type=int, default=None, help="Attention embedding dimension (overrides config)")
    parser.add_argument("--attention-heads", type=int, default=None, help="Number of attention heads (overrides config)")
    parser.add_argument("--attention-layers", type=int, default=None, help="Number of transformer layers (overrides config)")
    
    # Device
    parser.add_argument("--device", default=_defaults.device, choices=["cpu", "cuda", "auto"])
    
    args = parser.parse_args()
    
    # Load config from YAML if provided, otherwise use defaults
    if args.config:
        print(f"Loading configuration from {args.config}...")
        config = load_config_from_yaml(args.config)
        print(f"  Loaded config: {Path(args.config).stem}")
    else:
        config = TrainingConfig()
    
    # Override config with CLI arguments (CLI takes precedence)
    if args.simple != _defaults.simple_deck_path:
        config.simple_deck_path = args.simple
    if args.meta != _defaults.meta_deck_path:
        config.meta_deck_path = args.meta
    if args.save != _defaults.save_path:
        config.save_path = args.save
    if args.resume is not None:
        config.resume_path = args.resume
    if args.steps != _defaults.total_timesteps:
        config.total_timesteps = args.steps
    if args.checkpoint_freq != _defaults.checkpoint_freq:
        config.checkpoint_freq = args.checkpoint_freq
    if args.lr != _defaults.base_learning_rate:
        config.base_learning_rate = args.lr
    if args.min_lr != _defaults.min_learning_rate:
        config.min_learning_rate = args.min_lr
    if args.target_kl != _defaults.target_kl:
        config.target_kl = args.target_kl
    if args.batch_size != _defaults.batch_size:
        config.batch_size = args.batch_size
    if args.n_epochs != _defaults.n_epochs:
        config.n_epochs = args.n_epochs
    if args.ent_coef != _defaults.ent_coef:
        config.ent_coef = args.ent_coef
    if args.opponent_update_freq != _defaults.frozen_opponent_update_freq:
        config.frozen_opponent_update_freq = args.opponent_update_freq
    if args.n_envs != _defaults.n_envs:
        config.n_envs = args.n_envs
    if args.no_batched_env:
        config.use_batched_env = False
    if args.no_attention:
        config.use_attention = False
    if args.attention_dim is not None:
        config.attention_embed_dim = args.attention_dim
    if args.attention_heads is not None:
        config.attention_num_heads = args.attention_heads
    if args.attention_layers is not None:
        config.attention_num_layers = args.attention_layers
    if args.device != _defaults.device:
        config.device = args.device
    
    train(config)

