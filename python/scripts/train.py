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
from pathlib import Path
from typing import Optional

import numpy as np
import gymnasium as gym
import torch

# Enable TensorFloat32 for faster matmul on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from deckgym.env import DeckGymEnv
from deckgym.deck_loader import CurriculumDeckLoader, MetaDeckLoader
from deckgym.attention_policy import CardAttentionExtractor, create_attention_policy_kwargs
from deckgym.curriculum import CurriculumManager

import warnings
warnings.filterwarnings("ignore", message=".*nested tensors.*")

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
    base_learning_rate: float = 5e-5      # Lower LR for attention model (was 1e-4)
    min_learning_rate: float = 1e-5       # Lower floor LR (was 3e-5)
    n_steps: int = 8192                   # More experience per update
    batch_size: int = 512                 # Minibatch size for optimization
    n_epochs: int = 8                     # Reduced to compensate for larger n_steps
    gamma: float = 0.98                   # Reduced for shorter episodes (was 0.99)
    gae_lambda: float = 0.95              # GAE lambda
    ent_coef: float = 0.02                # Higher entropy for larger model (was 0.01)
    vf_coef: float = 0.5                  # Value function coefficient (default)
    clip_range: float = 0.2               # PPO clipping range
    target_kl: float = 0.015              # Stop epoch early if KL > target (stabilizes training)
    
    # Network architecture
    policy_layers: tuple = (512, 512, 256, 128)   # Deeper policy network (MLP only)
    value_layers: tuple = (512, 512, 256)         # Match policy depth for better value estimation (MLP only)
    use_attention: bool = True                    # Default to attention-based policy
    attention_embed_dim: int = 128                # Balance speed/quality
    attention_num_heads: int = 4                  # Balance speed/quality
    attention_num_layers: int = 2                 # Keep 2 layers for capacity
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
        Simple linear decay from base_lr to min_lr.
        
        Standard approach that works well for both curriculum and self-play.
        """
        base_lr = self.base_learning_rate
        min_lr = self.min_learning_rate
        
        def lr_schedule(progress_remaining: float) -> float:
            # Linear decay: base_lr at start, min_lr at end
            return min_lr + (base_lr - min_lr) * progress_remaining
        
        return lr_schedule


# Default configuration
DEFAULT_CONFIG = TrainingConfig()


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
                        # Determine reward from winner
                        state = self._env.game.get_state()
                        if state.winner and state.winner.winner == 0:
                            final_reward = 1.0  # Agent won
                        elif state.winner and state.winner.winner == 1:
                            final_reward = -1.0  # Opponent won
                        else:
                            final_reward = 0.0  # Draw
                
                if done:
                    break
        except BaseException as e:
            print(f"WARNING: Game error during opponent turn: {e}")
            return obs, info, 0.0, True
        
        return obs, info, final_reward, done
    
    def _select_opponent_action(self, obs) -> int:
        """Select action for the opponent based on opponent_type."""
        if self.opponent_type == "self":
            # Self-play: use frozen model
            action_mask = self._env.action_masks()
            if self.opponent_model is not None:
                action, _ = self.opponent_model.predict(obs, action_masks=action_mask, deterministic=False)
            else:
                valid_actions = np.where(action_mask)[0]
                action = np.random.choice(valid_actions)
            return action
        else:
            # Bot opponent: use play_tick() which returns the action
            # Note: play_tick() both selects AND executes the action
            # So we return -1 as a sentinel that action was already executed
            if self._bot_game is not None:
                self._bot_game.play_tick()
            return -1  # Sentinel: action already executed via play_tick()
    
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


class CurriculumEvalCallback(BaseCallback):
    """
    Periodically evaluates the agent and manages curriculum stage transitions.
    
    Uses quick_eval_vs_bot() for fast evaluation against the current stage opponent.
    Automatically advances stages when win rate threshold is reached.
    """
    
    def __init__(
        self,
        env,
        curriculum,  # CurriculumManager instance
        deck_loader: MetaDeckLoader,
        n_envs: int = 1,
        eval_freq: int = 100_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.env = env
        self.curriculum = curriculum
        self.deck_loader = deck_loader
        self.n_envs = n_envs
        self.eval_freq = eval_freq
        self.last_eval = 0
    
    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval < self.eval_freq:
            return True
        
        self.last_eval = self.num_timesteps
        stage = self.curriculum.current_stage
        
        # Skip evaluation for self-play stage
        if stage.opponent == "self":
            if self.verbose > 0:
                print(f"[Step {self.num_timesteps:,}] In self-play stage, skipping eval")
            return True
        
        # Run quick evaluation
        if self.verbose > 0:
            print(f"[Step {self.num_timesteps:,}] Evaluating vs {stage.opponent} ({stage.eval_games} games)...")
        
        # Import here to avoid circular imports
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from evaluate import quick_eval_vs_bot
        
        win_rate = quick_eval_vs_bot(
            self.model,
            stage.opponent,
            self.deck_loader,
            n_games=stage.eval_games,
            verbose=False,
        )
        
        if self.verbose > 0:
            print(f"[Step {self.num_timesteps:,}] WR vs {stage.opponent}: {win_rate:.1%} (threshold: {stage.win_rate_threshold:.0%})")
        
        # Check for stage advancement
        if self.curriculum.check_advancement(win_rate):
            new_stage = self.curriculum.current_stage
            self._update_opponent_type(new_stage.opponent)
        
        return True
    
    def _update_opponent_type(self, opponent_type: str):
        """Update all environments to use the new opponent type."""
        if self.n_envs == 1:
            inner_env = self.env.env.env if hasattr(self.env.env, 'env') else self.env.env
            inner_env.set_opponent_type(opponent_type)
        else:
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
    else:
        # Multiple environments with DummyVecEnv (no IPC overhead)
        env = DummyVecEnv([make_env(deck_loader, config, initial_opponent) for _ in range(config.n_envs)])
        single_env = None  # Will be set after model init
    
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
        model.learning_rate = config.get_learning_rate_schedule()
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
            learning_rate=config.get_learning_rate_schedule(),  # Cyclical LR
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
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
        CurriculumEvalCallback(
            env,
            curriculum=curriculum,
            deck_loader=meta_loader,
            n_envs=config.n_envs,
            eval_freq=100_000,  # Evaluate every 100k steps
            verbose=1,
        ),
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
        else:
            for i in range(config.n_envs):
                env.envs[i].env.env.set_opponent_model(initial_frozen)
    
    # Train
    print("[4/4] Starting training...")
    print(f"      Checkpoints: every {config.checkpoint_freq:,} steps")
    print(f"      Curriculum eval: every 100,000 steps")
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
    
    # Paths
    parser.add_argument("--simple", default="simple_deck.json", help="Simple deck definitions")
    parser.add_argument("--meta", default="meta_deck.json", help="Meta deck definitions")
    parser.add_argument("--save", default="models/rl_bot", help="Model save path")
    
    # Training
    parser.add_argument("--steps", type=int, default=30_000_000, help="Total training steps")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000, help="Checkpoint frequency")
    
    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=5e-5, help="Base learning rate (cyclical)")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Minimum learning rate floor")
    parser.add_argument("--target-kl", type=float, default=0.015, help="Target KL for early stopping")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--n-epochs", type=int, default=8, help="PPO epochs per update")
    parser.add_argument("--ent-coef", type=float, default=0.02, help="Entropy coefficient")
    
    # Self-play
    parser.add_argument("--opponent-update-freq", type=int, default=75_000, help="Frozen opponent update frequency")
    parser.add_argument("--warmup-ratio", type=float, default=0.10, help="Curriculum warmup ratio (0-1)")
    
    # Parallelization
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments (DummyVecEnv)")
    
    # Resume
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume training from")
    
    # Attention policy (enabled by default now)
    parser.add_argument("--no-attention", action="store_true", help="Disable attention, use MLP instead")
    parser.add_argument("--attention-dim", type=int, default=128, help="Attention embedding dimension")
    parser.add_argument("--attention-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--attention-layers", type=int, default=2, help="Number of transformer layers")
    
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
        use_attention=not args.no_attention,
        attention_embed_dim=args.attention_dim,
        attention_num_heads=args.attention_heads,
        attention_num_layers=args.attention_layers,
        device=args.device,
    )
    
    train(config)

