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
torch.set_float32_matmul_precision("high")

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from deckgym.env import DeckGymEnv
from deckgym.batched_env import BatchedDeckGymEnv
from deckgym.deck_loader import MetaDeckLoader
from deckgym.attention_policy import (
    CardAttentionExtractor,
    create_attention_policy_kwargs,
)
from deckgym.pfsp_callback import PFSPCallback

# Import configuration and constants
from deckgym.config import TrainingConfig, DEFAULT_CONFIG, OBSERVATION_SIZE

# =============================================================================
# Configuration
# =============================================================================


# Configuration is now centralized in deckgym/config.py


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
        from deckgym.diagnostic_logger import get_logger

        self.diagnostic_logger = get_logger()

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
        self.diagnostic_logger.clear_history(0)  # env_idx 0 for single env

        return obs, info

    def _replace_game_with_bot(self, deck_a: str, deck_b: str, seed=None):
        """Replace _env.game with a PyGame that has bot opponent."""
        import deckgym

        # Write decks to temp files (PyGame requires file paths for bot support)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(deck_a)
            path_a = f.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(deck_b)
            path_b = f.name

        try:
            # Create game with: player 0 = random (agent controls), player 1 = bot
            bot_game = deckgym.Game(
                path_a, path_b, players=["r", self.opponent_type], seed=seed
            )
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
        self.diagnostic_logger.record_action(0, action)
        try:
            obs, reward, done, truncated, info = self._env.step(action)
        except BaseException as e:
            print(f"WARNING: Game error during agent turn: {e}")
            self.diagnostic_logger.log_error(
                "agent_panic", 0, self._env.game.get_state(), {"error": str(e)}
            )
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
            while (
                self._env.game.current_player() == 1
                and not self._env.game.is_game_over()
            ):
                turn_actions += 1
                self._episode_actions += 1

                # Safety limits
                if turn_actions > self.config.max_actions_per_turn:
                    self._log_freeze_warning("opponent_turn_limit", turn_actions)
                    return obs, info, 0.0, True

                if self._episode_actions > self.config.max_total_actions:
                    self._log_freeze_warning(
                        "episode_action_limit", self._episode_actions
                    )
                    return obs, info, 0.0, True

                if self.opponent_type == "self":
                    # Self-play: select action with frozen model, execute in _env
                    action = self._select_opponent_action(obs)
                    obs, reward, done, truncated, info = self._env.step(action)
                    final_reward = (
                        -reward
                    )  # Invert reward (opponent's loss = agent's gain)
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
                                # Agent won: (1.0 + diff/6) * speed
                                # 3-0: (1 + 3/6) = 1.5 * speed
                                # 3-2: (1 + 1/6) = 1.16 * speed
                                final_reward = (1.0 + (point_diff / 6.0)) * max(
                                    1.0, speed_factor
                                )
                            else:
                                # Opponent won: (-1.0 + diff/6) * speed
                                # 0-3 (diff -3): (-1 + -0.5) = -1.5 * speed (Heavy penalty)
                                # 2-3 (diff -1): (-1 + -0.16) = -1.16 * speed (Lighter penalty)
                                final_reward = (
                                    -1.0 + (point_diff / 6.0)
                                ) * speed_factor
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
            action, _ = self.opponent_model.predict(
                obs, action_masks=action_mask, deterministic=False
            )
        else:
            valid_actions = np.where(action_mask)[0]
            action = np.random.choice(valid_actions)
        return action

    def _get_turn_count(self) -> int:
        """Get current turn count from game state."""
        return self._env.game.get_state().turn_count

    def _end_episode_truncated(self, reason: str):
        """End episode due to limit exceeded."""
        self.diagnostic_logger.log_error(reason, 0, self._env.game.get_state())
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
        print(
            f"WARNING: {reason} triggered (count={count}, turn={turn}, player={player})"
        )
        self.diagnostic_logger.log_error(
            reason, 0, self._env.game.get_state(), {"count": count}
        )


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
                    self._actions_per_turn_buffer.append(
                        episode_info["actions_per_turn"]
                    )

        return True


class FrozenOpponentCallback(BaseCallback):
    """
    Periodically updates the frozen opponent with current agent weights.

    Updates happen every N rollouts (not timesteps) for more predictable behavior.
    Exports policy to ONNX for fast Rust-side inference.
    """

    def __init__(
        self,
        env,
        n_envs: int = 1,
        update_every_n_rollouts: int = 8,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.env = env
        self.n_envs = n_envs
        self.update_every_n_rollouts = update_every_n_rollouts
        self.rollout_count = 0
        self._frozen_model = None

    def _on_rollout_end(self) -> None:
        """Called at end of each rollout collection."""
        self.rollout_count += 1

        if self.rollout_count % self.update_every_n_rollouts == 0:
            self._update_frozen_opponent()

    def _on_step(self) -> bool:
        return True

    def _update_frozen_opponent(self):
        """Export current model to ONNX and set as opponent."""
        import gc

        # Clean up old frozen model
        if self._frozen_model is not None:
            del self._frozen_model
            gc.collect()

        if isinstance(self.env, BatchedDeckGymEnv):
            # BatchedDeckGymEnv: export to ONNX for Rust-side inference
            try:
                from deckgym.onnx_export import export_policy_to_onnx

                # Export current policy to ONNX file
                onnx_path = os.path.join(tempfile.gettempdir(), "frozen_opponent.onnx")
                export_policy_to_onnx(self.model, onnx_path, validate=False)

                # Set the ONNX model as opponent in VecGame
                self.env.vec_game.set_onnx_opponent(onnx_path, deterministic=False)

                if self.verbose > 0:
                    print(
                        f"[Rollout {self.rollout_count}] Updated frozen opponent (ONNX)"
                    )
            except AttributeError as e:
                print(f"[WARNING] ONNX opponent not available: {e}")
            except Exception as e:
                print(f"[WARNING] Failed to set ONNX opponent: {e}")
        else:
            # CPU-based frozen model for non-batched envs
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "frozen_model")
                self.model.save(path)
                self._frozen_model = MaskablePPO.load(path, device="cpu")

            if self.n_envs == 1:
                inner_env = (
                    self.env.env.env if hasattr(self.env.env, "env") else self.env.env
                )
                inner_env.set_opponent_model(self._frozen_model)
            else:
                for i in range(self.n_envs):
                    self.env.envs[i].env.env.set_opponent_model(self._frozen_model)

            if self.verbose > 0:
                print(f"[Rollout {self.rollout_count}] Updated frozen opponent (CPU)")


# =============================================================================
# Action Masking & Environment Factory
# =============================================================================


def mask_fn(env: SelfPlayEnv) -> np.ndarray:
    """Provide action mask to MaskablePPO."""
    return env.action_masks()


def make_env(
    deck_loader: MetaDeckLoader,
    config: TrainingConfig,
) -> callable:
    """
    Factory function to create training environments.

    Args:
        deck_loader: Deck loader for sampling
        config: Training config

    Uses self-play (opponent_type=None) by default.
    """

    def _init() -> gym.Env:
        env = SelfPlayEnv(deck_loader, opponent_type=None, config=config)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
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
        "warmup_ratio": config.warmup_ratio,
        "n_steps": config.n_steps,
        "batch_size": config.batch_size,
        "n_epochs": config.n_epochs,
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "ent_coef": config.ent_coef,
        "vf_coef": config.vf_coef,
        "clip_range": config.clip_range,
        "max_grad_norm": config.max_grad_norm,
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
        "frozen_opponent_update_rollouts": config.frozen_opponent_update_rollouts,
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
    print("  TRAINING CONFIGURATION (Self-Play)")
    print(f"{'─' * 60}")

    print("\n  [▸] Paths:")
    print(f"     Meta decks:        {config.meta_deck_path}")
    print(f"     Checkpoint dir:    {config.checkpoint_dir}")
    if config.resume_path:
        print(f"     Resume from:       {config.resume_path}")

    print("\n  [◷] Training Duration:")
    print(f"     Total timesteps:   {config.total_timesteps:,}")
    print(f"     Checkpoint freq:   {config.checkpoint_freq:,}")

    print("\n  [⬢] PPO Hyperparameters:")
    print(
        f"     Learning rate:     0 → {config.base_learning_rate} → {config.min_learning_rate} (warmup {config.warmup_ratio:.0%})"
    )
    print(f"     Batch size:        {config.batch_size:,}")
    print(
        f"     Rollout buffer:    {config.n_steps * config.n_envs:,} ({config.n_steps} steps × {config.n_envs} envs)"
    )
    print(f"     N epochs:          {config.n_epochs}")
    print(f"     Gradient clip:     {config.max_grad_norm}")

    print("\n  [△] Network Architecture:")
    if config.use_attention:
        print(f"     Type:              Attention (CardAttentionExtractor)")
        print(f"     Embed dim:         {config.attention_embed_dim}")
        print(f"     Heads:             {config.attention_num_heads}")
        print(f"     Layers:            {config.attention_num_layers}")
        print(f"     Policy head:       {list(config.policy_layers)}")
        print(f"     Value head:        {list(config.value_layers)}")
    else:
        print(f"     Type:              MLP")
        print(f"     Policy layers:     {config.policy_layers}")
        print(f"     Value layers:      {config.value_layers}")

    print("\n  [↻] Self-Play:")
    print(
        f"     Opponent update:   every {config.frozen_opponent_update_rollouts} rollouts"
    )
    print(f"     Mode:              Pure self-play (ONNX frozen opponent)")

    print(f"{'─' * 60}\n")

    # Setup environment(s)
    print("[1/4] Loading meta decks...")
    deck_loader = MetaDeckLoader(config.meta_deck_path)

    print(f"[2/4] Creating {config.n_envs} environment(s) for self-play...")
    if config.n_envs == 1:
        # Single environment (simpler)
        single_env = SelfPlayEnv(deck_loader, opponent_type="self", config=config)
        env = ActionMasker(single_env, mask_fn)
        env = Monitor(env)
    elif config.use_batched_env:
        # High-performance Rust-side batched environment
        print(f"      Using BatchedDeckGymEnv (Rust-side batching)")
        env = BatchedDeckGymEnv(
            n_envs=config.n_envs,
            deck_loader=deck_loader,
            opponent_type=None,  # Self-play via ONNX
            config=config,
        )
        single_env = None
    else:
        # Legacy: Multiple environments with DummyVecEnv
        env = DummyVecEnv([make_env(deck_loader, config) for _ in range(config.n_envs)])
        single_env = None

    # Verify observation space
    expected_obs_size = OBSERVATION_SIZE
    if env.observation_space.shape[0] != expected_obs_size:
        raise ValueError(
            f"Observation size mismatch! Expected {expected_obs_size}, "
            f"but environment returned {env.observation_space.shape[0]}. "
            "Check observation.rs and attention_policy.py constants."
        )
    print(f"      Observation space verified: {env.observation_space.shape}")

    # Setup model
    print("[3/4] Initializing model...")
    if torch.cuda.is_available():
        print(f"      PyTorch Training Device: {torch.cuda.get_device_name(0)} (CUDA)")
    else:
        print(f"      PyTorch Training Device: CPU")
    activation_fn = torch.nn.SiLU if config.use_silu else torch.nn.ReLU

    if config.use_attention:
        # Use attention-based policy with CardAttentionExtractor
        print(
            f"      Using attention policy (embed_dim={config.attention_embed_dim}, heads={config.attention_num_heads}, layers={config.attention_num_layers})"
        )
        policy_kwargs = create_attention_policy_kwargs(config)
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
        print(
            f"      Updated hyperparameters (LR={config.base_learning_rate}, target_kl={config.target_kl})"
        )
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
    if hasattr(model, "logger") and model.logger is not None:
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
        EpisodeMetricsCallback(verbose=0),
    ]

    # Add PFSP or simple frozen opponent callback
    if config.use_pfsp:
        print(f"      Using PFSP with pool_size={config.pfsp_pool_size}")
        callbacks.append(
            PFSPCallback(
                env,
                n_envs=config.n_envs,
                pool_size=config.pfsp_pool_size,
                add_to_pool_every_n_rollouts=config.pfsp_add_every_n_rollouts,
                select_opponent_every_n_rollouts=config.pfsp_select_every_n_rollouts,
                priority_exponent=config.pfsp_priority_exponent,
                winrate_window=config.pfsp_winrate_window,
                checkpoint_dir=config.pfsp_checkpoint_dir,
                verbose=1,
            )
        )
    else:
        callbacks.append(
            FrozenOpponentCallback(
                env,
                n_envs=config.n_envs,
                update_every_n_rollouts=config.frozen_opponent_update_rollouts,
                verbose=1,
            )
        )

    # Initialize frozen opponent with initial weights
    print("      Initializing frozen opponent...")
    if isinstance(env, BatchedDeckGymEnv):
        try:
            from deckgym.onnx_export import export_policy_to_onnx

            onnx_path = os.path.join(tempfile.gettempdir(), "frozen_opponent.onnx")
            export_policy_to_onnx(model, onnx_path, validate=False)
            env.vec_game.set_onnx_opponent(
                onnx_path, deterministic=False, device=config.onnx_device
            )
            print(f"      ONNX opponent initialized")
        except Exception as e:
            print(f"      [WARNING] Could not initialize ONNX opponent: {e}")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "initial_frozen")
            model.save(path)
            initial_frozen = MaskablePPO.load(path, device="cpu")

        if config.n_envs == 1:
            env.env.env.set_opponent_model(initial_frozen)
        else:
            for i in range(config.n_envs):
                env.envs[i].env.env.set_opponent_model(initial_frozen)
        print(f"      CPU frozen opponent initialized")

    # Train
    print("[4/4] Starting training...")
    print(f"      Checkpoints: every {config.checkpoint_freq:,} steps")
    if config.use_pfsp:
        print(
            f"      PFSP: pool_size={config.pfsp_pool_size}, priority_exp={config.pfsp_priority_exponent}"
        )
        print(
            f"      PFSP: add every {config.pfsp_add_every_n_rollouts} rollouts, select every {config.pfsp_select_every_n_rollouts} rollouts"
        )
    else:
        print(
            f"      Opponent updates: every {config.frozen_opponent_update_rollouts} rollouts"
        )

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
    _defaults = DEFAULT_CONFIG

    # Config file (YAML) - load entire config from file
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Paths
    parser.add_argument(
        "--meta", default=_defaults.meta_deck_path, help="Meta deck definitions"
    )
    parser.add_argument("--save", default=_defaults.save_path, help="Model save path")

    # Training
    parser.add_argument(
        "--steps",
        type=int,
        default=_defaults.total_timesteps,
        help="Total training steps",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=_defaults.checkpoint_freq,
        help="Checkpoint frequency",
    )

    # PPO hyperparameters
    parser.add_argument(
        "--lr",
        type=float,
        default=_defaults.base_learning_rate,
        help="Peak learning rate (after warmup)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=_defaults.min_learning_rate,
        help="Final learning rate",
    )
    parser.add_argument(
        "--batch-size", type=int, default=_defaults.batch_size, help="Batch size"
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=_defaults.n_steps,
        help="Steps per env per rollout",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=_defaults.n_epochs, help="PPO epochs per update"
    )
    parser.add_argument(
        "--ent-coef", type=float, default=_defaults.ent_coef, help="Entropy coefficient"
    )

    # Self-play
    parser.add_argument(
        "--opponent-update-rollouts",
        type=int,
        default=_defaults.frozen_opponent_update_rollouts,
        help="Update frozen opponent every N rollouts",
    )

    # Parallelization
    parser.add_argument(
        "--n-envs",
        type=int,
        default=_defaults.n_envs,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--no-batched-env", action="store_true", help="Disable Rust-side batching"
    )

    # Resume
    parser.add_argument(
        "--resume",
        default=_defaults.resume_path,
        help="Path to checkpoint to resume from",
    )

    # Attention policy
    parser.add_argument(
        "--no-attention", action="store_true", help="Disable attention, use MLP instead"
    )
    parser.add_argument(
        "--attention-dim", type=int, default=None, help="Attention embedding dimension"
    )
    parser.add_argument(
        "--attention-heads", type=int, default=None, help="Number of attention heads"
    )
    parser.add_argument(
        "--attention-layers",
        type=int,
        default=None,
        help="Number of transformer layers",
    )

    # Device
    parser.add_argument(
        "--device", default=_defaults.device, choices=["cpu", "cuda", "auto"]
    )

    args = parser.parse_args()

    # Load config from YAML if provided, otherwise use defaults
    if args.config:
        print(f"Loading configuration from {args.config}...")
        config = TrainingConfig.from_yaml(args.config)
        print(f"  Loaded config: {Path(args.config).stem}")
    else:
        config = TrainingConfig()

    # Override config with CLI arguments (CLI takes precedence)
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
    if args.batch_size != _defaults.batch_size:
        config.batch_size = args.batch_size
    if args.n_steps != _defaults.n_steps:
        config.n_steps = args.n_steps
    if args.n_epochs != _defaults.n_epochs:
        config.n_epochs = args.n_epochs
    if args.ent_coef != _defaults.ent_coef:
        config.ent_coef = args.ent_coef
    if args.opponent_update_rollouts != _defaults.frozen_opponent_update_rollouts:
        config.frozen_opponent_update_rollouts = args.opponent_update_rollouts
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
