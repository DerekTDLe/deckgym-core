#!/usr/bin/env python3
"""
EpisodeMetricsCallback - Logs custom episode metrics to TensorBoard.

Tracks metrics from BatchedDeckGymEnv's episode info dict:
- actions_per_turn: Average actions taken between EndTurn calls

Also shows training phase indicators (Collecting vs Optimizing).
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


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
