#!/usr/bin/env python3
"""
Environment module for DeckGym RL training.

Provides Gymnasium-compatible environments for training RL agents
on the Pokemon TCG Pocket simulator.
"""

from deckgym.envs.base import DeckGymEnv
from deckgym.envs.batched import BatchedDeckGymEnv
from deckgym.envs.self_play import SelfPlayEnv

__all__ = [
    "DeckGymEnv",
    "BatchedDeckGymEnv",
    "SelfPlayEnv",
]
