#!/usr/bin/env python3
"""
Feature extractors for RL policy networks.

Provides attention-based and other extractors for processing game observations.
"""

from deckgym.models.extractors.attention import (
    CardAttentionExtractor,
    CardAttentionPolicy,
    create_attention_policy_kwargs,
    diagnose_gradients,
)
from deckgym.models.extractors.layers import (
    OnnxSafeAttention,
    MultiHeadAttentionPooling,
)

__all__ = [
    "CardAttentionExtractor",
    "CardAttentionPolicy",
    "create_attention_policy_kwargs",
    "diagnose_gradients",
    "OnnxSafeAttention",
    "MultiHeadAttentionPooling",
]
