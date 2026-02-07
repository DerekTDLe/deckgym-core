#!/usr/bin/env python3
"""
Reusable attention layers for RL policy networks.

Provides ONNX-compatible attention modules with gradient stability improvements.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from deckgym.config import (
    MODEL_ATTENTION_DROPOUT,
    MODEL_ATTENTION_TEMPERATURE,
    MODEL_INIT_RESIDUAL_SCALE,
    MODEL_NUM_HEADS_DEFAULT,
    MODEL_ATTENTION_POOL_QUERIES,
)


class OnnxSafeAttention(nn.Module):
    """
    ONNX-compatible multi-head self-attention with gradient stability improvements.

    Changes from V1:
        - Explicit Xavier initialization for QKV projections
        - Scaled initialization for output projection (residual scaling)
        - Optional temperature parameter for attention logits
        - Attention stats tracking for debugging
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = MODEL_ATTENTION_DROPOUT,
        attention_temperature: float = MODEL_ATTENTION_TEMPERATURE,
        init_residual_scale: bool = MODEL_INIT_RESIDUAL_SCALE,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.temperature = attention_temperature

        # Fused QKV projection
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        # For monitoring attention saturation
        self.register_buffer("_attn_logit_std", torch.tensor(0.0))
        self.register_buffer("_attn_entropy", torch.tensor(0.0))

        # Initialize weights properly
        self._init_weights(init_residual_scale)

    def _init_weights(self, init_residual_scale: bool):
        """
        Initialize weights for stable gradient flow.

        - QKV: Xavier uniform for balanced variance
        - Output: Scaled by 1/sqrt(2*num_layers) for residual streams (GPT-2 style)
        """
        nn.init.xavier_uniform_(self.qkv_proj.weight)

        if init_residual_scale:
            # Scale down output projection for stable residuals
            # This makes attention blocks start closer to identity
            nn.init.xavier_uniform_(
                self.out_proj.weight, gain=1.0 / math.sqrt(2 * self.num_layers)
            )
        else:
            nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        track_stats: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            key_padding_mask: [batch, seq_len] where True = masked (ignored)
            track_stats: If True, update attention statistics for monitoring

        Returns:
            [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Fused QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.embed_dim)
        q, k, v = qkv.unbind(dim=2)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Convert key_padding_mask to attention mask for SDPA
        # SDPA expects: True = attend, False = mask (opposite of our convention)
        attn_mask = None
        if key_padding_mask is not None:
            # Expand mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
            attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)

        # Track attention statistics for debugging (only when requested)
        if track_stats:
            with torch.no_grad():
                # Manual attention for stats (not used for output)
                attn_logits = (
                    torch.matmul(q, k.transpose(-2, -1)) * self.scale / self.temperature
                )
                if key_padding_mask is not None:
                    mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
                    attn_logits = attn_logits.masked_fill(mask, float("-inf"))
                valid_logits = attn_logits[~attn_logits.isinf()]
                if valid_logits.numel() > 0:
                    self._attn_logit_std = valid_logits.std()
                attn_probs_for_stats = F.softmax(attn_logits, dim=-1)
                entropy = -(
                    attn_probs_for_stats * (attn_probs_for_stats + 1e-10).log()
                ).sum(dim=-1)
                self._attn_entropy = entropy.mean()

        # Use PyTorch SDPA (FlashAttention when available) - 2-4x faster
        # Apply temperature by scaling q (equivalent to dividing attention logits)
        q_scaled = q * (self.scale / self.temperature) ** 0.5
        k_scaled = k * (self.scale / self.temperature) ** 0.5

        attn_output = F.scaled_dot_product_attention(
            q_scaled,
            k_scaled,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        # Reshape back
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

        return self.out_proj(attn_output)

    def get_attention_stats(self) -> Dict[str, float]:
        """Return attention statistics for monitoring."""
        return {
            "attn_logit_std": self._attn_logit_std.item(),
            "attn_entropy": self._attn_entropy.item(),
        }


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-head attention pooling with gradient stability improvements.

    Changes from V1:
        - Proper initialization matching the attention layers
        - LayerNorm on queries for stable cross-attention
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = MODEL_NUM_HEADS_DEFAULT,
        num_queries: int = MODEL_ATTENTION_POOL_QUERIES,
        attention_temperature: float = MODEL_ATTENTION_TEMPERATURE,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.temperature = attention_temperature

        # Learned query vectors with proper initialization
        self.queries = nn.Parameter(torch.zeros(num_queries, embed_dim))

        # LayerNorm for queries (stabilizes cross-attention)
        self.query_norm = nn.LayerNorm(embed_dim)

        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize for stable gradients."""
        # Queries: small init so pooling starts near uniform
        nn.init.normal_(self.queries, std=0.02)

        # Projections: Xavier
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)  # Slightly scaled down

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            key_padding_mask: [batch, seq_len] where True = masked

        Returns:
            [batch, num_queries * embed_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Expand and normalize queries
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        queries = self.query_norm(queries)

        # Project queries
        q = self.q_proj(queries)

        # Fused KV projection
        kv = self.kv_proj(x)
        kv = kv.view(batch_size, seq_len, 2, self.embed_dim)
        k, v = kv.unbind(dim=2)

        # Reshape for multi-head
        q = q.view(
            batch_size, self.num_queries, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn_scores = (
            torch.matmul(q, k.transpose(-2, -1)) * self.scale / self.temperature
        )

        # Apply mask
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)

        # Weighted combination
        pooled = torch.matmul(attn_probs, v)

        # Reshape
        pooled = (
            pooled.transpose(1, 2)
            .contiguous()
            .view(batch_size, self.num_queries, self.embed_dim)
        )

        # Output projection
        pooled = self.out_proj(pooled)

        # Flatten
        return pooled.view(batch_size, -1)
