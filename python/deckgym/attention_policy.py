"""
Custom Attention-Based Policy for MaskablePPO.

This policy uses multi-head self-attention over card features to create a 
permutation-invariant representation of the game state.

Architecture:
    1. Split observation into (global_features, card_features)
    2. Pass cards through self-attention layers
    3. Pool attended cards (mean pooling)
    4. Concatenate with global features
    5. Pass through shared MLP
    6. Output policy (action logits) and value heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Tuple, Dict, Any, Optional, List

# Import constants from the Rust side (via deckgym)
try:
    import deckgym
    GLOBAL_FEATURES = deckgym.Game.global_features_size()
    FEATURES_PER_CARD = deckgym.Game.features_per_card()
    MAX_CARDS = deckgym.Game.max_cards_in_game()
except ImportError:
    # Fallback for development - must match observation.rs constants
    # GLOBAL_FEATURES = 1 (turn) + 2 (points) + 2 (deck_size) + 2 (hand_size) + 2 (discard_size) + 32 (2×2×8 deck_energy with dual type)
    GLOBAL_FEATURES = 41
    FEATURES_PER_CARD = 117  # intrinsic (108) + position (9, no visibility) features per card
    MAX_CARDS = 24  # 20 self + 4 opponent board
    # Total observation size: 41 + 24 × 117 = 2849 dims


class OnnxSafeAttention(nn.Module):
    """
    ONNX-compatible multi-head self-attention.
    
    Uses only basic linear layers and tensor ops that ONNX supports.
    PyTorch's nn.MultiheadAttention uses _native_multi_head_attention which isn't supported.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V projections 
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            key_padding_mask: [batch, seq_len] where True = masked (ignored)
        
        Returns:
            [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # [batch, seq_len, embed_dim]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention: [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores: [batch, num_heads, seq_len, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if key_padding_mask is not None:
            # Expand mask: [batch, 1, 1, seq_len]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)  # [batch, num_heads, seq_len, head_dim]
        
        # Reshape back: [batch, seq_len, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        return self.out_proj(attn_output)

class AttentionPooling(nn.Module):
    """
    Attention-based pooling using a learned query vector.
    
    Instead of mean-pooling, we learn a query that attends to all cards
    and produces a weighted combination based on relevance.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Learned query vector (like a [CLS] token but as a parameter)
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Projections for cross-attention
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim] - card embeddings after self-attention
            key_padding_mask: [batch, seq_len] where True = masked (empty card slot)
        
        Returns:
            [batch, embed_dim] - single pooled representation
        """
        batch_size, seq_len, _ = x.shape
        
        # Expand learned query for batch: [batch, 1, embed_dim]
        query = self.query.expand(batch_size, -1, -1)
        
        # Project query, keys, values
        q = self.q_proj(query)  # [batch, 1, embed_dim]
        k = self.k_proj(x)      # [batch, seq_len, embed_dim]
        v = self.v_proj(x)      # [batch, seq_len, embed_dim]
        
        # Reshape for multi-head: [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores: [batch, num_heads, 1, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask for empty card slots
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # Weighted combination: [batch, num_heads, 1, head_dim]
        pooled = torch.matmul(attn_probs, v)
        
        # Reshape: [batch, embed_dim]
        pooled = pooled.transpose(1, 2).contiguous().view(batch_size, self.embed_dim)
        
        return self.out_proj(pooled)

class CardAttentionExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that uses self-attention over card features.
    
    Input: Flat observation vector [global_features, card_0, card_1, ..., card_n, padding]
    Output: Fixed-size feature vector for policy/value heads
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        global_features: int = GLOBAL_FEATURES,
        features_per_card: int = FEATURES_PER_CARD,
        max_cards: int = MAX_CARDS,
    ):
        # Calculate output features dimension
        features_dim = embed_dim + 64  # Attended cards + processed global
        super().__init__(observation_space, features_dim)
        
        self.global_features = global_features
        self.features_per_card = features_per_card
        self.max_cards = max_cards
        self.embed_dim = embed_dim
        
        # Project global features
        self.global_proj = nn.Sequential(
            nn.Linear(global_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        
        # Project card features to embedding dimension
        self.card_embed = nn.Sequential(
            nn.Linear(features_per_card, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Positional encoding (learnable, for card order within zones if needed)
        # We don't use this for permutation invariance, but keeping for potential future use
        
        # Manual attention blocks for ONNX compatibility
        # Uses OnnxSafeAttention instead of PyTorch's MultiheadAttention
        self.attention_layers = nn.ModuleList([
            OnnxSafeAttention(embed_dim, num_heads, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.ReLU(),
                nn.Linear(embed_dim * 2, embed_dim),
            )
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers * 2)
        ])
        self.num_layers = num_layers
        
        self.attention_pool = AttentionPooling(embed_dim, num_heads)
        
        # Final projection after pooling
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # Split into global and card features
        global_feats = observations[:, :self.global_features]
        card_feats_flat = observations[:, self.global_features:]
        
        # Reshape cards: [batch, max_cards, features_per_card]
        card_feats = card_feats_flat.view(batch_size, self.max_cards, self.features_per_card)
        
        # Create mask for empty card slots (all zeros = padding)
        # A card is considered "present" if it has any non-zero features
        card_sum = card_feats.abs().sum(dim=-1)  # [batch, max_cards]
        card_mask = card_sum < 1e-6  # True = masked (empty slot)
        
        # Process global features
        global_out = self.global_proj(global_feats)
        
        # Embed cards
        x = self.card_embed(card_feats)  # [batch, max_cards, embed_dim]
        
        # Apply manual attention layers (ONNX compatible)
        for i in range(self.num_layers):
            # Self-attention with residual (OnnxSafeAttention returns just output)
            attn_out = self.attention_layers[i](x, key_padding_mask=card_mask)
            x = self.layer_norms[i * 2](x + attn_out)
            
            # Feed-forward with residual
            ff_out = self.ff_layers[i](x)
            x = self.layer_norms[i * 2 + 1](x + ff_out)
        
        # Attention-based pooling (learned query attends to cards)
        pooled = self.attention_pool(x, key_padding_mask=card_mask)  # [batch, embed_dim]
        
        # Final projection
        cards_out = self.output_proj(pooled)
        
        # Concatenate global and card representations
        combined = torch.cat([global_out, cards_out], dim=1)
        
        return combined


class CardAttentionPolicy(ActorCriticPolicy):
    """
    Custom policy that uses CardAttentionExtractor for feature extraction.
    
    This is compatible with MaskablePPO from sb3-contrib.
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: type = nn.ReLU,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        *args,
        **kwargs,
    ):
        # Set default net_arch for policy/value heads after attention
        if net_arch is None:
            net_arch = [256, 128]
        
        # Store attention params
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=CardAttentionExtractor,
            features_extractor_kwargs={
                "embed_dim": embed_dim,
                "num_heads": num_heads,
                "num_layers": num_layers,
            },
            *args,
            **kwargs,
        )


# For sb3-contrib MaskablePPO compatibility
def create_attention_policy_kwargs(
    embed_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    net_arch: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Create policy_kwargs dict for MaskablePPO.
    
    Usage:
        from attention_policy import create_attention_policy_kwargs
        
        model = MaskablePPO(
            CardAttentionPolicy,
            env,
            policy_kwargs=create_attention_policy_kwargs(embed_dim=128),
            ...
        )
    """
    if net_arch is None:
        net_arch = dict(pi=[256, 128], vf=[256, 128])
    
    return {
        "features_extractor_class": CardAttentionExtractor,
        "features_extractor_kwargs": {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
        },
        "net_arch": net_arch,
    }


if __name__ == "__main__":
    # Test the policy
    import numpy as np
    
    # Create dummy observation space
    obs_size = GLOBAL_FEATURES + MAX_CARDS * FEATURES_PER_CARD
    print(f"Expected observation size: {obs_size}")
    
    observation_space = spaces.Box(
        low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
    )
    action_space = spaces.Discrete(100)
    
    # Create extractor
    extractor = CardAttentionExtractor(observation_space)
    print(f"Features dim: {extractor.features_dim}")
    
    # Test forward pass
    dummy_obs = torch.rand(4, obs_size)  # batch of 4
    features = extractor(dummy_obs)
    print(f"Output shape: {features.shape}")
    
    print("✓ CardAttentionExtractor works!")
