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
    # Use PyGame class which is exposed by the Rust bindings
    game_cls = getattr(deckgym, "PyGame", None) or getattr(deckgym, "Game", None)
    if game_cls is None:
        raise ImportError("Could not find Game or PyGame class in deckgym")
        
    GLOBAL_FEATURES = game_cls.global_features_size()
    FEATURES_PER_CARD = game_cls.features_per_card()
    # Calculate MAX_CARDS from total observation size
    # obs_size = global + max_cards * features
    MAX_CARDS = (game_cls.observation_size() - GLOBAL_FEATURES) // FEATURES_PER_CARD
except (ImportError, AttributeError):
    # Fallback for development - must match observation.rs constants
    # GLOBAL_FEATURES = 1 (turn) + 2 (points) + 2 (deck_size) + 2 (hand_size) + 2 (discard_size) + 32 (2×2×8 deck_energy with dual type)
    GLOBAL_FEATURES = 41
    FEATURES_PER_CARD = 116  # intrinsic (108) + position (8) features per card
    MAX_CARDS = 18  # 10 hand + 4 self board + 4 opponent board
    # Total observation size: 41 + 18 × 116 = 2129 dims


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
        
        # Fused QKV projection (FlashAttention/xFormers style)
        # Single linear layer computes Q, K, V simultaneously (3× faster)
        # This improves memory bandwidth and reduces kernel launches
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
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
        
        # Fused QKV projection: single matmul instead of 3
        qkv = self.qkv_proj(x)  # [batch, seq_len, 3 * embed_dim]
        
        # Split into Q, K, V
        qkv = qkv.view(batch_size, seq_len, 3, self.embed_dim)
        q, k, v = qkv.unbind(dim=2)  # Each: [batch, seq_len, embed_dim]
        
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

class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-head attention pooling using multiple learned query vectors.
    
    Instead of a single query (24× compression), we use multiple queries
    to capture different aspects of the card state (e.g., offensive cards,
    defensive cards, energy state, etc.).
    
    This reduces information loss and provides richer features for downstream tasks.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4, num_queries: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Multiple learned query vectors (like multiple [CLS] tokens)
        # Each query can specialize in capturing different aspects
        self.queries = nn.Parameter(torch.zeros(num_queries, 1, embed_dim))
        nn.init.xavier_uniform_(self.queries)  # Better gradient flow than normal(0.02)
        
        # Projections for cross-attention
        # Q is separate (from learned queries), but K and V are fused for efficiency
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=False)  # Fused K, V
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    
    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim] - card embeddings after self-attention
            key_padding_mask: [batch, seq_len] where True = masked (empty card slot)
        
        Returns:
            [batch, num_queries * embed_dim] - multiple pooled representations
        """
        batch_size, seq_len, _ = x.shape
        
        # Expand learned queries for batch: [batch, num_queries, embed_dim]
        # queries shape: [num_queries, 1, embed_dim]
        # We need to expand the middle dimension (1) to batch_size
        queries = self.queries.transpose(0, 1).expand(batch_size, -1, -1)  # [batch, num_queries, embed_dim]
        
        # Project queries
        q = self.q_proj(queries)  # [batch, num_queries, embed_dim]
        
        # Fused KV projection
        kv = self.kv_proj(x)  # [batch, seq_len, 2 * embed_dim]
        kv = kv.view(batch_size, seq_len, 2, self.embed_dim)
        k, v = kv.unbind(dim=2)  # Each: [batch, seq_len, embed_dim]
        
        # Reshape for multi-head: [batch, num_heads, *, head_dim]
        q = q.view(batch_size, self.num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores: [batch, num_heads, num_queries, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask for empty card slots
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # Weighted combination: [batch, num_heads, num_queries, head_dim]
        pooled = torch.matmul(attn_probs, v)
        
        # Reshape: [batch, num_queries, embed_dim]
        pooled = pooled.transpose(1, 2).contiguous().view(batch_size, self.num_queries, self.embed_dim)
        
        # Apply output projection to each query
        pooled = self.out_proj(pooled)  # [batch, num_queries, embed_dim]
        
        # Flatten to single vector: [batch, num_queries * embed_dim]
        return pooled.view(batch_size, -1)

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
        # Expanded from 320 to 640 to provide richer features for action net
        # global (128) + cards (512) = 640
        features_dim = 128 + embed_dim * 2  # Attended cards + processed global
        super().__init__(observation_space, features_dim)
        
        self.global_features = global_features
        self.features_per_card = features_per_card
        self.max_cards = max_cards
        self.embed_dim = embed_dim
        
        # Project global features (GELU activation for smoother gradients)
        # Expanded from 64 to 128 dims for richer global representation
        self.global_proj = nn.Sequential(
            nn.Linear(global_features, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )
        
        # Project card features to embedding dimension (GELU activation)
        self.card_embed = nn.Sequential(
            nn.Linear(features_per_card, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Manual attention blocks for ONNX compatibility
        # Uses OnnxSafeAttention instead of PyTorch's MultiheadAttention
        self.attention_layers = nn.ModuleList([
            OnnxSafeAttention(embed_dim, num_heads, dropout=0.15)
            for _ in range(num_layers)
        ])
        # Feed-forward layers with modern Transformer improvements:
        # - 4× expansion (standard in GPT/BERT/LLaMA)
        # - GELU activation (smoother than ReLU)
        # - Dropout for regularization
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(0.15),
            )
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers * 2)
        ])
        self.num_layers = num_layers
        
        # Multi-head pooling with 4 queries (reduces compression from 24× to 6×)
        # Output: [batch, 4 * embed_dim] = [batch, 1024] for embed_dim=256
        self.attention_pool = MultiHeadAttentionPooling(embed_dim, num_heads, num_queries=4)
        
        # Final projection after pooling (GELU activation)
        # Input: 4 * embed_dim (1024 for embed_dim=256)
        # Output: 2 * embed_dim (512 for embed_dim=256)
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.GELU(),
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
        
        # Apply attention layers with PRE-NORM (modern standard)
        # Pre-norm: normalize BEFORE attention, then add clean residual
        # This gives much better gradient flow than post-norm
        for i in range(self.num_layers):
            # Self-attention with pre-norm
            normed = self.layer_norms[i * 2](x)
            attn_out = self.attention_layers[i](normed, key_padding_mask=card_mask)
            x = x + attn_out  # Clean residual connection
            
            # Feed-forward with pre-norm
            normed = self.layer_norms[i * 2 + 1](x)
            ff_out = self.ff_layers[i](normed)
            x = x + ff_out  # Clean residual connection
        
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
    ortho_init: bool = False,  # Disabled by default - SB3's gain=0.01 is too conservative
) -> Dict[str, Any]:
    """
    Create policy_kwargs dict for MaskablePPO.
    
    Args:
        embed_dim: Embedding dimension for attention layers
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        net_arch: Network architecture for policy/value nets
        ortho_init: Whether to use orthogonal initialization (default: False)
                   SB3's ortho_init uses gain=0.01 for action_net, which is 
                   100× smaller than normal and causes gradient imbalance.
                   Setting to False uses PyTorch's default Kaiming uniform.
    
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
        "ortho_init": ortho_init,  # Disable SB3's conservative init
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
