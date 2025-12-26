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
    # GLOBAL_FEATURES = 1 (turn) + 2 (points) + 2 (deck_size) + 2 (hand_size) + 2 (discard_size) + 16 (2×8 deck_energy)
    GLOBAL_FEATURES = 25
    FEATURES_PER_CARD = 118  # intrinsic (108) + position (10) features per card
    MAX_CARDS = 24  # 20 self + 4 opponent board
    # Total observation size: 25 + 40 × 118 = 4745 dims


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
        
        # Self-attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation='relu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
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
        # A card is considered "present" if it has non-zero visibility feature
        # Visibility is the last position feature (index -1 in card features)
        visibility_idx = self.features_per_card - 1  # Last feature is visibility
        card_mask = card_feats[:, :, visibility_idx] < 0.5  # True = masked (padding)
        
        # Process global features
        global_out = self.global_proj(global_feats)
        
        # Embed cards
        card_embeds = self.card_embed(card_feats)  # [batch, max_cards, embed_dim]
        
        # Apply self-attention with mask
        # Note: Transformer expects mask where True = ignore
        attended = self.transformer(card_embeds, src_key_padding_mask=card_mask)
        
        # Mean pooling over non-masked cards
        mask_expanded = (~card_mask).unsqueeze(-1).float()  # [batch, max_cards, 1]
        num_cards = mask_expanded.sum(dim=1).clamp(min=1.0)  # [batch, 1]
        pooled = (attended * mask_expanded).sum(dim=1) / num_cards  # [batch, embed_dim]
        
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
