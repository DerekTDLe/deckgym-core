// RL Module - Reinforcement Learning utilities for deckgym-core
//
// This module exposes game state as tensors for neural network consumption.

mod observation;
mod action_mask;
mod supporter_categories;
mod attack_categories;

pub use observation::get_observation_tensor;
pub use observation::OBSERVATION_SIZE;
pub use action_mask::get_action_mask;
pub use action_mask::get_indexed_actions;
pub use action_mask::ACTION_SPACE_SIZE;
pub use supporter_categories::{
    SupporterEffectCategory, 
    get_supporter_effect_categories, 
    encode_supporter_categories,
    NUM_SUPPORTER_EFFECT_CATEGORIES,
};
pub use attack_categories::{
    AttackEffectCategory,
    get_attack_effect_categories,
    encode_attack_categories,
    encode_attack_effect_text,
    NUM_ATTACK_EFFECT_CATEGORIES,
};

