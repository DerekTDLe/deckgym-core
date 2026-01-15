// RL Module - Reinforcement Learning utilities for deckgym-core
//
// This module exposes game state as tensors for neural network consumption.

mod ability_categories;
mod action_mask;
mod attack_categories;
mod observation;
mod supporter_categories;

pub use ability_categories::{
    encode_ability_categories, encode_ability_from_card_id, get_ability_effect_categories,
    AbilityEffectCategory, NUM_ABILITY_EFFECT_CATEGORIES,
};
pub use action_mask::get_action_mask;
pub use action_mask::get_indexed_actions;
pub use action_mask::ACTION_SPACE_SIZE;
pub use attack_categories::{
    encode_attack_categories, encode_attack_effect_text, get_attack_effect_categories,
    AttackEffectCategory, NUM_ATTACK_EFFECT_CATEGORIES,
};
pub use observation::get_observation_tensor;
pub use observation::FEATURES_PER_CARD;
pub use observation::GLOBAL_FEATURES;
pub use observation::MAX_CARDS_IN_OBS;
pub use observation::OBSERVATION_SIZE;
pub use supporter_categories::{
    encode_supporter_categories, get_supporter_effect_categories, SupporterEffectCategory,
    NUM_SUPPORTER_EFFECT_CATEGORIES,
};
