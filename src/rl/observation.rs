// Observation Tensor Generation
//
// Converts game State into a flat f32 vector for neural network input.
//
// This version includes feature-based card representation for generalization:
// - Pokemon cards encoded by their stats (stage, HP, energy type, attack damage, effect categories)
// - Trainer cards in hand encoded by their effect categories
// - No card ID lookup - the agent learns from card properties, not identity

use crate::card_ids::CardId;
use crate::hooks::{get_retreat_cost, get_stage};
use crate::models::{Card, EnergyType};
use crate::state::State;

use super::attack_categories::{encode_attack_effect_text, NUM_ATTACK_EFFECT_CATEGORIES};
use super::supporter_categories::{encode_supporter_categories, NUM_SUPPORTER_EFFECT_CATEGORIES};

// --- Constants ---

/// Number of energy types in the game
pub const NUM_ENERGY_TYPES: usize = 10;

/// Number of board slots per player (1 Active + 3 Bench)
pub const SLOTS_PER_PLAYER: usize = 4;

/// Features per board slot (enhanced with effect categories)
pub const FEATURES_PER_SLOT: usize = 1                                     // Stage (0=Basic, 0.5=Stage1, 1.0=Stage2)
    + 2                                   // HP current ratio, HP max normalized
    + NUM_ENERGY_TYPES                    // Card's energy type (one-hot)
    + NUM_ENERGY_TYPES                    // Energy attached (count per type)
    + 1                                   // Energy delta (missing energy for strongest attack)
    + 4                                   // Attack 1: damage, cost, has_effect, effect categories...
    + NUM_ATTACK_EFFECT_CATEGORIES        // Attack 1 effect categories
    + 4                                   // Attack 2: damage, cost, has_effect, has_attack
    + NUM_ATTACK_EFFECT_CATEGORIES        // Attack 2 effect categories
    + 4                                   // Status: Poison, Sleep, Paralyze, Confusion
    + 1                                   // Has ability
    + 1; // Retreat cost

/// Hand features per player
pub const HAND_FEATURES_PER_PLAYER: usize = 3                                     // Pokemon counts: Basic, Stage1, Stage2
    + NUM_SUPPORTER_EFFECT_CATEGORIES     // Aggregated trainer effect categories
    + 1                                   // Can evolve active
    + 1; // Total playable cards count

/// Global features
pub const GLOBAL_FEATURES: usize = 1                                     // Turn count (normalized)
    + 2                                   // Points (self, opponent)
    + 4                                   // Deck sizes, Discard sizes
    + NUM_ENERGY_TYPES; // Deck energy composition

/// Board features (all slots for both players)
pub const BOARD_FEATURES: usize = 2 * SLOTS_PER_PLAYER * FEATURES_PER_SLOT;

/// Hand features for both players
pub const HAND_FEATURES: usize = 2 * HAND_FEATURES_PER_PLAYER;

/// Total observation size
pub const OBSERVATION_SIZE: usize = GLOBAL_FEATURES + BOARD_FEATURES + HAND_FEATURES;

// --- Main Function ---

/// Converts the game state into a flat observation tensor.
pub fn get_observation_tensor(state: &State, perspective: usize) -> Vec<f32> {
    let mut obs = Vec::with_capacity(OBSERVATION_SIZE);
    let opponent = 1 - perspective;

    // --- Global Features ---
    encode_global_features(state, perspective, &mut obs);

    // --- Board Features ---
    encode_board_state(state, perspective, &mut obs);
    encode_board_state(state, opponent, &mut obs);

    // --- Hand Features ---
    encode_hand_features(state, perspective, &mut obs);
    encode_hand_features(state, opponent, &mut obs);

    debug_assert_eq!(
        obs.len(),
        OBSERVATION_SIZE,
        "Observation size mismatch! Expected {}, got {}",
        OBSERVATION_SIZE,
        obs.len()
    );
    obs
}

// --- Helper Functions ---

fn encode_global_features(state: &State, perspective: usize, obs: &mut Vec<f32>) {
    let opponent = 1 - perspective;

    // Turn count (normalized, clamped)
    obs.push((state.turn_count as f32 / 30.0).clamp(0.0, 1.0));

    // Points (max 3 to win)
    obs.push(state.points[perspective] as f32 / 3.0);
    obs.push(state.points[opponent] as f32 / 3.0);

    // Deck sizes (normalized)
    obs.push((state.decks[perspective].cards.len() as f32 / 15.0).clamp(0.0, 1.0));
    obs.push((state.decks[opponent].cards.len() as f32 / 15.0).clamp(0.0, 1.0));

    // Discard pile sizes
    obs.push((state.discard_piles[perspective].len() as f32 / 20.0).clamp(0.0, 1.0));
    obs.push((state.discard_piles[opponent].len() as f32 / 20.0).clamp(0.0, 1.0));

    // Deck energy composition (multi-hot)
    let deck_energies = &state.decks[perspective].energy_types;
    for energy in all_energy_types() {
        obs.push(if deck_energies.contains(&energy) {
            1.0
        } else {
            0.0
        });
    }
}

fn encode_board_state(state: &State, player: usize, obs: &mut Vec<f32>) {
    for slot in 0..SLOTS_PER_PLAYER {
        if let Some(pokemon) = &state.in_play_pokemon[player][slot] {
            // Stage (0=Basic, 0.5=Stage1, 1.0=Stage2)
            let stage = get_stage(pokemon);
            obs.push(stage as f32 / 2.0);

            // HP
            obs.push(pokemon.remaining_hp as f32 / pokemon.total_hp as f32);
            obs.push((pokemon.total_hp as f32 / 250.0).clamp(0.0, 1.0));

            // Card's energy type (one-hot)
            let card_energy = pokemon.get_energy_type();
            for energy in all_energy_types() {
                obs.push(if card_energy == Some(energy) {
                    1.0
                } else {
                    0.0
                });
            }

            // Energy attached (count per type)
            let energy_counts = count_energy(&pokemon.attached_energy);
            for energy in all_energy_types() {
                obs.push(*energy_counts.get(&energy).unwrap_or(&0) as f32);
            }

            // Energy delta for strongest attack
            let attacks = pokemon.get_attacks();
            let strongest_attack = if attacks.len() > 1 {
                attacks.get(1)
            } else {
                attacks.first()
            };
            let missing_energy = if let Some(attack) = strongest_attack {
                calculate_missing_energy(&pokemon.attached_energy, &attack.energy_required)
            } else {
                0
            };
            obs.push((missing_energy as f32 / 4.0).clamp(0.0, 1.0));

            // Attack 1 features
            if let Some(attack) = attacks.first() {
                obs.push((attack.fixed_damage as f32 / 200.0).clamp(0.0, 1.0));
                obs.push((attack.energy_required.len() as f32 / 5.0).clamp(0.0, 1.0));
                obs.push(if attack.effect.is_some() { 1.0 } else { 0.0 });
                obs.push(1.0); // has_attack

                // Effect categories
                let effect_cats = encode_attack_effect_text(attack.effect.as_deref());
                obs.extend_from_slice(&effect_cats);
            } else {
                // No attack 1
                obs.extend_from_slice(&[0.0; 4 + NUM_ATTACK_EFFECT_CATEGORIES]);
            }

            // Attack 2 features
            if let Some(attack) = attacks.get(1) {
                obs.push((attack.fixed_damage as f32 / 200.0).clamp(0.0, 1.0));
                obs.push((attack.energy_required.len() as f32 / 5.0).clamp(0.0, 1.0));
                obs.push(if attack.effect.is_some() { 1.0 } else { 0.0 });
                obs.push(1.0); // has_attack

                let effect_cats = encode_attack_effect_text(attack.effect.as_deref());
                obs.extend_from_slice(&effect_cats);
            } else {
                // No attack 2
                obs.extend_from_slice(&[0.0; 4 + NUM_ATTACK_EFFECT_CATEGORIES]);
            }

            // Status conditions
            obs.push(if pokemon.poisoned { 1.0 } else { 0.0 });
            obs.push(if pokemon.asleep { 1.0 } else { 0.0 });
            obs.push(if pokemon.paralyzed { 1.0 } else { 0.0 });
            obs.push(if pokemon.confused { 1.0 } else { 0.0 });

            // Has ability
            obs.push(if pokemon.card.get_ability().is_some() {
                1.0
            } else {
                0.0
            });

            // Retreat cost
            obs.push((get_retreat_cost(state, pokemon).len() as f32 / 4.0).clamp(0.0, 1.0));
        } else {
            // Empty slot
            for _ in 0..FEATURES_PER_SLOT {
                obs.push(0.0);
            }
        }
    }
}

fn encode_hand_features(state: &State, player: usize, obs: &mut Vec<f32>) {
    let hand = &state.hands[player];

    // Count Pokemon by stage
    let mut basic_count = 0;
    let mut stage1_count = 0;
    let mut stage2_count = 0;

    // Aggregate trainer effect categories
    let mut trainer_cats = [0.0_f32; NUM_SUPPORTER_EFFECT_CATEGORIES];

    for card in hand {
        match card {
            Card::Pokemon(pokemon) => match pokemon.stage {
                0 => basic_count += 1,
                1 => stage1_count += 1,
                _ => stage2_count += 1,
            },
            Card::Trainer(trainer) => {
                // Aggregate trainer effect categories
                if let Some(card_id) = CardId::from_card_id(&trainer.id) {
                    let cats = encode_supporter_categories(card_id);
                    for (i, &cat) in cats.iter().enumerate() {
                        trainer_cats[i] += cat;
                    }
                }
            }
        }
    }

    // Pokemon counts (normalized)
    obs.push((basic_count as f32 / 5.0).clamp(0.0, 1.0));
    obs.push((stage1_count as f32 / 3.0).clamp(0.0, 1.0));
    obs.push((stage2_count as f32 / 2.0).clamp(0.0, 1.0));

    // Trainer effect categories (normalized by count)
    for cat in trainer_cats {
        obs.push((cat / 3.0).clamp(0.0, 1.0));
    }

    // Can evolve active
    let can_evolve = can_evolve_active(state, player);
    obs.push(if can_evolve { 1.0 } else { 0.0 });

    // Total playable cards
    let playable_count = basic_count
        + stage1_count
        + stage2_count
        + trainer_cats.iter().filter(|&&c| c > 0.0).count();
    obs.push((playable_count as f32 / 10.0).clamp(0.0, 1.0));
}

fn can_evolve_active(state: &State, player: usize) -> bool {
    if let Some(active) = &state.in_play_pokemon[player][0] {
        let active_name = active.get_name();
        for card in &state.hands[player] {
            if let Card::Pokemon(pokemon) = card {
                if let Some(ref evolves_from) = pokemon.evolves_from {
                    if evolves_from == &active_name {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn calculate_missing_energy(attached: &[EnergyType], required: &[EnergyType]) -> usize {
    let mut attached_counts = count_energy(attached);
    let mut missing = 0;
    let mut colorless_needed = 0;

    for energy in required {
        if *energy == EnergyType::Colorless {
            colorless_needed += 1;
        } else {
            let available = attached_counts.entry(*energy).or_insert(0);
            if *available > 0 {
                *available -= 1;
            } else {
                missing += 1;
            }
        }
    }

    let total_remaining: usize = attached_counts.values().sum();
    if total_remaining < colorless_needed {
        missing += colorless_needed - total_remaining;
    }

    missing
}

fn all_energy_types() -> [EnergyType; NUM_ENERGY_TYPES] {
    [
        EnergyType::Grass,
        EnergyType::Fire,
        EnergyType::Water,
        EnergyType::Lightning,
        EnergyType::Psychic,
        EnergyType::Fighting,
        EnergyType::Darkness,
        EnergyType::Metal,
        EnergyType::Dragon,
        EnergyType::Colorless,
    ]
}

fn count_energy(energies: &[EnergyType]) -> std::collections::HashMap<EnergyType, usize> {
    let mut counts = std::collections::HashMap::new();
    for e in energies {
        *counts.entry(*e).or_insert(0) += 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observation_size() {
        let state = State::default();
        let obs = get_observation_tensor(&state, 0);
        assert_eq!(obs.len(), OBSERVATION_SIZE);
    }

    #[test]
    fn test_observation_values_in_range() {
        let state = State::default();
        let obs = get_observation_tensor(&state, 0);
        for (i, &val) in obs.iter().enumerate() {
            assert!(val >= 0.0, "Value at index {} is negative: {}", i, val);
            // Most values should be <= 1.0, but energy counts can be higher
        }
    }
}
