// Observation Tensor Generation
//
// Converts game State into a flat f32 vector for neural network input.

use crate::models::EnergyType;
use crate::state::State;

// --- Constants ---
// These define the structure of our observation tensor.

/// Number of energy types in the game
pub const NUM_ENERGY_TYPES: usize = 10; // Grass, Fire, Water, Lightning, Psychic, Fighting, Darkness, Metal, Dragon, Colorless

/// Number of board slots per player (1 Active + 3 Bench)
pub const SLOTS_PER_PLAYER: usize = 4;

/// Features per board slot
pub const FEATURES_PER_SLOT: usize = 2  // HP current, HP max (normalized)
    + NUM_ENERGY_TYPES                   // Energy attached (count per type)
    + 1                                  // Energy delta (attached - cost of strongest attack)
    + 4                                  // Status: Poison, Sleep, Paralyze, Confusion
    + 1; // Card ID index (for embedding lookup)

/// Global features
pub const GLOBAL_FEATURES: usize = 1   // Turn count (normalized)
    + 2                                 // Points (self, opponent)
    + 4                                 // Deck sizes (self, opponent), Discard sizes (self, opponent)
    + 2                                 // Hand sizes (self, opponent)
    + NUM_ENERGY_TYPES; // Deck energy composition (one-hot)

/// Board features (all slots for both players)
pub const BOARD_FEATURES: usize = 2 * SLOTS_PER_PLAYER * FEATURES_PER_SLOT;

// TODO: Add hand/discard multi-hot encoding later
// For now, we'll use a simpler representation

/// Total observation size
pub const OBSERVATION_SIZE: usize = GLOBAL_FEATURES + BOARD_FEATURES;

// --- Main Function ---

/// Converts the game state into a flat observation tensor.
///
/// The tensor is structured as:
/// [Global Features | Player 0 Board | Player 1 Board]
///
/// # Arguments
/// * `state` - The current game state
/// * `perspective` - Which player's perspective (0 or 1). The observation is always
///                   from the perspective of "self" vs "opponent".
///
/// # Returns
/// A Vec<f32> of size OBSERVATION_SIZE
pub fn get_observation_tensor(state: &State, perspective: usize) -> Vec<f32> {
    let mut obs = Vec::with_capacity(OBSERVATION_SIZE);

    // --- Global Features ---
    encode_global_features(state, perspective, &mut obs);

    // --- Board Features ---
    // Self's board first, then opponent's
    let opponent = 1 - perspective;
    encode_board_state(state, perspective, &mut obs);
    encode_board_state(state, opponent, &mut obs);

    debug_assert_eq!(obs.len(), OBSERVATION_SIZE, "Observation size mismatch!");
    obs
}

// --- Helper Functions ---

fn encode_global_features(state: &State, perspective: usize, obs: &mut Vec<f32>) {
    let opponent = 1 - perspective;

    // Turn count (normalized to 0-1, clamped for long games)
    obs.push((state.turn_count as f32 / 30.0).clamp(0.0, 1.0));

    // Points (max 3 to win)
    obs.push(state.points[perspective] as f32 / 3.0);
    obs.push(state.points[opponent] as f32 / 3.0);

    // Deck sizes (normalized to 15, clamped for rare edge cases)
    obs.push((state.decks[perspective].cards.len() as f32 / 15.0).clamp(0.0, 1.0));
    obs.push((state.decks[opponent].cards.len() as f32 / 15.0).clamp(0.0, 1.0));

    // Discard pile sizes (max = 20, the full deck)
    obs.push(state.discard_piles[perspective].len() as f32 / 20.0);
    obs.push(state.discard_piles[opponent].len() as f32 / 20.0);

    // Hand sizes (hardcoded max = 10, no clamp needed)
    obs.push(state.hands[perspective].len() as f32 / 10.0);
    obs.push(state.hands[opponent].len() as f32 / 10.0);

    // Deck energy composition (multi-hot: can have up to 3 energy types)
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
            // HP (normalized)
            obs.push(pokemon.remaining_hp as f32 / pokemon.total_hp as f32);
            obs.push(pokemon.total_hp as f32 / 200.0); // Normalize to typical max HP

            // Energy attached (count per type)
            let energy_counts = count_energy(&pokemon.attached_energy);
            for energy in all_energy_types() {
                obs.push(*energy_counts.get(&energy).unwrap_or(&0) as f32);
            }

            // Energy delta: missing energy to use strongest attack (Attack 2 if exists, else Attack 1)
            let attacks = pokemon.get_attacks();
            let strongest_attack = if attacks.len() > 1 {
                attacks.get(1) // Attack 2 is always stronger
            } else {
                attacks.first()
            };
            
            let missing_energy = if let Some(attack) = strongest_attack {
                calculate_missing_energy(&pokemon.attached_energy, &attack.energy_required)
            } else {
                0 // No attacks (shouldn't happen for Pokemon)
            };
            // Normalize: 0 = ready, 4+ = far from ready
            obs.push((missing_energy as f32 / 4.0).clamp(0.0, 1.0));

            // Status conditions
            obs.push(if pokemon.poisoned { 1.0 } else { 0.0 });
            obs.push(if pokemon.asleep { 1.0 } else { 0.0 });
            obs.push(if pokemon.paralyzed { 1.0 } else { 0.0 });
            obs.push(if pokemon.confused { 1.0 } else { 0.0 });

            // Card ID index (placeholder - will implement canonical mapping later)
            obs.push(0.0); // TODO: Map card ID to canonical index
        } else {
            // Empty slot - push zeros
            let slot_size = FEATURES_PER_SLOT;
            for _ in 0..slot_size {
                obs.push(0.0);
            }
        }
    }
}

/// Calculate how many energy are missing to pay for an attack cost.
/// Handles Colorless energy (can be paid by any type) properly.
fn calculate_missing_energy(attached: &[EnergyType], required: &[EnergyType]) -> usize {
    let mut attached_counts = count_energy(attached);
    let mut missing = 0;
    let mut colorless_needed = 0;

    // First pass: count specific energy requirements
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

    // Second pass: colorless can be paid by any remaining energy
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
        // Create a default state and verify observation size
        let state = State::default();
        let obs = get_observation_tensor(&state, 0);
        assert_eq!(obs.len(), OBSERVATION_SIZE);
    }
}
