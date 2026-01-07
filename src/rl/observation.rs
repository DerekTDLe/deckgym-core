// Observation Tensor Generation - V2: Set-Based Card Encoding
//
// Encodes all cards in the game as a set with position features.
// Each card gets the same feature template regardless of its location.
//
// Order: Allied (board → hand → discard → deck) then Opponent (board → discard, hand/deck as counts)
// Total: 40 card slots × ~60 features + global features

use crate::card_ids::CardId;
use crate::hooks::{get_retreat_cost, get_stage};
use crate::models::{Card, EnergyType, PlayedCard};
use crate::state::State;
use crate::tool_ids::ToolId;

use super::ability_categories::{encode_ability_from_card_id, NUM_ABILITY_EFFECT_CATEGORIES};
use super::attack_categories::{encode_attack_effect_text, NUM_ATTACK_EFFECT_CATEGORIES};
use super::supporter_categories::{encode_supporter_categories, NUM_SUPPORTER_EFFECT_CATEGORIES};

// =============================================================================
// CONSTANTS
// =============================================================================

/// Number of energy types in the game for the cards
pub const NUM_ENERGY_TYPES: usize = 10;

/// Number of energy types in the game for the decks (Colorless and Dragon are excluded)
pub const NUM_ENERGY_TYPES_DECK: usize = 8;

/// Number of board slots per player (1 Active + 3 Bench)
pub const SLOTS_PER_PLAYER: usize = 4;

/// Number of distinct tool types for one-hot encoding
pub const NUM_TOOL_TYPES: usize = 8; // none + 7 tools

/// Maximum cards in observation (20 self + 4 opponent board = 24)
/// Self cards: full visibility (board, hand, discard, deck).
/// Opponent cards: only board visible (4 slots max). Hand/deck counts in global features.
pub const MAX_CARDS_IN_GAME: usize = 24;

// --- Normalization Constants ---

const MAX_TURNS: f32 = 30.0;
const MAX_POINTS: f32 = 3.0;
const MAX_DECK_SIZE: f32 = 20.0;
const MAX_DISCARD_SIZE: f32 = 20.0;
const MAX_HAND_SIZE: f32 = 10.0;
const MAX_HP: f32 = 250.0;
const MAX_ENERGY_ATTACHED: f32 = 5.0;
const MAX_ATTACK_DAMAGE: f32 = 200.0;
const MAX_ENERGY_COST: f32 = 4.0;
const MAX_RETREAT_COST: f32 = 4.0;

// --- Zone and Owner encoding ---

/// Zone one-hot indices
const ZONE_BOARD: usize = 0;
const ZONE_HAND: usize = 1;
const ZONE_DISCARD: usize = 2;
const ZONE_DECK: usize = 3;
const NUM_ZONES: usize = 4;

/// Number of board slots for position encoding (1 active + 3 bench)
const NUM_BOARD_SLOTS: usize = 4;

// --- Feature Dimensions ---

/// Card features (intrinsic properties)
pub const CARD_INTRINSIC_FEATURES: usize = 1  // is_pokemon (0=trainer, 1=pokemon)
    + 1                                        // stage (0/0.5/1.0)
    + NUM_ENERGY_TYPES                         // card energy type (one-hot)
    + 2                                        // hp_current, hp_max (normalized)
    + NUM_ENERGY_TYPES                         // weakness (one-hot)
    + 1                                        // ko_points (normalized)
    + NUM_ENERGY_TYPES                         // energy attached (per type)
    + 4                                        // attack1: damage, cost, has_effect, exists
    + NUM_ATTACK_EFFECT_CATEGORIES             // attack1 effect categories
    + 4                                        // attack2: damage, cost, has_effect, exists
    + NUM_ATTACK_EFFECT_CATEGORIES             // attack2 effect categories 
    + 4                                        // status: poison, sleep, para, confuse
    + NUM_ABILITY_EFFECT_CATEGORIES            // ability categories
    + NUM_SUPPORTER_EFFECT_CATEGORIES          // supporter categories (for trainers)
    + 1                                        // retreat cost
    + NUM_TOOL_TYPES;                          // attached tool (one-hot)

/// Position features (where the card is)
pub const CARD_POSITION_FEATURES: usize = NUM_ZONES  // zone one-hot (board/hand/discard/deck)
    + 1                                              // owner (0=self, 1=opponent)
    + 1                                              // is_active (1=active slot, 0=bench/other)
    + NUM_BOARD_SLOTS;                               // slot one-hot (if on board: 0=active, 1-3=bench)
    // NOTE: visibility removed - all encoded cards are now visible (24 cards only)

/// Total features per card slot
pub const FEATURES_PER_CARD: usize = CARD_INTRINSIC_FEATURES + CARD_POSITION_FEATURES;

/// Global features (game state)
pub const GLOBAL_FEATURES: usize = 1   // turn count
    + 2                                 // points (self, opponent)  
    + 2                                 // deck sizes (self, opponent)
    + 2                                 // hand sizes (self, opponent)
    + 2                                 // discard sizes (self, opponent)
    + 2 * 2 * NUM_ENERGY_TYPES_DECK;    // deck energy composition (2 players × 2 types max × 8 energies)

/// Total observation size
pub const OBSERVATION_SIZE: usize = GLOBAL_FEATURES + (MAX_CARDS_IN_GAME * FEATURES_PER_CARD);

// =============================================================================
// MAIN OBSERVATION FUNCTION
// =============================================================================

/// Converts the game state into a flat observation tensor.
pub fn get_observation_tensor(state: &State, perspective: usize) -> Vec<f32> {
    let mut obs = Vec::with_capacity(OBSERVATION_SIZE);
    let opponent = 1 - perspective;

    // --- Global Features ---
    encode_global_features(state, perspective, &mut obs);

    // --- All Cards (set-based encoding) ---
    let mut card_count = 0;

    // === SELF CARDS (full visibility) ===
    
    // Self board (Active + Bench)
    for slot in 0..SLOTS_PER_PLAYER {
        if let Some(pokemon) = &state.in_play_pokemon[perspective][slot] {
            encode_in_play_pokemon(pokemon, state, ZONE_BOARD, 0.0, slot, &mut obs);
            card_count += 1;
        }
    }

    // Self hand
    for card in &state.hands[perspective] {
        encode_card(card, None, state, ZONE_HAND, 0.0, None, &mut obs);
        card_count += 1;
    }

    // Self discard
    for card in &state.discard_piles[perspective] {
        encode_card(card, None, state, ZONE_DISCARD, 0.0, None, &mut obs);
        card_count += 1;
    }

    // Self deck (visible to self, composition known but not order)
    for card in &state.decks[perspective].cards {
        encode_card(card, None, state, ZONE_DECK, 0.0, None, &mut obs);
        card_count += 1;
    }

    // === OPPONENT CARDS (partial visibility) ===

    // Opponent board (fully visible)
    for slot in 0..SLOTS_PER_PLAYER {
        if let Some(pokemon) = &state.in_play_pokemon[opponent][slot] {
            encode_in_play_pokemon(pokemon, state, ZONE_BOARD, 1.0, slot, &mut obs);
            card_count += 1;
        }
    }

    // NOTE: Opponent discard, hand, and deck are NOT encoded as cards.
    // Their sizes are available in global features (lines 215-216).
    // This reduces observation from 40 to 24 cards for faster training.

    // === Pad remaining slots with zeros ===
    while card_count < MAX_CARDS_IN_GAME {
        encode_empty_slot(&mut obs);
        card_count += 1;
    }

    debug_assert_eq!(
        obs.len(),
        OBSERVATION_SIZE,
        "Observation size mismatch! Expected {}, got {}",
        OBSERVATION_SIZE,
        obs.len()
    );
    obs
}

// =============================================================================
// ENCODING FUNCTIONS
// =============================================================================

fn encode_global_features(state: &State, perspective: usize, obs: &mut Vec<f32>) {
    let opponent = 1 - perspective;

    // Turn count
    obs.push((state.turn_count as f32 / MAX_TURNS).clamp(0.0, 1.0));

    // Points
    obs.push(state.points[perspective] as f32 / MAX_POINTS);
    obs.push(state.points[opponent] as f32 / MAX_POINTS);

    // Deck sizes
    obs.push((state.decks[perspective].cards.len() as f32 / MAX_DECK_SIZE).clamp(0.0, 1.0));
    obs.push((state.decks[opponent].cards.len() as f32 / MAX_DECK_SIZE).clamp(0.0, 1.0));

    // Hand sizes
    obs.push((state.hands[perspective].len() as f32 / MAX_HAND_SIZE).clamp(0.0, 1.0));
    obs.push((state.hands[opponent].len() as f32 / MAX_HAND_SIZE).clamp(0.0, 1.0));

    // Discard sizes
    obs.push((state.discard_piles[perspective].len() as f32 / MAX_DISCARD_SIZE).clamp(0.0, 1.0));
    obs.push((state.discard_piles[opponent].len() as f32 / MAX_DISCARD_SIZE).clamp(0.0, 1.0));

    // Deck energy composition (self and opponent - visible info, supports dual type)
    for player in [perspective, opponent] {
        let deck_energies = &state.decks[player].energy_types;
        // Encode up to 2 energy types per deck (dual type support)
        for slot in 0..2 {
            for energy in deck_energy_types() {
                let has_energy = deck_energies.get(slot).map_or(false, |e| *e == energy);
                obs.push(if has_energy { 1.0 } else { 0.0 });
            }
        }
    }
}

/// Encode an InPlayPokemon (on board)
fn encode_in_play_pokemon(
    pokemon: &PlayedCard,
    state: &State,
    zone: usize,
    owner: f32,
    slot: usize,
    obs: &mut Vec<f32>,
) {
    // --- Intrinsic Features ---
    
    // is_pokemon
    obs.push(1.0);

    // Stage
    let stage = get_stage(pokemon);
    obs.push(stage as f32 / 2.0);

    // Card energy type (one-hot)
    let card_energy = pokemon.get_energy_type();
    for energy in all_energy_types() {
        obs.push(if card_energy == Some(energy) { 1.0 } else { 0.0 });
    }

    // HP
    obs.push(pokemon.remaining_hp as f32 / pokemon.total_hp as f32);
    obs.push((pokemon.total_hp as f32 / MAX_HP).clamp(0.0, 1.0));

    // Weakness
    let weakness = if let Card::Pokemon(p) = &pokemon.card {
        p.weakness
    } else {
        None
    };
    for energy in all_energy_types() {
        obs.push(if weakness == Some(energy) { 1.0 } else { 0.0 });
    }

    // KO points
    let ko_points = pokemon.card.get_knockout_points();
    obs.push(ko_points as f32 / MAX_POINTS);

    // Energy attached
    let energy_counts = count_energy_array(&pokemon.attached_energy);
    for (idx, _energy) in all_energy_types().iter().enumerate() {
        obs.push((energy_counts[idx] / MAX_ENERGY_ATTACHED).clamp(0.0, 1.0));
    }

    // Attacks
    let attacks = pokemon.get_attacks();
    if let Some(attack) = attacks.first() {
        encode_attack(attack, obs);
    } else {
        encode_no_attack(obs);
    }
    if let Some(attack) = attacks.get(1) {
        encode_attack(attack, obs);
    } else {
        encode_no_attack(obs);
    }

    // Status
    obs.push(if pokemon.poisoned { 1.0 } else { 0.0 });
    obs.push(if pokemon.asleep { 1.0 } else { 0.0 });
    obs.push(if pokemon.paralyzed { 1.0 } else { 0.0 });
    obs.push(if pokemon.confused { 1.0 } else { 0.0 });

    // Ability
    let ability_cats = encode_ability_from_card_id(&pokemon.card.get_id());
    obs.extend_from_slice(&ability_cats);

    // Supporter categories (not applicable for pokemon, zeros)
    for _ in 0..NUM_SUPPORTER_EFFECT_CATEGORIES {
        obs.push(0.0);
    }

    // Retreat cost
    obs.push((get_retreat_cost(state, pokemon).len() as f32 / MAX_RETREAT_COST).clamp(0.0, 1.0));

    // Tool
    encode_tool(&pokemon.attached_tool, obs);

    // --- Position Features ---
    encode_position(zone, owner, Some(slot), obs);
}

/// Encode a Card (in hand/discard/deck)
fn encode_card(
    card: &Card,
    _in_play: Option<&PlayedCard>,
    _state: &State,
    zone: usize,
    owner: f32,
    slot: Option<usize>,
    obs: &mut Vec<f32>,
) {
    match card {
        Card::Pokemon(pokemon_card) => {
            // is_pokemon
            obs.push(1.0);

            // Stage
            obs.push(pokemon_card.stage as f32 / 2.0);

            // Card energy type
            let card_energy = pokemon_card.energy_type;
            for energy in all_energy_types() {
                obs.push(if card_energy == energy { 1.0 } else { 0.0 });
            }

            // HP (max HP only for cards not in play)
            obs.push(1.0); // current HP = max (not in play)
            obs.push((pokemon_card.hp as f32 / MAX_HP).clamp(0.0, 1.0));

            // Weakness
            for energy in all_energy_types() {
                obs.push(if pokemon_card.weakness == Some(energy) { 1.0 } else { 0.0 });
            }

            // KO points
            obs.push(card.get_knockout_points() as f32 / MAX_POINTS);

            // Energy attached (none for cards not in play)
            for _ in 0..NUM_ENERGY_TYPES {
                obs.push(0.0);
            }

            // Attacks
            if let Some(attack) = pokemon_card.attacks.first() {
                encode_attack(attack, obs);
            } else {
                encode_no_attack(obs);
            }
            if let Some(attack) = pokemon_card.attacks.get(1) {
                encode_attack(attack, obs);
            } else {
                encode_no_attack(obs);
            }

            // Status (none for cards not in play)
            obs.extend_from_slice(&[0.0; 4]);

            // Ability
            let ability_cats = encode_ability_from_card_id(&pokemon_card.id);
            obs.extend_from_slice(&ability_cats);

            // Supporter categories (not applicable)
            for _ in 0..NUM_SUPPORTER_EFFECT_CATEGORIES {
                obs.push(0.0);
            }

            // Retreat cost
            obs.push((pokemon_card.retreat_cost.len() as f32 / MAX_RETREAT_COST).clamp(0.0, 1.0));

            // Tool (none for cards not in play)
            encode_tool(&None, obs);
        }
        Card::Trainer(trainer) => {
            // is_pokemon
            obs.push(0.0);

            // Stage (not applicable for trainers)
            obs.push(0.0);

            // Energy type (not applicable)
            for _ in 0..NUM_ENERGY_TYPES {
                obs.push(0.0);
            }

            // HP (not applicable)
            obs.push(0.0);
            obs.push(0.0);

            // Weakness (not applicable)
            for _ in 0..NUM_ENERGY_TYPES {
                obs.push(0.0);
            }

            // KO points (not applicable)
            obs.push(0.0);

            // Energy attached (not applicable)
            for _ in 0..NUM_ENERGY_TYPES {
                obs.push(0.0);
            }

            // Attacks (not applicable)
            encode_no_attack(obs);
            encode_no_attack(obs);

            // Status (not applicable)
            obs.extend_from_slice(&[0.0; 4]);

            // Ability (not applicable)
            obs.extend_from_slice(&[0.0; NUM_ABILITY_EFFECT_CATEGORIES]);

            // Supporter categories
            if let Some(card_id) = CardId::from_card_id(&trainer.id) {
                let cats = encode_supporter_categories(card_id);
                obs.extend_from_slice(&cats);
            } else {
                obs.extend_from_slice(&[0.0; NUM_SUPPORTER_EFFECT_CATEGORIES]);
            }

            // Retreat cost (not applicable)
            obs.push(0.0);

            // Tool (not applicable)
            encode_tool(&None, obs);
        }
    }

    // --- Position Features ---
    encode_position(zone, owner, slot, obs);
}

/// Encode an empty slot (padding)
fn encode_empty_slot(obs: &mut Vec<f32>) {
    for _ in 0..FEATURES_PER_CARD {
        obs.push(0.0);
    }
}

/// Encode position features (without visibility - all cards are visible now)
fn encode_position(zone: usize, owner: f32, slot: Option<usize>, obs: &mut Vec<f32>) {
    // Zone one-hot
    for i in 0..NUM_ZONES {
        obs.push(if i == zone { 1.0 } else { 0.0 });
    }

    // Owner
    obs.push(owner);

    // Is active (explicit feature for active slot)
    obs.push(if slot == Some(0) { 1.0 } else { 0.0 });

    // Slot one-hot (if on board: 0=active, 1-3=bench)
    for i in 0..NUM_BOARD_SLOTS {
        obs.push(if slot == Some(i) { 1.0 } else { 0.0 });
    }
}

/// Encode an attack
fn encode_attack(attack: &crate::models::Attack, obs: &mut Vec<f32>) {
    obs.push((attack.fixed_damage as f32 / MAX_ATTACK_DAMAGE).clamp(0.0, 1.0));
    obs.push((attack.energy_required.len() as f32 / MAX_ENERGY_COST).clamp(0.0, 1.0));
    obs.push(if attack.effect.is_some() { 1.0 } else { 0.0 });
    obs.push(1.0); // has_attack

    let effect_cats = encode_attack_effect_text(attack.effect.as_deref());
    obs.extend_from_slice(&effect_cats);
}

/// Encode absence of an attack (zeros)
fn encode_no_attack(obs: &mut Vec<f32>) {
    obs.extend_from_slice(&[0.0; 4 + NUM_ATTACK_EFFECT_CATEGORIES]);
}

/// Encode attached tool as one-hot vector
fn encode_tool(tool: &Option<ToolId>, obs: &mut Vec<f32>) {
    let tool_idx = match tool {
        None => 0,
        Some(ToolId::A2147GiantCape) => 1,
        Some(ToolId::A2148RockyHelmet) => 2,
        Some(ToolId::A3146PoisonBarb) => 3,
        Some(ToolId::A3147LeafCape) => 4,
        Some(ToolId::A3a065ElectricalCord)
        | Some(ToolId::A4b318ElectricalCord)
        | Some(ToolId::A4b319ElectricalCord) => 5,
        Some(ToolId::A4a067InflatableBoat) => 6,
        Some(ToolId::B1219HeavyHelmet) => 7,
    };

    for i in 0..NUM_TOOL_TYPES {
        obs.push(if i == tool_idx { 1.0 } else { 0.0 });
    }
}

// =============================================================================
// HELPERS
// =============================================================================

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

/// Energy types that can appear in decks (excludes Colorless and Dragon)
fn deck_energy_types() -> [EnergyType; NUM_ENERGY_TYPES_DECK] {
    [
        EnergyType::Grass,
        EnergyType::Fire,
        EnergyType::Water,
        EnergyType::Lightning,
        EnergyType::Psychic,
        EnergyType::Fighting,
        EnergyType::Darkness,
        EnergyType::Metal,
    ]
}

/// Count energy by type using a fixed-size array (faster than HashMap)
fn count_energy_array(energies: &[EnergyType]) -> [f32; NUM_ENERGY_TYPES] {
    let mut counts = [0.0f32; NUM_ENERGY_TYPES];
    for e in energies {
        let idx = energy_to_index(*e);
        counts[idx] += 1.0;
    }
    counts
}

/// Map energy type to array index
fn energy_to_index(e: EnergyType) -> usize {
    match e {
        EnergyType::Grass => 0,
        EnergyType::Fire => 1,
        EnergyType::Water => 2,
        EnergyType::Lightning => 3,
        EnergyType::Psychic => 4,
        EnergyType::Fighting => 5,
        EnergyType::Darkness => 6,
        EnergyType::Metal => 7,
        EnergyType::Dragon => 8,
        EnergyType::Colorless => 9,
    }
}

// =============================================================================
// TESTS
// =============================================================================

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
            assert!(val <= 1.0, "Value at index {} is > 1.0: {}", i, val);
        }
    }

    #[test]
    fn test_features_per_card_calculation() {
        // Verify our constant calculations are correct
        println!("CARD_INTRINSIC_FEATURES: {}", CARD_INTRINSIC_FEATURES);
        println!("CARD_POSITION_FEATURES: {}", CARD_POSITION_FEATURES);
        println!("FEATURES_PER_CARD: {}", FEATURES_PER_CARD);
        println!("GLOBAL_FEATURES: {}", GLOBAL_FEATURES);
        println!("OBSERVATION_SIZE: {}", OBSERVATION_SIZE);
        
        assert!(FEATURES_PER_CARD > 50, "Expected ~60 features per card");
        assert!(OBSERVATION_SIZE > 2000, "Expected ~2400 total features");
    }
}
