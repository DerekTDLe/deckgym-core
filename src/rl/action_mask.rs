// Action Mask Generation
//
// Generates a boolean mask indicating which actions are valid for the current state.
// Maps ALL SimpleAction variants to fixed indices for RL compatibility.

use crate::actions::{Action, SimpleAction};
use crate::move_generation::generate_possible_actions;
use crate::state::State;

// --- Action Space Layout ---
// 
// Total size: 120 actions
//
// Range       | Action Type                      | Count
// ------------|----------------------------------|-------
// 0           | EndTurn                          | 1
// 1-2         | Attack(0), Attack(1)             | 2
// 3-5         | Retreat to bench 1, 2, 3         | 3
// 6-9         | Attach energy to pos 0-3         | 4
// 10-19       | Place card from hand (slot 0-9)  | 10
// 20-29       | Evolve card from hand (slot 0-9) | 10
// 30-39       | Play trainer from hand (slot 0-9)| 10
// 40-43       | UseAbility (position 0-3)        | 4
// 44          | DrawCard                         | 1
// 45          | Noop (say "no" to prompt)        | 1
// 46-49       | Activate own bench pos 1-3 (+1)  | 4
// 50-53       | AttachTool to position 0-3       | 4
// 54-57       | Heal position 0-3                | 4
// 58-61       | DiscardFossil position 0-3       | 4
// 62-65       | AttachFromDiscard pos 0-3        | 4
// 66-75       | CommunicatePokemon hand 0-9      | 10
// 76-85       | ShufflePokemonIntoDeck hand 0-9  | 10
// 86-95       | ShuffleOpponentSupporter 0-9     | 10
// 96-105      | DiscardOpponentSupporter 0-9     | 10
// 106-115     | DiscardOwnCard hand 0-9          | 10
// 116         | ApplyEeveeBagDamageBoost         | 1
// 117         | HealAllEeveeEvolutions           | 1
// 118         | Non-turn energy attach (rare)    | 1
// 119         | Reserved                         | 1

pub const ACTION_SPACE_SIZE: usize = 120;

/// Converts a SimpleAction to a canonical action index.
/// Returns None if the action doesn't map to a fixed index (shouldn't happen).
fn action_to_index(action: &SimpleAction, hand_index_map: &std::collections::HashMap<String, usize>) -> Option<usize> {
    match action {
        // Core actions
        SimpleAction::EndTurn => Some(0),
        SimpleAction::Attack(attack_idx) => Some(1 + attack_idx.min(&1)), // 1-2
        SimpleAction::Retreat(bench_idx) => {
            if *bench_idx >= 1 && *bench_idx <= 3 {
                Some(3 + bench_idx - 1) // 3-5
            } else {
                None
            }
        }
        
        // Turn energy attachment (once per turn)
        SimpleAction::Attach { attachments, is_turn_energy: true } => {
            if let Some((_, _, in_play_idx)) = attachments.first() {
                if *in_play_idx <= 3 {
                    Some(6 + in_play_idx) // 6-9
                } else {
                    None
                }
            } else {
                None
            }
        }
        
        // Non-turn energy attach (from effects like Lusamine)
        SimpleAction::Attach { is_turn_energy: false, .. } => Some(118),
        
        // Hand card actions - Place Pokémon
        SimpleAction::Place(card, _) => {
            hand_index_map.get(&card.get_id()).and_then(|&idx| {
                if idx < 10 { Some(10 + idx) } else { None }
            })
        }
        
        // Hand card actions - Evolve
        SimpleAction::Evolve(card, _) => {
            hand_index_map.get(&card.get_id()).and_then(|&idx| {
                if idx < 10 { Some(20 + idx) } else { None }
            })
        }
        
        // Hand card actions - Play trainer
        SimpleAction::Play { trainer_card } => {
            hand_index_map.get(&trainer_card.id).and_then(|&idx| {
                if idx < 10 { Some(30 + idx) } else { None }
            })
        }
        
        // Ability usage
        SimpleAction::UseAbility { in_play_idx } => {
            if *in_play_idx <= 3 {
                Some(40 + in_play_idx) // 40-43
            } else {
                None
            }
        }
        
        // Draw card (beginning of turn)
        SimpleAction::DrawCard { .. } => Some(44),
        
        // Noop (decline an optional action)
        SimpleAction::Noop => Some(45),
        
        // Activate - bring bench Pokémon to active
        SimpleAction::Activate { in_play_idx, .. } => {
            if *in_play_idx >= 1 && *in_play_idx <= 3 {
                Some(46 + in_play_idx - 1) // 46-48
            } else if *in_play_idx == 0 {
                Some(49) // Special case: already active (shouldn't happen often)
            } else {
                None
            }
        }
        
        // Attach tool
        SimpleAction::AttachTool { in_play_idx, .. } => {
            if *in_play_idx <= 3 {
                Some(50 + in_play_idx) // 50-53
            } else {
                None
            }
        }
        
        // Heal
        SimpleAction::Heal { in_play_idx, .. } => {
            if *in_play_idx <= 3 {
                Some(54 + in_play_idx) // 54-57
            } else {
                None
            }
        }
        
        // Discard fossil
        SimpleAction::DiscardFossil { in_play_idx } => {
            if *in_play_idx <= 3 {
                Some(58 + in_play_idx) // 58-61
            } else {
                None
            }
        }
        
        // Attach from discard (Lusamine effect)
        SimpleAction::AttachFromDiscard { in_play_idx, .. } => {
            if *in_play_idx <= 3 {
                Some(62 + in_play_idx) // 62-65
            } else {
                None
            }
        }
        
        // Communicate Pokémon (swap hand card with deck)
        SimpleAction::CommunicatePokemon { hand_pokemon } => {
            hand_index_map.get(&hand_pokemon.get_id()).and_then(|&idx| {
                if idx < 10 { Some(66 + idx) } else { None }
            })
        }
        
        // Shuffle Pokémon into deck (May effect)
        SimpleAction::ShufflePokemonIntoDeck { hand_pokemon } => {
            // Use first card's index for simplicity
            if let Some(first_card) = hand_pokemon.first() {
                hand_index_map.get(&first_card.get_id()).and_then(|&idx| {
                    if idx < 10 { Some(76 + idx) } else { None }
                })
            } else {
                None
            }
        }
        
        // Shuffle opponent's supporter (Silver effect)
        // Note: We don't have opponent hand map, use card position heuristic
        SimpleAction::ShuffleOpponentSupporter { .. } => {
            // For opponent cards, we use a simplified mapping
            // Since we can't track opponent hand indices easily,
            // we'll map based on the action itself being valid
            Some(86) // Use first slot - actual choice is made by the simulator
        }
        
        // Discard opponent's supporter (Mega Absol effect)
        SimpleAction::DiscardOpponentSupporter { .. } => {
            Some(96)
        }
        
        // Discard own card (Sableye's Dirty Throw)
        SimpleAction::DiscardOwnCard { card } => {
            hand_index_map.get(&card.get_id()).and_then(|&idx| {
                if idx < 10 { Some(106 + idx) } else { None }
            })
        }
        
        // Eevee Bag options
        SimpleAction::ApplyEeveeBagDamageBoost => Some(116),
        SimpleAction::HealAllEeveeEvolutions => Some(117),
        
        // Complex actions that need special handling
        SimpleAction::MoveEnergy { .. } => {
            // MoveEnergy is rare - map to a single slot
            // The actual energy/position choice is handled by the simulator
            Some(118) // Shares with non-turn energy
        }
        
        SimpleAction::MoveAllDamage { .. } => {
            // Also rare, use reserved slot
            Some(119)
        }
        
        // ApplyDamage is internal, not a player choice
        SimpleAction::ApplyDamage { .. } => None,
    }
}

/// Generates a boolean action mask for the current state.
/// 
/// # Arguments
/// * `state` - The current game state
/// 
/// # Returns
/// A Vec<bool> of size ACTION_SPACE_SIZE where true = valid action
pub fn get_action_mask(state: &State) -> Vec<bool> {
    let mut mask = vec![false; ACTION_SPACE_SIZE];
    
    let (actor, valid_actions) = generate_possible_actions(state);
    
    // Build hand index map for this state
    let mut hand_index_map = std::collections::HashMap::new();
    for (idx, card) in state.hands[actor].iter().enumerate() {
        hand_index_map.insert(card.get_id(), idx);
    }
    
    // Mark valid actions in the mask
    for action in &valid_actions {
        if let Some(index) = action_to_index(&action.action, &hand_index_map) {
            if index < ACTION_SPACE_SIZE {
                mask[index] = true;
            }
        }
    }
    
    mask
}

/// Get the list of valid actions with their indices.
/// Useful for debugging and for stepping by index.
pub fn get_indexed_actions(state: &State) -> Vec<(usize, Action)> {
    let (actor, valid_actions) = generate_possible_actions(state);
    
    // Build hand index map
    let mut hand_index_map = std::collections::HashMap::new();
    for (idx, card) in state.hands[actor].iter().enumerate() {
        hand_index_map.insert(card.get_id(), idx);
    }
    
    valid_actions
        .into_iter()
        .filter_map(|action| {
            action_to_index(&action.action, &hand_index_map)
                .map(|idx| (idx, action))
        })
        .collect()
}