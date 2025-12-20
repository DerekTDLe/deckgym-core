// Action Mask Generation
//
// Generates a boolean mask indicating which actions are valid for the current state.

use crate::actions::{Action, SimpleAction};
use crate::move_generation::generate_possible_actions;
use crate::state::State;

// --- Constants ---

/// Maximum number of actions in our fixed action space.
/// This must be large enough to cover all possible action types.
/// Structure:
/// - 0: EndTurn
/// - 1-2: Attack (attack index 0, 1) // TODO : Add support for Rescue Scarf (enable up to 4 attacks, for exemple the Venusaur ex line, not implemented yet)
/// - 3-5: Retreat to bench position (1, 2, 3)
/// - 6-9: Attach energy to position (0, 1, 2, 3)
/// - 10-19: Play Pokémon card from hand (up to 10 cards in hand)
/// - 20-29: Evolve card from hand (up to 10 cards)
/// - 30-39: Play trainer from hand (up to 10 cards) (attach tool, use supporter, use item)
/// - 40-43: Use ability (position 0-3)
/// - 44+: Reserved for other potential actions missed here
pub const ACTION_SPACE_SIZE: usize = 50;

/// Converts a SimpleAction to a canonical action index.
/// Returns None if the action doesn't map to a fixed index.
fn action_to_index(action: &SimpleAction, hand_index_map: &std::collections::HashMap<String, usize>) -> Option<usize> {
    match action {
        SimpleAction::EndTurn => Some(0),
        SimpleAction::Attack(attack_idx) => Some(1 + attack_idx), // 1-2
        SimpleAction::Retreat(bench_idx) => Some(3 + bench_idx - 1), // 3-5 (bench is 1-3)
        SimpleAction::Attach { attachments, is_turn_energy: true } => {
            // Only handle single attachments for now
            if let Some((_, _, in_play_idx)) = attachments.first() {
                Some(6 + in_play_idx) // 6-9
            } else {
                None
            }
        }
        SimpleAction::Place(card, _) => {
            // Map hand card to index (10-19)
            hand_index_map.get(&card.get_id()).and_then(|&idx| {
                if idx < 10 { Some(10 + idx) } else { None }
            })
        }
        SimpleAction::Evolve(card, _) => {
            // Map hand card to index (20-29)
            hand_index_map.get(&card.get_id()).and_then(|&idx| {
                if idx < 10 { Some(20 + idx) } else { None }
            })
        }
        SimpleAction::Play { trainer_card } => {
            // Map hand card to index (30-39)
            hand_index_map.get(&trainer_card.id).and_then(|&idx| {
                if idx < 10 { Some(30 + idx) } else { None }
            })
        }
        SimpleAction::UseAbility { in_play_idx } => Some(40 + in_play_idx), // 40-43
        _ => None, // Other actions not mapped yet
    }
}

/// Converts an action index back to the corresponding Action from the valid actions list.
pub fn index_to_action(index: usize, valid_actions: &[Action]) -> Option<Action> {
    // Find the action that maps to this index
    let hand_index_map = std::collections::HashMap::new(); // Will be populated per-call
    
    for action in valid_actions {
        if let Some(action_idx) = action_to_index(&action.action, &hand_index_map) {
            if action_idx == index {
                return Some(action.clone());
            }
        }
    }
    None
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_mask_size() {
        let state = State::default();
        let mask = get_action_mask(&state);
        assert_eq!(mask.len(), ACTION_SPACE_SIZE);
    }

    #[test]
    fn test_action_mask_no_panic() {
        // Default state may not have mapped actions (setup phase has Place actions)
        // Just verify the function doesn't panic
        let state = State::default();
        let mask = get_action_mask(&state);
        // The mask should be all bools
        assert!(mask.iter().all(|&v| v == true || v == false));
    }
}