// Action Mask Generation
//
// Generates a boolean mask indicating which actions are valid for the current state.
// Maps ALL SimpleAction variants to fixed indices for RL compatibility.

use crate::actions::{Action, SimpleAction};
use crate::models::Card;
use crate::move_generation::generate_possible_actions;
use crate::state::State;
use crate::tool_ids::ToolId;

// Action Space 2

pub const ACTION_SPACE_SIZE: usize = 175;

/// Converts a SimpleAction to a canonical action index.
fn action_to_index(
    action: &SimpleAction,
    hand_index_map: &std::collections::HashMap<String, usize>,
    tool_index_map: &std::collections::HashMap<ToolId, usize>,
) -> Option<usize> {
    let index = match action {
        // --- 1. Board Actions (0-29) ---
        SimpleAction::EndTurn => Some(0),
        SimpleAction::Attack(attack_idx) => Some(1 + attack_idx.min(&1)),
        SimpleAction::Retreat(bench_idx) => {
            if *bench_idx >= 1 && *bench_idx <= 3 {
                Some(3 + bench_idx - 1)
            } else {
                None
            }
        }
        SimpleAction::UseAbility { in_play_idx } => {
            if *in_play_idx <= 3 {
                Some(6 + in_play_idx)
            } else {
                None
            }
        }
        SimpleAction::Attach {
            attachments,
            is_turn_energy: true,
        } => {
            if let Some((_, _, in_play_idx)) = attachments.first() {
                if *in_play_idx <= 3 {
                    Some(10 + in_play_idx)
                } else {
                    None
                }
            } else {
                None
            }
        }
        SimpleAction::Activate { in_play_idx, .. } => {
            if *in_play_idx >= 1 && *in_play_idx <= 3 {
                Some(14 + in_play_idx - 1)
            } else if *in_play_idx == 0 {
                None
            } else {
                None
            }
        }
        SimpleAction::DiscardFossil { in_play_idx } => {
            if *in_play_idx <= 3 {
                Some(17 + in_play_idx)
            } else {
                None
            }
        }
        SimpleAction::Heal { in_play_idx, .. } => {
            if *in_play_idx <= 3 {
                Some(21 + in_play_idx)
            } else {
                None
            }
        }
        SimpleAction::AttachFromDiscard { in_play_idx, .. } => {
            if *in_play_idx <= 3 {
                Some(25 + in_play_idx)
            } else {
                None
            }
        }

        // --- 2. Hand Actions (30-129) ---
        // Base: 30 + (HandIdx * 5)

        // Play Trainer (No Target)
        SimpleAction::Play { trainer_card } => {
            hand_index_map.get(&trainer_card.id).and_then(|&idx| {
                if idx < 20 {
                    Some(30 + (idx * 5) + 0)
                } else {
                    None
                }
            })
        }

        // Place Pokemon (Bench)
        SimpleAction::Place(card, in_play_idx) => {
            if let Some(&idx) = hand_index_map.get(&card.get_id()) {
                if idx < 20 && *in_play_idx <= 3 {
                    // Place is targeted to a slot.
                    // 1 + slot.
                    Some(30 + (idx * 5) + 1 + in_play_idx)
                } else {
                    None
                }
            } else {
                // Resolution Place (e.g. from Deck Search)
                if *in_play_idx <= 3 {
                    Some(135 + in_play_idx)
                } else {
                    None
                }
            }
        }

        // Evolve Pokemon
        SimpleAction::Evolve(card, in_play_idx) => {
            if let Some(&idx) = hand_index_map.get(&card.get_id()) {
                if idx < 20 && *in_play_idx <= 3 {
                    // 0 = Play/Evolve
                    Some(30 + (idx * 5) + 0)
                } else {
                    None
                }
            } else {
                // Resolution Evolve (e.g. from Deck Search)
                if *in_play_idx <= 3 {
                    Some(135 + in_play_idx)
                } else {
                    None
                }
            }
        }

        // Attach Tool (Targeted)
        SimpleAction::AttachTool {
            in_play_idx,
            tool_id,
        } => {
            // Try to find the tool in hand (Direct Play)
            if let Some(&idx) = tool_index_map.get(tool_id) {
                if idx < 20 && *in_play_idx <= 3 {
                    // Hand Action: Play from Hand to Slot
                    Some(30 + (idx * 5) + 1 + in_play_idx)
                } else {
                    None
                }
            } else {
                // If not in hand, it's a Resolution Action (Target Selection)
                // Map to Resolution Indices 135-138
                if *in_play_idx <= 3 {
                    Some(135 + in_play_idx)
                } else {
                    None
                }
            }
        }

        // --- 3. Resolution Actions (130-174) ---

        // ApplyDamage Targets (Target Selection)
        SimpleAction::ApplyDamage { targets, .. } => {
            if let Some((_, _, in_play_idx)) = targets.first() {
                if *in_play_idx <= 3 {
                    Some(130 + in_play_idx)
                } else {
                    None
                }
            } else {
                None
            }
        }

        // Hand Resolution Actions (Select Hand Card)
        SimpleAction::DiscardOwnCard { card } => {
            hand_index_map.get(&card.get_id()).and_then(|&idx| {
                if idx < 20 {
                    Some(140 + idx)
                } else {
                    None
                }
            })
        }
        SimpleAction::CommunicatePokemon { hand_pokemon: card } => hand_index_map
            .get(&card.get_id())
            .and_then(|&idx| if idx < 20 { Some(140 + idx) } else { None }),
        SimpleAction::ShufflePokemonIntoDeck { hand_pokemon } => {
            // For ShufflePokemonIntoDeck, it's a Vec<Card>. Use first.
            if let Some(c) = hand_pokemon.first() {
                hand_index_map.get(&c.get_id()).and_then(|&idx| {
                    if idx < 20 {
                        Some(140 + idx)
                    } else {
                        None
                    }
                })
            } else {
                None
            }
        }

        // Misc
        SimpleAction::DrawCard { .. } => Some(160),
        SimpleAction::Noop => Some(161),
        SimpleAction::ApplyEeveeBagDamageBoost => Some(162),
        SimpleAction::HealAllEeveeEvolutions => Some(163),
        SimpleAction::MoveEnergy { .. } => Some(164),
        SimpleAction::MoveAllDamage { .. } => Some(165),
        SimpleAction::ShuffleOpponentSupporter { .. } => Some(166),
        SimpleAction::DiscardOpponentSupporter { .. } => Some(167),
        SimpleAction::Attach {
            is_turn_energy: false,
            ..
        } => Some(168), // Non-turn attach
    };

    if index.is_none() {
        // eprint to see it in python stdout/stderr
        eprintln!("Warning: Unmapped Action: {:?}", action);
    }

    index
}

/// Generates a boolean action mask for the current state.
pub fn get_action_mask(state: &State) -> Vec<bool> {
    let mut mask = vec![false; ACTION_SPACE_SIZE];

    let (actor, valid_actions) = generate_possible_actions(state);

    // Build hand index map and tool index map for this state
    let mut hand_index_map = std::collections::HashMap::new();
    let mut tool_index_map = std::collections::HashMap::new();

    for (idx, card) in state.hands[actor].iter().enumerate() {
        hand_index_map.insert(card.get_id(), idx);

        if let Card::Trainer(t) = card {
            if let Some(tool_id) = ToolId::from_trainer_card(t) {
                // If we haven't seen this tool yet, map it.
                // If there are duplicates, the first one (lowest index) is used, which is fine.
                tool_index_map.entry(*tool_id).or_insert(idx);
            }
        }
    }

    // Mark valid actions in the mask
    for action in &valid_actions {
        if let Some(index) = action_to_index(&action.action, &hand_index_map, &tool_index_map) {
            if index < ACTION_SPACE_SIZE {
                mask[index] = true;
            }
        }
    }

    mask
}

/// Get the list of valid actions with their indices.
pub fn get_indexed_actions(state: &State) -> Vec<(usize, Action)> {
    let (actor, valid_actions) = generate_possible_actions(state);

    // Build hand index map and tool map
    let mut hand_index_map = std::collections::HashMap::new();
    let mut tool_index_map = std::collections::HashMap::new();

    for (idx, card) in state.hands[actor].iter().enumerate() {
        hand_index_map.insert(card.get_id(), idx);

        if let Card::Trainer(t) = card {
            if let Some(tool_id) = ToolId::from_trainer_card(t) {
                tool_index_map.entry(*tool_id).or_insert(idx);
            }
        }
    }

    valid_actions
        .into_iter()
        .filter_map(|action| {
            action_to_index(&action.action, &hand_index_map, &tool_index_map)
                .map(|idx| (idx, action))
        })
        .collect()
}
