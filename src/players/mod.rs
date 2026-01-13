mod attach_attack_player;
mod end_turn_player;
mod evolution_rusher_player;
mod expectiminimax_player;
mod human_player;
mod mcts_player;
#[cfg(feature = "onnx")]
mod onnx_player;
mod random_player;
mod value_function_player;
pub mod value_functions;
mod weighted_random_player;

pub use attach_attack_player::AttachAttackPlayer;
pub use end_turn_player::EndTurnPlayer;
pub use evolution_rusher_player::EvolutionRusherPlayer;
pub use expectiminimax_player::{ExpectiMiniMaxPlayer, ValueFunction};
pub use human_player::HumanPlayer;
pub use mcts_player::MctsPlayer;
#[cfg(feature = "onnx")]
pub use onnx_player::{print_available_providers, BatchedOnnxInference, OnnxPlayer};
pub use random_player::RandomPlayer;
pub use value_function_player::ValueFunctionPlayer;
pub use value_functions::*;
pub use weighted_random_player::WeightedRandomPlayer;

use crate::{actions::Action, Deck, State};
use rand::rngs::StdRng;
use std::fmt::Debug;
#[cfg(feature = "onnx")]
use std::path::Path;

pub trait Player: Debug {
    fn get_deck(&self) -> Deck;
    fn decision_fn(
        &mut self,
        rng: &mut StdRng,
        state: &State,
        possible_actions: &[Action],
    ) -> Action;
}

/// Enum for allowed player strategies
#[derive(Debug, Clone, PartialEq)]
pub enum PlayerCode {
    AA,
    ET,
    R,
    H,
    W,
    M {
        iterations: u64,
    }, // MCTS with configurable iterations
    V,
    E {
        max_depth: usize,
    },
    ER, // Evolution Rusher
    #[cfg(feature = "onnx")]
    Onnx {
        path: String,
        device: String,
    },
    #[cfg(feature = "onnx")]
    O {
        index: usize,   // 1-indexed: o1 = newest model, o2 = second newest, etc.
        device: String, // c=cpu, g=cuda, t=trt, or auto
    },
}
/// Custom parser function enforcing case-insensitivity
pub fn parse_player_code(s: &str) -> Result<PlayerCode, String> {
    let lower = s.to_ascii_lowercase();

    // Handle 'et' and 'er' explicitly first since they start with 'e'
    if lower == "et" {
        return Ok(PlayerCode::ET);
    }
    if lower == "er" {
        return Ok(PlayerCode::ER);
    }

    // Check if it starts with 'e' followed by digits (e.g., e2, e4)
    if lower.starts_with('e') && lower.len() > 1 {
        let rest = &lower[1..];
        if let Ok(max_depth) = rest.parse::<usize>() {
            return Ok(PlayerCode::E { max_depth });
        }
        return Err(format!("Invalid player code: {s}. Use 'e<number>' for ExpectiMiniMax with depth, e.g., 'e2', 'e5'"));
    }

    // Check if it starts with 'm' followed by digits (e.g., m100, m500)
    if lower.starts_with('m') && lower.len() > 1 {
        let rest = &lower[1..];
        if let Ok(iterations) = rest.parse::<u64>() {
            return Ok(PlayerCode::M { iterations });
        }
        return Err(format!("Invalid player code: {s}. Use 'm<number>' for MCTS with iterations, e.g., 'm100', 'm500'"));
    }

    // Check if it starts with 'o' followed by digits and optional device suffix
    // o1 = auto, o1c = cpu, o1g = cuda, o1t = tensorrt
    #[cfg(feature = "onnx")]
    if lower.starts_with('o') && lower.len() > 1 {
        let rest = &lower[1..];

        // Check for device suffix (last char: c, g, t)
        let (num_part, device) = if rest.ends_with('c') {
            (&rest[..rest.len() - 1], "cpu")
        } else if rest.ends_with('g') {
            (&rest[..rest.len() - 1], "cuda")
        } else if rest.ends_with('t') {
            (&rest[..rest.len() - 1], "trt")
        } else {
            (rest, "auto")
        };

        if let Ok(index) = num_part.parse::<usize>() {
            if index == 0 {
                return Err("o0 is invalid. Use o1 for the newest model.".to_string());
            }
            return Ok(PlayerCode::O {
                index,
                device: device.to_string(),
            });
        }
    }

    // Check if it starts with 'onnx:' (e.g., onnx:/path/to/model.onnx)
    #[cfg(feature = "onnx")]
    if lower.starts_with("onnx:") {
        let parts: Vec<&str> = s[5..].split(':').collect();
        let path = parts[0].to_string();
        let device = if parts.len() > 1 {
            parts[1].to_string()
        } else {
            "auto".to_string()
        };
        return Ok(PlayerCode::Onnx { path, device });
    }

    match lower.as_str() {
        "aa" => Ok(PlayerCode::AA),
        "et" => Ok(PlayerCode::ET),
        "r" => Ok(PlayerCode::R),
        "h" => Ok(PlayerCode::H),
        "w" => Ok(PlayerCode::W),
        "m" => Ok(PlayerCode::M { iterations: 100 }), // Default 100 iterations
        "v" => Ok(PlayerCode::V),
        "e" => Ok(PlayerCode::E { max_depth: 3 }), // Default depth
        "er" => Ok(PlayerCode::ER),
        _ => Err(format!("Invalid player code: {s}")),
    }
}

pub fn parse_player_code_generic(s: String) -> Result<PlayerCode, String> {
    parse_player_code(s.as_ref())
}

pub fn fill_code_array(maybe_players: Option<Vec<PlayerCode>>) -> Vec<PlayerCode> {
    match maybe_players {
        Some(mut player_codes) => {
            if player_codes.is_empty() || player_codes.len() > 2 {
                panic!("Invalid number of players");
            } else if player_codes.len() == 1 {
                player_codes.push(PlayerCode::R);
            }
            player_codes
        }
        None => vec![PlayerCode::R, PlayerCode::R],
    }
}

pub fn create_players(
    deck_a: Deck,
    deck_b: Deck,
    players: Vec<PlayerCode>,
) -> Result<Vec<Box<dyn Player>>, String> {
    let player_a: Box<dyn Player> = get_player(deck_a.clone(), &players[0])?;
    let player_b: Box<dyn Player> = get_player(deck_b.clone(), &players[1])?;
    Ok(vec![player_a, player_b])
}

fn get_player(deck: Deck, player: &PlayerCode) -> Result<Box<dyn Player>, String> {
    match player {
        PlayerCode::AA => Ok(Box::new(AttachAttackPlayer { deck })),
        PlayerCode::ET => Ok(Box::new(EndTurnPlayer { deck })),
        PlayerCode::R => Ok(Box::new(RandomPlayer { deck })),
        PlayerCode::H => Ok(Box::new(HumanPlayer { deck })),
        PlayerCode::W => Ok(Box::new(WeightedRandomPlayer { deck })),
        PlayerCode::M { iterations } => Ok(Box::new(MctsPlayer::new(deck, *iterations))),
        PlayerCode::V => Ok(Box::new(ValueFunctionPlayer { deck })),
        PlayerCode::E { max_depth } => Ok(Box::new(ExpectiMiniMaxPlayer {
            deck,
            max_depth: *max_depth,
            write_debug_trees: false,
            value_function: Box::new(value_functions::baseline_value_function),
        })),
        PlayerCode::ER => Ok(Box::new(EvolutionRusherPlayer { deck })),
        #[cfg(feature = "onnx")]
        PlayerCode::Onnx { path, device } => Ok(Box::new(
            OnnxPlayer::new(&path, deck, true, device)
                .map_err(|e| format!("Failed to load ONNX model: {}", e))?,
        )),
        #[cfg(feature = "onnx")]
        PlayerCode::O { index, device } => {
            let path = get_nth_onnx_model(*index).map_err(|e| {
                format!(
                "Failed to find ONNX model o{}. Make sure models/ contains .onnx files. Error: {}",
                index, e
            )
            })?;
            Ok(Box::new(
                OnnxPlayer::new(&path, deck, true, device)
                    .map_err(|e| format!("Failed to load ONNX model from {}: {}", path, e))?,
            ))
        }
    }
}

/// Get the Nth newest ONNX model from models/ directory.
/// Index is 1-based: 1 = newest, 2 = second newest, etc.
#[cfg(feature = "onnx")]
pub fn get_nth_onnx_model(index: usize) -> Result<String, String> {
    let models_dir = Path::new("models");
    if !models_dir.exists() {
        return Err("models/ directory not found".to_string());
    }

    // Collect all .onnx files with their modification times
    let mut onnx_files: Vec<(std::path::PathBuf, std::time::SystemTime)> =
        std::fs::read_dir(models_dir)
            .map_err(|e| format!("Failed to read models/ directory: {}", e))?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("onnx") {
                    let mtime = entry.metadata().ok()?.modified().ok()?;
                    Some((path, mtime))
                } else {
                    None
                }
            })
            .collect();

    if onnx_files.is_empty() {
        return Err("No .onnx files found in models/ directory".to_string());
    }

    // Sort by modification time, newest first
    onnx_files.sort_by(|a, b| b.1.cmp(&a.1));

    // Get the Nth file (1-indexed)
    if index > onnx_files.len() {
        return Err(format!(
            "o{} requested but only {} ONNX models found in models/",
            index,
            onnx_files.len()
        ));
    }

    let selected = &onnx_files[index - 1];

    Ok(selected.0.to_string_lossy().to_string())
}
