//! Batched Game Runner for GPU-Optimized ONNX Simulation
//!
//! This module provides `BatchedGameRunner`, which runs multiple games simultaneously
//! and batches all ONNX inference calls for maximum GPU utilization.
//!
//! Unlike the standard `Simulation::run()` which executes games independently,
//! this runner steps all games in lockstep, collecting observations from games
//! that need ONNX inference and running a single batched prediction.

use indicatif::ProgressBar;
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashMap;

use crate::{
    actions::apply_action,
    deck::Deck,
    generate_possible_actions,
    players::{create_players, Player, PlayerCode},
    rl::{get_action_mask, get_indexed_actions, get_observation_tensor},
    simulate::create_progress_bar,
    state::{GameOutcome, State},
};


#[cfg(feature = "onnx")]
use crate::players::BatchedOnnxInference;

/// Configuration for a player in the batched runner
#[derive(Clone)]
#[allow(dead_code)]
enum PlayerConfig {
    /// ONNX model with path and device
    #[cfg(feature = "onnx")]
    Onnx { path: String, device: String },
    /// CPU-based player with code
    Cpu { code: PlayerCode },
}

/// A single game instance within the batched runner
#[allow(dead_code)]
struct GameInstance {
    /// Current game state
    state: State,
    /// RNG for this game
    rng: StdRng,
    /// Parsed decks for player creation
    deck_a: Deck,
    deck_b: Deck,
    /// CPU players (only populated if using CPU players)
    cpu_players: Option<[Box<dyn Player>; 2]>,
}

impl GameInstance {
    fn new(deck_a: Deck, deck_b: Deck, seed: u64, cpu_codes: &[Option<PlayerCode>; 2]) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let state = State::initialize(&deck_a, &deck_b, &mut rng);

        // Create CPU players if needed
        let cpu_players = if cpu_codes[0].is_some() || cpu_codes[1].is_some() {
            let player_codes = vec![
                cpu_codes[0].clone().unwrap_or(PlayerCode::R),
                cpu_codes[1].clone().unwrap_or(PlayerCode::R),
            ];
            let players = create_players(deck_a.clone(), deck_b.clone(), player_codes)
                .expect("Failed to create CPU players");
            // Convert Vec to array
            let mut iter = players.into_iter();
            Some([iter.next().unwrap(), iter.next().unwrap()])
        } else {
            None
        };

        Self {
            state,
            rng,
            deck_a,
            deck_b,
            cpu_players,
        }
    }

    fn is_game_over(&self) -> bool {
        self.state.is_game_over()
    }

    fn get_outcome(&self) -> Option<GameOutcome> {
        self.state.winner
    }

    fn current_player(&self) -> usize {
        self.state.current_player
    }
}

/// Batched game runner for GPU-optimized ONNX simulation
pub struct BatchedGameRunner {
    /// All game instances
    games: Vec<GameInstance>,
    /// Number of games
    n_games: usize,
    /// Player configurations
    player_configs: [PlayerConfig; 2],
    /// ONNX sessions keyed by (path, device) - allows sharing when same model+device
    #[cfg(feature = "onnx")]
    onnx_sessions: HashMap<(String, String), BatchedOnnxInference>,
    /// Keys into onnx_sessions for each player (None if CPU player)
    #[cfg(feature = "onnx")]
    player_onnx_keys: [Option<(String, String)>; 2],
    /// Shared RNG for ONNX sampling
    #[cfg(feature = "onnx")]
    onnx_rng: StdRng,
    /// Progress bar
    progress: Option<ProgressBar>,
    /// Track completed games
    completed_count: usize,
}

impl BatchedGameRunner {
    /// Create a new batched game runner
    ///
    /// # Arguments
    /// * `deck_a_path` - Path to deck A file
    /// * `deck_b_path` - Path to deck B file
    /// * `player_codes` - Player codes for both players
    /// * `num_games` - Number of games to run
    /// * `seed` - Optional base seed
    pub fn new(
        deck_a_path: &str,
        deck_b_path: &str,
        player_codes: Vec<PlayerCode>,
        num_games: u32,
        seed: Option<u64>,
    ) -> Result<Self, String> {
        let n_games = num_games as usize;
        let base_seed = seed.unwrap_or_else(rand::random);

        // Load decks
        let deck_a = Deck::from_file(deck_a_path)
            .map_err(|e| format!("Failed to load deck A: {}", e))?;
        let deck_b = Deck::from_file(deck_b_path)
            .map_err(|e| format!("Failed to load deck B: {}", e))?;

        // Parse player configurations
        let (player_configs, cpu_codes) = Self::parse_player_configs(&player_codes)?;

        // Initialize ONNX sessions if needed
        #[cfg(feature = "onnx")]
        let (onnx_sessions, player_onnx_keys) =
            Self::init_onnx_sessions(&player_configs, base_seed)?;

        // Create game instances
        let games: Vec<GameInstance> = (0..n_games)
            .map(|i| {
                GameInstance::new(
                    deck_a.clone(),
                    deck_b.clone(),
                    base_seed.wrapping_add(i as u64),
                    &cpu_codes,
                )
            })
            .collect();

        Ok(Self {
            games,
            n_games,
            player_configs,
            #[cfg(feature = "onnx")]
            onnx_sessions,
            #[cfg(feature = "onnx")]
            player_onnx_keys,
            #[cfg(feature = "onnx")]
            onnx_rng: StdRng::seed_from_u64(base_seed.wrapping_add(999999)),
            progress: None,
            completed_count: 0,
        })
    }

    /// Parse player codes into configurations
    fn parse_player_configs(
        player_codes: &[PlayerCode],
    ) -> Result<([PlayerConfig; 2], [Option<PlayerCode>; 2]), String> {
        if player_codes.len() != 2 {
            return Err("Exactly 2 player codes required".to_string());
        }

        let mut configs = Vec::with_capacity(2);
        let mut cpu_codes: [Option<PlayerCode>; 2] = [None, None];

        for (i, code) in player_codes.iter().enumerate() {
            match code {
                #[cfg(feature = "onnx")]
                PlayerCode::Onnx { path, device } => {
                    configs.push(PlayerConfig::Onnx {
                        path: path.clone(),
                        device: device.clone(),
                    });
                }
                #[cfg(feature = "onnx")]
                PlayerCode::O { index, device } => {
                    let path = crate::players::get_nth_onnx_model(*index)
                        .map_err(|e| format!("Failed to find model o{}: {}", index, e))?;
                    configs.push(PlayerConfig::Onnx {
                        path,
                        device: device.clone(),
                    });
                }
                _ => {
                    // CPU player
                    configs.push(PlayerConfig::Cpu { code: code.clone() });
                    cpu_codes[i] = Some(code.clone());
                }
            }
        }

        Ok(([configs[0].clone(), configs[1].clone()], cpu_codes))
    }

    /// Initialize ONNX sessions, sharing when path and device match
    #[cfg(feature = "onnx")]
    fn init_onnx_sessions(
        player_configs: &[PlayerConfig; 2],
        _base_seed: u64,
    ) -> Result<
        (
            HashMap<(String, String), BatchedOnnxInference>,
            [Option<(String, String)>; 2],
        ),
        String,
    > {
        let mut sessions: HashMap<(String, String), BatchedOnnxInference> = HashMap::new();
        let mut keys: [Option<(String, String)>; 2] = [None, None];

        for (i, config) in player_configs.iter().enumerate() {
            if let PlayerConfig::Onnx { path, device } = config {
                let key = (path.clone(), device.clone());

                // Only create session if not already cached
                if !sessions.contains_key(&key) {
                    let session = BatchedOnnxInference::new(path, false, device)
                        .map_err(|e| format!("Failed to create ONNX session: {}", e))?;
                    sessions.insert(key.clone(), session);
                }

                keys[i] = Some(key);
            }
        }

        Ok((sessions, keys))
    }

    /// Run all games to completion
    pub fn run_all(mut self) -> Vec<Option<GameOutcome>> {
        // Create progress bar
        self.progress = Some(create_progress_bar(self.n_games as u64));

        // Main loop: step all games until all complete
        loop {
            // Find active games for each player
            let p0_active = self.find_active_games_for_player(0);
            let p1_active = self.find_active_games_for_player(1);

            if p0_active.is_empty() && p1_active.is_empty() {
                // All games done
                break;
            }

            // Step player 0's games
            if !p0_active.is_empty() {
                self.step_player(0, &p0_active);
            }

            // Step player 1's games
            if !p1_active.is_empty() {
                self.step_player(1, &p1_active);
            }

            // Update progress for newly completed games
            let new_completed = self.games.iter().filter(|g| g.is_game_over()).count();
            if new_completed > self.completed_count {
                if let Some(ref pb) = self.progress {
                    pb.set_position(new_completed as u64);
                }
                self.completed_count = new_completed;
            }
        }

        // Finish progress bar
        if let Some(pb) = self.progress.take() {
            pb.finish_with_message("Simulation complete!");
        }

        // Collect outcomes
        self.games.iter().map(|g| g.get_outcome()).collect()
    }

    /// Find indices of active (not finished) games where it's the given player's turn
    fn find_active_games_for_player(&self, player: usize) -> Vec<usize> {
        self.games
            .iter()
            .enumerate()
            .filter(|(_, g)| !g.is_game_over() && g.current_player() == player)
            .map(|(i, _)| i)
            .collect()
    }

    /// Step all games in the given indices for the specified player
    fn step_player(&mut self, player: usize, game_indices: &[usize]) {
        match &self.player_configs[player] {
            #[cfg(feature = "onnx")]
            PlayerConfig::Onnx { .. } => {
                self.step_onnx_player(player, game_indices);
            }
            PlayerConfig::Cpu { .. } => {
                self.step_cpu_player(player, game_indices);
            }
        }
    }

    /// Step ONNX player with batched inference
    #[cfg(feature = "onnx")]
    fn step_onnx_player(&mut self, player: usize, game_indices: &[usize]) {
        if game_indices.is_empty() {
            return;
        }

        let key = match &self.player_onnx_keys[player] {
            Some(k) => k.clone(),
            None => return,
        };

        // Collect observations from all games
        let obs: Vec<f32> = game_indices
            .iter()
            .flat_map(|&i| get_observation_tensor(&self.games[i].state, player))
            .collect();

        // Collect action masks from all games
        let masks: Vec<bool> = game_indices
            .iter()
            .flat_map(|&i| get_action_mask(&self.games[i].state))
            .collect();

        // Run batched inference
        let actions = {
            let session = self.onnx_sessions.get_mut(&key).unwrap();
            session.predict_batch(&obs, &masks, game_indices.len(), &mut self.onnx_rng)
        };

        // Apply actions to each game
        for (j, &i) in game_indices.iter().enumerate() {
            let action_idx = actions[j];
            let game = &mut self.games[i];

            let indexed_actions = get_indexed_actions(&game.state);
            if let Some((_, action)) = indexed_actions.iter().find(|(idx, _)| *idx == action_idx) {
                apply_action(&mut game.rng, &mut game.state, action);
            } else if let Some((_, action)) = indexed_actions.first() {
                // Fallback to first valid action
                apply_action(&mut game.rng, &mut game.state, action);
            }
        }
    }

    /// Step CPU player (sequential for now, could be parallelized)
    fn step_cpu_player(&mut self, player: usize, game_indices: &[usize]) {
        for &i in game_indices {
            let game = &mut self.games[i];

            // Get possible actions
            let (_, actions) = generate_possible_actions(&game.state);
            if actions.is_empty() {
                continue;
            }

            // Use CPU player to select action
            let action = if actions.len() == 1 {
                actions[0].clone()
            } else if let Some(ref mut players) = game.cpu_players {
                players[player].decision_fn(&mut game.rng, &game.state, &actions)
            } else {
                // Fallback to random
                use rand::seq::SliceRandom;
                actions.choose(&mut game.rng).unwrap().clone()
            };

            apply_action(&mut game.rng, &mut game.state, &action);
        }
    }
}

/// Check if a player code is an ONNX player
#[cfg(feature = "onnx")]
pub fn is_onnx_player(code: &PlayerCode) -> bool {
    matches!(code, PlayerCode::Onnx { .. } | PlayerCode::O { .. })
}

#[cfg(not(feature = "onnx"))]
pub fn is_onnx_player(_code: &PlayerCode) -> bool {
    false
}

