//! Vectorized Game Environment for High-Performance RL Training
//!
//! This module provides `VecGame`, a batched environment that manages multiple
//! games simultaneously and returns batched tensors. This reduces Python/Rust FFI
//! overhead from O(n_envs) calls per step to O(1) calls per batch step.

use rand::{rngs::StdRng, SeedableRng};

use log::warn;
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::{
    actions::apply_action,
    deck::Deck,
    generate_possible_actions,
    players::{create_players, parse_player_code, Player},
    rl::{
        get_action_mask, get_indexed_actions, get_observation_tensor, ACTION_SPACE_SIZE,
        OBSERVATION_SIZE,
    },
    state::{GameOutcome, State},
};

#[cfg(feature = "onnx")]
use crate::players::BatchedOnnxInference;

/// Result of a batch step operation
pub struct BatchStepResult {
    /// Observations for all environments (n_envs × obs_size), flattened
    pub observations: Vec<f32>,
    /// Rewards for each environment
    pub rewards: Vec<f32>,
    /// Done flags for each environment
    pub dones: Vec<bool>,
    /// Action masks for all environments (n_envs × action_size), flattened
    pub action_masks: Vec<bool>,
    /// Terminal observations (only set for environments that finished)
    /// Format: Vec of (env_idx, terminal_obs_vector)
    pub terminal_observations: Vec<(usize, Vec<f32>)>,
}

/// A single environment instance within VecGame
struct EnvInstance {
    /// The deck strings for resetting
    deck_a: String,
    deck_b: String,
    /// Current game state
    state: State,
    /// RNG for this environment
    rng: StdRng,
    /// Opponent type for bot games (None = self-play, Some("e2") = expectiminimax, etc.)
    opponent_type: Option<String>,
    /// Bot players (only used when opponent_type is set)
    players: Option<Vec<Box<dyn Player>>>,
}

impl EnvInstance {
    fn new(deck_a: String, deck_b: String, seed: u64, opponent_type: Option<String>) -> Self {
        let deck_a_parsed = Deck::from_string(&deck_a).expect("Failed to parse deck A");
        let deck_b_parsed = Deck::from_string(&deck_b).expect("Failed to parse deck B");
        let mut rng = StdRng::seed_from_u64(seed);
        let state = State::initialize(&deck_a_parsed, &deck_b_parsed, &mut rng);

        // Create bot players if opponent_type is specified
        let players = opponent_type.as_ref().map(|opp_type| {
            let player_codes = vec![
                parse_player_code("r").unwrap(), // Player 0: random (agent controls)
                parse_player_code(opp_type).unwrap(), // Player 1: bot
            ];
            create_players(deck_a_parsed.clone(), deck_b_parsed.clone(), player_codes)
        });

        Self {
            deck_a,
            deck_b,
            state,
            rng,
            opponent_type,
            players,
        }
    }

    fn reset(&mut self, seed: u64) {
        let deck_a_parsed = Deck::from_string(&self.deck_a).expect("Failed to parse deck A");
        let deck_b_parsed = Deck::from_string(&self.deck_b).expect("Failed to parse deck B");
        self.rng = StdRng::seed_from_u64(seed);
        self.state = State::initialize(&deck_a_parsed, &deck_b_parsed, &mut self.rng);

        // Recreate bot players if needed (but not for "onnx" - that's handled separately)
        if let Some(opp_type) = &self.opponent_type {
            if opp_type != "onnx" {
                let player_codes = vec![
                    parse_player_code("r").unwrap(),
                    parse_player_code(opp_type).unwrap(),
                ];
                self.players = Some(create_players(deck_a_parsed, deck_b_parsed, player_codes));
            }
        }
    }

    fn reset_with_new_decks(&mut self, deck_a: String, deck_b: String, seed: u64) {
        self.deck_a = deck_a;
        self.deck_b = deck_b;
        self.reset(seed);
    }

    fn get_observation(&self) -> Vec<f32> {
        // Agent is always player 0 - always return observation from agent's perspective
        get_observation_tensor(&self.state, 0)
    }

    #[allow(dead_code)]
    fn get_action_mask(&self) -> Vec<bool> {
        get_action_mask(&self.state)
    }

    /// Step the environment with an action index
    /// Returns (reward, done) where reward is score-based
    fn step(&mut self, action_idx: usize) -> (f32, bool) {
        let current_player = self.state.current_player;
        let indexed_actions = get_indexed_actions(&self.state);

        // Find and apply the action
        if let Some((_, action)) = indexed_actions.iter().find(|(idx, _)| *idx == action_idx) {
            apply_action(&mut self.rng, &mut self.state, action);
        } else {
            // Invalid action - this shouldn't happen if mask is used correctly
            // Apply first available action as fallback
            if let Some((_, action)) = indexed_actions.first() {
                apply_action(&mut self.rng, &mut self.state, action);
            }
        }

        let done = self.state.is_game_over();
        let reward = if done {
            self.calculate_reward(current_player)
        } else {
            0.0
        };

        (reward, done)
    }

    /// Play bot's turn(s) until it's the agent's turn or game over
    /// Returns (reward, done) - reward is non-zero only if game ended during bot's turn
    fn play_bot_turns(&mut self) -> (f32, bool) {
        if self.opponent_type.is_none() || self.players.is_none() {
            return (0.0, false);
        }

        let agent_player = 0; // Agent is always player 0
        let mut final_reward = 0.0;

        while self.state.current_player != agent_player && !self.state.is_game_over() {
            // Bot (player 1) plays
            let (_, actions) = generate_possible_actions(&self.state);
            if actions.is_empty() {
                break;
            }

            let action = if actions.len() == 1 {
                actions[0].clone()
            } else {
                let players = self.players.as_mut().unwrap();
                players[1].decision_fn(&mut self.rng, &self.state, &actions)
            };

            apply_action(&mut self.rng, &mut self.state, &action);

            if self.state.is_game_over() {
                final_reward = self.calculate_reward(agent_player);
                break;
            }
        }

        (final_reward, self.state.is_game_over())
    }

    /// Calculate turn-based reward from agent's perspective
    fn calculate_reward(&self, agent_player: usize) -> f32 {
        match self.state.winner {
            Some(GameOutcome::Win(winner)) => {
                let my_points = self.state.points[agent_player] as f32;
                let opp_points = self.state.points[1 - agent_player] as f32;
                let point_diff = my_points - opp_points;
                let turn_count = self.state.turn_count as f32;

                // Speed factor: 1 + (13 - turns) / 13
                let speed_factor = 1.0 + ((13.0 - turn_count) / 13.0);

                if winner == agent_player {
                    // Victory: (1.0 + diff/6) * speed [1.0 to 1.5 * speed]
                    let base = 1.0 + (point_diff / 6.0);
                    base * speed_factor.max(1.0)
                } else {
                    // Loss: (-1.0 + diff/6) * speed [-1.5 to -1.0 * speed]
                    // Remember point_diff is negative here
                    let base = -1.0 + (point_diff / 6.0);
                    base * speed_factor
                }
            }
            Some(GameOutcome::Tie) => 0.0,
            None => 0.0,
        }
    }
}

/// Vectorized game manager for batched RL training
pub struct VecGame {
    envs: Vec<EnvInstance>,
    n_envs: usize,
    /// Seed counter for generating unique seeds
    seed_counter: u64,
    /// ONNX opponent for self-play (optional, requires 'onnx' feature)
    #[cfg(feature = "onnx")]
    onnx_opponent: Option<BatchedOnnxInference>,
    /// RNG for ONNX opponent sampling
    #[cfg(feature = "onnx")]
    onnx_rng: StdRng,
}

impl VecGame {
    /// Create a new VecGame with n_envs parallel environments
    ///
    /// # Arguments
    /// * `deck_pairs` - Vec of (deck_a_string, deck_b_string) for each environment
    /// * `base_seed` - Base seed for RNG (each env gets base_seed + env_idx)
    /// * `opponent_type` - Optional opponent type ("e2", "e3", etc.) for bot games
    pub fn new(
        deck_pairs: Vec<(String, String)>,
        base_seed: u64,
        opponent_type: Option<String>,
    ) -> Self {
        let n_envs = deck_pairs.len();
        let envs = deck_pairs
            .into_iter()
            .enumerate()
            .map(|(i, (deck_a, deck_b))| {
                EnvInstance::new(deck_a, deck_b, base_seed + i as u64, opponent_type.clone())
            })
            .collect();

        Self {
            envs,
            n_envs,
            seed_counter: base_seed + n_envs as u64,
            #[cfg(feature = "onnx")]
            onnx_opponent: None,
            #[cfg(feature = "onnx")]
            onnx_rng: StdRng::seed_from_u64(base_seed.wrapping_add(999)),
        }
    }

    /// Set ONNX model as opponent for self-play
    ///
    /// Once set, all environments will use the ONNX model as player 1 (opponent).
    /// The model is shared across all environments and runs batched inference.
    ///
    /// # Arguments
    /// * `model_path` - Path to the .onnx model file
    /// * `deterministic` - If true, always pick best action; if false, sample from distribution
    #[cfg(feature = "onnx")]
    pub fn set_onnx_opponent(
        &mut self,
        model_path: &str,
        deterministic: bool,
    ) -> Result<(), String> {
        let inference = BatchedOnnxInference::new(model_path, deterministic)?;
        self.onnx_opponent = Some(inference);

        // Clear bot players since we're using ONNX now
        for env in &mut self.envs {
            env.opponent_type = Some("onnx".to_string());
            env.players = None;
        }

        Ok(())
    }

    /// Clear ONNX opponent and revert to no opponent (self-play mode handled in Python)
    #[cfg(feature = "onnx")]
    pub fn clear_onnx_opponent(&mut self) {
        self.onnx_opponent = None;
        for env in &mut self.envs {
            env.opponent_type = None;
            env.players = None;
        }
    }

    /// Get number of environments
    pub fn num_envs(&self) -> usize {
        self.n_envs
    }

    /// Get observation size
    pub fn observation_size(&self) -> usize {
        OBSERVATION_SIZE
    }

    /// Get action space size
    pub fn action_space_size(&self) -> usize {
        ACTION_SPACE_SIZE
    }

    /// Reset all environments and return batched observations
    /// Returns flattened observations (n_envs × obs_size)
    pub fn reset_all(&mut self) -> Vec<f32> {
        let mut observations = Vec::with_capacity(self.n_envs * OBSERVATION_SIZE);

        for (i, env) in self.envs.iter_mut().enumerate() {
            let _ = catch_unwind(AssertUnwindSafe(|| {
                self.seed_counter += 1;
                env.reset(self.seed_counter);

                // If using bots, play bot turns first if bot goes first
                if env.opponent_type.is_some() {
                    env.play_bot_turns();
                }
            })).map_err(|_| {
                warn!("Panic caught in environment {} during reset_all! Forcing secondary reset.", i);
                self.seed_counter += 1;
                let _ = catch_unwind(AssertUnwindSafe(|| env.reset(self.seed_counter)));
            });

            // Get observation (from new episode if reset happened or fallback to zeros)
            let obs = catch_unwind(AssertUnwindSafe(|| env.get_observation()))
                .unwrap_or_else(|_| vec![0.0; OBSERVATION_SIZE]);
            observations.extend(obs);
        }

        observations
    }

    /// Reset a single environment with new decks
    pub fn reset_single(&mut self, env_idx: usize, deck_a: String, deck_b: String) {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            self.seed_counter += 1;
            self.envs[env_idx].reset_with_new_decks(deck_a, deck_b, self.seed_counter);

            // If using bots, play bot turns first if bot goes first
            if self.envs[env_idx].opponent_type.is_some() {
                self.envs[env_idx].play_bot_turns();
            }
        })).map_err(|_| {
            warn!("Panic caught in environment {} during reset_single!", env_idx);
        });
    }

    /// Get batched action masks for all environments
    /// Returns flattened masks (n_envs × action_size)
    pub fn get_action_masks(&self) -> Vec<bool> {
        let mut masks = Vec::with_capacity(self.n_envs * ACTION_SPACE_SIZE);

        for (i, env) in self.envs.iter().enumerate() {
            let mask = catch_unwind(AssertUnwindSafe(|| env.get_action_mask()))
                .unwrap_or_else(|_| {
                    warn!("Panic caught in environment {} during get_action_mask! Returning safe mask.", i);
                    vec![false; ACTION_SPACE_SIZE]
                });
            masks.extend(mask);
        }

        masks
    }

    /// Get batched observations for all environments
    /// Returns flattened observations (n_envs × obs_size)
    pub fn get_observations(&self) -> Vec<f32> {
        let mut observations = Vec::with_capacity(self.n_envs * OBSERVATION_SIZE);

        for (i, env) in self.envs.iter().enumerate() {
            let obs = catch_unwind(AssertUnwindSafe(|| env.get_observation()))
                .unwrap_or_else(|_| {
                    warn!("Panic caught in environment {} during get_observation!", i);
                    vec![0.0; OBSERVATION_SIZE]
                });
            observations.extend(obs);
        }

        observations
    }

    /// Step all environments with batched actions
    ///
    /// This is the key method - ONE FFI call steps all environments!
    /// Environments that finish are NOT auto-reset here (caller handles that).
    ///
    /// # Arguments
    /// * `actions` - Action indices for each environment (length = n_envs)
    ///
    /// # Returns
    /// BatchStepResult containing observations, rewards, dones, masks, and terminal observations
    pub fn step_batch(&mut self, actions: &[usize]) -> BatchStepResult {
        assert_eq!(
            actions.len(),
            self.n_envs,
            "Actions length must match n_envs"
        );

        let mut observations = Vec::with_capacity(self.n_envs * OBSERVATION_SIZE);
        let mut rewards = Vec::with_capacity(self.n_envs);
        let mut dones = Vec::with_capacity(self.n_envs);
        let mut terminal_observations = Vec::new();

        // First pass: step agent actions
        for (i, (env, &action)) in self.envs.iter_mut().zip(actions.iter()).enumerate() {
            // Step the agent's action with panic protection
            let result = catch_unwind(AssertUnwindSafe(|| {
                let (mut reward, mut done) = env.step(action);

                // If using classical bots (not ONNX) and game not over, play bot turns
                #[cfg(feature = "onnx")]
                let is_onnx = self.onnx_opponent.is_some();
                #[cfg(not(feature = "onnx"))]
                let is_onnx = false;

                if !done && env.opponent_type.is_some() && !is_onnx {
                    let (bot_reward, bot_done) = env.play_bot_turns();
                    if bot_done {
                        done = true;
                        reward = bot_reward;
                    }
                }
                (reward, done)
            }));

            match result {
                Ok((reward, done)) => {
                    rewards.push(reward);
                    dones.push(done);
                }
                Err(_) => {
                    warn!("Panic caught in environment {} during step! Forcing reset.", i);
                    rewards.push(0.0);
                    dones.push(true); // Force reset in next loop
                }
            }
        }

        // ONNX opponent turns (batched) - take onnx temporarily to avoid borrow issues
        #[cfg(feature = "onnx")]
        {
            if let Some(mut onnx) = self.onnx_opponent.take() {
                self.play_onnx_opponent_turns_internal(&mut onnx, &mut rewards, &mut dones);
                self.onnx_opponent = Some(onnx);
            }
        }

        // Handle resets and collect observations
        #[cfg(feature = "onnx")]
        let has_onnx = self.onnx_opponent.is_some();
        #[cfg(not(feature = "onnx"))]
        let has_onnx = false;

        for (i, env) in self.envs.iter_mut().enumerate() {
            if dones[i] {
                // Wrap observation and reset in catch_unwind as well
                let _ = catch_unwind(AssertUnwindSafe(|| {
                    // Store terminal observation before auto-reset
                    terminal_observations.push((i, env.get_observation()));

                    // Auto-reset this environment
                    self.seed_counter += 1;
                    env.reset(self.seed_counter);

                    // Play bot turns if bot goes first in new game (non-ONNX only)
                    if env.opponent_type.is_some() && !has_onnx {
                        env.play_bot_turns();
                    }
                })).map_err(|_| {
                    warn!("Panic caught in environment {} during reset! Forcing a secondary reset.", i);
                    // If reset itself panics, we try one more time or just hope next one works
                    self.seed_counter += 1;
                    let _ = catch_unwind(AssertUnwindSafe(|| {
                        env.reset(self.seed_counter);
                    }));
                });
            }

            // Get observation (from new episode if reset happened)
            // Wrap in case state is corrupted
            let obs = catch_unwind(AssertUnwindSafe(|| env.get_observation()))
                .unwrap_or_else(|_| vec![0.0; OBSERVATION_SIZE]);
            observations.extend(obs);
        }

        // ONNX opponent initial turns for newly reset envs
        #[cfg(feature = "onnx")]
        {
            if let Some(mut onnx) = self.onnx_opponent.take() {
                self.play_onnx_initial_turns(&mut onnx, &dones, &mut observations);
                self.onnx_opponent = Some(onnx);
            }
        }

        // Get action masks for current state (after any resets)
        let action_masks = self.get_action_masks();

        BatchStepResult {
            observations,
            rewards,
            dones,
            action_masks,
            terminal_observations,
        }
    }

    /// Play ONNX opponent turns for all eligible environments (batched)
    #[cfg(feature = "onnx")]
    fn play_onnx_opponent_turns_internal(
        &mut self,
        onnx: &mut BatchedOnnxInference,
        rewards: &mut [f32],
        dones: &mut [bool],
    ) {
        const MAX_OPPONENT_TURNS: usize = 50; // Prevent infinite loops

        for _ in 0..MAX_OPPONENT_TURNS {
            // Find envs where it's opponent's turn and game not over
            let mut needs_onnx: Vec<usize> = Vec::new();
            for (i, env) in self.envs.iter().enumerate() {
                if !dones[i] && env.state.current_player == 1 && !env.state.is_game_over() {
                    needs_onnx.push(i);
                }
            }

            if needs_onnx.is_empty() {
                break;
            }

            // Collect observations and masks for those envs
            let mut obs_batch: Vec<f32> = Vec::with_capacity(needs_onnx.len() * OBSERVATION_SIZE);
            let mut mask_batch: Vec<bool> =
                Vec::with_capacity(needs_onnx.len() * ACTION_SPACE_SIZE);

            for &i in &needs_onnx {
                obs_batch.extend(get_observation_tensor(&self.envs[i].state, 1)); // Player 1's view
                mask_batch.extend(get_action_mask(&self.envs[i].state));
            }

            // Run batched ONNX inference
            let actions = onnx.predict_batch(
                &obs_batch,
                &mask_batch,
                needs_onnx.len(),
                &mut self.onnx_rng,
            );

            // Apply actions with panic protection
            for (j, &env_idx) in needs_onnx.iter().enumerate() {
                let action_idx = actions[j];
                let _ = catch_unwind(AssertUnwindSafe(|| {
                    let env = &mut self.envs[env_idx];

                    let indexed_actions = get_indexed_actions(&env.state);
                    if let Some((_, action)) =
                        indexed_actions.iter().find(|(idx, _)| *idx == action_idx)
                    {
                        apply_action(&mut env.rng, &mut env.state, action);
                    } else if let Some((_, action)) = indexed_actions.first() {
                        apply_action(&mut env.rng, &mut env.state, action);
                    }

                    if env.state.is_game_over() {
                        dones[env_idx] = true;
                        rewards[env_idx] = env.calculate_reward(0); // Agent is player 0
                    }
                })).map_err(|_| {
                    warn!("Panic caught in environment {} during ONNX action! Marking done.", env_idx);
                    dones[env_idx] = true;
                    rewards[env_idx] = 0.0;
                });
            }
        }
    }

    /// Play ONNX initial turns for newly reset envs where opponent goes first
    #[cfg(feature = "onnx")]
    fn play_onnx_initial_turns(
        &mut self,
        onnx: &mut BatchedOnnxInference,
        was_done: &[bool],
        observations: &mut Vec<f32>,
    ) {
        // Find reset envs where opponent goes first
        let mut needs_initial: Vec<usize> = Vec::new();
        for (i, env) in self.envs.iter().enumerate() {
            if was_done[i] && env.state.current_player == 1 && !env.state.is_game_over() {
                needs_initial.push(i);
            }
        }

        if needs_initial.is_empty() {
            return;
        }

        // Play opponent turns until agent's turn
        const MAX_TURNS: usize = 50;
        for _ in 0..MAX_TURNS {
            let mut active: Vec<usize> = Vec::new();
            for &i in &needs_initial {
                if self.envs[i].state.current_player == 1 && !self.envs[i].state.is_game_over() {
                    active.push(i);
                }
            }

            if active.is_empty() {
                break;
            }

            // Batch inference
            let mut obs_batch: Vec<f32> = Vec::with_capacity(active.len() * OBSERVATION_SIZE);
            let mut mask_batch: Vec<bool> = Vec::with_capacity(active.len() * ACTION_SPACE_SIZE);

            for &i in &active {
                obs_batch.extend(get_observation_tensor(&self.envs[i].state, 1));
                mask_batch.extend(get_action_mask(&self.envs[i].state));
            }

            let actions =
                onnx.predict_batch(&obs_batch, &mask_batch, active.len(), &mut self.onnx_rng);

            for (j, &env_idx) in active.iter().enumerate() {
                let action_idx = actions[j];
                let _ = catch_unwind(AssertUnwindSafe(|| {
                    let env = &mut self.envs[env_idx];

                    let indexed_actions = get_indexed_actions(&env.state);
                    if let Some((_, action)) =
                        indexed_actions.iter().find(|(idx, _)| *idx == action_idx)
                    {
                        apply_action(&mut env.rng, &mut env.state, action);
                    } else if let Some((_, action)) = indexed_actions.first() {
                        apply_action(&mut env.rng, &mut env.state, action);
                    }
                })).map_err(|_| {
                    warn!("Panic caught in environment {} during ONNX initial turn! Forcing end of turns.", env_idx);
                    // Just let it be, the agent will handle it in the main loop if still cur_player 1
                });
            }
        }

        // Update observations for reset envs
        for &i in &needs_initial {
            let start = i * OBSERVATION_SIZE;
            let end = start + OBSERVATION_SIZE;
            let new_obs = self.envs[i].get_observation();
            observations[start..end].copy_from_slice(&new_obs);
        }
    }

    /// Update deck pairs for auto-reset (called from Python when curriculum changes)
    pub fn set_deck_pairs(&mut self, deck_pairs: Vec<(String, String)>) {
        assert_eq!(
            deck_pairs.len(),
            self.n_envs,
            "Deck pairs length must match n_envs"
        );

        for (env, (deck_a, deck_b)) in self.envs.iter_mut().zip(deck_pairs.into_iter()) {
            env.deck_a = deck_a;
            env.deck_b = deck_b;
        }
    }

    /// Get current turn counts for all environments (for debugging/logging)
    pub fn get_turn_counts(&self) -> Vec<u8> {
        self.envs.iter().map(|env| env.state.turn_count).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_deck_string() -> String {
        r#"Energy: Grass
2 Bulbasaur A1 001
2 Ivysaur A1 002
2 Venusaur A1 003
2 Oddish A1 043
2 Gloom A1 044
2 Vileplume A1 045
2 Exeggcute A1 076
2 Exeggutor A1 077
2 Tangela A1 078
2 Pinsir A1 080"#
            .to_string()
    }

    #[test]
    fn test_vec_game_creation() {
        let deck_pairs = vec![
            (test_deck_string(), test_deck_string()),
            (test_deck_string(), test_deck_string()),
        ];

        let vec_game = VecGame::new(deck_pairs, 42, None);
        assert_eq!(vec_game.num_envs(), 2);
    }

    #[test]
    fn test_vec_game_reset_all() {
        let deck_pairs = vec![
            (test_deck_string(), test_deck_string()),
            (test_deck_string(), test_deck_string()),
        ];

        let mut vec_game = VecGame::new(deck_pairs, 42, None);
        let obs = vec_game.reset_all();

        assert_eq!(obs.len(), 2 * OBSERVATION_SIZE);
    }

    #[test]
    fn test_vec_game_get_masks() {
        let deck_pairs = vec![(test_deck_string(), test_deck_string())];

        let vec_game = VecGame::new(deck_pairs, 42, None);
        let masks = vec_game.get_action_masks();

        assert_eq!(masks.len(), ACTION_SPACE_SIZE);
        // At least one action should be valid
        assert!(masks.iter().any(|&m| m));
    }
}
