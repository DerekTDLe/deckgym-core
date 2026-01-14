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

/// Reward for a tie game
const REWARD_DRAW_BASE: f32 = -0.5;

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
                .expect("Failed to create players")
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
                self.players = Some(
                    create_players(deck_a_parsed, deck_b_parsed, player_codes)
                        .expect("Failed to create players"),
                );
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

                // Speed factor: bonus for fast games (< 13 turns), clamped to min 1.0
                // At 13 turns: factor = 1.0, at 6 turns: factor = 1.54
                let speed_factor = (1.0 + ((13.0 - turn_count) / 13.0)).max(1.0);

                if winner == agent_player {
                    // Victory: (1.0 + diff/6) * speed [1.0 to 1.5 * speed]
                    let base = 1.0 + (point_diff / 6.0);
                    base * speed_factor
                } else {
                    // Loss: (-1.0 + diff/6) * speed [-1.5 to -1.0 * speed]
                    // speed_factor is already clamped to >= 1.0, so loss stays negative
                    let base = -1.0 + (point_diff / 6.0);
                    base * speed_factor
                }
            }
            Some(GameOutcome::Tie) => REWARD_DRAW_BASE,
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
    /// ONNX opponent for self-play (single model, legacy mode)
    #[cfg(feature = "onnx")]
    onnx_opponent: Option<BatchedOnnxInference>,
    /// ONNX opponent pool for per-episode selection (indexed by pool position)
    #[cfg(feature = "onnx")]
    onnx_pool: Vec<Option<BatchedOnnxInference>>,
    /// Pool opponent names (for external API), index matches onnx_pool
    #[cfg(feature = "onnx")]
    onnx_pool_names: Vec<String>,
    /// Per-env opponent index into onnx_pool (None = use onnx_opponent legacy mode)
    #[cfg(feature = "onnx")]
    env_opponent_indices: Vec<Option<usize>>,
    /// Pre-allocated grouping buffer: one Vec per pool opponent
    #[cfg(feature = "onnx")]
    opponent_groups: Vec<Vec<usize>>,
    /// RNG for ONNX opponent sampling
    #[cfg(feature = "onnx")]
    onnx_rng: StdRng,
    /// Baseline bot codes: maps opponent name to bot code (e.g., "baseline_e2" -> "e2")
    /// Baselines in the pool use built-in bots instead of ONNX
    baseline_codes: std::collections::HashMap<String, String>,
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
            onnx_pool: Vec::new(),
            #[cfg(feature = "onnx")]
            onnx_pool_names: Vec::new(),
            #[cfg(feature = "onnx")]
            env_opponent_indices: vec![None; n_envs],
            #[cfg(feature = "onnx")]
            opponent_groups: Vec::new(),
            #[cfg(feature = "onnx")]
            onnx_rng: StdRng::seed_from_u64(base_seed.wrapping_add(999)),
            baseline_codes: std::collections::HashMap::new(),
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
        device: &str,
    ) -> Result<(), String> {
        let inference = BatchedOnnxInference::new(model_path, deterministic, device)?;
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
        // Also clear the cache to be safe
        crate::players::onnx_player::clear_onnx_cache();
    }

    /// Add an ONNX model to the opponent pool
    ///
    /// Models in the pool can be assigned to individual environments using set_env_opponent.
    /// This enables per-episode opponent selection (AlphaStar-style PFSP).
    ///
    /// # Arguments
    /// * `name` - Unique name for this opponent (e.g., "pfsp_6815k")
    /// * `model_path` - Path to the .onnx model file
    /// * `deterministic` - If true, always pick best action; if false, sample from distribution
    #[cfg(feature = "onnx")]
    pub fn add_onnx_to_pool(
        &mut self,
        name: &str,
        model_path: &str,
        deterministic: bool,
        device: &str,
    ) -> Result<(), String> {
        // Check if already exists
        if self.onnx_pool_names.contains(&name.to_string()) {
            return Err(format!("Opponent '{}' already in pool", name));
        }

        let inference = BatchedOnnxInference::new(model_path, deterministic, device)?;
        self.onnx_pool.push(Some(inference));
        self.onnx_pool_names.push(name.to_string());
        // Add corresponding grouping buffer with capacity for all envs
        self.opponent_groups.push(Vec::with_capacity(self.n_envs));
        Ok(())
    }

    /// Add a built-in baseline bot to the opponent pool
    ///
    /// This allows PFSP to include non-ONNX opponents like Expectiminimax.
    /// Baselines are handled differently: they don't use batched ONNX inference,
    /// instead they use the EnvInstance's built-in bot system.
    ///
    /// # Arguments
    /// * `name` - Unique name for this opponent (e.g., "baseline_e2")
    /// * `bot_code` - Bot code like "e2", "v", "aa", "er" (see players module)
    pub fn add_baseline_to_pool(&mut self, name: &str, bot_code: &str) -> Result<(), String> {
        // Validate bot code
        parse_player_code(bot_code)?;

        // Check if already exists in baseline_codes or onnx pool
        if self.baseline_codes.contains_key(name) {
            return Err(format!("Baseline '{}' already in pool", name));
        }
        #[cfg(feature = "onnx")]
        if self.onnx_pool_names.contains(&name.to_string()) {
            return Err(format!("Opponent '{}' already in pool", name));
        }

        // Store the baseline code mapping
        self.baseline_codes
            .insert(name.to_string(), bot_code.to_string());

        // Also add to pool names for unified lookup (but NOT to onnx_pool itself)
        #[cfg(feature = "onnx")]
        {
            self.onnx_pool.push(None); // Placeholder for baseline
            self.onnx_pool_names.push(name.to_string());
            // Add a placeholder for opponent_groups to keep indices aligned
            self.opponent_groups.push(Vec::with_capacity(self.n_envs));
        }

        Ok(())
    }

    /// Set the opponent for a specific environment (from the pool)
    ///
    /// The environment will use this opponent until changed again.
    /// This is called per-episode to implement PFSP opponent selection.
    /// Supports both ONNX models and built-in baseline bots.
    ///
    /// # Arguments
    /// * `env_idx` - Environment index
    /// * `opponent_name` - Name of opponent in pool (must exist)
    #[cfg(feature = "onnx")]
    pub fn set_env_opponent(&mut self, env_idx: usize, opponent_name: &str) -> Result<(), String> {
        if env_idx >= self.n_envs {
            return Err(format!(
                "env_idx {} out of range (n_envs={})",
                env_idx, self.n_envs
            ));
        }

        // Check if this is a baseline opponent
        if let Some(bot_code) = self.baseline_codes.get(opponent_name) {
            // Use built-in bot for this env
            let deck_a = Deck::from_string(&self.envs[env_idx].deck_a)
                .map_err(|e| format!("Failed to parse deck A: {}", e))?;
            let deck_b = Deck::from_string(&self.envs[env_idx].deck_b)
                .map_err(|e| format!("Failed to parse deck B: {}", e))?;

            let player_codes = vec![
                parse_player_code("r").unwrap(), // Player 0: random (agent controls)
                parse_player_code(bot_code).map_err(|e| e.to_string())?, // Player 1: baseline bot
            ];

            self.envs[env_idx].opponent_type = Some(bot_code.clone());
            self.envs[env_idx].players = Some(
                create_players(deck_a, deck_b, player_codes)
                    .map_err(|e| format!("Failed to create baseline players: {}", e))?,
            );
            self.env_opponent_indices[env_idx] = None; // Not using ONNX pool index

            return Ok(());
        }

        // Otherwise, it's an ONNX opponent
        let opp_idx = self
            .onnx_pool_names
            .iter()
            .position(|n| n == opponent_name)
            .ok_or_else(|| format!("Opponent '{}' not in pool", opponent_name))?;

        self.env_opponent_indices[env_idx] = Some(opp_idx);
        self.envs[env_idx].opponent_type = Some("onnx".to_string());
        self.envs[env_idx].players = None;

        Ok(())
    }

    /// Clear all opponents from pool and reset per-env assignments
    #[cfg(feature = "onnx")]
    pub fn clear_onnx_pool(&mut self) {
        self.onnx_pool.clear();
        self.onnx_pool_names.clear();
        self.opponent_groups.clear();
        self.baseline_codes.clear();
        for idx in &mut self.env_opponent_indices {
            *idx = None;
        }
        // Force-clear the global session cache to release VRAM
        crate::players::onnx_player::clear_onnx_cache();
    }

    /// Remove a specific opponent from the pool (frees GPU memory for ONNX)
    ///
    /// Returns true if opponent was found and removed, false otherwise.
    /// If any environment was using this opponent, its assignment is cleared.
    /// Works for both ONNX models and baseline bots.
    #[cfg(feature = "onnx")]
    pub fn remove_onnx_from_pool(&mut self, opponent_name: &str) -> bool {
        // Check if it's a baseline first
        let is_baseline = self.baseline_codes.contains_key(opponent_name);

        if is_baseline {
            // Remove baseline
            self.baseline_codes.remove(opponent_name);

            // Also remove from pool names
            if let Some(opp_idx) = self.onnx_pool_names.iter().position(|n| n == opponent_name) {
                self.onnx_pool_names.remove(opp_idx);
                self.opponent_groups.remove(opp_idx);

                // Clear any env using this opponent (baselines use opponent_type, not indices)
                for env in &mut self.envs {
                    if env.opponent_type.as_deref()
                        == Some(
                            self.baseline_codes
                                .get(opponent_name)
                                .map(|s| s.as_str())
                                .unwrap_or(opponent_name),
                        )
                    {
                        env.opponent_type = None;
                        env.players = None;
                    }
                }
            }
            return true;
        }

        // Find the ONNX opponent index
        let opp_idx = match self.onnx_pool_names.iter().position(|n| n == opponent_name) {
            Some(idx) => idx,
            None => return false,
        };

        // If it's an ONNX opponent, remove from global cache to release VRAM
        if let Some(Some(onnx)) = self.onnx_pool.get(opp_idx) {
            crate::players::onnx_player::remove_model_from_cache(&onnx.model_path, &onnx.device);
        }

        self.onnx_pool.remove(opp_idx);
        self.onnx_pool_names.remove(opp_idx);
        self.opponent_groups.remove(opp_idx);

        // Update env assignments: clear any using this opponent, shift others down
        for env_opp_idx in &mut self.env_opponent_indices {
            if let Some(idx) = *env_opp_idx {
                if idx == opp_idx {
                    // This env was using the removed opponent
                    *env_opp_idx = None;
                } else if idx > opp_idx {
                    // Shift down indices above the removed one
                    *env_opp_idx = Some(idx - 1);
                }
            }
        }

        true
    }

    /// Get list of opponent names in the pool
    #[cfg(feature = "onnx")]
    pub fn get_pool_opponent_names(&self) -> Vec<String> {
        self.onnx_pool_names.clone()
    }

    /// Check if using per-env opponents (pool mode) vs single opponent (legacy mode)
    #[cfg(feature = "onnx")]
    pub fn is_using_pool(&self) -> bool {
        !self.onnx_pool.is_empty() || !self.baseline_codes.is_empty()
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
            }))
            .map_err(|_| {
                warn!(
                    "Panic caught in environment {} during reset_all! Forcing secondary reset.",
                    i
                );
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
        }))
        .map_err(|_| {
            warn!(
                "Panic caught in environment {} during reset_single!",
                env_idx
            );
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
            let obs =
                catch_unwind(AssertUnwindSafe(|| env.get_observation())).unwrap_or_else(|_| {
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
                    warn!(
                        "Panic caught in environment {} during step! Forcing reset.",
                        i
                    );
                    rewards.push(0.0);
                    dones.push(true); // Force reset in next loop
                }
            }
        }

        // ONNX opponent turns (batched)
        // Two modes: legacy single-opponent OR pool with per-env opponents
        #[cfg(feature = "onnx")]
        {
            if self.is_using_pool() {
                // Pool mode: group by opponent and run batched inference per group
                self.play_pool_opponent_turns(&mut rewards, &mut dones);
            } else if let Some(mut onnx) = self.onnx_opponent.take() {
                // Legacy mode: single opponent for all envs
                self.play_onnx_opponent_turns_internal(&mut onnx, &mut rewards, &mut dones);
                self.onnx_opponent = Some(onnx);
            }
        }

        // Handle resets and collect observations
        #[cfg(feature = "onnx")]
        let has_onnx = self.onnx_opponent.is_some() || self.is_using_pool();
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
                }))
                .map_err(|_| {
                    warn!(
                        "Panic caught in environment {} during reset! Forcing a secondary reset.",
                        i
                    );
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
            if self.is_using_pool() {
                self.play_pool_initial_turns(&dones, &mut observations);
            } else if let Some(mut onnx) = self.onnx_opponent.take() {
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
                }))
                .map_err(|_| {
                    warn!(
                        "Panic caught in environment {} during ONNX action! Marking done.",
                        env_idx
                    );
                    dones[env_idx] = true;
                    rewards[env_idx] = 0.0;
                });
            }
        }
    }

    /// Play opponent turns using pool mode (per-env opponent assignment)
    /// Groups environments by their assigned opponent and runs batched inference per group
    #[cfg(feature = "onnx")]
    fn play_pool_opponent_turns(&mut self, rewards: &mut [f32], dones: &mut [bool]) {
        const MAX_OPPONENT_TURNS: usize = 50;

        // First, handle baseline opponents (they have env_opponent_indices[i] = None)
        // Baselines use built-in bots via play_bot_turns(), not ONNX
        for i in 0..self.n_envs {
            if !dones[i]
                && self.env_opponent_indices[i].is_none()
                && self.envs[i].players.is_some()
                && self.envs[i].state.current_player == 1
                && !self.envs[i].state.is_game_over()
            {
                // This is a baseline env - play bot turns until agent's turn or game over
                let (bot_reward, bot_done) = self.envs[i].play_bot_turns();
                if bot_done {
                    dones[i] = true;
                    rewards[i] = bot_reward;
                }
            }
        }

        // Then handle ONNX opponents
        for _ in 0..MAX_OPPONENT_TURNS {
            // Clear grouping buffers (keeps capacity)
            for group in &mut self.opponent_groups {
                group.clear();
            }

            // Group envs by opponent index (only ONNX opponents, not baselines)
            for (i, env) in self.envs.iter().enumerate() {
                if !dones[i] && env.state.current_player == 1 && !env.state.is_game_over() {
                    if let Some(opp_idx) = self.env_opponent_indices[i] {
                        if opp_idx < self.opponent_groups.len() {
                            self.opponent_groups[opp_idx].push(i);
                        }
                    }
                }
            }

            // Check if any groups have work
            let has_work = self.opponent_groups.iter().any(|g| !g.is_empty());
            if !has_work {
                break;
            }

            // Process each opponent group by index (use min to handle eviction during batch)
            let pool_len = self.onnx_pool.len().min(self.opponent_groups.len());
            for opp_idx in 0..pool_len {
                if self.opponent_groups[opp_idx].is_empty() {
                    continue;
                }

                // Skip if this index corresponds to a baseline (no ONNX model)
                let onnx_model = match &mut self.onnx_pool[opp_idx] {
                    Some(m) => m,
                    None => continue,
                };

                // Collect env indices for this opponent
                let env_indices: Vec<usize> = self.opponent_groups[opp_idx].clone();

                // Collect observations and masks for this group
                let mut obs_batch: Vec<f32> =
                    Vec::with_capacity(env_indices.len() * OBSERVATION_SIZE);
                let mut mask_batch: Vec<bool> =
                    Vec::with_capacity(env_indices.len() * ACTION_SPACE_SIZE);

                for &i in &env_indices {
                    obs_batch.extend(get_observation_tensor(&self.envs[i].state, 1));
                    mask_batch.extend(get_action_mask(&self.envs[i].state));
                }

                // Run batched inference for this opponent
                let actions = onnx_model.predict_batch(
                    &obs_batch,
                    &mask_batch,
                    env_indices.len(),
                    &mut self.onnx_rng,
                );

                // Apply actions
                for (j, &env_idx) in env_indices.iter().enumerate() {
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
                            rewards[env_idx] = env.calculate_reward(0);
                        }
                    }))
                    .map_err(|_| {
                        warn!(
                            "Panic caught in environment {} during pool ONNX action! Marking done.",
                            env_idx
                        );
                        dones[env_idx] = true;
                        rewards[env_idx] = 0.0;
                    });
                }
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

    /// Play initial opponent turns for reset envs in pool mode
    #[cfg(feature = "onnx")]
    fn play_pool_initial_turns(&mut self, was_done: &[bool], observations: &mut Vec<f32>) {
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

        // First, handle baseline opponents (they have env_opponent_indices[i] = None)
        // Baselines use built-in bots via play_bot_turns(), not ONNX
        for &i in &needs_initial {
            if self.env_opponent_indices[i].is_none() && self.envs[i].players.is_some() {
                // This is a baseline env - play bot turns
                while self.envs[i].state.current_player == 1 && !self.envs[i].state.is_game_over() {
                    self.envs[i].play_bot_turns();
                }
            }
        }

        // Play ONNX opponent turns until agent's turn
        const MAX_TURNS: usize = 50;
        for _ in 0..MAX_TURNS {
            // Clear grouping buffers (keeps capacity)
            for group in &mut self.opponent_groups {
                group.clear();
            }

            // Group active envs by opponent index (only ONNX opponents, not baselines)
            for &i in &needs_initial {
                if self.envs[i].state.current_player == 1 && !self.envs[i].state.is_game_over() {
                    if let Some(opp_idx) = self.env_opponent_indices[i] {
                        if opp_idx < self.opponent_groups.len() {
                            self.opponent_groups[opp_idx].push(i);
                        }
                    }
                }
            }

            // Check if any groups have work
            let has_work = self.opponent_groups.iter().any(|g| !g.is_empty());
            if !has_work {
                break;
            }

            // Process each opponent group by index (use min to handle eviction during batch)
            let pool_len = self.onnx_pool.len().min(self.opponent_groups.len());
            for opp_idx in 0..pool_len {
                if self.opponent_groups[opp_idx].is_empty() {
                    continue;
                }

                // Skip if this index corresponds to a baseline (no ONNX model)
                let onnx_model = match &mut self.onnx_pool[opp_idx] {
                    Some(m) => m,
                    None => continue,
                };

                let active: Vec<usize> = self.opponent_groups[opp_idx].clone();

                let mut obs_batch: Vec<f32> = Vec::with_capacity(active.len() * OBSERVATION_SIZE);
                let mut mask_batch: Vec<bool> =
                    Vec::with_capacity(active.len() * ACTION_SPACE_SIZE);

                for &i in &active {
                    obs_batch.extend(get_observation_tensor(&self.envs[i].state, 1));
                    mask_batch.extend(get_action_mask(&self.envs[i].state));
                }

                let actions = onnx_model.predict_batch(
                    &obs_batch,
                    &mask_batch,
                    active.len(),
                    &mut self.onnx_rng,
                );

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
                    }))
                    .map_err(|_| {
                        warn!(
                            "Panic caught in environment {} during pool initial turn!",
                            env_idx
                        );
                    });
                }
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

    /// Get current state of a specific environment
    pub fn get_state(&self, env_idx: usize) -> State {
        self.envs[env_idx].state.clone()
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
