//! Vectorized Game Environment for High-Performance RL Training
//!
//! This module provides `VecGame`, a batched environment that manages multiple
//! games simultaneously and returns batched tensors. This reduces Python/Rust FFI
//! overhead from O(n_envs) calls per step to O(1) calls per batch step.

use rand::{rngs::StdRng, SeedableRng};

use crate::{
    actions::apply_action,
    deck::Deck,
    generate_possible_actions,
    players::{create_players, parse_player_code, Player},
    rl::{get_action_mask, get_indexed_actions, get_observation_tensor, ACTION_SPACE_SIZE, OBSERVATION_SIZE},
    state::{GameOutcome, State},
};

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
                parse_player_code("r").unwrap(),  // Player 0: random (agent controls)
                parse_player_code(opp_type).unwrap(),  // Player 1: bot
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
        
        // Recreate bot players if needed
        if let Some(opp_type) = &self.opponent_type {
            let player_codes = vec![
                parse_player_code("r").unwrap(),
                parse_player_code(opp_type).unwrap(),
            ];
            self.players = Some(create_players(deck_a_parsed, deck_b_parsed, player_codes));
        }
    }
    
    fn reset_with_new_decks(&mut self, deck_a: String, deck_b: String, seed: u64) {
        self.deck_a = deck_a;
        self.deck_b = deck_b;
        self.reset(seed);
    }
    
    fn get_observation(&self) -> Vec<f32> {
        get_observation_tensor(&self.state, self.state.current_player)
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
    
    /// Calculate score-based reward from agent's perspective
    fn calculate_reward(&self, agent_player: usize) -> f32 {
        match self.state.winner {
            Some(GameOutcome::Win(winner)) => {
                let my_points = self.state.points[agent_player] as f32;
                let opp_points = self.state.points[1 - agent_player] as f32;
                let point_diff = my_points - opp_points;
                
                if winner == agent_player {
                    1.0 + (point_diff / 6.0)
                } else {
                    -1.0 + (point_diff / 6.0)
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
        
        for env in self.envs.iter_mut() {
            self.seed_counter += 1;
            env.reset(self.seed_counter);
            
            // If using bots, play bot turns first if bot goes first
            if env.opponent_type.is_some() {
                env.play_bot_turns();
            }
            
            observations.extend(env.get_observation());
        }
        
        observations
    }
    
    /// Reset a single environment with new decks
    pub fn reset_single(&mut self, env_idx: usize, deck_a: String, deck_b: String) {
        self.seed_counter += 1;
        self.envs[env_idx].reset_with_new_decks(deck_a, deck_b, self.seed_counter);
        
        // If using bots, play bot turns first if bot goes first
        if self.envs[env_idx].opponent_type.is_some() {
            self.envs[env_idx].play_bot_turns();
        }
    }
    
    /// Get batched action masks for all environments
    /// Returns flattened masks (n_envs × action_size)
    pub fn get_action_masks(&self) -> Vec<bool> {
        let mut masks = Vec::with_capacity(self.n_envs * ACTION_SPACE_SIZE);
        
        for env in &self.envs {
            masks.extend(env.get_action_mask());
        }
        
        masks
    }
    
    /// Get batched observations for all environments
    /// Returns flattened observations (n_envs × obs_size)
    pub fn get_observations(&self) -> Vec<f32> {
        let mut observations = Vec::with_capacity(self.n_envs * OBSERVATION_SIZE);
        
        for env in &self.envs {
            observations.extend(env.get_observation());
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
        assert_eq!(actions.len(), self.n_envs, "Actions length must match n_envs");
        
        let mut observations = Vec::with_capacity(self.n_envs * OBSERVATION_SIZE);
        let mut rewards = Vec::with_capacity(self.n_envs);
        let mut dones = Vec::with_capacity(self.n_envs);
        let mut terminal_observations = Vec::new();
        
        for (i, (env, &action)) in self.envs.iter_mut().zip(actions.iter()).enumerate() {
            // Step the agent's action
            let (mut reward, mut done) = env.step(action);
            
            // If using bots and game not over, play bot turns
            if !done && env.opponent_type.is_some() {
                let (bot_reward, bot_done) = env.play_bot_turns();
                if bot_done {
                    done = true;
                    reward = bot_reward;
                }
            }
            
            if done {
                // Store terminal observation before auto-reset
                terminal_observations.push((i, env.get_observation()));
                
                // Auto-reset this environment
                self.seed_counter += 1;
                env.reset(self.seed_counter);
                
                // Play bot turns if bot goes first in new game
                if env.opponent_type.is_some() {
                    env.play_bot_turns();
                }
            }
            
            // Get observation (from new episode if reset happened)
            observations.extend(env.get_observation());
            rewards.push(reward);
            dones.push(done);
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
    
    /// Update deck pairs for auto-reset (called from Python when curriculum changes)
    pub fn set_deck_pairs(&mut self, deck_pairs: Vec<(String, String)>) {
        assert_eq!(deck_pairs.len(), self.n_envs, "Deck pairs length must match n_envs");
        
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
2 Pinsir A1 080"#.to_string()
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
        let deck_pairs = vec![
            (test_deck_string(), test_deck_string()),
        ];
        
        let vec_game = VecGame::new(deck_pairs, 42, None);
        let masks = vec_game.get_action_masks();
        
        assert_eq!(masks.len(), ACTION_SPACE_SIZE);
        // At least one action should be valid
        assert!(masks.iter().any(|&m| m));
    }
}
