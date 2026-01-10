#!/usr/bin/env python3
import unittest
from unittest.mock import MagicMock, patch
from deckgym.pfsp_callback import PFSPCallback

class TestPFSPCallback(unittest.TestCase):
    def setUp(self):
        self.mock_env = MagicMock()
        self.mock_env.config = MagicMock()
        self.mock_env.config.pfsp_opponent_device = "trt"
        self.mock_env.config.pfsp_baseline_curriculum = [(0, ["v"])]
        self.mock_env.vec_game = MagicMock()
        
        self.callback = PFSPCallback(
            env=self.mock_env,
            n_envs=4,
            pool_size=5,
            add_to_pool_every_n_rollouts=10,
            verbose=1
        )
        # Mock SB3 model
        self.callback.model = MagicMock()
        self.callback._logger = MagicMock()
        
        # Ensure config returns values, not mocks, for comparisons
        self.mock_env.config.pfsp_min_winrate_to_add = 0.50
        self.mock_env.config.draw_reward = -0.5

    def test_on_training_start(self):
        with patch.object(self.callback, "_add_to_pool") as mock_add:
            self.callback._on_training_start()
            
            # Should have added initial model and baselines
            mock_add.assert_called_once()
            self.assertTrue(self.callback.pool_mode)
            # Rust bridge should have been called for initial opponents
            self.assertEqual(self.mock_env.vec_game.set_env_opponent.call_count, 4)

    def test_on_step_recording(self):
        # Setup env state
        self.callback.env_opponent_names = ["opp1", "opp1", "opp1", "opp1"]
        self.callback.skip_next_episode = [False] * 4
        
        # Mock env info
        self.callback.locals = {
            "infos": [
                {"episode": {"r": 1.0}},  # Agent win
                {"episode": {"r": -1.0}}, # Agent loss (opp win)
                {},                        # No episode info
                {"episode": {"r": -0.5}}  # Draw (default draw_reward is -0.5)
            ]
        }
        
        self.callback._on_step()
        
        # Results should be recorded for env 0, 1, 3
        expected = [("opp1", "agent_win"), ("opp1", "opp_win"), ("opp1", "draw")]
        self.assertEqual(self.callback.episode_results, expected)

    def test_rollout_end_logic(self):
        # Mock components
        self.callback.pool = MagicMock()
        self.callback.selector = MagicMock()
        self.callback.league_logger = MagicMock()
        self.callback.bridge = MagicMock()
        
        # Force a rollout end that triggers updates
        self.callback.rollout_count = 9
        self.callback.add_to_pool_every_n_rollouts = 10
        self.callback.episode_results = [("m1", "agent_win")] * 20 # Enough for update
        
        self.callback.selector.update_curriculum.return_value = ([], [])
        self.callback.league_logger.get_self_play_winrate.return_value = 0.8
        
        with patch.object(self.callback, "_add_to_pool") as mock_add:
            self.callback._on_rollout_end()
            
            # Rollout count should increment to 10
            self.assertEqual(self.callback.rollout_count, 10)
            # Should trigger curriculum update (every 10)
            self.callback.selector.update_curriculum.assert_called_once()
            # Should trigger stats update
            self.callback.pool.update_results.assert_called_once()
            self.callback.league_logger.log_metrics.assert_called_once()
            # Should trigger pool addition (every 10)
            mock_add.assert_called_once()
            # Should reset pool stats
            self.callback.pool.reset_statistics.assert_called_once()

if __name__ == "__main__":
    unittest.main()
