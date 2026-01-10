#!/usr/bin/env python3
import unittest
from unittest.mock import MagicMock
import numpy as np
from deckgym.league.logger import LeagueLogger
from deckgym.league.pool import OpponentPool

class TestLeagueLogger(unittest.TestCase):
    def setUp(self):
        self.pool = MagicMock(spec=OpponentPool)
        self.sb3_logger = MagicMock()
        self.league_logger = LeagueLogger(pool=self.pool, logger=self.sb3_logger, verbose=1)

    def test_is_fair_baseline(self):
        self.assertTrue(self.league_logger._is_fair_baseline("er"))
        self.assertTrue(self.league_logger._is_fair_baseline("et"))
        self.assertFalse(self.league_logger._is_fair_baseline("e2"))
        self.assertFalse(self.league_logger._is_fair_baseline("m1"))
        self.assertTrue(self.league_logger._is_fair_baseline("v"))

    def test_calculate_winrate(self):
        self.pool.get_data.side_effect = lambda name: {
            "m1": {"wins": 10, "losses": 0, "draws": 0}, # WR (opp losses/total) = 0.0
            "m2": {"wins": 0, "losses": 10, "draws": 0}  # WR (opp losses/total) = 1.0
        }.get(name)
        
        wr = self.league_logger.calculate_winrate(["m1", "m2"])
        self.assertEqual(wr, 0.5)

    def test_get_self_play_winrate(self):
        self.pool.opponents = {
            "m1": {"wins": 0, "losses": 10, "draws": 0, "is_baseline": False},
            "b1": {"wins": 10, "losses": 0, "draws": 0, "is_baseline": True}
        }
        self.pool.get_data.side_effect = lambda name: self.pool.opponents.get(name)
        
        wr = self.league_logger.get_self_play_winrate()
        self.assertEqual(wr, 1.0) # Only m1 is considered

    def test_get_baseline_winrate(self):
        self.pool.opponents = {
            "b1": {"wins": 10, "losses": 0, "draws": 0, "is_baseline": True, "baseline_code": "e2"}, # Unfair
            "b2": {"wins": 0, "losses": 10, "draws": 0, "is_baseline": True, "baseline_code": "er"}, # Fair
            "m1": {"wins": 5, "losses": 5, "draws": 0, "is_baseline": False}
        }
        self.pool.get_data.side_effect = lambda name: self.pool.opponents.get(name)
        
        # Fair only
        wr_fair = self.league_logger.get_baseline_winrate(only_fair=True)
        self.assertEqual(wr_fair, 1.0) # Only b2 (er) is considered
        
        # All baselines
        wr_all = self.league_logger.get_baseline_winrate(only_fair=False)
        self.assertEqual(wr_all, 0.5) # b1 and b2 considered

    def test_log_metrics(self):
        # Mock various winrates
        self.league_logger.get_self_play_winrate = MagicMock(return_value=0.7)
        self.league_logger.get_baseline_winrate = MagicMock(return_value=0.5)
        self.league_logger.calculate_winrate = MagicMock(side_effect=lambda names: 0.4 if "baseline_e2" in names else 0.6)
        
        self.pool.opponents = {"baseline_e2": {}, "baseline_er": {}}
        
        self.league_logger.log_metrics(rollout_count=100)
        
        # Check SB3 logger records
        self.sb3_logger.record.assert_any_call("pfsp/winrate_self_play", 0.7)
        self.sb3_logger.record.assert_any_call("pfsp/winrate_baselines_fair", 0.5)
        self.sb3_logger.record.assert_any_call("pfsp/winrate_e2", 0.4)
        self.sb3_logger.record.assert_any_call("pfsp/winrate_er", 0.6)
        self.sb3_logger.record.assert_any_call("pfsp/winrate_global_fair", 0.6) # mean(0.7, 0.5)

if __name__ == "__main__":
    unittest.main()
