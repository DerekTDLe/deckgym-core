#!/usr/bin/env python3
import unittest
from unittest.mock import MagicMock
import numpy as np
from deckgym.league.selector import OpponentSelector
from deckgym.league.pool import OpponentPool

class TestOpponentSelector(unittest.TestCase):
    def setUp(self):
        self.pool = MagicMock(spec=OpponentPool)
        self.curriculum = [
            (0, ["v"]),
            (100, ["v", "e2"]),
            (200, ["e2", "er"])
        ]
        self.selector = OpponentSelector(
            pool=self.pool,
            priority_exponent=1.0,
            baseline_curriculum=self.curriculum,
            baseline_max_allocation=0.25,
            n_envs=8
        )

    def test_baseline_envs_count(self):
        # 8 envs * 0.25 = 2
        self.assertEqual(self.selector.baseline_envs_count, 2)

    def test_update_curriculum(self):
        # Step 50 -> stage 0
        added, removed = self.selector.update_curriculum(50)
        self.assertEqual(self.selector.current_baselines, ["v"])
        self.assertEqual(added, ["v"])
        self.assertEqual(removed, [])

        # Step 150 -> stage 1
        added, removed = self.selector.update_curriculum(150)
        self.assertEqual(set(self.selector.current_baselines), {"v", "e2"})
        self.assertEqual(added, ["e2"])
        self.assertEqual(removed, [])

        # Step 250 -> stage 2
        added, removed = self.selector.update_curriculum(250)
        self.assertEqual(set(self.selector.current_baselines), {"e2", "er"})
        self.assertEqual(added, ["er"])
        self.assertEqual(removed, ["v"])

    def test_get_priorities(self):
        self.pool.opponents = {
            "m1": {"wins": 10, "losses": 0, "draws": 0, "is_baseline": False}, # WR ~ 0.91
            "m2": {"wins": 0, "losses": 10, "draws": 0, "is_baseline": False}, # WR ~ 0.08
            "b1": {"wins": 5, "losses": 5, "draws": 0, "is_baseline": True}    # WR ~ 0.5
        }
        
        # Test with exp = 1.0
        prios = self.selector.get_priorities(exclude_baselines=True)
        self.assertIn("m1", prios)
        self.assertIn("m2", prios)
        self.assertNotIn("b1", prios)
        self.assertAlmostEqual(prios["m1"], 11/12)
        self.assertAlmostEqual(prios["m2"], 1/12)

        # Test with exp = 2.0
        self.selector.priority_exponent = 2.0
        prios = self.selector.get_priorities(exclude_baselines=True)
        self.assertAlmostEqual(prios["m1"], (11/12)**2)
        self.assertAlmostEqual(prios["m2"], (1/12)**2)

    def test_select_opponent_pfsp(self):
        self.pool.opponents = {
            "m1": {"wins": 100, "losses": 0, "draws": 0, "is_baseline": False},
            "m2": {"wins": 0, "losses": 100, "draws": 0, "is_baseline": False}
        }
        
        # With high wins, m1 should be selected much more often than m2
        selections = [self.selector.select_opponent_pfsp() for _ in range(100)]
        self.assertGreater(selections.count("m1"), selections.count("m2"))

    def test_select_for_env_assignment(self):
        self.selector.current_baselines = ["e2"]
        # Envs 0, 1 should get baseline
        # Envs 2-7 should get PFSP
        
        self.pool.opponents = {
            "baseline_e2": {"is_baseline": True},
            "m1": {"wins": 0, "losses": 0, "draws": 0, "is_baseline": False}
        }
        
        self.assertEqual(self.selector.select_for_env(0), "baseline_e2")
        self.assertEqual(self.selector.select_for_env(1), "baseline_e2")
        self.assertEqual(self.selector.select_for_env(2), "m1")

if __name__ == "__main__":
    unittest.main()
