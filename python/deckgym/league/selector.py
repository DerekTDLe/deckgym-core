#!/usr/bin/env python3
"""
Opponent Selection Strategy for League Training.

Encapsulates the prioritization (PFSP) and curriculum-based selection logic.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from deckgym.league.pool import OpponentPool


class OpponentSelector:
    """
    Handles the logic for selecting opponents for each training environment.
    
    Responsibilities:
    - Calculating PFSP priorities based on winrates.
    - Managing baseline curriculum stages.
    - Assigning specific opponents to environments.
    """

    def __init__(
        self,
        pool: OpponentPool,
        priority_exponent: float,
        baseline_curriculum: List[Tuple[int, List[str]]],
        baseline_max_allocation: float,
        n_envs: int,
        brutal_resume: bool = False,
    ):
        self.pool = pool
        self.priority_exponent = priority_exponent
        self.baseline_curriculum = baseline_curriculum
        self.baseline_max_allocation = baseline_max_allocation
        self.n_envs = n_envs
        self.brutal_resume = brutal_resume
        
        self.baseline_envs_count = max(1, int(n_envs * baseline_max_allocation))
        self.current_baselines: List[str] = []

    def update_curriculum(self, current_step: int) -> Tuple[List[str], List[str]]:
        """
        Update baseline curriculum based on current training step.
        Returns: (new_baseline_codes, removed_baseline_codes)
        """
        # Find the appropriate curriculum stage
        new_baselines = []
        for step_threshold, baselines in sorted(
            self.baseline_curriculum, key=lambda x: x[0], reverse=True
        ):
            if self.brutal_resume or current_step >= step_threshold:
                new_baselines = baselines
                break
        
        if set(new_baselines) == set(self.current_baselines):
            return [], []
            
        removed = list(set(self.current_baselines) - set(new_baselines))
        added = list(set(new_baselines) - set(self.current_baselines))
        
        self.current_baselines = new_baselines
        return added, removed

    def get_priorities(self, exclude_baselines: bool = False) -> Dict[str, float]:
        """Calculate PFSP priorities for opponents in the pool."""
        pool_data = self.pool.opponents
        priorities = {}
        
        for name, data in pool_data.items():
            if exclude_baselines and data.get("is_baseline", False):
                continue
                
            total = data["wins"] + data["losses"] + data["draws"]
            # Laplace smoothing: (wins + 1) / (total + 2)
            opp_winrate = (data["wins"] + 1) / (total + 2)
            priorities[name] = float(opp_winrate**self.priority_exponent)
            
        return priorities

    def select_opponent_pfsp(self, exclude_baselines: bool = False) -> Optional[str]:
        """Select an opponent using PFSP probabilities."""
        priorities = self.get_priorities(exclude_baselines=exclude_baselines)
        
        if not priorities:
            # Fallback to any opponent if filtered pool is empty
            if exclude_baselines:
                priorities = self.get_priorities(exclude_baselines=False)
            if not priorities:
                return None

        total_priority = sum(priorities.values())
        names = list(priorities.keys())
        
        if total_priority <= 0:
            return np.random.choice(names)

        probs = [priorities[name] / total_priority for name in names]
        return np.random.choice(names, p=probs)

    def select_for_env(self, env_idx: int) -> str:
        """
        Select an opponent for a specific environment index.
        First N envs are reserved for baselines.
        """
        baseline_names = [f"baseline_{bl}" for bl in self.current_baselines]
        
        if env_idx < self.baseline_envs_count and baseline_names:
            # Reserved for baselines - uniform selection
            return np.random.choice(baseline_names)
        else:
            # PFSP selection (preferring self-play checkpoints for diversity)
            selected = self.select_opponent_pfsp(exclude_baselines=True)
            if selected is None:
                selected = self.select_opponent_pfsp(exclude_baselines=False)
            return selected
