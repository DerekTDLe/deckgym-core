#!/usr/bin/env python3
"""
CurriculumManager - Progressive training curriculum for Pokemon TCG Pocket RL.

Manages both opponent difficulty and deck complexity in a unified system.
Transitions are based on win rate thresholds rather than fixed timesteps.

Stages:
    1. warmup:   e2 + simple decks → learn basic mechanics
    2. meta:     e2 + meta decks → learn competitive strategies
    3. advanced: e3 + meta decks → push beyond e2 level
    4. mastery:  self-play → emergent strategies and refinement
"""

import random
from dataclasses import dataclass, field
from typing import Optional, Callable, List

from deckgym.deck_loader import MetaDeckLoader


@dataclass
class CurriculumStage:
    """Definition of a training stage."""
    name: str
    opponent: str           # "e2", "e3", "self"
    deck_source: str        # "simple", "meta"
    win_rate_threshold: Optional[float]  # None = final stage
    eval_games: int = 50    # Games for win rate evaluation
    
    def __repr__(self) -> str:
        threshold = f"{self.win_rate_threshold:.0%}" if self.win_rate_threshold else "∞"
        return f"Stage({self.name}: vs {self.opponent}, {self.deck_source} decks, threshold={threshold})"


# Default curriculum stages
DEFAULT_STAGES = [
    CurriculumStage("warmup",   "e2", "simple", 0.55, 50),  # Lower threshold for faster transition
    CurriculumStage("meta",     "e2", "meta",   0.55, 50),
    CurriculumStage("advanced", "e3", "meta",   0.50, 30),
    CurriculumStage("mastery",  "self", "meta", None, 0),
]


class CurriculumManager:
    """
    Manages training curriculum with automatic stage transitions.
    
    Usage:
        curriculum = CurriculumManager(simple_loader, meta_loader)
        
        # In training loop:
        deck_a, deck_b = curriculum.sample_pair()
        opponent_code = curriculum.current_stage.opponent
        
        # After evaluation:
        if curriculum.check_advancement(win_rate):
            print(f"Advanced to {curriculum.current_stage.name}!")
    """
    
    def __init__(
        self,
        simple_loader: MetaDeckLoader,
        meta_loader: MetaDeckLoader,
        stages: List[CurriculumStage] = None,
        on_stage_change: Optional[Callable[[CurriculumStage], None]] = None,
    ):
        """
        Initialize curriculum manager.
        
        Args:
            simple_loader: Deck loader for simple/beginner decks
            meta_loader: Deck loader for competitive meta decks
            stages: List of curriculum stages (default: warmup → meta → advanced → mastery)
            on_stage_change: Callback when stage changes
        """
        self.simple_loader = simple_loader
        self.meta_loader = meta_loader
        self.stages = stages or DEFAULT_STAGES
        self.on_stage_change = on_stage_change
        
        self._current_stage_idx = 0
        self._win_history: List[float] = []
        self._stage_stats = {s.name: {"evals": 0, "best_wr": 0.0} for s in self.stages}
    
    @property
    def current_stage(self) -> CurriculumStage:
        """Get current training stage."""
        return self.stages[self._current_stage_idx]
    
    @property
    def current_stage_idx(self) -> int:
        """Get current stage index."""
        return self._current_stage_idx
    
    @property
    def is_self_play(self) -> bool:
        """Check if current stage uses self-play."""
        return self.current_stage.opponent == "self"
    
    @property 
    def is_final_stage(self) -> bool:
        """Check if at final stage."""
        return self._current_stage_idx >= len(self.stages) - 1
    
    def get_deck_loader(self) -> MetaDeckLoader:
        """Get the deck loader for current stage."""
        if self.current_stage.deck_source == "simple":
            return self.simple_loader
        return self.meta_loader
    
    def sample_deck(self) -> str:
        """Sample a deck appropriate for current stage."""
        return self.get_deck_loader().sample_deck()
    
    def sample_pair(self) -> tuple[str, str]:
        """Sample a pair of decks for a game."""
        loader = self.get_deck_loader()
        return loader.sample_deck(), loader.sample_deck()
    
    def check_advancement(self, win_rate: float) -> bool:
        """
        Check if agent should advance to next stage based on win rate.
        
        Args:
            win_rate: Current win rate against stage opponent (0.0-1.0)
            
        Returns:
            True if advanced to next stage, False otherwise
        """
        stage = self.current_stage
        
        # Track stats
        self._win_history.append(win_rate)
        stats = self._stage_stats[stage.name]
        stats["evals"] += 1
        stats["best_wr"] = max(stats["best_wr"], win_rate)
        
        # Check threshold
        if stage.win_rate_threshold is None:
            return False  # Final stage, no advancement
        
        if win_rate >= stage.win_rate_threshold:
            return self._advance_stage()
        
        return False
    
    def _advance_stage(self) -> bool:
        """Advance to next stage."""
        if self.is_final_stage:
            return False
        
        old_stage = self.current_stage
        self._current_stage_idx += 1
        new_stage = self.current_stage
        
        print(f"\n{'='*60}")
        print(f"[CURRICULUM] Stage transition: {old_stage.name} → {new_stage.name}")
        print(f"  Opponent: {old_stage.opponent} → {new_stage.opponent}")
        print(f"  Decks: {old_stage.deck_source} → {new_stage.deck_source}")
        print(f"{'='*60}\n")
        
        if self.on_stage_change:
            self.on_stage_change(new_stage)
        
        return True
    
    def force_stage(self, stage_name: str) -> bool:
        """Force transition to a specific stage by name."""
        for idx, stage in enumerate(self.stages):
            if stage.name == stage_name:
                self._current_stage_idx = idx
                print(f"[CURRICULUM] Forced to stage: {stage.name}")
                return True
        return False
    
    def get_status(self) -> dict:
        """Get current curriculum status for logging."""
        stage = self.current_stage
        return {
            "stage": stage.name,
            "stage_idx": self._current_stage_idx,
            "opponent": stage.opponent,
            "deck_source": stage.deck_source,
            "threshold": stage.win_rate_threshold,
            "is_self_play": self.is_self_play,
            "stats": self._stage_stats,
        }
    
    def __repr__(self) -> str:
        return f"CurriculumManager(stage={self.current_stage.name}, idx={self._current_stage_idx}/{len(self.stages)})"


if __name__ == "__main__":
    # Test curriculum
    print("Testing CurriculumManager...")
    
    # Mock loaders
    class MockLoader:
        def __init__(self, name):
            self.name = name
        def sample_deck(self):
            return f"deck_from_{self.name}"
    
    simple = MockLoader("simple")
    meta = MockLoader("meta")
    
    curriculum = CurriculumManager(simple, meta)
    
    print(f"Initial: {curriculum}")
    print(f"Current stage: {curriculum.current_stage}")
    print(f"Sample deck: {curriculum.sample_deck()}")
    
    # Simulate progression
    for wr in [0.50, 0.60, 0.65, 0.70]:  # Should advance at 0.65
        advanced = curriculum.check_advancement(wr)
        print(f"WR={wr:.0%} → Advanced: {advanced}, Stage: {curriculum.current_stage.name}")
    
    print("\n✓ CurriculumManager works!")
