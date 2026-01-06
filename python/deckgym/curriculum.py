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

from collections import deque
from dataclasses import dataclass
from typing import Optional, Callable, List

from deckgym.deck_loader import MetaDeckLoader


class WinRateTracker:
    """
    Sliding window tracker for continuous win rate monitoring.
    
    Uses a fixed-size deque to maintain the last N episode results,
    providing a smoothed win rate estimate without separate evaluation runs.
    """
    
    def __init__(self, window_size: int = 270):
        """
        Initialize tracker with given window size.
        
        Args:
            window_size: Number of episodes to track
        """
        self.window_size = window_size
        self.results = deque(maxlen=window_size)
    
    def record(self, won: bool) -> None:
        """Record an episode result (True = win, False = loss)."""
        self.results.append(1 if won else 0)
    
    @property
    def win_rate(self) -> float:
        """Get current win rate (0.5 if no data yet)."""
        if not self.results:
            return 0.5
        return sum(self.results) / len(self.results)
    
    @property
    def count(self) -> int:
        """Number of episodes recorded in current window."""
        return len(self.results)
    
    @property
    def is_reliable(self) -> bool:
        """True if we have enough data for reliable estimates (>= 50 episodes)."""
        return len(self.results) >= 50
    
    def reset(self) -> None:
        """Clear all recorded results (use on stage transition)."""
        self.results.clear()


@dataclass
class CurriculumStage:
    """Definition of a training stage."""
    name: str
    opponent: str           # "e2", "e3", "self"
    deck_source: str        # "simple", "meta"
    win_rate_threshold: Optional[float]  # None = final stage
    
    def __repr__(self) -> str:
        threshold = f"{self.win_rate_threshold:.0%}" if self.win_rate_threshold else "∞"
        return f"Stage({self.name}: vs {self.opponent}, {self.deck_source} decks, threshold={threshold})"


# Default curriculum stages
DEFAULT_STAGES = [
    CurriculumStage("warmup",   "e2", "simple", 0.40),  # Reduced from 0.55 - faster progression
    CurriculumStage("meta",     "e2", "meta",   0.55),  # Keep at 0.55 - important milestone
    CurriculumStage("advanced", "e3", "meta",   0.50),  # Reduced from 0.55 - e3 is harder
    CurriculumStage("mastery",  "self", "meta", None),
]


class CurriculumManager:
    """
    Manages training curriculum with automatic stage transitions.
    
    Uses continuous win rate tracking from training episodes instead of
    separate evaluation runs. Stage advancement is checked at each rollout end.
    
    Usage:
        curriculum = CurriculumManager(simple_loader, meta_loader)
        
        # After each episode:
        curriculum.record_episode(won=(reward > 0))
        
        # At rollout end:
        if curriculum.check_advancement():
            update_opponent_type(curriculum.current_stage.opponent)
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
        self._tracker = WinRateTracker(window_size=270) # Agent WR%>=55% with ~95% Confidence level 
        self._stage_stats = {s.name: {"checks": 0, "best_wr": 0.0} for s in self.stages}
    
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
    
    def record_episode(self, won: bool) -> None:
        """
        Record a training episode result for continuous win rate tracking.
        
        Call this after each episode ends during training.
        
        Args:
            won: True if agent won, False if lost
        """
        self._tracker.record(won)
    
    @property
    def win_rate(self) -> float:
        """Current tracked win rate."""
        return self._tracker.win_rate
    
    @property
    def episode_count(self) -> int:
        """Number of episodes tracked in current window."""
        return self._tracker.count
    
    def check_advancement(self) -> bool:
        """
        Check if agent should advance to next stage based on tracked win rate.
        
        Uses the continuous win rate from training episodes.
        Requires at least 50 episodes for reliable estimates.
        
        Returns:
            True if advanced to next stage, False otherwise
        """
        stage = self.current_stage
        
        # Need enough data for reliable estimate
        if not self._tracker.is_reliable:
            return False
        
        win_rate = self._tracker.win_rate
        
        # Track stats
        stats = self._stage_stats[stage.name]
        stats["checks"] += 1
        stats["best_wr"] = max(stats["best_wr"], win_rate)
        
        # Check threshold (None = final stage)
        if stage.win_rate_threshold is None:
            return False
        
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
        
        # Reset tracker for fresh statistics against new opponent
        self._tracker.reset()
        
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
    
    # Simulate progression with continuous tracking
    import random
    random.seed(42)
    for episode in range(100):
        curriculum.record_episode(won=(random.random() < 0.60))  # 60% win rate
    
    advanced = curriculum.check_advancement()
    print(f"After 100 eps, WR={curriculum.win_rate:.1%} → Advanced: {advanced}")
    
    print("\n✓ CurriculumManager works!")
