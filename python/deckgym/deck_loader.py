#!/usr/bin/env python3
"""
MetaDeckLoader - Load competitive decks from JSON for RL training.

Supports two JSON formats:
1. Simple format (simple_deck.json): flat list of decks
2. Meta format (meta_deck.json): archetypes with nested decks
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DeckInfo:
    """Information about a deck."""
    archetype: str
    win_rate: float
    energy: str
    deck_string: str
    player: Optional[str] = None


class MetaDeckLoader:
    """
    Load and sample competitive decks from JSON data.
    
    Supports two JSON formats:
    - Simple: List of decks with energy_type and cards
    - Meta: Archetypes structure with nested decks and win rates
    
    Usage:
        loader = MetaDeckLoader("simple_deck.json")  # or meta_deck.json
        deck_str = loader.sample_deck()
        deck_str = loader.sample_deck(weighted=True)  # weighted by win rate
    """
    
    def __init__(self, json_path: str):
        """
        Load decks from JSON file.
        
        Args:
            json_path: Path to the deck JSON file
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.decks: list[DeckInfo] = []
        self._load_decks()
        
        archetypes = len(set(d.archetype for d in self.decks))
        print(f"Loaded {len(self.decks)} decks from {archetypes} archetypes")
    
    def _load_decks(self):
        """Parse all decks from the JSON data (auto-detect format)."""
        if isinstance(self.data, list):
            # Simple format: flat list of decks
            self._load_simple_format()
        elif "archetypes" in self.data:
            # Meta format: archetypes with nested decks
            self._load_meta_format()
        else:
            raise ValueError("Unknown JSON format: expected list or archetypes structure")
    
    def _load_simple_format(self):
        """Load decks from simple format (flat list)."""
        for deck in self.data:
            name = deck.get("name", "Unknown")
            energy = deck.get("energy_type", "")
            cards = deck.get("cards", [])
            
            if not cards:
                continue
            
            deck_string = self._cards_to_string(cards, energy)
            
            self.decks.append(DeckInfo(
                archetype=name,
                win_rate=50.0,  # No win rate in simple format
                energy=energy,
                deck_string=deck_string,
                player=None
            ))
    
    def _load_meta_format(self):
        """Load decks from meta format (archetypes structure)."""
        for archetype in self.data.get("archetypes", []):
            archetype_name = archetype.get("name", "Unknown")
            win_rate = archetype.get("win_rate", 50.0)
            
            for deck in archetype.get("decks", []):
                energy = deck.get("energy", "")
                cards = deck.get("cards", [])
                
                if not cards:
                    continue
                
                deck_string = self._cards_to_string(cards, energy)
                
                self.decks.append(DeckInfo(
                    archetype=archetype_name,
                    win_rate=win_rate,
                    energy=energy,
                    deck_string=deck_string,
                    player=deck.get("player")
                ))
    
    def _cards_to_string(self, cards: list[dict], energy: str = "") -> str:
        """
        Convert deck cards to deck string format.
        
        Args:
            cards: List of {"count": N, "name": "...", "set": "...", "number": "..."}
            energy: Energy type string (e.g., "Lightning" or "Fire/Darkness")
        
        Returns:
            Deck string in the format expected by deckgym
        """
        lines = []
        
        # Add energy line if specified
        if energy:
            # Handle dual energy (e.g., "Fire/Darkness" -> "Energy: Fire")
            primary_energy = energy.split("/")[0].strip()
            lines.append(f"Energy: {primary_energy}")
        
        # Add each card
        for card in cards:
            count = card.get("count", 1)
            name = card.get("name", "Unknown")
            set_code = card.get("set", "A1")
            number = card.get("number", "001")
            
            # Pad number to 3 digits
            if isinstance(number, int):
                number = f"{number:03d}"
            elif isinstance(number, str) and number.isdigit():
                number = number.zfill(3)
            
            lines.append(f"{count} {name} {set_code} {number}")
        
        return "\n".join(lines)
    
    def sample_deck(self, weighted: bool = False) -> str:
        """
        Sample a random deck.
        
        Args:
            weighted: If True, weight sampling by win rate
        
        Returns:
            Deck string for use with deckgym.Game.from_deck_strings()
        """
        if not self.decks:
            raise ValueError("No decks loaded")
        
        if weighted:
            weights = [max(d.win_rate, 1.0) for d in self.decks]
            deck = random.choices(self.decks, weights=weights, k=1)[0]
        else:
            deck = random.choice(self.decks)
        
        return deck.deck_string
    
    def sample_deck_info(self, weighted: bool = False) -> DeckInfo:
        """
        Sample a random deck with full info.
        
        Args:
            weighted: If True, weight sampling by win rate
        
        Returns:
            DeckInfo object with archetype, win_rate, etc.
        """
        if not self.decks:
            raise ValueError("No decks loaded")
        
        if weighted:
            weights = [max(d.win_rate, 1.0) for d in self.decks]
            return random.choices(self.decks, weights=weights, k=1)[0]
        else:
            return random.choice(self.decks)
    
    def get_decks_by_archetype(self, archetype: str) -> list[DeckInfo]:
        """Get all decks of a specific archetype."""
        return [d for d in self.decks if d.archetype.lower() == archetype.lower()]
    
    def get_archetypes(self) -> list[str]:
        """Get list of all archetypes."""
        return list(set(d.archetype for d in self.decks))
    
    def filter_by_win_rate(self, min_rate: float = 0.0, max_rate: float = 100.0) -> list[DeckInfo]:
        """Get decks within a win rate range."""
        return [d for d in self.decks if min_rate <= d.win_rate <= max_rate]
    
    def __len__(self) -> int:
        return len(self.decks)


class CurriculumDeckLoader:
    """
    Manages curriculum learning with two deck pools (simple → meta).
    
    Starts with simple decks and progressively introduces meta decks
    based on training progress.
    
    Usage:
        loader = CurriculumDeckLoader("simple_deck.json", "meta_deck.json")
        loader.set_difficulty(0.0)  # 100% simple decks
        loader.set_difficulty(0.5)  # 50% simple, 50% meta
        loader.set_difficulty(1.0)  # 100% meta decks
    """
    
    def __init__(self, simple_path: str, meta_path: str):
        """
        Load both deck pools.
        
        Args:
            simple_path: Path to simple_deck.json
            meta_path: Path to meta_deck.json
        """
        print("Loading simple decks...")
        self.simple_loader = MetaDeckLoader(simple_path)
        print("Loading meta decks...")
        self.meta_loader = MetaDeckLoader(meta_path)
        
        self.difficulty = 0.0  # 0 = all simple, 1 = all meta
    
    def set_difficulty(self, difficulty: float):
        """
        Set curriculum difficulty.
        
        Args:
            difficulty: 0.0 = 100% simple, 1.0 = 100% meta
        """
        self.difficulty = max(0.0, min(1.0, difficulty))
    
    def sample_deck(self, weighted: bool = False) -> str:
        """
        Sample a deck based on current difficulty.
        
        At difficulty=0: 100% simple decks
        At difficulty=0.5: 50% simple, 50% meta
        At difficulty=1: 100% meta decks
        """
        if random.random() < self.difficulty:
            return self.meta_loader.sample_deck(weighted=weighted)
        else:
            return self.simple_loader.sample_deck(weighted=weighted)
    
    def sample_pair(self, weighted: bool = False) -> tuple[str, str]:
        """
        Sample two decks for self-play.
        
        Returns:
            (deck_a_str, deck_b_str)
        """
        return self.sample_deck(weighted), self.sample_deck(weighted)
    
    def __len__(self) -> int:
        return len(self.simple_loader) + len(self.meta_loader)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python deck_loader.py <path_to_json> [path_to_meta_json]")
        print("\nExamples:")
        print("  python deck_loader.py simple_deck.json")
        print("  python deck_loader.py simple_deck.json meta_deck.json")
        sys.exit(1)
    
    if len(sys.argv) == 2:
        # Single loader mode
        loader = MetaDeckLoader(sys.argv[1])
        print(f"\nArchetypes: {loader.get_archetypes()[:5]}...")
        
        print("\n--- Sample deck ---")
        deck_info = loader.sample_deck_info()
        print(f"Archetype: {deck_info.archetype}")
        print(f"Win rate: {deck_info.win_rate}%")
        print(f"Energy: {deck_info.energy}")
        print(f"\nDeck string:\n{deck_info.deck_string}")
    else:
        # Curriculum mode
        curriculum = CurriculumDeckLoader(sys.argv[1], sys.argv[2])
        
        print("\n--- Testing curriculum ---")
        curriculum.set_difficulty(0.0)
        print(f"Difficulty 0.0: {curriculum.sample_deck()[:50]}...")
        
        curriculum.set_difficulty(1.0)
        print(f"Difficulty 1.0: {curriculum.sample_deck()[:50]}...")
