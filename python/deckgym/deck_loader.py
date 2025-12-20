#!/usr/bin/env python3
"""
MetaDeckLoader - Load competitive decks from JSON for RL training.

This module loads deck data from the Limitless TCG scraped JSON and converts
them into deck strings usable by the deckgym simulator.
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
    Load and sample competitive decks from Limitless TCG JSON data.
    
    Usage:
        loader = MetaDeckLoader("pokemon_pocket_complete.json")
        deck_str = loader.sample_deck()  # Random deck
        deck_str = loader.sample_deck(weighted=True)  # Weighted by win rate
    """
    
    def __init__(self, json_path: str):
        """
        Load decks from JSON file.
        
        Args:
            json_path: Path to the pokemon_pocket_complete.json file
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.decks: list[DeckInfo] = []
        self._load_decks()
        
        print(f"Loaded {len(self.decks)} decks from {len(self.data['archetypes'])} archetypes")
    
    def _load_decks(self):
        """Parse all decks from the JSON data."""
        for archetype in self.data.get("archetypes", []):
            archetype_name = archetype.get("archetype_name", "Unknown")
            win_rate = archetype.get("win_rate", 50.0)
            
            for deck in archetype.get("decks", []):
                energy = deck.get("energy", "")
                composition = deck.get("composition", [])
                
                if not composition:
                    continue
                
                deck_string = self._composition_to_string(composition, energy)
                
                self.decks.append(DeckInfo(
                    archetype=archetype_name,
                    win_rate=win_rate,
                    energy=energy,
                    deck_string=deck_string,
                    player=deck.get("player")
                ))
    
    def _composition_to_string(self, composition: list[dict], energy: str = "") -> str:
        """
        Convert deck composition to deck string format.
        
        Args:
            composition: List of {"count": N, "name": "...", "set": "...", "number": "..."}
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
        for card in composition:
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


if __name__ == "__main__":
    # Test the loader
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python deck_loader.py <path_to_json>")
        sys.exit(1)
    
    loader = MetaDeckLoader(sys.argv[1])
    print(f"\nArchetypes: {loader.get_archetypes()[:5]}...")
    
    print("\n--- Sample deck ---")
    deck_info = loader.sample_deck_info()
    print(f"Archetype: {deck_info.archetype}")
    print(f"Win rate: {deck_info.win_rate}%")
    print(f"Energy: {deck_info.energy}")
    print(f"\nDeck string:\n{deck_info.deck_string}")
