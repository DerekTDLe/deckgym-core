"""
MetaDeckLoader - Load competitive decks from JSON for RL training.

Supports two JSON formats:
1. Simple format (simple_deck.json): flat list of decks with energy_type
2. Meta format (meta_deck.json): archetypes with score/strength per deck

Sampling strategy for generalization:
- Hierarchical: First pick archetype, then deck (ensures archetype diversity)
- Weighted: Sample weighted by deck strength

Meta deck data sourced from:
  https://github.com/chase-manning/pokemon-tcg-pocket-tier-list
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DeckInfo:
    """Information about a deck."""

    archetype: str
    strength: float  # Pure deck power (0-1)
    score: float  # Power × popularity (0-1)
    deck_string: str


@dataclass
class ArchetypeInfo:
    """Aggregated statistics for an archetype."""

    name: str
    decks: list[DeckInfo] = field(default_factory=list)

    @property
    def mean_strength(self) -> float:
        if not self.decks:
            return 0.0
        return sum(d.strength for d in self.decks) / len(self.decks)

    @property
    def max_strength(self) -> float:
        return max((d.strength for d in self.decks), default=0.0)

    @property
    def deck_count(self) -> int:
        return len(self.decks)


class MetaDeckLoader:
    """
    Load and sample competitive decks with diversity-focused sampling.

    Sampling modes:
    - uniform: Random deck (may oversample popular archetypes)
    - hierarchical: Random archetype, then random deck (archetype diversity)
    - weighted: Weighted by strength
    """

    def __init__(
        self,
        json_path: str,
        max_archetypes: int = None,
        max_decks_per_archetype: int = None,
    ):
        """
        Load decks from JSON file.

        Args:
            json_path: Path to the deck JSON file
            max_archetypes: Limit archetypes (None = load all)
            max_decks_per_archetype: Limit decks per archetype (None = load all)
        """
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.decks: list[DeckInfo] = []
        self.archetypes: dict[str, ArchetypeInfo] = {}

        self._load_decks(max_archetypes, max_decks_per_archetype)

    def _load_decks(
        self, max_archetypes: int = None, max_decks_per_archetype: int = None
    ):
        """Parse all decks from the JSON data."""
        if isinstance(self.data, list):
            self._load_simple_format()
        elif "archetypes" in self.data:
            self._load_meta_format(max_archetypes, max_decks_per_archetype)
        else:
            raise ValueError("Unknown JSON format")

    def _load_simple_format(self):
        """Load decks from simple format (flat list)."""
        for deck in self.data:
            name = deck.get("name", "Unknown")
            energy = deck.get("energy_type", "")
            cards = deck.get("cards", [])

            if not cards:
                continue

            deck_string = self._cards_to_string_simple(cards, energy)
            deck_info = DeckInfo(
                archetype=name,
                strength=0.5,
                score=0.5,
                deck_string=deck_string,
            )

            self.decks.append(deck_info)

            if name not in self.archetypes:
                self.archetypes[name] = ArchetypeInfo(name=name)
            self.archetypes[name].decks.append(deck_info)

    def _load_meta_format(
        self, max_archetypes: int = None, max_decks_per_archetype: int = None
    ):
        """Load decks from meta format with score/strength."""
        archetypes = self.data.get("archetypes", [])

        if max_archetypes:
            archetypes = archetypes[:max_archetypes]

        for archetype in archetypes:
            archetype_name = archetype.get("name", "Unknown")
            decks_data = archetype.get("decks", [])

            if max_decks_per_archetype:
                # Keep best decks by strength
                decks_data = sorted(
                    decks_data, key=lambda d: d.get("strength", 0), reverse=True
                )
                decks_data = decks_data[:max_decks_per_archetype]

            if archetype_name not in self.archetypes:
                self.archetypes[archetype_name] = ArchetypeInfo(name=archetype_name)

            for deck in decks_data:
                cards = deck.get("cards", [])
                if not cards:
                    continue

                deck_string = self._cards_to_string_meta(cards)
                if deck_string is None:
                    continue

                deck_info = DeckInfo(
                    archetype=archetype_name,
                    strength=deck.get("strength", 0.5),
                    score=deck.get("score", 0.5),
                    deck_string=deck_string,
                )

                self.decks.append(deck_info)
                self.archetypes[archetype_name].decks.append(deck_info)

    def _cards_to_string_simple(self, cards: list[dict], energy: str = "") -> str:
        """Convert simple format cards to deck string."""
        lines = []
        if energy:
            lines.append(f"Energy: {energy.split('/')[0].strip()}")

        for card in cards:
            count = card.get("count", 1)
            name = card.get("name", "Unknown")
            set_code = card.get("set", "A1")
            number = card.get("number", "001")
            if isinstance(number, int):
                number = f"{number:03d}"
            elif isinstance(number, str) and number.isdigit():
                number = number.zfill(3)
            lines.append(f"{count} {name} {set_code} {number}")

        return "\n".join(lines)

    def _cards_to_string_meta(self, cards: list[dict]) -> Optional[str]:
        """Convert meta format cards (set/number only) to deck string."""
        lines = []
        for card in cards:
            count = card.get("count", 1)
            set_code = card.get("set", "")
            number = card.get("number", "")
            if not set_code or not number:
                return None
            if isinstance(number, int):
                number = f"{number:03d}"
            elif isinstance(number, str) and number.isdigit():
                number = number.zfill(3)
            lines.append(f"{count} {set_code} {number}")
        return "\n".join(lines)

    # =========================================================================
    # Sampling Methods
    # =========================================================================

    def sample_deck(self, mode: str = "hierarchical") -> str:
        """
        Sample a deck with the specified strategy.

        Args:
            mode: Sampling mode
                - "uniform": Pure random (may oversample popular archetypes)
                - "hierarchical": Random archetype, then random deck
                - "weighted": Weighted by strength
        """
        if mode == "uniform":
            return random.choice(self.decks).deck_string
        elif mode == "hierarchical":
            return self._sample_hierarchical()
        elif mode == "weighted":
            return self._sample_weighted()
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _sample_hierarchical(self) -> str:
        """Sample: random archetype → random deck."""
        archetype = random.choice(list(self.archetypes.values()))
        deck = random.choice(archetype.decks)
        return deck.deck_string

    def _sample_weighted(self) -> str:
        """Sample weighted by strength."""
        weights = [max(d.strength, 0.01) for d in self.decks]
        return random.choices(self.decks, weights=weights, k=1)[0].deck_string

    def sample_deck_info(self, mode: str = "hierarchical") -> DeckInfo:
        """Sample a deck with full info."""
        if mode == "hierarchical":
            archetype = random.choice(list(self.archetypes.values()))
            return random.choice(archetype.decks)
        return random.choice(self.decks)

    def get_archetypes(self) -> list[str]:
        """Get list of all archetypes."""
        return list(self.archetypes.keys())

    def get_top_decks(self, n: int = 100) -> list[DeckInfo]:
        """Get top N decks by strength."""
        return sorted(self.decks, key=lambda d: d.strength, reverse=True)[:n]

    def filter_by_win_rate(
        self, min_rate: float = 0.0, max_rate: float = 100.0
    ) -> list[DeckInfo]:
        """
        Filter decks by win rate equivalent (strength * 100).

        Since strength is 0-1, we treat it as win rate percentage.

        Args:
            min_rate: Minimum win rate % (e.g., 55.0 for 55%)
            max_rate: Maximum win rate % (e.g., 100.0 for 100%)

        Returns:
            List of decks within the win rate range.
        """
        min_strength = min_rate / 100.0
        max_strength = max_rate / 100.0
        return [d for d in self.decks if min_strength <= d.strength <= max_strength]

    def sample_pair(self, mode: str = "hierarchical") -> tuple[str, str]:
        """Sample two decks for self-play."""
        return self.sample_deck(mode), self.sample_deck(mode)

    def __len__(self) -> int:
        return len(self.decks)


class CurriculumDeckLoader:
    """
    Curriculum learning: simple → meta decks.

    Uses hierarchical sampling for maximum diversity.
    """

    def __init__(
        self,
        simple_path: str,
        meta_path: str,
        max_meta_archetypes: int = None,
        max_decks_per_archetype: int = 20,
    ):
        """
        Args:
            simple_path: Path to simple_deck.json
            meta_path: Path to meta_deck.json
            max_meta_archetypes: Limit meta archetypes (None = all)
            max_decks_per_archetype: Limit decks per archetype (keeps best)
        """
        self.simple_loader = MetaDeckLoader(simple_path)
        self.meta_loader = MetaDeckLoader(
            meta_path,
            max_archetypes=max_meta_archetypes,
            max_decks_per_archetype=max_decks_per_archetype,
        )

        self.difficulty = 0.0  # 0 = all simple, 1 = all meta
        self.sampling_mode = "hierarchical"  # or "weighted"

    def set_difficulty(self, difficulty: float):
        """Set curriculum difficulty (0.0 = simple, 1.0 = meta)."""
        self.difficulty = max(0.0, min(1.0, difficulty))

    def set_sampling_mode(self, mode: str):
        """Set sampling mode: 'hierarchical' or 'weighted'."""
        self.sampling_mode = mode

    def sample_deck(self) -> str:
        """Sample a deck based on difficulty and mode."""
        if random.random() < self.difficulty:
            return self.meta_loader.sample_deck(mode=self.sampling_mode)
        else:
            return self.simple_loader.sample_deck(mode=self.sampling_mode)

    def sample_pair(self) -> tuple[str, str]:
        """Sample two decks for self-play."""
        return self.sample_deck(), self.sample_deck()

    def __len__(self) -> int:
        return len(self.simple_loader) + len(self.meta_loader)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python deck_loader.py <meta_deck.json> [simple_deck.json]")
        sys.exit(1)

    loader = MetaDeckLoader(sys.argv[1])
    print(f"Loaded {len(loader.decks)} decks from {len(loader.archetypes)} archetypes")

    print("\n--- Archetype Statistics ---")
    archetypes = sorted(
        loader.archetypes.values(), key=lambda a: a.mean_strength, reverse=True
    )
    for arch in archetypes[:10]:
        print(
            f"  {arch.name[:40]:40s} | decks={arch.deck_count:3d} | strength={arch.mean_strength:.3f}"
        )

    print("\n--- Sample decks (hierarchical) ---")
    for _ in range(3):
        deck = loader.sample_deck_info(mode="hierarchical")
        print(f"  {deck.archetype[:30]} (strength={deck.strength:.3f})")
