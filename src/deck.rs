use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::hash::{Hash, Hasher};

use crate::card_ids::CardId;
use crate::database::get_card_by_enum;
use crate::models::{Card, EnergyType};

/// Represents a deck of cards.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Deck {
    pub cards: Vec<Card>,
    pub(crate) energy_types: Vec<EnergyType>,
}

impl Hash for Deck {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.cards.hash(state);
        // order energy types alphabetically to ensure consistent hash
        let mut energy_types: Vec<_> = self.energy_types.iter().collect();
        energy_types.sort();
        energy_types.hash(state);
    }
}

impl Deck {
    /// Parses a deck file and returns a `Deck` struct with cards flattened based on their counts.
    pub fn from_file(file_path: &str) -> Result<Self, String> {
        let contents = fs::read_to_string(file_path)
            .map_err(|err| format!("Failed to read file {file_path}: {err}"))?;

        Self::from_string(&contents)
    }

    pub fn from_string(contents: &str) -> Result<Self, String> {
        let mut energy_types = HashSet::new();
        let mut cards = Vec::new();
        for line in contents.lines() {
            // if line is empty or starts with "Pokemon:" or "Trainer:, skip it
            let trimmed = line.trim();
            if trimmed.is_empty()
                || trimmed.starts_with("Pokémon:")
                || trimmed.starts_with("Trainer:")
            {
                continue;
            }
            if trimmed.starts_with("Energy:") {
                let energy_type: &str = trimmed
                    .split_whitespace()
                    .last()
                    .expect("Energy: line should have an energy type");
                energy_types
                    .insert(EnergyType::from_str(energy_type).expect("Invalid energy type"));
                continue;
            }

            let (count, card) = Card::from_str_with_count(trimmed)?;
            cards.extend(vec![card; count as usize]);
        }

        // If empty energy types set, infer from attack costs using expert heuristic
        if energy_types.is_empty() {
            energy_types = Self::infer_energy_types_from_attacks(&cards);
        }

        Ok(Self {
            cards,
            energy_types: energy_types.into_iter().collect(),
        })
    }

    pub fn is_valid(&self) -> bool {
        let basic = self.cards.iter().filter(|x| x.is_basic()).count();
        let has_correct_size = self.cards.len() == 20 && basic >= 1;

        if !has_correct_size {
            return false;
        }

        // Check that no card name appears more than twice
        let mut card_counts = std::collections::HashMap::new();
        for card in &self.cards {
            let count = card_counts.entry(card.get_name()).or_insert(0);
            *count += 1;
            if *count > 2 {
                return false;
            }
        }

        true
    }

    /// Draws a card from the deck.
    /// Returns `Some(Card)` if the deck is not empty, otherwise returns `None`.
    pub fn draw(&mut self) -> Option<Card> {
        if self.cards.is_empty() {
            None
        } else {
            Some(self.cards.remove(0))
        }
    }

    /// Shuffles the deck of cards.
    pub fn shuffle(&mut self, initial_shuffle: bool, rng: &mut impl Rng) {
        if initial_shuffle {
            // Ensure there is at least 1 basic pokemon in the initial 5 cards
            let (mut matching, mut non_matching): (Vec<_>, Vec<_>) =
                self.cards.clone().into_iter().partition(is_basic);
            matching.shuffle(rng);
            non_matching.shuffle(rng);

            let shuffled_cards: Vec<Card> =
                vec![matching.pop().expect("Decks must have at least 1 basic")];

            let mut remaining = [matching, non_matching].concat();
            remaining.shuffle(rng);

            self.cards = [shuffled_cards, remaining].concat();
        } else {
            self.cards.shuffle(rng);
        }
    }

    /// Expert heuristic for inferring deck energy types from attack costs.
    ///
    /// Algorithm:
    /// 1. Analyze attack costs (prefer 2nd attack), excluding splashable/self-charging Pokemon
    /// 2. If no types found: re-include passe-partout, find baby types, exclude them, use remaining
    /// 3. Fallback to card energy_type field if single non-Colorless type
    /// 4. If Irida trainer present → Water
    /// 5. Ultimate fallback → Lightning
    fn infer_energy_types_from_attacks(cards: &[Card]) -> HashSet<EnergyType> {
        // Pokemon to exclude from initial analysis (splashable + self-charging)
        const EXCLUDED_POKEMON: &[&str] = &[
            // Splashable (passe-partout)
            "Froakie",
            "Frogadier",
            "Greninja",
            "Greninja ex",
            "Druddigon",
            "Oricorio",
            "Sylveon ex",
            "Decidueye ex",
            // Self-charging abilities
            "Magnemite",
            "Magneton",
            "Magnezone",
            "Ralts",
            "Kirlia",
            "Gardevoir",
            "Leafeon ex",
            "Flareon ex",
            "Charmeleon",
            "Giratina ex",
            "Zeraora",
            "Deino",
            "Zweilous",
            "Hydreigon",
            // Self-charging attacks
            "Manaphy",
            "Moltres ex",
        ];

        // Passe-partout cards (subset that we re-include in step 2)
        const PASSE_PARTOUT: &[&str] = &[
            "Froakie",
            "Frogadier",
            "Greninja",
            "Greninja ex",
            "Druddigon",
            "Oricorio",
            "Sylveon ex",
            "Decidueye ex",
        ];

        // Helper: get non-Colorless energy types from attack (prefer 2nd attack)
        let get_attack_energies = |pokemon: &crate::models::PokemonCard| -> Vec<EnergyType> {
            let attack = if pokemon.attacks.len() > 1 {
                &pokemon.attacks[1]
            } else if !pokemon.attacks.is_empty() {
                &pokemon.attacks[0]
            } else {
                return vec![];
            };
            attack
                .energy_required
                .iter()
                .filter(|e| **e != EnergyType::Colorless)
                .copied()
                .collect()
        };

        // Helper: check if Pokemon is a "baby" (basic with no attacks or colorless-only)
        let is_baby = |pokemon: &crate::models::PokemonCard| -> bool {
            if pokemon.stage != 0 {
                return false;
            }
            // Check if all attacks are colorless-only
            pokemon
                .attacks
                .iter()
                .all(|atk| atk.energy_required.is_empty())
        };

        // Step 1: Analyze attack costs, excluding specific cards
        let mut energy_types: HashSet<EnergyType> = HashSet::new();
        for card in cards {
            if let Card::Pokemon(pokemon) = card {
                if EXCLUDED_POKEMON.contains(&pokemon.name.as_str()) {
                    continue;
                }
                for energy in get_attack_energies(pokemon) {
                    energy_types.insert(energy);
                }
            }
        }

        if !energy_types.is_empty() {
            return energy_types;
        }

        // Step 2: No types found - re-include passe-partout, exclude baby types
        let mut baby_types: HashSet<EnergyType> = HashSet::new();
        let mut passe_partout_types: HashSet<EnergyType> = HashSet::new();

        for card in cards {
            if let Card::Pokemon(pokemon) = card {
                if is_baby(pokemon) {
                    baby_types.insert(pokemon.energy_type);
                }
                if PASSE_PARTOUT.contains(&pokemon.name.as_str()) {
                    for energy in get_attack_energies(pokemon) {
                        passe_partout_types.insert(energy);
                    }
                }
            }
        }

        // Remove baby types from passe-partout types
        for bt in &baby_types {
            passe_partout_types.remove(bt);
        }

        if !passe_partout_types.is_empty() {
            return passe_partout_types;
        }

        // Step 3: Fallback to card energy_type field
        let mut card_types: HashSet<EnergyType> = HashSet::new();
        for card in cards {
            if let Card::Pokemon(pokemon) = card {
                if pokemon.energy_type != EnergyType::Colorless {
                    card_types.insert(pokemon.energy_type);
                }
            }
        }
        if card_types.len() == 1 {
            return card_types;
        }

        // Step 4: Check for Irida trainer → Water
        let has_irida = cards.iter().any(|card| {
            if let Card::Trainer(trainer) = card {
                trainer.name == "Irida"
            } else {
                false
            }
        });
        if has_irida {
            let mut result = HashSet::new();
            result.insert(EnergyType::Water);
            return result;
        }

        // Step 5: Ultimate fallback → Lightning
        let mut result = HashSet::new();
        result.insert(EnergyType::Lightning);
        result
    }
}

impl Card {
    /// Parses a line and returns a tuple of count and a `Card`.
    pub fn from_str_with_count(line: &str) -> Result<(u32, Card), String> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Err(format!("Invalid card format: {line}"));
        }

        let count = parts[0]
            .parse::<u32>()
            .map_err(|_| format!("Invalid count: {}", parts[0]))?;
        let set = parts[parts.len() - 2];
        // maybe pad number with 0 on the left if missing 0s
        let number = parts[parts.len() - 1];
        let padded_number = format!("{number:0>3}");
        let id = format!("{set} {padded_number}");

        let card_id =
            CardId::from_card_id(&id).ok_or_else(|| format!("Card ID not found for id: {id}"))?;
        let card = get_card_by_enum(card_id);

        Ok((count, card.clone()))
    }
}

pub fn is_basic(card: &Card) -> bool {
    card.is_basic()
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use super::*;

    #[test]
    fn test_deck_from_file() {
        let file_path = "example_decks/venusaur-exeggutor.txt";
        let deck = Deck::from_file(file_path).expect("Failed to parse deck from file");

        // Add assertions to verify the deck content
        assert!(!deck.cards.is_empty(), "Deck should not be empty");

        // Example assertion: Check if the first card is Bulbasaur
        let first_card = &deck.cards[0];
        if let Card::Pokemon(card) = first_card {
            assert_eq!(card.name, "Bulbasaur");
        } else {
            panic!("Expected first card to be Bulbasaur");
        }
    }

    #[test]
    fn test_draw_card() {
        let file_path = "example_decks/venusaur-exeggutor.txt";
        let mut deck = Deck::from_file(file_path).expect("Failed to parse deck from file");

        let initial_count = deck.cards.len();
        let drawn_card = deck.draw();
        assert!(drawn_card.is_some(), "Should draw a card");
        assert_eq!(
            deck.cards.len(),
            initial_count - 1,
            "Deck size should decrease by 1"
        );
    }

    #[test]
    fn test_initial_shuffle_deck() {
        let file_path = "example_decks/venusaur-exeggutor.txt";
        let mut deck = Deck::from_file(file_path).expect("Failed to parse deck from file");

        let mut rng = thread_rng();
        deck.shuffle(true, &mut rng);

        // Ensure there is at least 1 basic pokemon in the initial 5 cards
        let initial_five_cards = &deck.cards[..5];
        assert!(
            initial_five_cards.iter().any(is_basic),
            "There should be at least 1 basic pokemon in the initial 5 cards"
        );
    }

    #[test]
    fn test_from_string() {
        let string = r#"Pokémon: 8
2 Ekans A1 164
2 Arbok A1 165
2 Koffing A1 176
2 Weezing A1 177

Trainer: 12
2 Professor's Research P-A 007
2 Koga A1 222
2 Poké Ball P-A 005
2 Sabrina A1 225
2 Potion P-A 001
1 X Speed P-A 002
1 Giovanni A1 223"#;
        let deck = Deck::from_string(string).expect("Failed to parse deck from string");

        assert_eq!(deck.cards.len(), 20);
    }

    #[test]
    fn test_from_string_with_energy() {
        let string = r#"Energy: Grass
Pokémon: 8
2 Ekans A1 164
2 Arbok A1 165
2 Koffing A1 176
2 Weezing A1 177

Trainer: 12
2 Professor's Research P-A 007
2 Koga A1 222
2 Poké Ball P-A 005
2 Sabrina A1 225
2 Potion P-A 001
1 X Speed P-A 002
1 Giovanni A1 223"#;
        let deck = Deck::from_string(string).expect("Failed to parse deck from string");

        assert_eq!(deck.cards.len(), 20);
        assert_eq!(deck.energy_types.len(), 1);
        assert_eq!(deck.energy_types[0], EnergyType::Grass);
    }

    #[test]
    fn test_from_string_without_leading_zeros() {
        let string = r#"Energy: Grass
2 Weedle A1 8
2 Kakuna A1 9
2 Beedrill A1 10
2 Cottonee A1 27
2 Whimsicott A1 28
1 Giovanni A1 270
1 Sabrina A1 272
2 Potion P-A 1
2 X Speed P-A 2
2 Poke Ball P-A 5
2 Professor's Research P-A 7"#;
        let deck = Deck::from_string(string).expect("Failed to parse deck from string");
        assert_eq!(deck.cards.len(), 20);
    }

    #[test]
    fn test_infer_energy_from_attacks_water() {
        // Deck with Water pokemon (Suicune ex), no Energy: line
        // Suicune ex has Water attacks, should infer Water type
        let string = r#"2 Suicune ex A4a 20
2 Suicune ex A4a 20
2 Suicune ex A4a 20
2 Suicune ex A4a 20
2 Suicune ex A4a 20
2 Suicune ex A4a 20
2 Suicune ex A4a 20
2 Professor's Research P-A 007
2 Poké Ball P-A 005
2 Potion P-A 001"#;
        let deck = Deck::from_string(string).expect("Failed to parse deck");
        assert!(deck.energy_types.contains(&EnergyType::Water));
    }

    #[test]
    fn test_infer_excludes_greninja_and_uses_fallback() {
        // Deck with only Greninja (excluded) + Irida trainer should fallback to Water
        let string = r#"2 Froakie A1 87
2 Frogadier A1 88
2 Greninja A1 89
2 Froakie A1 87
2 Frogadier A1 88
2 Greninja A1 89
2 Irida A2a 72
2 Professor's Research P-A 007
2 Poké Ball P-A 005
2 Potion P-A 001"#;
        let deck = Deck::from_string(string).expect("Failed to parse deck");
        assert!(deck.energy_types.contains(&EnergyType::Water));
    }

    #[test]
    fn test_infer_ultimate_fallback_lightning() {
        // Deck with only colorless Pokemon, no Irida → Lightning fallback
        let string = r#"2 Chansey A1 107
2 Chansey A1 107
2 Chansey A1 107
2 Chansey A1 107
2 Chansey A1 107
2 Chansey A1 107
2 Chansey A1 107
2 Professor's Research P-A 007
2 Poké Ball P-A 005
2 Potion P-A 001"#;
        let deck = Deck::from_string(string).expect("Failed to parse deck");
        assert!(deck.energy_types.contains(&EnergyType::Lightning));
    }

    #[test]
    fn test_infer_passe_partout_with_baby_exclusion() {
        // Deck with Oricorio (passe-partout), Mantyke (Water baby), Greninja line (passe-partout Water)
        // Algorithm: exclude all passe-partout+self-charging → empty
        // Step 2: re-include passe-partout (Greninja=Water, Oricorio=Lightning), find baby (Mantyke=Water)
        // Exclude Water from types → Oricorio A3066 has Lightning attacks
        let string = r#"2 Oricorio A3 66
2 Mantyke A4a 23
2 Froakie A1 87
2 Frogadier A1 88
2 Greninja A1 89
2 Froakie A1 87
2 Frogadier A1 88
2 Greninja A1 89
2 Professor's Research P-A 007
2 Poké Ball P-A 005"#;
        let deck = Deck::from_string(string).expect("Failed to parse deck");
        // Should fallback to Lightning since all Water energy is excluded by Mantyke baby
        assert!(deck.energy_types.contains(&EnergyType::Lightning));
    }

    #[test]
    fn test_infer_giratina_darkrai_darkness() {
        // Giratina ex is excluded (self-charging), but Darkrai ex is not
        // Should infer Darkness type from Darkrai ex's attack costs
        let string = r#"2 Darkrai ex A2 110
2 Giratina ex A2b 35
2 Darkrai ex A2 110
2 Giratina ex A2b 35
2 Giratina ex A2b 35
2 Giratina ex A2b 35
2 Professor's Research P-A 007
2 Poké Ball P-A 005
2 Potion P-A 001
2 Rocky Helmet A2 148"#;
        let deck = Deck::from_string(string).expect("Failed to parse deck");
        assert!(deck.energy_types.contains(&EnergyType::Darkness));
    }
}
