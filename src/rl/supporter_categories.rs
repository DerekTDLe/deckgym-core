// Supporter/Trainer Effect Categories for RL Observation Encoding
//
// This module provides a way to categorize supporter/trainer cards by their effect type
// for use in the observation tensor. This allows the RL agent to generalize
// across cards with similar effects.

use crate::card_ids::CardId;

/// Categories of supporter/trainer card effects for observation encoding.
/// A card can belong to multiple categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SupporterEffectCategory {
    /// Heals damage from Pokemon (Potion, Erika, Pokemon Center Lady)
    Heal,
    /// Draws cards (Professor's Research, Copycat)
    Draw,
    /// Searches deck for cards (Poké Ball, May)
    Search,
    /// Attaches or manipulates energy (Misty, Brock, Elemental Switch)
    Energy,
    /// Boosts attack damage (Giovanni, Blaine, Red)
    DamageBoost,
    /// Switches Pokemon positions (Sabrina, Lyra, Koga)
    Switch,
    /// Reduces retreat cost (X Speed, Leaf)
    Retreat,
    /// Disrupts opponent's hand or board (Red Card, Mars, Silver)
    Disrupt,
    /// Evolves Pokemon (Rare Candy)
    Evolution,
    /// Card is a Supporter
    Supporter,
    /// Card is a Tool
    Tool,
    /// Card is an Item
    Item,
    /// Card is specialized toward a Pokemon or a type (Brock, Misty...)
    Specialized,
    /// Card is generalist (any Pokemon, any type)
    Generalist,
}

/// Number of effect categories (for observation vector sizing)
pub const NUM_SUPPORTER_EFFECT_CATEGORIES: usize = 14;

/// Returns the effect categories for a given trainer CardId.
/// Returns a slice of applicable categories.
pub fn get_supporter_effect_categories(card_id: CardId) -> &'static [SupporterEffectCategory] {
    use SupporterEffectCategory::*;

    match card_id {
        // === HEAL ===
        CardId::PA001Potion => &[Heal, Item],
        CardId::A1219Erika | CardId::A1266Erika | CardId::A4b328Erika | CardId::A4b329Erika => {
            &[Heal, Supporter, Specialized]
        }
        CardId::A2a072Irida | CardId::A2a087Irida | CardId::A4b330Irida | CardId::A4b331Irida => {
            &[Heal, Supporter, Specialized]
        }
        CardId::A2b070PokemonCenterLady | CardId::A2b089PokemonCenterLady => &[Heal, Supporter, Generalist],
        CardId::A3155Lillie
        | CardId::A3197Lillie
        | CardId::A3209Lillie
        | CardId::A4b348Lillie
        | CardId::A4b349Lillie
        | CardId::A4b374Lillie => &[Heal, Supporter, Specialized],

        // === DRAW ===
        CardId::PA007ProfessorsResearch | CardId::A4b373ProfessorsResearch => &[Draw, Supporter, Generalist],
        CardId::B1225Copycat | CardId::B1270Copycat => &[Draw, Supporter, Generalist],
        CardId::A2b069Iono | CardId::A2b088Iono | CardId::A4b340Iono | CardId::A4b341Iono => {
            &[Draw, Disrupt, Supporter, Generalist]
        }

        // === SEARCH ===
        CardId::PA005PokeBall | CardId::A2b111PokeBall => &[Search, Item, Specialized],
        CardId::A1a065MythicalSlab => &[Search, Item, Specialized],
        CardId::A2146PokemonCommunication
        | CardId::A4b316PokemonCommunication
        | CardId::A4b317PokemonCommunication => &[Search, Item, Generalist],
        CardId::A3a067Gladion | CardId::A3a081Gladion => &[Search, Supporter, Specialized],
        CardId::B1223May | CardId::B1268May => &[Search, Supporter, Generalist],
        CardId::B1226Lisia | CardId::B1271Lisia => &[Search, Supporter, Specialized],
        CardId::A2a073CelesticTownElder | CardId::A2a088CelesticTownElder => &[Search, Supporter, Specialized],
        CardId::B1a068Clemont | CardId::B1a081Clemont => &[Search, Supporter, Specialized],

        // === ENERGY ===
        CardId::A1220Misty | CardId::A1267Misty => &[Energy, Supporter, Specialized],
        CardId::A1224Brock | CardId::A1271Brock => &[Energy, Supporter, Specialized],
        CardId::A4151ElementalSwitch
        | CardId::A4b310ElementalSwitch
        | CardId::A4b311ElementalSwitch => &[Energy, Item, Generalist],
        CardId::A3a069Lusamine
        | CardId::A3a083Lusamine
        | CardId::A4b350Lusamine
        | CardId::A4b351Lusamine
        | CardId::A4b375Lusamine => &[Energy, Supporter, Specialized],
        CardId::B1217FlamePatch | CardId::B1331FlamePatch => &[Energy, Item, Specialized],

        // === DAMAGE BOOST ===
        CardId::A1223Giovanni
        | CardId::A1270Giovanni
        | CardId::A4b334Giovanni
        | CardId::A4b335Giovanni => &[DamageBoost, Supporter, Generalist],
        CardId::A1221Blaine | CardId::A1268Blaine => &[DamageBoost, Supporter, Specialized],
        CardId::A2b071Red | CardId::A2b090Red | CardId::A4b352Red | CardId::A4b353Red => {
            &[DamageBoost, Supporter, Generalist]
        }
        CardId::B1a066ClemontsBackpack => &[DamageBoost, Item, Specialized],

        // === SWITCH ===
        CardId::A1225Sabrina
        | CardId::A1272Sabrina
        | CardId::A4b338Sabrina
        | CardId::A4b339Sabrina => &[Switch, Supporter, Generalist],
        CardId::A1222Koga | CardId::A1269Koga => &[Switch, Supporter, Specialized],
        CardId::A4157Lyra | CardId::A4197Lyra | CardId::A4b332Lyra | CardId::A4b333Lyra => {
            &[Switch, Supporter, Specialized]
        }
        CardId::A3a064Repel => &[Switch, Item, Generalist],
        CardId::B1a069Serena | CardId::B1a082Serena => &[Switch, Supporter, Specialized],

        // === RETREAT ===
        CardId::PA002XSpeed => &[Retreat, Item, Generalist],
        CardId::A1a068Leaf | CardId::A1a082Leaf | CardId::A4b346Leaf | CardId::A4b347Leaf => {
            &[Retreat, Supporter, Generalist]
        }

        // === DISRUPT ===
        CardId::PA006RedCard => &[Disrupt, Item, Generalist],
        CardId::A2150Cyrus | CardId::A2190Cyrus | CardId::A4b326Cyrus | CardId::A4b327Cyrus => {
            &[Switch, Disrupt, Supporter, Generalist]
        }
        CardId::A2155Mars | CardId::A2195Mars | CardId::A4b344Mars | CardId::A4b345Mars => {
            &[Disrupt, Supporter, Generalist]
        }
        CardId::A4158Silver | CardId::A4198Silver | CardId::A4b336Silver | CardId::A4b337Silver => {
            &[Disrupt, Supporter, Generalist]
        }

        // === TOOL ===
        CardId::A2147GiantCape | CardId::A4b320GiantCape | CardId::A4b321GiantCape => &[Tool, Heal, Generalist],
        CardId::A2148RockyHelmet | CardId::A4b322RockyHelmet | CardId::A4b323RockyHelmet => {
            &[Tool, DamageBoost, Generalist]
        }
        CardId::A3146PoisonBarb => &[Tool, DamageBoost, Generalist],
        CardId::A3147LeafCape => &[Tool, Heal, Specialized],
        CardId::A3a065ElectricalCord
        | CardId::A4b318ElectricalCord
        | CardId::A4b319ElectricalCord => &[Tool, Energy, Specialized],
        CardId::A4a067InflatableBoat => &[Tool, Retreat, Specialized],
        CardId::B1219HeavyHelmet => &[Tool, Heal, Specialized],

        // === EVOLUTION ===
        CardId::A3144RareCandy
        | CardId::A4b314RareCandy
        | CardId::A4b315RareCandy
        | CardId::A4b379RareCandy => &[Evolution, Item, Generalist],
        CardId::B1a067QuickGrowExtract | CardId::B1a103QuickGrowExtract => &[Evolution, Item, Specialized],

        // === SPECIAL / MULTI-CATEGORY ===
        CardId::A3b066EeveeBag
        | CardId::A3b107EeveeBag
        | CardId::A4b308EeveeBag
        | CardId::A4b309EeveeBag => &[Heal, DamageBoost, Item, Specialized],

        // Default: unknown cards get empty slice
        _ => &[],
    }
}

/// Convert supporter effect categories to a one-hot/multi-hot vector.
pub fn encode_supporter_categories(card_id: CardId) -> [f32; NUM_SUPPORTER_EFFECT_CATEGORIES] {
    let categories = get_supporter_effect_categories(card_id);
    let mut encoding = [0.0; NUM_SUPPORTER_EFFECT_CATEGORIES];

    for cat in categories {
        encoding[*cat as usize] = 1.0;
    }

    encoding
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_potion_is_heal_and_item() {
        let cats = get_supporter_effect_categories(CardId::PA001Potion);
        assert!(cats.contains(&SupporterEffectCategory::Heal));
        assert!(cats.contains(&SupporterEffectCategory::Item));
    }

    #[test]
    fn test_iono_is_draw_disrupt_supporter() {
        let cats = get_supporter_effect_categories(CardId::A2b069Iono);
        assert!(cats.contains(&SupporterEffectCategory::Draw));
        assert!(cats.contains(&SupporterEffectCategory::Disrupt));
        assert!(cats.contains(&SupporterEffectCategory::Supporter));
    }

    #[test]
    fn test_encoding_size() {
        let encoding = encode_supporter_categories(CardId::PA001Potion);
        assert_eq!(encoding.len(), NUM_SUPPORTER_EFFECT_CATEGORIES);
        assert_eq!(encoding[SupporterEffectCategory::Heal as usize], 1.0);
        assert_eq!(encoding[SupporterEffectCategory::Item as usize], 1.0);
    }

    #[test]
    fn test_pokeball_is_search_item() {
        let cats = get_supporter_effect_categories(CardId::PA005PokeBall);
        assert!(cats.contains(&SupporterEffectCategory::Search));
        assert!(cats.contains(&SupporterEffectCategory::Item));
    }
}
