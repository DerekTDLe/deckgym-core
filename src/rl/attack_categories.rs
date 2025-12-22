// Attack Effect Categories for RL Observation Encoding
//
// This module categorizes Pokemon attack effects by their mechanics
// to enable generalization in the RL observation tensor.

/// TODO : refine the categories, it is a simple implementation
use crate::actions::Mechanic;

/// Categories of attack effects for observation encoding.
/// An attack can belong to multiple categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttackEffectCategory {
    // === CORE EFFECTS ===
    /// Heals self or allies
    Heal,
    /// Inflicts status conditions on opponent
    StatusInflict,
    /// Involves randomness (coin flips)
    Variance,
    
    // === RESOURCES ===
    /// Manipulates energy (discard, charge, move)
    EnergyManip,
    /// Searches or draws cards
    CardAdvantage,
    
    // === DAMAGE MODIFIERS ===
    /// Conditional extra damage
    ConditionalDamage,
    /// Damages bench Pokemon
    SpreadDamage,
    /// Recoil or self-damage
    SelfDamage,
    
    // === TACTICAL ===
    /// Blocks, prevents, or reduces future actions
    Protection,
    /// Disrupts opponent's board or hand
    Disruption,
    /// Movement effects (switch, retreat)
    Movement,
    
    // === SETUP ===
    /// Board development (bench, evolution)
    BoardDevelopment,
}

/// Number of attack effect categories
pub const NUM_ATTACK_EFFECT_CATEGORIES: usize = 12;

/// Returns the effect categories for a given Mechanic.
pub fn get_attack_effect_categories(mechanic: &Mechanic) -> &'static [AttackEffectCategory] {
    use AttackEffectCategory::*;

    match mechanic {
        // === HEALS ===
        Mechanic::SelfHeal { .. } => &[Heal],

        // === STATUS ===
        Mechanic::InflictStatusConditions { .. } => &[StatusInflict],
        Mechanic::ChanceStatusAttack { .. } => &[StatusInflict, Variance],
        Mechanic::ExtraDamageForEachHeadsWithStatus { .. } => &[ConditionalDamage, StatusInflict, Variance],

        // === COIN FLIP ===
        Mechanic::CoinFlipExtraDamage { .. } => &[ConditionalDamage, Variance],
        Mechanic::CoinFlipExtraDamageOrSelfDamage { .. } => &[ConditionalDamage, SelfDamage, Variance],
        Mechanic::ExtraDamageForEachHeads { .. } => &[ConditionalDamage, Variance],
        Mechanic::CoinFlipNoEffect => &[Variance],
        Mechanic::ExtraDamageIfBothHeads { .. } => &[ConditionalDamage, Variance],
        Mechanic::CoinFlipToBlockAttackNextTurn => &[Variance, Protection],

        // === ENERGY MANIPULATION ===
        Mechanic::SelfDiscardEnergy { .. } => &[EnergyManip],
        Mechanic::SelfDiscardAllEnergy => &[EnergyManip],
        Mechanic::SelfDiscardRandomEnergy => &[EnergyManip, Variance],
        Mechanic::DiscardRandomGlobalEnergy => &[EnergyManip, Variance],
        Mechanic::DiscardEnergyFromOpponentActive => &[EnergyManip],
        Mechanic::SelfChargeActive { .. } => &[EnergyManip],
        Mechanic::ChargeBench { .. } => &[EnergyManip],
        Mechanic::ManaphyOceanicGift => &[EnergyManip],
        Mechanic::MoltresExInfernoDance => &[EnergyManip, Variance],
        Mechanic::VaporeonHyperWhirlpool => &[EnergyManip],

        // === CONDITIONAL DAMAGE ===
        Mechanic::ExtraDamageIfEx { .. } => &[ConditionalDamage],
        Mechanic::ExtraDamageIfExtraEnergy { .. } => &[ConditionalDamage],
        Mechanic::ExtraDamageIfHurt { .. } => &[ConditionalDamage],
        Mechanic::ExtraDamageIfKnockedOutLastTurn { .. } => &[ConditionalDamage],
        Mechanic::ExtraDamageIfToolAttached { .. } => &[ConditionalDamage],
        Mechanic::ExtraDamagePerEnergy { .. } => &[ConditionalDamage],
        Mechanic::ExtraDamagePerSpecificEnergy { .. } => &[ConditionalDamage],
        Mechanic::DamagePerEnergyAll { .. } => &[ConditionalDamage],
        Mechanic::ExtraDamagePerTrainerInOpponentDeck { .. } => &[ConditionalDamage],
        Mechanic::ExtraDamageIfCardInDiscard { .. } => &[ConditionalDamage],
        Mechanic::BenchCountDamage { .. } => &[ConditionalDamage],
        Mechanic::EvolutionBenchCountDamage { .. } => &[ConditionalDamage],
        Mechanic::DamageReducedBySelfDamage => &[ConditionalDamage, SelfDamage],
        Mechanic::DamageEqualToSelfDamage => &[ConditionalDamage],
        Mechanic::ExtraDamageEqualToSelfDamage => &[ConditionalDamage],

        // === BENCH DAMAGE ===
        Mechanic::DamageAllOpponentPokemon { .. } => &[SpreadDamage],
        Mechanic::AlsoBenchDamage { .. } => &[SpreadDamage],
        Mechanic::AlsoChoiceBenchDamage { .. } => &[SpreadDamage],
        Mechanic::DirectDamage { .. } => &[SpreadDamage],
        Mechanic::ConditionalBenchDamage { .. } => &[SpreadDamage, ConditionalDamage],
        Mechanic::PalkiaExDimensionalStorm => &[SpreadDamage, EnergyManip],

        // === CARD EFFECTS ===
        Mechanic::DamageAndCardEffect { .. } => &[Protection],
        Mechanic::DamageAndMultipleCardEffects { .. } => &[Protection],
        Mechanic::DamageAndTurnEffect { .. } => &[Protection],
        Mechanic::BlockBasicAttack => &[Protection],
        Mechanic::ShuffleOpponentActiveIntoDeck => &[Disruption],

        // === SEARCH ===
        Mechanic::SearchToHandByEnergy { .. } => &[CardAdvantage],
        Mechanic::SearchToBenchByName { .. } => &[CardAdvantage],
        Mechanic::MagikarpWaterfallEvolution => &[BoardDevelopment],

        // === SELF DAMAGE ===
        Mechanic::SelfDamage { .. } => &[SelfDamage],
        Mechanic::RecoilIfKo { .. } => &[SelfDamage],

        // === SPECIAL / UTILITY ===
        Mechanic::SwitchSelfWithBench => &[Movement],
        Mechanic::CelebiExPowerfulBloom => &[Variance],
        Mechanic::MegaBlazikenExMegaBurningAttack => &[EnergyManip, StatusInflict],
        Mechanic::HoOhExPhoenixTurbo => &[EnergyManip],
        Mechanic::HealBenchedBasic { .. } => &[Heal],
    }
}

/// Convert attack effect categories to a multi-hot vector.
pub fn encode_attack_categories(mechanic: &Mechanic) -> [f32; NUM_ATTACK_EFFECT_CATEGORIES] {
    let categories = get_attack_effect_categories(mechanic);
    let mut encoding = [0.0; NUM_ATTACK_EFFECT_CATEGORIES];

    for cat in categories {
        encoding[*cat as usize] = 1.0;
    }

    encoding
}

/// Get categories from an attack's effect text by looking up the mechanic.
/// Returns zeros if the effect is not found or has no special mechanic.
pub fn encode_attack_effect_text(effect_text: Option<&str>) -> [f32; NUM_ATTACK_EFFECT_CATEGORIES] {
    use crate::actions::EFFECT_MECHANIC_MAP;

    if let Some(text) = effect_text {
        if let Some(mechanic) = EFFECT_MECHANIC_MAP.get(text) {
            return encode_attack_categories(mechanic);
        }
    }

    [0.0; NUM_ATTACK_EFFECT_CATEGORIES]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::EnergyType;

    #[test]
    fn test_self_heal_is_heals() {
        let mechanic = Mechanic::SelfHeal { amount: 20 };
        let cats = get_attack_effect_categories(&mechanic);
        assert!(cats.contains(&AttackEffectCategory::Heal));
    }

    #[test]
    fn test_coin_flip_multi_category() {
        let mechanic = Mechanic::CoinFlipExtraDamageOrSelfDamage {
            extra_damage: 40,
            self_damage: 20,
        };
        let cats = get_attack_effect_categories(&mechanic);
        assert!(cats.contains(&AttackEffectCategory::Variance));
        assert!(cats.contains(&AttackEffectCategory::SelfDamage));
    }

    #[test]
    fn test_encoding_size() {
        let mechanic = Mechanic::SelfDiscardEnergy {
            energies: vec![EnergyType::Fire],
        };
        let encoding = encode_attack_categories(&mechanic);
        assert_eq!(encoding.len(), NUM_ATTACK_EFFECT_CATEGORIES);
        assert_eq!(encoding[AttackEffectCategory::EnergyManip as usize], 1.0);
    }
}
