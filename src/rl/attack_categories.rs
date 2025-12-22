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
    /// Heals self (SelfHeal, MegaDrain)
    Heals,
    /// Inflicts status conditions (Poison, Sleep, Paralyze, Confuse, Burn)
    Status,
    /// Involves coin flips (CoinFlipExtraDamage, CoinFlipNoEffect, etc.)
    CoinFlip,
    /// Manipulates energy (discard, charge, move)
    EnergyManip,
    /// Conditional extra damage (if ex, if hurt, if extra energy, etc.)
    ConditionalDamage,
    /// Damages bench Pokemon
    BenchDamage,
    /// Applies card effects (prevent attack, prevent retreat, etc.)
    CardEffect,
    /// Searches deck for cards
    Search,
    /// Deals damage to self (recoil)
    SelfDamage,
}

/// Number of attack effect categories
pub const NUM_ATTACK_EFFECT_CATEGORIES: usize = 9;

/// Returns the effect categories for a given Mechanic.
pub fn get_attack_effect_categories(mechanic: &Mechanic) -> &'static [AttackEffectCategory] {
    use AttackEffectCategory::*;
    
    match mechanic {
        // === HEALS ===
        Mechanic::SelfHeal { .. } => &[Heals],
        
        // === STATUS ===
        Mechanic::InflictStatusConditions { .. } => &[Status],
        Mechanic::ChanceStatusAttack { .. } => &[Status, CoinFlip],
        Mechanic::ExtraDamageForEachHeadsWithStatus { .. } => &[CoinFlip, Status],
        
        // === COIN FLIP ===
        Mechanic::CoinFlipExtraDamage { .. } => &[CoinFlip],
        Mechanic::CoinFlipExtraDamageOrSelfDamage { .. } => &[CoinFlip, SelfDamage],
        Mechanic::ExtraDamageForEachHeads { .. } => &[CoinFlip],
        Mechanic::CoinFlipNoEffect => &[CoinFlip],
        Mechanic::ExtraDamageIfBothHeads { .. } => &[CoinFlip],
        Mechanic::CoinFlipToBlockAttackNextTurn => &[CoinFlip, CardEffect],
        
        // === ENERGY MANIPULATION ===
        Mechanic::SelfDiscardEnergy { .. } => &[EnergyManip],
        Mechanic::SelfDiscardAllEnergy => &[EnergyManip],
        Mechanic::SelfDiscardRandomEnergy => &[EnergyManip],
        Mechanic::DiscardRandomGlobalEnergy => &[EnergyManip],
        Mechanic::DiscardEnergyFromOpponentActive => &[EnergyManip],
        Mechanic::SelfChargeActive { .. } => &[EnergyManip],
        Mechanic::ChargeBench { .. } => &[EnergyManip],
        Mechanic::ManaphyOceanicGift => &[EnergyManip],
        Mechanic::MoltresExInfernoDance => &[EnergyManip, CoinFlip],
        Mechanic::VaporeonHyperWhirlpool => &[EnergyManip, CoinFlip],
        
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
        Mechanic::DamageAllOpponentPokemon { .. } => &[BenchDamage],
        Mechanic::AlsoBenchDamage { .. } => &[BenchDamage],
        Mechanic::AlsoChoiceBenchDamage { .. } => &[BenchDamage],
        Mechanic::DirectDamage { .. } => &[BenchDamage],
        Mechanic::ConditionalBenchDamage { .. } => &[BenchDamage, ConditionalDamage],
        Mechanic::PalkiaExDimensionalStorm => &[BenchDamage, EnergyManip],
        
        // === CARD EFFECTS ===
        Mechanic::DamageAndCardEffect { .. } => &[CardEffect],
        Mechanic::DamageAndMultipleCardEffects { .. } => &[CardEffect],
        Mechanic::DamageAndTurnEffect { .. } => &[CardEffect],
        Mechanic::BlockBasicAttack => &[CardEffect],
        Mechanic::ShuffleOpponentActiveIntoDeck => &[CardEffect],
        
        // === SEARCH ===
        Mechanic::SearchToHandByEnergy { .. } => &[Search],
        Mechanic::SearchToBenchByName { .. } => &[Search],
        Mechanic::MagikarpWaterfallEvolution => &[Search],
        
        // === SELF DAMAGE ===
        Mechanic::SelfDamage { .. } => &[SelfDamage],
        Mechanic::RecoilIfKo { .. } => &[SelfDamage],
        
        // === SPECIAL / UTILITY ===
        Mechanic::SwitchSelfWithBench => &[CardEffect],
        Mechanic::CelebiExPowerfulBloom => &[CoinFlip],
        Mechanic::MegaBlazikenExMegaBurningAttack => &[EnergyManip, Status],
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
        assert!(cats.contains(&AttackEffectCategory::Heals));
    }
    
    #[test]
    fn test_coin_flip_multi_category() {
        let mechanic = Mechanic::CoinFlipExtraDamageOrSelfDamage { 
            extra_damage: 40, 
            self_damage: 20 
        };
        let cats = get_attack_effect_categories(&mechanic);
        assert!(cats.contains(&AttackEffectCategory::CoinFlip));
        assert!(cats.contains(&AttackEffectCategory::SelfDamage));
    }
    
    #[test]
    fn test_encoding_size() {
        let mechanic = Mechanic::SelfDiscardEnergy { 
            energies: vec![EnergyType::Fire] 
        };
        let encoding = encode_attack_categories(&mechanic);
        assert_eq!(encoding.len(), NUM_ATTACK_EFFECT_CATEGORIES);
        assert_eq!(encoding[AttackEffectCategory::EnergyManip as usize], 1.0);
    }
}
