# AI Roadmap - Imperfect Information & Advanced RL

## Goals
- Transform the current AlphaZero agent into one capable of handling imperfect information
- Bayesian inference to guess opponent decks
- Explore R-NaD/DeepNash for Nash equilibrium approach

---

## 1. Deck Guesser (Bayesian Inference)

### Concept
Train an agent to guess the complete composition of a deck based on revealed cards (board + discard). Once the deck is estimated, we can sample multiple possible states and transform the game into perfect information via ISMCTS.

### Proposed Architecture

```
python/deckgym/belief/
├── __init__.py
├── card_features.py      # Feature extraction from database.json
├── belief_state.py       # BeliefState class
├── bayesian_updater.py   # Update logic for P(deck|revealed)
├── archetype_prior.py    # Load meta_deck.json as priors
└── evolution_graph.py    # Graph Basic→Stage1→Stage2
```

### Available Data
- `meta_deck.json`: 396 archetypes with scores and variants
- `card-scores.json`: Scores and popularity per card
- `database.json`: Complete structural features (stage, evolves_from, attacks, etc.)

### Card Feature Vector (Draft)
```python
card_features = {
    # Meta features (from card-scores.json)
    "meta_score": float,       # Global score (0 if absent)
    "meta_popularity": float,  # Popularity (0 if absent)
    
    # Structural features (from database.json)
    "stage": int,              # 0=Basic, 1=Stage1, 2=Stage2
    "is_ex": bool,
    "energy_type": List[int],  # one-hot (11 types)
    "hp_normalized": float,    # HP / 280
    
    # Attack semantics
    "max_damage_per_energy": float,
    "has_draw_effect": bool,
    "has_energy_accel": bool,
    "has_switch_effect": bool,
    "has_heal": bool,
    "has_bench_snipe": bool,
    
    # Evolution info
    "has_evolution_target": bool,  # Stage 1/2 need pre-evolution
    "can_be_evolved": bool,        # Basic can evolve
    "evolution_depth": int,        # 0=final or no evolution
}
```

### BeliefState Class
```python
class BeliefState:
    revealed_cards: List[CardId]           # board + discard
    deck_probs: Dict[str, float]      # P(deck | revealed)
    card_probs: Dict[CardId, float]        # Fallback for rogue decks
```

### Bayesian Update
```
P(deck | revealed) ∝ P(revealed | deck) × P(deck)
```
Hierarchical, first find the corresponding archetype if there is one, then find the corresponding deck and it's variants.

### Evolutionary Fallback (Generalization)
- If no archetypes matches → "rogue deck" mode
- Use evolution constraints: Stage 2 seen → Basic/Stage1 present
- Auto-include cards (Professor's Research, Poké Ball) have high prior
- Infer cards following deck energy type and popularity

### Implementation Steps
- [ ] **Step 1**: `evolution_graph.py` - Evolution graph from database.json
- [ ] **Step 2**: `card_features.py` - Merge database.json + card-scores.json
- [ ] **Step 3**: `archetype_prior.py` - Load meta_deck.json as priors
- [ ] **Step 4**: `belief_state.py` + `bayesian_updater.py`
- [ ] **Step 5**: Generate synthetic training data
- [ ] **Step 6**: Train multi-label classification model

---

## 2. 🎮 AlphaZero for Imperfect Information (ISMCTS)

### Concept
Combine the Deck Guesser with AlphaZero via Information Set Monte Carlo Tree Search:
1. Sample N complete states consistent with observation
2. Run AlphaZero/MCTS on each determinized state
3. Aggregate decisions (majority vote, weighted average)

### Architecture
```
┌─────────────────────────────────────────────────┐
│           Opponent Belief State                 │
│  P(deck | revealed_cards, game_state)           │
└───────────────────┬─────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│         Informed State Sampler                  │
│  Sample N complete game states from belief      │
└───────────────────┬─────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│       Batched AlphaZero-style Policy            │
│  (Existing Rust infra with ONNX batching)       │
└───────────────────┬─────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│         Action Aggregation                      │
│  • Majority vote                                │
│  • Weighted average by P(sample)                │
│  • Robust averaging (median)                    │
└─────────────────────────────────────────────────┘
```

### Implementation Steps
- [ ] Integrate BeliefState into agent observation
- [ ] Implement Informed State Sampler on Rust side
- [ ] Modify MCTS to support information sets
- [ ] Benchmark: compare with current agent (e2 with near perfect information or AI agents)

---

## 3. 📚 Research: R-NaD & DeepNash

---

## 4. 📋 Priorities

| Priority | Task |
|----------|------|
| 0 | Deck Guesser MVP (archetype matching) |
| 1 | Evolutionary fallback for rogue decks |
| 2 | ISMCTS integration |
| 3 | R-NaD/DeepNash research |

---

## 5. 📝 Notes

### Advantages of this Approach
- `meta_deck.json` data provides excellent priors
- Existing Rust + ONNX batching infra can evaluate N samples in parallel
- Decks are small (20 cards) → manageable sampling space

### Risks
- Implementation complexity of belief updater
- Tuning number of samples vs performance (latency)
- Edge cases where belief is too diffuse (early game)
- High variance if posteriors are flat
