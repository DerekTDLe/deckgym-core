# Embedding Module for DeckGym

This module generates text embeddings for Pokemon TCG card effects.
Trained on the full TCG database and applied to the Pocket subset.

## Usage

```python
from deckgym.embeddings import generate_all

# Generate all embeddings (train on TCG, apply to Pocket)
generate_all()
```

## Configuration

Edit `config.py` to change:
- `EMBEDDING_DIM`: Output dimension (default: 128)
- `SENTENCE_MODEL`: SentenceTransformer model name
